"""Generate timed full-game JSONL files for every three-agent pairing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agents.agent import RandomAgent
from agents.rl_agent import RLAgent
from agents.rl_nn import PolicyNetwork
from agents.heuristic_agent import StrategicAgent
from middleware.domino_engine import DominoEngine
from training.rl.rollout import (
    FINAL_PIP_PENALTY,
    LEARNER_DRAW_PENALTY,
    LEARNER_PASS_PENALTY,
    OPPONENT_DRAW_REWARD,
    OPPONENT_PASS_REWARD,
    TERMINAL_LOSS_REWARD,
    TERMINAL_WIN_REWARD,
)


DEFAULT_GAMES = 10_000
DEFAULT_BASE_SEED = 42
DEFAULT_WEIGHTS = SCRIPT_DIR / "neural_policy_weights.npz"
MANIFEST_FILE = "full_game_pair_generation_manifest.json"
AGENT_ORDER = ("random", "heuristic", "neural")
MATCHUPS = (
    ("random", "random"),
    ("random", "heuristic"),
    ("random", "neural"),
    ("heuristic", "heuristic"),
    ("heuristic", "neural"),
    ("neural", "neural"),
)


def matchup_key(pair):
    """Return the stable filename/configuration key for one pairing."""
    return f"{pair[0]}_vs_{pair[1]}"


def output_path(output_dir, pair):
    """Return the raw JSONL path for a pairing."""
    return output_dir / f"{matchup_key(pair)}_full_games.jsonl"


def elapsed_us(start_ns):
    """Return rounded wall-clock microseconds from a perf-counter timestamp."""
    return int((time.perf_counter_ns() - start_ns + 500) // 1_000)


def file_sha256(path):
    """Hash one local artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_model():
    """Return a concise CPU identifier for timing provenance."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def machine_metadata():
    """Describe the single-process environment behind wall timings."""
    clock = time.get_clock_info("perf_counter")
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu_model": cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "neural_device": "cpu",
        "execution": "single_process_sequential",
        "timer": "time.perf_counter_ns",
        "timer_monotonic": bool(clock.monotonic),
        "timer_adjustable": bool(clock.adjustable),
        "timer_resolution_ns": int(round(clock.resolution * 1_000_000_000)),
        "stored_timing_unit": "integer_microseconds",
    }


def load_neural_network(weights_path):
    """Load and warm one shared CPU policy outside all measured games."""
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing copied neural weights: {weights_path}")
    network = PolicyNetwork.load(
        weights_path,
        use_value_head=False,
        device="cpu",
    )
    input_size = int(network.W1.shape[1])
    probe = np.zeros((input_size, 1), dtype=np.float32)
    warmup_started = time.perf_counter_ns()
    for _ in range(10):
        network.forward(probe)
    return network, {
        "file": weights_path.name,
        "bytes": weights_path.stat().st_size,
        "sha256": file_sha256(weights_path),
        "input_size": input_size,
        "output_size": int(getattr(network, f"W{network.layer_count}").shape[0]),
        "hidden_sizes": [
            int(getattr(network, f"W{index}").shape[0])
            for index in range(1, network.layer_count)
        ],
        "device": "cpu",
        "warmup_forwards": 10,
        "warmup_wall_us": elapsed_us(warmup_started),
    }


class AgentFactory:
    """Create fresh per-game agents while sharing immutable neural weights."""

    def __init__(self, network):
        self.network = network

    def create(self, agent_name):
        if agent_name == "random":
            return RandomAgent()
        if agent_name == "heuristic":
            return StrategicAgent()
        if agent_name == "neural":
            return RLAgent(self.network, mode="evaluation")
        raise ValueError(f"Unknown agent type: {agent_name!r}")


def seat_agents(pair, game_index):
    """Alternate starting seats for mixed matchups, preserving self-pairs."""
    if pair[0] == pair[1] or game_index % 2 == 0:
        return pair
    return (pair[1], pair[0])


def action_kind(action):
    """Return the serialized action category."""
    if action is None:
        return "pass"
    if (
        isinstance(action, (list, tuple))
        and len(action) == 2
        and action[0] == "DRAW"
    ):
        return "draw"
    return "tile_play"


def decision_class(action, tile_option_count):
    """Separate forced rules-engine turns from genuine policy choices."""
    kind = action_kind(action)
    if kind == "draw":
        return "forced_draw"
    if kind == "pass":
        return "forced_pass"
    if tile_option_count <= 1:
        return "forced_tile"
    return "voluntary_choice"


def raw_event_reward_by_seat(action, acting_player):
    """Return undiscounted draw/pass shaping from each seat's perspective."""
    rewards = [0.0, 0.0]
    other_player = 1 - int(acting_player)
    kind = action_kind(action)
    if kind == "draw":
        rewards[acting_player] = float(LEARNER_DRAW_PENALTY)
        rewards[other_player] = float(OPPONENT_DRAW_REWARD)
    elif kind == "pass":
        rewards[acting_player] = float(LEARNER_PASS_PENALTY)
        rewards[other_player] = float(OPPONENT_PASS_REWARD)
    return rewards


def raw_terminal_components(final_state):
    """Return terminal outcome and final-pip terms separately for both seats."""
    winner = int(final_state["winner"])
    if winner not in (0, 1):
        raise ValueError(f"Unexpected terminal winner {winner!r}")
    outcome = [
        float(TERMINAL_WIN_REWARD if seat == winner else TERMINAL_LOSS_REWARD)
        for seat in range(2)
    ]
    pip_penalty = [
        -float(FINAL_PIP_PENALTY)
        * float(sum(int(tile[0]) + int(tile[1]) for tile in hand))
        for hand in final_state["hands"]
    ]
    return outcome, pip_penalty


def raw_reward_schema():
    """Describe the fixed training reward constants captured by this analysis."""
    return {
        "terminal_win": float(TERMINAL_WIN_REWARD),
        "terminal_loss": float(TERMINAL_LOSS_REWARD),
        "final_pip_penalty_per_pip": -float(FINAL_PIP_PENALTY),
        "self_draw": float(LEARNER_DRAW_PENALTY),
        "opponent_draw": float(OPPONENT_DRAW_REWARD),
        "self_pass": float(LEARNER_PASS_PENALTY),
        "opponent_pass": float(OPPONENT_PASS_REWARD),
        "temporal_discount_applied": False,
        "alpha_mixing_applied": False,
    }


def play_timed_game(game_index, seed, pair, factory):
    """Play one full game and retain integer wall timings for every action."""
    game_started = time.perf_counter_ns()
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)

    setup_started = time.perf_counter_ns()
    seats = seat_agents(pair, game_index)
    engine = DominoEngine(player_count=2)
    # Stable ids make exact-belief resets and files directly comparable across
    # all six matchups using the same seed/deal index.
    engine.game_id = game_index + 1
    agents = [factory.create(agent_name) for agent_name in seats]
    initial_state = engine.to_dict()
    setup_wall_us = elapsed_us(setup_started)

    simulation_started = time.perf_counter_ns()
    history = []
    while not engine.game_over:
        turn_started = time.perf_counter_ns()

        section_started = time.perf_counter_ns()
        state = engine._get_state()
        state_wall_us = elapsed_us(section_started)
        current_player = int(state["current_player"])

        section_started = time.perf_counter_ns()
        legal_actions = engine.valid_actions(current_player)
        legal_actions_wall_us = elapsed_us(section_started)
        tile_option_count = sum(
            action is not None
            and not (
                isinstance(action, (list, tuple))
                and len(action) == 2
                and action[0] == "DRAW"
            )
            for action in legal_actions
        )

        section_started = time.perf_counter_ns()
        chosen_action = agents[current_player].choose_move(state, legal_actions)
        decision_wall_us = elapsed_us(section_started)

        section_started = time.perf_counter_ns()
        _, game_over, _info = engine.step(
            chosen_action,
            return_state=False,
            legal_actions=legal_actions,
        )
        transition_wall_us = elapsed_us(section_started)
        turn_wall_us = elapsed_us(turn_started)

        history.append({
            "state": state,
            "target_action": chosen_action,
            "acting_player": current_player,
            "acting_agent": seats[current_player],
            "raw_event_reward_by_seat": raw_event_reward_by_seat(
                chosen_action, current_player
            ),
            "legal_action_count": len(legal_actions),
            "tile_option_count": int(tile_option_count),
            "decision_class": decision_class(
                chosen_action,
                tile_option_count,
            ),
            "timing_us": {
                "state": state_wall_us,
                "legal_actions": legal_actions_wall_us,
                "agent_decision": decision_wall_us,
                "engine_transition": transition_wall_us,
                "turn_wall": turn_wall_us,
            },
        })
        if game_over:
            break

    simulation_wall_us = elapsed_us(simulation_started)
    final_state = engine.to_dict()
    terminal_outcome, final_pip_penalty = raw_terminal_components(final_state)
    event_sum = [
        float(sum(item["raw_event_reward_by_seat"][seat] for item in history))
        for seat in range(2)
    ]
    total_raw = [
        event_sum[seat] + terminal_outcome[seat] + final_pip_penalty[seat]
        for seat in range(2)
    ]
    return {
        "game": game_index + 1,
        "seed": seed,
        "matchup": matchup_key(pair),
        "seat_agents": list(seats),
        "initial_state": initial_state,
        "history": history,
        "final_state": final_state,
        "raw_rewards": {
            "event_sum_by_seat": event_sum,
            "terminal_outcome_by_seat": terminal_outcome,
            "final_pip_penalty_by_seat": final_pip_penalty,
            "total_by_seat": total_raw,
        },
        "timing_us": {
            "setup": setup_wall_us,
            "simulation": simulation_wall_us,
            "game_wall": elapsed_us(game_started),
        },
    }


def generate_matchup(pair, args, factory):
    """Atomically write one complete timed pairing and return its manifest."""
    destination = output_path(args.output_dir, pair)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    digest = hashlib.sha256()
    matchup_started = time.perf_counter_ns()
    simulation_sum_us = 0
    turn_count = 0
    try:
        with temporary.open("wb") as stream:
            for game_index in range(args.games):
                seed = int(args.base_seed) + game_index
                record = play_timed_game(
                    game_index,
                    seed,
                    pair,
                    factory,
                )
                simulation_sum_us += int(record["timing_us"]["simulation"])
                turn_count += len(record["history"])
                payload = (
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                stream.write(payload)
                digest.update(payload)
                completed = game_index + 1
                if args.progress_every and completed % args.progress_every == 0:
                    elapsed = max(1, elapsed_us(matchup_started))
                    rate = completed * 1_000_000 / elapsed
                    print(
                        f"{matchup_key(pair)}: {completed:,}/{args.games:,} "
                        f"games ({rate:.1f} games/s)",
                        flush=True,
                    )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    matchup_wall_us = elapsed_us(matchup_started)
    return {
        "matchup": matchup_key(pair),
        "canonical_agents": list(pair),
        "mixed_matchup_seats_alternate": pair[0] != pair[1],
        "games": int(args.games),
        "turns": int(turn_count),
        "base_seed": int(args.base_seed),
        "last_seed": int(args.base_seed) + int(args.games) - 1,
        "output_file": destination.name,
        "output_bytes": destination.stat().st_size,
        "output_sha256": digest.hexdigest(),
        "summed_game_simulation_wall_us": int(simulation_sum_us),
        "matchup_end_to_end_wall_us": matchup_wall_us,
        "simulation_games_per_second": (
            args.games * 1_000_000 / simulation_sum_us
            if simulation_sum_us else None
        ),
        "end_to_end_games_per_second": (
            args.games * 1_000_000 / matchup_wall_us
            if matchup_wall_us else None
        ),
    }


def selected_matchups(pattern):
    """Resolve ``all`` or a comma-separated list of exact matchup keys."""
    if pattern == "all":
        return MATCHUPS
    requested = {item.strip() for item in pattern.split(",") if item.strip()}
    known = {matchup_key(pair): pair for pair in MATCHUPS}
    unknown = requested - set(known)
    if unknown:
        raise ValueError(
            f"Unknown matchups {sorted(unknown)}; choose from {sorted(known)}"
        )
    return tuple(pair for pair in MATCHUPS if matchup_key(pair) in requested)


def write_manifest(args, pair_manifests, weights_metadata, run_wall_us):
    """Publish timing provenance only after every selected file succeeds."""
    manifest = {
        "format_version": 2,
        "purpose": (
            "timed full games plus undiscounted raw RL reward components for "
            "every random/heuristic/neural pairing"
        ),
        "games_per_matchup": int(args.games),
        "base_seed": int(args.base_seed),
        "matchup_order": [item["matchup"] for item in pair_manifests],
        "seat_policy": (
            "mixed matchups swap player 0/player 1 every game; self matchups "
            "have identical agent types in both seats"
        ),
        "raw_reward_schema": raw_reward_schema(),
        "raw_reward_semantics": (
            "Every action stores the immediate draw/pass shaping reward from both "
            "seat perspectives. Terminal win/loss and final-pip penalty are stored "
            "separately once per game. No gamma/event decay or alpha mixture is "
            "applied anywhere in these analysis fields."
        ),
        "timing_scope": {
            "state": "DominoEngine._get_state",
            "legal_actions": "DominoEngine.valid_actions",
            "agent_decision": "Agent.choose_move",
            "engine_transition": "DominoEngine.step",
            "turn_wall": "complete turn including timing/bookkeeping overhead",
            "simulation": "complete game loop, excluding setup and JSON serialization",
            "game_wall": "RNG setup, engine/agent setup, and game simulation",
            "matchup_end_to_end": "simulation plus JSON serialization and file I/O",
        },
        "machine": machine_metadata(),
        "neural_policy": weights_metadata,
        "matchups": pair_manifests,
        "run_end_to_end_wall_us": int(run_wall_us),
    }
    path = args.output_dir / MANIFEST_FILE
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def parse_args():
    """Parse reproducible workload and output controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument(
        "--matchups",
        default="all",
        help="'all' or comma-separated canonical matchup names.",
    )
    parser.add_argument("--progress-every", type=int, default=1_000)
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    try:
        args.selected_matchups = selected_matchups(args.matchups)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    """Generate every selected matchup using one warmed CPU neural policy."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    network, weights_metadata = load_neural_network(args.weights)
    factory = AgentFactory(network)
    run_started = time.perf_counter_ns()
    manifests = []
    print(
        f"Generating {len(args.selected_matchups)} matchups x "
        f"{args.games:,} games with integer-microsecond wall timings."
    )
    for index, pair in enumerate(args.selected_matchups, start=1):
        print(
            f"\n[{index}/{len(args.selected_matchups)}] {matchup_key(pair)}",
            flush=True,
        )
        manifests.append(generate_matchup(pair, args, factory))
    manifest_path = write_manifest(
        args,
        manifests,
        weights_metadata,
        elapsed_us(run_started),
    )
    print(f"\nSaved generation manifest to {manifest_path}")


if __name__ == "__main__":
    main()
