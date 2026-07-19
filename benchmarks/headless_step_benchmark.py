#!/usr/bin/env python3
"""Compare legacy-equivalent and optimized headless engine turn loops.

The baseline deliberately calls ``step(action)`` after the caller has already
computed state and legal actions. The optimized path passes the fresh legal
collection and suppresses the discarded post-action state. Both modes use the
same per-game seeds and must produce identical result fingerprints.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.agent import RandomAgent
from agents.heuristic_agent import StrategicAgent
from agents.neural_agent import NeuralAgent
from agents.rl_agent import RLAgent
from agents.rl_nn import PolicyNetwork
from middleware.domino_engine import DominoEngine
from training.self_play import (
    DEFAULT_GAMMA,
    DEFAULT_REWARD_SCHEMA,
    REWARD_SCHEMAS,
    _collect_self_play_steps,
)


@dataclass
class EngineCounters:
    """Structural work performed by all engines inside one benchmark case."""

    step_calls: int = 0
    valid_action_calls: int = 0
    state_snapshot_calls: int = 0
    serialized_history_actions: int = 0


@contextlib.contextmanager
def _count_engine_calls(*, force_default_step: bool = False):
    """Instrument engine calls and optionally emulate the discarded-state path."""
    counters = EngineCounters()
    original_step = DominoEngine.step
    original_valid_actions = DominoEngine.valid_actions
    original_get_state = DominoEngine._get_state
    original_serialize_action = DominoEngine._serialize_action

    def counted_step(self, action, *args, **kwargs):
        counters.step_calls += 1
        if force_default_step:
            return original_step(self, action)
        return original_step(self, action, *args, **kwargs)

    def counted_valid_actions(self, player=None):
        counters.valid_action_calls += 1
        return original_valid_actions(self, player)

    def counted_get_state(self):
        counters.state_snapshot_calls += 1
        return original_get_state(self)

    def counted_serialize_action(self, action):
        counters.serialized_history_actions += 1
        return original_serialize_action(self, action)

    DominoEngine.step = counted_step
    DominoEngine.valid_actions = counted_valid_actions
    DominoEngine._get_state = counted_get_state
    DominoEngine._serialize_action = counted_serialize_action
    try:
        yield counters
    finally:
        DominoEngine.step = original_step
        DominoEngine.valid_actions = original_valid_actions
        DominoEngine._get_state = original_get_state
        DominoEngine._serialize_action = original_serialize_action


def _agent_pair(agent_name: str):
    """Build one evaluated agent and the common random opponent outside timing."""
    if agent_name == "random":
        evaluated = RandomAgent()
    elif agent_name == "heuristic":
        evaluated = StrategicAgent()
    elif agent_name == "neural":
        evaluated = NeuralAgent.load(
            str(ROOT / "models" / "domino_sl_weights.npz"),
            device="cpu",
        )
    elif agent_name == "rl":
        network = PolicyNetwork.load(
            str(ROOT / "models" / "domino_rl_weights.npz"),
            device="cpu",
        )
        evaluated = RLAgent(network, mode="evaluation")
    else:  # pragma: no cover - argparse and internal callers constrain names.
        raise ValueError(f"Unknown benchmark agent: {agent_name}")
    return evaluated, RandomAgent()


def _run_matchup(agent_name: str, game_count: int, base_seed: int, optimized: bool):
    """Benchmark one agent-vs-random loop and return counters plus fingerprint."""
    evaluated, opponent = _agent_pair(agent_name)
    agents_by_position = (
        (evaluated, opponent),
        (opponent, evaluated),
    )
    digest = hashlib.sha256()

    with _count_engine_calls() as counters:
        start = time.perf_counter()
        for game_index in range(game_count):
            seed = base_seed + game_index
            random.seed(seed)
            np.random.seed(seed & 0xFFFFFFFF)
            engine = DominoEngine(player_count=2)
            agents = agents_by_position[game_index % 2]
            while not engine.game_over:
                state = engine._get_state()
                current_player = state["current_player"]
                legal_actions = engine.valid_actions(current_player)
                action = agents[current_player].choose_move(state, legal_actions)
                if optimized:
                    engine.step(
                        action,
                        return_state=False,
                        legal_actions=legal_actions,
                    )
                else:
                    engine.step(action)
            final_state = engine.to_dict()
            # Game ids are process-global allocation metadata and therefore
            # differ because the baseline and optimized cases run sequentially.
            # Normalize only that identifier; every rule-bearing field remains
            # part of the equivalence fingerprint.
            final_state["game_id"] = game_index + 1
            digest.update(
                json.dumps(
                    final_state,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        elapsed = time.perf_counter() - start

    return {
        "games": game_count,
        "elapsed_seconds": elapsed,
        "games_per_second": game_count / elapsed,
        "fingerprint": digest.hexdigest(),
        "counters": asdict(counters),
    }


def _sample_fingerprint(digest, samples, events, winner, learner_position):
    """Add one complete RL rollout result to a stable comparison digest."""
    digest.update(f"{winner}:{learner_position}:{asdict(events)}".encode("utf-8"))
    for sample in samples:
        digest.update(np.asarray(sample.x).tobytes())
        digest.update(np.asarray(sample.legal_mask).tobytes())
        digest.update(
            repr((
                sample.action_index,
                sample.policy_reward,
                sample.raw_reward,
                sample.local_reward,
                sample.terminal_reward,
                sample.multiplier,
                sample.option_count,
            )).encode("utf-8")
        )


def _run_rl_rollouts(game_count: int, base_seed: int, optimized: bool):
    """Benchmark the production RL rollout loop with or without the fast path."""
    network = PolicyNetwork.load(
        str(ROOT / "models" / "domino_rl_weights.npz"),
        device="cpu",
    )
    schema = REWARD_SCHEMAS[DEFAULT_REWARD_SCHEMA]
    digest = hashlib.sha256()

    with _count_engine_calls(force_default_step=not optimized) as counters:
        start = time.perf_counter()
        for game_index in range(game_count):
            seed = base_seed + game_index
            random.seed(seed)
            np.random.seed(seed & 0xFFFFFFFF)
            samples, events, winner, learner_position = _collect_self_play_steps(
                network,
                [],
                schema,
                DEFAULT_GAMMA,
            )
            _sample_fingerprint(
                digest,
                samples,
                events,
                winner,
                learner_position,
            )
        elapsed = time.perf_counter() - start

    return {
        "games": game_count,
        "elapsed_seconds": elapsed,
        "games_per_second": game_count / elapsed,
        "fingerprint": digest.hexdigest(),
        "counters": asdict(counters),
    }


def _comparison(name: str, baseline: dict, optimized: dict) -> dict:
    """Validate equivalence and return one compact benchmark comparison."""
    if baseline["fingerprint"] != optimized["fingerprint"]:
        raise RuntimeError(f"Result fingerprint changed for {name}.")
    return {
        "workload": name,
        "fingerprints_identical": True,
        "speedup_percent": (
            100.0
            * (optimized["games_per_second"] / baseline["games_per_second"] - 1.0)
        ),
        "baseline": baseline,
        "optimized": optimized,
    }


def run_benchmark(game_count: int, rollout_game_count: int, seed: int) -> dict:
    """Run diagnostic-style matchups and production RL rollout comparisons."""
    comparisons = []
    for offset, agent_name in enumerate(("random", "heuristic", "neural", "rl")):
        if agent_name in {"neural", "rl"}:
            model_name = (
                "domino_sl_weights.npz"
                if agent_name == "neural"
                else "domino_rl_weights.npz"
            )
            if not (ROOT / "models" / model_name).exists():
                continue
        case_seed = seed + offset * 1_000_000
        baseline = _run_matchup(agent_name, game_count, case_seed, False)
        optimized = _run_matchup(agent_name, game_count, case_seed, True)
        comparisons.append(_comparison(
            f"{agent_name}_vs_random",
            baseline,
            optimized,
        ))

    rl_weights = ROOT / "models" / "domino_rl_weights.npz"
    if rl_weights.exists() and rollout_game_count > 0:
        baseline = _run_rl_rollouts(rollout_game_count, seed + 9_000_000, False)
        optimized = _run_rl_rollouts(rollout_game_count, seed + 9_000_000, True)
        comparisons.append(_comparison("rl_rollout_generation", baseline, optimized))

    return {
        "game_count_per_matchup": game_count,
        "rl_rollout_game_count": rollout_game_count,
        "seed": seed,
        "comparisons": comparisons,
    }


def _print_report(report: dict) -> None:
    """Print throughput and structural call reductions without fixed assertions."""
    header = (
        "workload",
        "baseline game/s",
        "optimized game/s",
        "speedup",
        "valid calls",
        "state calls",
        "serialized actions",
    )
    print(" | ".join(header))
    print("-" * 132)
    for comparison in report["comparisons"]:
        baseline = comparison["baseline"]
        optimized = comparison["optimized"]
        before = baseline["counters"]
        after = optimized["counters"]
        print(
            f"{comparison['workload']} | "
            f"{baseline['games_per_second']:.2f} | "
            f"{optimized['games_per_second']:.2f} | "
            f"{comparison['speedup_percent']:+.1f}% | "
            f"{before['valid_action_calls']} -> {after['valid_action_calls']} | "
            f"{before['state_snapshot_calls']} -> {after['state_snapshot_calls']} | "
            f"{before['serialized_history_actions']} -> "
            f"{after['serialized_history_actions']}"
        )
    print("All result fingerprints are identical.")


def parse_args(argv=None):
    """Parse reproducible benchmark sizes and optional JSON output."""
    parser = argparse.ArgumentParser(
        description="Benchmark the backward-compatible headless engine step path."
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--rollout-games", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.games < 1 or args.rollout_games < 0:
        parser.error("--games must be positive and --rollout-games non-negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    report = run_benchmark(args.games, args.rollout_games, args.seed)
    _print_report(report)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
