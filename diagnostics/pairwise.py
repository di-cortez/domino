"""
Evaluate one domino agent against another and write compact diagnostics.

This module is the pairwise diagnostics helper. The high-level entry point
``diagnostics.evaluate`` calls this module repeatedly to compare every agent
with the common random baseline, but this file can still be executed directly
when only one matchup is needed:

    python -m diagnostics.pairwise --agent heuristic --opponent random

The evaluated agent alternates between player 0 and player 1 so the result is
not dominated by first-player advantage. Outputs are written to a dedicated
folder containing JSON, CSV, and optional PNG plots.
"""

import argparse
import contextlib
import csv
import io
import json
import math
import secrets
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.plots import LABEL, generate_plots as write_diagnostic_plots, summarize
from diagnostics.parallel_runner import (
    ESTIMATED_DIAGNOSTIC_RECORD_BYTES,
    MAX_DIAGNOSTIC_WORKERS,
    ParallelRunInfo,
    ParallelSafetyConfig,
    evaluate_game_specs,
    game_seed,
)
from middleware.domino_engine import DominoEngine
from utils.resource_limits import ensure_ram_available
from utils.runtime_status import format_duration, print_memory_report

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

DEFAULT_AGENT = "rl"
DEFAULT_OPPONENT = "heuristic"
DEFAULT_GAME_COUNT = 10000
DEFAULT_SEED = None
DEFAULT_AGENT_WEIGHTS = None
DEFAULT_OPPONENT_WEIGHTS = None
DEFAULT_OUTPUT_DIR = None
DEFAULT_GENERATE_PLOTS = True
DEFAULT_WORKERS = 1

CANONICAL_AGENTS = ("rl", "neural", "heuristic", "random")

DEFAULT_WEIGHTS = {
    "rl": ROOT / "models" / "domino_rl_weights.npz",
    "neural": ROOT / "models" / "domino_sl_weights.npz",
}
VALUE_WEIGHT_NAMES = ("Wv", "bv")


def normalize_agent_name(agent_name):
    """Return the canonical diagnostics name for an agent."""
    normalized = agent_name.strip().lower()
    if normalized not in CANONICAL_AGENTS:
        raise ValueError(f"Unknown agent {agent_name!r}. Options: {CANONICAL_AGENTS}")
    return normalized


def resolve_weights_path(agent_name, weights_path=None):
    """Return an existing checkpoint path for a checkpoint-backed agent."""
    agent_name = normalize_agent_name(agent_name)
    if agent_name not in DEFAULT_WEIGHTS:
        return None

    path = Path(weights_path or DEFAULT_WEIGHTS[agent_name])
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {agent_name} checkpoint: {path}. "
            "Generate the model first or pass an explicit weights path."
        )
    return path


def _checkpoint_has_value_head(path):
    """Return whether an RL checkpoint contains the complete value head."""
    with np.load(path, allow_pickle=False) as weights:
        return all(name in weights.files for name in VALUE_WEIGHT_NAMES)


def create_agent(agent_name, weights_path=None):
    """Create an agent by name, importing checkpoint-backed classes only when used."""
    agent_name = normalize_agent_name(agent_name)

    if agent_name == "rl":
        from agents.rl_agent import RLAgent

        path = resolve_weights_path("rl", weights_path)
        return RLAgent.load(
            str(path),
            mode="evaluation",
            use_value_head=_checkpoint_has_value_head(path),
        )
    if agent_name == "neural":
        from agents.neural_agent import NeuralAgent

        return NeuralAgent.load(str(resolve_weights_path("neural", weights_path)))
    if agent_name == "heuristic":
        from agents.heuristic_agent import StrategicAgent

        return StrategicAgent()
    if agent_name == "random":
        from agents.agent import RandomAgent

        return RandomAgent()

    raise ValueError(f"Unknown agent {agent_name!r}. Options: {CANONICAL_AGENTS}")


def is_forced_draw(legal_actions):
    """Return True when the only legal action is drawing from the stock."""
    return len(legal_actions) == 1 and legal_actions[0] == ("DRAW", None)


def is_forced_pass(legal_actions):
    """Return True when the only legal action is passing."""
    return len(legal_actions) == 1 and legal_actions[0] is None


def count_tile_play_options(legal_actions):
    """Count voluntary tile-play options, excluding forced draw/pass."""
    return sum(
        1
        for action in legal_actions
        if action is not None and action != ("DRAW", None)
    )


def empty_choice_stats():
    """Return counters for how often the evaluated agent really had a choice."""
    return {
        "agent_real_decision_turns": 0,
        "agent_forced_tile_turns": 0,
        "agent_forced_draws": 0,
        "agent_forced_passes": 0,
        "agent_choice_histogram": {},
    }


def update_choice_stats(stats, legal_actions):
    """Update choice counters and return whether this was a real decision."""
    if is_forced_draw(legal_actions):
        stats["agent_forced_draws"] += 1
        return False

    if is_forced_pass(legal_actions):
        stats["agent_forced_passes"] += 1
        return False

    option_count = count_tile_play_options(legal_actions)
    stats["agent_choice_histogram"][str(option_count)] = (
        stats["agent_choice_histogram"].get(str(option_count), 0) + 1
    )

    if option_count >= 2:
        stats["agent_real_decision_turns"] += 1
        return True
    else:
        stats["agent_forced_tile_turns"] += 1
        return False


def _new_value_head_stats(agent):
    """Create sufficient statistics when the evaluated agent has a critic."""
    network = getattr(agent, "network", None)
    if network is None or not getattr(network, "use_value_head", False):
        return None
    return {
        "sample_count": 0,
        "finite_count": 0,
        "nonfinite_count": 0,
        "sum": 0.0,
        "sum_squares": 0.0,
        "min": None,
        "max": None,
    }


def _record_value_head_prediction(agent, stats):
    """Record V(s) from the policy forward cache without another forward pass."""
    if stats is None:
        return
    network = agent.network
    hidden = network.cache.get("A2")
    if hidden is None:
        raise RuntimeError("RL value diagnostics require a completed policy forward pass.")
    value = network.xp.dot(network.Wv, hidden) + network.bv
    if hasattr(value, "get"):
        value = value.get()
    scalar = float(np.asarray(value).reshape(-1)[0])
    stats["sample_count"] += 1
    if not math.isfinite(scalar):
        stats["nonfinite_count"] += 1
        return
    stats["finite_count"] += 1
    stats["sum"] += scalar
    stats["sum_squares"] += scalar * scalar
    stats["min"] = scalar if stats["min"] is None else min(stats["min"], scalar)
    stats["max"] = scalar if stats["max"] is None else max(stats["max"], scalar)


def _add_game_runtime(runtime_profile, section, started):
    if runtime_profile is None or started is None:
        return
    sections = runtime_profile.setdefault("sections_seconds", {})
    sections[section] = sections.get(section, 0.0) + (
        time.perf_counter() - started
    )


def _game_runtime_start(runtime_profile):
    return time.perf_counter() if runtime_profile is not None else None


def _merge_numeric_runtime(target, source):
    for key, value in source.items():
        if isinstance(value, dict):
            _merge_numeric_runtime(target.setdefault(key, {}), value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            target[key] = target.get(key, 0) + value


def _play_game_unprofiled(agent, opponent, agent_position, suppress_agent_output):
    """Profiler-free diagnostic hot path for non-sampled games."""
    agents = [None, None]
    agents[agent_position] = agent
    agents[1 - agent_position] = opponent
    engine = DominoEngine(player_count=2)
    choice_stats = empty_choice_stats()
    value_head_stats = _new_value_head_stats(agent)
    while not engine.game_over:
        state = engine._get_state()
        current_player = state["current_player"]
        legal_actions = engine.valid_actions(current_player)
        evaluated_real_decision = False
        if current_player == agent_position:
            evaluated_real_decision = update_choice_stats(
                choice_stats,
                legal_actions,
            )
        if suppress_agent_output:
            with contextlib.redirect_stdout(io.StringIO()):
                action = agents[current_player].choose_move(state, legal_actions)
        else:
            action = agents[current_player].choose_move(state, legal_actions)
        if evaluated_real_decision:
            _record_value_head_prediction(agent, value_head_stats)
        engine.step(
            action,
            return_state=False,
            legal_actions=legal_actions,
        )

    final_state = engine.to_dict()
    winner = final_state["winner"]
    if winner == -1:
        result = "draw"
    elif winner == agent_position:
        result = "win"
    else:
        result = "loss"
    pips = [sum(tile[0] + tile[1] for tile in hand) for hand in final_state["hands"]]
    initial_hands = final_state["initial_hands"]
    record = {
        "game": None,
        "agent_position": agent_position,
        "result": result,
        "turns": final_state["turn"],
        "agent_initial_hand": initial_hands[agent_position],
        "opponent_initial_hand": initial_hands[1 - agent_position],
        "agent_remaining_pips": pips[agent_position],
        "opponent_remaining_pips": pips[1 - agent_position],
        **choice_stats,
    }
    if value_head_stats is not None:
        record["_agent_value_head_stats"] = value_head_stats
    return record


def play_game(
    agent,
    opponent,
    agent_position,
    suppress_agent_output=True,
    runtime_profile=None,
):
    """Play one game and return the outcome from the evaluated agent's view."""
    if runtime_profile is None:
        return _play_game_unprofiled(
            agent,
            opponent,
            agent_position,
            suppress_agent_output,
        )
    section_started = _game_runtime_start(runtime_profile)
    agents = [None, None]
    agents[agent_position] = agent
    agents[1 - agent_position] = opponent

    engine = DominoEngine(player_count=2)
    choice_stats = empty_choice_stats()
    value_head_stats = _new_value_head_stats(agent)
    _add_game_runtime(
        runtime_profile,
        "agent_pair_and_engine_initialization",
        section_started,
    )

    while not engine.game_over:
        section_started = _game_runtime_start(runtime_profile)
        state = engine._get_state()
        current_player = state["current_player"]
        legal_actions = engine.valid_actions(current_player)
        _add_game_runtime(
            runtime_profile,
            "state_and_legal_action_generation",
            section_started,
        )

        evaluated_real_decision = False
        if current_player == agent_position:
            section_started = _game_runtime_start(runtime_profile)
            evaluated_real_decision = update_choice_stats(
                choice_stats,
                legal_actions,
            )
            _add_game_runtime(
                runtime_profile,
                "evaluated_agent_choice_statistics",
                section_started,
            )

        section_started = _game_runtime_start(runtime_profile)
        if suppress_agent_output:
            with contextlib.redirect_stdout(io.StringIO()):
                action = agents[current_player].choose_move(state, legal_actions)
        else:
            action = agents[current_player].choose_move(state, legal_actions)
        _add_game_runtime(
            runtime_profile,
            (
                "evaluated_agent_decisions"
                if current_player == agent_position
                else "opponent_agent_decisions"
            ),
            section_started,
        )
        if evaluated_real_decision:
            section_started = _game_runtime_start(runtime_profile)
            _record_value_head_prediction(agent, value_head_stats)
            _add_game_runtime(
                runtime_profile,
                "evaluated_agent_value_head_statistics",
                section_started,
            )

        section_started = _game_runtime_start(runtime_profile)
        engine.step(
            action,
            return_state=False,
            legal_actions=legal_actions,
        )
        _add_game_runtime(
            runtime_profile,
            "engine_state_transition",
            section_started,
        )

    section_started = _game_runtime_start(runtime_profile)
    final_state = engine.to_dict()
    winner = final_state["winner"]

    if winner == -1:
        result = "draw"
    elif winner == agent_position:
        result = "win"
    else:
        result = "loss"

    pips = [sum(tile[0] + tile[1] for tile in hand) for hand in final_state["hands"]]
    initial_hands = final_state["initial_hands"]

    result = {
        "game": None,
        "agent_position": agent_position,
        "result": result,
        "turns": final_state["turn"],
        "agent_initial_hand": initial_hands[agent_position],
        "opponent_initial_hand": initial_hands[1 - agent_position],
        "agent_remaining_pips": pips[agent_position],
        "opponent_remaining_pips": pips[1 - agent_position],
        **choice_stats,
    }
    if value_head_stats is not None:
        result["_agent_value_head_stats"] = value_head_stats
    _add_game_runtime(
        runtime_profile,
        "final_state_and_outcome_serialization",
        section_started,
    )
    return result


def _effective_seed(seed):
    """Return the explicit run seed used to derive stable per-game seeds."""
    return int(seed) if seed is not None else secrets.randbits(63)


def evaluate_pair(
    agent_name,
    opponent_name,
    game_count=DEFAULT_GAME_COUNT,
    weights=None,
    opponent_weights=None,
    seed=None,
    progress_callback=None,
    suppress_agent_output=True,
    workers=DEFAULT_WORKERS,
    safety_config=None,
    game_indices=None,
    effective_seed=None,
    return_run_info=False,
):
    """Run deterministically seeded games in one or more CPU-only workers."""
    agent_name = normalize_agent_name(agent_name)
    opponent_name = normalize_agent_name(opponent_name)
    if game_count < 1:
        raise ValueError("game_count must be positive")
    workers = int(workers)
    if not 1 <= workers <= MAX_DIAGNOSTIC_WORKERS:
        raise ValueError(
            f"workers must be between 1 and {MAX_DIAGNOSTIC_WORKERS}"
        )
    resolved_weights = resolve_weights_path(agent_name, weights)
    resolved_opponent_weights = resolve_weights_path(opponent_name, opponent_weights)
    effective_seed = _effective_seed(seed) if effective_seed is None else int(effective_seed)
    indices = list(range(game_count)) if game_indices is None else [int(i) for i in game_indices]
    if any(index < 0 or index >= game_count for index in indices):
        raise ValueError("game_indices must be inside the requested game_count")
    if len(set(indices)) != len(indices):
        raise ValueError("game_indices contains duplicates")
    specs = [(index, game_seed(effective_seed, index)) for index in indices]

    games, run_info = evaluate_game_specs(
        agent_name=agent_name,
        opponent_name=opponent_name,
        game_specs=specs,
        weights=resolved_weights,
        opponent_weights=resolved_opponent_weights,
        requested_workers=workers,
        suppress_agent_output=suppress_agent_output,
        progress_callback=progress_callback,
        safety=safety_config or ParallelSafetyConfig(),
    )

    metadata = {
        "requested_seed": seed,
        "effective_seed": effective_seed,
        "parallel": run_info.to_dict(),
    }
    if return_run_info:
        return games, metadata
    return games


def save_csv(games, path):
    """Write compact per-game records to CSV."""
    fields = [
        "game",
        "game_seed",
        "agent_position",
        "result",
        "turns",
        "agent_initial_hand",
        "opponent_initial_hand",
        "agent_remaining_pips",
        "opponent_remaining_pips",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                field: (
                    json.dumps(game.get(field), separators=(",", ":"))
                    if field in {"agent_initial_hand", "opponent_initial_hand"}
                    else game.get(field)
                )
                for field in fields
            }
            for game in games
        )


def add_choice_summary(summary, games):
    """Attach aggregate choice-opportunity statistics to a pairwise summary."""
    histogram = {}
    for game in games:
        for option_count, count in game["agent_choice_histogram"].items():
            histogram[option_count] = histogram.get(option_count, 0) + count

    evaluated_turns = sum(
        game["agent_real_decision_turns"]
        + game["agent_forced_tile_turns"]
        + game["agent_forced_draws"]
        + game["agent_forced_passes"]
        for game in games
    )

    real_decisions = sum(game["agent_real_decision_turns"] for game in games)

    summary["choice_opportunities"] = {
        "evaluated_agent_turns": evaluated_turns,
        "real_decision_turns": real_decisions,
        "real_decision_rate": real_decisions / evaluated_turns if evaluated_turns else 0.0,
        "forced_tile_turns": sum(game["agent_forced_tile_turns"] for game in games),
        "forced_draws": sum(game["agent_forced_draws"] for game in games),
        "forced_passes": sum(game["agent_forced_passes"] for game in games),
        "choice_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
    }
    return summary


def add_value_head_summary(summary, games):
    """Attach aggregate V(s) statistics when the evaluated agent has a critic."""
    game_stats = [
        game["_agent_value_head_stats"]
        for game in games
        if "_agent_value_head_stats" in game
    ]
    if not game_stats:
        return summary

    sample_count = sum(stats["sample_count"] for stats in game_stats)
    finite_count = sum(stats["finite_count"] for stats in game_stats)
    nonfinite_count = sum(stats["nonfinite_count"] for stats in game_stats)
    total = sum(stats["sum"] for stats in game_stats)
    total_squares = sum(stats["sum_squares"] for stats in game_stats)
    if finite_count:
        mean = total / finite_count
        variance = max(0.0, total_squares / finite_count - mean * mean)
        minimum = min(
            stats["min"] for stats in game_stats if stats["min"] is not None
        )
        maximum = max(
            stats["max"] for stats in game_stats if stats["max"] is not None
        )
        std = math.sqrt(variance)
    else:
        mean = std = minimum = maximum = None

    summary["value_head_predictions"] = {
        "semantics": "V(s) on the evaluated agent's real decision states",
        "games_with_value_head": len(game_stats),
        "sample_count": sample_count,
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "mean": mean,
        "std": std,
        "min": minimum,
        "max": maximum,
    }
    return summary


def print_summary(summary, duration_s):
    """Print the main pairwise metrics in a compact console format."""
    game_count = summary["game_count"]
    games_per_second = game_count / duration_s if duration_s else float("inf")
    print(f"\n===== Diagnostics: {summary['agent']} vs {summary['opponent']} =====")
    print(
        f"Games: {game_count} | time: {format_duration(duration_s)} "
        f"({games_per_second:.1f} games/s)"
    )
    for key in ("win", "draw", "loss"):
        rate = summary["rates"][key]
        print(f"  {LABEL[key]:<8} {summary['counts'][key]:>5}  ({rate:6.1%})")
    lo, hi = summary["win_ci95"]
    print(f"  95% CI (win rate): [{lo:.1%}, {hi:.1%}]")
    for position in ("0", "1"):
        info = summary["by_position"][position]
        print(
            f"  As player {position}: {info['win_rate']:.1%} wins "
            f"across {info['games']} games"
        )
    print(f"  Turns per game: {summary['mean_turns']:.1f} +/- {summary['std_turns']:.1f}")
    print(
        "  Remaining pips (mean): "
        f"agent {summary['mean_agent_remaining_pips']:.1f} | "
        f"opponent {summary['mean_opponent_remaining_pips']:.1f}"
    )
    choice_info = summary.get("choice_opportunities")
    if choice_info:
        print(
            "  Choice opportunities: "
            f"{choice_info['real_decision_turns']}/"
            f"{choice_info['evaluated_agent_turns']} turns "
            f"({choice_info['real_decision_rate']:.1%})"
        )
        print(
            "  Forced turns: "
            f"tile {choice_info['forced_tile_turns']}, "
            f"draw {choice_info['forced_draws']}, "
            f"pass {choice_info['forced_passes']}"
        )
        print(f"  Choice histogram: {choice_info['choice_histogram']}")
    value_info = summary.get("value_head_predictions")
    if value_info:
        if value_info["finite_count"]:
            print(
                "  Value head V(s) mean/std/min/max: "
                f"{value_info['mean']:+.3f}/{value_info['std']:.3f}/"
                f"{value_info['min']:+.3f}/{value_info['max']:+.3f} "
                f"over {value_info['sample_count']} decisions"
            )
        if value_info["nonfinite_count"]:
            print(
                "  Value head non-finite predictions: "
                f"{value_info['nonfinite_count']}/"
                f"{value_info['sample_count']}"
            )


def _atomic_replace_directory(staging_dir, output_dir):
    """Commit a completed directory while preserving prior valid output."""
    staging_dir = Path(staging_dir)
    output_dir = Path(output_dir)
    backup_dir = output_dir.with_name(f".{output_dir.name}.backup-{time.time_ns()}")
    had_previous = output_dir.exists()
    try:
        if had_previous:
            output_dir.rename(backup_dir)
        staging_dir.rename(output_dir)
    except BaseException:
        if output_dir.exists() and not had_previous:
            shutil.rmtree(output_dir, ignore_errors=True)
        if backup_dir.exists() and not output_dir.exists():
            backup_dir.rename(output_dir)
        raise
    else:
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)


def run_pairwise(
    agent_name,
    opponent_name,
    game_count=DEFAULT_GAME_COUNT,
    weights=None,
    opponent_weights=None,
    seed=None,
    output_dir=None,
    generate_plots=DEFAULT_GENERATE_PLOTS,
    print_console_summary=True,
    suppress_agent_output=True,
    print_memory_summary=True,
    progress_callback=None,
    workers=DEFAULT_WORKERS,
    safety_config=None,
    precomputed_games=None,
    precomputed_duration_s=0.0,
    precomputed_runtime_profile=None,
    effective_seed=None,
    display_output_dir=None,
    save_game_records=True,
):
    """Run one matchup and atomically write its requested artifacts.

    ``save_game_records=False`` keeps the complete records in memory for the
    normal summary calculation but omits the potentially large ``games.csv``.
    This is useful for recurring monitors whose compact history is persisted
    elsewhere.
    """
    runtime_profile_started = time.perf_counter()
    runtime_sections = {}

    def add_runtime(section, started):
        runtime_sections[section] = runtime_sections.get(section, 0.0) + (
            time.perf_counter() - started
        )

    agent_name = normalize_agent_name(agent_name)
    opponent_name = normalize_agent_name(opponent_name)
    if game_count < 1:
        raise ValueError("game_count must be positive")
    workers = int(workers)
    if not 1 <= workers <= MAX_DIAGNOSTIC_WORKERS:
        raise ValueError(
            f"workers must be between 1 and {MAX_DIAGNOSTIC_WORKERS}"
        )
    output_dir = output_dir or (
        ROOT
        / "diagnostics"
        / "results"
        / "pairwise"
        / f"{agent_name}_vs_{opponent_name}"
    )
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    precomputed_games = list(precomputed_games or [])
    precomputed_by_index = {int(record["game"]) - 1: record for record in precomputed_games}
    if len(precomputed_by_index) != len(precomputed_games):
        raise ValueError("precomputed_games contains duplicate game ids")
    if any(index < 0 or index >= game_count for index in precomputed_by_index):
        raise ValueError("precomputed game id is outside game_count")
    missing_indices = [index for index in range(game_count) if index not in precomputed_by_index]
    # Results are deliberately aggregated in the parent, so reserve headroom
    # for the missing Python records plus the small numeric vectors used by
    # summaries/plots. Existing precomputed records are already reflected in
    # the current available-RAM probe and must not be counted twice.
    memory_reserve_mb = (
        safety_config.memory_reserve_mb
        if safety_config is not None
        else ParallelSafetyConfig().memory_reserve_mb
    )
    estimated_new_result_bytes = (
        len(missing_indices) * ESTIMATED_DIAGNOSTIC_RECORD_BYTES
        + game_count * 256
    )
    ensure_ram_available(
        estimated_new_result_bytes,
        memory_reserve_mb,
        f"retaining {game_count} diagnostic game records for "
        f"{agent_name} vs {opponent_name}",
    )
    runtime_sections["validation_setup_and_memory_preflight"] = (
        time.perf_counter() - runtime_profile_started
    )

    section_started = time.perf_counter()
    if print_console_summary:
        print(
            f"Evaluating {agent_name} vs {opponent_name} over {game_count} games "
            f"with {workers} worker(s) "
            "(starting position alternates every game)"
        )
    if print_console_summary and print_memory_summary:
        print_memory_report("Diagnostics startup memory")

    progress_bar = None
    if print_console_summary and tqdm is not None:
        progress_bar = tqdm(
            total=game_count,
            desc=f"{agent_name} vs {opponent_name}",
            unit="game",
            leave=True,
            initial=len(precomputed_games),
        )
    add_runtime("console_and_progress_setup", section_started)

    def progress(_done, _total):
        if progress_bar is not None:
            progress_bar.update(max(0, _done - (progress_bar.n - len(precomputed_games))))
        if progress_callback is not None:
            progress_callback(_done, _total)

    start_time = time.time()
    section_started = time.perf_counter()
    try:
        if missing_indices:
            new_games, execution_metadata = evaluate_pair(
                agent_name,
                opponent_name,
                game_count,
                weights=weights,
                opponent_weights=opponent_weights,
                seed=seed,
                progress_callback=progress,
                suppress_agent_output=suppress_agent_output,
                workers=workers,
                safety_config=safety_config,
                game_indices=missing_indices,
                effective_seed=effective_seed,
                return_run_info=True,
            )
        else:
            new_games = []
            resolved_seed = _effective_seed(seed) if effective_seed is None else int(effective_seed)
            execution_metadata = {
                "requested_seed": seed,
                "effective_seed": resolved_seed,
                "parallel": ParallelRunInfo(
                    requested_workers=workers,
                    initial_workers=workers,
                    final_workers=workers,
                ).to_dict(),
            }
    finally:
        if progress_bar is not None:
            progress_bar.close()
    add_runtime("new_game_execution", section_started)

    section_started = time.perf_counter()
    games_by_index = dict(precomputed_by_index)
    games_by_index.update({int(record["game"]) - 1: record for record in new_games})
    if len(games_by_index) != game_count:
        raise RuntimeError(
            f"diagnostics produced {len(games_by_index)}/{game_count} unique games"
        )
    games = [games_by_index[index] for index in range(game_count)]
    duration = float(precomputed_duration_s) + (time.time() - start_time)
    add_runtime("parent_result_ordering", section_started)

    section_started = time.perf_counter()
    summary = summarize(
        games,
        agent_name,
        opponent_name,
        execution_metadata["effective_seed"],
    )
    summary["requested_seed"] = execution_metadata["requested_seed"]
    summary["effective_seed"] = execution_metadata["effective_seed"]
    summary["parallel"] = execution_metadata["parallel"]
    summary["precomputed_games"] = len(precomputed_games)
    summary = add_choice_summary(summary, games)
    summary = add_value_head_summary(summary, games)
    summary["duration_s"] = duration
    add_runtime("summary_statistics", section_started)
    section_started = time.perf_counter()
    if print_console_summary:
        print_summary(summary, duration)
        parallel = summary["parallel"]
        print(
            "  Workers: "
            f"requested {parallel['requested_workers']}, "
            f"initial {parallel['initial_workers']}, final {parallel['final_workers']}"
        )
        if parallel["fallback_count"]:
            print(f"  Memory/error fallbacks: {parallel['fallback_count']}")
        print(
            "  Peak worker RAM: "
            f"{parallel['peak_worker_rss_mb']:.1f} MiB each, "
            f"{parallel['peak_total_children_rss_mb']:.1f} MiB total"
        )
    add_runtime("console_summary", section_started)

    section_started = time.perf_counter()
    staging_dir = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.tmp-",
        dir=output_dir.parent,
    ))
    add_runtime("output_staging_setup", section_started)
    try:
        section_started = time.perf_counter()
        if save_game_records:
            save_csv(games, staging_dir / "games.csv")
        add_runtime("games_csv_write", section_started)
        section_started = time.perf_counter()
        with open(staging_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        add_runtime("summary_json_write", section_started)
        section_started = time.perf_counter()
        if generate_plots:
            write_diagnostic_plots(games, summary, staging_dir)
        add_runtime("diagnostic_plot_generation", section_started)
        section_started = time.perf_counter()
        _atomic_replace_directory(staging_dir, output_dir)
        add_runtime("atomic_output_commit", section_started)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    section_started = time.perf_counter()
    if print_console_summary:
        shown_output_dir = Path(display_output_dir) if display_output_dir else output_dir
        print(f"\nResults saved in {shown_output_dir}/")
        if generate_plots:
            print(
                "  cumulative_rates.png, result_distribution.png, wins_by_position.png, "
                "game_lengths.png, choice_opportunities.png"
            )
        artifacts = ["summary.json"]
        if save_game_records:
            artifacts.insert(0, "games.csv")
        print("  " + ", ".join(artifacts))
    add_runtime("final_console_output", section_started)

    runtime_total_seconds = time.perf_counter() - runtime_profile_started
    runtime_sections["unaccounted"] = max(
        0.0,
        runtime_total_seconds - sum(runtime_sections.values()),
    )
    game_worker_profile = {}
    _merge_numeric_runtime(
        game_worker_profile,
        dict(precomputed_runtime_profile or {}),
    )
    _merge_numeric_runtime(
        game_worker_profile,
        execution_metadata["parallel"].get("runtime_profile", {}),
    )

    return {
        "summary": summary,
        "games": games,
        "output_dir": str(output_dir),
        "duration_s": duration,
        "runtime_profile_delta": {
            "execution_seconds": float(runtime_total_seconds),
            "games": int(game_count),
            "new_games": int(len(new_games)),
            "precomputed_games": int(len(precomputed_games)),
            "sections_seconds": {
                name: float(seconds) for name, seconds in runtime_sections.items()
            },
            "game_worker": game_worker_profile,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one domino agent against another over N games.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--agent", choices=CANONICAL_AGENTS, default=DEFAULT_AGENT)
    parser.add_argument("--opponent", choices=CANONICAL_AGENTS, default=DEFAULT_OPPONENT)
    parser.add_argument(
        "-n",
        "--games",
        type=int,
        default=DEFAULT_GAME_COUNT,
        help="Number of games to play in this matchup.",
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_AGENT_WEIGHTS)
    parser.add_argument("--opponent-weights", type=Path, default=DEFAULT_OPPONENT_WEIGHTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder. Defaults to diagnostics/results/pairwise/<agent>_vs_<opponent>.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=not DEFAULT_GENERATE_PLOTS,
        help="Skip PNG generation and write only CSV/JSON plus console output.",
    )
    parser.add_argument(
        "--show-agent-output",
        action="store_true",
        help="Do not suppress print calls made by the agents during evaluation.",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        choices=range(1, MAX_DIAGNOSTIC_WORKERS + 1),
        default=DEFAULT_WORKERS,
        metavar="N",
        help=f"CPU-only diagnostic workers (maximum {MAX_DIAGNOSTIC_WORKERS}).",
    )
    args = parser.parse_args()

    run_pairwise(
        args.agent,
        args.opponent,
        game_count=args.games,
        weights=args.weights,
        opponent_weights=args.opponent_weights,
        seed=args.seed,
        output_dir=args.output,
        generate_plots=not args.no_plots,
        suppress_agent_output=not args.show_agent_output,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
