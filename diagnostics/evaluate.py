"""
Evaluate every supported domino agent against the random baseline.

This is the high-level diagnostics entry point. It intentionally delegates each
single matchup to ``diagnostics.pairwise`` so the two-agent evaluator remains the
only place that knows how to play games, summarize them, and write per-matchup
artifacts. The workload is explicit: every canonical agent is evaluated
against ``random`` for the requested number of games per matchup.
"""

import argparse
import csv
import json
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

from diagnostics.pairwise import (
    CANONICAL_AGENTS,
    DEFAULT_GAME_COUNT,
    _atomic_replace_directory,
    resolve_weights_path,
    run_pairwise,
)
from diagnostics.parallel_runner import (
    DEFAULT_ESTIMATED_WORKER_MB,
    DEFAULT_MAX_WORKER_RSS_MB,
    DEFAULT_MEMORY_RESERVE_MB,
    MAX_DIAGNOSTIC_WORKERS,
    ParallelSafetyConfig,
    cap_parallel_workers,
    game_seed,
)
from diagnostics.plots import (
    plot_aggregate_choice_opportunities,
    plot_all_pairs_table,
    worst_case_margin_of_error,
)
from utils.artifacts import file_sha256
from utils.runtime_status import format_duration, print_memory_report
from diagnostics.worker_autotune import (
    DEFAULT_AUTOTUNE_FRACTION,
    DEFAULT_MINIMUM_GAIN,
    MatchupSpec,
    autotune_diagnostic_workers,
)

DEFAULT_OUTPUT_DIR = ROOT / "diagnostics" / "results" / "all_pairs"
RANDOM_BASELINE_OPPONENT = "random"
RANDOM_BASELINE_MATCHUPS = tuple(
    (agent, RANDOM_BASELINE_OPPONENT)
    for agent in CANONICAL_AGENTS
)
DEFAULT_DIAGNOSTIC_WORKERS = "auto"
POLICY_WEIGHT_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")
VALUE_WEIGHT_NAMES = ("Wv", "bv")
RANDOM_NN_ARCHITECTURE = (168, 256, 128, 56)


def _weights_for(agent_name, rl_weights=None, neural_weights=None):
    """Return the optional checkpoint path that belongs to ``agent_name``."""
    if agent_name == "rl":
        return rl_weights
    if agent_name == "neural":
        return neural_weights
    return None


def _matrix_rows(summaries):
    """Flatten matchup summaries into CSV-friendly rows."""
    rows = []
    for summary in summaries:
        counts = summary["counts"]
        rates = summary["rates"]
        rows.append({
            "agent": summary["agent"],
            "opponent": summary["opponent"],
            "games": summary["game_count"],
            "wins": counts["win"],
            "draws": counts["draw"],
            "losses": counts["loss"],
            "win_rate": rates["win"],
            "draw_rate": rates["draw"],
            "loss_rate": rates["loss"],
            "mean_turns": summary["mean_turns"],
        })
    return rows


def _save_matrix_csv(rows, path):
    """Write one row per evaluated matchup."""
    fields = [
        "agent",
        "opponent",
        "games",
        "wins",
        "draws",
        "losses",
        "win_rate",
        "draw_rate",
        "loss_rate",
        "mean_turns",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _selected_pairs(agents):
    """Return one random-baseline matchup for every selected agent."""
    return tuple((agent, RANDOM_BASELINE_OPPONENT) for agent in agents)


def diagnostic_plan():
    """Return the fixed five-agent random-baseline plan."""
    return CANONICAL_AGENTS, RANDOM_BASELINE_MATCHUPS


def _checkpoint_network_metadata(agent, weights_path):
    """Describe one checkpoint architecture without constructing its agent."""
    path = Path(weights_path)
    with np.load(path, allow_pickle=False) as weights:
        architecture = [
            int(weights["W1"].shape[1]),
            int(weights["W1"].shape[0]),
            int(weights["W2"].shape[0]),
            int(weights["W3"].shape[0]),
        ]
        available_names = set(weights.files)
        policy_parameters = sum(
            int(weights[name].size)
            for name in POLICY_WEIGHT_NAMES
        )
        value_head = all(name in available_names for name in VALUE_WEIGHT_NAMES)
        value_parameters = (
            sum(int(weights[name].size) for name in VALUE_WEIGHT_NAMES)
            if value_head
            else 0
        )
    return {
        "agent": agent,
        "architecture": architecture,
        "policy_parameters": policy_parameters,
        "value_head": value_head,
        "value_parameters": value_parameters,
        "total_parameters": policy_parameters + value_parameters,
        "checkpoint": str(path),
        "checkpoint_name": path.name,
        "checkpoint_sha256": file_sha256(path),
        "checkpoint_bytes": path.stat().st_size,
    }


def _network_metadata(agents, matchup_specs):
    """Collect compact architecture metadata for neural agents in the report."""
    checkpoint_paths = {
        matchup.agent: matchup.weights
        for matchup in matchup_specs
        if matchup.weights is not None
    }
    metadata = {}
    for agent in ("rl", "neural"):
        if agent in agents and agent in checkpoint_paths:
            metadata[agent] = _checkpoint_network_metadata(
                agent,
                checkpoint_paths[agent],
            )

    if "random_nn" in agents:
        architecture = list(RANDOM_NN_ARCHITECTURE)
        policy_parameters = (
            architecture[1] * architecture[0] + architecture[1]
            + architecture[2] * architecture[1] + architecture[2]
            + architecture[3] * architecture[2] + architecture[3]
        )
        metadata["random_nn"] = {
            "agent": "random_nn",
            "architecture": architecture,
            "policy_parameters": policy_parameters,
            "value_head": False,
            "value_parameters": 0,
            "total_parameters": policy_parameters,
            "checkpoint": None,
            "checkpoint_name": None,
            "initialization": "untrained fixed seed 0",
        }
    return metadata


def _remove_stale_pair_outputs(pairs_dir, pairs):
    """Remove pair folders that do not belong to the selected plan."""
    selected_folders = {f"{agent}_vs_{opponent}" for agent, opponent in pairs}
    for path in pairs_dir.iterdir():
        if path.is_dir() and path.name not in selected_folders:
            shutil.rmtree(path)


def _aggregate_choice_opportunities(summaries):
    """Accumulate choice-opportunity stats across all evaluated matchups."""
    totals = {
        "matchups": len(summaries),
        "evaluated_agent_turns": 0,
        "real_decision_turns": 0,
        "real_decision_rate": 0.0,
        "forced_tile_turns": 0,
        "forced_draws": 0,
        "forced_passes": 0,
        "choice_histogram": {},
    }

    for summary in summaries:
        choice_info = summary.get("choice_opportunities", {})
        totals["evaluated_agent_turns"] += choice_info.get("evaluated_agent_turns", 0)
        totals["real_decision_turns"] += choice_info.get("real_decision_turns", 0)
        totals["forced_tile_turns"] += choice_info.get("forced_tile_turns", 0)
        totals["forced_draws"] += choice_info.get("forced_draws", 0)
        totals["forced_passes"] += choice_info.get("forced_passes", 0)

        for option_count, count in choice_info.get("choice_histogram", {}).items():
            histogram = totals["choice_histogram"]
            histogram[option_count] = histogram.get(option_count, 0) + count

    if totals["evaluated_agent_turns"]:
        totals["real_decision_rate"] = (
            totals["real_decision_turns"] / totals["evaluated_agent_turns"]
        )

    totals["choice_histogram"] = dict(
        sorted(totals["choice_histogram"].items(), key=lambda item: int(item[0]))
    )
    return totals


def run_all_pairs(
    agents=None,
    game_count=DEFAULT_GAME_COUNT,
    output_dir=DEFAULT_OUTPUT_DIR,
    seed=None,
    rl_weights=None,
    neural_weights=None,
    generate_pair_plots=True,
    quiet=False,
    progress_callback=None,
    workers=DEFAULT_DIAGNOSTIC_WORKERS,
    safety_config=None,
    autotune_fraction=DEFAULT_AUTOTUNE_FRACTION,
    autotune_minimum_gain=DEFAULT_MINIMUM_GAIN,
    status_callback=None,
):
    """Evaluate canonical agents and write aggregate artifacts.

    Passing ``agents`` retains support for a custom subset, but every selected
    agent is still evaluated only against the random baseline. When omitted,
    all five canonical agents are evaluated.
    """
    if game_count < 1:
        raise ValueError("game_count must be positive")
    if agents is None:
        agents, pairs = diagnostic_plan()
    else:
        agents = tuple(agents)
        pairs = tuple(_selected_pairs(agents))

    if workers != "auto":
        workers = int(workers)
        if not 1 <= workers <= MAX_DIAGNOSTIC_WORKERS:
            raise ValueError(
                f"workers must be 'auto' or between 1 and {MAX_DIAGNOSTIC_WORKERS}"
            )
    safety_config = safety_config or ParallelSafetyConfig()
    effective_seed = int(seed) if seed is not None else secrets.randbits(63)

    final_output_dir = Path(output_dir)
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print_memory_report("Agent-vs-random diagnostics startup memory")

    summaries = []
    total_pairs = len(pairs)
    completed_games = 0
    start_time = time.time()

    matchup_specs = []
    for agent, opponent in pairs:
        agent_weights = _weights_for(
            agent,
            rl_weights=rl_weights,
            neural_weights=neural_weights,
        )
        resolved_opponent_weights = _weights_for(
            opponent,
            rl_weights=rl_weights,
            neural_weights=neural_weights,
        )
        matchup_specs.append(MatchupSpec(
            agent=agent,
            opponent=opponent,
            weights=resolve_weights_path(agent, agent_weights),
            opponent_weights=resolve_weights_path(opponent, resolved_opponent_weights),
        ))
    matchup_specs = tuple(matchup_specs)
    network_metadata = _network_metadata(agents, matchup_specs)

    output_dir = Path(tempfile.mkdtemp(
        prefix=f".{final_output_dir.name}.tmp-",
        dir=final_output_dir.parent,
    ))
    pairs_dir = output_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    fixed_workers = None
    if workers != "auto":
        fixed_workers, fixed_was_capped, fixed_cap_reason = cap_parallel_workers(
            workers,
            safety_config,
        )
        if fixed_was_capped:
            emit_status = status_callback or (
                lambda message: print(message, flush=True)
            )
            emit_status(
                f"Fixed workers reduced from {workers} to {fixed_workers} by "
                f"resource preflight: {fixed_cap_reason}."
            )

    selected_workers_by_matchup = {}
    autotune_by_matchup = {}
    total_reused_games = 0

    for index, (agent, opponent) in enumerate(pairs, start=1):
        if not quiet:
            print(f"\n[{index}/{total_pairs}] {agent} vs {opponent}", flush=True)
        pair_output = pairs_dir / f"{agent}_vs_{opponent}"

        def pair_progress(_done, _total):
            nonlocal completed_games
            completed_games += 1
            if progress_callback is not None:
                progress_callback(completed_games, total_pairs * game_count)

        matchup = matchup_specs[index - 1]
        matchup_report_key = f"{agent}_vs_{opponent}"
        pair_seed = game_seed(effective_seed, index - 1)
        try:
            if workers == "auto":
                tuning_progress_previous = 0

                def tuning_progress(done, _total):
                    nonlocal tuning_progress_previous, completed_games
                    increment = max(0, done - tuning_progress_previous)
                    tuning_progress_previous = done
                    completed_games += increment
                    if progress_callback is not None:
                        progress_callback(
                            completed_games,
                            total_pairs * game_count,
                        )

                tuning = autotune_diagnostic_workers(
                    matchups=(matchup,),
                    game_count=game_count,
                    base_seed=effective_seed,
                    safety=safety_config,
                    benchmark_fraction=autotune_fraction,
                    minimum_gain=autotune_minimum_gain,
                    progress_callback=tuning_progress,
                    status_callback=status_callback,
                    pair_seed_overrides={matchup.key: pair_seed},
                )
                selected_workers = tuning["optimal_workers"]
                retained_games = tuning["precomputed_games"].pop(matchup.key)
                retained_duration = tuning["durations_by_matchup"].pop(matchup.key)
                retained_runtime_profile = tuning[
                    "runtime_profiles_by_matchup"
                ].pop(matchup.key)
                total_reused_games += tuning["reused_game_count"]
                autotune_by_matchup[matchup_report_key] = {
                    "optimal_workers": tuning["optimal_workers"],
                    "candidate_workers": tuning["candidate_workers"],
                    "games_per_test": tuning["games_per_test"],
                    "reused_game_count": tuning["reused_game_count"],
                    "attempts": tuning["attempts"],
                }
            else:
                selected_workers = fixed_workers
                retained_games = []
                retained_duration = 0.0
                retained_runtime_profile = {}

            selected_workers_by_matchup[matchup_report_key] = selected_workers
            result = run_pairwise(
                agent,
                opponent,
                game_count=game_count,
                weights=matchup.weights,
                opponent_weights=matchup.opponent_weights,
                seed=seed,
                output_dir=pair_output,
                generate_plots=generate_pair_plots,
                print_console_summary=not quiet,
                print_memory_summary=False,
                progress_callback=pair_progress,
                workers=selected_workers,
                safety_config=safety_config,
                precomputed_games=retained_games,
                precomputed_duration_s=retained_duration,
                precomputed_runtime_profile=retained_runtime_profile,
                effective_seed=pair_seed,
                display_output_dir=(
                    final_output_dir / "pairs" / f"{agent}_vs_{opponent}"
                ),
            )
        except BaseException:
            shutil.rmtree(output_dir, ignore_errors=True)
            raise
        summaries.append(result["summary"])
        del result, retained_games

    rows = _matrix_rows(summaries)
    choice_opportunities = _aggregate_choice_opportunities(summaries)
    report = {
        "choice_opportunities": choice_opportunities,
        "comparison_opponent": RANDOM_BASELINE_OPPONENT,
        "report_layout": "single_row",
        "agents": list(agents),
        "game_count_per_matchup": game_count,
        "evaluated_matchups": total_pairs,
        "seed": seed,
        "effective_seed": effective_seed,
        "requested_workers": workers,
        "selected_workers_by_matchup": selected_workers_by_matchup,
        "autotune": {
            "scope": "per_matchup",
            "benchmark_fraction": (
                autotune_fraction if workers == "auto" else 0.0
            ),
            "minimum_gain": autotune_minimum_gain,
            "reused_game_count": total_reused_games,
            "matchups": autotune_by_matchup,
        },
        "duration_s": time.time() - start_time,
        "win_rate_margin_of_error_95": {
            "method": "normal approximation, worst case p=0.5",
            "proportion": worst_case_margin_of_error(game_count),
            "percentage_points": 100 * worst_case_margin_of_error(game_count),
        },
        "network_metadata": network_metadata,
        "summaries": summaries,
    }
    try:
        _save_matrix_csv(rows, output_dir / "all_pairs_matrix.csv")
        with open(output_dir / "all_pairs_summary.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        plot_all_pairs_table(
            summaries,
            agents,
            output_dir / "all_pairs_table.png",
            report_metadata=report,
        )
        plot_all_pairs_table(
            summaries,
            agents,
            output_dir / "all_pairs_table.pdf",
            report_metadata=report,
        )
        plot_aggregate_choice_opportunities(
            choice_opportunities,
            output_dir / "choice_opportunities.png",
        )
        _atomic_replace_directory(output_dir, final_output_dir)
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    return report


def _worker_count(value):
    """Parse ``auto`` or a diagnostic worker count within the hard limit."""
    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("workers must be 'auto' or an integer") from exc
    if not 1 <= parsed <= MAX_DIAGNOSTIC_WORKERS:
        raise argparse.ArgumentTypeError(
            f"workers must be between 1 and {MAX_DIAGNOSTIC_WORKERS}"
        )
    return parsed


def main():
    """Parse the CLI and run the fixed agent-vs-random comparisons."""
    parser = argparse.ArgumentParser(
        description="Evaluate every supported domino agent against random.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--games",
        type=int,
        default=DEFAULT_GAME_COUNT,
        help="Number of games per evaluated matchup.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rl-weights", type=Path, default=None)
    parser.add_argument("--neural-weights", type=Path, default=None)
    parser.add_argument(
        "--no-pair-plots",
        action="store_true",
        help=(
            "Skip per-matchup PNG plots. The aggregate PNG and PDF are still "
            "generated."
        ),
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=_worker_count,
        default=DEFAULT_DIAGNOSTIC_WORKERS,
        help=f"CPU-only workers or 'auto' for online tuning (maximum {MAX_DIAGNOSTIC_WORKERS}).",
    )
    parser.add_argument("--autotune-fraction", type=float, default=DEFAULT_AUTOTUNE_FRACTION)
    parser.add_argument("--autotune-min-gain", type=float, default=DEFAULT_MINIMUM_GAIN)
    parser.add_argument("--memory-reserve-mb", type=int, default=DEFAULT_MEMORY_RESERVE_MB)
    parser.add_argument("--estimated-worker-mb", type=int, default=DEFAULT_ESTIMATED_WORKER_MB)
    parser.add_argument("--max-worker-rss-mb", type=int, default=DEFAULT_MAX_WORKER_RSS_MB)
    args = parser.parse_args()

    report = run_all_pairs(
        game_count=args.games,
        output_dir=args.output,
        seed=args.seed,
        rl_weights=args.rl_weights,
        neural_weights=args.neural_weights,
        generate_pair_plots=not args.no_pair_plots,
        workers=args.workers,
        safety_config=ParallelSafetyConfig(
            memory_reserve_mb=args.memory_reserve_mb,
            estimated_worker_mb=args.estimated_worker_mb,
            max_worker_rss_mb=args.max_worker_rss_mb,
        ),
        autotune_fraction=args.autotune_fraction,
        autotune_minimum_gain=args.autotune_min_gain,
    )

    print("\n===== Agent-vs-random diagnostics complete =====")
    print(f"Agents: {', '.join(report['agents'])}")
    print(f"Games per matchup: {report['game_count_per_matchup']}")
    print(f"Evaluated matchups: {report['evaluated_matchups']}")
    print(f"Comparison opponent: {report['comparison_opponent']}")
    print(f"Elapsed time: {format_duration(report['duration_s'])}")
    print("Diagnostic workers selected per matchup:")
    for matchup, worker_count in report["selected_workers_by_matchup"].items():
        print(f"  {matchup}: {worker_count}")
    print(f"Results saved in {Path(args.output)}/")
    print("  all_pairs_table.png")
    print("  all_pairs_table.pdf")
    print("  choice_opportunities.png")
    print("  all_pairs_matrix.csv")
    print("  all_pairs_summary.json")
    print("  pairs/<agent>_vs_<opponent>/...")


if __name__ == "__main__":
    main()
