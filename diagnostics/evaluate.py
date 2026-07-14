"""
Run a selected diagnostics matrix for the supported domino agents.

This is the high-level diagnostics entry point. It intentionally delegates each
single matchup to ``diagnostics.pairwise`` so the two-agent evaluator remains the
only place that knows how to play games, summarize them, and write per-matchup
artifacts. The optional mode controls whether the run uses a focused two-pair
check, the historical four-agent matrix, or the complete five-agent matrix.
"""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.pairwise import (
    CANONICAL_AGENTS,
    DEFAULT_GAME_COUNT,
    remove_legacy_artifacts,
    run_pairwise,
)
from diagnostics.plots import (
    plot_aggregate_choice_opportunities,
    plot_all_pairs_table,
)
from utils.runtime_status import format_duration, print_memory_report

DEFAULT_OUTPUT_DIR = ROOT / "diagnostics" / "results" / "all_pairs"
DEFAULT_DIAGNOSTIC_MODE = "default"
DIAGNOSTIC_MODES = ("default", "fast", "complete")
DEFAULT_MATRIX_AGENTS = ("rl", "neural", "heuristic", "random")
FAST_MATRIX_AGENTS = ("rl", "heuristic", "random")
FAST_MATCHUPS = (("rl", "random"), ("heuristic", "random"))


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
    """Return only A-vs-B pairs from the upper triangle, keeping A-vs-A controls."""
    return [
        (agent, opponent)
        for agent_index, agent in enumerate(agents)
        for opponent in agents[agent_index:]
    ]


def diagnostic_plan(mode=DEFAULT_DIAGNOSTIC_MODE):
    """Return the displayed agents and evaluated pairs for a diagnostics mode."""
    if mode == "fast":
        return FAST_MATRIX_AGENTS, FAST_MATCHUPS
    if mode == "default":
        return DEFAULT_MATRIX_AGENTS, tuple(_selected_pairs(DEFAULT_MATRIX_AGENTS))
    if mode == "complete":
        return CANONICAL_AGENTS, tuple(_selected_pairs(CANONICAL_AGENTS))
    raise ValueError(f"Unknown diagnostics mode {mode!r}. Options: {DIAGNOSTIC_MODES}")


def _remove_stale_pair_outputs(pairs_dir, pairs):
    """Remove pair folders that do not belong to the selected diagnostics mode."""
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
    diagnostic_mode=DEFAULT_DIAGNOSTIC_MODE,
):
    """Evaluate one diagnostics mode and write its aggregate artifacts.

    Passing ``agents`` retains support for custom upper-triangle matrices. When
    omitted, ``diagnostic_mode`` selects one of the standard plans.
    """
    if agents is None:
        agents, pairs = diagnostic_plan(diagnostic_mode)
        report_mode = diagnostic_mode
    else:
        agents = tuple(agents)
        pairs = tuple(_selected_pairs(agents))
        report_mode = "custom"

    output_dir = Path(output_dir)
    pairs_dir = output_dir / "pairs"
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)
    remove_legacy_artifacts(output_dir)
    _remove_stale_pair_outputs(pairs_dir, pairs)

    if not quiet:
        print_memory_report("All-pairs diagnostics startup memory")

    summaries = []
    total_pairs = len(pairs)
    completed_games = 0
    start_time = time.time()

    for index, (agent, opponent) in enumerate(pairs, start=1):
        if not quiet:
            print(f"\n[{index}/{total_pairs}] {agent} vs {opponent}", flush=True)
        pair_output = pairs_dir / f"{agent}_vs_{opponent}"

        def pair_progress(_done, _total):
            nonlocal completed_games
            completed_games += 1
            if progress_callback is not None:
                progress_callback(completed_games, total_pairs * game_count)

        result = run_pairwise(
            agent,
            opponent,
            game_count=game_count,
            weights=_weights_for(agent, rl_weights=rl_weights, neural_weights=neural_weights),
            opponent_weights=_weights_for(
                opponent,
                rl_weights=rl_weights,
                neural_weights=neural_weights,
            ),
            seed=seed,
            output_dir=pair_output,
            generate_plots=generate_pair_plots,
            print_console_summary=not quiet,
            print_memory_summary=False,
            progress_callback=pair_progress,
        )
        summaries.append(result["summary"])

    rows = _matrix_rows(summaries)
    _save_matrix_csv(rows, output_dir / "all_pairs_matrix.csv")
    choice_opportunities = _aggregate_choice_opportunities(summaries)

    report = {
        "choice_opportunities": choice_opportunities,
        "diagnostic_mode": report_mode,
        "agents": list(agents),
        "game_count_per_matchup": game_count,
        "evaluated_matchups": total_pairs,
        "unevaluated_matrix_matchups": len(agents) * len(agents) - total_pairs,
        "seed": seed,
        "duration_s": time.time() - start_time,
        "summaries": summaries,
    }
    with open(output_dir / "all_pairs_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    plot_all_pairs_table(summaries, agents, output_dir / "all_pairs_table.png")
    plot_aggregate_choice_opportunities(
        choice_opportunities,
        output_dir / "choice_opportunities.png",
    )
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a selected matrix of supported domino agents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=DIAGNOSTIC_MODES,
        default=DEFAULT_DIAGNOSTIC_MODE,
        help=(
            "Diagnostic scope: default uses the historical 10 matchups, fast "
            "uses 2 focused matchups, and complete uses all 15 matchups."
        ),
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
        help="Skip the per-matchup PNG plots. The aggregate table image is still generated.",
    )
    args = parser.parse_args()

    report = run_all_pairs(
        game_count=args.games,
        output_dir=args.output,
        seed=args.seed,
        rl_weights=args.rl_weights,
        neural_weights=args.neural_weights,
        generate_pair_plots=not args.no_pair_plots,
        diagnostic_mode=args.mode,
    )

    print("\n===== All-pairs diagnostics complete =====")
    print(f"Mode: {report['diagnostic_mode']}")
    print(f"Agents: {', '.join(report['agents'])}")
    print(f"Games per matchup: {report['game_count_per_matchup']}")
    print(f"Evaluated matchups: {report['evaluated_matchups']}")
    print(f"Unevaluated matrix matchups: {report['unevaluated_matrix_matchups']}")
    print(f"Elapsed time: {format_duration(report['duration_s'])}")
    print(f"Results saved in {Path(args.output)}/")
    print("  all_pairs_table.png")
    print("  choice_opportunities.png")
    print("  all_pairs_matrix.csv")
    print("  all_pairs_summary.json")
    print("  pairs/<agent>_vs_<opponent>/...")


if __name__ == "__main__":
    main()
