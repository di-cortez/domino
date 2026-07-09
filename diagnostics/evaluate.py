"""
Run the complete ordered diagnostics matrix for all supported domino agents.

This is the high-level diagnostics entry point. It intentionally delegates each
single matchup to ``diagnostics.pairwise`` so the two-agent evaluator remains the
only place that knows how to play games, summarize them, and write per-matchup
artifacts.
"""

import argparse
import csv
import json
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.pairwise import CANONICAL_AGENTS, DEFAULT_GAME_COUNT, run_pairwise
from diagnostics.plots import plot_all_pairs_table

DEFAULT_OUTPUT_DIR = ROOT / "diagnostics" / "results" / "all_pairs"


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
    """Write one row per ordered matchup."""
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
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_all_pairs(
    agents=CANONICAL_AGENTS,
    game_count=DEFAULT_GAME_COUNT,
    output_dir=DEFAULT_OUTPUT_DIR,
    seed=None,
    rl_weights=None,
    neural_weights=None,
    generate_pair_plots=True,
):
    """Evaluate every ordered pair of agents and write the aggregate matrix."""
    output_dir = Path(output_dir)
    pairs_dir = output_dir / "pairs"
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    total_pairs = len(agents) * len(agents)
    start_time = time.time()

    for index, (agent, opponent) in enumerate(product(agents, agents), start=1):
        print(f"\n[{index}/{total_pairs}] {agent} vs {opponent}", flush=True)
        pair_output = pairs_dir / f"{agent}_vs_{opponent}"
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
            print_console_summary=True,
        )
        summaries.append(result["summary"])

    rows = _matrix_rows(summaries)
    _save_matrix_csv(rows, output_dir / "all_pairs_matrix.csv")

    report = {
        "agents": list(agents),
        "game_count_per_matchup": game_count,
        "ordered_matchups": total_pairs,
        "seed": seed,
        "duration_s": time.time() - start_time,
        "summaries": summaries,
    }
    with open(output_dir / "all_pairs_summary.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    plot_all_pairs_table(summaries, agents, output_dir / "all_pairs_table.png")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all ordered pairs of supported domino agents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-n", "--games", type=int, default=DEFAULT_GAME_COUNT)
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
    )

    print("\n===== All-pairs diagnostics complete =====")
    print(f"Agents: {', '.join(report['agents'])}")
    print(f"Games per ordered matchup: {report['game_count_per_matchup']}")
    print(f"Ordered matchups: {report['ordered_matchups']}")
    print(f"Results saved in {Path(args.output)}/")
    print("  all_pairs_table.png")
    print("  all_pairs_matrix.csv")
    print("  all_pairs_summary.json")
    print("  pairs/<agent>_vs_<opponent>/...")


if __name__ == "__main__":
    main()
