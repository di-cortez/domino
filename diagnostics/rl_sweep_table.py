"""
Build a comparative table from ``train_script/run_rl_parameter_sweep.sh`` output.

Each RL sweep point directory under ``diagnostics/results/`` (one per
training+diagnostics run, e.g. ``domino_rl[_critic]_default/``,
``domino_rl[_critic]_lr<LR>_gamma<GAMMA>_gpi<GPI>/`` for the grid search, or
``domino_rl_critic_<grid tag>_vc<VC>/`` for the critic-on value_coef axis)
contains two JSON files written by
that script: ``sweep_run.json`` (the exact RL hyperparameters used) and
``summary.json`` (the rl-vs-random win/draw/loss rates from
``diagnostics.pairwise``). This module discovers every such directory, joins
the two JSON files into one row per run, and writes a comparative table --
CSV, an aggregate JSON, a console summary, and a PNG image table -- mirroring
``diagnostics/evaluate.py``'s all-pairs matrix output (``_matrix_rows`` /
``_save_matrix_csv`` / ``plot_all_pairs_table``). Rows are sorted directly on
the four numeric hyperparameters (not by parsing the tag string), so both the
grid-search rows and the value_coef-axis rows sort sensibly together.

Usage:
    python -m diagnostics.rl_sweep_table
    python -m diagnostics.rl_sweep_table --results-dir diagnostics/results
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.plots import plot_sweep_comparison_table

DEFAULT_RESULTS_DIR = ROOT / "diagnostics" / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "rl_sweep_table"

CSV_FIELDS = [
    "run_name", "critic", "varied_parameter", "learning_rate", "gamma",
    "games_per_iteration", "value_coef", "games", "wins", "draws", "losses",
    "win_rate", "draw_rate", "loss_rate", "win_ci95_low", "win_ci95_high",
    "mean_turns", "duration_s", "model_path",
]


def discover_sweep_runs(results_dir):
    """Return every immediate subdirectory of ``results_dir`` with both JSON files."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []
    run_dirs = []
    for entry in sorted(results_dir.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "sweep_run.json").exists() and (entry / "summary.json").exists():
            run_dirs.append(entry)
    return run_dirs


def _load_run(run_dir):
    """Join one sweep point's hyperparameters with its vs-random diagnostics."""
    with open(run_dir / "sweep_run.json", "r", encoding="utf-8") as f:
        hyperparameters = json.load(f)
    with open(run_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)
    return hyperparameters, summary


def build_rows(run_dirs):
    """Flatten each sweep point's hyperparameters + vs-random results into one row."""
    rows = []
    for run_dir in run_dirs:
        hyperparameters, summary = _load_run(run_dir)
        rates = summary["rates"]
        counts = summary["counts"]
        win_ci_low, win_ci_high = summary["win_ci95"]
        critic_enabled = bool(hyperparameters["critic_enabled"])
        rows.append({
            "run_name": hyperparameters["run_name"],
            "critic": "on" if critic_enabled else "off",
            "critic_enabled": critic_enabled,
            "varied_parameter": hyperparameters["varied_parameter"],
            "learning_rate": hyperparameters["learning_rate"],
            "gamma": hyperparameters["gamma"],
            "games_per_iteration": hyperparameters["games_per_iteration"],
            "value_coef": hyperparameters["value_coef"],
            "rl_iterations": hyperparameters["rl_iterations"],
            "seed": hyperparameters["seed"],
            "games": summary["game_count"],
            "wins": counts["win"],
            "draws": counts["draw"],
            "losses": counts["loss"],
            "win_rate": rates["win"],
            "draw_rate": rates["draw"],
            "loss_rate": rates["loss"],
            "win_rate_pct": round(100 * rates["win"], 1),
            "draw_rate_pct": round(100 * rates["draw"], 1),
            "win_ci95_low": win_ci_low,
            "win_ci95_high": win_ci_high,
            "mean_turns": summary["mean_turns"],
            "duration_s": summary.get("duration_s"),
            "model_path": hyperparameters["model_path"],
        })

    # Sort directly on the actual hyperparameter values rather than parsing
    # the tag string -- robust to both the grid tags (e.g.
    # "lr0.0005_gamma0.9_gpi80", which vary three values at once) and the
    # single-axis value_coef tags.
    rows.sort(
        key=lambda row: (
            row["critic_enabled"],
            row["learning_rate"],
            row["gamma"],
            row["games_per_iteration"],
            row["value_coef"],
        )
    )
    return rows


def _save_csv(rows, path):
    """Write one row per sweep point (mirrors evaluate.py's ``_save_matrix_csv``)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in CSV_FIELDS} for row in rows)


def _print_console_table(rows):
    """Print a compact aligned comparison table to the console."""
    if not rows:
        print("No sweep runs found (looked for directories with both "
              "sweep_run.json and summary.json).")
        return

    headers = ["Run", "Critic", "Varied", "LR", "Gamma", "G/Iter", "VCoef", "Win%", "Draw%", "Games"]
    table_rows = []
    for row in rows:
        table_rows.append([
            row["run_name"],
            row["critic"],
            row["varied_parameter"],
            f"{row['learning_rate']:g}",
            f"{row['gamma']:g}",
            str(row["games_per_iteration"]),
            f"{row['value_coef']:g}",
            f"{row['win_rate_pct']:.1f}",
            f"{row['draw_rate_pct']:.1f}",
            str(row["games"]),
        ])

    widths = [
        max(len(headers[i]), *(len(table_row[i]) for table_row in table_rows))
        for i in range(len(headers))
    ]

    def _line(cells):
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths))

    print(_line(headers))
    print("  ".join("-" * width for width in widths))
    for table_row in table_rows:
        print(_line(table_row))


def build_report(results_dir=DEFAULT_RESULTS_DIR, output_dir=DEFAULT_OUTPUT_DIR, quiet=False):
    """Discover every RL sweep point under ``results_dir`` and write the comparative table.

    Writes ``rl_sweep_table.csv``, ``rl_sweep_table.json``, and
    ``rl_sweep_table.png`` to ``output_dir``. Returns the list of row dicts.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_sweep_runs(results_dir)
    rows = build_rows(run_dirs)

    _save_csv(rows, output_dir / "rl_sweep_table.csv")
    with open(output_dir / "rl_sweep_table.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    plot_sweep_comparison_table(rows, output_dir / "rl_sweep_table.png")

    if not quiet:
        print(f"\nRL hyperparameter sweep comparison ({len(rows)} runs found in {results_dir}/)")
        _print_console_table(rows)
        print(
            f"\nSaved: {output_dir}/rl_sweep_table.csv, "
            "rl_sweep_table.json, rl_sweep_table.png"
        )

    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Build a comparative table from train_script/run_rl_parameter_sweep.sh "
            "output: joins each sweep point's sweep_run.json (hyperparameters) with "
            "its summary.json (rl-vs-random win/draw/loss rates)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory to scan for sweep-point subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write the comparative table into.",
    )
    parser.add_argument("--quiet", action="store_true", help="Skip console output.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    build_report(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
