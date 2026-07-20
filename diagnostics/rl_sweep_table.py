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
CSV, an aggregate JSON, a console summary, and PNG/PDF visual tables. The raw
CSV/JSON retain one row per trained model. For the less cluttered console and
PNG/PDF presentation, runs that differ only by games-per-iteration are pivoted
into one row with win-rate columns labelled 40, 80, and 160. Rows are sorted
directly on numeric hyperparameters rather than by parsing tag strings.

Usage:
    python -m diagnostics.rl_sweep_table
    python -m diagnostics.rl_sweep_table --results-dir diagnostics/results
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.plots import plot_sweep_comparison_table

DEFAULT_RESULTS_DIR = ROOT / "diagnostics" / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "rl_sweep_table"
DEFAULT_GAMES_PER_ITERATION_COLUMNS = (40, 80, 160)
SWEEP_DIAGNOSTIC_PLOT_FILES = (
    "cumulative_rates.png",
    "result_distribution.png",
    "wins_by_position.png",
    "game_lengths.png",
    "choice_opportunities.png",
)

CSV_FIELDS = [
    "run_name", "critic", "varied_parameter", "learning_rate", "gamma",
    "games_per_iteration", "value_coef", "games", "wins", "draws", "losses",
    "win_rate", "draw_rate", "loss_rate", "win_ci95_low", "win_ci95_high",
    "mean_turns", "duration_s", "model_path",
]


def file_sha256(path):
    """Return the hexadecimal SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_reusable_sweep_diagnostic(
    run_dir,
    expected_metadata,
    model_path,
    require_plots=True,
):
    """Validate that a completed sweep diagnostic can be safely reused.

    The JSON configuration, model identity, requested seed, matchup, counts,
    CSV row count, and requested artifacts must all match the current sweep
    point. New metadata records a model SHA-256 digest. Older diagnostics are
    accepted for backward compatibility only when every artifact is at least
    as new as the exact numbered model checkpoint.

    Returns ``(is_valid, reason)`` so callers can log or test failures without
    treating stale output as a fatal error.
    """
    run_dir = Path(run_dir)
    model_path = Path(model_path)
    metadata_path = run_dir / "sweep_run.json"
    summary_path = run_dir / "summary.json"
    games_path = run_dir / "games.csv"
    artifact_paths = [metadata_path, summary_path, games_path]
    if require_plots:
        artifact_paths.extend(run_dir / name for name in SWEEP_DIAGNOSTIC_PLOT_FILES)

    if not model_path.is_file():
        return False, f"model checkpoint is missing: {model_path}"
    missing = [str(path) for path in artifact_paths if not path.is_file()]
    if missing:
        return False, f"diagnostic artifact is missing: {missing[0]}"

    try:
        with open(metadata_path, "r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        with open(summary_path, "r", encoding="utf-8") as stream:
            summary = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, f"diagnostic JSON is unreadable: {exc}"

    for key, expected in expected_metadata.items():
        actual = metadata.get(key)
        if isinstance(expected, bool):
            matches = isinstance(actual, bool) and actual is expected
        elif isinstance(expected, float):
            try:
                matches = math.isclose(
                    float(actual), expected, rel_tol=0.0, abs_tol=1e-15
                )
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            return False, (
                f"sweep metadata mismatch for {key}: "
                f"expected {expected!r}, found {actual!r}"
            )

    expected_games = int(expected_metadata["diagnostic_games"])
    expected_seed = int(expected_metadata["seed"])
    if summary.get("agent") != "rl" or summary.get("opponent") != "random":
        return False, "diagnostic matchup is not rl vs random"
    if summary.get("game_count") != expected_games:
        return False, "diagnostic game count does not match the current request"
    for seed_key in ("seed", "requested_seed", "effective_seed"):
        if summary.get(seed_key) != expected_seed:
            return False, f"diagnostic {seed_key} does not match the current request"

    counts = summary.get("counts")
    rates = summary.get("rates")
    if not isinstance(counts, dict) or not isinstance(rates, dict):
        return False, "diagnostic counts or rates are missing"
    try:
        count_total = sum(int(counts[key]) for key in ("win", "draw", "loss"))
        for key in ("win", "draw", "loss"):
            rate = float(rates[key])
            if not math.isfinite(rate) or not math.isclose(
                rate,
                int(counts[key]) / expected_games,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                return False, f"diagnostic {key} rate is inconsistent with its count"
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False, "diagnostic counts or rates are invalid"
    if count_total != expected_games:
        return False, "diagnostic result counts do not add up to the game count"
    win_ci95 = summary.get("win_ci95")
    if not isinstance(win_ci95, list) or len(win_ci95) != 2:
        return False, "diagnostic confidence interval is missing"
    if "mean_turns" not in summary:
        return False, "diagnostic mean-turn count is missing"

    try:
        with open(games_path, "r", encoding="utf-8", newline="") as stream:
            csv_row_count = sum(1 for _row in csv.reader(stream)) - 1
    except (OSError, UnicodeError, csv.Error) as exc:
        return False, f"diagnostic games CSV is unreadable: {exc}"
    if csv_row_count != expected_games:
        return False, (
            f"diagnostic games CSV has {csv_row_count} rows; "
            f"expected {expected_games}"
        )

    stored_hash = metadata.get("model_sha256")
    if stored_hash:
        try:
            if stored_hash != file_sha256(model_path):
                return False, "diagnostic model checksum does not match the checkpoint"
        except OSError as exc:
            return False, f"model checkpoint cannot be hashed: {exc}"
    else:
        model_mtime = model_path.stat().st_mtime_ns
        if any(path.stat().st_mtime_ns < model_mtime for path in artifact_paths):
            return False, "legacy diagnostic artifacts predate the model checkpoint"

    return True, "diagnostic is complete and compatible"


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


def build_display_rows(
    rows,
    games_per_iteration_values=DEFAULT_GAMES_PER_ITERATION_COLUMNS,
):
    """Pivot per-model rows into compact games-per-iteration win-rate columns.

    Training output remains untouched: every input row still represents one
    model and remains present in the raw CSV/JSON. This view groups only runs
    with identical critic, learning-rate, gamma, value-coefficient, iteration,
    and seed settings. Duplicate results for the same group and GPI are
    rejected instead of being silently overwritten.
    """
    observed_values = sorted({
        int(row["games_per_iteration"])
        for row in rows
    })
    column_values = tuple(dict.fromkeys(
        [int(value) for value in games_per_iteration_values] + observed_values
    ))
    grouped = {}

    for row in rows:
        key = (
            bool(row["critic_enabled"]),
            float(row["learning_rate"]),
            float(row["gamma"]),
            float(row["value_coef"]),
            int(row["rl_iterations"]),
            row.get("seed"),
        )
        display_row = grouped.setdefault(key, {
            "critic": row["critic"],
            "critic_enabled": bool(row["critic_enabled"]),
            "learning_rate": row["learning_rate"],
            "gamma": row["gamma"],
            "value_coef": row["value_coef"],
            "rl_iterations": row["rl_iterations"],
            "seed": row.get("seed"),
        })
        gpi = int(row["games_per_iteration"])
        field = f"win_rate_pct_gpi_{gpi}"
        if field in display_row:
            raise ValueError(
                "multiple sweep results have the same displayed configuration: "
                f"critic={row['critic']}, learning_rate={row['learning_rate']}, "
                f"gamma={row['gamma']}, value_coef={row['value_coef']}, "
                f"games_per_iteration={gpi}"
            )
        display_row[field] = row["win_rate_pct"]

    display_rows = list(grouped.values())
    for row in display_rows:
        for gpi in column_values:
            row.setdefault(f"win_rate_pct_gpi_{gpi}", "")
    display_rows.sort(
        key=lambda row: (
            row["critic_enabled"],
            row["learning_rate"],
            row["gamma"],
            row["value_coef"],
            row["rl_iterations"],
            -1 if row["seed"] is None else row["seed"],
        )
    )
    return display_rows, column_values


def _save_csv(rows, path):
    """Write one row per sweep point (mirrors evaluate.py's ``_save_matrix_csv``)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in CSV_FIELDS} for row in rows)


def _format_win_rate(value):
    """Format one optional percentage for the compact display."""
    return "" if value == "" or value is None else f"{float(value):.1f}%"


def _print_console_table(rows, games_per_iteration_values):
    """Print the pivoted games-per-iteration comparison to the console."""
    if not rows:
        print("No sweep runs found (looked for directories with both "
              "sweep_run.json and summary.json).")
        return

    headers = ["Critic", "LR", "Gamma", "VCoef"] + [
        str(value) for value in games_per_iteration_values
    ]
    table_rows = []
    for row in rows:
        base_cells = [
            row["critic"],
            f"{row['learning_rate']:g}",
            f"{row['gamma']:g}",
            f"{row['value_coef']:g}",
        ]
        win_cells = [
            _format_win_rate(row[f"win_rate_pct_gpi_{value}"])
            for value in games_per_iteration_values
        ]
        table_rows.append(base_cells + win_cells)

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

    Writes ``rl_sweep_table.csv``, ``rl_sweep_table.json``,
    ``rl_sweep_table.png``, and ``rl_sweep_table.pdf`` to ``output_dir``.
    CSV/JSON and the returned list retain one row per model; console/PNG/PDF
    use the compact pivoted view.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_sweep_runs(results_dir)
    rows = build_rows(run_dirs)
    display_rows, gpi_columns = build_display_rows(rows)

    _save_csv(rows, output_dir / "rl_sweep_table.csv")
    with open(output_dir / "rl_sweep_table.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    plot_sweep_comparison_table(
        display_rows,
        output_dir / "rl_sweep_table.png",
        games_per_iteration_values=gpi_columns,
    )
    plot_sweep_comparison_table(
        display_rows,
        output_dir / "rl_sweep_table.pdf",
        games_per_iteration_values=gpi_columns,
    )

    if not quiet:
        print(
            f"\nRL hyperparameter sweep comparison ({len(rows)} runs, "
            f"{len(display_rows)} displayed configurations found in {results_dir}/)"
        )
        _print_console_table(display_rows, gpi_columns)
        print(
            f"\nSaved: {output_dir}/rl_sweep_table.csv, "
            "rl_sweep_table.json, rl_sweep_table.png, rl_sweep_table.pdf"
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
