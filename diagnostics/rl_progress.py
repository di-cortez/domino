"""Reproducible periodic RL-vs-random monitoring and derived reports."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from diagnostics.pairwise import run_pairwise
from diagnostics.parallel_runner import ParallelSafetyConfig, cap_parallel_workers
from diagnostics.worker_autotune import (
    DEFAULT_AUTOTUNE_FRACTION,
    DEFAULT_MINIMUM_GAIN,
    MatchupSpec,
    autotune_diagnostic_workers,
)
from training.ppo import stable_seed
from utils.artifacts import atomic_write_json, atomic_write_text, file_sha256


FORMAT_VERSION = 1
PERIODIC_NAMESPACE = "periodic_rl_vs_random"
FINAL_NAMESPACE = "final_all_pairs_holdout"
CSV_FIELDS = (
    "rl_games",
    "rl_iterations",
    "optimizer_steps",
    "win_rate_percent",
    "score_percent",
    "draw_rate_percent",
    "ci95_low_percent",
    "ci95_high_percent",
    "diagnostic_games",
    "diagnostic_seconds",
    "checkpoint_path",
    "checkpoint_sha256",
)


def periodic_diagnostic_seed(seed):
    return stable_seed(int(seed), PERIODIC_NAMESPACE)


def final_diagnostic_seed(seed):
    return stable_seed(int(seed), FINAL_NAMESPACE)


def read_periodic_history(path):
    """Read valid JSONL points, tolerating only a corrupt final partial line."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise ValueError(
                f"Periodic diagnostic history has corrupt line {index + 1}."
            )
        required_identity = {
            "rl_games",
            "checkpoint_sha256",
            "diagnostic_seed",
            "diagnostic_games",
            "opponent",
        }
        if not isinstance(value, dict) or not required_identity.issubset(value):
            if index == len(lines) - 1:
                break
            raise ValueError(
                f"Periodic diagnostic history line {index + 1} has no valid identity."
            )
        rows.append(value)
    return rows


def _point_key(row):
    return (
        int(row["rl_games"]),
        row["checkpoint_sha256"],
        int(row["diagnostic_seed"]),
        int(row["diagnostic_games"]),
        row["opponent"],
    )


def _repair_final_partial_line(path, rows):
    """Atomically normalize a valid prefix when the JSONL tail is incomplete."""
    path = Path(path)
    if not path.exists():
        return False
    raw_text = path.read_text(encoding="utf-8")
    nonempty_lines = [line for line in raw_text.splitlines() if line.strip()]
    if (
        len(rows) == len(nonempty_lines)
        and (not raw_text or raw_text.endswith("\n"))
    ):
        return False
    valid_text = "".join(
        json.dumps(existing, sort_keys=True) + "\n" for existing in rows
    )
    atomic_write_text(path, valid_text)
    return True


def append_periodic_point(path, row):
    """Append one fsynced point unless its full diagnostic identity exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    key = _point_key(row)
    existing_rows = read_periodic_history(path)
    _repair_final_partial_line(path, existing_rows)
    for existing in existing_rows:
        if _point_key(existing) == key:
            return existing, False
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return row, True


def rebuild_progress_csv(run_dir):
    """Rebuild the derived CSV from JSONL, which remains the source of truth."""
    run_dir = Path(run_dir)
    rows = sorted(
        read_periodic_history(run_dir / "periodic_diagnostics.jsonl"),
        key=lambda row: int(row["rl_games"]),
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "rl_games": row["rl_games"],
            "rl_iterations": row["rl_iterations"],
            "optimizer_steps": row["optimizer_steps"],
            "win_rate_percent": 100.0 * row["win_rate"],
            "score_percent": 100.0 * row["score"],
            "draw_rate_percent": 100.0 * row["draw_rate"],
            "ci95_low_percent": 100.0 * row["ci95_win_rate_low"],
            "ci95_high_percent": 100.0 * row["ci95_win_rate_high"],
            "diagnostic_games": row["diagnostic_games"],
            "diagnostic_seconds": row["diagnostic_seconds"],
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_sha256": row["checkpoint_sha256"],
        })
    return atomic_write_text(run_dir / "rl_vs_random_progress.csv", stream.getvalue())


def rebuild_progress_plot(run_dir, *, log_x=False):
    """Atomically rebuild one learning-curve PNG using only JSONL points."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    run_dir = Path(run_dir)
    rows = sorted(
        read_periodic_history(run_dir / "periodic_diagnostics.jsonl"),
        key=lambda row: int(row["rl_games"]),
    )
    if not rows:
        raise ValueError("Cannot plot RL progress without diagnostic points.")
    x = np.asarray([row["rl_games"] for row in rows], dtype=np.float64)
    y = 100.0 * np.asarray([row["win_rate"] for row in rows])
    low = 100.0 * np.asarray([row["ci95_win_rate_low"] for row in rows])
    high = 100.0 * np.asarray([row["ci95_win_rate_high"] for row in rows])

    figure = Figure(figsize=(9.5, 5.5), facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(x, y, marker="o", linewidth=2.0, label="RL vs random win rate")
    axis.fill_between(x, low, high, alpha=0.2, label="95% confidence interval")
    zero_rows = [row for row in rows if int(row["rl_games"]) == 0]
    if zero_rows:
        axis.scatter(
            [0],
            [100.0 * zero_rows[-1]["win_rate"]],
            s=70,
            zorder=4,
            label="Canonical supervised starting point",
        )
    if log_x:
        axis.set_xscale("symlog", linthresh=100_000)
    axis.set_xlabel("RL games completed")
    axis.set_ylabel("Win rate vs random (%)")
    axis.set_title(
        f"RL learning progress — {rows[-1]['pipeline_level']} seed {rows[-1]['seed']}"
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    games = int(rows[-1]["diagnostic_games"])
    starting_point = zero_rows[-1] if zero_rows else rows[0]
    start_name = Path(starting_point["checkpoint_path"]).name
    start_hash = starting_point["checkpoint_sha256"][:12]
    figure.text(
        0.01,
        0.01,
        f"Start: {start_name} · sha256 {start_hash}...",
        ha="left",
        va="bottom",
        fontsize=8,
    )
    figure.text(
        0.99,
        0.01,
        f"Updated {updated} · {games:,} games per diagnostic",
        ha="right",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    filename = "rl_vs_random_progress_logx.png" if log_x else "rl_vs_random_progress.png"
    output = run_dir / filename
    temporary = output.with_name(
        f".{output.stem}.tmp-{os.getpid()}-{time.time_ns()}.png"
    )
    try:
        figure.savefig(temporary, dpi=150, format="png")
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def rebuild_progress_reports(run_dir, *, log_x=False):
    csv_path = rebuild_progress_csv(run_dir)
    plot_path = rebuild_progress_plot(run_dir)
    log_path = rebuild_progress_plot(run_dir, log_x=True) if log_x else None
    rebuild_best_checkpoint(run_dir)
    return csv_path, plot_path, log_path


def _update_best(run_dir, row):
    path = Path(run_dir) / "best_checkpoint.json"
    current = None
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = None
    if current is not None:
        try:
            if float(current["win_rate"]) >= float(row["win_rate"]):
                return current
        except (KeyError, TypeError, ValueError):
            current = None
    value = {
        "criterion": "win_rate_vs_random",
        "rl_games": int(row["rl_games"]),
        "win_rate": float(row["win_rate"]),
        "score": float(row["score"]),
        "checkpoint_path": row["checkpoint_path"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "diagnostic_seed": int(row["diagnostic_seed"]),
        "updated_at": row["created_at"],
    }
    atomic_write_json(path, value)
    return value


def rebuild_best_checkpoint(run_dir):
    """Rebuild the best pointer from JSONL without changing latest state."""
    rows = sorted(
        read_periodic_history(Path(run_dir) / "periodic_diagnostics.jsonl"),
        key=lambda row: int(row["rl_games"]),
    )
    if not rows:
        return None
    best = rows[0]
    for row in rows[1:]:
        if float(row["win_rate"]) > float(best["win_rate"]):
            best = row
    path = Path(run_dir) / "best_checkpoint.json"
    value = {
        "criterion": "win_rate_vs_random",
        "rl_games": int(best["rl_games"]),
        "win_rate": float(best["win_rate"]),
        "score": float(best["score"]),
        "checkpoint_path": best["checkpoint_path"],
        "checkpoint_sha256": best["checkpoint_sha256"],
        "diagnostic_seed": int(best["diagnostic_seed"]),
        "updated_at": best["created_at"],
    }
    atomic_write_json(path, value)
    return value


def run_periodic_diagnostic(
    *,
    run_dir,
    pipeline_level,
    seed,
    rl_games,
    rl_iterations,
    optimizer_steps,
    checkpoint_path,
    diagnostic_games,
    rl_elapsed_seconds,
    wall_clock_seconds,
    workers="auto",
    safety_config=None,
    autotune_fraction=DEFAULT_AUTOTUNE_FRACTION,
    autotune_minimum_gain=DEFAULT_MINIMUM_GAIN,
    status_callback=None,
):
    """Evaluate one checkpoint on the fixed monitor set and persist one point."""
    runtime_profile_started = time.perf_counter()
    runtime_sections = {}

    def add_runtime(section, started):
        runtime_sections[section] = runtime_sections.get(section, 0.0) + (
            time.perf_counter() - started
        )

    run_dir = Path(run_dir)
    checkpoint_path = Path(checkpoint_path)
    diagnostic_seed = periodic_diagnostic_seed(seed)
    checkpoint_hash = file_sha256(checkpoint_path)
    identity = {
        "rl_games": int(rl_games),
        "checkpoint_sha256": checkpoint_hash,
        "diagnostic_seed": int(diagnostic_seed),
        "diagnostic_games": int(diagnostic_games),
        "opponent": "random",
    }
    history_path = run_dir / "periodic_diagnostics.jsonl"
    existing_history = read_periodic_history(history_path)
    _repair_final_partial_line(history_path, existing_history)
    runtime_sections["identity_hash_and_history_read"] = (
        time.perf_counter() - runtime_profile_started
    )
    for existing in existing_history:
        if _point_key(existing) == _point_key(identity):
            section_started = time.perf_counter()
            rebuild_progress_csv(run_dir)
            add_runtime("progress_csv_rebuild", section_started)
            section_started = time.perf_counter()
            rebuild_progress_plot(run_dir)
            add_runtime("progress_plot_rebuild", section_started)
            section_started = time.perf_counter()
            rebuild_best_checkpoint(run_dir)
            _update_best(run_dir, existing)
            add_runtime("best_checkpoint_update", section_started)
            runtime_total_seconds = time.perf_counter() - runtime_profile_started
            runtime_sections["unaccounted"] = max(
                0.0,
                runtime_total_seconds - sum(runtime_sections.values()),
            )
            existing = dict(existing)
            existing["runtime_profile_delta"] = {
                "execution_count": 1,
                "reused_execution_count": 1,
                "games": 0,
                "execution_seconds": float(runtime_total_seconds),
                "sections_seconds": {
                    name: float(seconds)
                    for name, seconds in runtime_sections.items()
                },
                "pairwise_sections_seconds": {},
                "game_worker": {},
            }
            return existing, False

    safety_config = safety_config or ParallelSafetyConfig()
    output_dir = run_dir / "diagnostics" / f"games_{int(rl_games):010d}"
    section_started = time.perf_counter()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    add_runtime("rng_snapshot", section_started)
    started = time.time()
    try:
        section_started = time.perf_counter()
        if workers == "auto":
            matchup = MatchupSpec(
                agent="rl",
                opponent="random",
                weights=checkpoint_path,
            )
            tuning = autotune_diagnostic_workers(
                matchups=(matchup,),
                game_count=int(diagnostic_games),
                base_seed=int(diagnostic_seed),
                safety=safety_config,
                benchmark_fraction=autotune_fraction,
                minimum_gain=autotune_minimum_gain,
                status_callback=status_callback,
                pair_seed_overrides={matchup.key: int(diagnostic_seed)},
            )
            selected_workers = int(tuning["optimal_workers"])
            precomputed = tuning["precomputed_games"][matchup.key]
            precomputed_duration = tuning["durations_by_matchup"][matchup.key]
            precomputed_runtime_profile = tuning[
                "runtime_profiles_by_matchup"
            ][matchup.key]
        else:
            selected_workers, _capped, _reason = cap_parallel_workers(
                int(workers), safety_config
            )
            precomputed = ()
            precomputed_duration = 0.0
            precomputed_runtime_profile = {}
        add_runtime(
            "worker_autotune" if workers == "auto" else "worker_selection",
            section_started,
        )
        section_started = time.perf_counter()
        result = run_pairwise(
            "rl",
            "random",
            game_count=int(diagnostic_games),
            weights=checkpoint_path,
            seed=int(diagnostic_seed),
            effective_seed=int(diagnostic_seed),
            output_dir=output_dir,
            generate_plots=False,
            print_console_summary=False,
            print_memory_summary=False,
            workers=selected_workers,
            safety_config=safety_config,
            precomputed_games=precomputed,
            precomputed_duration_s=precomputed_duration,
            precomputed_runtime_profile=precomputed_runtime_profile,
        )
        add_runtime("pairwise_evaluation", section_started)
    finally:
        section_started = time.perf_counter()
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        add_runtime("rng_restore", section_started)
    diagnostic_seconds = time.time() - started
    section_started = time.perf_counter()
    summary = result["summary"]
    wins = int(summary["counts"]["win"])
    draws = int(summary["counts"]["draw"])
    losses = int(summary["counts"]["loss"])
    low, high = summary["win_ci95"]
    row = {
        "format_version": FORMAT_VERSION,
        "pipeline_level": pipeline_level,
        "seed": int(seed),
        "rl_games": int(rl_games),
        "rl_iterations": int(rl_iterations),
        "optimizer_steps": int(optimizer_steps),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "opponent": "random",
        "diagnostic_games": int(diagnostic_games),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / int(diagnostic_games),
        "draw_rate": draws / int(diagnostic_games),
        "loss_rate": losses / int(diagnostic_games),
        "score": (wins + 0.5 * draws) / int(diagnostic_games),
        "ci95_win_rate_low": float(low),
        "ci95_win_rate_high": float(high),
        "diagnostic_seed": int(diagnostic_seed),
        "diagnostic_seed_namespace": PERIODIC_NAMESPACE,
        "diagnostic_seconds": float(diagnostic_seconds),
        "rl_elapsed_seconds": float(rl_elapsed_seconds),
        "wall_clock_seconds": float(wall_clock_seconds),
        "selected_workers": selected_workers,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    add_runtime("diagnostic_summary_payload", section_started)
    section_started = time.perf_counter()
    row, appended = append_periodic_point(history_path, row)
    add_runtime("history_jsonl_append_and_fsync", section_started)
    section_started = time.perf_counter()
    rebuild_progress_csv(run_dir)
    add_runtime("progress_csv_rebuild", section_started)
    section_started = time.perf_counter()
    rebuild_progress_plot(run_dir)
    add_runtime("progress_plot_rebuild", section_started)
    section_started = time.perf_counter()
    rebuild_best_checkpoint(run_dir)
    _update_best(run_dir, row)
    add_runtime("best_checkpoint_update", section_started)
    runtime_total_seconds = time.perf_counter() - runtime_profile_started
    runtime_sections["unaccounted"] = max(
        0.0,
        runtime_total_seconds - sum(runtime_sections.values()),
    )
    pairwise_profile = result["runtime_profile_delta"]
    row["runtime_profile_delta"] = {
        "execution_count": 1,
        "reused_execution_count": 0,
        "games": int(diagnostic_games),
        "execution_seconds": float(runtime_total_seconds),
        "sections_seconds": {
            name: float(seconds) for name, seconds in runtime_sections.items()
        },
        "pairwise_sections_seconds": dict(pairwise_profile["sections_seconds"]),
        "game_worker": dict(pairwise_profile.get("game_worker", {})),
    }
    return row, appended


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Rebuild RL-vs-random CSV and plots from periodic JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--log-x", action="store_true")
    args = parser.parse_args(argv)
    csv_path, plot_path, log_path = rebuild_progress_reports(
        args.run_dir,
        log_x=args.log_x,
    )
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")
    if log_path is not None:
        print(f"Log-x plot: {log_path}")


if __name__ == "__main__":
    main()
