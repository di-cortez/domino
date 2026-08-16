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
import re
import time

import numpy as np

from diagnostics.pairwise import run_pairwise
from diagnostics.parallel_runner import ParallelSafetyConfig, cap_parallel_workers
from diagnostics.plots import wilson_interval
from diagnostics.worker_autotune import (
    DEFAULT_AUTOTUNE_FRACTION,
    DEFAULT_MINIMUM_GAIN,
    MatchupSpec,
    autotune_diagnostic_workers,
)
from training.canonical_run import load_run_config
from training.run_artifacts import (
    periodic_diagnostics_path,
    rl_progress_csv_path,
    rl_progress_png_path,
    run_dir_from_compact_diagnostic_path,
)
from training.utils.seeding import stable_seed
from utils.artifacts import atomic_write_json, atomic_write_text, file_sha256
from middleware.rulesets import DEFAULT_RULESET_NAME


FORMAT_VERSION = 3
PERIODIC_NAMESPACE = "periodic_rl_vs_random"
FINAL_NAMESPACE = "final_all_pairs_holdout"
PERIODIC_SUMMARY_RETENTION = 10
_PERIODIC_DIRECTORY_PATTERN = re.compile(r"games_(\d+)")
CSV_FIELDS = (
    "rl_games",
    "rl_iterations",
    "rl_elapsed_hours",
    "win_rate_percent",
    "ci95_low_percent",
    "ci95_high_percent",
)
FOOTER_FONT_SIZE = 8.0
FOOTER_MIN_FONT_SIZE = 5.0
# Left and right footer margins plus the gap kept between the two blocks.
FOOTER_ROW_MARGIN_FRACTION = 0.05
HISTORY_RECORD_TYPE = "periodic_rl_vs_random_history"
HISTORY_CHECKPOINT_BASE = "checkpoints"
HISTORY_STATIC_FIELDS = (
    "pipeline_level",
    "ruleset_name",
    "seed",
    "opponent",
    "diagnostic_games",
    "diagnostic_seed",
    "diagnostic_seed_namespace",
    "configuration_sha256",
)
HISTORY_DATA_FIELDS = (
    "rl_games",
    "rl_iterations",
    "rl_elapsed_seconds",
    "diagnostic_seconds",
    "checkpoint_path",
    "checkpoint_sha256",
    "wins",
    "selected_workers",
    "created_at",
)


def _format_machine_memory(byte_count):
    """Return a compact binary memory size for the plot footer."""
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024


def _machine_footer_lines(metadata):
    """Return separate CPU and GPU descriptions for the progress plot."""
    metadata = dict(metadata or {})
    cpu = metadata.get("cpu_model") or "unknown CPU"
    logical = metadata.get("logical_cpu_count")
    cpu_text = f"{cpu} ({logical} logical)" if logical else cpu
    ram = metadata.get("ram_total_bytes")
    ram_text = "RAM unknown" if ram is None else f"RAM {_format_machine_memory(ram)}"

    gpu = metadata.get("gpu_name") or "no detected GPU"
    vram = metadata.get("vram_total_bytes")
    vram_text = (
        "VRAM unavailable"
        if vram is None
        else f"VRAM {_format_machine_memory(vram)}"
    )
    return f"CPU: {cpu_text} · {ram_text}", f"GPU: {gpu} · {vram_text}"


def _regularizer_footer_text(coefficient):
    """Return one optional regularization coefficient for the plot footer.

    A run created before these controls existed stores no field at all; that
    absence is exactly the disabled value, so it reads as ``off`` instead of
    as an unknown setting.
    """
    try:
        value = float(coefficient)
    except (TypeError, ValueError):
        return "off"
    return f"{value:g}" if value > 0.0 else "off"


def _hidden_footer_text(network_architecture):
    """Return the hidden-layer count and every width, at any depth.

    ``network_architecture`` is the stored ``[input, *hidden, output]`` list,
    so the hidden stack is whatever sits between the encoder and action
    dimensions. The result reads ``4 layers 512x256x128x64``.
    """
    architecture = tuple(network_architecture or ())
    if len(architecture) < 3:
        return "unknown"
    hidden_sizes = [int(size) for size in architecture[1:-1]]
    layer_word = "layer" if len(hidden_sizes) == 1 else "layers"
    widths = "x".join(str(size) for size in hidden_sizes)
    return f"{len(hidden_sizes)} {layer_word} {widths}"


def _training_footer_line(run_config):
    """Return critic, hidden-layer, and regularization settings for the footer."""
    run_config = dict(run_config or {})
    rl_config = dict(run_config.get("rl_config", {}))
    hidden_text = _hidden_footer_text(run_config.get("network_architecture", ()))
    value_head_text = (
        "on" if bool(rl_config.get("use_value_head", False)) else "off"
    )
    dropout_text = _regularizer_footer_text(rl_config.get("dropout_rate"))
    decay_text = _regularizer_footer_text(rl_config.get("weight_decay"))
    return (
        f"Value head {value_head_text} · hidden {hidden_text} · "
        f"dropout {dropout_text} · weight decay {decay_text}"
    )


def _fitted_footer_fontsize(figure, text, reserved_width):
    """Return the largest footer size at which ``text`` clears the left block.

    The architecture summary grows with the hidden-layer count, so this row is
    measured rather than assumed to fit. ``reserved_width`` is the width the
    already-placed left-hand footer occupies, in display units.
    """
    renderer = figure.canvas.get_renderer()
    available = (
        figure.bbox.width * (1.0 - FOOTER_ROW_MARGIN_FRACTION) - reserved_width
    )
    probe = figure.text(0.0, 0.0, text, fontsize=FOOTER_FONT_SIZE)
    size = FOOTER_FONT_SIZE
    try:
        while size > FOOTER_MIN_FONT_SIZE:
            probe.set_fontsize(size)
            if probe.get_window_extent(renderer).width <= available:
                break
            size -= 0.5
    finally:
        probe.remove()
    return size


def _rl_elapsed_hours(row):
    """Return cumulative RL and periodic-diagnostic time in hours."""
    try:
        seconds = float(row["progress_elapsed_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Periodic diagnostic point has no valid cumulative "
            "progress_elapsed_seconds value."
        ) from exc
    if not np.isfinite(seconds) or seconds < 0.0:
        raise ValueError(
            "Periodic diagnostic progress time must be finite and non-negative."
        )
    return seconds / 3600.0


def periodic_diagnostic_seed(seed):
    return stable_seed(int(seed), PERIODIC_NAMESPACE)


def final_diagnostic_seed(seed):
    return stable_seed(int(seed), FINAL_NAMESPACE)


def _required_history_identity(row):
    required = {
        "rl_games",
        "checkpoint_sha256",
        "diagnostic_seed",
        "diagnostic_games",
        "opponent",
    }
    return isinstance(row, dict) and required.issubset(row)


def _history_header(rows):
    """Build and validate one compact history header from normalized rows."""
    first = rows[0]
    static = {name: first.get(name) for name in HISTORY_STATIC_FIELDS}
    for row in rows[1:]:
        differences = [
            name
            for name, expected in static.items()
            if row.get(name) != expected
        ]
        if differences:
            raise ValueError(
                "Periodic diagnostic history mixes incompatible static metadata: "
                + ", ".join(differences)
            )
    return {
        "record_type": HISTORY_RECORD_TYPE,
        "format_version": FORMAT_VERSION,
        "checkpoint_path_base": HISTORY_CHECKPOINT_BASE,
        "columns": list(HISTORY_DATA_FIELDS),
        "static": static,
    }


def _checkpoint_path_for_storage(path, history_path):
    """Return a checkpoint path relative to the history's checkpoints folder."""
    run_dir = run_dir_from_compact_diagnostic_path(history_path).resolve()
    checkpoint_base = run_dir / HISTORY_CHECKPOINT_BASE
    checkpoint = Path(path)
    if not checkpoint.is_absolute():
        checkpoint = run_dir / checkpoint
    return os.path.relpath(checkpoint.resolve(), checkpoint_base)


def _checkpoint_path_from_storage(path, history_path, checkpoint_base):
    """Resolve one stored checkpoint path for existing in-memory consumers."""
    return str(
        (
            run_dir_from_compact_diagnostic_path(history_path)
            / checkpoint_base
            / str(path)
        ).resolve()
    )


def _derive_history_values(rows):
    """Restore redundant report values without persisting them per point."""
    cumulative_diagnostic_seconds = 0.0
    derived = []
    for original in rows:
        row = dict(original)
        try:
            games = int(row["diagnostic_games"])
            wins = int(row["wins"])
            rl_seconds = float(row["rl_elapsed_seconds"])
            diagnostic_seconds = float(row["diagnostic_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Periodic diagnostic history has invalid numeric data."
            ) from exc
        if games < 1 or not 0 <= wins <= games:
            raise ValueError("Periodic diagnostic wins must be within game count.")
        if (
            not np.isfinite(rl_seconds)
            or not np.isfinite(diagnostic_seconds)
            or rl_seconds < 0.0
            or diagnostic_seconds < 0.0
        ):
            raise ValueError(
                "Periodic diagnostic elapsed times must be finite and non-negative."
            )
        cumulative_diagnostic_seconds += diagnostic_seconds
        losses = games - wins
        low, high = wilson_interval(wins, games)
        row.update({
            "format_version": FORMAT_VERSION,
            "losses": losses,
            "win_rate": wins / games,
            "loss_rate": losses / games,
            "ci95_win_rate_low": float(low),
            "ci95_win_rate_high": float(high),
            "progress_elapsed_seconds": (
                rl_seconds + cumulative_diagnostic_seconds
            ),
        })
        derived.append(row)
    return derived


def _legacy_row(value):
    """Normalize one version-two object before compact migration."""
    row = dict(value)
    row.setdefault("diagnostic_seed_namespace", PERIODIC_NAMESPACE)
    row.setdefault("configuration_sha256", None)
    if "wins" not in row and "win_rate" in row:
        row["wins"] = round(
            float(row["win_rate"]) * int(row["diagnostic_games"])
        )
    return row


def read_periodic_history(path):
    """Read compact or legacy JSONL, tolerating a corrupt final partial line."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    values = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            values.append((index, json.loads(line)))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise ValueError(
                f"Periodic diagnostic history has corrupt line {index + 1}."
            )
    if not values:
        return []

    _first_line, first = values[0]
    if (
        isinstance(first, dict)
        and first.get("record_type") == HISTORY_RECORD_TYPE
    ):
        if first.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                "Unsupported periodic diagnostic history format version "
                f"{first.get('format_version')!r}."
            )
        columns = first.get("columns")
        if columns != list(HISTORY_DATA_FIELDS):
            raise ValueError("Periodic diagnostic history has unexpected columns.")
        static = first.get("static")
        if not isinstance(static, dict):
            raise ValueError("Periodic diagnostic history has no static header.")
        checkpoint_base = first.get("checkpoint_path_base")
        if checkpoint_base != HISTORY_CHECKPOINT_BASE:
            raise ValueError(
                "Periodic diagnostic history has an unexpected checkpoint base."
            )
        rows = []
        for line_index, value in values[1:]:
            if not isinstance(value, list) or len(value) != len(columns):
                if line_index == len(lines) - 1:
                    break
                raise ValueError(
                    "Periodic diagnostic history line "
                    f"{line_index + 1} has invalid compact data."
                )
            row = {**static, **dict(zip(columns, value))}
            row["checkpoint_path"] = _checkpoint_path_from_storage(
                row["checkpoint_path"],
                path,
                checkpoint_base,
            )
            if not _required_history_identity(row):
                raise ValueError(
                    "Periodic diagnostic history line "
                    f"{line_index + 1} has no valid identity."
                )
            rows.append(row)
        return _derive_history_values(rows)

    rows = []
    for line_index, value in values:
        row = _legacy_row(value) if isinstance(value, dict) else value
        if not _required_history_identity(row):
            if line_index == len(lines) - 1:
                break
            raise ValueError(
                f"Periodic diagnostic history line {line_index + 1} "
                "has no valid identity."
            )
        rows.append(row)
    return _derive_history_values(rows)


def _point_key(row):
    return (
        int(row["rl_games"]),
        row["checkpoint_sha256"],
        row.get("configuration_sha256"),
        int(row["diagnostic_seed"]),
        int(row["diagnostic_games"]),
        row["opponent"],
    )


def _serialize_periodic_history(path, rows):
    """Return canonical header-plus-data JSONL for normalized rows."""
    if not rows:
        return ""
    header = _history_header(rows)
    lines = [json.dumps(header, sort_keys=True, separators=(",", ":"))]
    for row in rows:
        stored = dict(row)
        stored["checkpoint_path"] = _checkpoint_path_for_storage(
            row["checkpoint_path"],
            path,
        )
        values = [stored.get(name) for name in HISTORY_DATA_FIELDS]
        lines.append(json.dumps(values, separators=(",", ":")))
    return "\n".join(lines) + "\n"


def _repair_final_partial_line(path, rows):
    """Atomically migrate legacy data and normalize a valid compact prefix."""
    path = Path(path)
    if not path.exists():
        return False
    raw_text = path.read_text(encoding="utf-8")
    valid_text = _serialize_periodic_history(path, rows)
    if raw_text == valid_text:
        return False
    atomic_write_text(path, valid_text)
    return True


def append_periodic_point(path, row):
    """Persist one compact point unless its full diagnostic identity exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    key = _point_key(row)
    existing_rows = read_periodic_history(path)
    _repair_final_partial_line(path, existing_rows)
    for existing in existing_rows:
        if _point_key(existing) == key:
            return existing, False
    atomic_write_text(
        path,
        _serialize_periodic_history(path, [*existing_rows, row]),
    )
    persisted = read_periodic_history(path)[-1]
    return persisted, True


def prune_periodic_diagnostic_artifacts(
    run_dir,
    *,
    keep_summaries=PERIODIC_SUMMARY_RETENTION,
):
    """Remove bulky game records and bound per-point summary directories.

    The compact ``periodic_diagnostics.jsonl`` remains the complete source of
    truth. Only known generated files are removed, and an old directory is
    removed only when that leaves it empty.
    """
    keep_summaries = int(keep_summaries)
    if keep_summaries < 0:
        raise ValueError("keep_summaries must be non-negative")
    diagnostics_dir = Path(run_dir) / "diagnostics"
    if not diagnostics_dir.is_dir():
        return {
            "games_csv_removed": 0,
            "summary_json_removed": 0,
            "directories_removed": 0,
        }

    point_directories = []
    for path in diagnostics_dir.iterdir():
        match = _PERIODIC_DIRECTORY_PATTERN.fullmatch(path.name)
        if match and path.is_dir() and not path.is_symlink():
            point_directories.append((int(match.group(1)), path))
    point_directories.sort()
    retained = {
        path.resolve()
        for _games, path in (
            point_directories[-keep_summaries:]
            if keep_summaries else ()
        )
    }

    removed_games = 0
    removed_summaries = 0
    removed_directories = 0
    for _games, directory in point_directories:
        games_path = directory / "games.csv"
        if games_path.is_file():
            try:
                games_path.unlink()
            except OSError:
                pass
            else:
                removed_games += 1
        if directory.resolve() not in retained:
            summary_path = directory / "summary.json"
            if summary_path.is_file():
                try:
                    summary_path.unlink()
                except OSError:
                    pass
                else:
                    removed_summaries += 1
            try:
                directory.rmdir()
            except OSError:
                pass
            else:
                removed_directories += 1
    return {
        "games_csv_removed": removed_games,
        "summary_json_removed": removed_summaries,
        "directories_removed": removed_directories,
    }


def rebuild_progress_csv(run_dir):
    """Rebuild the derived CSV from JSONL, which remains the source of truth."""
    run_dir = Path(run_dir)
    rows = sorted(
        read_periodic_history(periodic_diagnostics_path(run_dir)),
        key=lambda row: int(row["rl_games"]),
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "rl_games": row["rl_games"],
            "rl_iterations": row["rl_iterations"],
            "rl_elapsed_hours": f"{_rl_elapsed_hours(row):.3f}",
            "win_rate_percent": f"{100.0 * row['win_rate']:.3f}",
            "ci95_low_percent": (
                f"{100.0 * row['ci95_win_rate_low']:.3f}"
            ),
            "ci95_high_percent": (
                f"{100.0 * row['ci95_win_rate_high']:.3f}"
            ),
        })
    return atomic_write_text(rl_progress_csv_path(run_dir), stream.getvalue())


def rebuild_progress_plot(run_dir, *, log_x=False):
    """Atomically rebuild one learning-curve PNG using only JSONL points."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    run_dir = Path(run_dir)
    rows = sorted(
        read_periodic_history(periodic_diagnostics_path(run_dir)),
        key=lambda row: int(row["rl_games"]),
    )
    if not rows:
        raise ValueError("Cannot plot RL progress without diagnostic points.")
    x = np.asarray([_rl_elapsed_hours(row) for row in rows], dtype=np.float64)
    y = 100.0 * np.asarray([row["win_rate"] for row in rows])
    low = 100.0 * np.asarray([row["ci95_win_rate_low"] for row in rows])
    high = 100.0 * np.asarray([row["ci95_win_rate_high"] for row in rows])

    window = 10
    y_smooth = np.asarray([
        np.mean(y[max(0, index - window + 1): index + 1])
        for index in range(len(y))
    ])

    figure = Figure(figsize=(9.5, 5.5), facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.plot(x, y, marker="o", linewidth=2.0, label="RL vs random win rate")
    axis.fill_between(x, low, high, alpha=0.2, label="95% confidence interval")
    axis.plot(
        x,
        y_smooth,
        linewidth=2.5,
        label=f"{window}-point trailing moving average",
    )
    zero_rows = [row for row in rows if int(row["rl_games"]) == 0]
    if zero_rows:
        starting_hours = _rl_elapsed_hours(zero_rows[-1])
        axis.scatter(
            [starting_hours],
            [100.0 * zero_rows[-1]["win_rate"]],
            s=70,
            zorder=4,
            label="Canonical supervised starting point",
        )
    if log_x:
        axis.set_xscale("symlog", linthresh=1.0)
    axis.set_xlabel("Cumulative RL and periodic-diagnostic time (hours)")
    axis.set_ylabel("Win rate vs random (%)")
    axis.set_title(
        f"RL learning progress — {rows[-1]['pipeline_level']} seed {rows[-1]['seed']}"
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    games = int(rows[-1]["diagnostic_games"])
    try:
        run_config = load_run_config(run_dir)
    except (FileNotFoundError, ValueError):
        run_config = {}
    starting_point = zero_rows[-1] if zero_rows else rows[0]
    start_name = Path(starting_point["checkpoint_path"]).name
    start_hash = starting_point["checkpoint_sha256"][:12]
    config_hash = (
        run_config.get("configuration_sha256")
        or rows[-1].get("configuration_sha256")
        or "unknown"
    )
    rl_config = run_config.get("rl_config", {})
    ppo_config = run_config.get("ppo_config", {})
    seed = run_config.get("seed", rows[-1].get("seed"))
    gpi = rl_config.get("games_per_iteration", rows[-1].get("gpi"))
    learning_rate = rl_config.get("learning_rate")
    if run_config.get("algorithm") == "reinforce_v1":
        ppo_text = "PPO off"
    else:
        epochs = ppo_config.get("max_epochs")
        ppo_text = f"PPO {epochs} ep" if epochs is not None else "PPO"
    gpi_text = f"GPI {int(gpi):,}" if gpi is not None else "GPI unknown"
    lr_text = (
        f"lr {float(learning_rate):g}"
        if learning_rate is not None else "lr unknown"
    )
    cpu_text, gpu_text = _machine_footer_lines(run_config.get("machine", {}))
    start_footer = figure.text(
        0.01,
        0.06,
        f"Start: {start_name} · sha256 {start_hash}...",
        ha="left",
        va="bottom",
        fontsize=FOOTER_FONT_SIZE,
    )
    figure.text(
        0.01,
        0.035,
        cpu_text,
        ha="left",
        va="bottom",
        fontsize=7,
    )
    figure.text(
        0.01,
        0.01,
        gpu_text,
        ha="left",
        va="bottom",
        fontsize=7,
    )
    training_footer = _training_footer_line(run_config)
    figure.text(
        0.99,
        0.06,
        training_footer,
        ha="right",
        va="bottom",
        fontsize=_fitted_footer_fontsize(
            figure,
            training_footer,
            start_footer.get_window_extent(
                figure.canvas.get_renderer()
            ).width,
        ),
    )
    figure.text(
        0.99,
        0.035,
        f"Updated {updated} · {games:,} games/diagnostic",
        ha="right",
        va="bottom",
        fontsize=8,
    )
    figure.text(
        0.99,
        0.01,
        f"seed {seed} · {gpi_text} · {ppo_text} · {lr_text} · "
        f"config {config_hash[:8]}...",
        ha="right",
        va="bottom",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.095, 1, 1))
    output = rl_progress_png_path(run_dir, log_x=log_x)
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
        "checkpoint_path": row["checkpoint_path"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "configuration_sha256": row.get("configuration_sha256"),
        "diagnostic_seed": int(row["diagnostic_seed"]),
        "updated_at": row["created_at"],
    }
    atomic_write_json(path, value)
    return value


def rebuild_best_checkpoint(run_dir):
    """Rebuild the best pointer from JSONL without changing latest state."""
    rows = sorted(
        read_periodic_history(periodic_diagnostics_path(run_dir)),
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
        "checkpoint_path": best["checkpoint_path"],
        "checkpoint_sha256": best["checkpoint_sha256"],
        "configuration_sha256": best.get("configuration_sha256"),
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
    checkpoint_path,
    diagnostic_games,
    rl_elapsed_seconds,
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
    try:
        run_config = load_run_config(run_dir)
    except (FileNotFoundError, ValueError):
        run_config = {}
    checkpoint_path = Path(checkpoint_path).resolve()
    ruleset_name = run_config.get("ruleset_name", DEFAULT_RULESET_NAME)
    diagnostic_seed = periodic_diagnostic_seed(seed)
    checkpoint_hash = file_sha256(checkpoint_path)
    identity = {
        "rl_games": int(rl_games),
        "checkpoint_sha256": checkpoint_hash,
        "configuration_sha256": run_config.get("configuration_sha256"),
        "diagnostic_seed": int(diagnostic_seed),
        "diagnostic_games": int(diagnostic_games),
        "opponent": "random",
        "ruleset_name": ruleset_name,
    }
    history_path = periodic_diagnostics_path(run_dir)
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
            section_started = time.perf_counter()
            prune_periodic_diagnostic_artifacts(run_dir)
            add_runtime("diagnostic_artifact_pruning", section_started)
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
                ruleset_name=ruleset_name,
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
            save_game_records=False,
            ruleset=ruleset_name,
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
    row = {
        "format_version": FORMAT_VERSION,
        "pipeline_level": pipeline_level,
        "seed": int(seed),
        "rl_games": int(rl_games),
        "rl_iterations": int(rl_iterations),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "configuration_sha256": run_config.get("configuration_sha256"),
        "opponent": "random",
        "diagnostic_games": int(diagnostic_games),
        "wins": wins,
        "diagnostic_seed": int(diagnostic_seed),
        "diagnostic_seed_namespace": PERIODIC_NAMESPACE,
        "diagnostic_seconds": float(diagnostic_seconds),
        "rl_elapsed_seconds": float(rl_elapsed_seconds),
        "selected_workers": selected_workers,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    add_runtime("diagnostic_summary_payload", section_started)
    section_started = time.perf_counter()
    row, appended = append_periodic_point(history_path, row)
    add_runtime("history_jsonl_atomic_update", section_started)
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
    section_started = time.perf_counter()
    prune_periodic_diagnostic_artifacts(run_dir)
    add_runtime("diagnostic_artifact_pruning", section_started)
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
