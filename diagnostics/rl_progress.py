"""Reproducible periodic RL-vs-random monitoring and derived reports."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
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
from training.canonical_run import (
    load_run_config,
    run_config_uses_opponent_bucket_features,
    run_config_uses_opponent_suit_features,
)
from training.run_artifacts import (
    periodic_diagnostics_path,
    rl_progress_csv_path,
    rl_progress_png_path,
    run_dir_from_compact_diagnostic_path,
)
from training.rl.reporting import read_training_metrics
from training.rl.statistics import RunningMoments, rounded_statistic
from training.utils.seeding import stable_seed
from utils.artifacts import atomic_write_json, atomic_write_text, file_sha256
from middleware.rulesets import DEFAULT_RULESET_NAME


FORMAT_VERSION = 4
# v3 is read but never written: every existing run's history is in that layout,
# and `rebuild_best_checkpoint`, the CSV rebuild and resume all go through the
# same reader.
LEGACY_HISTORY_VERSION = 3
SUPPORTED_HISTORY_VERSIONS = (LEGACY_HISTORY_VERSION, FORMAT_VERSION)
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
# The v3 positional columns, kept verbatim so a v3 file written before the
# schema change still decodes. Nothing new is ever written in this layout.
LEGACY_V3_DATA_FIELDS = (
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

# What a v4 record stores. Every other value a reader sees -- the win rate, the
# losses, the Wilson bounds, the cumulative progress clock -- stays derived by
# `_derive_history_values`, so there is exactly one source of truth for each.
#
# `checkpoint_sha256` is deliberately absent: it played no part in resuming a
# run, which goes entirely through `training_state.json`. `checkpoint_path`
# stays, so a record still names the weights it measured.
HISTORY_DATA_FIELDS = (
    "rl_games",
    "rl_iterations",
    "rl_elapsed_seconds",
    "diagnostic_seconds",
    "checkpoint_path",
    "wins",
    "selected_workers",
    "created_at",
)

# What a v4 record calls each stored field on disk. The internal row keeps the
# names every consumer already uses; only the file speaks this vocabulary, and
# the reader inverts the map on the way back in. Keeping the translation at the
# file boundary is what lets the record match the agreed schema without a rename
# rippling through the CSV rebuild, the best-checkpoint pointer and resume.
V4_RECORDED_NAMES = {
    "created_at": "timestamp",
    "rl_iterations": "iteration",
    "rl_games": "cumulative_games",
}
V4_INTERNAL_NAMES = {
    recorded: internal for internal, recorded in V4_RECORDED_NAMES.items()
}

# Two values the record states outright even though the reader recomputes them
# from `wins` and the elapsed clocks. They are in the schema because a record
# should be readable without knowing the derivation, and they are recomputed on
# read regardless, so the stored copy can never become the source of truth.
V4_RESTATED_FIELDS = ("elapsed_hours", "diagnostic_win_rate")


def v4_recorded_field_names():
    """Return the names a v4 record actually stores for the outcome fields."""
    return tuple(
        V4_RECORDED_NAMES.get(name, name) for name in HISTORY_DATA_FIELDS
    ) + V4_RESTATED_FIELDS


# The training-window statistics a v4 record carries beside the outcome. They
# are optional: a run whose metrics trace was pruned still records its win
# rate, and every one of them reads back as `None`.
HISTORY_WINDOW_FIELDS = (
    "window_first_iteration",
    "window_iterations",
    "window_games",
    "window_decisions",
    "window_restart_decisions",
    "max_kl",
    "clip_fraction",
    "entropy",
    "epochs_completed",
) + tuple(
    f"{prefix}_{statistic}"
    for _column, prefix in (
        ("terminal_empty_hand_moments", "R_E"),
        ("terminal_blocked_moments", "R_B"),
        ("draw_return_moments", "G_D"),
        ("pass_return_moments", "G_P"),
        ("baseline_moments", "baseline"),
    )
    for statistic in ("mean", "max", "min", "std")
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
        version = first.get("format_version")
        if version not in SUPPORTED_HISTORY_VERSIONS:
            raise ValueError(
                "Unsupported periodic diagnostic history format version "
                f"{version!r}."
            )
        static = first.get("static")
        if not isinstance(static, dict):
            raise ValueError("Periodic diagnostic history has no static header.")
        checkpoint_base = first.get("checkpoint_path_base")
        if checkpoint_base != HISTORY_CHECKPOINT_BASE:
            raise ValueError(
                "Periodic diagnostic history has an unexpected checkpoint base."
            )
        # v3 lines are positional arrays declared by the header's `columns`;
        # v4 lines are self-describing objects. Both decode to the same row
        # shape, so every consumer downstream is version-blind -- which is what
        # lets a v3 run keep its curve, its best-checkpoint pointer and its
        # rebuilt CSV after the schema change.
        columns = first.get("columns")
        if version == LEGACY_HISTORY_VERSION:
            if columns != list(LEGACY_V3_DATA_FIELDS):
                raise ValueError(
                    "Periodic diagnostic history has unexpected columns."
                )
        elif columns is not None:
            raise ValueError(
                "A version 4 periodic diagnostic history declares no columns."
            )
        rows = []
        for line_index, value in values[1:]:
            last_line = line_index == len(lines) - 1
            if version == LEGACY_HISTORY_VERSION:
                if not isinstance(value, list) or len(value) != len(columns):
                    if last_line:
                        break
                    raise ValueError(
                        "Periodic diagnostic history line "
                        f"{line_index + 1} has invalid compact data."
                    )
                stored = dict(zip(columns, value))
            else:
                if not isinstance(value, dict):
                    if last_line:
                        break
                    raise ValueError(
                        "Periodic diagnostic history line "
                        f"{line_index + 1} is not a record object."
                    )
                stored = {
                    V4_INTERNAL_NAMES.get(name, name): field
                    for name, field in value.items()
                    # The restated values are recomputed below, so the stored
                    # copies are dropped rather than allowed to disagree.
                    if name not in V4_RESTATED_FIELDS
                }
            row = {**static, **stored}
            # A v3 row carries a checkpoint hash that v4 no longer stores.
            # Reading it back is harmless and keeps the older records fully
            # auditable, so it is passed through rather than stripped.
            row["checkpoint_path"] = _checkpoint_path_from_storage(
                row["checkpoint_path"],
                path,
                checkpoint_base,
            )
            if not _required_history_identity(row):
                if last_line:
                    break
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


# The five scalar populations a periodic record summarises, mapped from the
# per-iteration metrics column that carries them to the prefix they take in the
# record. Adding a population is one entry here and one column in
# `training.rl.reporting`.
WINDOW_DISTRIBUTIONS = (
    ("terminal_empty_hand_moments", "R_E"),
    ("terminal_blocked_moments", "R_B"),
    ("draw_return_moments", "G_D"),
    ("pass_return_moments", "G_P"),
    ("baseline_moments", "baseline"),
)

# Per-iteration PPO scalars, and how a window of them collapses to one number.
# `max_kl` takes the worst update in the window because the trust region is a
# bound, not an average; the rest describe a typical update and take the mean.
WINDOW_SCALARS = (
    ("max_kl", "max_approx_kl", "max"),
    ("clip_fraction", "final_clip_fraction", "mean"),
    ("entropy", "entropy", "mean"),
    ("epochs_completed", "epochs_completed", "mean"),
)


def training_metrics_path(run_dir):
    """Return the per-iteration metrics trace a canonical run writes."""
    return Path(run_dir) / "training_metrics.jsonl"


def _window_scalar(rows, column, how):
    values = [
        float(row[column]) for row in rows if row.get(column) is not None
    ]
    if not values:
        return None
    collapsed = max(values) if how == "max" else math.fsum(values) / len(values)
    return rounded_statistic(collapsed)


def summarize_training_window(run_dir, *, first_iteration, last_iteration):
    """Summarise the training iterations one periodic record covers.

    A periodic diagnostic fires every ``periodic_diagnostic_every_games``, which
    is fifty iterations at the usual settings, so a record describes a *window*
    of training rather than a single update. The window is
    ``first_iteration <= iteration <= last_iteration``; the caller derives its
    lower bound from the previous record, so the windows tile the run without
    gaps or overlap.

    Distributions merge, which is exact. Scalars collapse by the rule
    ``WINDOW_SCALARS`` names for each. A statistic no iteration reported comes
    back ``None`` rather than zero, because zero is a value every one of these
    can legitimately take.

    Returns an empty mapping when the trace is missing or carries no iteration
    in range: a run whose metrics were pruned still records its win rate.
    """
    path = training_metrics_path(run_dir)
    if not path.exists():
        return {}
    try:
        _header, rows = read_training_metrics(path)
    except (OSError, ValueError):
        # A diagnostic must never be lost to an unreadable metrics trace; the
        # win rate is the point of the record and does not depend on it.
        return {}
    window = [
        row for row in rows
        if first_iteration <= int(row["iteration"]) <= last_iteration
    ]
    if not window:
        return {}
    summary = {
        "window_first_iteration": int(window[0]["iteration"]),
        "window_iterations": len(window),
        "window_games": sum(int(row["games"]) for row in window),
        "window_decisions": sum(int(row["normal_decisions"]) for row in window),
        "window_restart_decisions": sum(
            int(row["restart_decisions"]) for row in window
        ),
    }
    for column, prefix in WINDOW_DISTRIBUTIONS:
        merged = RunningMoments()
        for row in window:
            stored = row.get(column)
            if stored is not None:
                merged.merge(RunningMoments.from_list(stored))
        summary.update(merged.as_dict(prefix))
    for name, column, how in WINDOW_SCALARS:
        summary[name] = _window_scalar(window, column, how)
    return summary


def _point_key(row):
    """Return the identity two records must share to be the same measurement.

    The checkpoint hash used to be part of this. Dropping it makes the key
    coarser in exactly one way: two measurements at the same ``rl_games`` taken
    from *different* weights -- a rollback followed by a retrain -- now collide,
    and the second is discarded as a duplicate. An ordinary resume re-measures
    the same checkpoint and is unaffected.
    """
    return (
        int(row["rl_games"]),
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
    # The progress clock is cumulative across records, and the row being
    # appended has not been through `_derive_history_values` yet, so the writer
    # accumulates it here rather than reading a field that may not exist.
    cumulative_diagnostic_seconds = 0.0
    for row in rows:
        cumulative_diagnostic_seconds += float(row["diagnostic_seconds"])
        record = {
            V4_RECORDED_NAMES.get(name, name): row.get(name)
            for name in HISTORY_DATA_FIELDS + HISTORY_WINDOW_FIELDS
            # A window statistic the run never measured is omitted rather than
            # written as null, so a record stays as small as what it knows.
            if name in HISTORY_DATA_FIELDS or row.get(name) is not None
        }
        record["checkpoint_path"] = _checkpoint_path_for_storage(
            row["checkpoint_path"],
            path,
        )
        # Restated for readability; `_derive_history_values` recomputes both,
        # so the stored copies are never what a consumer actually reads.
        record["elapsed_hours"] = round(
            (float(row["rl_elapsed_seconds"]) + cumulative_diagnostic_seconds)
            / 3600.0,
            6,
        )
        record["diagnostic_win_rate"] = round(
            100.0 * int(row["wins"]) / int(row["diagnostic_games"]),
            6,
        )
        lines.append(json.dumps(record, separators=(",", ":")))
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
    # v3 rows still carry the hash; v4 does not record it. The footer prints
    # it when it is there and simply says less when it is not.
    start_hash = (starting_point.get("checkpoint_sha256") or "")[:12]
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
        "checkpoint_sha256": row.get("checkpoint_sha256"),
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
        "checkpoint_sha256": best.get("checkpoint_sha256"),
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
    use_opponent_suit_features = run_config_uses_opponent_suit_features(
        run_config
    )
    use_opponent_bucket_features = run_config_uses_opponent_bucket_features(
        run_config
    )
    diagnostic_seed = periodic_diagnostic_seed(seed)
    identity = {
        "rl_games": int(rl_games),
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
                use_opponent_suit_features=use_opponent_suit_features,
                use_opponent_bucket_features=use_opponent_bucket_features,
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
            use_opponent_suit_features=use_opponent_suit_features,
            use_opponent_bucket_features=use_opponent_bucket_features,
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
    # The training this record covers: everything since the previous point.
    # `existing_history` was read before the diagnostic ran, so its last entry
    # is the previous record and the windows tile the run exactly.
    previous_iteration = max(
        (int(existing["rl_iterations"]) for existing in existing_history),
        default=0,
    )
    row.update(
        summarize_training_window(
            run_dir,
            first_iteration=previous_iteration + 1,
            last_iteration=int(rl_iterations),
        )
    )
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
