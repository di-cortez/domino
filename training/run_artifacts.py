"""Stable paths for the compact, run-level analysis artifacts."""

from __future__ import annotations

from pathlib import Path

from utils.artifacts import atomic_copy


RUN_COMPACT_DIAGNOSTICS_DIRNAME = "run_compact_diagnostics"
RUN_CONFIG_FILENAME = "run_config.json"
PERIODIC_DIAGNOSTICS_FILENAME = "periodic_diagnostics.jsonl"
RL_PROGRESS_CSV_FILENAME = "rl_vs_random_progress.csv"
RL_PROGRESS_PNG_FILENAME = "rl_vs_random_progress.png"
RL_PROGRESS_LOGX_PNG_FILENAME = "rl_vs_random_progress_logx.png"

RUN_COMPACT_DIAGNOSTIC_FILENAMES = (
    RUN_CONFIG_FILENAME,
    PERIODIC_DIAGNOSTICS_FILENAME,
    RL_PROGRESS_CSV_FILENAME,
    RL_PROGRESS_PNG_FILENAME,
    RL_PROGRESS_LOGX_PNG_FILENAME,
)


def run_compact_diagnostics_dir(run_dir):
    """Return the directory containing one run's compact analysis bundle."""
    return Path(run_dir) / RUN_COMPACT_DIAGNOSTICS_DIRNAME


def run_dir_from_compact_diagnostic_path(path):
    """Return the owning run directory for current or former artifact paths."""
    parent = Path(path).parent
    if parent.name == RUN_COMPACT_DIAGNOSTICS_DIRNAME:
        return parent.parent
    return parent


def run_compact_diagnostic_path(run_dir, filename):
    """Return one validated path inside the compact analysis bundle."""
    if filename not in RUN_COMPACT_DIAGNOSTIC_FILENAMES:
        raise ValueError(f"Unknown compact run diagnostic artifact: {filename!r}.")
    return run_compact_diagnostics_dir(run_dir) / filename


def run_config_path(run_dir):
    return run_compact_diagnostic_path(run_dir, RUN_CONFIG_FILENAME)


def periodic_diagnostics_path(run_dir):
    return run_compact_diagnostic_path(run_dir, PERIODIC_DIAGNOSTICS_FILENAME)


def rl_progress_csv_path(run_dir):
    return run_compact_diagnostic_path(run_dir, RL_PROGRESS_CSV_FILENAME)


def rl_progress_png_path(run_dir, *, log_x=False):
    filename = (
        RL_PROGRESS_LOGX_PNG_FILENAME if log_x else RL_PROGRESS_PNG_FILENAME
    )
    return run_compact_diagnostic_path(run_dir, filename)


def existing_run_config_path(run_dir):
    """Locate a current config or the former run-root location."""
    current = run_config_path(run_dir)
    legacy = Path(run_dir) / RUN_CONFIG_FILENAME
    if current.is_file():
        return current
    if legacy.is_file():
        return legacy
    return current


def run_config_exists(run_dir):
    return existing_run_config_path(run_dir).is_file()


def migrate_legacy_compact_diagnostics(run_dir):
    """Copy former run-root summaries into the compact analysis directory.

    Existing runs remain resumable, while every subsequent write uses the new
    location. The former files are deliberately left untouched so migration
    cannot make a previously valid run less recoverable.
    """
    run_dir = Path(run_dir)
    destination = run_compact_diagnostics_dir(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    migrated = []
    for filename in RUN_COMPACT_DIAGNOSTIC_FILENAMES:
        source = run_dir / filename
        target = destination / filename
        if source.is_file() and not target.exists():
            atomic_copy(source, target)
            migrated.append(target)
    return tuple(migrated)
