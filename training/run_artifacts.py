"""Stable paths for the compact, run-level analysis artifacts.

One run keeps its shareable summary -- the configuration, the periodic history,
the progress CSV and its plots -- in a single bundle directory beside its
checkpoints. The bundle is named after the run that made it:

    20260904-138_diego_notebook_lr_0p02

so a directory copied away from its run still says when it was trained, on
which machine, and which experiment it was. The parts are the training *start*
date, the experiment ordinal from the shared log, the machine slug from
``utils.machine_identity``, and an optional tail naming the one parameter the
run tests.

The ordinal is written as the literal ``XXX`` and filled in by hand: it comes
from a log shared across machines, and no single machine can compute the next
value. Nothing may therefore key on the bundle's name -- a hand-rename would
break it -- which is why every lookup here goes through ``find_bundle_dir`` and
matches a pattern instead.

``run_compact_diagnostics`` remains readable forever: it is the name every run
predating this convention uses, and those runs must keep resuming, rebuilding
their reports and reporting their best checkpoint.
"""

from __future__ import annotations

import re
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

# The placeholder an automatically created bundle carries until an operator
# substitutes the real experiment number from the shared log.
ORDINAL_PLACEHOLDER = "XXX"

# ``<8-digit date>-<ordinal>_<machine slug>_``. The ordinal alternative accepts
# both the placeholder and a filled number, so renaming `XXX` to `138` by hand
# leaves the bundle discoverable.
BUNDLE_DIR_PATTERN = re.compile(
    rf"^\d{{8}}-(?:\d+|{ORDINAL_PLACEHOLDER})_[A-Za-z0-9_-]+_?$"
)

# Files that prove a directory is a run root rather than something inside one.
RUN_ROOT_MARKERS = ("training_state.json", RUN_CONFIG_FILENAME)

# Short names for the tested parameter in a bundle's tail. A directory name is
# read at a glance in a listing, so `lr_0p02` beats `learning_rate_0p02`;
# anything not listed keeps its own flag name.
FLAG_SHORT_NAMES = {
    "--learning-rate": "lr",
    "--entropy-coef": "entropy",
    "--opponent-buckets": "bucket",
    "--reward-distance-mode": "distance",
    "--baseline": "baseline",
    "--gpi": "gpi",
    "--reward-eta": "eta",
    "--gamma-f": "gamma_f",
    "--gamma-i": "gamma_i",
    "--terminal-empty-hand-weight": "aE",
    "--terminal-blocked-weight": "aB",
    "--immediate-draw-weight": "aD",
    "--immediate-pass-weight": "aP",
}


def bundle_suffix(flag, value):
    """Return the path-safe tail naming the one parameter a run tests.

    ``('--learning-rate', '0.02')`` becomes ``lr_0p02``. The point is that a
    bundle copied away from its run still says which experiment it is, so the
    tail has to survive as a directory name: a decimal point becomes ``p`` --
    ``0p02`` and ``0p2`` stay distinguishable, which ``002`` and ``02`` would
    not -- and every other separator becomes an underscore.
    """
    def normalized(text):
        text = str(text).replace(".", "p")
        for separator in ("-", ",", " ", "/", "\\"):
            text = text.replace(separator, "_")
        return text.strip("_")

    short = FLAG_SHORT_NAMES.get(str(flag))
    if short is None:
        # An unlisted flag keeps its own name, normalized the same way the
        # value is: otherwise it produces a tail the check below rejects.
        short = normalized(str(flag).lstrip("-"))
    cleaned = normalized(value)
    tail = f"{short}_{cleaned}" if cleaned else short
    if not re.fullmatch(r"[A-Za-z0-9_]+", tail):
        raise ValueError(f"Unsafe bundle suffix {tail!r} from {flag}={value!r}.")
    return tail


def bundle_dir_name(*, date, machine_slug, ordinal=None, suffix=None):
    """Assemble one bundle directory name from its parts.

    Without ``suffix`` the name ends in the separator that has always closed
    it. With one, the tested parameter closes it instead:
    ``20260904-XXX_diego_notebook_lr_0p02``.
    """
    if not re.fullmatch(r"\d{8}", str(date)):
        raise ValueError(f"Bundle date must be 8 digits, got {date!r}.")
    if ordinal is None:
        ordinal_part = ORDINAL_PLACEHOLDER
    else:
        ordinal_part = f"{int(ordinal):03d}"
    tail = "" if suffix is None else str(suffix)
    name = f"{date}-{ordinal_part}_{machine_slug}_{tail}"
    if not BUNDLE_DIR_PATTERN.match(name):
        raise ValueError(f"Assembled an invalid bundle directory name: {name!r}.")
    return name


def is_bundle_dir_name(name):
    """Return whether one directory name is a run's analysis bundle."""
    return name == RUN_COMPACT_DIAGNOSTICS_DIRNAME or bool(
        BUNDLE_DIR_PATTERN.match(str(name))
    )


def find_bundle_dir(run_dir):
    """Return the existing bundle inside ``run_dir``, or ``None``.

    A migrated run holds both the legacy directory and the new one by design,
    so the new name always wins. Two *new-style* bundles is genuine ambiguity,
    never migration, and is refused rather than guessed at.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None
    named = sorted(
        child for child in run_dir.iterdir()
        if child.is_dir() and BUNDLE_DIR_PATTERN.match(child.name)
    )
    if len(named) > 1:
        raise ValueError(
            f"{run_dir} holds more than one analysis bundle: "
            + ", ".join(child.name for child in named)
            + ". Remove or merge the extras; a run has exactly one."
        )
    if named:
        return named[0]
    legacy = run_dir / RUN_COMPACT_DIAGNOSTICS_DIRNAME
    return legacy if legacy.is_dir() else None


def run_compact_diagnostics_dir(run_dir, *, default_name=None):
    """Return the directory holding one run's compact analysis bundle.

    An existing bundle always wins, whatever it is called. ``default_name``
    names the bundle a run is about to create; without it the legacy name is
    used, which keeps every caller that only reads working unchanged.
    """
    existing = find_bundle_dir(run_dir)
    if existing is not None:
        return existing
    return Path(run_dir) / (default_name or RUN_COMPACT_DIAGNOSTICS_DIRNAME)


def run_dir_from_compact_diagnostic_path(path):
    """Return the run directory owning one artifact path.

    Resolving this wrongly is silent and destructive: every stored checkpoint
    path is relative to the run root, so a wrong answer sends
    ``rebuild_best_checkpoint`` at files that do not exist. The former
    implementation returned the parent unconditionally when it did not
    recognize the directory name, which a per-run bundle name would have
    triggered on every single run. It now recognizes both naming schemes, and
    the last-resort fallback has to *prove* it found a run root.
    """
    path = Path(path)
    parent = path.parent
    if is_bundle_dir_name(parent.name):
        return parent.parent
    # An artifact sitting directly in the run root, from before the bundle
    # existed. Accept it only when the directory actually looks like a run.
    if any((parent / marker).is_file() for marker in RUN_ROOT_MARKERS):
        return parent
    raise ValueError(
        f"Cannot identify the run directory owning {path}: {parent} is "
        "neither an analysis bundle nor a run root."
    )


def run_compact_diagnostic_path(run_dir, filename, *, default_name=None):
    """Return one validated path inside the compact analysis bundle."""
    if filename not in RUN_COMPACT_DIAGNOSTIC_FILENAMES:
        raise ValueError(f"Unknown compact run diagnostic artifact: {filename!r}.")
    return run_compact_diagnostics_dir(
        run_dir, default_name=default_name
    ) / filename


def run_config_path(run_dir, *, default_name=None):
    return run_compact_diagnostic_path(
        run_dir, RUN_CONFIG_FILENAME, default_name=default_name
    )


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


def migrate_bundle_to_named_dir(run_dir, bundle_name):
    """Copy a legacy ``run_compact_diagnostics`` bundle to its per-run name.

    Copy-then-verify, never move: a half-finished rename would leave a run
    without the history its own resume path reads. The legacy directory is left
    in place, and ``find_bundle_dir`` prefers the new one from then on.
    """
    run_dir = Path(run_dir)
    legacy = run_dir / RUN_COMPACT_DIAGNOSTICS_DIRNAME
    destination = run_dir / bundle_name
    if not legacy.is_dir() or destination.exists():
        return ()
    if not BUNDLE_DIR_PATTERN.match(bundle_name):
        raise ValueError(f"Not a valid bundle directory name: {bundle_name!r}.")
    destination.mkdir(parents=True)
    migrated = []
    for source in sorted(legacy.iterdir()):
        if not source.is_file():
            continue
        target = destination / source.name
        atomic_copy(source, target)
        if target.read_bytes() != source.read_bytes():
            raise OSError(f"Bundle migration produced a differing copy: {target}.")
        migrated.append(target)
    return tuple(migrated)
