"""Stable short names for the machines that train runs in this project.

A run bundle is named after the machine that produced it, so a directory copied
away from its run still says where it came from. The name comes from the
hardware the run already records in ``run_config.json`` -- see
``utils.runtime_status.machine_metadata`` -- rather than from anything the
operator has to type, so it cannot drift from the truth.

The GPU alone separates every machine in use, but the registry keys on the
logical CPU count as well: a GPU swap should produce an unknown machine and a
warning, not silently rename an existing one.

An unrecognized machine never stops a run. It gets a derived slug and a warning
telling the operator to add an entry here; a naming convention that can abort
an eight-hour training is worse than an imperfect name.
"""

from __future__ import annotations

import re
import warnings


# ``(gpu substring, logical CPU count) -> slug``. The GPU is matched as a
# substring because the recorded strings carry vendor and packaging noise
# ("NVIDIA GeForce RTX 3050 6GB Laptop GPU").
#
# Slug shape, matching the directories already in `models/rl`: ``_`` separates
# the owner from the role, and ``-`` joins a multi-word role.
MACHINE_REGISTRY = (
    ({"gpu": "RTX 3050", "logical_cpus": 20}, "diego_notebook"),
    ({"gpu": "GTX 1650", "logical_cpus": 20}, "rick_desktop"),
    ({"gpu": "GTX 960M", "logical_cpus": 8}, "rick_notebook-antigo"),
    ({"gpu": "RTX 4050", "logical_cpus": 16}, "rick_notebook-novo"),
)

UNKNOWN_MACHINE_WARNING = (
    "This machine is not in utils.machine_identity.MACHINE_REGISTRY, so its "
    "run bundle is named {slug!r}. Training continues normally. Add an entry "
    "for gpu={gpu!r}, logical_cpus={cpus!r} to give it a stable name."
)

# Characters a slug may contain, matching `_validated_run_label` in
# `training.canonical_run` so a slug is always a usable path component.
_ALLOWED = re.compile(r"[^A-Za-z0-9_-]+")


def _sanitize(value):
    """Return one path-safe fragment of a hardware description."""
    collapsed = _ALLOWED.sub("-", str(value or "").strip()).strip("-")
    return collapsed.lower() or "unknown"


def derived_slug(metadata):
    """Return a stable fallback name for a machine with no registry entry.

    Deterministic, so the same unknown machine names its bundles consistently
    across runs, and descriptive enough that an operator can recognize which
    machine it was and add the registry entry afterwards.
    """
    metadata = dict(metadata or {})
    gpu = metadata.get("gpu_name") or "no-gpu"
    # Drop the vendor and marketing words that every card repeats, keeping the
    # model, which is what actually distinguishes machines.
    words = [
        word for word in str(gpu).split()
        if word.lower() not in {"nvidia", "geforce", "laptop", "gpu"}
    ]
    cpus = metadata.get("logical_cpu_count") or 0
    return f"unknown_{_sanitize('-'.join(words) or gpu)}-{int(cpus)}cpu"


def machine_slug(metadata, *, override=None, warn=True):
    """Return the registered slug for one machine's recorded metadata.

    ``override`` short-circuits the registry so a new machine can run before
    its entry is committed. It is sanitized, never trusted verbatim, because it
    becomes a directory name.
    """
    if override:
        return _sanitize(override)
    metadata = dict(metadata or {})
    gpu = str(metadata.get("gpu_name") or "")
    cpus = metadata.get("logical_cpu_count")
    for signature, slug in MACHINE_REGISTRY:
        if signature["gpu"] in gpu and signature["logical_cpus"] == cpus:
            return slug
    slug = derived_slug(metadata)
    # Metadata with neither a GPU nor a CPU count is not an unrecognized
    # machine, it is a caller that never described one -- a test double, or a
    # config written before the field existed. Warning there would train the
    # operator to ignore the warning that matters.
    described = bool(gpu) or cpus is not None
    if warn and described:
        warnings.warn(
            UNKNOWN_MACHINE_WARNING.format(
                slug=slug, gpu=gpu or None, cpus=cpus
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    return slug
