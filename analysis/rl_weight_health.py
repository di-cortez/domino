"""Report the numerical health of an RL run's saved policy checkpoints.

Read-only triage for the failure described in
``references/atualizacoes/atualizacoes_3108/DOMINO_RL_NUMERICAL_STABILITY_IMPLEMENTATION_PLAN.md``:
a rollout worker raising ``NonFinitePolicyError`` means the policy weights
stopped being usable, and the first question is *when*. This walks every
checkpoint a run left on disk and reports, per checkpoint, whether it is
finite and how large its largest weight is.

``max_abs_weight`` is the observable that matters. The float32 forward pass
overflows as a function of the largest weight times the largest activation, so
a run whose ``max_abs_weight`` climbs monotonically is on its way to a
non-finite policy even while every gradient norm looks ordinary.

Nothing here writes into a run directory.

Usage:

    python -m analysis.rl_weight_health <run_directory> [--per-layer]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _weight_arrays(payload):
    """Return the trainable float arrays of one checkpoint, by name.

    The archive stores only weights; a resume checkpoint also carries the
    optimizer step count and the algorithm label, which are not parameters.
    """
    return {
        name: payload[name]
        for name in payload.files
        if payload[name].dtype.kind == "f" and payload[name].ndim > 0
    }


def inspect_checkpoint(path):
    """Return one health row for a single ``.npz`` policy checkpoint."""
    with np.load(path, allow_pickle=False) as payload:
        arrays = _weight_arrays(payload)
        per_layer = {}
        non_finite = 0
        max_abs = 0.0
        for name, value in sorted(arrays.items()):
            finite = np.isfinite(value)
            non_finite += int(value.size - np.count_nonzero(finite))
            if np.any(finite):
                layer_max = float(np.max(np.abs(value[finite])))
            else:
                layer_max = float("nan")
                # A layer with no finite entry at all cannot contribute a
                # maximum, so it must not silently lower the run's figure.
                max_abs = float("nan")
            per_layer[name] = {
                "max_abs": layer_max,
                "mean_abs": (
                    float(np.mean(np.abs(value[finite])))
                    if np.any(finite)
                    else float("nan")
                ),
                "non_finite": int(value.size - np.count_nonzero(finite)),
                "shape": list(value.shape),
            }
            if not np.isnan(max_abs):
                max_abs = max(max_abs, layer_max)
    return {
        "path": str(path),
        "name": path.name,
        "parameters": len(per_layer),
        "non_finite_parameters": non_finite,
        "max_abs_weight": max_abs,
        "healthy": non_finite == 0,
        "per_layer": per_layer,
    }


def _is_policy_checkpoint(path):
    """Return whether one ``.npz`` holds policy weights rather than metadata.

    ``optimizer_state.npz`` sits beside the weights and carries only scalars,
    so a file counts only when the first policy layer is present.
    """
    with np.load(path, allow_pickle=False) as payload:
        return "W1" in payload.files


def _iteration_of(path):
    """Return the iteration a checkpoint filename encodes, or -1 if it has none.

    Both layouts end in ``_iter<six digits>``: ``numbered_checkpoint_path`` in
    ``training/rl/resume.py`` and ``checkpoint_iter%06d.npz`` in
    ``training/rl/checkpoint_archive.py``.
    """
    stem = path.stem
    marker = "_iter"
    index = stem.rfind(marker)
    if index < 0:
        return -1
    digits = stem[index + len(marker):]
    return int(digits) if digits.isdigit() else -1


def collect_checkpoints(run_directory):
    """Return every policy checkpoint of one run, oldest iteration first.

    Unnumbered checkpoints -- the run's current weights -- sort last, because
    they are the newest state the run reached.
    """
    run_directory = Path(run_directory)
    if not run_directory.is_dir():
        raise NotADirectoryError(f"No run directory at {run_directory}.")
    paths = [
        path
        for path in run_directory.glob("*.npz")
        if not path.name.endswith(".resume.npz")
    ]
    paths.extend((run_directory / "checkpoint_archive").glob("*.npz"))
    paths = [path for path in paths if _is_policy_checkpoint(path)]
    # An unnumbered checkpoint is the run's current state, so it is newer than
    # every archived iteration and must sort after all of them.
    return sorted(
        paths,
        key=lambda path: (
            _iteration_of(path) if _iteration_of(path) >= 0 else float("inf"),
            path.name,
        ),
    )


def report(run_directory, *, per_layer=False):
    """Return the ordered health rows of every checkpoint in one run."""
    rows = [inspect_checkpoint(path) for path in collect_checkpoints(run_directory)]
    if not per_layer:
        for row in rows:
            row.pop("per_layer")
    return rows


def _format(rows, *, per_layer):
    lines = [
        f"{'iteration':>10}  {'finite':>6}  {'max_abs_weight':>16}  checkpoint",
        f"{'-' * 10}  {'-' * 6}  {'-' * 16}  {'-' * 40}",
    ]
    for row in rows:
        iteration = _iteration_of(Path(row["path"]))
        lines.append(
            f"{(iteration if iteration >= 0 else 'current'):>10}  "
            f"{('yes' if row['healthy'] else 'NO'):>6}  "
            f"{row['max_abs_weight']:>16.6g}  {row['name']}"
        )
        if per_layer:
            for name, detail in row.get("per_layer", {}).items():
                lines.append(
                    f"{'':>10}  {'':>6}  {detail['max_abs']:>16.6g}    "
                    f"{name} {tuple(detail['shape'])} "
                    f"mean_abs={detail['mean_abs']:.6g} "
                    f"non_finite={detail['non_finite']}"
                )
    unhealthy = [row for row in rows if not row["healthy"]]
    lines.append("")
    if not rows:
        lines.append("No policy checkpoints found in this run directory.")
    elif unhealthy:
        healthy = [row for row in rows if row["healthy"]]
        lines.append(
            f"{len(unhealthy)} of {len(rows)} checkpoints contain NaN or "
            "infinite parameters."
        )
        if healthy:
            last = healthy[-1]
            lines.append(f"Last finite checkpoint: {last['name']}")
    else:
        lines.append(
            f"All {len(rows)} checkpoints are finite. "
            f"max_abs_weight spans {min(row['max_abs_weight'] for row in rows):.6g} "
            f"to {max(row['max_abs_weight'] for row in rows):.6g}."
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Report NaN/Inf parameters and weight magnitude across the saved "
            "policy checkpoints of one RL run. Read-only."
        )
    )
    parser.add_argument(
        "run_directory",
        help="an RL run directory, for example models/rl/domino_rl_<...>",
    )
    parser.add_argument(
        "--per-layer",
        action="store_true",
        help="also report every weight and bias array of every checkpoint",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the rows as JSON instead of a table",
    )
    args = parser.parse_args(argv)
    rows = report(args.run_directory, per_layer=args.per_layer or args.json)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(_format(rows, per_layer=args.per_layer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
