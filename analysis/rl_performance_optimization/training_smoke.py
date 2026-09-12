"""Fixed-seed real ``train()`` smoke with an exact stop/resume counterpart.

Run it with ``PYTHONPATH`` pointing at the checkout under test. One invocation
trains the same run twice into a fresh output directory:

    full/   uninterrupted to ``--iterations``
    split/  stopped after ``--stop-after`` iterations, then resumed to the end

and writes ``fingerprint.json``: per-array SHA-256 of both final checkpoints,
the deterministic training-metrics rows, and the warmup trace when enabled.
``compare`` diffs two fingerprints, for example reference versus candidate.

Rollout workers stay CPU-only as always; ``--device`` selects the learner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

# Columns whose values depend on wall time, machine load, output location, or
# the length of one invocation rather than on learning. Everything else,
# including every PPO statistic, must match exactly.
VOLATILE_COLUMNS = frozenset({
    "selected_workers",
    "checkpoint_path",
    "total_iterations",
})


FAST_WARMUP = {
    "exponent": 3,
    "hold_iterations": 2,
    "cooldown_iterations": 1,
}


def _supervised_path(root, ruleset, weights):
    if weights is not None:
        return Path(weights)
    # pylint: disable=import-outside-toplevel
    from agents.encoder import DominoEncoder
    from agents.rl_nn import PolicyNetwork

    path = Path(root) / "supervised.npz"
    if not path.exists():
        encoder = DominoEncoder(ruleset)
        network = PolicyNetwork(
            input_size=encoder.vector_size,
            hidden_sizes=(64, 32),
            output_size=encoder.action_size,
            random_seed=7,
            device="cpu",
        )
        np.savez(path, **{
            name: np.asarray(getattr(network, name))
            for name in network.weight_names
        })
    return path


def _train(root, args, *, stop_after=None, resume_weights=None, resume_state=None):
    # pylint: disable=import-outside-toplevel
    from diagnostics.parallel_runner import ParallelSafetyConfig
    from training.rl.config import (
        RLExecutionOptions,
        RLResourceOptions,
        RLTrainingOptions,
    )
    from training.rl.training_loop import train

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    warmup = None
    if args.warmup:
        warmup = dict(FAST_WARMUP)
        if args.warmup_kl_threshold is not None:
            warmup["kl_threshold"] = args.warmup_kl_threshold
    return train(
        RLTrainingOptions(
            ruleset_name=args.ruleset,
            total_training_games=args.gpi * args.iterations,
            gpi=args.gpi,
            opponent_buckets=tuple(args.opponent_buckets.split(",")),
            baseline=(args.baseline,),
            use_value_head=args.baseline.startswith("value-head"),
            difficulty_weight=0.5,
            learning_rate=args.learning_rate,
            entropy_coef=args.entropy_coef,
            dropout_rate=args.dropout,
            warmup_lr=warmup,
            seed=args.seed,
            ppo_max_epochs=args.ppo_max_epochs,
        ),
        RLResourceOptions(
            sl_weights_path=_supervised_path(root.parent, args.ruleset, args.weights),
            rl_weights_path=root / "training.npz",
            device=args.device,
            workers=args.workers,
            safety_config=ParallelSafetyConfig(
                memory_reserve_mb=0,
                estimated_worker_mb=1,
                max_worker_rss_mb=2048,
            ),
        ),
        RLExecutionOptions(
            quiet=True,
            checkpoint_interval=1,
            numbered_checkpoints=True,
            fresh_from_sl=resume_weights is None,
            stop_after_training_games=stop_after,
            resume_weights_path=resume_weights,
            resume_state_file=resume_state,
        ),
    )


def _array_hashes(path):
    with np.load(path, allow_pickle=False) as data:
        return {
            name: {
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(data[name]).tobytes()
                ).hexdigest(),
                "dtype": str(data[name].dtype),
                "shape": list(data[name].shape),
            }
            for name in sorted(data.files)
        }


def _volatile(column):
    return column in VOLATILE_COLUMNS or column.endswith("_seconds")


def _metrics_rows(root):
    path = Path(root) / "training_training_metrics.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    columns = header.get("columns") if isinstance(header, dict) else None
    rows = [json.loads(line) for line in lines[1:]]
    if columns is None:
        return {"header": header, "rows": rows, "volatile_columns": []}
    kept = [index for index, column in enumerate(columns) if not _volatile(column)]
    return {
        "columns": [columns[index] for index in kept],
        "volatile_columns": [
            column for column in columns if _volatile(column)
        ],
        "rows": [[row[index] for index in kept] for row in rows],
    }


def _warmup_rows(root):
    # pylint: disable=import-outside-toplevel
    from training.rl.lr_warmup import warmup_trace_path

    path = warmup_trace_path(Path(root) / "training_training_metrics.jsonl")
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[1:]]


def cmd_run(args):
    """Train uninterrupted and split runs, then write their fingerprint."""
    # pylint: disable=import-outside-toplevel
    from training.rl.resume import resume_state_path

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"{output} is not empty; choose a fresh directory.")
    full = _train(output / "full", args)
    partial = _train(output / "split", args, stop_after=args.gpi * args.stop_after)
    resumed = _train(
        output / "split",
        args,
        resume_weights=partial["rl_weights_path"],
        resume_state=resume_state_path(partial["rl_weights_path"]),
    )
    fingerprint = {
        "kind": "training_smoke_fingerprint",
        "pythonpath": os.environ.get("PYTHONPATH"),
        "arguments": vars(args),
        "full": {
            "weights": _array_hashes(full["rl_weights_path"]),
            "metrics": _metrics_rows(output / "full"),
            "warmup": _warmup_rows(output / "full"),
            "optimizer_step_count": full.get("optimizer_step_count"),
            "completed_training_games": full.get("completed_training_games"),
        },
        "split": {
            "weights": _array_hashes(resumed["rl_weights_path"]),
            "metrics": _metrics_rows(output / "split"),
            "warmup": _warmup_rows(output / "split"),
            "optimizer_step_count": resumed.get("optimizer_step_count"),
            "completed_training_games": resumed.get("completed_training_games"),
        },
    }
    fingerprint["full_equals_split"] = _compare_sections(
        fingerprint["full"], fingerprint["split"]
    )
    (output / "fingerprint.json").write_text(
        json.dumps(fingerprint, indent=1),
        encoding="utf-8",
    )
    from training.rl.lr_warmup import WARMUP_TRACE_COLUMNS

    promotions = None
    if fingerprint["full"]["warmup"] is not None:
        rows = [
            dict(zip(WARMUP_TRACE_COLUMNS, row))
            for row in fingerprint["full"]["warmup"]
        ]
        promotions = [row["iteration"] for row in rows if row["promoted"] is True]
    print(json.dumps({
        "fingerprint": str(output / "fingerprint.json"),
        "full_equals_split": fingerprint["full_equals_split"],
        "warmup_promotions": promotions,
        "optimizer_steps": fingerprint["full"]["optimizer_step_count"],
    }, indent=1))


def _compare_sections(left, right):
    return {
        "weights": left["weights"] == right["weights"],
        "metrics_rows": left["metrics"]["rows"] == right["metrics"]["rows"],
        "warmup": left["warmup"] == right["warmup"],
        "optimizer_step_count": (
            left["optimizer_step_count"] == right["optimizer_step_count"]
        ),
    }


def cmd_compare(args):
    """Compare two fingerprints section by section."""
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    result = {
        section: _compare_sections(reference[section], candidate[section])
        for section in ("full", "split")
    }
    columns = reference["full"]["metrics"].get("columns")
    first_difference = None
    if columns is not None:
        for left, right in zip(
            reference["full"]["metrics"]["rows"],
            candidate["full"]["metrics"]["rows"],
        ):
            if left != right:
                first_difference = {
                    column: [a, b]
                    for column, a, b in zip(columns, left, right)
                    if a != b
                }
                break
    result["first_metrics_row_difference"] = first_difference
    result["reference_full_equals_split"] = reference["full_equals_split"]
    result["candidate_full_equals_split"] = candidate["full_equals_split"]
    print(json.dumps(result, indent=1))
    exact = all(
        all(section.values()) for section in (result["full"], result["split"])
    )
    sys.exit(0 if exact else 1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help=cmd_run.__doc__)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--ruleset", default="double-three")
    run.add_argument("--weights", default=None)
    run.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--gpi", type=int, default=2000)
    run.add_argument("--iterations", type=int, default=8)
    run.add_argument("--stop-after", type=int, default=3)
    run.add_argument("--opponent-buckets", default="heuristic,recent")
    run.add_argument("--baseline", default="batch-mean")
    run.add_argument("--learning-rate", type=float, default=0.01)
    run.add_argument("--entropy-coef", type=float, default=0.0)
    run.add_argument("--dropout", type=float, default=0.0)
    run.add_argument("--ppo-max-epochs", type=int, default=16)
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--warmup", action="store_true")
    run.add_argument("--warmup-kl-threshold", type=float, default=None)
    compare = commands.add_parser("compare", help=cmd_compare.__doc__)
    compare.add_argument("reference")
    compare.add_argument("candidate")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    {"run": cmd_run, "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    main()
