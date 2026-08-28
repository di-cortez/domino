#!/usr/bin/env python3
"""Run one profiled RL job while retaining every iteration policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from training.rl.config import (
    RLExecutionOptions,
    RLResourceOptions,
    RLTrainingOptions,
)
from training.rl.ppo import PPO_TRAINING_ALGORITHM
from training.rl.resume import _load_initial_network
from training.rl.training_loop import train
from utils.artifacts import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ruleset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=60_000)
    parser.add_argument("--gpi", type=int, default=2_000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_260_828)
    parser.add_argument("--ppo-epochs", type=int, default=16)
    return parser.parse_args()


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    args = parse_args()
    if args.games % args.gpi:
        raise ValueError("games must be an exact multiple of GPI")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    retained_dir = output_dir / "iteration_weights"
    retained_dir.mkdir()
    weights_path = output_dir / "rl_weights.npz"
    nonexistent_sl_path = output_dir / "intentionally_absent_sl_weights.npz"

    training = RLTrainingOptions(
        ruleset_name=args.ruleset,
        total_training_games=args.games,
        gpi=args.gpi,
        opponent_buckets=("heuristic", "recent"),
        difficulty_weight=0.5,
        seed=args.seed,
        ppo_max_epochs=args.ppo_epochs,
    )
    resources = RLResourceOptions(
        sl_weights_path=nonexistent_sl_path,
        rl_weights_path=weights_path,
        device="gpu",
        workers=args.workers,
    )

    # Save the exact random policy used by the training loader before iteration
    # one.  This supplies the learning curve's true zero-game baseline.
    initial = _load_initial_network(
        training.learning_rate,
        nonexistent_sl_path,
        None,
        quiet=True,
        use_value_head=training.use_value_head,
        device="gpu",
        fresh_from_sl=True,
        expected_training_algorithm=PPO_TRAINING_ALGORITHM,
        weight_decay=training.weight_decay,
        dropout_rate=training.dropout_rate,
        ruleset=args.ruleset,
        initialization_seed=args.seed,
        use_opponent_suit_features=training.use_opponent_suit_features,
    )
    initial.synchronize()
    initial_path = retained_dir / "iteration_000000.npz"
    initial.save(initial_path)
    del initial

    retained = {0: str(initial_path)}

    def retain_checkpoint(payload) -> None:
        iteration = int(payload["rl_iterations_completed"])
        source = Path(payload["rl_weights_path"])
        destination = retained_dir / f"iteration_{iteration:06d}.npz"
        shutil.copy2(source, destination)
        retained[iteration] = str(destination)

    execution = RLExecutionOptions(
        log_interval=max(1, args.games // args.gpi),
        checkpoint_interval=1,
        fresh_from_sl=True,
        numbered_checkpoints=True,
        quiet=True,
        checkpoint_callback=retain_checkpoint,
    )
    summary = train(training, resources, execution)
    expected_iterations = args.games // args.gpi
    missing = sorted(set(range(expected_iterations + 1)) - set(retained))
    if missing:
        raise RuntimeError(f"Missing retained iteration policies: {missing}")

    checkpoint_hashes = {
        str(iteration): file_sha256(path)
        for iteration, path in sorted(retained.items())
    }
    summary["retained_iteration_weights"] = {
        "count": len(retained),
        "directory": str(retained_dir),
        "sha256_by_iteration": checkpoint_hashes,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ruleset": args.ruleset,
                "games": summary["completed_training_games"],
                "decisions": summary["total_decision_samples"],
                "profile_s": summary["runtime_profile_delta"]["execution_seconds"],
                "retained_policies": len(retained),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"BENCHMARK_FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        raise
