#!/usr/bin/env python3
"""Run one isolated RL timing benchmark and persist its complete profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from training.rl.config import (
    RLExecutionOptions,
    RLResourceOptions,
    RLTrainingOptions,
)
from training.rl.training_loop import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ruleset", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=24_000)
    parser.add_argument("--gpi", type=int, default=2_000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_260_828)
    parser.add_argument("--ppo-epochs", type=int, default=16)
    return parser.parse_args()


def json_default(value):
    """Convert the few possible NumPy leaves in public summaries."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
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
    execution = RLExecutionOptions(
        log_interval=max(1, args.games // args.gpi),
        checkpoint_interval=1_000_000,
        fresh_from_sl=True,
        quiet=True,
    )
    summary = train(training, resources, execution)
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
                "duration_s": summary["duration_s"],
                "profile_s": summary["runtime_profile_delta"]["execution_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # Preserve a concise marker in the captured log.
        print(f"BENCHMARK_FAILED: {type(error).__name__}: {error}", file=sys.stderr)
        raise
