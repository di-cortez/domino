#!/usr/bin/env python3
"""Evaluate every retained learner policy against one fixed random panel."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.pairwise import run_pairwise
from diagnostics.parallel_runner import ParallelSafetyConfig
from diagnostics.plots import wilson_interval
from diagnostics.rl_progress import periodic_diagnostic_seed
from utils.artifacts import file_sha256


RAW = HERE / "raw_30_iterations"
BENCHMARK_MANIFEST = HERE / "benchmark_manifest_30_iterations.json"
CURVE_ROWS = HERE / "learning_curve_raw.jsonl"
CURVE_MANIFEST = HERE / "learning_curve_manifest.json"
OUTPUT_ROOT = HERE / "curve_diagnostics"
RULESETS = ("double-three", "double-four", "double-five", "double-six")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=10)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_existing() -> dict[tuple, dict]:
    if not CURVE_ROWS.is_file():
        return {}
    rows = {}
    with CURVE_ROWS.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (
                row["ruleset"],
                int(row["repetition"]),
                int(row["iteration"]),
                int(row["diagnostic_games"]),
            )
            rows[key] = row
    return rows


def append_row(row: dict) -> None:
    with CURVE_ROWS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def checkpoint_data(ruleset: str) -> tuple[list[dict], bool]:
    summaries = [
        load_json(RAW / ruleset / f"rep{repetition}" / "summary.json")
        for repetition in (1, 2)
    ]
    expected_iterations = set(range(31))
    hash_maps = []
    for summary in summaries:
        hashes = {
            int(iteration): digest
            for iteration, digest in summary["retained_iteration_weights"][
                "sha256_by_iteration"
            ].items()
        }
        if set(hashes) != expected_iterations:
            raise RuntimeError(f"Incomplete retained curve for {ruleset}")
        hash_maps.append(hashes)
    identical = hash_maps[0] == hash_maps[1]
    return summaries, identical


def main() -> None:
    args = parse_args()
    benchmark = load_json(BENCHMARK_MANIFEST)
    training_seed = int(benchmark["configuration"]["seed"])
    diagnostic_seed = int(periodic_diagnostic_seed(training_seed))
    existing = load_existing()
    manifest = {
        "format": "domino_rl_ruleset_learning_curve_diagnostics",
        "version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_games_per_checkpoint": args.games,
        "diagnostic_workers": args.workers,
        "training_seed": training_seed,
        "diagnostic_seed": diagnostic_seed,
        "diagnostic_opponent": "random",
        "policy_mode": "deterministic evaluation",
        "common_fixed_panel_across_checkpoints": True,
        "rulesets": {},
    }
    CURVE_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    total_points = 0
    evaluation_plan = []
    for ruleset in RULESETS:
        summaries, identical = checkpoint_data(ruleset)
        repetitions = (1,) if identical else (1, 2)
        manifest["rulesets"][ruleset] = {
            "byte_identical_iteration_weights": identical,
            "evaluated_repetitions": list(repetitions),
        }
        for repetition in repetitions:
            hashes = summaries[repetition - 1]["retained_iteration_weights"][
                "sha256_by_iteration"
            ]
            for iteration in range(31):
                evaluation_plan.append((ruleset, repetition, iteration, hashes[str(iteration)]))
    total_points = len(evaluation_plan)
    CURVE_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for point_index, (ruleset, repetition, iteration, expected_hash) in enumerate(
        evaluation_plan,
        start=1,
    ):
        key = (ruleset, repetition, iteration, args.games)
        checkpoint = (
            RAW
            / ruleset
            / f"rep{repetition}"
            / "iteration_weights"
            / f"iteration_{iteration:06d}.npz"
        )
        actual_hash = file_sha256(checkpoint)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Checkpoint hash changed: {checkpoint}")
        previous = existing.get(key)
        if previous and previous.get("checkpoint_sha256") == actual_hash:
            print(
                f"[{point_index}/{total_points}] {ruleset} iteration {iteration}: reused",
                flush=True,
            )
            continue

        print(
            f"[{point_index}/{total_points}] {ruleset} iteration {iteration}: evaluating",
            flush=True,
        )
        output_dir = (
            OUTPUT_ROOT
            / ruleset
            / f"rep{repetition}"
            / f"iteration_{iteration:06d}"
        )
        result = run_pairwise(
            "rl",
            "random",
            game_count=args.games,
            weights=checkpoint,
            seed=diagnostic_seed,
            effective_seed=diagnostic_seed,
            output_dir=output_dir,
            generate_plots=False,
            print_console_summary=False,
            print_memory_summary=False,
            workers=args.workers,
            safety_config=ParallelSafetyConfig(),
            save_game_records=False,
            ruleset=ruleset,
        )
        summary = result["summary"]
        wins = int(summary["counts"]["win"])
        losses = int(summary["counts"]["loss"])
        low, high = wilson_interval(wins, wins + losses)
        row = {
            "ruleset": ruleset,
            "repetition": repetition,
            "iteration": iteration,
            "training_games": iteration * int(benchmark["configuration"]["gpi"]),
            "checkpoint_path": str(checkpoint.relative_to(HERE)),
            "checkpoint_sha256": actual_hash,
            "diagnostic_games": args.games,
            "diagnostic_seed": diagnostic_seed,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / (wins + losses),
            "ci95_low": low,
            "ci95_high": high,
            "diagnostic_seconds": float(result["duration_s"]),
            "runtime_profile": result["runtime_profile_delta"],
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        append_row(row)
        existing[key] = row
        print(
            f"    {100.0 * row['win_rate']:.2f}% wins, "
            f"{row['diagnostic_seconds']:.2f}s",
            flush=True,
        )

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["completed_points"] = total_points
    CURVE_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("All learning-curve diagnostics completed.", flush=True)


if __name__ == "__main__":
    started = time.perf_counter()
    main()
    print(f"Elapsed: {time.perf_counter() - started:.2f}s", flush=True)
