#!/usr/bin/env python3
"""Run two 30-iteration timing/learning-curve jobs per ruleset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RAW = HERE / "raw_30_iterations"
MANIFEST = HERE / "benchmark_manifest_30_iterations.json"
RULESETS = ("double-three", "double-four", "double-five", "double-six")
SCHEDULE = (
    (1, "double-three"),
    (1, "double-four"),
    (1, "double-five"),
    (1, "double-six"),
    (2, "double-six"),
    (2, "double-five"),
    (2, "double-four"),
    (2, "double-three"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=60_000)
    parser.add_argument("--gpi", type=int, default=2_000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_260_828)
    parser.add_argument("--ppo-epochs", type=int, default=16)
    return parser.parse_args()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def machine_snapshot() -> dict:
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "logical_cpus": os.cpu_count(),
        "load_average": os.getloadavg(),
        "nvidia_smi": command_output([
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
    }


def write_manifest(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if RAW.exists():
        raise SystemExit(f"{RAW} already exists; refusing to overwrite artifacts.")
    RAW.mkdir(parents=True)
    manifest = {
        "format": "domino_rl_ruleset_learning_curve_benchmark",
        "version": 1,
        "configuration": {
            "rulesets": list(RULESETS),
            "repetitions": 2,
            "training_games_per_run": args.games,
            "iterations_per_run": args.games // args.gpi,
            "gpi": args.gpi,
            "workers": args.workers,
            "seed": args.seed,
            "ppo_max_epochs": args.ppo_epochs,
            "device": "gpu",
            "opponent_buckets": ["heuristic", "recent"],
            "difficulty_weight": 0.5,
            "initialization": "ruleset-default random policy from fixed seed",
            "periodic_diagnostics_during_training": False,
            "supervised_training": False,
            "retained_policy_interval_iterations": 1,
        },
        "schedule": [
            {"repetition": repetition, "ruleset": ruleset}
            for repetition, ruleset in SCHEDULE
        ],
        "machine_before": machine_snapshot(),
        "runs": [],
    }
    write_manifest(MANIFEST, manifest)

    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(ROOT), environment.get("PYTHONPATH"))
        if value
    )
    for index, (repetition, ruleset) in enumerate(SCHEDULE, start=1):
        run_dir = RAW / ruleset / f"rep{repetition}"
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        log_path = run_dir.parent / f"rep{repetition}.console.log"
        command = [
            sys.executable,
            str(HERE / "run_learning_curve_one.py"),
            "--ruleset", ruleset,
            "--output-dir", str(run_dir),
            "--games", str(args.games),
            "--gpi", str(args.gpi),
            "--workers", str(args.workers),
            "--seed", str(args.seed),
            "--ppo-epochs", str(args.ppo_epochs),
        ]
        print(f"[{index}/{len(SCHEDULE)}] {ruleset} repetition {repetition}", flush=True)
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.perf_counter() - started
        manifest["runs"].append({
            "ruleset": ruleset,
            "repetition": repetition,
            "returncode": process.returncode,
            "subprocess_wall_seconds": elapsed,
            "output_directory": str(run_dir.relative_to(HERE)),
            "console_log": str(log_path.relative_to(HERE)),
        })
        write_manifest(MANIFEST, manifest)
        print(f"    {elapsed:.2f}s, exit {process.returncode}", flush=True)
        if process.returncode:
            raise SystemExit(f"Benchmark failed; inspect {log_path}")

    manifest["machine_after"] = machine_snapshot()
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(MANIFEST, manifest)
    print("All 30-iteration benchmark runs completed.", flush=True)


if __name__ == "__main__":
    main()
