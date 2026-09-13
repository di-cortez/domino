"""Scaled canonical pipeline benchmark: synchronous versus asynchronous monitors.

Run it with ``PYTHONPATH`` pointing at the checkout under test. One invocation
runs one canonical ``big`` pipeline in a fresh artifact root with a copied
supervised checkpoint, keeping the production ratio of one monitor game per
RL game at a smaller cadence, and writes ``benchmark.json`` with:

- total RL-stage wall clock and the progress-clock end point;
- per-iteration ``rollout_seconds`` and ``update_seconds`` from the metrics;
- the cumulative runtime profile's RL and diagnostic totals;
- the recorded monitor points without timing fields;
- sampled whole-machine CPU utilization and the queue depth over time.

``compare`` summarizes two or more benchmark files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from types import SimpleNamespace


def _cpu_times():
    with open("/proc/stat", encoding="utf-8") as stream:
        fields = stream.readline().split()[1:]
    values = [int(value) for value in fields]
    idle = values[3] + values[4]
    return sum(values), idle


class _Sampler(threading.Thread):
    """Sample machine CPU utilization and the diagnostic queue depth."""

    def __init__(self, queue_dir, interval=1.0):
        super().__init__(daemon=True)
        self.queue_dir = Path(queue_dir)
        self.interval = interval
        self.samples = []
        self._halt = threading.Event()

    def run(self):
        previous = _cpu_times()
        started = time.monotonic()
        while not self._halt.wait(self.interval):
            current = _cpu_times()
            total = current[0] - previous[0]
            idle = current[1] - previous[1]
            previous = current
            depth = (
                len(list(self.queue_dir.glob("task_games_*.json")))
                if self.queue_dir.is_dir() else 0
            )
            self.samples.append({
                "t": time.monotonic() - started,
                "cpu_busy_percent": (
                    100.0 * (total - idle) / total if total else 0.0
                ),
                "queue_depth": depth,
            })

    def stop(self):
        self._halt.set()
        self.join()


def cmd_run(args):
    # pylint: disable=import-outside-toplevel
    import training.pipeline as pipeline
    from diagnostics.rl_progress import read_periodic_history
    from training.rl.reporting import read_training_metrics
    from training.run_artifacts import periodic_diagnostics_path
    from utils.artifacts import atomic_copy, file_sha256

    root = Path(args.output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"{root} is not empty.")
    root.mkdir(parents=True, exist_ok=True)
    supervised = atomic_copy(Path(args.weights), root / "supervised.npz")
    argv = [
        "big",
        "--artifact-root", str(root),
        "--seed", str(args.seed),
        "--total-training-games", str(args.total_games),
        "--gpi", str(args.gpi),
        "--opponent-buckets", "random",
        "--rl-workers", str(args.rl_workers),
        "--device", args.device,
        "--learning-rate", "0.0008",
        "--ppo-max-epochs", "16",
        "--periodic-diagnostic-games", str(args.every_games),
        "--periodic-diagnostic-every-games", str(args.every_games),
        "--diagnostic-workers", str(args.diagnostic_workers),
        "--skip-final-diagnostic",
    ]
    if args.mode == "async":
        argv += [
            "--async-periodic-diagnostics",
            "--async-diagnostic-workers", str(args.diagnostic_workers),
        ]
    parsed = pipeline.parse_args(argv)
    config = pipeline._build_config(parsed.scale)  # pylint: disable=protected-access
    pipeline.validate_args(parsed, config)
    assets = {
        "paths": SimpleNamespace(weights=supervised),
        "weights_metadata": {"artifact": {"sha256": file_sha256(supervised)}},
    }
    run_dir = pipeline._pipeline_run_dir(root, config, parsed)  # pylint: disable=protected-access
    sampler = _Sampler(run_dir / "diagnostics" / "periodic_queue")
    sampler.start()
    started = time.monotonic()
    try:
        result = pipeline.run_rl_pipeline(root, config, parsed, assets)
    finally:
        wall = time.monotonic() - started
        sampler.stop()
    run_dir = Path(result["run_dir"])
    _header, metrics = read_training_metrics(
        next(run_dir.glob("**/training_metrics.jsonl"))
    )
    history = read_periodic_history(periodic_diagnostics_path(run_dir))
    profile = json.loads(
        (run_dir / "diagnostics" / "runtime_profile.json").read_text()
    )
    cumulative = profile["cumulative"]
    payload = {
        "kind": "pipeline_benchmark",
        "mode": args.mode,
        "pythonpath": os.environ.get("PYTHONPATH"),
        "arguments": vars(args),
        "wall_seconds": wall,
        "rl_elapsed_seconds": float(result["elapsed_rl_seconds"]),
        "progress_clock_seconds": float(history[-1]["progress_elapsed_seconds"]),
        "completed_training_games": int(result["completed_training_games"]),
        "iterations": [
            {
                "iteration": int(row["iteration"]),
                "rollout_seconds": float(row["rollout_seconds"]),
                "update_seconds": float(row["update_seconds"]),
                "iteration_seconds": float(row["iteration_seconds"]),
            }
            for row in metrics
        ],
        "profile": {
            "rl_execution_seconds": cumulative["rl"]["execution_seconds"],
            "rollout_game_execution_seconds": cumulative["rl"]["sections_seconds"].get(
                "rollout_game_execution"
            ),
            "ppo_update_seconds": cumulative["rl"]["sections_seconds"].get("ppo_update"),
            "diagnostic_execution_seconds": cumulative["rl_vs_random_diagnostics"][
                "execution_seconds"
            ],
            "diagnostic_pairwise_seconds": cumulative["rl_vs_random_diagnostics"][
                "sections_seconds"
            ].get("pairwise_evaluation"),
            "async_queue_wait_seconds": cumulative["rl_vs_random_diagnostics"].get(
                "async_queue_wait_seconds"
            ),
            "async_blocking_seconds": cumulative["rl_vs_random_diagnostics"].get(
                "async_blocking_seconds"
            ),
        },
        "points": [
            {
                key: row[key]
                for key in ("rl_games", "rl_iterations", "wins", "diagnostic_seconds")
            }
            for row in history
        ],
        "samples": sampler.samples,
    }
    (root / "benchmark.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps({
        key: payload[key]
        for key in ("mode", "wall_seconds", "rl_elapsed_seconds", "progress_clock_seconds")
    }))


def _summary(path):
    data = json.loads(Path(path).read_text())
    iterations = data["iterations"]
    busy = [sample["cpu_busy_percent"] for sample in data["samples"]]
    depth = [sample["queue_depth"] for sample in data["samples"]]
    return {
        "file": str(path),
        "mode": data["mode"],
        "wall_seconds": round(data["wall_seconds"], 2),
        "rl_elapsed_seconds": round(data["rl_elapsed_seconds"], 2),
        "progress_clock_seconds": round(data["progress_clock_seconds"], 2),
        "median_rollout_seconds": round(
            statistics.median(row["rollout_seconds"] for row in iterations), 3
        ),
        "median_update_seconds": round(
            statistics.median(row["update_seconds"] for row in iterations), 3
        ),
        "diagnostic_execution_seconds": round(
            data["profile"]["diagnostic_execution_seconds"], 2
        ),
        "async_blocking_seconds": data["profile"]["async_blocking_seconds"],
        "mean_cpu_busy_percent": round(statistics.mean(busy), 1) if busy else None,
        "max_queue_depth": max(depth, default=0),
        "wins": [point["wins"] for point in data["points"]],
    }


def cmd_compare(args):
    rows = [_summary(path) for path in args.files]
    print(json.dumps(rows, indent=1))
    if args.output:
        Path(args.output).write_text(json.dumps(rows, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--mode", choices=("sync", "async"), required=True)
    run.add_argument("--weights", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--seed", type=int, default=52)
    run.add_argument("--gpi", type=int, default=2000)
    run.add_argument("--total-games", type=int, default=60000)
    run.add_argument("--every-games", type=int, default=10000)
    run.add_argument("--rl-workers", type=int, default=8)
    run.add_argument("--diagnostic-workers", type=int, default=4)
    run.add_argument("--device", default="gpu")
    compare = commands.add_parser("compare")
    compare.add_argument("files", nargs="+")
    compare.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    {"run": cmd_run, "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
