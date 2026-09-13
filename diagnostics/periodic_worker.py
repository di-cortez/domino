"""Measure one queued periodic diagnostic in its own low-priority process.

Started by ``diagnostics.periodic_queue`` with a task path:

    python -m diagnostics.periodic_worker <run>/diagnostics/periodic_queue/task_games_N.json

The process lowers its own scheduling priority, stays CPU-only, verifies the
milestone checkpoint against the hash recorded at enqueue time, plays the
point's games with the task's fixed worker count, and atomically writes one
result envelope. It writes nothing else the training process owns. It exits
when its parent dies, and turns SIGTERM into an exception so the diagnostic
runner terminates its game workers before the process ends.
"""

from __future__ import annotations

import os

# Before any project import: `agents.nn` probes CUDA at import time, and this
# process must never create a GPU context beside the learner.
os.environ["DOMINO_FORCE_CPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# pylint: disable=wrong-import-position
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import time

from diagnostics import periodic_queue
from diagnostics.parallel_runner import ParallelSafetyConfig
from diagnostics.rl_progress import measure_periodic_point
from utils.artifacts import atomic_write_json, file_sha256

_PR_SET_PDEATHSIG = 1


class WorkerTerminated(BaseException):
    """SIGTERM reached the measuring process."""


def _raise_terminated(signum, _frame):
    raise WorkerTerminated(signal.Signals(signum).name)


def _die_with_parent(expected_parent_pid):
    """Receive SIGTERM when the training process exits, where supported."""
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            libc.prctl(_PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0)
        except (OSError, AttributeError):
            pass
    # The parent may have exited before the request above took effect.
    if expected_parent_pid is not None and os.getppid() != int(expected_parent_pid):
        raise WorkerTerminated("parent exited before the worker started")


def run_task(path):
    """Measure the task at ``path`` and write its result envelope."""
    path = Path(path)
    task = json.loads(path.read_text(encoding="utf-8"))
    if (
        task.get("format") != periodic_queue.TASK_FORMAT
        or task.get("format_version") != periodic_queue.QUEUE_FORMAT_VERSION
    ):
        raise ValueError(f"Not a periodic diagnostic task: {path}.")
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    checkpoint = Path(task["checkpoint_path"])
    actual = file_sha256(checkpoint)
    if actual != task["checkpoint_sha256"]:
        raise ValueError(
            f"Checkpoint {checkpoint} changed after it was queued: recorded "
            f"{task['checkpoint_sha256']}, found {actual}."
        )
    sections = {}
    row, pairwise_profile, selected_workers = measure_periodic_point(
        run_dir=task["run_dir"],
        pipeline_level=task["pipeline_level"],
        seed=task["seed"],
        rl_games=task["rl_games"],
        rl_iterations=task["rl_iterations"],
        checkpoint_path=checkpoint,
        diagnostic_games=task["diagnostic_games"],
        rl_elapsed_seconds=task["rl_elapsed_seconds"],
        training_window=task["training_window"],
        workers=int(task["workers"]),
        safety_config=ParallelSafetyConfig(**task["safety_config"]),
        status_callback=lambda message: print(message, flush=True),
        runtime_sections=sections,
    )
    atomic_write_json(
        periodic_queue.result_path(task["run_dir"], task["rl_games"]),
        {
            "format": periodic_queue.RESULT_FORMAT,
            "format_version": periodic_queue.QUEUE_FORMAT_VERSION,
            "identity": task["identity"],
            "checkpoint_sha256": task["checkpoint_sha256"],
            "row": row,
            "measure_seconds": time.perf_counter() - started,
            "measure_sections_seconds": sections,
            "pairwise_runtime_profile": pairwise_profile,
            "selected_workers": int(selected_workers),
            "worker_pid": os.getpid(),
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return row


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) != 1:
        print(__doc__.strip().splitlines()[0], file=sys.stderr)
        print("usage: python -m diagnostics.periodic_worker TASK_PATH", file=sys.stderr)
        return 2
    task = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    signal.signal(signal.SIGTERM, _raise_terminated)
    signal.signal(signal.SIGINT, _raise_terminated)
    try:
        _die_with_parent(task.get("parent_pid"))
        try:
            os.nice(int(task.get("niceness", 0)))
        except OSError as exc:
            print(f"Could not lower the scheduling priority: {exc}", flush=True)
        row = run_task(argv[0])
    except WorkerTerminated as exc:
        print(f"Periodic diagnostic worker stopped: {exc}.", flush=True)
        return 143
    print(
        f"Periodic diagnostic at {int(row['rl_games']):,} RL games measured: "
        f"{int(row['wins']):,}/{int(row['diagnostic_games']):,} wins.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
