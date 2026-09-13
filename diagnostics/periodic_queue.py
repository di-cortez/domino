"""Durable queue that measures periodic diagnostics beside training.

A periodic RL-vs-random point is observational: nothing in training reads it
back. Measuring it synchronously nevertheless held the RL loop still for the
whole diagnostic. This queue lets the training process hand a finished
milestone checkpoint to one separate, low-priority, CPU-only process and keep
training while that process plays the games.

Ownership is deliberately one-sided. The training process is the only writer
of task records, the periodic history, the reports, the best pointer, and the
training-state markers. The measuring process (``diagnostics.periodic_worker``)
reads one task and writes exactly one file, its own result envelope, by atomic
replacement. Everything is durable, so a stop at any moment leaves either a
task to measure again or a result to publish, never a half-published point:

    enqueue          task_games_N.json  state "pending", window captured now
    start            state "running"
    worker finishes  result_games_N.json appears; state "completed"
    publish          history append (idempotent), reports, markers
    acknowledge      task and result removed

At most one task runs at a time and tasks run in checkpoint order, so results
are always published in order. Tasks reference the immutable milestone weights
``checkpoints/games_N_weights.npz``, which no retention policy removes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from utils.artifacts import atomic_write_json

TASK_FORMAT = "domino_periodic_diagnostic_task"
RESULT_FORMAT = "domino_periodic_diagnostic_result"
QUEUE_FORMAT_VERSION = 1
QUEUE_DIRNAME = "periodic_queue"

# The resource policy of the asynchronous path. It belongs to diagnostics and
# is independent of the synchronous path's autotuned selection in
# `periodic_diagnostic_tuning.json`, which it neither reads nor writes.
DEFAULT_ASYNC_WORKERS = 4
ASYNC_WORKER_NICENESS = 10
# Tasks not yet published, the running one included. Reaching it holds the RL
# loop at its next milestone until a task completes, so a diagnostic slower
# than training cannot grow the backlog without bound.
ASYNC_OUTSTANDING_LIMIT = 3
# Two attempts per task: one retry absorbs a transient failure, and a second
# failure stops the pipeline exactly as a failed synchronous diagnostic does.
MAX_TASK_ATTEMPTS = 2
DEFAULT_STOP_GRACE_SECONDS = 60.0
POLL_SECONDS = 1.0

PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PeriodicDiagnosticFailed(RuntimeError):
    """A queued periodic diagnostic failed on every allowed attempt."""


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def queue_dir(run_dir):
    return Path(run_dir) / "diagnostics" / QUEUE_DIRNAME


def task_path(run_dir, rl_games):
    return queue_dir(run_dir) / f"task_games_{int(rl_games):010d}.json"


def result_path(run_dir, rl_games):
    return queue_dir(run_dir) / f"result_games_{int(rl_games):010d}.json"


def log_path(run_dir, rl_games):
    return queue_dir(run_dir) / f"worker_games_{int(rl_games):010d}.log"


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _identity_key(identity):
    """The fields ``append_periodic_point`` deduplicates on, as a tuple."""
    return (
        int(identity["rl_games"]),
        identity.get("configuration_sha256"),
        int(identity["diagnostic_seed"]),
        int(identity["diagnostic_games"]),
        identity["opponent"],
    )


def build_task(
    *,
    run_dir,
    identity,
    pipeline_level,
    seed,
    rl_iterations,
    checkpoint_path,
    checkpoint_sha256,
    rl_elapsed_seconds,
    training_window,
    workers,
    safety_config,
):
    """Return one pending task record; nothing is written."""
    return {
        "format": TASK_FORMAT,
        "format_version": QUEUE_FORMAT_VERSION,
        "state": PENDING,
        "attempts": 0,
        "identity": dict(identity),
        "run_dir": str(Path(run_dir).resolve()),
        "pipeline_level": str(pipeline_level),
        "seed": int(seed),
        "rl_games": int(identity["rl_games"]),
        "rl_iterations": int(rl_iterations),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": str(checkpoint_sha256),
        "diagnostic_games": int(identity["diagnostic_games"]),
        "rl_elapsed_seconds": float(rl_elapsed_seconds),
        "training_window": dict(training_window),
        "workers": int(workers),
        "niceness": int(ASYNC_WORKER_NICENESS),
        "safety_config": {
            "memory_reserve_mb": int(safety_config.memory_reserve_mb),
            "estimated_worker_mb": int(safety_config.estimated_worker_mb),
            "max_worker_rss_mb": int(safety_config.max_worker_rss_mb),
        },
        # Time the RL loop spent waiting on the queue on this task's behalf;
        # see `PeriodicDiagnosticQueue.add_blocking_seconds`.
        "blocking_seconds": 0.0,
        "enqueued_at": _utc_now(),
        "started_at": None,
        "finished_at": None,
        "parent_pid": None,
        "worker_pid": None,
        "last_error": None,
    }


@dataclass(frozen=True)
class CompletedTask:
    """A measured point waiting for the training process to publish it."""

    task: dict
    result: dict


class PeriodicDiagnosticQueue:
    """Own the task records and the one measuring process of a run.

    ``worker_command`` exists for tests; production runs always use
    ``python -m diagnostics.periodic_worker``.
    """

    def __init__(
        self,
        run_dir,
        *,
        status_callback=None,
        worker_command=None,
        poll_seconds=POLL_SECONDS,
    ):
        self.run_dir = Path(run_dir).resolve()
        self.directory = queue_dir(self.run_dir)
        self.status = status_callback or (lambda _message: None)
        self.worker_command = worker_command or [
            sys.executable,
            "-m",
            "diagnostics.periodic_worker",
        ]
        self.poll_seconds = float(poll_seconds)
        self._process = None
        self._process_games = None
        self._log_stream = None

    # -- records ---------------------------------------------------------

    def tasks(self):
        """Return every task record, oldest checkpoint first."""
        if not self.directory.is_dir():
            return []
        records = []
        for path in sorted(self.directory.glob("task_games_*.json")):
            record = _read_json(path)
            if (
                isinstance(record, dict)
                and record.get("format") == TASK_FORMAT
                and record.get("format_version") == QUEUE_FORMAT_VERSION
            ):
                records.append(record)
            else:
                raise ValueError(f"Unreadable periodic diagnostic task: {path}.")
        return sorted(records, key=lambda record: int(record["rl_games"]))

    def _write(self, task):
        atomic_write_json(task_path(self.run_dir, task["rl_games"]), task)

    def find(self, identity):
        """Return the queued task with this identity, or ``None``."""
        key = _identity_key(identity)
        for task in self.tasks():
            if _identity_key(task["identity"]) == key:
                return task
        return None

    def outstanding(self):
        """Return how many tasks are not yet published."""
        return len(self.tasks())

    def latest_iteration_before(self, rl_games):
        """Return the largest queued iteration count below ``rl_games``."""
        return max(
            (
                int(task["rl_iterations"])
                for task in self.tasks()
                if int(task["rl_games"]) < int(rl_games)
            ),
            default=0,
        )

    def enqueue(self, task):
        """Persist one task unless an identical one is already queued."""
        existing = _read_json(task_path(self.run_dir, task["rl_games"]))
        if existing is not None:
            if _identity_key(existing["identity"]) != _identity_key(task["identity"]):
                raise ValueError(
                    "A different periodic diagnostic is already queued for "
                    f"{int(task['rl_games']):,} RL games."
                )
            return existing
        self._write(task)
        return task

    def add_blocking_seconds(self, seconds):
        """Charge time the RL loop waited on the queue to the newest task.

        That task's record then reports, as its ``diagnostic_seconds``, the
        time diagnostics actually held training back rather than their full
        duration, which ran beside training.
        """
        tasks = self.tasks()
        if not tasks or seconds <= 0.0:
            return
        newest = tasks[-1]
        newest["blocking_seconds"] = float(newest["blocking_seconds"]) + float(seconds)
        self._write(newest)

    # -- recovery ----------------------------------------------------------

    def reconcile(self, *, restored_rl_games):
        """Make the queue consistent with the checkpoint training resumed from.

        Called before any worker starts. A task for a checkpoint newer than
        the restored one belongs to a future this run abandoned, so it and any
        result are discarded rather than attached to different weights. A task
        left running by a process that no longer exists is measured again, and
        so is a task whose result never became readable.
        """
        if not self.directory.is_dir():
            return
        known = set()
        for task in self.tasks():
            games = int(task["rl_games"])
            if games > int(restored_rl_games):
                self.status(
                    "Discarding the queued periodic diagnostic at "
                    f"{games:,} RL games: training resumed from "
                    f"{int(restored_rl_games):,}."
                )
                self._remove(games, keep_log=False)
                continue
            known.add(games)
            state = task["state"]
            if state == COMPLETED and self._valid_result(task) is None:
                state = RUNNING
            if state in (RUNNING, FAILED):
                if state == FAILED:
                    self.status(
                        "Retrying the periodic diagnostic at "
                        f"{games:,} RL games that failed in an earlier "
                        f"session: {task.get('last_error')}"
                    )
                    task["attempts"] = 0
                task["state"] = PENDING
                task["worker_pid"] = None
                self._write(task)
        for path in self.directory.glob("result_games_*.json"):
            games = int(path.stem.rpartition("_")[2])
            if games not in known:
                path.unlink(missing_ok=True)

    # -- measuring process ---------------------------------------------------

    def running_games(self):
        return self._process_games

    def pump(self):
        """Advance the queue without blocking: reap, then start the next task."""
        self._reap()
        if self._process is not None:
            return
        for task in self.tasks():
            if task["state"] == PENDING:
                self._start(task)
                return
            if task["state"] != COMPLETED:
                return

    def _environment(self):
        environment = dict(os.environ)
        environment["DOMINO_FORCE_CPU"] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = ""
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(REPOSITORY_ROOT)
            if not existing
            else os.pathsep.join((str(REPOSITORY_ROOT), existing))
        )
        return environment

    def _start(self, task):
        games = int(task["rl_games"])
        result_path(self.run_dir, games).unlink(missing_ok=True)
        task["state"] = RUNNING
        task["attempts"] = int(task["attempts"]) + 1
        task["started_at"] = _utc_now()
        task["parent_pid"] = os.getpid()
        self._write(task)
        self._log_stream = open(  # pylint: disable=consider-using-with
            log_path(self.run_dir, games), "a", encoding="utf-8"
        )
        self._process = subprocess.Popen(  # pylint: disable=consider-using-with
            [*self.worker_command, str(task_path(self.run_dir, games))],
            stdin=subprocess.DEVNULL,
            stdout=self._log_stream,
            stderr=subprocess.STDOUT,
            env=self._environment(),
            cwd=str(REPOSITORY_ROOT),
        )
        self._process_games = games
        task["worker_pid"] = self._process.pid
        self._write(task)
        self.status(
            f"Periodic diagnostic at {games:,} RL games started in the "
            f"background (pid {self._process.pid}, {task['workers']} "
            f"workers, nice +{task['niceness']})."
        )

    def _valid_result(self, task):
        result = _read_json(result_path(self.run_dir, task["rl_games"]))
        if (
            not isinstance(result, dict)
            or result.get("format") != RESULT_FORMAT
            or result.get("format_version") != QUEUE_FORMAT_VERSION
            or _identity_key(result.get("identity", {})) != _identity_key(task["identity"])
            or result.get("checkpoint_sha256") != task["checkpoint_sha256"]
        ):
            return None
        return result

    def _reap(self):
        if self._process is None:
            return
        code = self._process.poll()
        if code is None:
            return
        games = self._process_games
        self._close_process()
        task = _read_json(task_path(self.run_dir, games))
        if task is None:
            return
        task["finished_at"] = _utc_now()
        task["worker_pid"] = None
        if code == 0 and self._valid_result(task) is not None:
            task["state"] = COMPLETED
            self._write(task)
            return
        log = log_path(self.run_dir, games)
        task["last_error"] = f"worker exited with code {code}; log: {log}"
        if int(task["attempts"]) < MAX_TASK_ATTEMPTS:
            task["state"] = PENDING
            self._write(task)
            self.status(
                f"Periodic diagnostic at {games:,} RL games failed "
                f"(exit code {code}); retrying. Log: {log}"
            )
            return
        task["state"] = FAILED
        self._write(task)
        raise PeriodicDiagnosticFailed(
            f"The periodic diagnostic at {games:,} RL games failed "
            f"{task['attempts']} times; the last worker exited with code "
            f"{code}. Its task stays queued for the next run. Log: {log}"
        )

    def _close_process(self):
        self._process = None
        self._process_games = None
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None

    def stop(self, grace_seconds=DEFAULT_STOP_GRACE_SECONDS):
        """Stop the measuring process, leaving its task queued for later.

        The worker turns SIGTERM into an ordinary exception, so its own game
        workers are terminated by the diagnostic runner on the way out. A
        worker that does not exit within ``grace_seconds`` is killed.
        """
        if self._process is None:
            return
        process = self._process
        games = self._process_games
        if process.poll() is None:
            self.status(
                f"Stopping the periodic diagnostic at {games:,} RL games; "
                "it stays queued for the next run."
            )
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=float(grace_seconds))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        code = process.returncode
        self._close_process()
        task = _read_json(task_path(self.run_dir, games))
        if task is None:
            return
        task["worker_pid"] = None
        task["finished_at"] = _utc_now()
        if code == 0 and self._valid_result(task) is not None:
            task["state"] = COMPLETED
        else:
            # A stop is not a failure: the attempt it interrupted is returned.
            task["state"] = PENDING
            task["attempts"] = max(0, int(task["attempts"]) - 1)
        self._write(task)

    # -- publication -----------------------------------------------------------

    def completed(self):
        """Return the measured tasks that can be published, in order.

        Only the completed prefix is returned, so a point is never published
        ahead of an older one that is still being measured.
        """
        ready = []
        for task in self.tasks():
            if task["state"] != COMPLETED:
                break
            result = self._valid_result(task)
            if result is None:
                break
            ready.append(CompletedTask(task=task, result=result))
        return ready

    def acknowledge(self, task):
        """Forget one published task; its history record is the durable copy."""
        self._remove(int(task["rl_games"]), keep_log=False)

    def _remove(self, games, *, keep_log):
        task_path(self.run_dir, games).unlink(missing_ok=True)
        result_path(self.run_dir, games).unlink(missing_ok=True)
        if not keep_log:
            log_path(self.run_dir, games).unlink(missing_ok=True)

    def wait(self, condition, *, publish, shutdown=None):
        """Block until ``condition()`` holds, measuring and publishing meanwhile.

        Returns the seconds waited. ``shutdown`` ends the wait early.
        """
        started = time.monotonic()
        while True:
            self.pump()
            publish()
            if condition() or (shutdown is not None and shutdown()):
                return time.monotonic() - started
            time.sleep(self.poll_seconds)
