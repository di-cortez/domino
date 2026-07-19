"""Low-overhead pipeline timing for exact opponent-model updates.

The pipeline runs work in a parent process and several short-lived CPU worker
processes.  Wall times from those workers overlap, so they cannot be subtracted
from the pipeline wall clock.  This module therefore uses aggregate process CPU
time for the exact-versus-rest split and records stage wall time separately.

Worker processes inherit a private run directory and stage name through the
environment.  A worker that calls ``ExactOpponentModel.update`` writes one
small timing shard when it exits.  The parent combines those shards with its
own counters after each worker pool has shut down.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from multiprocessing.util import Finalize
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - Linux provides resource.
    resource = None


RUN_DIRECTORY_ENV = "DOMINO_EXACT_UPDATE_TIMING_DIR"
STAGE_ENV = "DOMINO_EXACT_UPDATE_TIMING_STAGE"
REPORT_SCHEMA_VERSION = 1


@dataclass
class _ProcessProfile:
    """Process-local counters flushed to one unique JSON shard."""

    pid: int
    stage: str
    run_directory: Path
    token: str
    calls: int = 0
    cpu_ns: int = 0
    wall_ns: int = 0
    finalizer: Finalize | None = None

    @property
    def shard_path(self) -> Path:
        return self.run_directory / "shards" / (
            f"{self.stage}.{self.pid}.{self.token}.json"
        )


@dataclass(frozen=True)
class _StageToken:
    """Parent-process baseline used to close one pipeline stage."""

    name: str
    started_at: str
    wall_start_ns: int
    parent_cpu_start_ns: int
    children_cpu_start_s: float | None


_PROFILE: _ProcessProfile | None = None
_PROFILE_LOCK = threading.Lock()
_ACTIVE_RECORDER: "PipelineTimingRecorder | None" = None


def _now_iso() -> str:
    """Return a timezone-aware local timestamp suitable for JSON reports."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _children_cpu_seconds() -> float | None:
    """Return cumulative CPU used by reaped children of the current process."""
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return float(usage.ru_utime + usage.ru_stime)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace one JSON file without exposing a partial document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _profile_environment() -> tuple[Path, str] | None:
    """Return the inherited run directory and stage, if timing is active."""
    run_directory = os.environ.get(RUN_DIRECTORY_ENV)
    stage = os.environ.get(STAGE_ENV)
    if not run_directory or not stage:
        return None
    return Path(run_directory), stage


def _flush_profile(expected_pid: int | None = None, expected_token: str | None = None) -> None:
    """Write the current process counters when they match the caller's token."""
    profile = _PROFILE
    if profile is None:
        return
    if expected_pid is not None and profile.pid != expected_pid:
        return
    if expected_token is not None and profile.token != expected_token:
        return

    payload = {
        "pid": profile.pid,
        "stage": profile.stage,
        "exact_update_calls": profile.calls,
        "exact_update_cpu_ns": profile.cpu_ns,
        "exact_update_wall_ns": profile.wall_ns,
    }
    _atomic_write_json(profile.shard_path, payload)


def _finish_process_profile() -> None:
    """Flush and detach the current parent-process profile, if one exists."""
    global _PROFILE
    profile = _PROFILE
    if profile is None or profile.pid != os.getpid():
        _PROFILE = None
        return
    _flush_profile(profile.pid, profile.token)
    if profile.finalizer is not None:
        profile.finalizer.cancel()
    _PROFILE = None


def _ensure_process_profile() -> _ProcessProfile | None:
    """Lazily create counters in any process that performs an exact update."""
    global _PROFILE
    environment = _profile_environment()
    if environment is None:
        return None

    run_directory, stage = environment
    pid = os.getpid()
    profile = _PROFILE
    if (
        profile is not None
        and profile.pid == pid
        and profile.stage == stage
        and profile.run_directory == run_directory
    ):
        return profile

    # A fork can inherit the parent's Python globals. Never flush inherited
    # counters from the child, and start a fresh shard for the child's PID.
    if profile is not None and profile.pid == pid:
        _finish_process_profile()

    token = uuid.uuid4().hex
    profile = _ProcessProfile(
        pid=pid,
        stage=stage,
        run_directory=run_directory,
        token=token,
    )
    profile.finalizer = Finalize(
        None,
        _flush_profile,
        args=(pid, token),
        exitpriority=20,
    )
    _PROFILE = profile
    return profile


def begin_exact_update_timing() -> tuple[_ProcessProfile, int, int] | None:
    """Start timing one ``ExactOpponentModel.update`` call when enabled."""
    profile = _ensure_process_profile()
    if profile is None:
        return None
    return profile, time.process_time_ns(), time.perf_counter_ns()


def end_exact_update_timing(
    timing_token: tuple[_ProcessProfile, int, int] | None,
) -> None:
    """Add one completed or failed update call to process-local counters."""
    if timing_token is None:
        return
    profile, cpu_start_ns, wall_start_ns = timing_token
    cpu_elapsed_ns = time.process_time_ns() - cpu_start_ns
    wall_elapsed_ns = time.perf_counter_ns() - wall_start_ns
    with _PROFILE_LOCK:
        profile.calls += 1
        profile.cpu_ns += max(0, cpu_elapsed_ns)
        profile.wall_ns += max(0, wall_elapsed_ns)


class PipelineTimingRecorder:
    """Collect and persist exact-versus-rest timing for one pipeline run."""

    def __init__(self, report_path: str | Path, metadata: dict[str, Any]):
        self.report_path = Path(report_path).resolve()
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = (
            datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
            + f"-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        self.run_directory = Path(tempfile.mkdtemp(
            prefix=f".exact-update-timing-{self.run_id}-",
            dir=self.report_path.parent,
        ))
        (self.run_directory / "shards").mkdir()
        self.run = {
            "run_id": self.run_id,
            "status": "running",
            "started_at": _now_iso(),
            "finished_at": None,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "metadata": dict(metadata),
            "pipeline_elapsed_wall_seconds": None,
            "stages": {},
            "totals": None,
        }
        self._wall_start_ns = time.perf_counter_ns()
        self._active_stage: _StageToken | None = None
        self._closed = False
        os.environ[RUN_DIRECTORY_ENV] = str(self.run_directory)
        os.environ.pop(STAGE_ENV, None)

    def begin_stage(self, name: str) -> _StageToken:
        """Start one non-overlapping pipeline-stage measurement."""
        if self._closed:
            raise RuntimeError("The pipeline timing recorder is already closed.")
        if self._active_stage is not None:
            raise RuntimeError(
                f"Timing stage {self._active_stage.name!r} is still active."
            )
        os.environ[STAGE_ENV] = name
        token = _StageToken(
            name=name,
            started_at=_now_iso(),
            wall_start_ns=time.perf_counter_ns(),
            parent_cpu_start_ns=time.process_time_ns(),
            children_cpu_start_s=_children_cpu_seconds(),
        )
        self._active_stage = token
        return token

    def end_stage(
        self,
        token: _StageToken,
        *,
        status: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Close one stage after all of its child worker pools have exited."""
        if token is not self._active_stage:
            raise RuntimeError("Attempted to close a pipeline timing stage out of order.")

        _finish_process_profile()
        wall_seconds = (time.perf_counter_ns() - token.wall_start_ns) / 1e9
        parent_cpu_seconds = (
            time.process_time_ns() - token.parent_cpu_start_ns
        ) / 1e9
        children_cpu_end = _children_cpu_seconds()
        if token.children_cpu_start_s is None or children_cpu_end is None:
            children_cpu_seconds = None
            aggregate_cpu_seconds = parent_cpu_seconds
        else:
            children_cpu_seconds = max(
                0.0,
                children_cpu_end - token.children_cpu_start_s,
            )
            aggregate_cpu_seconds = parent_cpu_seconds + children_cpu_seconds

        exact_calls = 0
        exact_cpu_ns = 0
        exact_wall_ns = 0
        exact_processes: set[int] = set()
        for shard_path in sorted((self.run_directory / "shards").glob("*.json")):
            try:
                shard = json.loads(shard_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if shard.get("stage") != token.name:
                continue
            exact_calls += int(shard.get("exact_update_calls", 0))
            exact_cpu_ns += int(shard.get("exact_update_cpu_ns", 0))
            exact_wall_ns += int(shard.get("exact_update_wall_ns", 0))
            exact_processes.add(int(shard.get("pid", 0)))

        exact_cpu_seconds = exact_cpu_ns / 1e9
        exact_wall_seconds = exact_wall_ns / 1e9
        other_cpu_seconds = max(0.0, aggregate_cpu_seconds - exact_cpu_seconds)
        if aggregate_cpu_seconds > 0.0:
            exact_share = 100.0 * exact_cpu_seconds / aggregate_cpu_seconds
            other_share = 100.0 * other_cpu_seconds / aggregate_cpu_seconds
        else:
            exact_share = 0.0
            other_share = 0.0

        stage_result = {
            "status": status,
            "error": error,
            "started_at": token.started_at,
            "finished_at": _now_iso(),
            "elapsed_wall_seconds": wall_seconds,
            "cpu_measurement_includes_reaped_children": children_cpu_seconds is not None,
            "parent_process_cpu_seconds": parent_cpu_seconds,
            "child_processes_cpu_seconds": children_cpu_seconds,
            "aggregate_process_cpu_seconds": aggregate_cpu_seconds,
            "exact_opponent_model_update": {
                "calls": exact_calls,
                "processes_with_calls": len(exact_processes),
                "cpu_seconds": exact_cpu_seconds,
                "aggregate_call_wall_seconds": exact_wall_seconds,
                "share_of_aggregate_process_cpu_percent": exact_share,
                "mean_cpu_milliseconds_per_call": (
                    1000.0 * exact_cpu_seconds / exact_calls
                    if exact_calls else 0.0
                ),
            },
            "everything_else": {
                "cpu_seconds": other_cpu_seconds,
                "share_of_aggregate_process_cpu_percent": other_share,
            },
        }
        if exact_cpu_seconds > aggregate_cpu_seconds + 0.001:
            stage_result["measurement_warning"] = (
                "Exact-update CPU exceeded the aggregate stage CPU baseline; "
                "one or more worker processes may not have been reaped before "
                "the stage ended."
            )

        self.run["stages"][token.name] = stage_result
        self._active_stage = None
        os.environ.pop(STAGE_ENV, None)
        return stage_result

    def finish(self, *, status: str, error: str | None = None) -> None:
        """Append this run to the stable report and remove timing shards."""
        if self._closed:
            return
        if self._active_stage is not None:
            self.end_stage(self._active_stage, status="failed", error=error)

        self.run["status"] = status
        self.run["error"] = error
        self.run["finished_at"] = _now_iso()
        self.run["pipeline_elapsed_wall_seconds"] = (
            time.perf_counter_ns() - self._wall_start_ns
        ) / 1e9

        stages = list(self.run["stages"].values())
        aggregate_cpu = sum(
            float(stage["aggregate_process_cpu_seconds"])
            for stage in stages
        )
        exact_cpu = sum(
            float(stage["exact_opponent_model_update"]["cpu_seconds"])
            for stage in stages
        )
        other_cpu = max(0.0, aggregate_cpu - exact_cpu)
        self.run["totals"] = {
            "aggregate_process_cpu_seconds": aggregate_cpu,
            "exact_opponent_model_update_cpu_seconds": exact_cpu,
            "everything_else_cpu_seconds": other_cpu,
            "exact_opponent_model_update_share_percent": (
                100.0 * exact_cpu / aggregate_cpu if aggregate_cpu else 0.0
            ),
        }

        document: dict[str, Any]
        if self.report_path.exists():
            document = json.loads(self.report_path.read_text(encoding="utf-8"))
            if document.get("schema_version") != REPORT_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported timing report schema in {self.report_path}."
                )
            runs = document.get("runs")
            if not isinstance(runs, list):
                raise ValueError(f"Invalid timing report in {self.report_path}.")
        else:
            document = {
                "schema_version": REPORT_SCHEMA_VERSION,
                "report": "ExactOpponentModel.update pipeline timing",
                "timing_basis": {
                    "elapsed_wall_seconds": (
                        "Observed parent wall-clock duration of the stage."
                    ),
                    "aggregate_process_cpu_seconds": (
                        "Parent CPU plus CPU of reaped child workers. Parallel "
                        "worker CPU times are summed."
                    ),
                    "exact_vs_everything_else": (
                        "A non-overlapping split of aggregate process CPU. The "
                        "aggregate wall time of exact calls is informational and "
                        "must not be subtracted from stage wall time because worker "
                        "calls overlap."
                    ),
                },
                "runs": [],
            }
        document["runs"].append(self.run)
        _atomic_write_json(self.report_path, document)

        self._closed = True
        os.environ.pop(STAGE_ENV, None)
        os.environ.pop(RUN_DIRECTORY_ENV, None)
        shutil.rmtree(self.run_directory, ignore_errors=True)


def start_pipeline_timing(
    report_path: str | Path,
    metadata: dict[str, Any],
) -> PipelineTimingRecorder:
    """Create and register the recorder used by pipeline stage wrappers."""
    global _ACTIVE_RECORDER
    if _ACTIVE_RECORDER is not None and not _ACTIVE_RECORDER._closed:
        raise RuntimeError("A pipeline timing recorder is already active.")
    _ACTIVE_RECORDER = PipelineTimingRecorder(report_path, metadata)
    return _ACTIVE_RECORDER


def begin_pipeline_stage(name: str) -> _StageToken | None:
    """Start a stage when the pipeline recorder is active."""
    if _ACTIVE_RECORDER is None:
        return None
    return _ACTIVE_RECORDER.begin_stage(name)


def end_pipeline_stage(
    token: _StageToken | None,
    *,
    status: str,
    error: str | None = None,
) -> dict[str, Any] | None:
    """End a stage when the pipeline recorder is active."""
    if token is None or _ACTIVE_RECORDER is None:
        return None
    return _ACTIVE_RECORDER.end_stage(token, status=status, error=error)


def finish_pipeline_timing(*, status: str, error: str | None = None) -> None:
    """Persist and unregister the active pipeline recorder."""
    global _ACTIVE_RECORDER
    recorder = _ACTIVE_RECORDER
    if recorder is None:
        return
    try:
        recorder.finish(status=status, error=error)
    finally:
        _ACTIVE_RECORDER = None


def exception_summary(error: BaseException) -> str:
    """Return a compact JSON-safe exception description without a traceback."""
    message = str(error)
    name = type(error).__name__
    return f"{name}: {message}" if message else name
