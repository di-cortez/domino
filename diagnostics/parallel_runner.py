"""Deterministic, CPU-only multiprocessing for independent diagnostic games.

Jobs are small blocks pulled from a bounded dynamic queue.  The parent process
is the only process that aggregates results or writes output.  Per-game seeds
depend only on the run seed and zero-based game id, so worker count, scheduling,
fallbacks, and autotuning do not change a game's deal or random choices.
"""

from __future__ import annotations

import contextlib
import math
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from utils.resource_limits import MIB, available_ram_mb, process_rss_bytes


MAX_PARALLEL_WORKERS = 20
# Backward-compatible diagnostics name used by the public CLI and older code.
MAX_DIAGNOSTIC_WORKERS = MAX_PARALLEL_WORKERS
DEFAULT_MEMORY_RESERVE_MB = 512
DEFAULT_ESTIMATED_WORKER_MB = 256
DEFAULT_MAX_WORKER_RSS_MB = 1024
ESTIMATED_DIAGNOSTIC_RECORD_BYTES = 4096

_WORKER_AGENT = None
_WORKER_OPPONENT = None
_WORKER_SUPPRESS_OUTPUT = True


class DiagnosticMemoryPressure(RuntimeError):
    """Raised internally when live memory crosses a configured safety limit."""


class DiagnosticExecutionError(RuntimeError):
    """Unrecoverable pool failure with already completed records attached."""

    def __init__(self, message, records, run_info, cause):
        super().__init__(message)
        self.records = records
        self.run_info = run_info
        self.__cause__ = cause


@dataclass(frozen=True)
class ParallelSafetyConfig:
    """Memory and queue limits shared by CPU-only worker pools."""

    memory_reserve_mb: int = DEFAULT_MEMORY_RESERVE_MB
    estimated_worker_mb: int = DEFAULT_ESTIMATED_WORKER_MB
    max_worker_rss_mb: int = DEFAULT_MAX_WORKER_RSS_MB
    max_workers: int = MAX_PARALLEL_WORKERS
    max_in_flight_per_worker: int = 2
    target_jobs_per_worker: int = 8
    max_games_per_job: int = 32
    poll_interval_s: float = 0.05
    memory_check_interval_s: float = 0.10
    fallback_on_error: bool = True
    start_method: str = field(
        default_factory=lambda: "spawn" if "spawn" in mp.get_all_start_methods() else mp.get_start_method()
    )

    def __post_init__(self):
        if self.memory_reserve_mb < 0:
            raise ValueError("memory_reserve_mb must be non-negative")
        if self.estimated_worker_mb < 1:
            raise ValueError("estimated_worker_mb must be positive")
        if self.max_worker_rss_mb < 1:
            raise ValueError("max_worker_rss_mb must be positive")
        if not 1 <= self.max_workers <= MAX_PARALLEL_WORKERS:
            raise ValueError(
                f"max_workers must be between 1 and {MAX_PARALLEL_WORKERS}"
            )


@dataclass
class ParallelRunInfo:
    """Execution metadata retained in diagnostic and dataset summaries."""

    requested_workers: int
    initial_workers: int
    final_workers: int
    peak_worker_rss_mb: float = 0.0
    peak_total_children_rss_mb: float = 0.0
    min_available_memory_mb: float | None = None
    fallback_count: int = 0
    fallback_history: list[dict] = field(default_factory=list)
    attempted_worker_counts: list[int] = field(default_factory=list)
    safety_capped: bool = False
    memory_monitoring_available: bool = True
    workers_cpu_only: bool = True

    def to_dict(self) -> dict:
        """Return JSON-serializable worker, fallback, and memory metadata."""
        return {
            "requested_workers": self.requested_workers,
            "initial_workers": self.initial_workers,
            "final_workers": self.final_workers,
            "peak_worker_rss_mb": self.peak_worker_rss_mb,
            "peak_total_children_rss_mb": self.peak_total_children_rss_mb,
            "min_available_memory_mb": self.min_available_memory_mb,
            "fallback_count": self.fallback_count,
            "fallback_history": self.fallback_history,
            "attempted_worker_counts": self.attempted_worker_counts,
            "safety_capped": self.safety_capped,
            "memory_monitoring_available": self.memory_monitoring_available,
            "workers_cpu_only": self.workers_cpu_only,
        }


def game_seed(base_seed: int, game_index: int) -> int:
    """Return a stable SplitMix64-style seed for one absolute game id."""
    value = (int(base_seed) + 0x9E3779B97F4A7C15 * (int(game_index) + 1))
    value &= 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def _force_cpu_environment() -> None:
    """Disable GPU visibility and nested numerical-library thread pools."""
    os.environ["DOMINO_FORCE_CPU"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[name] = "1"


@contextlib.contextmanager
def cpu_only_worker_environment():
    """Set CPU-only environment before spawn imports any project module."""
    names = (
        "DOMINO_FORCE_CPU",
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    previous = {name: os.environ.get(name) for name in names}
    _force_cpu_environment()
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _worker_initializer(
    agent_name: str,
    opponent_name: str,
    weights: str | None,
    opponent_weights: str | None,
    suppress_agent_output: bool,
) -> None:
    """Construct one reusable agent pair inside each diagnostic worker."""
    global _WORKER_AGENT, _WORKER_OPPONENT, _WORKER_SUPPRESS_OUTPUT
    _force_cpu_environment()
    from diagnostics.pairwise import create_agent

    _WORKER_AGENT = create_agent(agent_name, weights)
    _WORKER_OPPONENT = create_agent(opponent_name, opponent_weights)
    _WORKER_SUPPRESS_OUTPUT = suppress_agent_output


def _worker_play_games(jobs: tuple[tuple[int, int], ...]) -> list[dict]:
    """Play one scheduled game block with stable absolute game seeds."""
    from diagnostics.pairwise import play_game

    records = []
    for game_index, seed in jobs:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        record = play_game(
            _WORKER_AGENT,
            _WORKER_OPPONENT,
            agent_position=game_index % 2,
            suppress_agent_output=_WORKER_SUPPRESS_OUTPUT,
        )
        record["game"] = game_index + 1
        record["game_seed"] = int(seed)
        records.append(record)
    return records


def cap_parallel_workers(
    requested_workers: int,
    safety: ParallelSafetyConfig,
) -> tuple[int, bool, str | None]:
    """Cap workers by the hard limit, CPU count, and current RAM headroom."""
    requested = max(1, int(requested_workers))
    cpu_limit = max(1, os.cpu_count() or 1)
    capped = min(requested, safety.max_workers, MAX_PARALLEL_WORKERS, cpu_limit)
    reasons = []
    if requested > MAX_PARALLEL_WORKERS:
        reasons.append(f"hard limit is {MAX_PARALLEL_WORKERS}")
    if requested > safety.max_workers:
        reasons.append(f"configured limit is {safety.max_workers}")
    if requested > cpu_limit:
        reasons.append(f"only {cpu_limit} logical CPUs detected")

    available_mb = available_ram_mb()
    if available_mb is not None:
        usable_mb = max(0.0, available_mb - safety.memory_reserve_mb)
        memory_limit = max(1, int(usable_mb // safety.estimated_worker_mb))
        if capped > memory_limit:
            reasons.append(
                f"RAM preflight allows {memory_limit} worker(s): {available_mb:.1f} MiB "
                f"available, {safety.memory_reserve_mb} MiB reserved"
            )
        capped = min(capped, memory_limit)

    was_capped = capped < requested
    return max(1, capped), was_capped, "; ".join(reasons) or None


# Compatibility alias retained for existing diagnostics imports.
safety_cap_workers = cap_parallel_workers


def executor_memory_snapshot(
    executor: ProcessPoolExecutor,
) -> tuple[float, float, float | None]:
    """Return peak child RSS, total child RSS, and available host RAM in MiB."""
    rss_values = []
    processes = getattr(executor, "_processes", {}) or {}
    for process in processes.values():
        rss = process_rss_bytes(process.pid)
        if rss is not None:
            rss_values.append(rss / MIB)
    return max(rss_values, default=0.0), sum(rss_values), available_ram_mb()


def terminate_executor(executor: ProcessPoolExecutor) -> None:
    """Terminate worker children after a failed pool without waiting forever."""
    processes = list((getattr(executor, "_processes", {}) or {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=1.0)
    executor.shutdown(wait=False, cancel_futures=True)


def _chunk_pending_jobs(
    pending_specs: list[tuple[int, int]],
    worker_count: int,
    safety: ParallelSafetyConfig,
) -> list[tuple[tuple[int, int], ...]]:
    """Split pending games into small blocks for dynamic load balancing."""
    target_jobs = max(1, worker_count * safety.target_jobs_per_worker)
    chunk_size = max(1, math.ceil(len(pending_specs) / target_jobs))
    chunk_size = min(chunk_size, safety.max_games_per_job)
    return [
        tuple(pending_specs[index:index + chunk_size])
        for index in range(0, len(pending_specs), chunk_size)
    ]


def _run_pending_jobs(
    queued_jobs: list[tuple[tuple[int, int], ...]],
    worker_count: int,
    worker_args: tuple,
    safety: ParallelSafetyConfig,
    on_record: Callable[[dict], None],
    run_info: ParallelRunInfo,
) -> None:
    """Run a bounded job queue while enforcing live process-memory limits."""
    if not queued_jobs:
        return

    executor = None
    try:
        context = mp.get_context(safety.start_method)
        with cpu_only_worker_environment():
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
                initializer=_worker_initializer,
                initargs=worker_args,
            )
            max_in_flight = max(
                worker_count,
                worker_count * safety.max_in_flight_per_worker,
            )
            jobs_iter = iter(queued_jobs)
            in_flight = {}

            def submit_next() -> bool:
                try:
                    job = next(jobs_iter)
                except StopIteration:
                    return False
                future = executor.submit(_worker_play_games, job)
                in_flight[future] = job
                return True

            for _ in range(min(max_in_flight, len(queued_jobs))):
                submit_next()

            last_memory_check = float("-inf")
            while in_flight:
                done, _ = wait(
                    in_flight,
                    timeout=safety.poll_interval_s,
                    return_when=FIRST_COMPLETED,
                )
                completed_jobs = 0
                first_job_error = None
                for future in done:
                    in_flight.pop(future, None)
                    try:
                        job_records = future.result()
                    except Exception as exc:
                        if first_job_error is None:
                            first_job_error = exc
                        continue
                    for record in job_records:
                        on_record(record)
                    completed_jobs += 1
                # Preserve every successful future from the same completed set
                # before reacting to one failed sibling job.
                if first_job_error is not None:
                    raise first_job_error

                now = time.monotonic()
                if now - last_memory_check >= safety.memory_check_interval_s:
                    peak_worker, total_children, available_mb = executor_memory_snapshot(
                        executor
                    )
                    run_info.peak_worker_rss_mb = max(
                        run_info.peak_worker_rss_mb,
                        peak_worker,
                    )
                    run_info.peak_total_children_rss_mb = max(
                        run_info.peak_total_children_rss_mb,
                        total_children,
                    )
                    if available_mb is None:
                        run_info.memory_monitoring_available = False
                    else:
                        if run_info.min_available_memory_mb is None:
                            run_info.min_available_memory_mb = available_mb
                        else:
                            run_info.min_available_memory_mb = min(
                                run_info.min_available_memory_mb,
                                available_mb,
                            )
                        if available_mb < safety.memory_reserve_mb:
                            raise DiagnosticMemoryPressure(
                                f"available RAM fell to {available_mb:.1f} MiB, below "
                                f"the {safety.memory_reserve_mb} MiB reserve"
                            )
                    if peak_worker > safety.max_worker_rss_mb:
                        raise DiagnosticMemoryPressure(
                            f"one worker reached {peak_worker:.1f} MiB RSS, above "
                            f"the {safety.max_worker_rss_mb} MiB limit"
                        )
                    last_memory_check = now

                for _ in range(completed_jobs):
                    submit_next()

            executor.shutdown(wait=True)
            executor = None
    except BaseException:
        if executor is not None:
            terminate_executor(executor)
        raise


def evaluate_game_specs(
    *,
    agent_name: str,
    opponent_name: str,
    game_specs: Iterable[tuple[int, int]],
    weights: str | Path | None,
    opponent_weights: str | Path | None,
    requested_workers: int,
    suppress_agent_output: bool,
    progress_callback: Callable[[int, int], None] | None = None,
    safety: ParallelSafetyConfig | None = None,
) -> tuple[list[dict], ParallelRunInfo]:
    """Execute arbitrary absolute game ids, retaining work across pool fallbacks."""
    safety = safety or ParallelSafetyConfig()
    specs = sorted((int(index), int(seed)) for index, seed in game_specs)
    if not specs:
        run_info = ParallelRunInfo(
            requested_workers=requested_workers,
            initial_workers=1,
            final_workers=1,
        )
        return [], run_info
    if len({index for index, _seed in specs}) != len(specs):
        raise ValueError("game_specs contains duplicate game ids")

    worker_count, was_capped, cap_reason = cap_parallel_workers(
        requested_workers,
        safety,
    )
    run_info = ParallelRunInfo(
        requested_workers=int(requested_workers),
        initial_workers=worker_count,
        final_workers=worker_count,
        safety_capped=was_capped,
    )
    if cap_reason:
        run_info.fallback_history.append({
            "from_workers": int(requested_workers),
            "to_workers": worker_count,
            "completed_games": 0,
            "reason": cap_reason,
            "phase": "preflight",
        })

    records_by_index = {}
    worker_args = (
        agent_name,
        opponent_name,
        str(weights) if weights is not None else None,
        str(opponent_weights) if opponent_weights is not None else None,
        suppress_agent_output,
    )

    def store(record: dict) -> None:
        game_index = int(record["game"]) - 1
        if game_index not in records_by_index:
            records_by_index[game_index] = record
            if progress_callback is not None:
                progress_callback(len(records_by_index), len(specs))

    recoverable = (MemoryError, DiagnosticMemoryPressure, BrokenProcessPool, OSError)
    while len(records_by_index) < len(specs):
        pending = [spec for spec in specs if spec[0] not in records_by_index]
        run_info.attempted_worker_counts.append(worker_count)
        try:
            _run_pending_jobs(
                _chunk_pending_jobs(pending, worker_count, safety),
                worker_count,
                worker_args,
                safety,
                store,
                run_info,
            )
        except recoverable as exc:
            if not safety.fallback_on_error or worker_count <= 1:
                records = [records_by_index[index] for index in sorted(records_by_index)]
                raise DiagnosticExecutionError(
                    f"diagnostics failed with {worker_count} worker(s): {exc}",
                    records,
                    run_info,
                    exc,
                ) from exc
            next_workers = max(1, worker_count // 2)
            run_info.fallback_count += 1
            run_info.fallback_history.append({
                "from_workers": worker_count,
                "to_workers": next_workers,
                "completed_games": len(records_by_index),
                "reason": f"{type(exc).__name__}: {exc}",
                "phase": "runtime",
            })
            worker_count = next_workers
            run_info.final_workers = worker_count
        except Exception as exc:
            records = [records_by_index[index] for index in sorted(records_by_index)]
            raise DiagnosticExecutionError(
                f"diagnostics failed with {worker_count} worker(s): "
                f"{type(exc).__name__}: {exc}",
                records,
                run_info,
                exc,
            ) from exc

    return [records_by_index[index] for index, _seed in specs], run_info
