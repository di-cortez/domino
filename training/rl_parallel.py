"""Deterministic CPU multiprocessing for reinforcement-learning rollouts.

Every game in one RL iteration observes the same frozen learner policy and
opponent-pool snapshots. Policy weights live in a bounded shared-memory bank,
so workers attach to read-only NumPy views instead of receiving a full network
copy with every job. Workers return finalized trajectories; the parent process
orders them by game id, assembles the batch, and remains solely responsible for
gradient updates, checkpoints, logging, and GPU use.

Per-game seeds depend only on the run seed and absolute game id. Scheduling,
worker count, autotuning, and memory fallback therefore do not change rollout
contents. Adaptive tuning uses separate labeled seed streams and discards all
benchmark trajectories before the real training-game counter starts.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from multiprocessing import shared_memory
import numpy as np

from diagnostics.parallel_runner import (
    MAX_PARALLEL_WORKERS,
    DiagnosticMemoryPressure,
    ParallelRunInfo,
    ParallelSafetyConfig,
    cap_parallel_workers,
    cpu_only_worker_environment,
    executor_memory_snapshot,
    game_seed,
    terminate_executor,
)
from utils.runtime_status import format_duration


DEFAULT_RL_WORKERS = "auto"
DEFAULT_RL_AUTOTUNE_FRACTION = 0.01
DEFAULT_RL_MINIMUM_GAIN = 0.10
DEFAULT_RL_WORKER_CANDIDATES = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
POLICY_WEIGHT_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")


class RLRolloutExecutionError(RuntimeError):
    """Unrecoverable rollout-pool error with completed results attached."""

    def __init__(self, message, results, run_info, cause):
        super().__init__(message)
        self.results = results
        self.run_info = run_info
        self.__cause__ = cause


@dataclass(frozen=True)
class SharedPolicyDescriptor:
    """Pickle-friendly description of one policy stored in shared memory."""

    name: str
    shapes: tuple[tuple[int, ...], ...]
    element_count: int
    dtype: str


class SharedPolicyBank:
    """Own the current policy and a fixed-size ring of opponent snapshots."""

    def __init__(self, network, max_pool_size):
        if max_pool_size < 0:
            raise ValueError("max_pool_size must be non-negative")
        self.shapes = tuple(
            tuple(int(value) for value in getattr(network, name).shape)
            for name in POLICY_WEIGHT_NAMES
        )
        first_weight = getattr(network, POLICY_WEIGHT_NAMES[0])
        if hasattr(first_weight, "get"):
            first_weight = first_weight.get()
        self.dtype = np.asarray(first_weight).dtype
        self.element_count = sum(math.prod(shape) for shape in self.shapes)
        self._segments = []
        self._descriptors = []
        self._closed = False
        try:
            for _slot in range(1 + max_pool_size):
                segment = shared_memory.SharedMemory(
                    create=True,
                    size=self.element_count * self.dtype.itemsize,
                )
                self._segments.append(segment)
                self._descriptors.append(SharedPolicyDescriptor(
                    name=segment.name,
                    shapes=self.shapes,
                    element_count=self.element_count,
                    dtype=self.dtype.str,
                ))
        except BaseException:
            self.close()
            raise

        self.max_pool_size = int(max_pool_size)
        self.pool_slots = []
        self._next_pool_slot = 0
        self.write_current(network)

    @property
    def current_descriptor(self):
        return self._descriptors[0]

    @property
    def pool_descriptors(self):
        return tuple(self._descriptors[1:])

    @property
    def allocated_bytes(self):
        return len(self._segments) * self.element_count * self.dtype.itemsize

    def _write_slot(self, slot_index, network):
        flat = np.ndarray(
            (self.element_count,),
            dtype=self.dtype,
            buffer=self._segments[slot_index].buf,
        )
        offset = 0
        for name, shape in zip(POLICY_WEIGHT_NAMES, self.shapes):
            value = getattr(network, name)
            if hasattr(value, "get"):
                value = value.get()
            value = np.asarray(value, dtype=self.dtype)
            if value.shape != shape:
                raise ValueError(
                    f"Policy weight {name} changed shape from {shape} to {value.shape}."
                )
            size = value.size
            np.copyto(flat[offset:offset + size].reshape(shape), value)
            offset += size

    def write_current(self, network):
        """Publish the learner policy after the previous gradient update."""
        self._write_slot(0, network)

    def append_pool_snapshot(self, network):
        """Append a frozen opponent snapshot, overwriting the oldest ring slot."""
        if self.max_pool_size == 0:
            return
        slot = self._next_pool_slot
        self._write_slot(slot + 1, network)
        if slot in self.pool_slots:
            self.pool_slots.remove(slot)
        self.pool_slots.append(slot)
        self._next_pool_slot = (slot + 1) % self.max_pool_size

    def export_pool_snapshots(self):
        """Copy opponent snapshots in their logical oldest-to-newest order."""
        snapshots = []
        for slot in self.pool_slots:
            flat = np.ndarray(
                (self.element_count,),
                dtype=self.dtype,
                buffer=self._segments[slot + 1].buf,
            )
            weights = {}
            offset = 0
            for name, shape in zip(POLICY_WEIGHT_NAMES, self.shapes):
                size = math.prod(shape)
                weights[name] = flat[offset:offset + size].reshape(shape).copy()
                offset += size
            snapshots.append(weights)
        return tuple(snapshots)

    def restore_pool_snapshots(self, snapshots):
        """Replace the ring with serialized snapshots from a resume state."""
        snapshots = tuple(snapshots)
        if len(snapshots) > self.max_pool_size:
            raise ValueError(
                f"Resume state contains {len(snapshots)} opponent snapshots, "
                f"but max_pool_size is {self.max_pool_size}."
            )
        self.pool_slots = []
        self._next_pool_slot = 0
        for weights in snapshots:
            missing = [name for name in POLICY_WEIGHT_NAMES if name not in weights]
            if missing:
                raise ValueError(
                    "Resume opponent snapshot is missing policy weights: "
                    + ", ".join(missing)
                )
            self.append_pool_snapshot(_CPUInferencePolicy(weights))

    def close(self):
        """Release every shared segment, even after a failed training run."""
        if self._closed:
            return
        self._closed = True
        for segment in self._segments:
            try:
                segment.close()
            finally:
                try:
                    segment.unlink()
                except FileNotFoundError:
                    pass


class _CPUInferencePolicy:
    """Minimal NumPy policy wrapper backed directly by shared arrays."""

    xp = np
    device = "cpu"

    def __init__(self, weights):
        for name, value in weights.items():
            setattr(self, name, value)

    def forward(self, x):
        x = np.asarray(x)
        z1 = np.dot(self.W1, x) + self.b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(self.W2, a1) + self.b2
        a2 = np.maximum(0, z2)
        z3 = np.dot(self.W3, a2) + self.b3
        exp_z = np.exp(z3 - np.max(z3, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)


_WORKER_SHARED_HANDLES = []
_WORKER_CURRENT_POLICY = None
_WORKER_POOL_POLICIES = ()
_WORKER_TRAINING_OPPONENT = None
_WORKER_SCHEMA = None
_WORKER_GAMMA = None


def _attach_policy(descriptor):
    """Attach one worker-side inference policy to a shared-memory segment."""
    segment = shared_memory.SharedMemory(name=descriptor.name)
    flat = np.ndarray(
        (descriptor.element_count,),
        dtype=np.dtype(descriptor.dtype),
        buffer=segment.buf,
    )
    weights = {}
    offset = 0
    for name, shape in zip(POLICY_WEIGHT_NAMES, descriptor.shapes):
        size = math.prod(shape)
        weights[name] = flat[offset:offset + size].reshape(shape)
        offset += size
    return _CPUInferencePolicy(weights), segment


def _worker_initializer(
    current_descriptor,
    pool_descriptors,
    training_opponent,
    schema,
    gamma,
):
    """Attach every reusable policy view inside one CPU-only worker."""
    global _WORKER_SHARED_HANDLES
    global _WORKER_CURRENT_POLICY, _WORKER_POOL_POLICIES
    global _WORKER_TRAINING_OPPONENT, _WORKER_SCHEMA, _WORKER_GAMMA

    # The environment is already set before spawn; repeating it here protects
    # direct initializer use and documents the invariant at the worker boundary.
    os.environ["DOMINO_FORCE_CPU"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    current, current_handle = _attach_policy(current_descriptor)
    pool_policies = []
    handles = [current_handle]
    for descriptor in pool_descriptors:
        policy, handle = _attach_policy(descriptor)
        pool_policies.append(policy)
        handles.append(handle)

    _WORKER_SHARED_HANDLES = handles
    _WORKER_CURRENT_POLICY = current
    _WORKER_POOL_POLICIES = tuple(pool_policies)
    _WORKER_TRAINING_OPPONENT = training_opponent
    _WORKER_SCHEMA = dict(schema)
    _WORKER_GAMMA = float(gamma)


def _event_stats_dict(event_stats):
    """Return the four compact event counters needed by the parent process."""
    return {
        "opponent_draws": int(event_stats.opponent_draws),
        "opponent_passes": int(event_stats.opponent_passes),
        "learner_draws": int(event_stats.learner_draws),
        "learner_passes": int(event_stats.learner_passes),
    }


def _worker_collect_rollouts(job):
    """Play one dynamic block of seeded training games."""
    game_specs, pool_slots = job
    from training.self_play import (
        _collect_self_play_steps,
        _collect_steps_vs_heuristic,
    )

    results = []
    pool = [_WORKER_POOL_POLICIES[index] for index in pool_slots]
    for game_index, seed in game_specs:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        if _WORKER_TRAINING_OPPONENT == "self_play":
            samples, events, winner, learner_position = _collect_self_play_steps(
                _WORKER_CURRENT_POLICY,
                pool,
                _WORKER_SCHEMA,
                _WORKER_GAMMA,
            )
        else:
            samples, events, winner, learner_position = _collect_steps_vs_heuristic(
                _WORKER_CURRENT_POLICY,
                _WORKER_SCHEMA,
                _WORKER_GAMMA,
            )
        results.append({
            "game_index": int(game_index),
            "game_seed": int(seed),
            "samples": samples,
            "event_stats": _event_stats_dict(events),
            "winner": int(winner),
            "learner_position": int(learner_position),
        })
    return results


def _worker_ready():
    """Confirm that a spawned worker initialized without GPU visibility."""
    return {
        "force_cpu": os.environ.get("DOMINO_FORCE_CPU"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "array_backend": _WORKER_CURRENT_POLICY.xp.__name__,
    }


def _chunk_specs(specs, worker_count, safety):
    """Split games into enough bounded blocks for dynamic load balancing."""
    target_jobs = max(1, worker_count * safety.target_jobs_per_worker)
    chunk_size = max(1, math.ceil(len(specs) / target_jobs))
    chunk_size = min(chunk_size, safety.max_games_per_job)
    return [
        tuple(specs[index:index + chunk_size])
        for index in range(0, len(specs), chunk_size)
    ]


class RLRolloutRunner:
    """Manage shared policies and a persistent, memory-monitored worker pool."""

    def __init__(
        self,
        network,
        *,
        training_opponent,
        schema,
        gamma,
        max_pool_size,
        safety=None,
    ):
        self.safety = safety or ParallelSafetyConfig()
        self.training_opponent = training_opponent
        self.schema = dict(schema)
        self.gamma = float(gamma)
        self.bank = SharedPolicyBank(network, max_pool_size)
        if training_opponent == "self_play":
            self.bank.append_pool_snapshot(network)
        self.executor = None
        self.requested_workers = 1
        self.worker_count = 1
        self._cap_reason = None
        self._safety_capped = False
        self._closed = False

    @property
    def allocated_shared_bytes(self):
        return self.bank.allocated_bytes

    def sync_current(self, network):
        """Publish weights only while no worker task is in flight."""
        self.bank.write_current(network)

    def append_pool_snapshot(self, network):
        """Publish a new frozen opponent after the iteration update."""
        self.bank.append_pool_snapshot(network)

    def export_pool_snapshots(self):
        """Return copies suitable for an atomic parent-side resume file."""
        return self.bank.export_pool_snapshots()

    def restore_pool_snapshots(self, snapshots):
        """Restore the exact opponent pool before starting resumed rollouts."""
        self._shutdown_executor()
        self.bank.restore_pool_snapshots(snapshots)

    def _shutdown_executor(self, terminate=False):
        if self.executor is None:
            return
        if terminate:
            terminate_executor(self.executor)
        else:
            self.executor.shutdown(wait=True, cancel_futures=False)
        self.executor = None

    def set_workers(self, requested_workers):
        """Select a capped pool size and restart workers only when it changes."""
        capped, was_capped, reason = cap_parallel_workers(
            requested_workers,
            self.safety,
        )
        requested_workers = int(requested_workers)
        if (
            self.executor is not None
            and capped == self.worker_count
            and requested_workers == self.requested_workers
        ):
            return capped, was_capped, reason
        self._shutdown_executor()
        self.requested_workers = requested_workers
        self.worker_count = capped
        self._cap_reason = reason
        self._safety_capped = was_capped
        return capped, was_capped, reason

    def _ensure_executor(self):
        if self.executor is not None:
            return
        context = mp.get_context(self.safety.start_method)
        try:
            with cpu_only_worker_environment():
                self.executor = ProcessPoolExecutor(
                    max_workers=self.worker_count,
                    mp_context=context,
                    initializer=_worker_initializer,
                    initargs=(
                        self.bank.current_descriptor,
                        self.bank.pool_descriptors,
                        self.training_opponent,
                        self.schema,
                        self.gamma,
                    ),
                )
                # ProcessPoolExecutor starts children lazily. Submitting one
                # warmup per worker while the parent environment is still
                # CPU-only prevents later tasks from spawning CUDA-visible
                # children after the context manager restores the environment.
                warmups = [
                    self.executor.submit(_worker_ready)
                    for _worker in range(self.worker_count)
                ]
                for future in warmups:
                    worker_state = future.result()
                    if worker_state != {
                        "force_cpu": "1",
                        "cuda_visible_devices": "",
                        "array_backend": "numpy",
                    }:
                        raise RuntimeError(
                            f"RL worker violated the CPU-only invariant: {worker_state}"
                        )
        except BaseException:
            self._shutdown_executor(terminate=True)
            raise

    def _run_jobs(self, jobs, worker_function, on_result, run_info):
        """Run a bounded queue on the persistent executor and monitor memory."""
        if not jobs:
            return
        self._ensure_executor()
        max_in_flight = max(
            self.worker_count,
            self.worker_count * self.safety.max_in_flight_per_worker,
        )
        jobs_iter = iter(jobs)
        in_flight = {}

        def submit_next():
            try:
                job = next(jobs_iter)
            except StopIteration:
                return False
            future = self.executor.submit(worker_function, job)
            in_flight[future] = job
            return True

        for _ in range(min(max_in_flight, len(jobs))):
            submit_next()

        last_memory_check = float("-inf")
        try:
            while in_flight:
                done, _pending = wait(
                    in_flight,
                    timeout=self.safety.poll_interval_s,
                    return_when=FIRST_COMPLETED,
                )
                completed_jobs = 0
                first_error = None
                for future in done:
                    in_flight.pop(future, None)
                    try:
                        results = future.result()
                    except Exception as exc:
                        if first_error is None:
                            first_error = exc
                        continue
                    for result in results:
                        on_result(result)
                    completed_jobs += 1
                if first_error is not None:
                    raise first_error

                now = time.monotonic()
                if now - last_memory_check >= self.safety.memory_check_interval_s:
                    peak_worker, total_children, available_mb = (
                        executor_memory_snapshot(self.executor)
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
                        if available_mb < self.safety.memory_reserve_mb:
                            raise DiagnosticMemoryPressure(
                                f"available RAM fell to {available_mb:.1f} MiB, "
                                f"below the {self.safety.memory_reserve_mb} MiB reserve"
                            )
                    if peak_worker > self.safety.max_worker_rss_mb:
                        raise DiagnosticMemoryPressure(
                            f"one RL worker reached {peak_worker:.1f} MiB RSS, above "
                            f"the {self.safety.max_worker_rss_mb} MiB limit"
                        )
                    last_memory_check = now

                for _ in range(completed_jobs):
                    submit_next()
        except BaseException:
            self._shutdown_executor(terminate=True)
            raise

    def _execute_specs(self, specs, worker_function, job_builder):
        """Execute unique ids, retaining completed rollouts across fallbacks."""
        specs = sorted((int(index), int(seed)) for index, seed in specs)
        if len({index for index, _seed in specs}) != len(specs):
            raise ValueError("RL game specs contain duplicate game ids")
        run_info = ParallelRunInfo(
            requested_workers=self.requested_workers,
            initial_workers=self.worker_count,
            final_workers=self.worker_count,
            safety_capped=self._safety_capped,
        )
        if self._cap_reason:
            run_info.fallback_history.append({
                "from_workers": self.requested_workers,
                "to_workers": self.worker_count,
                "completed_games": 0,
                "reason": self._cap_reason,
                "phase": "preflight",
            })

        results_by_index = {}

        def store(result):
            index = int(result["game_index"])
            results_by_index.setdefault(index, result)

        recoverable = (
            MemoryError,
            DiagnosticMemoryPressure,
            BrokenProcessPool,
            OSError,
        )
        while len(results_by_index) < len(specs):
            pending = [spec for spec in specs if spec[0] not in results_by_index]
            run_info.attempted_worker_counts.append(self.worker_count)
            jobs = job_builder(_chunk_specs(pending, self.worker_count, self.safety))
            try:
                self._run_jobs(jobs, worker_function, store, run_info)
            except recoverable as exc:
                if not self.safety.fallback_on_error or self.worker_count <= 1:
                    results = [results_by_index[index] for index in sorted(results_by_index)]
                    raise RLRolloutExecutionError(
                        f"RL rollout generation failed with {self.worker_count} "
                        f"worker(s): {exc}",
                        results,
                        run_info,
                        exc,
                    ) from exc
                next_workers = max(1, self.worker_count // 2)
                run_info.fallback_count += 1
                run_info.fallback_history.append({
                    "from_workers": self.worker_count,
                    "to_workers": next_workers,
                    "completed_games": len(results_by_index),
                    "reason": f"{type(exc).__name__}: {exc}",
                    "phase": "runtime",
                })
                self.worker_count = next_workers
                run_info.final_workers = next_workers
            except Exception as exc:
                results = [results_by_index[index] for index in sorted(results_by_index)]
                raise RLRolloutExecutionError(
                    f"RL rollout generation failed with {self.worker_count} "
                    f"worker(s): {type(exc).__name__}: {exc}",
                    results,
                    run_info,
                    exc,
                ) from exc

        return [results_by_index[index] for index, _seed in specs], run_info

    def collect_games(self, first_absolute_game, game_count, base_seed):
        """Collect an exact absolute game-id range, including partial batches."""
        first_absolute_game = int(first_absolute_game)
        game_count = int(game_count)
        if first_absolute_game < 0 or game_count < 1:
            raise ValueError("RL absolute game offset must be non-negative and count positive")
        specs = [
            (
                first_absolute_game + local_index,
                game_seed(base_seed, first_absolute_game + local_index),
            )
            for local_index in range(game_count)
        ]
        pool_slots = tuple(self.bank.pool_slots)
        return self._execute_specs(
            specs,
            _worker_collect_rollouts,
            lambda chunks: [(chunk, pool_slots) for chunk in chunks],
        )

    def collect_training_iteration(self, iteration_index, game_count, base_seed):
        """Backward-compatible fixed-size wrapper around :meth:`collect_games`."""
        return self.collect_games(
            int(iteration_index) * int(game_count),
            game_count,
            base_seed,
        )

    def close(self):
        """Stop workers before unlinking the policy bank they are viewing."""
        if self._closed:
            return
        self._closed = True
        try:
            self._shutdown_executor()
        finally:
            self.bank.close()


def _candidate_counts(candidates, safety, games_per_iteration):
    """Return useful candidates bounded by games, CPUs, RAM config, and 20."""
    cpu_limit = max(1, os.cpu_count() or 1)
    hard_limit = min(
        MAX_PARALLEL_WORKERS,
        safety.max_workers,
        cpu_limit,
        max(1, int(games_per_iteration)),
    )
    values = sorted({int(value) for value in candidates if int(value) >= 1})
    values = tuple(value for value in values if value <= hard_limit)
    if not values or values[0] != 1:
        values = (1, *values)
    return values


class RetainedRLWorkerAutotuner:
    """Benchmark worker counts across complete iterations that remain trained."""

    def __init__(
        self,
        *,
        total_iterations,
        games_per_iteration,
        safety,
        benchmark_fraction=DEFAULT_RL_AUTOTUNE_FRACTION,
        minimum_gain=DEFAULT_RL_MINIMUM_GAIN,
        candidates=DEFAULT_RL_WORKER_CANDIDATES,
        status_callback=None,
    ):
        if total_iterations < 1:
            raise ValueError("total_iterations must be positive")
        if games_per_iteration < 1:
            raise ValueError("games_per_iteration must be positive")
        if not 0 < benchmark_fraction <= 1:
            raise ValueError("benchmark_fraction must be in (0, 1]")
        if minimum_gain < 0:
            raise ValueError("minimum_gain must be non-negative")
        self.total_iterations = int(total_iterations)
        self.games_per_iteration = int(games_per_iteration)
        self.benchmark_fraction = float(benchmark_fraction)
        self.minimum_gain = float(minimum_gain)
        self.candidates = _candidate_counts(
            candidates,
            safety,
            games_per_iteration,
        )
        self.iterations_per_test = max(
            1,
            math.ceil(total_iterations * benchmark_fraction),
        )
        self.games_per_test = self.iterations_per_test * games_per_iteration
        self.emit = status_callback or (lambda message: print(message, flush=True))
        self.candidate_index = 0
        self.current_workers = self.candidates[0]
        self.optimal_workers = 1
        self.previous_success = None
        self.attempts = []
        self.finished = False
        self._attempt_iterations = 0
        self._attempt_games = 0
        self._attempt_duration = 0.0
        self._attempt_failed = False
        self._attempt_failure_reason = None
        self._attempt_runs = []
        self._finished_message_emitted = False
        self.emit(
            "Testing the optimal RL rollout worker count... each test trains "
            f"and retains {self.iterations_per_test} iteration(s) "
            f"({self.games_per_test} games, about {benchmark_fraction:.1%} of "
            "the planned RL workload)."
        )

    def reject_current_before_allocation(self, reason):
        """Stop safely when RAM/CPU preflight cannot honor a candidate."""
        self.attempts.append({
            "requested_workers": self.current_workers,
            "planned_iterations": self.iterations_per_test,
            "completed_iterations": 0,
            "planned_games": self.games_per_test,
            "completed_games": 0,
            "duration_s": 0.0,
            "games_per_second": 0.0,
            "improvement_over_previous": None,
            "passed": False,
            "failure_reason": reason,
            "runs": [],
        })
        self.emit(
            f"RL test with {self.current_workers} worker(s) failed before "
            f"allocation; reason: {reason}."
        )
        if self.previous_success is None:
            raise RuntimeError(
                "The one-worker RL baseline could not start, so no safe worker "
                "configuration can be selected."
            )
        self.finished = True
        self._emit_finished()

    def record_iteration(self, duration_s, run_info, completed_iteration):
        """Record one retained rollout batch and advance or stop the benchmark."""
        if self.finished:
            return
        self._attempt_iterations += 1
        self._attempt_games += self.games_per_iteration
        self._attempt_duration += float(duration_s)
        self._attempt_runs.append(run_info.to_dict())
        if run_info.safety_capped or run_info.fallback_count:
            self._attempt_failed = True
            history = run_info.fallback_history
            self._attempt_failure_reason = (
                history[-1]["reason"] if history else "worker safety fallback"
            )
        if self._attempt_iterations < self.iterations_per_test:
            return

        throughput = (
            self._attempt_games / self._attempt_duration
            if self._attempt_duration
            else float("inf")
        )
        improvement = None
        if self.previous_success is not None:
            improvement = (
                throughput / self.previous_success["games_per_second"] - 1.0
            )
        attempt = {
            "requested_workers": self.current_workers,
            "planned_iterations": self.iterations_per_test,
            "completed_iterations": self._attempt_iterations,
            "planned_games": self.games_per_test,
            "completed_games": self._attempt_games,
            "duration_s": self._attempt_duration,
            "games_per_second": throughput,
            "improvement_over_previous": improvement,
            "passed": not self._attempt_failed,
            "failure_reason": self._attempt_failure_reason,
            "runs": self._attempt_runs,
        }
        self.attempts.append(attempt)
        workers = self.current_workers
        worker_label = "worker" if workers == 1 else "workers"
        duration_text = (
            f"{self._attempt_duration / 60.0:.3f} min "
            f"({format_duration(self._attempt_duration)})"
        )

        if not attempt["passed"]:
            self.emit(
                f"RL test with {workers} {worker_label} failed after retaining "
                f"{self._attempt_games} games; reason: "
                f"{self._attempt_failure_reason or 'worker fallback'}."
            )
            if self.previous_success is None:
                raise RuntimeError(
                    "The one-worker RL baseline could not complete, so no safe "
                    "worker configuration can be selected."
                )
            self.finished = True
            self._emit_finished()
            return

        if self.previous_success is None:
            self.emit(
                f"RL test with {workers} {worker_label} passed; baseline "
                f"{duration_text}; {self._attempt_games} games retained."
            )
            self.optimal_workers = workers
            self.previous_success = attempt
        else:
            self.emit(
                f"RL test with {workers} {worker_label} passed; {duration_text}; "
                f"{improvement:.1%} improvement over the previous test; "
                f"{self._attempt_games} games retained."
            )
            if improvement < self.minimum_gain:
                self.emit(
                    f"Marginal gain is below {self.minimum_gain:.0%}; the current "
                    "test remains in RL training but will not be selected."
                )
                self.finished = True
                self._emit_finished()
                return
            self.optimal_workers = workers
            self.previous_success = attempt

        next_index = self.candidate_index + 1
        enough_iterations = (
            self.total_iterations - int(completed_iteration)
            >= self.iterations_per_test
        )
        if next_index >= len(self.candidates) or not enough_iterations:
            if not enough_iterations and next_index < len(self.candidates):
                self.emit(
                    "RL worker autotuning stopped because there are not enough "
                    "untrained iterations for another complete retained test."
                )
            self.finished = True
            self._emit_finished()
            return

        self.candidate_index = next_index
        self.current_workers = self.candidates[next_index]
        self._attempt_iterations = 0
        self._attempt_games = 0
        self._attempt_duration = 0.0
        self._attempt_failed = False
        self._attempt_failure_reason = None
        self._attempt_runs = []

    def _emit_finished(self):
        if self._finished_message_emitted:
            return
        self._finished_message_emitted = True
        retained_games = sum(item["completed_games"] for item in self.attempts)
        total_games = self.total_iterations * self.games_per_iteration
        self.emit(f"Optimal RL rollout configuration: {self.optimal_workers} worker(s).")
        self.emit(f"RL autotuning games retained: {retained_games}/{total_games}.")

    def to_dict(self):
        """Return JSON-serializable retained benchmark metadata."""
        return {
            "optimal_workers": self.optimal_workers,
            "candidate_workers": list(self.candidates),
            "benchmark_fraction": self.benchmark_fraction,
            "minimum_gain": self.minimum_gain,
            "iterations_per_test": self.iterations_per_test,
            "games_per_test": self.games_per_test,
            "reused_iteration_count": sum(
                item["completed_iterations"] for item in self.attempts
            ),
            "reused_game_count": sum(
                item["completed_games"] for item in self.attempts
            ),
            "attempts": self.attempts,
        }


def worker_count(value):
    """Parse ``auto`` or an RL worker count bounded by the hard limit."""
    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("RL workers must be 'auto' or an integer") from exc
    if not 1 <= parsed <= MAX_PARALLEL_WORKERS:
        raise ValueError(
            f"RL workers must be between 1 and {MAX_PARALLEL_WORKERS}"
        )
    return parsed
