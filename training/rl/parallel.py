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
from multiprocessing import shared_memory
import numpy as np

from agents.network_architecture import (
    hidden_layer_count_from_weights,
    policy_layer_names,
)
from diagnostics.parallel_runner import (
    DEEP_PROFILE_SAMPLE_INTERVAL,
    MAX_PARALLEL_WORKERS,
    DiagnosticMemoryPressure,
    ParallelRunInfo,
    ParallelSafetyConfig,
    cap_parallel_workers,
    cpu_only_worker_environment,
    executor_memory_snapshot,
    game_seed,
    ignore_parent_shutdown_signals,
    terminate_executor,
)
from training.rl.pool import (
    OpponentPool,
    SharedPolicyBank,
    SharedPolicyDescriptor,
    unique_neural_capacity,
)
from training.rl.matchmaking import OpponentPerformanceTracker
from middleware.rulesets import DEFAULT_RULESET_NAME, resolve_ruleset


DEFAULT_RL_WORKERS = "auto"
DEFAULT_RL_WORKER_CANDIDATES = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)


class RLRolloutExecutionError(RuntimeError):
    """Unrecoverable rollout-pool error with completed results attached."""

    def __init__(self, message, results, run_info, cause):
        super().__init__(message)
        self.results = results
        self.run_info = run_info
        self.__cause__ = cause


class _CPUInferencePolicy:
    """Minimal NumPy policy wrapper backed directly by shared arrays."""

    xp = np
    device = "cpu"

    def __init__(self, weights):
        for name, value in weights.items():
            setattr(self, name, value)
        self.layer_count = hidden_layer_count_from_weights(weights) + 1

    @property
    def weight_names(self):
        return policy_layer_names(self.layer_count - 1)

    def forward(self, x):
        # The encoder emits float64, so cast to match the float32 weights.
        # Without it NumPy promotes every weight matrix per call, which is both
        # far slower and inconsistent with PolicyNetwork.forward.
        activation = np.asarray(x, dtype=np.float32)
        for index in range(1, self.layer_count):
            activation = np.maximum(
                0,
                np.dot(getattr(self, f"W{index}"), activation)
                + getattr(self, f"b{index}"),
            )
        logits = (
            np.dot(getattr(self, f"W{self.layer_count}"), activation)
            + getattr(self, f"b{self.layer_count}")
        )
        exp_z = np.exp(logits - np.max(logits, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)


_WORKER_SHARED_HANDLES = []
_WORKER_CURRENT_POLICY = None
_WORKER_POOL_POLICIES = ()
_WORKER_SCHEMA = None
_WORKER_GAMMA = None
_WORKER_RULESET_NAME = DEFAULT_RULESET_NAME


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
    for name, shape in zip(descriptor.weight_names, descriptor.shapes):
        size = math.prod(shape)
        weights[name] = flat[offset:offset + size].reshape(shape)
        offset += size
    return _CPUInferencePolicy(weights), segment


def _worker_initializer(
    current_descriptor,
    pool_descriptors,
    schema,
    gamma,
    ruleset_name,
):
    """Attach every reusable policy view inside one CPU-only worker."""
    global _WORKER_SHARED_HANDLES
    global _WORKER_CURRENT_POLICY, _WORKER_POOL_POLICIES
    global _WORKER_SCHEMA, _WORKER_GAMMA
    global _WORKER_RULESET_NAME

    ignore_parent_shutdown_signals()
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
    _WORKER_SCHEMA = dict(schema)
    _WORKER_GAMMA = float(gamma)
    _WORKER_RULESET_NAME = resolve_ruleset(ruleset_name).name


def _event_stats_dict(event_stats):
    """Return the four compact event counters needed by the parent process."""
    return {
        "opponent_draws": int(event_stats.opponent_draws),
        "opponent_passes": int(event_stats.opponent_passes),
        "learner_draws": int(event_stats.learner_draws),
        "learner_passes": int(event_stats.learner_passes),
    }


def _add_numeric_tree(target, source):
    """Recursively sum a compact worker timing payload."""
    for key, value in source.items():
        if isinstance(value, dict):
            _add_numeric_tree(target.setdefault(key, {}), value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            target[key] = target.get(key, 0) + value


def _add_worker_section(profile, section, started):
    sections = profile.setdefault("sections_seconds", {})
    sections[section] = sections.get(section, 0.0) + (
        time.perf_counter() - started
    )


def _worker_collect_rollouts(job):
    """Play one dynamic block of seeded training games."""
    profile_started = time.perf_counter()
    game_specs = job
    from training.rl.rollout import collect_steps_for_assignment

    results = []
    runtime_profile = {
        "games": 0,
        "profiled_games": 0,
        "profiled_game_cpu_seconds": 0.0,
        "worker_cpu_seconds": 0.0,
        "sections_seconds": {},
        "learner_policy": {},
        "opponent_policy": {},
    }
    for assignment, seed, capture_restarts in game_specs:
        game_index = assignment.game_index
        profile_game = game_index % DEEP_PROFILE_SAMPLE_INTERVAL == 0
        game_profile = runtime_profile if profile_game else None
        sampled_game_started = time.perf_counter() if profile_game else None
        section_started = time.perf_counter() if profile_game else None
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        if profile_game:
            _add_worker_section(runtime_profile, "per_game_rng_setup", section_started)
        opponent_network = (
            None
            if assignment.bank_slot is None
            else _WORKER_POOL_POLICIES[assignment.bank_slot]
        )
        collected = collect_steps_for_assignment(
            _WORKER_CURRENT_POLICY,
            assignment.opponent_kind,
            opponent_network,
            _WORKER_SCHEMA,
            _WORKER_GAMMA,
            runtime_profile=game_profile,
            ruleset_name=_WORKER_RULESET_NAME,
        )
        samples, events, winner, learner_position = collected[:4]
        captured_restarts = collected[4] if capture_restarts else ()
        section_started = time.perf_counter() if profile_game else None
        results.append({
            "game_index": int(game_index),
            "game_seed": int(seed),
            "samples": samples,
            "event_stats": _event_stats_dict(events),
            "winner": int(winner),
            "learner_position": int(learner_position),
            "bucket_name": assignment.bucket_name,
            "opponent_id": assignment.opponent_id,
            "opponent_kind": assignment.opponent_kind,
            "bank_slot": assignment.bank_slot,
            "captured_restarts": captured_restarts,
        })
        runtime_profile["games"] += 1
        if profile_game:
            _add_worker_section(
                runtime_profile,
                "result_payload_construction",
                section_started,
            )
            runtime_profile["profiled_games"] += 1
            runtime_profile["profiled_game_cpu_seconds"] += (
                time.perf_counter() - sampled_game_started
            )
    worker_seconds = time.perf_counter() - profile_started
    runtime_profile["worker_cpu_seconds"] = float(worker_seconds)
    sections = runtime_profile["sections_seconds"]
    sections["unaccounted"] = max(
        0.0,
        runtime_profile["profiled_game_cpu_seconds"] - sum(sections.values()),
    )
    for policy_key in ("learner_policy", "opponent_policy"):
        policy_profile = runtime_profile[policy_key]
        policy_sections = policy_profile.setdefault("sections_seconds", {})
        policy_sections["unaccounted"] = max(
            0.0,
            float(policy_profile.get("total_seconds", 0.0))
            - sum(policy_sections.values()),
        )
    return {
        "records": results,
        "runtime_profile": runtime_profile,
    }


def _worker_collect_restarts(job):
    """Run one dynamic block of deterministic restart continuations."""
    from training.rl.rollout import collect_steps_from_restart

    profile_started = time.perf_counter()
    results = []
    runtime_profile = {
        "games": 0,
        "profiled_games": 0,
        "profiled_game_cpu_seconds": 0.0,
        "worker_cpu_seconds": 0.0,
        "sections_seconds": {},
        "learner_policy": {},
        "opponent_policy": {},
    }
    for restart, seed in job:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        opponent_network = (
            None
            if restart.bank_slot is None
            else _WORKER_POOL_POLICIES[restart.bank_slot]
        )
        samples, events, winner, learner_position = collect_steps_from_restart(
            _WORKER_CURRENT_POLICY,
            restart.opponent_kind,
            opponent_network,
            restart,
            _WORKER_SCHEMA,
            _WORKER_GAMMA,
            ruleset_name=_WORKER_RULESET_NAME,
        )
        results.append({
            "restart_index": int(restart.restart_index),
            "restart_seed": int(seed),
            "source_game_index": int(restart.source_game_index),
            "snapshot_ordinal": int(restart.snapshot_ordinal),
            "source_turn": int(restart.source_turn),
            "source_legal_tile_action_count": int(
                restart.source_legal_tile_action_count
            ),
            "samples": samples,
            "event_stats": _event_stats_dict(events),
            "winner": int(winner),
            "learner_position": int(learner_position),
            "bucket_name": restart.bucket_name,
            "opponent_id": restart.opponent_id,
            "opponent_kind": restart.opponent_kind,
            "bank_slot": restart.bank_slot,
        })
        runtime_profile["games"] += 1
    runtime_profile["worker_cpu_seconds"] = float(
        time.perf_counter() - profile_started
    )
    return {
        "records": results,
        "runtime_profile": runtime_profile,
    }


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
        opponent_buckets,
        schema,
        gamma,
        ruleset_name=DEFAULT_RULESET_NAME,
        safety=None,
    ):
        self.safety = safety or ParallelSafetyConfig()
        self.schema = dict(schema)
        self.gamma = float(gamma)
        self.ruleset_name = resolve_ruleset(ruleset_name).name
        self.bank = SharedPolicyBank(
            network,
            unique_neural_capacity(opponent_buckets),
        )
        self.opponent_pool = OpponentPool(
            self.bank,
            selected_buckets=opponent_buckets,
            initial_network=network,
        )
        self.performance_tracker = OpponentPerformanceTracker()
        self.performance_tracker.ensure(
            record.opponent_id
            for record in self.opponent_pool.active_opponents()
        )
        self.executor = None
        self.requested_workers = 1
        self.worker_count = 1
        self._cap_reason = None
        self._safety_capped = False
        self._closed = False
        self.last_runtime_profile = {}

    @property
    def allocated_shared_bytes(self):
        return self.bank.allocated_bytes

    def sync_current(self, network):
        """Publish weights only while no worker task is in flight."""
        self.bank.write_current(network)

    def restore_opponent_pool(self, state, weights, performance_state):
        """Restore the exact opponent pool before starting resumed rollouts."""
        self._shutdown_executor()
        self.opponent_pool.restore_state(state, weights)
        self.performance_tracker.restore_state(
            performance_state,
            (
                record.opponent_id
                for record in self.opponent_pool.active_opponents()
            ),
        )

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
                        self.bank.opponent_descriptors,
                        self.schema,
                        self.gamma,
                        self.ruleset_name,
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

    def _run_jobs(
        self,
        jobs,
        worker_function,
        on_result,
        run_info,
    ):
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
                        payload = future.result()
                    except Exception as exc:
                        if first_error is None:
                            first_error = exc
                        continue
                    if (
                        isinstance(payload, dict)
                        and "records" in payload
                        and "runtime_profile" in payload
                    ):
                        results = payload["records"]
                        _add_numeric_tree(
                            run_info.runtime_profile,
                            payload["runtime_profile"],
                        )
                    else:
                        # Preserve compatibility with custom/test worker
                        # functions returning the original bare record list.
                        results = payload
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

    def _execute_specs(
        self,
        specs,
        worker_function,
        job_builder,
        *,
        index_attribute="game_index",
        result_index_key="game_index",
    ):
        """Execute unique ids, retaining completed rollouts across fallbacks."""
        specs = sorted(
            specs,
            key=lambda item: int(getattr(item[0], index_attribute)),
        )
        spec_ids = [int(getattr(item[0], index_attribute)) for item in specs]
        if len(set(spec_ids)) != len(specs):
            raise ValueError(f"RL specs contain duplicate {index_attribute} values")
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
            index = int(result[result_index_key])
            results_by_index.setdefault(index, result)

        recoverable = (
            MemoryError,
            DiagnosticMemoryPressure,
            BrokenProcessPool,
            OSError,
        )
        while len(results_by_index) < len(specs):
            pending = [
                spec
                for spec in specs
                if getattr(spec[0], index_attribute) not in results_by_index
            ]
            run_info.attempted_worker_counts.append(self.worker_count)
            jobs = job_builder(_chunk_specs(pending, self.worker_count, self.safety))
            try:
                self._run_jobs(
                    jobs,
                    worker_function,
                    store,
                    run_info,
                )
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

        self.last_runtime_profile = dict(run_info.runtime_profile)
        return [
            results_by_index[getattr(spec[0], index_attribute)]
            for spec in specs
        ], run_info

    def collect_games(
        self,
        match_plan,
        base_seed,
        *,
        capture_opponent_decision_restarts=False,
    ):
        """Execute one immutable MatchPlan without discovering opponents."""
        if not match_plan.assignments:
            raise ValueError("RL MatchPlan must contain at least one assignment")
        specs = [
            (
                assignment,
                game_seed(base_seed, assignment.game_index),
                bool(capture_opponent_decision_restarts),
            )
            for assignment in match_plan.assignments
        ]
        return self._execute_specs(
            specs,
            _worker_collect_rollouts,
            lambda chunks: list(chunks),
        )

    def collect_restarts(self, rollout_results, base_seed, iteration):
        """Run every captured continuation in deterministic source order."""
        restarts = []
        for result in sorted(rollout_results, key=lambda item: item["game_index"]):
            captures = sorted(
                result.get("captured_restarts", ()),
                key=lambda item: item.snapshot_ordinal,
            )
            for capture in captures:
                restarts.append(OpponentDecisionRestart(
                    restart_index=len(restarts),
                    source_iteration=int(iteration),
                    source_game_index=int(result["game_index"]),
                    snapshot_ordinal=int(capture.snapshot_ordinal),
                    source_turn=int(capture.source_turn),
                    original_learner_position=int(
                        capture.original_learner_position
                    ),
                    source_legal_tile_action_count=int(
                        capture.source_legal_tile_action_count
                    ),
                    opponent_kind=result["opponent_kind"],
                    opponent_id=result["opponent_id"],
                    bucket_name=result["bucket_name"],
                    bank_slot=result["bank_slot"],
                    engine_state=capture.engine_state,
                ))
        if not restarts:
            self.last_runtime_profile = {}
            return [], ParallelRunInfo(
                requested_workers=self.requested_workers,
                initial_workers=self.worker_count,
                final_workers=self.worker_count,
                safety_capped=self._safety_capped,
            )
        specs = [
            (
                restart,
                stable_seed(
                    base_seed,
                    "opponent_decision_restart",
                    iteration,
                    restart.source_game_index,
                    restart.snapshot_ordinal,
                ),
            )
            for restart in restarts
        ]
        return self._execute_specs(
            specs,
            _worker_collect_restarts,
            lambda chunks: list(chunks),
            index_attribute="restart_index",
            result_index_key="restart_index",
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
