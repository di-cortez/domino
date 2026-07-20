"""Deterministic CPU multiprocessing for supervised dataset games.

Each worker plays independent heuristic-vs-heuristic games and returns compact
JSONL payloads. The parent process is solely responsible for aggregation and
disk writes. Per-game seeds are derived from the run seed and absolute game id,
so scheduling, worker count, and memory fallback never change game contents.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from typing import Callable, Iterable

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


DEFAULT_DATASET_WORKER_CANDIDATES = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
DEFAULT_DATASET_AUTOTUNE_FRACTION = 0.01
DEFAULT_DATASET_MINIMUM_GAIN = 0.10


class DatasetExecutionError(RuntimeError):
    """Unrecoverable dataset pool error with completed metadata attached."""

    def __init__(self, message, results, run_info, cause):
        super().__init__(message)
        self.results = results
        self.run_info = run_info
        self.__cause__ = cause


def _normalize_action(action):
    """Return a normalized tile-play action or ``None`` for forced actions."""
    if action is None or action == ["DRAW", None] or action == ("DRAW", None):
        return None
    if isinstance(action[0], list):
        return tuple(action[0]), action[1]
    return action


def _legal_tile_actions_from_state(state):
    """Reconstruct legal tile-play actions from one serialized state."""
    hand = [tuple(tile) for tile in state["current_player_hand"]]
    ends = state.get("ends", [])

    if not ends:
        doubles = [tile for tile in hand if tile[0] == tile[1]]
        if doubles:
            opening_double = max(doubles, key=lambda tile: tile[0])
            return [(opening_double, 0)]
        return [(tile, 0) for tile in hand]

    left_end, right_end = ends
    actions = []
    for tile in hand:
        if left_end in tile:
            actions.append((tile, 0))
        if right_end in tile:
            actions.append((tile, 1))
    if left_end == right_end:
        actions = [(tile, 0) for tile, _side in actions]
    return list(dict.fromkeys(actions))


def _is_real_decision_state(state):
    """Return whether the player had at least two voluntary tile plays."""
    return len(_legal_tile_actions_from_state(state)) >= 2


def generate_dataset_game(game_index: int, seed: int) -> dict:
    """Play and serialize one deterministic heuristic training game."""
    # Seed both random APIs even though the current heuristic only needs the
    # standard-library shuffle. This protects determinism if agents later use
    # NumPy without changing the dataset scheduling contract.
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)

    from agents.heuristic_agent import StrategicAgent
    from middleware.domino_engine import DominoEngine
    from middleware.middleware import GameManager

    engine = DominoEngine(player_count=2)
    # Engine counters are process-local. Replacing the id before the first
    # observed state gives every dataset game a globally stable absolute id.
    engine.game_id = int(game_index) + 1
    manager = GameManager(engine, [StrategicAgent(), StrategicAgent()])
    _, game_history = manager.play_full_game()

    saved_records = []
    skipped_turn_count = 0
    for turn in game_history:
        target_action = _normalize_action(turn["target_action"])
        if target_action is None or not _is_real_decision_state(turn["state"]):
            skipped_turn_count += 1
            continue
        saved_records.append(turn)

    # Serializing in the worker bounds inter-process Python-object overhead.
    # The parent stores this payload as one SQLite row keyed by game id.
    import json

    payload = "".join(
        json.dumps(record, ensure_ascii=False) + "\n"
        for record in saved_records
    )
    return {
        "game_index": int(game_index),
        "game_seed": int(seed),
        "jsonl_payload": payload,
        "saved_turn_count": len(saved_records),
        "skipped_turn_count": skipped_turn_count,
    }


def _dataset_worker_play_games(jobs: tuple[tuple[int, int], ...]) -> list[dict]:
    """Execute one dynamically scheduled block inside a CPU-only worker."""
    return [generate_dataset_game(game_index, seed) for game_index, seed in jobs]


def _chunk_pending_jobs(
    pending_specs: list[tuple[int, int]],
    worker_count: int,
    safety: ParallelSafetyConfig,
) -> list[tuple[tuple[int, int], ...]]:
    """Create enough small blocks for dynamic load balancing."""
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
    safety: ParallelSafetyConfig,
    on_result: Callable[[dict], None],
    run_info: ParallelRunInfo,
) -> None:
    """Run a bounded dynamic queue while monitoring child and system RAM."""
    if not queued_jobs:
        return

    executor = None
    try:
        context = mp.get_context(safety.start_method)
        with cpu_only_worker_environment():
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
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
                future = executor.submit(_dataset_worker_play_games, job)
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
                        job_results = future.result()
                    except Exception as exc:
                        if first_job_error is None:
                            first_job_error = exc
                        continue
                    for result in job_results:
                        on_result(result)
                    completed_jobs += 1
                if first_job_error is not None:
                    raise first_job_error

                now = time.monotonic()
                if now - last_memory_check >= safety.memory_check_interval_s:
                    peak_worker, total_children, available_mb = (
                        executor_memory_snapshot(executor)
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
                                f"available RAM fell to {available_mb:.1f} MiB, "
                                f"below the {safety.memory_reserve_mb} MiB reserve"
                            )
                    if peak_worker > safety.max_worker_rss_mb:
                        raise DiagnosticMemoryPressure(
                            f"one dataset worker reached {peak_worker:.1f} MiB RSS, "
                            f"above the {safety.max_worker_rss_mb} MiB limit"
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


def evaluate_dataset_game_specs(
    *,
    game_specs: Iterable[tuple[int, int]],
    requested_workers: int,
    result_callback: Callable[[dict], None] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    safety: ParallelSafetyConfig | None = None,
    retain_results: bool = True,
) -> tuple[list[dict], ParallelRunInfo]:
    """Execute absolute game ids and retain completed work across fallbacks."""
    safety = safety or ParallelSafetyConfig()
    specs = sorted((int(index), int(seed)) for index, seed in game_specs)
    if len({index for index, _seed in specs}) != len(specs):
        raise ValueError("game_specs contains duplicate game ids")
    if not specs:
        return [], ParallelRunInfo(requested_workers, 1, 1)

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

    completed_ids = set()
    retained_by_index = {}

    def store(result: dict) -> None:
        game_index = int(result["game_index"])
        if game_index in completed_ids:
            return
        if result_callback is not None:
            result_callback(result)
        completed_ids.add(game_index)
        if retain_results:
            retained_by_index[game_index] = result
        if progress_callback is not None:
            progress_callback(len(completed_ids), len(specs))

    recoverable = (MemoryError, DiagnosticMemoryPressure, BrokenProcessPool, OSError)
    while len(completed_ids) < len(specs):
        pending = [spec for spec in specs if spec[0] not in completed_ids]
        run_info.attempted_worker_counts.append(worker_count)
        try:
            _run_pending_jobs(
                _chunk_pending_jobs(pending, worker_count, safety),
                worker_count,
                safety,
                store,
                run_info,
            )
        except recoverable as exc:
            if not safety.fallback_on_error or worker_count <= 1:
                results = [retained_by_index[index] for index in sorted(retained_by_index)]
                raise DatasetExecutionError(
                    f"dataset generation failed with {worker_count} worker(s): {exc}",
                    results,
                    run_info,
                    exc,
                ) from exc
            next_workers = max(1, worker_count // 2)
            run_info.fallback_count += 1
            run_info.fallback_history.append({
                "from_workers": worker_count,
                "to_workers": next_workers,
                "completed_games": len(completed_ids),
                "reason": f"{type(exc).__name__}: {exc}",
                "phase": "runtime",
            })
            worker_count = next_workers
            run_info.final_workers = worker_count
        except Exception as exc:
            results = [retained_by_index[index] for index in sorted(retained_by_index)]
            raise DatasetExecutionError(
                f"dataset generation failed with {worker_count} worker(s): "
                f"{type(exc).__name__}: {exc}",
                results,
                run_info,
                exc,
            ) from exc

    results = [retained_by_index[index] for index, _seed in specs] if retain_results else []
    return results, run_info


def _candidate_counts(candidates, safety):
    """Return sorted worker candidates within CPU and hard safety limits."""
    cpu_limit = max(1, os.cpu_count() or 1)
    hard_limit = min(MAX_PARALLEL_WORKERS, safety.max_workers, cpu_limit)
    values = sorted({int(value) for value in candidates if int(value) >= 1})
    values = tuple(value for value in values if value <= hard_limit)
    if not values or values[0] != 1:
        values = (1, *values)
    return values


def autotune_dataset_workers(
    *,
    game_count: int,
    base_seed: int,
    safety: ParallelSafetyConfig,
    result_callback: Callable[[dict], None],
    benchmark_fraction: float = DEFAULT_DATASET_AUTOTUNE_FRACTION,
    minimum_gain: float = DEFAULT_DATASET_MINIMUM_GAIN,
    candidates=DEFAULT_DATASET_WORKER_CANDIDATES,
    progress_callback: Callable[[int, int], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> dict:
    """Benchmark dataset workers on unique games retained by the caller."""
    if game_count < 1:
        raise ValueError("game_count must be positive")
    if not 0 < benchmark_fraction <= 1:
        raise ValueError("benchmark_fraction must be in (0, 1]")
    if minimum_gain < 0:
        raise ValueError("minimum_gain must be non-negative")

    emit = status_callback or (lambda message: print(message, flush=True))
    candidate_counts = _candidate_counts(candidates, safety)
    games_per_test = max(1, math.ceil(game_count * benchmark_fraction))
    completed_ids = set()
    attempts = []
    optimal_workers = 1
    previous_success = None

    emit(
        "Testing the optimal dataset worker count... "
        f"each test plays and retains {games_per_test} games "
        f"({benchmark_fraction:.1%} of the dataset workload)."
    )

    for workers in candidate_counts:
        planned = [
            game_index
            for game_index in range(game_count)
            if game_index not in completed_ids
        ][:games_per_test]
        if len(planned) < games_per_test:
            emit(
                "Dataset worker autotuning stopped because there are not enough "
                f"unplayed games for another {games_per_test}-game test."
            )
            break

        attempt_completed = 0

        def store(result):
            nonlocal attempt_completed
            game_index = int(result["game_index"])
            if game_index in completed_ids:
                return
            result_callback(result)
            completed_ids.add(game_index)
            attempt_completed += 1
            if progress_callback is not None:
                progress_callback(len(completed_ids), game_count)

        started = time.perf_counter()
        attempt_failed = False
        failure_reason = None
        try:
            _results, run_info = evaluate_dataset_game_specs(
                game_specs=[
                    (game_index, game_seed(base_seed, game_index))
                    for game_index in planned
                ],
                requested_workers=workers,
                result_callback=store,
                safety=safety,
                retain_results=False,
            )
        except DatasetExecutionError as exc:
            run_info = exc.run_info
            attempt_failed = True
            failure_reason = str(exc)

        elapsed = time.perf_counter() - started
        throughput = attempt_completed / elapsed if elapsed else float("inf")
        improvement = None
        if previous_success is not None:
            improvement = throughput / previous_success["games_per_second"] - 1.0

        if run_info.safety_capped:
            attempt_failed = True
            failure_reason = run_info.fallback_history[0]["reason"]
        if run_info.fallback_count:
            attempt_failed = True
            failure_reason = run_info.fallback_history[-1]["reason"]

        attempt = {
            "requested_workers": workers,
            "planned_games": games_per_test,
            "completed_games": attempt_completed,
            "duration_s": elapsed,
            "duration_minutes": elapsed / 60.0,
            "games_per_second": throughput,
            "improvement_over_previous": improvement,
            "passed": not attempt_failed and attempt_completed == games_per_test,
            "failure_reason": failure_reason,
            "run": run_info.to_dict(),
        }
        attempts.append(attempt)

        worker_label = "worker" if workers == 1 else "workers"
        if not attempt["passed"]:
            emit(
                f"Dataset test with {workers} {worker_label} failed; "
                f"{attempt_completed}/{games_per_test} games retained; "
                f"reason: {failure_reason or 'incomplete execution'}."
            )
            break

        duration_text = f"{elapsed / 60.0:.3f} min ({format_duration(elapsed)})"
        if previous_success is None:
            emit(
                f"Dataset test with {workers} {worker_label} passed; baseline "
                f"{duration_text}; {attempt_completed} games retained."
            )
            optimal_workers = workers
            previous_success = attempt
            continue

        emit(
            f"Dataset test with {workers} {worker_label} passed; {duration_text}; "
            f"{improvement:.1%} improvement over the previous test; "
            f"{attempt_completed} games retained."
        )
        if improvement < minimum_gain:
            emit(
                f"Marginal gain is below {minimum_gain:.0%}; the current test "
                "remains in the dataset but will not be selected."
            )
            break
        optimal_workers = workers
        previous_success = attempt

    if previous_success is None:
        raise RuntimeError(
            "The one-worker dataset baseline could not complete, so no safe "
            "worker configuration can be selected."
        )

    emit(f"Optimal dataset configuration: {optimal_workers} worker(s).")
    emit(f"Dataset autotuning games retained: {len(completed_ids)}/{game_count}.")
    return {
        "optimal_workers": optimal_workers,
        "candidate_workers": list(candidate_counts),
        "benchmark_fraction": benchmark_fraction,
        "minimum_gain": minimum_gain,
        "games_per_test": games_per_test,
        "reused_game_count": len(completed_ids),
        "completed_game_ids": completed_ids,
        "attempts": attempts,
    }
