"""Online worker autotuning whose benchmark games remain in the final report."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from diagnostics.parallel_runner import (
    ESTIMATED_DIAGNOSTIC_RECORD_BYTES,
    MAX_DIAGNOSTIC_WORKERS,
    DiagnosticExecutionError,
    ParallelSafetyConfig,
    evaluate_game_specs,
    game_seed,
)
from utils.resource_limits import MemorySafetyError, ensure_ram_available
from utils.runtime_status import format_duration


DEFAULT_AUTOTUNE_FRACTION = 0.01
DEFAULT_MINIMUM_GAIN = 0.10
DEFAULT_WORKER_CANDIDATES = (1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)


@dataclass(frozen=True)
class MatchupSpec:
    """Agent names and optional checkpoints needed to recreate one matchup."""

    agent: str
    opponent: str
    weights: str | Path | None = None
    opponent_weights: str | Path | None = None

    @property
    def key(self):
        return self.agent, self.opponent


def _candidate_counts(candidates, safety):
    """Return sorted candidates within CPU, configuration, and hard limits."""
    cpu_limit = max(1, os.cpu_count() or 1)
    hard_limit = min(MAX_DIAGNOSTIC_WORKERS, safety.max_workers, cpu_limit)
    values = sorted({int(value) for value in candidates if int(value) >= 1})
    values = tuple(value for value in values if value <= hard_limit)
    if not values or values[0] != 1:
        values = (1, *values)
    return values


def _allocate_sample(total_sample, matchups, candidate_index):
    """Distribute exactly one benchmark slice across matchups, rotating extras."""
    count = len(matchups)
    base, remainder = divmod(total_sample, count)
    allocation = {matchup.key: base for matchup in matchups}
    for offset in range(remainder):
        selected = matchups[(candidate_index + offset) % count]
        allocation[selected.key] += 1
    return allocation


def _first_missing_indices(completed, game_count, count):
    """Return the earliest unplayed absolute game ids for one new slice."""
    indices = []
    for game_index in range(game_count):
        if game_index not in completed:
            indices.append(game_index)
            if len(indices) == count:
                break
    return indices


def autotune_diagnostic_workers(
    *,
    matchups: tuple[MatchupSpec, ...],
    game_count: int,
    base_seed: int,
    safety: ParallelSafetyConfig,
    benchmark_fraction: float = DEFAULT_AUTOTUNE_FRACTION,
    minimum_gain: float = DEFAULT_MINIMUM_GAIN,
    candidates=DEFAULT_WORKER_CANDIDATES,
    progress_callback: Callable[[int, int], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    suppress_agent_output: bool = True,
    pair_seed_overrides: dict[tuple[str, str], int] | None = None,
):
    """Benchmark workers on unique game slices and return every produced record.

    The all-pairs orchestrator calls this function independently for each
    matchup. Each attempted worker count receives ``benchmark_fraction`` of
    that matchup. Slices use absolute game ids and remain in the final records,
    so autotuning performs useful diagnostic work.
    """
    if not matchups:
        raise ValueError("at least one matchup is required")
    if game_count < 1:
        raise ValueError("game_count must be positive")
    if not 0 < benchmark_fraction <= 1:
        raise ValueError("benchmark_fraction must be in (0, 1]")
    if minimum_gain < 0:
        raise ValueError("minimum_gain must be non-negative")

    emit = status_callback or (lambda message: print(message, flush=True))
    candidate_counts = _candidate_counts(candidates, safety)
    total_games = len(matchups) * game_count
    games_per_test = max(1, math.ceil(total_games * benchmark_fraction))
    precomputed = {matchup.key: [] for matchup in matchups}
    completed_ids = {matchup.key: set() for matchup in matchups}
    durations_by_matchup = {matchup.key: 0.0 for matchup in matchups}
    pair_seeds = {
        matchup.key: (
            int(pair_seed_overrides[matchup.key])
            if pair_seed_overrides and matchup.key in pair_seed_overrides
            else game_seed(base_seed, pair_index)
        )
        for pair_index, matchup in enumerate(matchups)
    }
    attempts = []
    total_completed = 0
    optimal_workers = 1
    previous_success = None
    matchup_label = ", ".join(
        f"{matchup.agent} vs {matchup.opponent}" for matchup in matchups
    )

    emit(
        f"Testing the optimal diagnostic worker count for {matchup_label}... "
        f"each test plays and retains {games_per_test} games "
        f"({benchmark_fraction:.1%} of this matchup)."
    )

    for candidate_index, workers in enumerate(candidate_counts):
        try:
            ensure_ram_available(
                games_per_test * ESTIMATED_DIAGNOSTIC_RECORD_BYTES,
                safety.memory_reserve_mb,
                f"retaining the {workers}-worker diagnostic benchmark slice",
            )
        except MemorySafetyError as exc:
            attempts.append({
                "requested_workers": workers,
                "planned_games": games_per_test,
                "completed_games": 0,
                "duration_s": 0.0,
                "duration_minutes": 0.0,
                "games_per_second": 0.0,
                "improvement_over_previous": None,
                "passed": False,
                "failure_reason": str(exc),
                "matchup_runs": [],
            })
            emit(
                f"Test with {workers} worker(s) failed before allocation; "
                f"reason: {exc}."
            )
            break
        allocation = _allocate_sample(games_per_test, matchups, candidate_index)
        planned = {}
        for matchup in matchups:
            planned[matchup.key] = _first_missing_indices(
                completed_ids[matchup.key],
                game_count,
                allocation[matchup.key],
            )
        planned_count = sum(len(indices) for indices in planned.values())
        if planned_count < games_per_test:
            emit(
                "Worker autotuning stopped because there are not enough unplayed "
                f"games for another complete {games_per_test}-game test."
            )
            break

        attempt_start = time.perf_counter()
        attempt_completed = 0
        attempt_infos = []
        attempt_failed = False
        failure_reason = None

        for matchup in matchups:
            indices = planned[matchup.key]
            if not indices:
                continue
            specs = [
                (game_index, game_seed(pair_seeds[matchup.key], game_index))
                for game_index in indices
            ]
            pair_progress_previous = 0

            def pair_progress(done, _total):
                nonlocal pair_progress_previous, total_completed, attempt_completed
                increment = max(0, done - pair_progress_previous)
                pair_progress_previous = done
                total_completed += increment
                attempt_completed += increment
                if progress_callback is not None:
                    progress_callback(total_completed, total_games)

            pair_start = time.perf_counter()
            try:
                records, run_info = evaluate_game_specs(
                    agent_name=matchup.agent,
                    opponent_name=matchup.opponent,
                    game_specs=specs,
                    weights=matchup.weights,
                    opponent_weights=matchup.opponent_weights,
                    requested_workers=workers,
                    suppress_agent_output=suppress_agent_output,
                    progress_callback=pair_progress,
                    safety=safety,
                )
            except DiagnosticExecutionError as exc:
                records = exc.records
                run_info = exc.run_info
                attempt_failed = True
                failure_reason = str(exc)
            pair_elapsed = time.perf_counter() - pair_start
            durations_by_matchup[matchup.key] += pair_elapsed
            attempt_infos.append(run_info.to_dict())

            existing = completed_ids[matchup.key]
            for record in records:
                game_index = int(record["game"]) - 1
                if game_index not in existing:
                    precomputed[matchup.key].append(record)
                    existing.add(game_index)

            if run_info.safety_capped:
                attempt_failed = True
                failure_reason = run_info.fallback_history[0]["reason"]
            if run_info.fallback_count:
                attempt_failed = True
                failure_reason = run_info.fallback_history[-1]["reason"]
            if attempt_failed:
                break

        elapsed = time.perf_counter() - attempt_start
        throughput = attempt_completed / elapsed if elapsed else float("inf")
        improvement = None
        if previous_success is not None:
            improvement = throughput / previous_success["games_per_second"] - 1.0

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
            "matchup_runs": attempt_infos,
        }
        attempts.append(attempt)

        worker_label = "worker" if workers == 1 else "workers"
        if not attempt["passed"]:
            emit(
                f"Test with {workers} {worker_label} failed; "
                f"{attempt_completed}/{games_per_test} games retained; "
                f"reason: {failure_reason or 'incomplete execution'}."
            )
            break

        duration_text = f"{elapsed / 60.0:.3f} min ({format_duration(elapsed)})"
        if previous_success is None:
            emit(
                f"Test with {workers} {worker_label} passed; baseline "
                f"{duration_text}; {attempt_completed} games retained."
            )
            optimal_workers = workers
            previous_success = attempt
            continue

        emit(
            f"Test with {workers} {worker_label} passed; {duration_text}; "
            f"{improvement:.1%} improvement over the previous test; "
            f"{attempt_completed} games retained."
        )
        if improvement < minimum_gain:
            emit(
                f"Marginal gain is below {minimum_gain:.0%}; the current test "
                "remains in the diagnostics but will not be selected."
            )
            break
        optimal_workers = workers
        previous_success = attempt

    if previous_success is None:
        raise RuntimeError(
            "The one-worker diagnostic baseline could not complete, so no safe "
            "worker configuration can be selected."
        )

    for records in precomputed.values():
        records.sort(key=lambda record: int(record["game"]))
    reused_game_count = sum(len(records) for records in precomputed.values())
    emit(
        f"Optimal configuration for {matchup_label}: "
        f"{optimal_workers} worker(s)."
    )
    emit(f"Autotuning games retained: {reused_game_count}/{total_games}.")

    return {
        "optimal_workers": optimal_workers,
        "candidate_workers": list(candidate_counts),
        "benchmark_fraction": benchmark_fraction,
        "minimum_gain": minimum_gain,
        "games_per_test": games_per_test,
        "reused_game_count": reused_game_count,
        "attempts": attempts,
        "precomputed_games": precomputed,
        "durations_by_matchup": durations_by_matchup,
        "pair_seeds": pair_seeds,
    }
