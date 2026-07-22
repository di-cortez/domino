"""Isolated throughput tuning for RL games-per-iteration and rollout workers."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import subprocess
import time

import numpy as np

from training.ppo import PPOBuffer, stable_seed
from training.rl_parallel import (
    DEFAULT_RL_MINIMUM_GAIN,
    DEFAULT_RL_WORKER_CANDIDATES,
    RLRolloutRunner,
    _candidate_counts,
)
from utils.resource_limits import (
    MIB,
    effective_gpu_available_bytes,
    gpu_memory_info,
    process_rss_bytes,
)


TUNING_VERSION = 3
DEFAULT_GPI_CANDIDATES = (100, 200, 400, 600, 800, 1000, 2000)
DEFAULT_GPI_BENCHMARK_GAMES_TARGET = 2000
DEFAULT_GPI_BENCHMARK_WORKERS = 10
DEFAULT_WORKER_BENCHMARK_FRACTION = 0.01
DEFAULT_GPI_TIE_FRACTION = 0.03
MINIMUM_SAFE_VRAM_MB = 256


def gpi_benchmark_iterations(gpi, target=DEFAULT_GPI_BENCHMARK_GAMES_TARGET):
    """Return the required ``floor(target / gpi)`` benchmark iterations."""
    gpi = int(gpi)
    target = int(target)
    if gpi < 1 or target < 1:
        raise ValueError("GPI and benchmark target must be positive.")
    return target // gpi


def _policy_arrays(network):
    arrays = {}
    names = ["W1", "b1", "W2", "b2", "W3", "b3"]
    if hasattr(network, "Wv") and hasattr(network, "bv"):
        names.extend(("Wv", "bv"))
    for name in names:
        value = getattr(network, name)
        if hasattr(value, "get"):
            value = value.get()
        arrays[name] = np.asarray(value).copy()
    return arrays


def policy_sha256(network):
    """Hash policy names, shapes, dtypes, and bytes deterministically."""
    digest = hashlib.sha256()
    for name, value in _policy_arrays(network).items():
        digest.update(name.encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def pool_sha256(pool_snapshots):
    digest = hashlib.sha256()
    for snapshot_index, snapshot in enumerate(pool_snapshots):
        digest.update(str(snapshot_index).encode("ascii"))
        for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
            value = np.asarray(snapshot[name])
            digest.update(name.encode("ascii"))
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def capture_isolation_state(network, pool_snapshots=()):
    """Capture every mutable parent-side state that tuning must not consume."""
    return {
        "weights": _policy_arrays(network),
        "weights_sha256": policy_sha256(network),
        "optimizer": dict(network.optimizer_state_dict()),
        "rl_training_algorithm": getattr(network, "rl_training_algorithm", None),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "pool_snapshots": tuple(
            {
                name: np.asarray(value).copy()
                for name, value in snapshot.items()
            }
            for snapshot in pool_snapshots
        ),
        "pool_sha256": pool_sha256(pool_snapshots),
    }


def restore_isolation_state(network, snapshot, pool_snapshots=()):
    """Restore captured state and fail if any isolation invariant differs."""
    for name, value in snapshot["weights"].items():
        setattr(network, name, network.xp.asarray(value, dtype=network.xp.float32))
    network.load_optimizer_state_dict(snapshot["optimizer"])
    saved_algorithm = snapshot["rl_training_algorithm"]
    if saved_algorithm is None:
        if hasattr(network, "rl_training_algorithm"):
            del network.rl_training_algorithm
    else:
        network.rl_training_algorithm = saved_algorithm
    random.setstate(snapshot["python_rng"])
    np.random.set_state(snapshot["numpy_rng"])
    saved_pool = snapshot["pool_snapshots"]
    if len(pool_snapshots) != len(saved_pool):
        raise RuntimeError("Adaptive tuning changed the opponent-pool size.")
    for target, saved in zip(pool_snapshots, saved_pool):
        for name, value in saved.items():
            target[name] = value.copy()
    if policy_sha256(network) != snapshot["weights_sha256"]:
        raise RuntimeError("Adaptive tuning changed RL policy weights.")
    if network.optimizer_state_dict() != snapshot["optimizer"]:
        raise RuntimeError("Adaptive tuning changed the optimizer state.")
    if getattr(network, "rl_training_algorithm", None) != saved_algorithm:
        raise RuntimeError("Adaptive tuning changed RL algorithm metadata.")
    if pool_sha256(pool_snapshots) != snapshot["pool_sha256"]:
        raise RuntimeError("Adaptive tuning changed the opponent pool.")


def _flatten_samples(results):
    samples = []
    for result in results:
        samples.extend(result["samples"])
    if not samples:
        return 0, 0
    buffer = PPOBuffer.from_samples(samples, normalize=False)
    return buffer.size, buffer.nbytes


def _resource_sample(extra_host_bytes=0):
    parent_rss = process_rss_bytes()
    gpu = gpu_memory_info()
    remaining = effective_gpu_available_bytes()
    return {
        "parent_ram_mb": (
            0.0
            if parent_rss is None
            else float((parent_rss + int(extra_host_bytes)) / MIB)
        ),
        "vram_used_mb": None if gpu is None else float(gpu.used / MIB),
        "remaining_vram_mb": None if remaining is None else float(remaining / MIB),
    }


def _memory_snapshot(run_infos, resource_samples=()):
    samples = [*resource_samples, _resource_sample()]
    parent_mb = max(sample["parent_ram_mb"] for sample in samples)
    child_peak_mb = max(
        (info.peak_total_children_rss_mb for info in run_infos),
        default=0.0,
    )
    vram_samples = [
        sample["vram_used_mb"]
        for sample in samples
        if sample["vram_used_mb"] is not None
    ]
    remaining_samples = [
        sample["remaining_vram_mb"]
        for sample in samples
        if sample["remaining_vram_mb"] is not None
    ]
    return {
        "peak_ram_mb": float(parent_mb + child_peak_mb),
        "peak_worker_rss_mb": float(max(
            (info.peak_worker_rss_mb for info in run_infos),
            default=0.0,
        )),
        "peak_vram_mb": max(vram_samples) if vram_samples else None,
        "remaining_vram_mb": min(remaining_samples) if remaining_samples else None,
    }


def _failure_result(
    base,
    exc,
    run_infos=(),
    *,
    completed_games=0,
    decision_count=0,
    trajectory_bytes=0,
    elapsed_seconds=0.0,
    resource_samples=(),
):
    cause = getattr(exc, "__cause__", None)
    is_oom = isinstance(exc, MemoryError) or isinstance(cause, MemoryError)
    return {
        **base,
        "actual_games": int(completed_games),
        "completed_games": int(completed_games),
        "elapsed_seconds": float(elapsed_seconds),
        "games_per_second": 0.0,
        "decision_count": int(decision_count),
        "trajectory_bytes": int(trajectory_bytes),
        **_memory_snapshot(run_infos, resource_samples),
        "success": False,
        "status": "oom" if is_oom else "error",
        "failure": f"{type(exc).__name__}: {exc}",
        "worker_runs": [info.to_dict() for info in run_infos],
    }


def _new_runner(
    network,
    *,
    training_opponent,
    schema,
    gamma,
    max_pool_size,
    safety,
    pool_snapshots,
):
    runner = RLRolloutRunner(
        network,
        training_opponent=training_opponent,
        schema=schema,
        gamma=gamma,
        max_pool_size=max_pool_size if training_opponent == "self_play" else 0,
        safety=safety,
    )
    if pool_snapshots:
        runner.restore_pool_snapshots(pool_snapshots)
    return runner


def _valid_run_infos(run_infos):
    return all(
        not info.safety_capped and not info.fallback_count
        for info in run_infos
    )


def benchmark_gpi_candidates(
    network,
    *,
    candidates,
    benchmark_games_target,
    base_seed,
    training_opponent,
    schema,
    gamma,
    max_pool_size,
    safety,
    pool_snapshots=(),
    status_callback=None,
):
    """Benchmark all requested GPIs with ten frozen-policy workers."""
    emit = status_callback or (lambda _message: None)
    candidates = tuple(int(value) for value in candidates)
    if not candidates or any(value < 1 for value in candidates):
        raise ValueError("GPI candidates must be positive and non-empty.")
    runner = _new_runner(
        network,
        training_opponent=training_opponent,
        schema=schema,
        gamma=gamma,
        max_pool_size=max_pool_size,
        safety=safety,
        pool_snapshots=pool_snapshots,
    )
    results = []
    try:
        actual_workers, was_capped, cap_reason = runner.set_workers(
            DEFAULT_GPI_BENCHMARK_WORKERS
        )
        if was_capped or actual_workers != DEFAULT_GPI_BENCHMARK_WORKERS:
            detail = f" ({cap_reason})" if cap_reason else ""
            raise RuntimeError(
                "GPI tuning requires exactly "
                f"{DEFAULT_GPI_BENCHMARK_WORKERS} workers, but resource safety "
                f"allowed {actual_workers}{detail}."
            )
        # Process creation and one short rollout initialize worker imports and
        # internal structures without entering any candidate's timer.
        warmup_seed = stable_seed(base_seed, "gpi_autotune", "warmup")
        warmup_results, _warmup_info = runner.collect_games(0, 2, warmup_seed)
        _flatten_samples(warmup_results)
        for gpi in candidates:
            benchmark_iterations = gpi_benchmark_iterations(
                gpi,
                benchmark_games_target,
            )
            actual_games = benchmark_iterations * gpi
            base = {
                "gpi": gpi,
                "benchmark_iterations": benchmark_iterations,
                "planned_games": actual_games,
            }
            if benchmark_iterations < 1:
                results.append(_failure_result(
                    base,
                    ValueError("GPI exceeds the benchmark-games target"),
                ))
                continue
            candidate_seed = stable_seed(base_seed, "gpi_autotune", gpi)
            run_infos = []
            decision_count = 0
            trajectory_bytes = 0
            resource_samples = []
            completed_games = 0
            started = None
            try:
                started = time.perf_counter()
                for local_iteration in range(benchmark_iterations):
                    batch_results, run_info = runner.collect_games(
                        local_iteration * gpi,
                        gpi,
                        candidate_seed,
                    )
                    run_infos.append(run_info)
                    if len(batch_results) != gpi:
                        raise RuntimeError(
                            f"Expected {gpi} benchmark games, got {len(batch_results)}."
                        )
                    decisions, buffer_bytes = _flatten_samples(batch_results)
                    decision_count += decisions
                    trajectory_bytes = max(trajectory_bytes, buffer_bytes)
                    completed_games += len(batch_results)
                    resource_samples.append(_resource_sample(buffer_bytes))
                network.synchronize()
                elapsed = time.perf_counter() - started
                remaining_vram = effective_gpu_available_bytes()
                if not _valid_run_infos(run_infos):
                    raise RuntimeError("candidate required worker memory recovery")
                if (
                    network.device == "gpu"
                    and remaining_vram is not None
                    and remaining_vram < MINIMUM_SAFE_VRAM_MB * MIB
                ):
                    raise MemoryError("candidate left unsafe VRAM headroom")
                result = {
                    **base,
                    "actual_games": int(completed_games),
                    "completed_games": int(completed_games),
                    "elapsed_seconds": float(elapsed),
                    "games_per_second": float(actual_games / elapsed),
                    "decision_count": int(decision_count),
                    "trajectory_bytes": int(trajectory_bytes),
                    **_memory_snapshot(run_infos, resource_samples),
                    "success": True,
                    "status": "success",
                    "failure": None,
                    "worker_runs": [info.to_dict() for info in run_infos],
                }
            except Exception as exc:
                elapsed = 0.0 if started is None else time.perf_counter() - started
                result = _failure_result(
                    base,
                    exc,
                    run_infos,
                    completed_games=completed_games,
                    decision_count=decision_count,
                    trajectory_bytes=trajectory_bytes,
                    elapsed_seconds=elapsed,
                    resource_samples=resource_samples,
                )
            results.append(result)
            if result["success"]:
                emit(
                    f"  GPI {gpi:4d}: {actual_games} games, "
                    f"{benchmark_iterations:2d} iterations, "
                    f"{result['games_per_second']:.1f} games/s"
                )
            else:
                emit(f"  GPI {gpi:4d}: failed ({result['failure']})")
    finally:
        runner.close()
    return results


def select_fastest(results, *, key, tie_fraction):
    """Select fastest valid candidate, preferring the smaller key within a tie."""
    valid = [row for row in results if row.get("success")]
    if not valid:
        raise RuntimeError("No adaptive RL tuning candidate completed safely.")
    best_rate = max(float(row["games_per_second"]) for row in valid)
    tied = [
        row for row in valid
        if float(row["games_per_second"]) >= best_rate * (1.0 - float(tie_fraction))
    ]
    return min(tied, key=lambda row: int(row[key]))


def benchmark_worker_candidates(
    network,
    *,
    selected_gpi,
    total_training_games,
    benchmark_fraction,
    minimum_gain=DEFAULT_RL_MINIMUM_GAIN,
    candidates,
    base_seed,
    training_opponent,
    schema,
    gamma,
    max_pool_size,
    safety,
    pool_snapshots=(),
    status_callback=None,
):
    """Benchmark workers until marginal throughput gain falls below the limit."""
    emit = status_callback or (lambda _message: None)
    if float(minimum_gain) < 0:
        raise ValueError("minimum_gain must be non-negative.")
    test_games = max(1, int(int(total_training_games) * float(benchmark_fraction)))
    candidates = _candidate_counts(candidates, safety, max(selected_gpi, test_games))
    runner = _new_runner(
        network,
        training_opponent=training_opponent,
        schema=schema,
        gamma=gamma,
        max_pool_size=max_pool_size,
        safety=safety,
        pool_snapshots=pool_snapshots,
    )
    results = []
    previous_success = None
    try:
        for requested_workers in candidates:
            base = {
                "requested_workers": int(requested_workers),
                "planned_games": int(test_games),
            }
            run_infos = []
            decision_count = 0
            trajectory_bytes = 0
            resource_samples = []
            completed = 0
            started = None
            try:
                selected_workers, was_capped, cap_reason = runner.set_workers(
                    requested_workers
                )
                if was_capped or selected_workers != requested_workers:
                    raise MemoryError(cap_reason or "worker candidate was safety-capped")
                candidate_seed = stable_seed(
                    base_seed,
                    "worker_autotune",
                    requested_workers,
                )
                started = time.perf_counter()
                block_count = 0
                while completed < test_games:
                    block_games = min(int(selected_gpi), test_games - completed)
                    batch_results, run_info = runner.collect_games(
                        completed,
                        block_games,
                        candidate_seed,
                    )
                    run_infos.append(run_info)
                    if len(batch_results) != block_games:
                        raise RuntimeError(
                            f"Expected {block_games} worker-test games, "
                            f"got {len(batch_results)}."
                        )
                    decisions, buffer_bytes = _flatten_samples(batch_results)
                    decision_count += decisions
                    trajectory_bytes = max(trajectory_bytes, buffer_bytes)
                    completed += block_games
                    block_count += 1
                    resource_samples.append(_resource_sample(buffer_bytes))
                network.synchronize()
                elapsed = time.perf_counter() - started
                if completed != test_games:
                    raise RuntimeError("Worker benchmark did not use its exact game budget.")
                if not _valid_run_infos(run_infos):
                    raise RuntimeError("candidate required worker memory recovery")
                result = {
                    **base,
                    "actual_games": int(completed),
                    "completed_games": int(completed),
                    "selected_workers": int(selected_workers),
                    "blocks": int(block_count),
                    "elapsed_seconds": float(elapsed),
                    "games_per_second": float(test_games / elapsed),
                    "decision_count": int(decision_count),
                    "trajectory_bytes": int(trajectory_bytes),
                    **_memory_snapshot(run_infos, resource_samples),
                    "success": True,
                    "status": "success",
                    "failure": None,
                    "worker_runs": [info.to_dict() for info in run_infos],
                }
            except Exception as exc:
                elapsed = 0.0 if started is None else time.perf_counter() - started
                result = _failure_result(
                    base,
                    exc,
                    run_infos,
                    completed_games=completed,
                    decision_count=decision_count,
                    trajectory_bytes=trajectory_bytes,
                    elapsed_seconds=elapsed,
                    resource_samples=resource_samples,
                )
                result["selected_workers"] = int(runner.worker_count)
                result["blocks"] = len(run_infos)
            if result["success"]:
                improvement = (
                    None
                    if previous_success is None
                    else (
                        float(result["games_per_second"])
                        / float(previous_success["games_per_second"])
                        - 1.0
                    )
                )
                accepted = improvement is None or improvement >= float(minimum_gain)
                result["improvement_over_previous"] = improvement
                result["accepted"] = bool(accepted)
                results.append(result)
                comparison = (
                    "baseline"
                    if improvement is None
                    else f"{improvement:+.1%} vs previous"
                )
                emit(
                    f"  workers {requested_workers:2d}: "
                    f"{test_games} games, {result['games_per_second']:.1f} games/s "
                    f"({comparison})"
                )
                if not accepted:
                    emit(
                        "  Marginal worker gain is below "
                        f"{float(minimum_gain):.0%}; keeping "
                        f"{previous_success['requested_workers']} workers and "
                        "stopping worker tuning."
                    )
                    break
                previous_success = result
            else:
                result["improvement_over_previous"] = None
                result["accepted"] = False
                results.append(result)
                emit(
                    f"  workers {requested_workers:2d}: failed "
                    f"({result['failure']})"
                )
                # Match the pipeline's other worker tuners: preserve the last
                # accepted candidate and do not probe beyond a failed one.
                break
    finally:
        runner.close()
    return test_games, results


def selected_worker_candidate(results):
    """Return the last candidate accepted by marginal-gain worker tuning."""
    accepted = [row for row in results if row.get("success") and row.get("accepted")]
    if not accepted:
        raise RuntimeError(
            "The one-worker RL baseline could not complete, so no safe worker "
            "configuration can be selected."
        )
    return accepted[-1]


def _gpu_name():
    try:
        import cupy

        properties = cupy.cuda.runtime.getDeviceProperties(0)
        value = properties.get("name", "CUDA device")
        return value.decode(errors="replace") if isinstance(value, bytes) else str(value)
    except Exception:
        return None


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        return None


def hardware_metadata(device):
    return {
        "device": str(device),
        "gpu_name": _gpu_name(),
        "cpu_count": int(os.cpu_count() or 1),
    }


def hardware_warning(saved, current):
    changed = [
        key for key in ("device", "gpu_name", "cpu_count")
        if saved.get(key) != current.get(key)
    ]
    if not changed:
        return None
    details = ", ".join(
        f"{key}: saved={saved.get(key)!r}, current={current.get(key)!r}"
        for key in changed
    )
    return f"Adaptive-tuning hardware changed ({details}); saved choices were reused."


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
    )
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_adaptive_tuning(
    network,
    *,
    total_training_games,
    manual_gpi,
    adaptive_gpi,
    workers,
    retune_gpi,
    retune_workers,
    saved_tuning,
    gpi_candidates,
    gpi_benchmark_games_target,
    worker_benchmark_fraction,
    worker_minimum_gain,
    worker_candidates,
    base_seed,
    training_opponent,
    schema,
    gamma,
    max_pool_size,
    safety,
    pool_snapshots=(),
    output_path=None,
    status_callback=None,
):
    """Select frozen GPI/workers while restoring all parent training state."""
    emit = status_callback or (lambda _message: None)
    snapshot = capture_isolation_state(network, pool_snapshots)
    current_hardware = hardware_metadata(network.device)
    try:
        if saved_tuning and not retune_gpi:
            selected_gpi = int(saved_tuning["selected_gpi"])
            gpi_results = list(saved_tuning.get("gpi_results", []))
            gpi_source = "resume"
        elif adaptive_gpi or retune_gpi:
            emit(
                "Selecting GPI for rollout throughput with "
                f"{DEFAULT_GPI_BENCHMARK_WORKERS} workers..."
            )
            gpi_results = benchmark_gpi_candidates(
                network,
                candidates=gpi_candidates,
                benchmark_games_target=gpi_benchmark_games_target,
                base_seed=base_seed,
                training_opponent=training_opponent,
                schema=schema,
                gamma=gamma,
                max_pool_size=max_pool_size,
                safety=safety,
                pool_snapshots=pool_snapshots,
                status_callback=emit,
            )
            selected_gpi = int(select_fastest(
                gpi_results,
                key="gpi",
                tie_fraction=DEFAULT_GPI_TIE_FRACTION,
            )["gpi"])
            gpi_source = "autotune"
        else:
            selected_gpi = int(manual_gpi)
            gpi_results = []
            gpi_source = "manual"
        emit(f"Selected GPI: {selected_gpi} ({gpi_source}).")

        if saved_tuning and not retune_workers:
            selected_workers = int(saved_tuning["selected_workers"])
            worker_results = list(saved_tuning.get("worker_results", []))
            worker_test_games = int(saved_tuning.get("worker_test_games", 0))
            worker_source = "resume"
        elif workers == "auto" or retune_workers:
            worker_test_games = max(
                1,
                int(int(total_training_games) * float(worker_benchmark_fraction)),
            )
            emit(
                f"Selecting worker count with {worker_test_games} benchmark games "
                f"per candidate and {float(worker_minimum_gain):.0%} minimum "
                "marginal gain..."
            )
            worker_test_games, worker_results = benchmark_worker_candidates(
                network,
                selected_gpi=selected_gpi,
                total_training_games=total_training_games,
                benchmark_fraction=worker_benchmark_fraction,
                minimum_gain=worker_minimum_gain,
                candidates=worker_candidates,
                base_seed=base_seed,
                training_opponent=training_opponent,
                schema=schema,
                gamma=gamma,
                max_pool_size=max_pool_size,
                safety=safety,
                pool_snapshots=pool_snapshots,
                status_callback=emit,
            )
            selected_workers = int(
                selected_worker_candidate(worker_results)["requested_workers"]
            )
            worker_source = "autotune"
        else:
            selected_workers = int(workers)
            worker_results = []
            worker_test_games = 0
            worker_source = "manual"
        emit(f"Selected workers: {selected_workers} ({worker_source}).")

        metadata = {
            "version": TUNING_VERSION,
            "base_seed": int(base_seed),
            "total_training_games": int(total_training_games),
            "gpi_candidates": [int(value) for value in gpi_candidates],
            "gpi_benchmark_games_target": int(gpi_benchmark_games_target),
            "gpi_benchmark_workers": DEFAULT_GPI_BENCHMARK_WORKERS,
            "gpi_results": gpi_results,
            "selected_gpi": selected_gpi,
            "gpi_source": gpi_source,
            "worker_test_games": int(worker_test_games),
            "worker_benchmark_fraction": float(worker_benchmark_fraction),
            "worker_minimum_gain": float(worker_minimum_gain),
            "worker_results": worker_results,
            "selected_workers": selected_workers,
            "worker_source": worker_source,
            **current_hardware,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "initial_weights_sha256": snapshot["weights_sha256"],
            "isolation_verified": True,
        }
    finally:
        restore_isolation_state(network, snapshot, pool_snapshots)

    if output_path is not None:
        atomic_write_json(output_path, metadata)
    return metadata
