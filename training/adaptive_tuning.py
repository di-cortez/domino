"""Isolated throughput tuning for RL rollout workers."""

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


TUNING_VERSION = 4
DEFAULT_WORKER_BENCHMARK_FRACTION = 0.01


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


def benchmark_worker_candidates(
    network,
    *,
    gpi,
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
    candidates = _candidate_counts(candidates, safety, max(gpi, test_games))
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
                    block_games = min(int(gpi), test_games - completed)
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


def run_worker_tuning(
    network,
    *,
    gpi,
    total_training_games,
    workers,
    retune_workers,
    saved_tuning,
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
    """Select rollout workers while restoring all parent training state."""
    emit = status_callback or (lambda _message: None)
    snapshot = capture_isolation_state(network, pool_snapshots)
    current_hardware = hardware_metadata(network.device)
    try:
        saved_gpi = None
        if saved_tuning:
            saved_gpi = saved_tuning.get(
                "gpi",
                saved_tuning.get("selected_gpi"),
            )
        reuse_saved_workers = (
            saved_tuning
            and not retune_workers
            and (saved_gpi is None or int(saved_gpi) == int(gpi))
        )
        if reuse_saved_workers:
            selected_workers = int(saved_tuning["selected_workers"])
            worker_results = list(saved_tuning.get("worker_results", []))
            worker_test_games = int(saved_tuning.get("worker_test_games", 0))
            worker_source = "resume"
        elif workers == "auto" or retune_workers:
            if saved_tuning and not retune_workers and saved_gpi is not None:
                emit(
                    f"Saved worker tuning used GPI {int(saved_gpi)}; "
                    f"retuning workers for fixed GPI {int(gpi)}."
                )
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
                gpi=gpi,
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
            "gpi": int(gpi),
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
