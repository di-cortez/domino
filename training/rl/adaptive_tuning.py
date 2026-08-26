"""Isolated throughput tuning for RL rollout workers."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import time

import numpy as np

from agents.network_architecture import (
    hidden_layer_count_from_weights,
    policy_layer_names,
)
from training.utils.seeding import stable_seed
from training.rl.constants import (
    RL_WORKER_AUTOTUNE_FRACTION,
    RL_WORKER_AUTOTUNE_MINIMUM_GAIN,
)
from training.rl.parallel import (
    DEFAULT_RL_WORKER_CANDIDATES,
    RLRolloutRunner,
    _candidate_counts,
)
from training.rl.matchmaking import build_match_plan
from training.rl.matchmaking import matchmaking_policy_manifest
from training.rl.pool import pool_policy_manifest
from utils.resource_limits import (
    MIB,
    effective_gpu_available_bytes,
    gpu_memory_info,
    process_rss_bytes,
)
from utils.repository import current_git_commit
from middleware.rulesets import DEFAULT_RULESET_NAME


TUNING_VERSION = 5


def _policy_arrays(network):
    arrays = {}
    names = list(network.weight_names)
    # Includes a separate critic network's stack, so two runs that differ only
    # in how the critic is wired never hash to the same policy digest.
    names.extend(getattr(network, "critic_parameter_names", ()))
    for name in names:
        value = network.parameter_array(name)
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


def pool_sha256(
    pool_state=None,
    pool_weights=None,
    performance_state=None,
    rotation_state=None,
):
    """Hash complete logical pool inputs without relying on bank positions."""
    digest = hashlib.sha256()
    for value in (pool_state or {}, performance_state or {}, rotation_state or {}):
        digest.update(json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"))
    for opponent_id, snapshot in sorted((pool_weights or {}).items()):
        digest.update(opponent_id.encode("utf-8"))
        for name in policy_layer_names(hidden_layer_count_from_weights(snapshot)):
            value = np.asarray(snapshot[name])
            digest.update(name.encode("ascii"))
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def capture_isolation_state(
    network,
    pool_state=None,
    pool_weights=None,
    performance_state=None,
    rotation_state=None,
):
    """Capture every mutable parent-side state that tuning must not consume."""
    return {
        "weights": _policy_arrays(network),
        "weights_sha256": policy_sha256(network),
        "optimizer": dict(network.optimizer_state_dict()),
        "rl_training_algorithm": getattr(network, "rl_training_algorithm", None),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "pool_sha256": pool_sha256(
            pool_state,
            pool_weights,
            performance_state,
            rotation_state,
        ),
        "pool_state": deepcopy(pool_state),
        "pool_weights": {
            opponent_id: {
                name: np.asarray(value).copy()
                for name, value in weights.items()
            }
            for opponent_id, weights in (pool_weights or {}).items()
        },
        "performance_state": deepcopy(performance_state),
        "rotation_state": deepcopy(rotation_state),
    }


def restore_isolation_state(
    network,
    snapshot,
    pool_state=None,
    pool_weights=None,
    performance_state=None,
    rotation_state=None,
):
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
    if pool_state is not None:
        pool_state.clear()
        pool_state.update(deepcopy(snapshot["pool_state"]))
    if pool_weights is not None:
        pool_weights.clear()
        pool_weights.update({
            opponent_id: {
                name: value.copy()
                for name, value in weights.items()
            }
            for opponent_id, weights in snapshot["pool_weights"].items()
        })
    if performance_state is not None:
        performance_state.clear()
        performance_state.update(deepcopy(snapshot["performance_state"]))
    if rotation_state is not None:
        rotation_state.clear()
        rotation_state.update(deepcopy(snapshot["rotation_state"]))
    if policy_sha256(network) != snapshot["weights_sha256"]:
        raise RuntimeError("Adaptive tuning changed RL policy weights.")
    if network.optimizer_state_dict() != snapshot["optimizer"]:
        raise RuntimeError("Adaptive tuning changed the optimizer state.")
    if getattr(network, "rl_training_algorithm", None) != saved_algorithm:
        raise RuntimeError("Adaptive tuning changed RL algorithm metadata.")
    if pool_sha256(
        pool_state,
        pool_weights,
        performance_state,
        rotation_state,
    ) != snapshot["pool_sha256"]:
        raise RuntimeError("Adaptive tuning changed the opponent pool.")


def _flatten_samples(results):
    """Estimate retained trajectory bytes without constructing a PPO buffer."""
    samples = []
    for result in results:
        samples.extend(result["samples"])
    if not samples:
        return 0, 0
    array_bytes = sum(
        int(np.asarray(sample.x).nbytes)
        + int(np.asarray(sample.legal_mask).nbytes)
        for sample in samples
    )
    # action index plus old log-prob, advantage/return, and the two persisted
    # reward components used by update diagnostics.
    scalar_bytes = len(samples) * (
        np.dtype(np.int64).itemsize + 5 * np.dtype(np.float32).itemsize
    )
    return len(samples), int(array_bytes + scalar_bytes)


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
    opponent_buckets,
    schema,
    gamma_f,
    ruleset_name=DEFAULT_RULESET_NAME,
    use_opponent_suit_features=True,
    use_opponent_bucket_features=False,
    safety,
    pool_state,
    pool_weights,
    performance_state,
    rotation_state=None,
):
    runner = RLRolloutRunner(
        network,
        opponent_buckets=opponent_buckets,
        schema=schema,
        gamma_f=gamma_f,
        ruleset_name=ruleset_name,
        safety=safety,
        use_opponent_suit_features=use_opponent_suit_features,
        use_opponent_bucket_features=use_opponent_bucket_features,
    )
    if pool_state is not None:
        # The throwaway runner owns its own rotation object, so benchmark plans
        # are representative of the real anchors without being able to advance
        # them.
        runner.restore_opponent_pool(
            pool_state,
            pool_weights,
            performance_state,
            rotation_state,
        )
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
    base_seed,
    opponent_buckets,
    difficulty_weight,
    schema,
    gamma_f,
    ruleset_name=DEFAULT_RULESET_NAME,
    use_opponent_suit_features=True,
    use_opponent_bucket_features=False,
    safety,
    pool_state=None,
    pool_weights=None,
    performance_state=None,
    rotation_state=None,
    status_callback=None,
):
    """Benchmark workers until marginal throughput gain falls below the limit."""
    emit = status_callback or (lambda _message: None)
    test_games = max(
        1,
        int(int(total_training_games) * RL_WORKER_AUTOTUNE_FRACTION),
    )
    candidates = _candidate_counts(
        DEFAULT_RL_WORKER_CANDIDATES,
        safety,
        max(gpi, test_games),
    )
    runner = _new_runner(
        network,
        opponent_buckets=opponent_buckets,
        schema=schema,
        gamma_f=gamma_f,
        ruleset_name=ruleset_name,
        use_opponent_suit_features=use_opponent_suit_features,
        use_opponent_bucket_features=use_opponent_bucket_features,
        safety=safety,
        pool_state=pool_state,
        pool_weights=pool_weights,
        performance_state=performance_state,
        rotation_state=rotation_state,
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
                    match_plan = build_match_plan(
                        opponent_pool=runner.opponent_pool,
                        performance_tracker=runner.performance_tracker,
                        uniform_rotation=runner.uniform_rotation,
                        selected_buckets=opponent_buckets,
                        difficulty_weight=difficulty_weight,
                        iteration=block_count + 1,
                        first_absolute_game=completed,
                        game_count=block_games,
                        base_seed=candidate_seed,
                    )
                    batch_results, run_info = runner.collect_games(
                        match_plan,
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
                accepted = (
                    improvement is None
                    or improvement >= RL_WORKER_AUTOTUNE_MINIMUM_GAIN
                )
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
                        f"{RL_WORKER_AUTOTUNE_MINIMUM_GAIN:.0%}; keeping "
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
    base_seed,
    opponent_buckets,
    difficulty_weight,
    schema,
    gamma_f,
    ruleset_name=DEFAULT_RULESET_NAME,
    use_opponent_suit_features=True,
    use_opponent_bucket_features=False,
    safety,
    pool_state=None,
    pool_weights=None,
    performance_state=None,
    rotation_state=None,
    output_path=None,
    status_callback=None,
):
    """Select rollout workers while restoring all parent training state."""
    emit = status_callback or (lambda _message: None)
    snapshot = capture_isolation_state(
        network,
        pool_state,
        pool_weights,
        performance_state,
        rotation_state,
    )
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
            and int(saved_tuning.get("version", -1)) == TUNING_VERSION
            and (saved_gpi is None or int(saved_gpi) == int(gpi))
            and saved_tuning.get("ruleset_name", DEFAULT_RULESET_NAME)
            == ruleset_name
            and tuple(saved_tuning.get("opponent_buckets", ()))
            == tuple(opponent_buckets)
            and float(saved_tuning.get("difficulty_weight", -1.0))
            == float(difficulty_weight)
            and saved_tuning.get("pool_policy")
            == pool_policy_manifest(opponent_buckets)
            and saved_tuning.get("matchmaking_policy")
            == matchmaking_policy_manifest()
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
                int(int(total_training_games) * RL_WORKER_AUTOTUNE_FRACTION),
            )
            emit(
                f"Selecting worker count with {worker_test_games} benchmark games "
                f"per candidate and {RL_WORKER_AUTOTUNE_MINIMUM_GAIN:.0%} minimum "
                "marginal gain..."
            )
            worker_test_games, worker_results = benchmark_worker_candidates(
                network,
                gpi=gpi,
                total_training_games=total_training_games,
                base_seed=base_seed,
                opponent_buckets=opponent_buckets,
                difficulty_weight=difficulty_weight,
                schema=schema,
                gamma_f=gamma_f,
                ruleset_name=ruleset_name,
                use_opponent_suit_features=use_opponent_suit_features,
                use_opponent_bucket_features=use_opponent_bucket_features,
                safety=safety,
                pool_state=pool_state,
                pool_weights=pool_weights,
                performance_state=performance_state,
                rotation_state=rotation_state,
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
            "ruleset_name": ruleset_name,
            "opponent_buckets": list(opponent_buckets),
            "difficulty_weight": float(difficulty_weight),
            "pool_policy": pool_policy_manifest(opponent_buckets),
            "matchmaking_policy": matchmaking_policy_manifest(),
            "worker_test_games": int(worker_test_games),
            "worker_benchmark_fraction": RL_WORKER_AUTOTUNE_FRACTION,
            "worker_minimum_gain": RL_WORKER_AUTOTUNE_MINIMUM_GAIN,
            "worker_results": worker_results,
            "selected_workers": selected_workers,
            "worker_source": worker_source,
            **current_hardware,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": current_git_commit(),
            "initial_weights_sha256": snapshot["weights_sha256"],
            "isolation_verified": True,
        }
    finally:
        restore_isolation_state(
            network,
            snapshot,
            pool_state,
            pool_weights,
            performance_state,
            rotation_state,
        )

    if output_path is not None:
        atomic_write_json(output_path, metadata)
    return metadata
