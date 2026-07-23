"""Atomic policy checkpoints and exact RL resume-state compatibility."""

from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random
import secrets

import numpy as np

from agents.encoder import DominoEncoder
from agents.rl_nn import PolicyNetwork


RESUME_STATE_VERSION = 3
SUPPORTED_RESUME_STATE_VERSIONS = (2, RESUME_STATE_VERSION)
RESUME_POLICY_WEIGHT_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")
PPO_TRAINING_ALGORITHM = "ppo_v1"
LEGACY_TRAINING_ALGORITHM = "reinforce_v1"
NUMBERED_CHECKPOINT_WEIGHT_RETENTION = 5


def _checkpoint_matches_encoder(network):
    """Return True when a loaded checkpoint matches the current encoder shape."""
    encoder = DominoEncoder()
    return (
        network.W1.shape[1] == encoder.VECTOR_SIZE
        and network.W3.shape[0] == len(encoder.all_actions)
    )


def _load_initial_network(
    learning_rate,
    sl_weights_path,
    rl_weights_path,
    quiet=False,
    use_value_head=False,
    device="auto",
    sl_weights_data=None,
    fresh_from_sl=False,
    expected_training_algorithm=None,
):
    """Load an RL checkpoint or initialize from compatible SL weights.

    ``sl_weights_data`` accepts a pre-loaded mapping of SL weight arrays (see
    ``PolicyNetwork.load_from_sl``), so a caller warm-starting many runs from
    the same SL checkpoint can read it from disk once and reuse it. Unused when
    resuming from an existing RL checkpoint. ``fresh_from_sl=True`` ignores
    ``rl_weights_path`` as an initialization source while leaving that file
    intact until the completed new model atomically replaces it.
    """
    if rl_weights_path is not None and not fresh_from_sl:
        try:
            network = PolicyNetwork.load(
                rl_weights_path,
                learning_rate=learning_rate,
                use_value_head=use_value_head,
                device=device,
            )
            if not _checkpoint_matches_encoder(network):
                raise ValueError(
                    f"RL checkpoint {rl_weights_path} has shape "
                    f"input={network.W1.shape[1]}, output={network.W3.shape[0]}, "
                    "but the current encoder expects input=168, output=56."
                )
            saved_algorithm = getattr(network, "rl_training_algorithm", None)
            if expected_training_algorithm is not None:
                if saved_algorithm is None:
                    if expected_training_algorithm != LEGACY_TRAINING_ALGORITHM:
                        raise ValueError(
                            f"RL checkpoint {rl_weights_path} predates algorithm "
                            "metadata and cannot be continued as PPO implicitly. "
                            "Use --fresh-from-sl for a new PPO run or --no-ppo "
                            "to continue the historical update rule."
                        )
                elif saved_algorithm != expected_training_algorithm:
                    raise ValueError(
                        "RL checkpoint rl_training_algorithm is "
                        f"{saved_algorithm!r}, but "
                        f"{expected_training_algorithm!r} was requested."
                    )
                network.rl_training_algorithm = expected_training_algorithm
            if not quiet:
                print(f"Resuming RL training from {rl_weights_path}")
            return network
        except FileNotFoundError:
            pass

    network = PolicyNetwork.load_from_sl(
        sl_weights_path,
        learning_rate=learning_rate,
        use_value_head=use_value_head,
        device=device,
        data=sl_weights_data,
    )
    if not _checkpoint_matches_encoder(network):
        raise ValueError(
            f"SL checkpoint {sl_weights_path} has shape "
            f"input={network.W1.shape[1]}, output={network.W3.shape[0]}, "
            "but the current encoder expects input=168, output=56. "
            "Regenerate the supervised dataset and retrain SL first."
        )
    if expected_training_algorithm is not None:
        network.rl_training_algorithm = expected_training_algorithm
    if not quiet:
        print(f"Initializing RL policy from supervised weights: {sl_weights_path}")
    return network


def numbered_checkpoint_path(base_path, iteration):
    """Return an iteration-suffixed checkpoint path derived from ``base_path``."""
    path = Path(base_path)
    suffix = path.suffix or ".npz"
    stem = path.name[:-len(path.suffix)] if path.suffix else path.name
    return path.with_name(f"{stem}_iter{int(iteration):06d}{suffix}")


def resume_state_path(weights_path):
    """Return the auxiliary resume-state path paired with a weights checkpoint."""
    path = Path(weights_path)
    suffix = path.suffix or ".npz"
    stem = path.name[:-len(path.suffix)] if path.suffix else path.name
    return path.with_name(f"{stem}.resume{suffix}")


def _prune_numbered_checkpoint_weights(
    base_path,
    current_weights_path,
    *,
    keep=NUMBERED_CHECKPOINT_WEIGHT_RETENTION,
):
    """Retain a small rolling window of policy-only numbered checkpoints."""
    keep = int(keep)
    if keep < 1:
        raise ValueError("keep must be positive")
    base = Path(base_path)
    suffix = base.suffix or ".npz"
    stem = base.name[:-len(base.suffix)] if base.suffix else base.name
    prefix = f"{stem}_iter"
    candidates = []
    for path in base.parent.glob(f"{prefix}*{suffix}"):
        iteration_text = path.name[len(prefix):-len(suffix)]
        if iteration_text.isdigit() and path.is_file():
            candidates.append((int(iteration_text), path))

    current = Path(current_weights_path).resolve()
    retained = {current}
    for _iteration, path in sorted(candidates, reverse=True):
        if len(retained) >= keep:
            break
        retained.add(path.resolve())

    removed = []
    for _iteration, path in candidates:
        if path.resolve() not in retained:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            else:
                removed.append(path)
    return removed


def _file_sha256(path):
    """Return a streaming SHA-256 digest without duplicating checkpoint memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_network_save(network, path):
    """Publish a complete network file with one same-directory atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.tmp-{os.getpid()}-{secrets.token_hex(4)}.npz"
    )
    try:
        network.save(str(temporary))
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_resume_state_save(path, metadata, pool_snapshots):
    """Atomically save metadata and the exact self-play opponent pool."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "pool_count": np.asarray(len(pool_snapshots), dtype=np.int64),
    }
    for snapshot_index, snapshot in enumerate(pool_snapshots):
        for name in RESUME_POLICY_WEIGHT_NAMES:
            arrays[f"pool_{snapshot_index:03d}_{name}"] = np.asarray(snapshot[name])
    temporary = path.with_name(
        f".{path.stem}.tmp-{os.getpid()}-{secrets.token_hex(4)}.npz"
    )
    try:
        np.savez_compressed(temporary, **arrays)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_resume_state(weights_path, state_path):
    """Validate and load one complete weights/state checkpoint pair."""
    weights_path = Path(weights_path)
    state_path = Path(state_path)
    with np.load(state_path, allow_pickle=False) as state:
        metadata = json.loads(str(state["metadata_json"].item()))
        if metadata.get("version") not in SUPPORTED_RESUME_STATE_VERSIONS:
            raise ValueError(
                f"Unsupported RL resume-state version in {state_path}: "
                f"{metadata.get('version')!r}."
            )
        actual_hash = _file_sha256(weights_path)
        if metadata.get("weights_sha256") != actual_hash:
            raise ValueError(
                f"RL checkpoint pair is inconsistent: {weights_path} does not "
                f"match {state_path}."
            )
        pool_count = int(state["pool_count"])
        snapshots = []
        for snapshot_index in range(pool_count):
            weights = {}
            for name in RESUME_POLICY_WEIGHT_NAMES:
                key = f"pool_{snapshot_index:03d}_{name}"
                if key not in state:
                    raise ValueError(
                        f"RL resume state {state_path} is missing {key}."
                    )
                weights[name] = np.asarray(state[key]).copy()
            snapshots.append(weights)
    return metadata, tuple(snapshots)


def _rng_state_metadata():
    """Return JSON-safe parent RNG state for exact checkpoint continuation."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
    }


def _nested_tuple(value):
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _restore_rng_state(metadata):
    """Restore parent RNG state saved by :func:`_rng_state_metadata`."""
    if not metadata:
        raise ValueError("RL resume state is missing parent RNG state.")
    random.setstate(_nested_tuple(metadata["python"]))
    numpy_state = metadata["numpy"]
    np.random.set_state((
        numpy_state["bit_generator"],
        np.asarray(numpy_state["keys"], dtype=np.uint32),
        int(numpy_state["position"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))


def _resume_configuration(
    *,
    total_training_games,
    selected_gpi,
    selected_workers,
    rl_training_algorithm,
    training_opponent,
    learning_rate,
    entropy_coef,
    pool_refresh_games,
    max_pool_size,
    use_value_head,
    value_coef,
    gamma,
    reward_schema,
    clip_grad_norm,
    normalize_advantages,
    effective_seed,
    device,
    sl_weights_sha256,
    ppo_clip_epsilon,
    ppo_target_kl,
    ppo_stop_kl,
    ppo_max_epochs,
    ppo_min_minibatches,
    ppo_max_minibatches,
    ppo_games_per_minibatch_scale,
    ppo_min_decisions_per_minibatch,
    prefer_gpu_buffer,
    gpu_buffer_safety_fraction,
):
    """Return every setting that can affect post-checkpoint RL computation."""
    return {
        "total_training_games": int(total_training_games),
        "selected_gpi": int(selected_gpi),
        "selected_workers": int(selected_workers),
        "rl_training_algorithm": rl_training_algorithm,
        "training_opponent": training_opponent,
        "learning_rate": float(learning_rate),
        "entropy_coef": float(entropy_coef),
        "pool_refresh_games": int(pool_refresh_games),
        "max_pool_size": int(max_pool_size),
        "use_value_head": bool(use_value_head),
        "value_coef": float(value_coef),
        "gamma": float(gamma),
        "reward_schema": reward_schema,
        "clip_grad_norm": (
            None if clip_grad_norm is None else float(clip_grad_norm)
        ),
        "normalize_advantages": bool(normalize_advantages),
        "effective_seed": int(effective_seed),
        "device": device,
        "sl_weights_sha256": sl_weights_sha256,
        "ppo_clip_epsilon": float(ppo_clip_epsilon),
        "ppo_target_kl": float(ppo_target_kl),
        "ppo_stop_kl": float(ppo_stop_kl),
        "ppo_max_epochs": int(ppo_max_epochs),
        "ppo_min_minibatches": int(ppo_min_minibatches),
        "ppo_max_minibatches": int(ppo_max_minibatches),
        "ppo_games_per_minibatch_scale": int(ppo_games_per_minibatch_scale),
        "ppo_min_decisions_per_minibatch": int(ppo_min_decisions_per_minibatch),
        "prefer_gpu_buffer": bool(prefer_gpu_buffer),
        "gpu_buffer_safety_fraction": float(gpu_buffer_safety_fraction),
    }


def _validate_resume_configuration(metadata, expected, *, ignored_keys=()):
    """Reject a resume that would silently continue a different experiment."""
    saved = metadata.get("configuration")
    ignored_keys = set(ignored_keys)
    comparable_saved = {
        key: value
        for key, value in (saved or {}).items()
        if key not in ignored_keys
    }
    comparable_expected = {
        key: value for key, value in expected.items() if key not in ignored_keys
    }
    if comparable_saved != comparable_expected:
        differences = []
        saved = saved or {}
        for key in sorted(set(saved) | set(expected)):
            if key in ignored_keys:
                continue
            if saved.get(key) != expected.get(key):
                differences.append(
                    f"{key}: checkpoint={saved.get(key)!r}, "
                    f"requested={expected.get(key)!r}"
                )
        raise ValueError(
            "RL resume configuration does not match the checkpoint: "
            + "; ".join(differences)
        )


def _save_numbered_resume_checkpoint(
    network,
    runner,
    base_path,
    iteration,
    configuration,
    runtime_workers,
    completed_training_games,
    adaptive_tuning,
    training_state,
):
    """Save one atomic resumable checkpoint and retain only its latest state."""
    weights_path = numbered_checkpoint_path(base_path, iteration)
    state_path = resume_state_path(weights_path)

    # Invalidate an older same-iteration pair before replacing its weights.
    # A sudden interruption then leaves either the previous iteration pair or
    # the newly completed pair, never new weights with stale resume metadata.
    state_path.unlink(missing_ok=True)
    _atomic_network_save(network, weights_path)
    metadata = {
        "version": RESUME_STATE_VERSION,
        "completed_iteration": int(iteration),
        "weights_file": weights_path.name,
        "weights_sha256": _file_sha256(weights_path),
        "runtime_workers": int(runtime_workers),
        "completed_training_games": int(completed_training_games),
        "configuration": configuration,
        "optimizer_state": network.optimizer_state_dict(),
        "rng_state": _rng_state_metadata(),
        "adaptive_tuning": adaptive_tuning,
        "training_state": training_state,
        "opponent_pool_metadata": list(runner.export_pool_metadata()),
    }
    _atomic_resume_state_save(
        state_path,
        metadata,
        runner.export_pool_snapshots(),
    )

    # Pool snapshots are much larger than policy-only checkpoints. The newest
    # state is sufficient for continuation; older numbered weight files remain
    # available for analysis while their superseded resume states are removed.
    base = Path(base_path)
    base_stem = base.name[:-len(base.suffix)] if base.suffix else base.name
    for older_state in base.parent.glob(f"{base_stem}_iter*.resume.npz"):
        if older_state != state_path:
            older_state.unlink(missing_ok=True)
    _prune_numbered_checkpoint_weights(base, weights_path)
    return weights_path, state_path


def _sl_checkpoint_sha256(path, data=None):
    path = Path(path)
    if path.is_file():
        return _file_sha256(path)
    if data is None:
        return None
    digest = hashlib.sha256()
    for name in RESUME_POLICY_WEIGHT_NAMES:
        value = np.asarray(data[name])
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _training_state_payload(
    *,
    win_rate_window,
    value_loss_window,
    ppo_window,
    total_decision_samples,
    ppo_updates_completed,
    clipped_iteration_count,
    total_rollout_duration_s,
    total_update_duration_s,
    elapsed_rl_seconds,
):
    return {
        "win_rate_window": list(win_rate_window),
        "value_loss_window": list(value_loss_window),
        "ppo_window": list(ppo_window),
        "total_decision_samples": int(total_decision_samples),
        "trainable_decisions_seen": int(total_decision_samples),
        "ppo_updates_completed": int(ppo_updates_completed),
        "clipped_iteration_count": int(clipped_iteration_count),
        "total_rollout_duration_s": float(total_rollout_duration_s),
        "total_update_duration_s": float(total_update_duration_s),
        "elapsed_rl_seconds": float(elapsed_rl_seconds),
    }


def _restore_training_windows(metadata, moving_average_window):
    state = (metadata or {}).get("training_state", {})
    return (
        deque(state.get("win_rate_window", ()), maxlen=moving_average_window),
        deque(state.get("value_loss_window", ()), maxlen=moving_average_window),
        deque(state.get("ppo_window", ()), maxlen=10),
        state,
    )
