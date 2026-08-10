"""Atomic policy checkpoints and exact RL resume-state compatibility."""

from collections import deque
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import random
import secrets

import numpy as np

from agents.encoder import DominoEncoder
from agents.network_architecture import (
    hidden_layer_count_from_weights,
    policy_layer_names,
)
from agents.nn import DISABLED_DROPOUT_RATE
from agents.rl_nn import PolicyNetwork
from middleware.domino_engine import RULESET_VERSION
from training.rl_constants import (
    PPO_GPU_BUFFER_SAFETY_FRACTION,
    PPO_MAX_MINIBATCHES,
    PPO_MIN_MINIBATCHES,
)
from training.ppo import DEFAULT_TARGET_KL
from utils.repository import current_git_commit


RESUME_STATE_VERSION = 4
SUPPORTED_RESUME_STATE_VERSIONS = (RESUME_STATE_VERSION,)
# Optional regularizers added after version 4 was published. A saved state
# without them was produced with the regularizer disabled, so filling in the
# disabled value reproduces that run exactly instead of rejecting the resume.
DEFAULTED_CONFIGURATION_KEYS = {
    "weight_decay": 0.0,
    "dropout_rate": 0.0,
}
PPO_TRAINING_ALGORITHM = "ppo_v1"
LEGACY_TRAINING_ALGORITHM = "reinforce_v1"
NUMBERED_CHECKPOINT_WEIGHT_RETENTION = 5


@dataclass(frozen=True)
class RLTrainingConfiguration:
    """Typed durable RL resume configuration and source provenance."""

    total_training_games: int
    selected_gpi: int
    selected_workers: int
    rl_training_algorithm: str
    training_opponent: str
    learning_rate: float
    entropy_coef: float
    max_pool_size: int
    use_value_head: bool
    value_coef: float
    gamma: float
    reward_schema: str
    clip_grad_norm: float | None
    normalize_advantages: bool
    weight_decay: float
    dropout_rate: float
    effective_seed: int
    device: str
    sl_weights_sha256: str | None
    ppo_clip_epsilon: float
    ppo_target_kl: float
    ppo_stop_kl: float
    ppo_max_epochs: int
    ppo_games_per_minibatch_scale: int
    ppo_min_decisions_per_minibatch: int
    prefer_gpu_buffer: bool
    run_configuration_sha256: str | None = None
    git_commit: str | None = None

    @classmethod
    def from_mapping(cls, value):
        """Build the typed configuration from a runtime/checkpoint mapping."""
        data = dict(value)
        data.setdefault("ppo_target_kl", DEFAULT_TARGET_KL)
        data.setdefault("run_configuration_sha256", None)
        data.setdefault("git_commit", None)
        return cls(**{field.name: data[field.name] for field in fields(cls)})

    @classmethod
    def from_run_config(
        cls,
        run_config,
        *,
        total_training_games,
        selected_workers,
        device,
    ):
        """Build the exact RL identity from one immutable canonical run config."""
        rl = run_config["rl_config"]
        ppo = run_config["ppo_config"]
        return cls.from_mapping({
            "total_training_games": int(total_training_games),
            "selected_gpi": int(rl["games_per_iteration"]),
            "selected_workers": int(selected_workers),
            "rl_training_algorithm": run_config["algorithm"],
            "training_opponent": rl["training_opponent"],
            "learning_rate": float(rl["learning_rate"]),
            "entropy_coef": float(rl["entropy_coef"]),
            "max_pool_size": int(rl["max_pool_size"]),
            "use_value_head": bool(rl["use_value_head"]),
            "value_coef": float(rl["value_coef"]),
            "gamma": float(rl["gamma"]),
            "reward_schema": rl["reward_schema"],
            "clip_grad_norm": rl.get("clip_grad_norm"),
            "normalize_advantages": bool(rl["normalize_advantages"]),
            "weight_decay": float(rl.get("weight_decay", 0.0)),
            "dropout_rate": float(rl.get("dropout_rate", 0.0)),
            "effective_seed": int(run_config["seed"]),
            "device": device,
            "sl_weights_sha256": run_config["supervised_weights_sha256"],
            "ppo_clip_epsilon": float(ppo["clip_epsilon"]),
            "ppo_target_kl": float(ppo.get("target_kl", DEFAULT_TARGET_KL)),
            "ppo_stop_kl": float(ppo["stop_kl"]),
            "ppo_max_epochs": int(ppo["max_epochs"]),
            "ppo_games_per_minibatch_scale": int(
                ppo["games_per_minibatch_scale"]
            ),
            "ppo_min_decisions_per_minibatch": int(
                ppo["min_decisions_per_minibatch"]
            ),
            "prefer_gpu_buffer": bool(ppo["prefer_gpu_buffer"]),
            "run_configuration_sha256": run_config["configuration_sha256"],
            "git_commit": run_config.get("git_commit"),
        })

    def to_dict(self):
        """Return the durable checkpoint representation, including constants."""
        return {
            **asdict(self),
            "ruleset_version": RULESET_VERSION,
            "ppo_min_minibatches": PPO_MIN_MINIBATCHES,
            "ppo_max_minibatches": PPO_MAX_MINIBATCHES,
            "gpu_buffer_safety_fraction": PPO_GPU_BUFFER_SAFETY_FRACTION,
        }

    def warn_if_commit_changed(self, emit_status):
        """Warn, but never reject, when code changed since the run began."""
        current = current_git_commit()
        if self.git_commit and current and self.git_commit != current:
            emit_status(
                "Warning: the repository commit changed since this RL run "
                f"started ({self.git_commit[:12]} -> {current[:12]}). "
                "Resume will continue with the original saved configuration."
            )


def _checkpoint_shape(network):
    """Return the encoder-facing input and output widths of one network."""
    output_layer = getattr(network, f"W{network.layer_count}")
    return int(network.W1.shape[1]), int(output_layer.shape[0])


def _checkpoint_matches_encoder(network):
    """Return True when a loaded checkpoint matches the current encoder shape."""
    encoder = DominoEncoder()
    input_size, output_size = _checkpoint_shape(network)
    return (
        input_size == encoder.VECTOR_SIZE
        and output_size == len(encoder.all_actions)
    )


def _load_initial_network(
    learning_rate,
    sl_weights_path,
    rl_weights_path,
    quiet=False,
    use_value_head=False,
    device="auto",
    fresh_from_sl=False,
    expected_training_algorithm=None,
    weight_decay=0.0,
    dropout_rate=DISABLED_DROPOUT_RATE,
):
    """Load an RL checkpoint or initialize from compatible SL weights.

    ``fresh_from_sl=True`` ignores ``rl_weights_path`` as an initialization
    source while leaving that file intact until the completed new model
    atomically replaces it.
    """
    if rl_weights_path is not None and not fresh_from_sl:
        try:
            network = PolicyNetwork.load(
                rl_weights_path,
                learning_rate=learning_rate,
                use_value_head=use_value_head,
                device=device,
                weight_decay=weight_decay,
                dropout_rate=dropout_rate,
            )
            if not _checkpoint_matches_encoder(network):
                input_size, output_size = _checkpoint_shape(network)
                raise ValueError(
                    f"RL checkpoint {rl_weights_path} has shape "
                    f"input={input_size}, output={output_size}, "
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
        weight_decay=weight_decay,
        dropout_rate=dropout_rate,
    )
    if not _checkpoint_matches_encoder(network):
        input_size, output_size = _checkpoint_shape(network)
        raise ValueError(
            f"SL checkpoint {sl_weights_path} has shape "
            f"input={input_size}, output={output_size}, "
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
    """Retain a small rolling window of numbered policy checkpoints."""
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
        # The snapshot itself carries the layer count, so a deeper policy is
        # stored under the same pool_<index>_<name> convention.
        for name in policy_layer_names(
            hidden_layer_count_from_weights(snapshot)
        ):
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
            prefix = f"pool_{snapshot_index:03d}_"
            stored = {
                key[len(prefix):] for key in state.files if key.startswith(prefix)
            }
            try:
                names = policy_layer_names(
                    hidden_layer_count_from_weights(stored)
                )
            except ValueError as exc:
                raise ValueError(
                    f"RL resume state {state_path} has no usable policy "
                    f"weights for opponent snapshot {snapshot_index}."
                ) from exc
            weights = {}
            for name in names:
                weights[name] = np.asarray(state[prefix + name]).copy()
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


def _validate_resume_configuration(
    metadata,
    expected,
    *,
    ignored_keys=(),
    emit_status=lambda _message: None,
):
    """Reject a resume that would silently continue a different experiment."""
    saved = dict(metadata.get("configuration") or {})
    for key, disabled_value in DEFAULTED_CONFIGURATION_KEYS.items():
        saved.setdefault(key, disabled_value)
    saved.setdefault("ppo_target_kl", DEFAULT_TARGET_KL)
    saved.setdefault("git_commit", None)
    expected = expected.to_dict()
    if saved.get("git_commit") != expected.get("git_commit"):
        emit_status(
            "Warning: the checkpoint commit differs from the run's initial "
            "commit. Resume will continue with the saved training parameters."
        )
    # These legacy keys no longer affect computation. ``ppo_target_kl`` was
    # informational: PPO reports target KL, but only stop KL controls early
    # stopping. ``pool_refresh_games`` configured an opponent-snapshot cadence
    # that is now always one snapshot per iteration, controlled by gpi. Neither
    # must ever reject an otherwise exact run.
    ignored_keys = {
        "git_commit",
        "ppo_target_kl",
        "pool_refresh_games",
        *ignored_keys,
    }
    comparable_saved = {
        key: value
        for key, value in saved.items()
        if key not in ignored_keys
    }
    comparable_expected = {
        key: value for key, value in expected.items() if key not in ignored_keys
    }
    if comparable_saved != comparable_expected:
        differences = []
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
        "configuration": configuration.to_dict(),
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

    # Pool snapshots are much larger than policy checkpoints. The newest
    # state is sufficient for continuation; older numbered weight files remain
    # available for analysis while their superseded resume states are removed.
    base = Path(base_path)
    base_stem = base.name[:-len(base.suffix)] if base.suffix else base.name
    for older_state in base.parent.glob(f"{base_stem}_iter*.resume.npz"):
        if older_state != state_path:
            older_state.unlink(missing_ok=True)
    _prune_numbered_checkpoint_weights(base, weights_path)
    return weights_path, state_path


def _sl_checkpoint_sha256(path):
    path = Path(path)
    if path.is_file():
        return _file_sha256(path)
    return None


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
