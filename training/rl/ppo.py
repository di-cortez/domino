"""Masked PPO minibatch updates for the domino self-play policy.

The canonical decision buffer always lives in host RAM.  A GPU-backed learner
may additionally keep a complete device copy when conservative VRAM preflight
and a real first-minibatch workspace probe both succeed.  Otherwise the same
immutable host buffer is streamed one minibatch at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable

import numpy as np

from training.utils.seeding import stable_seed
from utils.resource_limits import effective_gpu_available_bytes


# PPO IMPLEMENTATION CONSTANTS
#
# ``PPO_MAX_EPOCHS`` is the only PPO training control exposed to callers. A
# value of one deliberately selects the separate single-update REINFORCE path;
# values from two through sixteen select PPO. Everything else below is a fixed
# implementation policy and is recorded in manifests for provenance, not
# threaded through the training loop as configurable state.
PPO_DISABLED_EPOCHS = 1
DEFAULT_PPO_MAX_EPOCHS = 4
MAX_PPO_EPOCHS = 16
PPO_TRAINING_ALGORITHM = "ppo_v1"
REINFORCE_TRAINING_ALGORITHM = "reinforce_v1"
PPO_CLIP_EPSILON = 0.2
PPO_TARGET_KL = 0.01
PPO_STOP_KL = 0.015
PPO_GAMES_PER_MINIBATCH_SCALE = 125
PPO_MIN_DECISIONS_PER_MINIBATCH = 128
PPO_MIN_MINIBATCHES = 4
PPO_MAX_MINIBATCHES = 16
PPO_PREFER_GPU_BUFFER = True
PPO_GPU_BUFFER_SAFETY_FRACTION = 0.70
POLICY_GRADIENT_CLIP_NORM = 5.0
ADVANTAGE_EPSILON = 1e-8


def validate_ppo_max_epochs(max_epochs):
    """Return a valid user-facing PPO epoch budget."""
    max_epochs = int(max_epochs)
    if not PPO_DISABLED_EPOCHS <= max_epochs <= MAX_PPO_EPOCHS:
        raise ValueError(
            f"ppo_max_epochs must be between {PPO_DISABLED_EPOCHS} and "
            f"{MAX_PPO_EPOCHS}"
        )
    return max_epochs


def ppo_is_enabled(max_epochs):
    """Return whether an epoch budget selects PPO instead of REINFORCE."""
    return validate_ppo_max_epochs(max_epochs) > PPO_DISABLED_EPOCHS


def fixed_ppo_policy(max_epochs):
    """Return the complete persisted PPO policy without configurable aliases."""
    max_epochs = validate_ppo_max_epochs(max_epochs)
    return {
        "algorithm": (
            PPO_TRAINING_ALGORITHM
            if ppo_is_enabled(max_epochs)
            else REINFORCE_TRAINING_ALGORITHM
        ),
        "max_epochs": max_epochs,
        "fixed_policy": {
            "clip_epsilon": PPO_CLIP_EPSILON,
            "target_kl": PPO_TARGET_KL,
            "stop_kl": PPO_STOP_KL,
            "min_minibatches": PPO_MIN_MINIBATCHES,
            "max_minibatches": PPO_MAX_MINIBATCHES,
            "games_per_minibatch_scale": PPO_GAMES_PER_MINIBATCH_SCALE,
            "min_decisions_per_minibatch": PPO_MIN_DECISIONS_PER_MINIBATCH,
            "prefer_gpu_buffer": PPO_PREFER_GPU_BUFFER,
            "gpu_buffer_safety_fraction": PPO_GPU_BUFFER_SAFETY_FRACTION,
        },
    }


def _to_numpy(value, *, dtype=None):
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=dtype)


def normalize_advantages(values, epsilon=ADVANTAGE_EPSILON):
    """Normalize one complete iteration globally, safely handling zero variance."""
    advantages = np.asarray(values, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(advantages)):
        raise ValueError("PPO advantages contain NaN or infinity.")
    mean = float(np.mean(advantages, dtype=np.float64))
    std = float(np.std(advantages, dtype=np.float64))
    centered = advantages - np.float32(mean)
    if std <= float(epsilon):
        return np.ascontiguousarray(centered, dtype=np.float32), True, mean, std
    normalized = centered / np.float32(std + float(epsilon))
    return np.ascontiguousarray(normalized, dtype=np.float32), False, mean, std


@dataclass(frozen=True)
class PPOBuffer:
    """One immutable on-policy decision batch in contiguous host arrays."""

    states: np.ndarray
    actions: np.ndarray
    legal_masks: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray | None
    advantages: np.ndarray
    returns: np.ndarray
    local_rewards: np.ndarray
    terminal_rewards: np.ndarray
    advantage_std_zero: bool
    raw_advantage_mean: float
    raw_advantage_std: float

    @classmethod
    def from_samples(
        cls,
        samples: Iterable,
        *,
        normalize=True,
        old_values=None,
    ):
        samples = list(samples)
        if not samples:
            raise ValueError("Cannot build a PPO buffer without real decisions.")
        states = np.ascontiguousarray(
            np.hstack([
                _to_numpy(sample.x, dtype=np.float32)
                for sample in samples
            ]),
            dtype=np.float32,
        )
        legal_masks = np.ascontiguousarray(
            np.hstack([
                _to_numpy(sample.legal_mask) > 0
                for sample in samples
            ]),
            dtype=np.bool_,
        )
        actions = np.ascontiguousarray(
            [sample.action_index for sample in samples],
            dtype=np.int64,
        )
        old_log_probs = np.ascontiguousarray(
            [sample.old_log_prob for sample in samples],
            dtype=np.float32,
        )
        returns = np.ascontiguousarray(
            [sample.policy_reward for sample in samples],
            dtype=np.float32,
        )
        local_rewards = np.ascontiguousarray(
            [sample.local_reward for sample in samples],
            dtype=np.float32,
        )
        terminal_rewards = np.ascontiguousarray(
            [sample.terminal_reward for sample in samples],
            dtype=np.float32,
        )
        if old_values is not None:
            old_values = np.ascontiguousarray(
                _to_numpy(old_values, dtype=np.float32).reshape(-1),
                dtype=np.float32,
            )
            if old_values.size != len(samples):
                raise ValueError(
                    "PPO old_values must contain one value per decision."
                )
            if not np.all(np.isfinite(old_values)):
                raise ValueError("PPO old_values contain NaN or infinity.")
        if states.ndim != 2 or states.shape[1] != len(samples):
            raise ValueError("PPO states must have one column per decision.")
        if legal_masks.ndim != 2 or legal_masks.shape[1] != len(samples):
            raise ValueError("PPO legal masks must have one column per decision.")
        if np.any(legal_masks.sum(axis=0) < 2):
            raise ValueError("PPO buffer contains a forced or single-option action.")
        if np.any(actions < 0) or np.any(actions >= legal_masks.shape[0]):
            raise ValueError("PPO buffer contains an out-of-range action index.")
        if np.any(~legal_masks[actions, np.arange(len(samples))]):
            raise ValueError("PPO buffer contains an action outside its legal mask.")
        if not np.all(np.isfinite(old_log_probs)):
            raise ValueError("PPO old_log_probs contain NaN or infinity.")
        raw_advantages = (
            returns.copy()
            if old_values is None
            else returns - old_values
        )
        if normalize:
            advantages, std_zero, raw_mean, raw_std = normalize_advantages(
                raw_advantages
            )
        else:
            advantages = raw_advantages.copy()
            std_zero = False
            raw_mean = float(np.mean(raw_advantages, dtype=np.float64))
            raw_std = float(np.std(raw_advantages, dtype=np.float64))
        buffer = cls(
            states=states,
            actions=actions,
            legal_masks=legal_masks,
            old_log_probs=old_log_probs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            local_rewards=local_rewards,
            terminal_rewards=terminal_rewards,
            advantage_std_zero=std_zero,
            raw_advantage_mean=raw_mean,
            raw_advantage_std=raw_std,
        )
        for array in (
            buffer.states,
            buffer.actions,
            buffer.legal_masks,
            buffer.old_log_probs,
            buffer.advantages,
            buffer.returns,
            buffer.local_rewards,
            buffer.terminal_rewards,
        ):
            array.setflags(write=False)
        if buffer.old_values is not None:
            buffer.old_values.setflags(write=False)
        return buffer

    @property
    def size(self):
        return int(self.actions.size)

    @property
    def nbytes(self):
        arrays = [
            self.states,
            self.actions,
            self.legal_masks,
            self.old_log_probs,
            self.advantages,
            self.returns,
            self.local_rewards,
            self.terminal_rewards,
        ]
        if self.old_values is not None:
            arrays.append(self.old_values)
        return int(sum(
            array.nbytes
            for array in arrays
        ))


@dataclass(frozen=True)
class PPOIterationUpdate:
    """Completed PPO update plus the two parent-level timing boundaries."""

    metrics: dict
    buffer_seconds: float
    update_seconds: float


def requested_minibatches(
    actual_games,
):
    """Return the requested 4..16 minibatch count based on games collected."""
    if actual_games < 1:
        raise ValueError("actual_games must be positive.")
    raw = math.ceil(int(actual_games) / PPO_GAMES_PER_MINIBATCH_SCALE)
    return max(PPO_MIN_MINIBATCHES, min(PPO_MAX_MINIBATCHES, raw))


def effective_minibatches(
    decision_count,
    requested,
):
    """Cap minibatches so every non-final slice remains operationally useful."""
    if decision_count < 1 or requested < 1:
        raise ValueError("Decision/minibatch counts must be positive.")
    maximum_useful = max(
        1,
        int(decision_count) // PPO_MIN_DECISIONS_PER_MINIBATCH,
    )
    return min(int(requested), maximum_useful, int(decision_count))


def minibatch_indices(decision_count, minibatch_count, seed):
    """Return a deterministic no-drop partition with every index exactly once."""
    if not 1 <= int(minibatch_count) <= int(decision_count):
        raise ValueError("minibatch_count must be between one and decision_count.")
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    permutation = rng.permutation(int(decision_count))
    partitions = tuple(
        np.ascontiguousarray(part, dtype=np.int64)
        for part in np.array_split(permutation, int(minibatch_count))
    )
    if any(part.size == 0 for part in partitions):
        raise AssertionError("PPO created an empty minibatch.")
    combined = np.concatenate(partitions)
    if combined.size != decision_count or np.unique(combined).size != decision_count:
        raise AssertionError("PPO minibatches lost or duplicated a decision.")
    return partitions


def clipped_surrogate(ratios, advantages):
    """Return NumPy PPO surrogate terms, exposed for exact unit tests."""
    ratios = np.asarray(ratios, dtype=np.float64)
    advantages = np.asarray(advantages, dtype=np.float64)
    clipped = np.clip(
        ratios,
        1.0 - PPO_CLIP_EPSILON,
        1.0 + PPO_CLIP_EPSILON,
    )
    return np.minimum(ratios * advantages, clipped * advantages)


def log_ratio_statistics(new_log_probs, old_log_probs):
    """Return whole-buffer PPO ratio, KL, and clipping statistics."""
    new = np.asarray(new_log_probs, dtype=np.float64)
    old = np.asarray(old_log_probs, dtype=np.float64)
    log_ratio = new - old
    ratio = np.exp(log_ratio)
    if not np.all(np.isfinite(ratio)):
        raise FloatingPointError("PPO probability ratio contains NaN or infinity.")
    approx_kl = float(np.mean((ratio - 1.0) - log_ratio))
    lower = 1.0 - PPO_CLIP_EPSILON
    upper = 1.0 + PPO_CLIP_EPSILON
    return {
        "approx_kl": max(0.0, approx_kl),
        "clip_fraction": float(np.mean((ratio < lower) | (ratio > upper))),
        "ratio_mean": float(np.mean(ratio)),
        "ratio_min": float(np.min(ratio)),
        "ratio_max": float(np.max(ratio)),
    }


class PPOBufferStorage:
    """Backend view of a canonical host PPO buffer."""

    def __init__(self, network, buffer):
        self.network = network
        self.buffer = buffer
        self.location = "ram_streamed" if network.device == "gpu" else "ram"
        self._device_arrays = None
        self.preflight = {
            "requested_gpu": bool(
                PPO_PREFER_GPU_BUFFER and network.device == "gpu"
            ),
            "reported_free_vram_bytes": None,
            "usable_free_vram_bytes": None,
            "buffer_bytes": int(buffer.nbytes),
            "fallback_reason": None,
        }
        if PPO_PREFER_GPU_BUFFER and network.device == "gpu":
            self._try_full_gpu_copy()

    def _try_full_gpu_copy(self):
        free_bytes = effective_gpu_available_bytes()
        usable = (
            None
            if free_bytes is None
            else int(free_bytes * PPO_GPU_BUFFER_SAFETY_FRACTION)
        )
        self.preflight["reported_free_vram_bytes"] = free_bytes
        self.preflight["usable_free_vram_bytes"] = usable
        if usable is None or self.buffer.nbytes > usable:
            self.location = "ram_streamed"
            self.preflight["fallback_reason"] = (
                "VRAM preflight could not reserve the complete PPO buffer and workspace"
            )
            return
        xp = self.network.xp
        try:
            self._device_arrays = {
                "states": xp.asarray(self.buffer.states, dtype=xp.float32),
                "actions": xp.asarray(self.buffer.actions, dtype=xp.int64),
                "legal_masks": xp.asarray(self.buffer.legal_masks, dtype=xp.bool_),
                "old_log_probs": xp.asarray(self.buffer.old_log_probs, dtype=xp.float32),
                "advantages": xp.asarray(self.buffer.advantages, dtype=xp.float32),
                "returns": xp.asarray(self.buffer.returns, dtype=xp.float32),
            }
            if self.buffer.old_values is not None:
                self._device_arrays["old_values"] = xp.asarray(
                    self.buffer.old_values,
                    dtype=xp.float32,
                )
            self.network.synchronize()
            self.location = "gpu"
        except Exception as exc:
            if not self.network._is_backend_memory_error(exc):
                raise
            self._device_arrays = None
            self.network.release_disposable_cache()
            self.location = "ram_streamed"
            self.preflight["fallback_reason"] = f"{type(exc).__name__}: {exc}"

    def batch(self, indices):
        xp = self.network.xp
        indices = np.asarray(indices, dtype=np.int64)
        if self._device_arrays is not None:
            backend_indices = xp.asarray(indices, dtype=xp.int64)
            arrays = self._device_arrays
            batch = {
                "states": arrays["states"][:, backend_indices],
                "actions": arrays["actions"][backend_indices],
                "legal_masks": arrays["legal_masks"][:, backend_indices],
                "old_log_probs": arrays["old_log_probs"][backend_indices],
                "advantages": arrays["advantages"][backend_indices],
                "returns": arrays["returns"][backend_indices],
            }
            batch["old_values"] = (
                None
                if self.buffer.old_values is None
                else arrays["old_values"][backend_indices]
            )
            return batch
        batch = {
            "states": xp.asarray(self.buffer.states[:, indices], dtype=xp.float32),
            "actions": xp.asarray(self.buffer.actions[indices], dtype=xp.int64),
            "legal_masks": xp.asarray(self.buffer.legal_masks[:, indices], dtype=xp.bool_),
            "old_log_probs": xp.asarray(self.buffer.old_log_probs[indices], dtype=xp.float32),
            "advantages": xp.asarray(self.buffer.advantages[indices], dtype=xp.float32),
            "returns": xp.asarray(self.buffer.returns[indices], dtype=xp.float32),
        }
        batch["old_values"] = (
            None
            if self.buffer.old_values is None
            else xp.asarray(self.buffer.old_values[indices], dtype=xp.float32)
        )
        return batch

    def fallback_to_streaming(self, reason):
        """Discard only the optional GPU copy; the canonical RAM buffer survives."""
        self._device_arrays = None
        self.network.release_disposable_cache()
        self.location = "ram_streamed" if self.network.device == "gpu" else "ram"
        self.preflight["fallback_reason"] = str(reason)

    def close(self):
        self._device_arrays = None
        self.network.release_disposable_cache()


def _validate_first_minibatch(network, storage, indices):
    batch = storage.batch(indices)
    log_probs, entropy, _policy = network.evaluate_actions(
        batch["states"],
        batch["legal_masks"],
        batch["actions"],
    )
    xp = network.xp
    finite = xp.all(xp.isfinite(log_probs)) & xp.all(xp.isfinite(entropy))
    if not bool(network._as_float(finite)):
        raise FloatingPointError("PPO first-minibatch validation produced NaN/Inf.")
    # Probe the large simultaneous arrays needed by manual backpropagation
    # while the complete GPU buffer is still present. This is deliberately a
    # dry allocation: no weight or optimizer state changes before the real
    # first minibatch, so a GPU OOM can safely switch to RAM streaming.
    xp = network.xp
    workspace = [
        xp.empty_like(network.cache[f"Z{index}"])
        for index in range(1, network.layer_count + 1)
    ]
    workspace.extend(
        xp.empty_like(network.cache[f"A{index}"])
        for index in range(1, network.layer_count)
    )
    parameter_names = list(network.weight_names)
    if getattr(network, "use_value_head", False):
        parameter_names.extend(("Wv", "bv"))
        workspace.append(xp.empty((1, batch["states"].shape[1]), dtype=xp.float32))
    workspace.extend(
        xp.empty_like(getattr(network, name))
        for name in parameter_names
    )
    network.synchronize()
    del workspace
    network.release_disposable_cache()


def prepare_storage(
    network,
    buffer,
    first_indices,
):
    """Allocate and workspace-probe storage before the first optimizer step."""
    storage = PPOBufferStorage(network, buffer)
    try:
        _validate_first_minibatch(network, storage, first_indices)
    except Exception as exc:
        if storage.location != "gpu" or not network._is_backend_memory_error(exc):
            storage.close()
            raise
        storage.fallback_to_streaming(f"workspace probe: {type(exc).__name__}: {exc}")
        _validate_first_minibatch(network, storage, first_indices)
    return storage


def _merge_numeric_profile(target, source):
    """Recursively sum numeric profiler leaves."""
    for key, value in source.items():
        if isinstance(value, dict):
            _merge_numeric_profile(target.setdefault(key, {}), value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            target[key] = target.get(key, 0) + value


def evaluate_full_buffer(
    network,
    storage,
    partitions,
    clip_epsilon,
    runtime_profile=None,
):
    """Compute exact whole-buffer PPO metrics, streaming when necessary."""
    profile_started = time.perf_counter()
    timing = {}

    def finish_phase(name, started):
        timing[name] = timing.get(name, 0.0) + (
            time.perf_counter() - started
        )

    total = 0
    surrogate_sum = 0.0
    entropy_sum = 0.0
    kl_sum = 0.0
    clipped_count = 0
    ratio_sum = 0.0
    ratio_min = float("inf")
    ratio_max = float("-inf")
    value_loss_sum = 0.0
    value_sum = 0.0
    value_square_sum = 0.0
    return_sum = 0.0
    return_square_sum = 0.0
    value_error_sum = 0.0
    value_error_square_sum = 0.0
    value_clipped_count = 0
    lower = 1.0 - float(clip_epsilon)
    upper = 1.0 + float(clip_epsilon)
    xp = network.xp
    for indices in partitions:
        phase_started = time.perf_counter()
        batch = storage.batch(indices)
        finish_phase("partition_batch_materialization", phase_started)
        phase_started = time.perf_counter()
        new_log_probs, entropy, _policy = network.evaluate_actions(
            batch["states"], batch["legal_masks"], batch["actions"]
        )
        values = None
        value_losses = None
        value_delta = None
        if getattr(network, "use_value_head", False):
            values = xp.dot(
                network.Wv,
                network.cache[network.last_hidden_activation_key],
            ) + network.bv
            value_losses, _value_gradient, value_delta = (
                network.clipped_value_loss_terms(
                    values,
                    batch["returns"].reshape(1, -1),
                    batch["old_values"].reshape(1, -1),
                    clip_epsilon,
                )
            )
        finish_phase("policy_forward_and_action_mask_validation", phase_started)
        phase_started = time.perf_counter()
        log_ratio = new_log_probs - batch["old_log_probs"]
        ratio = xp.exp(log_ratio)
        finite = xp.all(xp.isfinite(ratio)) & xp.all(xp.isfinite(entropy))
        if values is not None:
            finite = finite & xp.all(xp.isfinite(values)) & xp.all(
                xp.isfinite(value_losses)
            )
        if not bool(network._as_float(finite)):
            raise FloatingPointError("PPO full-buffer metrics produced NaN/Inf.")
        finish_phase("ratio_construction_and_finite_validation", phase_started)
        phase_started = time.perf_counter()
        clipped_ratio = xp.clip(ratio, lower, upper)
        surrogate = xp.minimum(
            ratio * batch["advantages"],
            clipped_ratio * batch["advantages"],
        )
        count = int(len(indices))
        total += count
        surrogate_sum += network._as_float(xp.sum(surrogate))
        entropy_sum += network._as_float(xp.sum(entropy))
        kl_sum += network._as_float(xp.sum((ratio - 1.0) - log_ratio))
        clipped_count += int(network._as_float(xp.sum((ratio < lower) | (ratio > upper))))
        ratio_sum += network._as_float(xp.sum(ratio))
        ratio_min = min(ratio_min, network._as_float(xp.min(ratio)))
        ratio_max = max(ratio_max, network._as_float(xp.max(ratio)))
        if values is not None:
            flat_values = values.reshape(-1)
            returns = batch["returns"].reshape(-1)
            errors = returns - flat_values
            value_loss_sum += network._as_float(xp.sum(value_losses))
            value_sum += network._as_float(xp.sum(flat_values))
            value_square_sum += network._as_float(xp.sum(flat_values ** 2))
            return_sum += network._as_float(xp.sum(returns))
            return_square_sum += network._as_float(xp.sum(returns ** 2))
            value_error_sum += network._as_float(xp.sum(errors))
            value_error_square_sum += network._as_float(xp.sum(errors ** 2))
            value_clipped_count += int(network._as_float(
                xp.sum(xp.abs(value_delta) > float(clip_epsilon))
            ))
        finish_phase("surrogate_metric_reductions_and_host_transfers", phase_started)
    phase_started = time.perf_counter()
    network.synchronize()
    if total != storage.buffer.size:
        raise AssertionError("Whole-buffer PPO metrics did not visit every decision.")
    result = {
        "policy_loss": float(-surrogate_sum / total),
        "entropy": float(entropy_sum / total),
        "approx_kl": max(0.0, float(kl_sum / total)),
        "clip_fraction": float(clipped_count / total),
        "ratio_mean": float(ratio_sum / total),
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
        "value_loss": None,
        "value_clip_fraction": None,
        "value_mean": None,
        "value_std": None,
        "explained_variance": None,
    }
    if getattr(network, "use_value_head", False):
        value_mean = value_sum / total
        value_variance = max(0.0, value_square_sum / total - value_mean ** 2)
        return_mean = return_sum / total
        return_variance = max(
            0.0,
            return_square_sum / total - return_mean ** 2,
        )
        error_mean = value_error_sum / total
        error_variance = max(
            0.0,
            value_error_square_sum / total - error_mean ** 2,
        )
        result.update({
            "value_loss": float(value_loss_sum / total),
            "value_clip_fraction": float(value_clipped_count / total),
            "value_mean": float(value_mean),
            "value_std": float(math.sqrt(value_variance)),
            "explained_variance": (
                None
                if return_variance <= ADVANTAGE_EPSILON
                else float(1.0 - error_variance / return_variance)
            ),
        })
    finish_phase("final_device_synchronization_and_accounting", phase_started)
    total_seconds = time.perf_counter() - profile_started
    timing["unaccounted"] = max(0.0, total_seconds - sum(timing.values()))
    if runtime_profile is not None:
        _merge_numeric_profile(runtime_profile, {
            "calls": 1,
            "execution_seconds": float(total_seconds),
            "gpu_calls": int(network.device == "gpu"),
            "cpu_calls": int(network.device == "cpu"),
            "gpu_buffer_calls": int(storage.location == "gpu"),
            "ram_buffer_calls": int(storage.location != "gpu"),
            "sections_seconds": timing,
        })
    return result


def ppo_update(
    network,
    buffer,
    *,
    actual_games,
    base_seed,
    iteration,
    entropy_coef,
    value_coef=0.5,
    max_epochs=DEFAULT_PPO_MAX_EPOCHS,
):
    """Run deterministic, KL-limited PPO epochs over one on-policy buffer."""
    profile_started = time.perf_counter()
    timing = {
        "setup_and_first_partition": 0.0,
        "storage_prepare": 0.0,
        "later_epoch_partitioning": 0.0,
        "minibatch_materialization": 0.0,
        "optimizer_steps": 0.0,
        "optimizer_synchronization": 0.0,
        "full_buffer_evaluation": 0.0,
        "epoch_metrics_and_kl_control": 0.0,
        "storage_cleanup": 0.0,
    }
    optimizer_step_detail = {}
    full_buffer_detail = {}
    max_epochs = validate_ppo_max_epochs(max_epochs)
    if not ppo_is_enabled(max_epochs):
        raise ValueError("ppo_update requires ppo_max_epochs of at least 2")
    if float(value_coef) < 0:
        raise ValueError("PPO value_coef must be non-negative.")
    use_value_head = bool(getattr(network, "use_value_head", False))
    if use_value_head and buffer.old_values is None:
        raise ValueError(
            "PPO old_values are required when the value head is enabled."
        )
    if not use_value_head and buffer.old_values is not None:
        raise ValueError(
            "PPO old_values were provided, but the value head is disabled."
        )

    requested = requested_minibatches(actual_games)
    effective = effective_minibatches(buffer.size, requested)
    first_partitions = minibatch_indices(
        buffer.size,
        effective,
        stable_seed(base_seed, "ppo_shuffle", iteration, 0),
    )
    timing["setup_and_first_partition"] = time.perf_counter() - profile_started
    storage_started = time.perf_counter()
    storage = prepare_storage(
        network,
        buffer,
        first_partitions[0],
    )
    timing["storage_prepare"] = time.perf_counter() - storage_started
    epoch_rows = []
    stopped_by_kl = False
    optimizer_steps = 0
    result = None
    try:
        for epoch in range(int(max_epochs)):
            if epoch == 0:
                partitions = first_partitions
            else:
                partition_started = time.perf_counter()
                partitions = minibatch_indices(
                    buffer.size,
                    effective,
                    stable_seed(base_seed, "ppo_shuffle", iteration, epoch),
                )
                timing["later_epoch_partitioning"] += (
                    time.perf_counter() - partition_started
                )
            batch_grad_norms = []
            batch_applied_grad_norms = []
            batch_clipped = 0
            step_before_epoch = int(getattr(network, "optimizer_step_count", 0))
            for indices in partitions:
                materialization_started = time.perf_counter()
                batch = storage.batch(indices)
                timing["minibatch_materialization"] += (
                    time.perf_counter() - materialization_started
                )
                optimizer_started = time.perf_counter()
                step_metrics = network.backward_ppo(
                    batch["states"],
                    batch["actions"],
                    batch["legal_masks"],
                    batch["old_log_probs"],
                    batch["advantages"],
                    returns=(batch["returns"] if use_value_head else None),
                    old_values=(
                        batch["old_values"] if use_value_head else None
                    ),
                    value_coef=value_coef,
                    clip_epsilon=PPO_CLIP_EPSILON,
                    entropy_coef=entropy_coef,
                    clip_grad_norm=POLICY_GRADIENT_CLIP_NORM,
                )
                timing["optimizer_steps"] += time.perf_counter() - optimizer_started
                step_detail = step_metrics.pop("runtime_profile_detail", {})
                _merge_numeric_profile(optimizer_step_detail, step_detail)
                batch_grad_norms.append(float(step_metrics["grad_norm"]))
                batch_applied_grad_norms.append(
                    float(step_metrics["applied_grad_norm"])
                )
                batch_clipped += int(step_metrics["grad_clipped"])
            synchronization_started = time.perf_counter()
            network.synchronize()
            timing["optimizer_synchronization"] += (
                time.perf_counter() - synchronization_started
            )
            steps_this_epoch = (
                int(getattr(network, "optimizer_step_count", 0))
                - step_before_epoch
            )
            if steps_this_epoch != len(partitions):
                raise AssertionError("PPO optimizer-step count does not match minibatches.")
            optimizer_steps += steps_this_epoch
            evaluation_started = time.perf_counter()
            whole = evaluate_full_buffer(
                network,
                storage,
                partitions,
                PPO_CLIP_EPSILON,
                runtime_profile=full_buffer_detail,
            )
            timing["full_buffer_evaluation"] += (
                time.perf_counter() - evaluation_started
            )
            metrics_started = time.perf_counter()
            row = {
                "epoch": epoch + 1,
                **whole,
                "gradient_norm_mean": float(np.mean(batch_grad_norms)),
                "gradient_norm_max": float(np.max(batch_grad_norms)),
                "applied_gradient_norm_mean": float(
                    np.mean(batch_applied_grad_norms)
                ),
                "clipped_gradient_minibatches": int(batch_clipped),
                "optimizer_steps": int(steps_this_epoch),
                "decisions": int(buffer.size),
                "minibatches": int(len(partitions)),
                "minibatch_sizes": [int(len(indices)) for indices in partitions],
            }
            epoch_rows.append(row)
            if whole["approx_kl"] > PPO_STOP_KL:
                stopped_by_kl = True
                timing["epoch_metrics_and_kl_control"] += (
                    time.perf_counter() - metrics_started
                )
                break
            timing["epoch_metrics_and_kl_control"] += (
                time.perf_counter() - metrics_started
            )
        final = epoch_rows[-1]
        result = {
            "requested_minibatches": int(requested),
            "effective_minibatches": int(effective),
            "minibatch_sizes": list(epoch_rows[0]["minibatch_sizes"]),
            "epochs_completed": int(len(epoch_rows)),
            "stopped_by_kl": bool(stopped_by_kl),
            "optimizer_steps": int(optimizer_steps),
            "final_approx_kl": float(final["approx_kl"]),
            "max_approx_kl": float(max(row["approx_kl"] for row in epoch_rows)),
            "final_clip_fraction": float(final["clip_fraction"]),
            "final_entropy": float(final["entropy"]),
            "final_policy_loss": float(final["policy_loss"]),
            "final_value_loss": (
                None
                if final["value_loss"] is None
                else float(final["value_loss"])
            ),
            "final_value_clip_fraction": (
                None
                if final["value_clip_fraction"] is None
                else float(final["value_clip_fraction"])
            ),
            "final_value_mean": (
                None
                if final["value_mean"] is None
                else float(final["value_mean"])
            ),
            "final_value_std": (
                None
                if final["value_std"] is None
                else float(final["value_std"])
            ),
            "final_explained_variance": (
                None
                if final["explained_variance"] is None
                else float(final["explained_variance"])
            ),
            "gradient_norm_mean": float(np.mean([
                row["gradient_norm_mean"] for row in epoch_rows
            ])),
            "gradient_norm_max": float(max(
                row["gradient_norm_max"] for row in epoch_rows
            )),
            "buffer_location": storage.location,
            "buffer_bytes": int(buffer.nbytes),
            "buffer_preflight": dict(storage.preflight),
            "advantage_std_zero": bool(buffer.advantage_std_zero),
            "raw_advantage_mean": float(buffer.raw_advantage_mean),
            "raw_advantage_std": float(buffer.raw_advantage_std),
            "target_kl": PPO_TARGET_KL,
            "stop_kl": PPO_STOP_KL,
            "clip_epsilon": PPO_CLIP_EPSILON,
            "epoch_metrics": epoch_rows,
            # Compatibility with the old one-update metric keys.
            "entropy": float(final["entropy"]),
            "grad_norm": float(max(
                row["gradient_norm_max"] for row in epoch_rows
            )),
            "applied_grad_norm": float(max(
                row["applied_gradient_norm_mean"] for row in epoch_rows
            )),
            "grad_clipped": bool(any(
                row["clipped_gradient_minibatches"] for row in epoch_rows
            )),
            "value_loss": (
                None
                if final["value_loss"] is None
                else float(final["value_loss"])
            ),
        }
    finally:
        cleanup_started = time.perf_counter()
        storage.close()
        timing["storage_cleanup"] += time.perf_counter() - cleanup_started

    total_seconds = time.perf_counter() - profile_started
    accounted_seconds = sum(timing.values())
    timing["unaccounted"] = max(0.0, total_seconds - accounted_seconds)
    result["runtime_timing_seconds"] = {
        "total": float(total_seconds),
        **{key: float(value) for key, value in timing.items()},
    }
    optimizer_step_detail["gpu_calls"] = int(
        optimizer_step_detail.get("gpu_calls", 0)
    )
    optimizer_step_detail["cpu_calls"] = int(
        optimizer_step_detail.get("cpu_calls", 0)
    )
    result["runtime_profile_detail"] = {
        "optimizer_step": optimizer_step_detail,
        "full_buffer_evaluation": full_buffer_detail,
    }
    return result


def _value_prediction_summary(values):
    """Summarize pre-update value predictions without another forward pass."""
    host_values = values.get() if hasattr(values, "get") else values
    flattened = np.asarray(host_values, dtype=np.float64).reshape(-1)
    return {
        "sample_count": int(flattened.size),
        "mean": float(np.mean(flattened)),
        "std": float(np.std(flattened)),
        "min": float(np.min(flattened)),
        "max": float(np.max(flattened)),
    }


def update_from_samples(
    network,
    samples,
    *,
    actual_games,
    base_seed,
    iteration,
    entropy_coef,
    value_coef,
    normalize_advantages,
    max_epochs,
    collect_value_predictions=False,
):
    """Build the immutable PPO buffer and update one on-policy iteration."""
    if not ppo_is_enabled(max_epochs):
        raise ValueError("PPO sample updates require ppo_max_epochs of at least 2")
    buffer_started = time.perf_counter()
    old_values = None
    value_predictions = None
    if getattr(network, "use_value_head", False):
        value_states = network.xp.hstack([
            network.xp.asarray(sample.x, dtype=network.xp.float32)
            for sample in samples
        ])
        backend_old_values = network.predict_values(value_states)
        if collect_value_predictions:
            value_predictions = _value_prediction_summary(backend_old_values)
        old_values = np.ascontiguousarray(
            (
                backend_old_values.get()
                if hasattr(backend_old_values, "get")
                else backend_old_values
            ),
            dtype=np.float32,
        ).reshape(-1)
    decision_buffer = PPOBuffer.from_samples(
        samples,
        normalize=normalize_advantages,
        old_values=old_values,
    )
    buffer_seconds = time.perf_counter() - buffer_started

    update_started = time.perf_counter()
    metrics = ppo_update(
        network,
        decision_buffer,
        actual_games=actual_games,
        base_seed=base_seed,
        iteration=iteration,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_epochs=max_epochs,
    )
    network.synchronize()
    update_seconds = time.perf_counter() - update_started
    if value_predictions is not None:
        metrics["value_predictions_before_update"] = value_predictions
    return PPOIterationUpdate(
        metrics=metrics,
        buffer_seconds=float(buffer_seconds),
        update_seconds=float(update_seconds),
    )
