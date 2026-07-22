"""Masked PPO minibatch updates for the domino self-play policy.

The canonical decision buffer always lives in host RAM.  A GPU-backed learner
may additionally keep a complete device copy when conservative VRAM preflight
and a real first-minibatch workspace probe both succeed.  Otherwise the same
immutable host buffer is streamed one minibatch at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

import numpy as np

from utils.resource_limits import effective_gpu_available_bytes


DEFAULT_CLIP_EPSILON = 0.2
DEFAULT_TARGET_KL = 0.01
DEFAULT_STOP_KL = 0.015
DEFAULT_MAX_EPOCHS = 4
DEFAULT_MIN_MINIBATCHES = 4
DEFAULT_MAX_MINIBATCHES = 16
DEFAULT_GAMES_PER_MINIBATCH_SCALE = 125
DEFAULT_MIN_DECISIONS_PER_MINIBATCH = 128
DEFAULT_GPU_BUFFER_SAFETY_FRACTION = 0.70
ADVANTAGE_EPSILON = 1e-8


def stable_seed(base_seed: int, *parts: object) -> int:
    """Return a process-independent 64-bit seed for a labeled operation."""
    digest = hashlib.sha256()
    digest.update(str(int(base_seed)).encode("ascii"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "little", signed=False)


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
    advantages: np.ndarray
    returns: np.ndarray
    local_rewards: np.ndarray
    terminal_rewards: np.ndarray
    advantage_std_zero: bool
    raw_advantage_mean: float
    raw_advantage_std: float

    @classmethod
    def from_samples(cls, samples: Iterable, *, normalize=True):
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
        if normalize:
            advantages, std_zero, raw_mean, raw_std = normalize_advantages(returns)
        else:
            advantages = returns.copy()
            std_zero = False
            raw_mean = float(np.mean(returns, dtype=np.float64))
            raw_std = float(np.std(returns, dtype=np.float64))
        buffer = cls(
            states=states,
            actions=actions,
            legal_masks=legal_masks,
            old_log_probs=old_log_probs,
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
        return buffer

    @property
    def size(self):
        return int(self.actions.size)

    @property
    def nbytes(self):
        return int(sum(
            array.nbytes
            for array in (
                self.states,
                self.actions,
                self.legal_masks,
                self.old_log_probs,
                self.advantages,
                self.returns,
                self.local_rewards,
                self.terminal_rewards,
            )
        ))


def requested_minibatches(
    actual_games,
    *,
    minimum=DEFAULT_MIN_MINIBATCHES,
    maximum=DEFAULT_MAX_MINIBATCHES,
    games_scale=DEFAULT_GAMES_PER_MINIBATCH_SCALE,
):
    """Return the requested 4..16 minibatch count based on games collected."""
    if actual_games < 1 or games_scale < 1:
        raise ValueError("actual_games and games_scale must be positive.")
    if minimum < 1 or maximum < minimum:
        raise ValueError("Invalid PPO minibatch bounds.")
    raw = math.ceil(int(actual_games) / int(games_scale))
    return max(int(minimum), min(int(maximum), raw))


def effective_minibatches(
    decision_count,
    requested,
    *,
    minimum_decisions=DEFAULT_MIN_DECISIONS_PER_MINIBATCH,
):
    """Cap minibatches so every non-final slice remains operationally useful."""
    if decision_count < 1 or requested < 1 or minimum_decisions < 1:
        raise ValueError("Decision/minibatch counts must be positive.")
    maximum_useful = max(1, int(decision_count) // int(minimum_decisions))
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


def clipped_surrogate(ratios, advantages, clip_epsilon=DEFAULT_CLIP_EPSILON):
    """Return NumPy PPO surrogate terms, exposed for exact unit tests."""
    ratios = np.asarray(ratios, dtype=np.float64)
    advantages = np.asarray(advantages, dtype=np.float64)
    clipped = np.clip(
        ratios,
        1.0 - float(clip_epsilon),
        1.0 + float(clip_epsilon),
    )
    return np.minimum(ratios * advantages, clipped * advantages)


def log_ratio_statistics(new_log_probs, old_log_probs, clip_epsilon=DEFAULT_CLIP_EPSILON):
    """Return whole-buffer PPO ratio, KL, and clipping statistics."""
    new = np.asarray(new_log_probs, dtype=np.float64)
    old = np.asarray(old_log_probs, dtype=np.float64)
    log_ratio = new - old
    ratio = np.exp(log_ratio)
    if not np.all(np.isfinite(ratio)):
        raise FloatingPointError("PPO probability ratio contains NaN or infinity.")
    approx_kl = float(np.mean((ratio - 1.0) - log_ratio))
    lower = 1.0 - float(clip_epsilon)
    upper = 1.0 + float(clip_epsilon)
    return {
        "approx_kl": max(0.0, approx_kl),
        "clip_fraction": float(np.mean((ratio < lower) | (ratio > upper))),
        "ratio_mean": float(np.mean(ratio)),
        "ratio_min": float(np.min(ratio)),
        "ratio_max": float(np.max(ratio)),
    }


class PPOBufferStorage:
    """Backend view of a canonical host PPO buffer."""

    def __init__(self, network, buffer, *, prefer_gpu=True, safety_fraction=0.70):
        self.network = network
        self.buffer = buffer
        self.location = "ram_streamed" if network.device == "gpu" else "ram"
        self._device_arrays = None
        self.preflight = {
            "requested_gpu": bool(prefer_gpu and network.device == "gpu"),
            "reported_free_vram_bytes": None,
            "usable_free_vram_bytes": None,
            "buffer_bytes": int(buffer.nbytes),
            "fallback_reason": None,
        }
        if prefer_gpu and network.device == "gpu":
            self._try_full_gpu_copy(safety_fraction)

    def _try_full_gpu_copy(self, safety_fraction):
        if not 0 < float(safety_fraction) <= 1:
            raise ValueError("GPU buffer safety fraction must be in (0, 1].")
        free_bytes = effective_gpu_available_bytes()
        usable = None if free_bytes is None else int(free_bytes * float(safety_fraction))
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
            return {
                "states": arrays["states"][:, backend_indices],
                "actions": arrays["actions"][backend_indices],
                "legal_masks": arrays["legal_masks"][:, backend_indices],
                "old_log_probs": arrays["old_log_probs"][backend_indices],
                "advantages": arrays["advantages"][backend_indices],
                "returns": arrays["returns"][backend_indices],
            }
        return {
            "states": xp.asarray(self.buffer.states[:, indices], dtype=xp.float32),
            "actions": xp.asarray(self.buffer.actions[indices], dtype=xp.int64),
            "legal_masks": xp.asarray(self.buffer.legal_masks[:, indices], dtype=xp.bool_),
            "old_log_probs": xp.asarray(self.buffer.old_log_probs[indices], dtype=xp.float32),
            "advantages": xp.asarray(self.buffer.advantages[indices], dtype=xp.float32),
            "returns": xp.asarray(self.buffer.returns[indices], dtype=xp.float32),
        }

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
        xp.empty_like(network.cache["Z3"]),
        xp.empty_like(network.cache["A2"]),
        xp.empty_like(network.cache["Z2"]),
        xp.empty_like(network.cache["A1"]),
        xp.empty_like(network.cache["Z1"]),
    ]
    workspace.extend(
        xp.empty_like(getattr(network, name))
        for name in ("W1", "b1", "W2", "b2", "W3", "b3")
    )
    network.synchronize()
    del workspace
    network.release_disposable_cache()


def prepare_storage(
    network,
    buffer,
    first_indices,
    *,
    prefer_gpu=True,
    safety_fraction=DEFAULT_GPU_BUFFER_SAFETY_FRACTION,
):
    """Allocate and workspace-probe storage before the first optimizer step."""
    storage = PPOBufferStorage(
        network,
        buffer,
        prefer_gpu=prefer_gpu,
        safety_fraction=safety_fraction,
    )
    try:
        _validate_first_minibatch(network, storage, first_indices)
    except Exception as exc:
        if storage.location != "gpu" or not network._is_backend_memory_error(exc):
            storage.close()
            raise
        storage.fallback_to_streaming(f"workspace probe: {type(exc).__name__}: {exc}")
        _validate_first_minibatch(network, storage, first_indices)
    return storage


def evaluate_full_buffer(network, storage, partitions, clip_epsilon):
    """Compute exact whole-buffer PPO metrics, streaming when necessary."""
    total = 0
    surrogate_sum = 0.0
    entropy_sum = 0.0
    kl_sum = 0.0
    clipped_count = 0
    ratio_sum = 0.0
    ratio_min = float("inf")
    ratio_max = float("-inf")
    lower = 1.0 - float(clip_epsilon)
    upper = 1.0 + float(clip_epsilon)
    xp = network.xp
    for indices in partitions:
        batch = storage.batch(indices)
        new_log_probs, entropy, _policy = network.evaluate_actions(
            batch["states"], batch["legal_masks"], batch["actions"]
        )
        log_ratio = new_log_probs - batch["old_log_probs"]
        ratio = xp.exp(log_ratio)
        finite = xp.all(xp.isfinite(ratio)) & xp.all(xp.isfinite(entropy))
        if not bool(network._as_float(finite)):
            raise FloatingPointError("PPO full-buffer metrics produced NaN/Inf.")
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
    network.synchronize()
    if total != storage.buffer.size:
        raise AssertionError("Whole-buffer PPO metrics did not visit every decision.")
    return {
        "policy_loss": float(-surrogate_sum / total),
        "entropy": float(entropy_sum / total),
        "approx_kl": max(0.0, float(kl_sum / total)),
        "clip_fraction": float(clipped_count / total),
        "ratio_mean": float(ratio_sum / total),
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
    }


def ppo_update(
    network,
    buffer,
    *,
    actual_games,
    base_seed,
    iteration,
    entropy_coef,
    clip_grad_norm,
    clip_epsilon=DEFAULT_CLIP_EPSILON,
    target_kl=DEFAULT_TARGET_KL,
    stop_kl=DEFAULT_STOP_KL,
    max_epochs=DEFAULT_MAX_EPOCHS,
    min_minibatches=DEFAULT_MIN_MINIBATCHES,
    max_minibatches=DEFAULT_MAX_MINIBATCHES,
    games_per_minibatch_scale=DEFAULT_GAMES_PER_MINIBATCH_SCALE,
    min_decisions_per_minibatch=DEFAULT_MIN_DECISIONS_PER_MINIBATCH,
    prefer_gpu_buffer=True,
    gpu_buffer_safety_fraction=DEFAULT_GPU_BUFFER_SAFETY_FRACTION,
):
    """Run up to four deterministic PPO epochs over one on-policy buffer."""
    if not 0 < float(clip_epsilon) < 1:
        raise ValueError("PPO clip_epsilon must be in (0, 1).")
    if target_kl <= 0 or stop_kl <= 0 or stop_kl < target_kl:
        raise ValueError("PPO KL thresholds must satisfy 0 < target_kl <= stop_kl.")
    if not 1 <= int(max_epochs) <= 4:
        raise ValueError("PPO max_epochs must be between one and four.")
    if not 0 < float(gpu_buffer_safety_fraction) <= 1:
        raise ValueError("GPU buffer safety fraction must be in (0, 1].")

    requested = requested_minibatches(
        actual_games,
        minimum=min_minibatches,
        maximum=max_minibatches,
        games_scale=games_per_minibatch_scale,
    )
    effective = effective_minibatches(
        buffer.size,
        requested,
        minimum_decisions=min_decisions_per_minibatch,
    )
    first_partitions = minibatch_indices(
        buffer.size,
        effective,
        stable_seed(base_seed, "ppo_shuffle", iteration, 0),
    )
    storage = prepare_storage(
        network,
        buffer,
        first_partitions[0],
        prefer_gpu=prefer_gpu_buffer,
        safety_fraction=gpu_buffer_safety_fraction,
    )
    epoch_rows = []
    stopped_by_kl = False
    optimizer_steps = 0
    try:
        for epoch in range(int(max_epochs)):
            partitions = (
                first_partitions
                if epoch == 0
                else minibatch_indices(
                    buffer.size,
                    effective,
                    stable_seed(base_seed, "ppo_shuffle", iteration, epoch),
                )
            )
            batch_grad_norms = []
            batch_applied_grad_norms = []
            batch_clipped = 0
            step_before_epoch = int(getattr(network, "optimizer_step_count", 0))
            for indices in partitions:
                batch = storage.batch(indices)
                step_metrics = network.backward_ppo(
                    batch["states"],
                    batch["actions"],
                    batch["legal_masks"],
                    batch["old_log_probs"],
                    batch["advantages"],
                    clip_epsilon=clip_epsilon,
                    entropy_coef=entropy_coef,
                    clip_grad_norm=clip_grad_norm,
                )
                batch_grad_norms.append(float(step_metrics["grad_norm"]))
                batch_applied_grad_norms.append(
                    float(step_metrics["applied_grad_norm"])
                )
                batch_clipped += int(step_metrics["grad_clipped"])
            network.synchronize()
            steps_this_epoch = (
                int(getattr(network, "optimizer_step_count", 0))
                - step_before_epoch
            )
            if steps_this_epoch != len(partitions):
                raise AssertionError("PPO optimizer-step count does not match minibatches.")
            optimizer_steps += steps_this_epoch
            whole = evaluate_full_buffer(
                network,
                storage,
                partitions,
                clip_epsilon,
            )
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
            if whole["approx_kl"] > float(stop_kl):
                stopped_by_kl = True
                break
        final = epoch_rows[-1]
        return {
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
            "target_kl": float(target_kl),
            "stop_kl": float(stop_kl),
            "clip_epsilon": float(clip_epsilon),
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
            "value_loss": None,
        }
    finally:
        storage.close()
