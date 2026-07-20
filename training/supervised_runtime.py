"""Retained batch tuning and bounded supervised dataset residency."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass

import numpy as np

from utils.resource_limits import (
    MIB,
    MemorySafetyError,
    effective_gpu_available_bytes,
    gpu_allocation_status,
    host_allocation_status,
    process_rss_bytes,
    system_memory_info,
)


DEFAULT_SUPERVISED_CPU_BATCH_SIZE = 1024
DEFAULT_SUPERVISED_GPU_BATCH_SIZE = 2048
SUPERVISED_BATCH_AUTOTUNE_EPOCHS = 10
SUPERVISED_BATCH_AUTOTUNE_MINIMUM_GAIN = 0.10
MAX_SUPERVISED_BATCH_SIZE = 1024 * 1024
SUPERVISED_GPU_MEMORY_RESERVE_MB = 512

SUPERVISED_CPU_BATCH_CANDIDATES = tuple(
    2 ** exponent for exponent in range(10, 21)
)
SUPERVISED_GPU_BATCH_CANDIDATES = tuple(
    2 ** exponent for exponent in range(11, 21)
)
SUPERVISED_GPU_RESIDENCY_CANDIDATES = SUPERVISED_GPU_BATCH_CANDIDATES


def effective_batch_candidate_pairs(device, training_examples):
    """Return unique ``(requested, effective)`` batch-size candidates."""
    if training_examples < 1:
        raise ValueError("training_examples must be positive")
    requested = (
        SUPERVISED_GPU_BATCH_CANDIDATES
        if device == "gpu"
        else SUPERVISED_CPU_BATCH_CANDIDATES
    )
    candidates = []
    effective_sizes = set()
    for requested_size in requested:
        effective = min(int(requested_size), int(training_examples))
        if effective not in effective_sizes:
            candidates.append((int(requested_size), effective))
            effective_sizes.add(effective)
        if effective == training_examples:
            break
    return tuple(candidates)


def effective_batch_candidates(device, training_examples):
    """Return unique effective candidates capped at the training set."""
    return tuple(
        effective
        for _requested, effective in effective_batch_candidate_pairs(
            device,
            training_examples,
        )
    )


def effective_residency_candidates(total_examples):
    """Return unique GPU-residency probe sizes capped at the full dataset."""
    if total_examples < 1:
        raise ValueError("total_examples must be positive")
    candidates = []
    for size in SUPERVISED_GPU_RESIDENCY_CANDIDATES:
        effective = min(int(size), int(total_examples))
        if effective not in candidates:
            candidates.append(effective)
        if effective == total_examples:
            break
    return tuple(candidates)


def estimate_supervised_workspace_bytes(
    batch_size,
    input_size,
    hidden1_size,
    hidden2_size,
    output_size,
):
    """Conservatively estimate one float32 forward/backward live workspace."""
    # Per-example tensors include inputs/labels, all forward activations,
    # softmax temporaries, backward deltas, and fancy-index materialization.
    per_example_floats = (
        input_size
        + output_size
        + 2 * hidden1_size
        + 2 * hidden2_size
        + 2 * output_size
        + output_size
        + 2 * hidden2_size
        + 2 * hidden1_size
        + input_size
        + output_size
    )
    parameter_floats = (
        hidden1_size * input_size
        + hidden1_size
        + hidden2_size * hidden1_size
        + hidden2_size
        + output_size * hidden2_size
        + output_size
    )
    # Two parameter-sized buffers cover gradients and matrix-library
    # temporaries; 1.25x covers allocator rounding and transient expressions.
    raw_bytes = (
        int(batch_size) * per_example_floats * np.dtype(np.float32).itemsize
        + 2 * parameter_floats * np.dtype(np.float32).itemsize
        + int(batch_size) * np.dtype(np.int64).itemsize
    )
    return int(math.ceil(raw_bytes * 1.25))


class RetainedBatchAutotuner:
    """Select a batch size while retaining every completed benchmark epoch."""

    def __init__(
        self,
        *,
        device,
        training_examples,
        total_epochs,
        preflight,
        enabled=True,
        fixed_batch_size=None,
        epochs_per_candidate=SUPERVISED_BATCH_AUTOTUNE_EPOCHS,
        minimum_gain=SUPERVISED_BATCH_AUTOTUNE_MINIMUM_GAIN,
        status_callback=None,
    ):
        if total_epochs < 1:
            raise ValueError("total_epochs must be positive")
        if epochs_per_candidate < 1:
            raise ValueError("epochs_per_candidate must be positive")
        if minimum_gain < 0:
            raise ValueError("minimum_gain must be non-negative")
        self.device = device
        self.training_examples = int(training_examples)
        self.total_epochs = int(total_epochs)
        self.preflight = preflight
        self.epochs_per_candidate = int(epochs_per_candidate)
        self.minimum_gain = float(minimum_gain)
        self.emit = status_callback or (lambda _message: None)
        self.enabled_requested = bool(enabled and fixed_batch_size is None)
        self.attempts = []
        self.autotune_epochs_retained = 0
        self.accepted_examples_per_second = None
        self.candidate_index = 0
        self._durations = []
        self._attempt_start_epoch = 0
        self.finished = not enabled or total_epochs < epochs_per_candidate

        if fixed_batch_size is not None:
            requested_batch_size = int(fixed_batch_size)
            fixed_batch_size = max(
                1,
                min(requested_batch_size, self.training_examples),
            )
            self.candidate_pairs = ((requested_batch_size, fixed_batch_size),)
            self.candidates = (fixed_batch_size,)
            self.current_batch_size = fixed_batch_size
            self.current_requested_batch_size = requested_batch_size
            self.selected_batch_size = fixed_batch_size
            self.finished = True
        else:
            self.candidate_pairs = effective_batch_candidate_pairs(
                device,
                self.training_examples,
            )
            self.candidates = tuple(
                effective for _requested, effective in self.candidate_pairs
            )
            self.current_batch_size = self.candidates[0]
            self.current_requested_batch_size = self.candidate_pairs[0][0]
            self.selected_batch_size = self.current_batch_size

        if not self.finished:
            self.emit(
                "Testing the optimal supervised batch size on "
                f"{self.device.upper()}... each test trains and retains "
                f"{self.epochs_per_candidate} complete epoch(s)."
            )
        self._preflight_or_stop(self.current_batch_size, baseline=True)

    def _memory_attempt(
        self,
        requested_batch_size,
        effective_batch_size,
        memory_result,
        reason,
    ):
        return {
            "requested_batch_size": int(requested_batch_size),
            "effective_batch_size": int(effective_batch_size),
            "epoch_start": None,
            "epoch_end": None,
            "completed_epochs": 0,
            "epoch_durations_s": [],
            "median_epoch_duration_s": None,
            "examples_per_second": None,
            "optimizer_updates_per_second": None,
            "gain_over_accepted": None,
            "accepted": False,
            "memory_result": memory_result,
            "failure_reason": reason,
        }

    def _preflight_or_stop(self, batch_size, *, baseline=False):
        memory_result = dict(self.preflight(batch_size))
        if memory_result.get("safe", False):
            self._active_memory_result = memory_result
            return True
        reason = memory_result.get("reason", "memory preflight rejected candidate")
        self.attempts.append(
            self._memory_attempt(
                self.current_requested_batch_size,
                batch_size,
                memory_result,
                reason,
            )
        )
        if baseline:
            raise MemorySafetyError(
                "No safe initial supervised batch size: " + reason
            )
        self.finished = True
        self.emit(
            f"Supervised batch test with {batch_size:,} stopped before "
            f"training by memory preflight: {reason}."
        )
        self.emit(
            "Optimal supervised batch size: "
            f"{self.selected_batch_size:,}."
        )
        return False

    def record_epoch(self, epoch_index, duration_s):
        """Record one real epoch and advance the retained benchmark."""
        if self.finished:
            return
        self._durations.append(float(duration_s))
        self.autotune_epochs_retained += 1
        if len(self._durations) < self.epochs_per_candidate:
            return

        median_duration = statistics.median(self._durations)
        examples_per_second = self.training_examples / median_duration
        update_count = math.ceil(
            self.training_examples / self.current_batch_size
        )
        optimizer_updates_per_second = update_count / median_duration
        gain = None
        accepted = self.accepted_examples_per_second is None
        if self.accepted_examples_per_second is not None:
            gain = (
                examples_per_second - self.accepted_examples_per_second
            ) / self.accepted_examples_per_second
            accepted = gain >= self.minimum_gain

        attempt = {
            "requested_batch_size": int(self.current_requested_batch_size),
            "effective_batch_size": int(self.current_batch_size),
            "epoch_start": int(self._attempt_start_epoch + 1),
            "epoch_end": int(epoch_index + 1),
            "completed_epochs": self.epochs_per_candidate,
            "epoch_durations_s": list(self._durations),
            "median_epoch_duration_s": float(median_duration),
            "examples_per_second": float(examples_per_second),
            "optimizer_updates_per_second": float(
                optimizer_updates_per_second
            ),
            "gain_over_accepted": gain,
            "accepted": bool(accepted),
            "memory_result": self._active_memory_result,
            "failure_reason": None,
        }
        self.attempts.append(attempt)

        if gain is None:
            self.emit(
                "Supervised batch test with "
                f"{self.current_batch_size:,} passed; median epoch "
                f"{median_duration:.3f}s; test total "
                f"{sum(self._durations):.1f}s; "
                f"{examples_per_second:,.0f} examples/s (baseline); "
                f"{self.epochs_per_candidate} epochs retained."
            )
        else:
            decision = "accepted" if accepted else "rejected"
            self.emit(
                "Supervised batch test with "
                f"{self.current_batch_size:,} passed; median epoch "
                f"{median_duration:.3f}s; test total "
                f"{sum(self._durations):.1f}s; "
                f"{examples_per_second:,.0f} examples/s; "
                f"{gain:+.1%} improvement over the last accepted batch; "
                f"{decision}; {self.epochs_per_candidate} epochs retained."
            )

        if not accepted:
            self.finished = True
            self.current_batch_size = self.selected_batch_size
            self.emit(
                f"Marginal gain is below {self.minimum_gain:.0%}; the current "
                "test remains in supervised training but its batch size will "
                "not be selected."
            )
            self.emit(
                "Optimal supervised batch size: "
                f"{self.selected_batch_size:,}."
            )
            return

        self.selected_batch_size = self.current_batch_size
        self.accepted_examples_per_second = examples_per_second
        next_index = self.candidate_index + 1
        completed = epoch_index + 1
        remaining = self.total_epochs - completed
        if (
            next_index >= len(self.candidates)
            or remaining < self.epochs_per_candidate
        ):
            self.finished = True
            self.emit(
                "Optimal supervised batch size: "
                f"{self.selected_batch_size:,}."
            )
            return

        self.candidate_index = next_index
        self.current_batch_size = self.candidates[next_index]
        self.current_requested_batch_size = self.candidate_pairs[next_index][0]
        self._durations = []
        self._attempt_start_epoch = completed
        if not self._preflight_or_stop(self.current_batch_size):
            self.current_batch_size = self.selected_batch_size
            return

    def handle_runtime_memory_failure(self, epoch_index, exc):
        """Reject an unaccepted live candidate after an allocator failure.

        A conservative preflight should make this path rare. Completed epochs
        and their weight updates are retained; only the failed, incomplete
        epoch is retried with the last accepted batch.
        """
        reason = f"{type(exc).__name__}: {exc}"
        can_retry = (
            self.accepted_examples_per_second is not None
            and self.current_batch_size != self.selected_batch_size
        )
        self.attempts.append({
            "requested_batch_size": int(self.current_requested_batch_size),
            "effective_batch_size": int(self.current_batch_size),
            "epoch_start": int(self._attempt_start_epoch + 1),
            "epoch_end": (
                int(self._attempt_start_epoch + len(self._durations))
                if self._durations
                else None
            ),
            "completed_epochs": len(self._durations),
            "epoch_durations_s": list(self._durations),
            "median_epoch_duration_s": (
                statistics.median(self._durations)
                if self._durations
                else None
            ),
            "examples_per_second": None,
            "optimizer_updates_per_second": None,
            "gain_over_accepted": None,
            "accepted": False,
            "memory_result": self._active_memory_result,
            "failure_reason": reason,
        })
        if not can_retry:
            return False
        self.emit(
            f"Supervised batch {self.current_batch_size:,}: runtime memory "
            "failure before "
            f"completing epoch {epoch_index + 1}; retrying with accepted "
            f"batch {self.selected_batch_size:,}."
        )
        self.current_batch_size = self.selected_batch_size
        accepted_attempt = next(
            attempt
            for attempt in reversed(self.attempts[:-1])
            if attempt.get("accepted") is True
            and attempt.get("effective_batch_size")
            == self.selected_batch_size
        )
        self.current_requested_batch_size = accepted_attempt[
            "requested_batch_size"
        ]
        self._durations = []
        self.finished = True
        return True

    def to_dict(self):
        attempts = list(self.attempts)
        if self._durations and (
            not attempts
            or attempts[-1].get("epoch_end") != self._attempt_start_epoch + len(self._durations)
        ):
            attempts.append({
                "requested_batch_size": int(self.current_requested_batch_size),
                "effective_batch_size": int(self.current_batch_size),
                "epoch_start": int(self._attempt_start_epoch + 1),
                "epoch_end": int(self._attempt_start_epoch + len(self._durations)),
                "completed_epochs": len(self._durations),
                "epoch_durations_s": list(self._durations),
                "median_epoch_duration_s": statistics.median(self._durations),
                "examples_per_second": None,
                "optimizer_updates_per_second": None,
                "gain_over_accepted": None,
                "accepted": None,
                "memory_result": self._active_memory_result,
                "failure_reason": "candidate incomplete when training ended",
            })
        return {
            "enabled": self.enabled_requested,
            "device": self.device,
            "candidate_batch_sizes": list(self.candidates),
            "candidate_requested_batch_sizes": [
                requested for requested, _effective in self.candidate_pairs
            ],
            "epochs_per_candidate": self.epochs_per_candidate,
            "minimum_gain": self.minimum_gain,
            "selected_batch_size": self.selected_batch_size,
            "autotune_epochs_retained": self.autotune_epochs_retained,
            "attempts": attempts,
        }


@dataclass
class GPUResidencyProbe:
    capacity_examples: int
    full_dataset: bool
    attempts: list
    minimum_effective_free_vram_bytes: int | None
    peak_pool_used_bytes: int


def probe_gpu_residency(
    x_host,
    y_host,
    *,
    reserve_mb=SUPERVISED_GPU_MEMORY_RESERVE_MB,
):
    """Probe dataset residency with temporary arrays and no weight updates."""
    import cupy as cp

    total_examples = int(x_host.shape[1])
    feature_count = int(x_host.shape[0] + y_host.shape[0])
    minimum_batch = min(DEFAULT_SUPERVISED_GPU_BATCH_SIZE, total_examples)
    minimum_workspace = estimate_supervised_workspace_bytes(
        minimum_batch,
        x_host.shape[0],
        256,
        128,
        y_host.shape[0],
    )
    attempts = []
    capacity = 0
    minimum_free = None
    peak_pool_used = 0
    pool = cp.get_default_memory_pool()

    for candidate in effective_residency_candidates(total_examples):
        dataset_bytes = (
            candidate * feature_count * np.dtype(np.float32).itemsize
        )
        safe, memory_result = gpu_allocation_status(
            dataset_bytes + minimum_workspace,
            reserve_mb,
        )
        attempt = {
            "examples": int(candidate),
            "estimated_dataset_bytes": int(dataset_bytes),
            "memory_result": memory_result,
            "passed": False,
            "actual_pool_used_bytes": None,
            "failure_reason": None,
        }
        if not safe:
            attempt["failure_reason"] = "VRAM preflight rejected candidate"
            attempts.append(attempt)
            break

        x_probe = None
        y_probe = None
        try:
            x_probe = cp.asarray(
                x_host[:, :candidate],
                dtype=cp.float32,
            )
            y_probe = cp.asarray(
                y_host[:, :candidate],
                dtype=cp.float32,
            )
            cp.cuda.Stream.null.synchronize()
            pool_used = int(pool.used_bytes())
            peak_pool_used = max(peak_pool_used, pool_used)
            available = effective_gpu_available_bytes()
            if available is not None:
                minimum_free = (
                    available
                    if minimum_free is None
                    else min(minimum_free, available)
                )
            post_safe = (
                available is not None
                and available
                >= reserve_mb * MIB + minimum_workspace
            )
            if not post_safe:
                attempt["failure_reason"] = (
                    "actual allocation did not preserve the VRAM reserve and "
                    "minimum training workspace"
                )
                attempts.append(attempt)
                break
            attempt["passed"] = True
            attempt["actual_pool_used_bytes"] = pool_used
            attempts.append(attempt)
            capacity = candidate
        except (
            cp.cuda.memory.OutOfMemoryError,
            cp.cuda.runtime.CUDARuntimeError,
        ) as exc:
            attempt["failure_reason"] = f"{type(exc).__name__}: {exc}"
            attempts.append(attempt)
            break
        finally:
            del x_probe, y_probe
            pool.free_all_blocks()

        if candidate == total_examples:
            break

    return GPUResidencyProbe(
        capacity_examples=int(capacity),
        full_dataset=capacity == total_examples and capacity > 0,
        attempts=attempts,
        minimum_effective_free_vram_bytes=minimum_free,
        peak_pool_used_bytes=peak_pool_used,
    )


class SupervisedDataPlan:
    """Serve each epoch from RAM, mmap, full GPU, or a reusable GPU window."""

    def __init__(
        self,
        x_host,
        y_host,
        *,
        train_count,
        host_storage_mode,
        device,
        resident_capacity=None,
        index_observer=None,
    ):
        self.x_host = x_host
        self.y_host = y_host
        self.train_count = int(train_count)
        self.total_examples = int(x_host.shape[1])
        self.validation_count = self.total_examples - self.train_count
        self.host_storage_mode = host_storage_mode
        self.device = device
        self.index_observer = index_observer
        self.resident_window_examples = None
        self.full_dataset_on_gpu = False
        self.full_upload_seconds = None
        self.x_gpu = None
        self.y_gpu = None
        self.x_window = None
        self.y_window = None
        self.peak_gpu_pool_used_bytes = 0
        self.minimum_effective_free_vram_bytes = None

        if device == "cpu":
            self.storage_mode = host_storage_mode
            return

        import cupy as cp

        capacity = int(resident_capacity or 0)
        if capacity < 1:
            raise MemorySafetyError("GPU resident capacity must be positive")
        self.resident_window_examples = capacity
        pool = cp.get_default_memory_pool()
        try:
            if capacity >= self.total_examples:
                started = time.perf_counter()
                self.x_gpu = cp.asarray(x_host, dtype=cp.float32)
                self.y_gpu = cp.asarray(y_host, dtype=cp.float32)
                cp.cuda.Stream.null.synchronize()
                self.full_upload_seconds = time.perf_counter() - started
                self.full_dataset_on_gpu = True
                self.storage_mode = "gpu_full"
            else:
                self.x_window = cp.empty(
                    (x_host.shape[0], capacity),
                    dtype=cp.float32,
                    order="F",
                )
                self.y_window = cp.empty(
                    (y_host.shape[0], capacity),
                    dtype=cp.float32,
                    order="F",
                )
                cp.cuda.Stream.null.synchronize()
                self.storage_mode = "gpu_windowed"
            self.peak_gpu_pool_used_bytes = int(pool.used_bytes())
            self._observe_gpu_memory()
        except (
            cp.cuda.memory.OutOfMemoryError,
            cp.cuda.runtime.CUDARuntimeError,
        ) as exc:
            self.close()
            raise MemorySafetyError(
                f"GPU dataset residency allocation failed: {exc}"
            ) from exc

    def _observe_indices(self, indices):
        if self.index_observer is not None:
            self.index_observer(np.asarray(indices, dtype=np.int64))

    def _observe_gpu_memory(self):
        if self.device != "gpu":
            return
        import cupy as cp

        pool = cp.get_default_memory_pool()
        self.peak_gpu_pool_used_bytes = max(
            self.peak_gpu_pool_used_bytes,
            int(pool.used_bytes()),
        )
        available = effective_gpu_available_bytes()
        if available is not None:
            self.minimum_effective_free_vram_bytes = (
                available
                if self.minimum_effective_free_vram_bytes is None
                else min(self.minimum_effective_free_vram_bytes, available)
            )

    def batch_memory_preflight(self, network, batch_size, reserve_mb):
        """Return a full CPU/GPU candidate working-set decision."""
        workspace = estimate_supervised_workspace_bytes(
            batch_size,
            network.W1.shape[1],
            network.W1.shape[0],
            network.W2.shape[0],
            network.W3.shape[0],
        )
        if self.device == "cpu":
            permutation_bytes = self.train_count * np.dtype(np.int64).itemsize
            safe, result = host_allocation_status(
                workspace + permutation_bytes,
                reserve_mb,
            )
            result.update({
                "backend": "cpu",
                "workspace_bytes": int(workspace),
                "permutation_bytes": int(permutation_bytes),
                "reason": None if safe else "insufficient safe host RAM",
            })
            return result

        if (
            self.storage_mode == "gpu_windowed"
            and batch_size > self.resident_window_examples
        ):
            return {
                "safe": False,
                "backend": "gpu",
                "workspace_bytes": int(workspace),
                "reason": (
                    f"batch {batch_size} exceeds resident window capacity "
                    f"{self.resident_window_examples}"
                ),
            }
        permutation_bytes = self.train_count * np.dtype(np.int64).itemsize
        safe, result = gpu_allocation_status(
            workspace + permutation_bytes,
            reserve_mb,
        )
        result.update({
            "backend": "gpu",
            "workspace_bytes": int(workspace),
            "permutation_bytes": int(permutation_bytes),
            "reason": None if safe else "insufficient safe effective VRAM",
        })
        if self.storage_mode == "gpu_windowed":
            host_window_bytes = (
                self.resident_window_examples
                * (self.x_host.shape[0] + self.y_host.shape[0])
                * np.dtype(np.float32).itemsize
                + permutation_bytes
            )
            host_safe, host_result = host_allocation_status(
                host_window_bytes,
                reserve_mb,
            )
            result["host_window_memory_result"] = host_result
            if not host_safe:
                result["safe"] = False
                result["reason"] = "insufficient host RAM for one GPU window"
        return result

    def _gpu_batch_update(self, network, x_batch, y_batch):
        network.forward(x_batch)
        return network.backward(y_batch)

    def train_epoch(self, network, batch_size, _epoch_index):
        """Train one epoch and return loss, update count, and window rotations."""
        if self.device == "cpu":
            permutation = np.random.permutation(self.train_count)
            weighted_loss = 0.0
            updates = 0
            for start in range(0, self.train_count, batch_size):
                indices = permutation[start:start + batch_size]
                self._observe_indices(indices)
                x_batch = self.x_host[:, indices]
                y_batch = self.y_host[:, indices]
                network.forward(x_batch)
                loss = network.backward(y_batch)
                weighted_loss += network._as_float(loss) * len(indices)
                updates += 1
            return weighted_loss / self.train_count, updates, 0

        import cupy as cp

        if self.storage_mode == "gpu_full":
            permutation = cp.random.permutation(self.train_count)
            weighted_loss = 0.0
            updates = 0
            for start in range(0, self.train_count, batch_size):
                indices = permutation[start:start + batch_size]
                if self.index_observer is not None:
                    self._observe_indices(cp.asnumpy(indices))
                loss = self._gpu_batch_update(
                    network,
                    self.x_gpu[:, indices],
                    self.y_gpu[:, indices],
                )
                batch_count = int(indices.size)
                weighted_loss += network._as_float(loss) * batch_count
                updates += 1
            self._observe_gpu_memory()
            return weighted_loss / self.train_count, updates, 0

        permutation = np.random.permutation(self.train_count)
        weighted_loss = 0.0
        updates = 0
        rotations = 0
        capacity = self.resident_window_examples
        for window_start in range(0, self.train_count, capacity):
            indices = permutation[window_start:window_start + capacity]
            window_count = len(indices)
            self._observe_indices(indices)
            x_host_window = np.asfortranarray(
                self.x_host[:, indices],
                dtype=np.float32,
            )
            y_host_window = np.asfortranarray(
                self.y_host[:, indices],
                dtype=np.float32,
            )
            self.x_window[:, :window_count].set(x_host_window)
            self.y_window[:, :window_count].set(y_host_window)
            rotations += 1
            for start in range(0, window_count, batch_size):
                stop = min(start + batch_size, window_count)
                loss = self._gpu_batch_update(
                    network,
                    self.x_window[:, start:stop],
                    self.y_window[:, start:stop],
                )
                batch_count = stop - start
                weighted_loss += network._as_float(loss) * batch_count
                updates += 1
        self._observe_gpu_memory()
        return weighted_loss / self.train_count, updates, rotations

    def validation_loss(self, network, batch_size):
        """Evaluate validation loss using the selected safe residency mode."""
        if self.validation_count < 1:
            return float("nan")
        epsilon = network.xp.asarray(1e-8, dtype=network.xp.float32)
        total_loss = 0.0

        def consume(x_batch, y_batch):
            probabilities = network.forward(x_batch)
            loss_sum = -network.xp.sum(
                y_batch * network.xp.log(probabilities + epsilon)
            )
            return network._as_float(loss_sum)

        if self.device == "cpu":
            for start in range(0, self.validation_count, batch_size):
                stop = min(start + batch_size, self.validation_count)
                absolute_start = self.train_count + start
                absolute_stop = self.train_count + stop
                total_loss += consume(
                    self.x_host[:, absolute_start:absolute_stop],
                    self.y_host[:, absolute_start:absolute_stop],
                )
        elif self.storage_mode == "gpu_full":
            for start in range(0, self.validation_count, batch_size):
                stop = min(start + batch_size, self.validation_count)
                absolute_start = self.train_count + start
                absolute_stop = self.train_count + stop
                total_loss += consume(
                    self.x_gpu[:, absolute_start:absolute_stop],
                    self.y_gpu[:, absolute_start:absolute_stop],
                )
        else:
            capacity = self.resident_window_examples
            for window_start in range(0, self.validation_count, capacity):
                window_stop = min(
                    window_start + capacity,
                    self.validation_count,
                )
                window_count = window_stop - window_start
                absolute_start = self.train_count + window_start
                absolute_stop = self.train_count + window_stop
                self.x_window[:, :window_count].set(
                    np.asfortranarray(
                        self.x_host[:, absolute_start:absolute_stop],
                        dtype=np.float32,
                    )
                )
                self.y_window[:, :window_count].set(
                    np.asfortranarray(
                        self.y_host[:, absolute_start:absolute_stop],
                        dtype=np.float32,
                    )
                )
                for start in range(0, window_count, batch_size):
                    stop = min(start + batch_size, window_count)
                    total_loss += consume(
                        self.x_window[:, start:stop],
                        self.y_window[:, start:stop],
                    )
        network.synchronize()
        network.release_disposable_cache()
        self._observe_gpu_memory()
        return total_loss / self.validation_count

    def close(self):
        if self.device != "gpu":
            return
        try:
            import cupy as cp

            self.x_gpu = None
            self.y_gpu = None
            self.x_window = None
            self.y_window = None
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


class SupervisedResourceTracker:
    """Track host and GPU high/low watermarks across one training run."""

    def __init__(self, device="cpu"):
        self.device = device
        self.peak_host_rss_bytes = 0
        self.minimum_available_host_ram_bytes = None
        self.peak_gpu_pool_used_bytes = 0
        self.minimum_effective_free_vram_bytes = None
        self.observe()

    def observe(self):
        rss = process_rss_bytes()
        if rss is not None:
            self.peak_host_rss_bytes = max(self.peak_host_rss_bytes, rss)
        memory = system_memory_info()
        if memory is not None:
            self.minimum_available_host_ram_bytes = (
                memory.available
                if self.minimum_available_host_ram_bytes is None
                else min(
                    self.minimum_available_host_ram_bytes,
                    memory.available,
                )
            )
        if self.device != "gpu":
            return
        try:
            import cupy as cp

            pool_used = int(cp.get_default_memory_pool().used_bytes())
            self.peak_gpu_pool_used_bytes = max(
                self.peak_gpu_pool_used_bytes,
                pool_used,
            )
            available = effective_gpu_available_bytes()
            if available is not None:
                self.minimum_effective_free_vram_bytes = (
                    available
                    if self.minimum_effective_free_vram_bytes is None
                    else min(
                        self.minimum_effective_free_vram_bytes,
                        available,
                    )
                )
        except Exception:
            pass

    def to_dict(self):
        return {
            "peak_host_rss_bytes": self.peak_host_rss_bytes or None,
            "minimum_available_host_ram_bytes": (
                self.minimum_available_host_ram_bytes
            ),
            "peak_gpu_pool_used_bytes": self.peak_gpu_pool_used_bytes or None,
            "minimum_effective_free_vram_bytes": (
                self.minimum_effective_free_vram_bytes
            ),
        }
