"""Fixed-batch safety checks and bounded supervised dataset residency."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from agents.network_architecture import DEFAULT_HIDDEN_SIZES
from utils.resource_limits import (
    MIB,
    MemorySafetyError,
    effective_gpu_available_bytes,
    gpu_allocation_status,
    host_allocation_status,
    process_rss_bytes,
    system_memory_info,
)


DEFAULT_SUPERVISED_BATCH_SIZE = 8192
SUPERVISED_GPU_MEMORY_RESERVE_MB = 512

SUPERVISED_BATCH_SIZE_CHOICES = tuple(
    2 ** exponent for exponent in range(10, 21)
)
SUPERVISED_GPU_RESIDENCY_CANDIDATES = tuple(
    2 ** exponent for exponent in range(11, 21)
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
    hidden_sizes,
    output_size,
):
    """Conservatively estimate one float32 forward/backward live workspace."""
    dimensions = (
        int(input_size),
        *(int(size) for size in hidden_sizes),
        int(output_size),
    )
    # Per-example tensors include inputs/labels, all forward activations,
    # softmax temporaries, backward deltas, and fancy-index materialization.
    # Every hidden layer contributes one forward and one backward pair.
    per_example_floats = (
        2 * dimensions[0]
        + 4 * dimensions[-1]
        + 4 * sum(dimensions[1:-1])
    )
    parameter_floats = sum(
        dimensions[index] * dimensions[index - 1] + dimensions[index]
        for index in range(1, len(dimensions))
    )
    # Two parameter-sized buffers cover gradients and matrix-library
    # temporaries; 1.25x covers allocator rounding and transient expressions.
    raw_bytes = (
        int(batch_size) * per_example_floats * np.dtype(np.float32).itemsize
        + 2 * parameter_floats * np.dtype(np.float32).itemsize
        + int(batch_size) * np.dtype(np.int64).itemsize
    )
    return int(math.ceil(raw_bytes * 1.25))


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
    hidden_sizes=DEFAULT_HIDDEN_SIZES,
):
    """Probe dataset residency with temporary arrays and no weight updates."""
    import cupy as cp

    total_examples = int(x_host.shape[1])
    feature_count = int(x_host.shape[0] + y_host.shape[0])
    minimum_batch = min(DEFAULT_SUPERVISED_BATCH_SIZE, total_examples)
    minimum_workspace = estimate_supervised_workspace_bytes(
        minimum_batch,
        x_host.shape[0],
        hidden_sizes,
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
            network.hidden_sizes,
            getattr(network, f"W{network.layer_count}").shape[0],
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
        network.forward(x_batch, training=True)
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
                network.forward(x_batch, training=True)
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
