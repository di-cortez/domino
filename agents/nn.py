"""Small float32 NumPy/CuPy multilayer perceptron used by domino agents."""

from __future__ import annotations

import os
import time

import numpy as host_np


DEVICES = ("auto", "cpu", "gpu")
NETWORK_DTYPE = host_np.float32
GPU_UNAVAILABLE_REASON = None
_cupy = None

try:
    if os.environ.get("DOMINO_FORCE_CPU") == "1":
        raise ImportError("CPU forced by DOMINO_FORCE_CPU")
    import cupy as _cupy

    device_count = int(_cupy.cuda.runtime.getDeviceCount())
    if device_count < 1:
        raise RuntimeError("CuPy did not find a CUDA-capable device")
    # A visible device is not necessarily usable (for example, a busy display
    # GPU can reject a new context). One synchronized allocation verifies the
    # backend before ``auto`` is allowed to select it.
    _backend_probe = _cupy.zeros(1, dtype=_cupy.float32)
    _cupy.cuda.Stream.null.synchronize()
    del _backend_probe
    _cupy.get_default_memory_pool().free_all_blocks()
    GPU_ENABLED = True
except Exception as exc:
    GPU_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"
    GPU_ENABLED = False


def resolve_device(device="auto"):
    """Return the array backend and concrete device for one network."""
    if device not in DEVICES:
        raise ValueError(f"Unknown device {device!r}; expected one of {DEVICES}.")
    if device == "cpu":
        return host_np, "cpu"
    if device == "gpu":
        if not GPU_ENABLED or _cupy is None:
            reason = GPU_UNAVAILABLE_REASON or "CuPy/CUDA is unavailable"
            raise ValueError(f"device='gpu' requested but {reason}.")
        return _cupy, "gpu"
    if GPU_ENABLED and _cupy is not None:
        return _cupy, "gpu"
    return host_np, "cpu"


if GPU_ENABLED:
    # Cap this process's CuPy pool when concurrent sweep jobs share one GPU.
    _vram_limit_mb = os.environ.get("DOMINO_VRAM_LIMIT_MB")
    if _vram_limit_mb:
        _cupy.get_default_memory_pool().set_limit(
            size=int(float(_vram_limit_mb) * 1024 * 1024)
        )


class SupervisedNeuralNetwork:
    """Three-layer float32 MLP: 168 -> 256 -> 128 -> 56."""

    def __init__(
        self,
        input_size=168,
        hidden1_size=256,
        hidden2_size=128,
        output_size=56,
        learning_rate=0.01,
        random_seed=None,
        weight_decay=0.0,
        device="auto",
    ):
        self.xp, self.device = resolve_device(device)
        self.lr = float(learning_rate)
        self.weight_decay = float(weight_decay)
        random_source = (
            self.xp.random
            if random_seed is None
            else self.xp.random.RandomState(random_seed)
        )

        def initialized(shape, scale):
            values = random_source.randn(*shape).astype(
                self.xp.float32,
                copy=False,
            )
            return values * self.xp.asarray(scale, dtype=self.xp.float32)

        self.W1 = initialized(
            (hidden1_size, input_size),
            host_np.sqrt(2.0 / input_size),
        )
        self.b1 = self.xp.zeros((hidden1_size, 1), dtype=self.xp.float32)
        self.W2 = initialized(
            (hidden2_size, hidden1_size),
            host_np.sqrt(2.0 / hidden1_size),
        )
        self.b2 = self.xp.zeros((hidden2_size, 1), dtype=self.xp.float32)
        self.W3 = initialized(
            (output_size, hidden2_size),
            host_np.sqrt(1.0 / hidden2_size),
        )
        self.b3 = self.xp.zeros((output_size, 1), dtype=self.xp.float32)
        self.cache = {}
        self.last_gradient_dtypes = {}
        self.last_training_summary = {}

    @property
    def weight_names(self):
        return ("W1", "b1", "W2", "b2", "W3", "b3")

    def relu(self, z):
        return self.xp.maximum(self.xp.asarray(0, dtype=z.dtype), z)

    def relu_derivative(self, z):
        return (z > 0).astype(z.dtype, copy=False)

    def softmax(self, z):
        exp_z = self.xp.exp(z - self.xp.max(z, axis=0, keepdims=True))
        return exp_z / self.xp.sum(exp_z, axis=0, keepdims=True)

    def _to_backend(self, array):
        """Move one array to this network's backend without promoting dtype."""
        return self.xp.asarray(array, dtype=self.xp.float32)

    def to_host(self, array):
        """Return one backend array as a plain float32 NumPy array."""
        if hasattr(array, "get"):
            array = array.get()
        return host_np.asarray(array, dtype=NETWORK_DTYPE)

    def load_policy_weights(self, data):
        """Load compatible NumPy/CuPy weights and cast legacy float64 safely."""
        for name in self.weight_names:
            setattr(
                self,
                name,
                self.xp.asarray(data[name], dtype=self.xp.float32),
            )

    def forward(self, x):
        x = self._to_backend(x)
        z1 = self.xp.dot(self.W1, x) + self.b1
        a1 = self.relu(z1)
        z2 = self.xp.dot(self.W2, a1) + self.b2
        a2 = self.relu(z2)
        z3 = self.xp.dot(self.W3, a2) + self.b3
        a3 = self.softmax(z3)

        self.cache = {
            "X": x,
            "Z1": z1,
            "A1": a1,
            "Z2": z2,
            "A2": a2,
            "Z3": z3,
            "A3": a3,
        }
        return a3

    def backward(self, y_target):
        y_target = self._to_backend(y_target)
        sample_count = y_target.shape[1]
        a3 = self.cache["A3"]
        a2 = self.cache["A2"]
        a1 = self.cache["A1"]
        x = self.cache["X"]
        inverse_count = self.xp.asarray(
            1.0 / sample_count,
            dtype=x.dtype,
        )

        dz3 = a3 - y_target
        dW3 = inverse_count * self.xp.dot(dz3, a2.T)
        db3 = inverse_count * self.xp.sum(dz3, axis=1, keepdims=True)

        da2 = self.xp.dot(self.W3.T, dz3)
        dz2 = da2 * self.relu_derivative(self.cache["Z2"])
        dW2 = inverse_count * self.xp.dot(dz2, a1.T)
        db2 = inverse_count * self.xp.sum(dz2, axis=1, keepdims=True)

        da1 = self.xp.dot(self.W2.T, dz2)
        dz1 = da1 * self.relu_derivative(self.cache["Z1"])
        dW1 = inverse_count * self.xp.dot(dz1, x.T)
        db1 = inverse_count * self.xp.sum(dz1, axis=1, keepdims=True)

        gradients = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3,
        }
        self.last_gradient_dtypes = {
            name: gradient.dtype for name, gradient in gradients.items()
        }
        decay = self.xp.asarray(self.weight_decay, dtype=x.dtype)
        learning_rate = self.xp.asarray(self.lr, dtype=x.dtype)
        self.W3 -= learning_rate * (dW3 + decay * self.W3)
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * (dW2 + decay * self.W2)
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * (dW1 + decay * self.W1)
        self.b1 -= learning_rate * db1

        epsilon = self.xp.asarray(1e-8, dtype=x.dtype)
        return -inverse_count * self.xp.sum(
            y_target * self.xp.log(a3 + epsilon)
        )

    def _as_float(self, value):
        if hasattr(value, "get"):
            value = value.get()
        return float(value)

    def synchronize(self):
        """Wait for queued CUDA work so wall-clock timings are valid."""
        if self.device == "gpu":
            self.xp.cuda.Stream.null.synchronize()

    def _is_backend_memory_error(self, exc):
        if isinstance(exc, MemoryError):
            return True
        if self.device != "gpu":
            return False
        return isinstance(
            exc,
            (
                self.xp.cuda.memory.OutOfMemoryError,
                self.xp.cuda.runtime.CUDARuntimeError,
            ),
        )

    def release_disposable_cache(self):
        """Release forward intermediates and unused GPU pool blocks."""
        self.cache = {}
        if self.device == "gpu":
            self.xp.get_default_memory_pool().free_all_blocks()

    def _run_array_training_epoch(self, x_train, y_train, batch_size):
        """Train one complete shuffled epoch from host or backend arrays."""
        sample_count = x_train.shape[1]
        permutation = host_np.random.permutation(sample_count)
        weighted_loss = 0.0
        update_count = 0
        for start in range(0, sample_count, batch_size):
            indices = permutation[start:start + batch_size]
            batch_count = len(indices)
            self.forward(x_train[:, indices])
            loss = self.backward(y_train[:, indices])
            weighted_loss += self._as_float(loss) * batch_count
            update_count += 1
        return weighted_loss / sample_count, update_count, 0

    def _batched_validation_loss(self, x_val, y_val, batch_size):
        total_loss = 0.0
        sample_count = x_val.shape[1]
        for start in range(0, sample_count, batch_size):
            x_batch = x_val[:, start:start + batch_size]
            y_batch = self._to_backend(y_val[:, start:start + batch_size])
            batch_count = x_batch.shape[1]
            probabilities = self.forward(x_batch)
            epsilon = self.xp.asarray(1e-8, dtype=probabilities.dtype)
            batch_loss = -self.xp.sum(
                y_batch * self.xp.log(probabilities + epsilon)
            ) / self.xp.asarray(batch_count, dtype=probabilities.dtype)
            total_loss += self._as_float(batch_loss) * batch_count
        self.release_disposable_cache()
        return total_loss / sample_count

    def train(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        epochs=1500,
        batch_size=128,
        on_validation=None,
        progress_callback=None,
        quiet=False,
        early_stopping_patience=None,
        lr_decay_factor=None,
        lr_decay_patience=5,
        validation_interval=10,
        epoch_runner=None,
        validation_runner=None,
        batch_controller=None,
        epoch_metrics_callback=None,
        training_plateau_window=None,
        training_plateau_patience=4,
        training_plateau_min_epochs=100,
        training_plateau_min_relative_improvement=0.001,
    ):
        """Train sequential epochs with independent plateau counters.

        ``epoch_runner`` and ``validation_runner`` let the supervised pipeline
        provide RAM, mmap, full-GPU, or windowed-GPU data access without moving
        storage policy into the network. The default path retains the public
        array-based API used by tests and small direct callers. Passing a
        ``training_plateau_window`` enables conservative block-median early
        stopping; direct callers remain opt-in while the supervised pipeline
        enables it by default.
        """
        if epochs < 1:
            raise ValueError("epochs must be positive")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if lr_decay_factor is not None and not 0 < lr_decay_factor < 1:
            raise ValueError("lr_decay_factor must be between zero and one")
        if lr_decay_patience < 1:
            raise ValueError("lr_decay_patience must be positive")
        if validation_interval < 1:
            raise ValueError("validation_interval must be positive")
        if training_plateau_window is not None and training_plateau_window < 1:
            raise ValueError("training_plateau_window must be positive")
        if training_plateau_patience < 1:
            raise ValueError("training_plateau_patience must be positive")
        if training_plateau_min_epochs < 1:
            raise ValueError("training_plateau_min_epochs must be positive")
        if training_plateau_min_relative_improvement < 0:
            raise ValueError(
                "training_plateau_min_relative_improvement must be non-negative"
            )

        loss_history = []
        best_validation_loss = float("inf")
        lr_checks_without_improvement = 0
        early_checks_without_improvement = 0
        lr_decay_count = 0
        initial_learning_rate = self.lr
        completed_epochs = 0
        training_plateau_checks_without_improvement = 0
        training_plateau_last_relative_improvement = None
        training_plateau_stopped = False
        stopping_reason = "epoch_limit"
        plateau_loss_start = (
            0
            if batch_controller is None
            or getattr(batch_controller, "finished", True)
            else None
        )

        for epoch in range(epochs):
            current_batch_size = (
                batch_controller.current_batch_size
                if batch_controller is not None
                else batch_size
            )
            self.synchronize()
            training_started = time.perf_counter()
            while True:
                try:
                    if epoch_runner is None:
                        mean_loss, optimizer_updates, window_rotations = (
                            self._run_array_training_epoch(
                                x_train,
                                y_train,
                                current_batch_size,
                            )
                        )
                    else:
                        mean_loss, optimizer_updates, window_rotations = epoch_runner(
                            self,
                            current_batch_size,
                            epoch,
                        )
                    break
                except Exception as exc:
                    if not self._is_backend_memory_error(exc):
                        raise
                    self.release_disposable_cache()
                    retry = (
                        batch_controller is not None
                        and batch_controller.handle_runtime_memory_failure(
                            epoch,
                            exc,
                        )
                    )
                    if not retry:
                        raise MemoryError(
                            "Supervised training exhausted memory at batch "
                            f"{current_batch_size}; no smaller accepted batch "
                            "is available."
                        ) from exc
                    current_batch_size = batch_controller.current_batch_size
            self.synchronize()
            training_seconds = time.perf_counter() - training_started
            loss_history.append(mean_loss)
            completed_epochs += 1

            if batch_controller is not None:
                batch_controller.record_epoch(epoch, training_seconds)
                if plateau_loss_start is None and batch_controller.finished:
                    # The epoch that completed/rejected the benchmark may use
                    # a different batch. Begin plateau evidence on the next
                    # epoch, after the selected batch is stable.
                    plateau_loss_start = completed_epochs

            validation_loss = None
            if epoch % validation_interval == 0 and x_val is not None:
                if validation_runner is None:
                    validation_loss = self._batched_validation_loss(
                        x_val,
                        y_val,
                        batch_size=current_batch_size,
                    )
                else:
                    validation_loss = validation_runner(
                        self,
                        current_batch_size,
                    )
                if on_validation is not None:
                    on_validation(epoch, validation_loss, self)

                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    lr_checks_without_improvement = 0
                    early_checks_without_improvement = 0
                else:
                    lr_checks_without_improvement += 1
                    if early_stopping_patience is not None:
                        early_checks_without_improvement += 1

                    if (
                        lr_decay_factor is not None
                        and lr_checks_without_improvement >= lr_decay_patience
                    ):
                        old_learning_rate = self.lr
                        self.lr *= lr_decay_factor
                        lr_checks_without_improvement = 0
                        lr_decay_count += 1
                        if not quiet:
                            print(
                                "  -> Validation loss did not improve for "
                                f"{lr_decay_patience} consecutive checks; "
                                "learning rate reduced from "
                                f"{old_learning_rate:.8f} to {self.lr:.8f}."
                            )

                if not quiet:
                    print(
                        f"Epoch {epoch} | training loss: {mean_loss:.4f} | "
                        f"validation loss: {validation_loss:.4f}"
                    )
            elif epoch % validation_interval == 0 and not quiet:
                print(f"Epoch {epoch} | training loss: {mean_loss:.4f}")

            training_plateau_checked = False
            training_plateau_relative_improvement = None
            if (
                training_plateau_window is not None
                and plateau_loss_start is not None
                and completed_epochs >= training_plateau_min_epochs
            ):
                stable_losses = loss_history[plateau_loss_start:]
                stable_epoch_count = len(stable_losses)
                if (
                    stable_epoch_count >= 2 * training_plateau_window
                    and stable_epoch_count % training_plateau_window == 0
                ):
                    previous_block = stable_losses[
                        -2 * training_plateau_window:-training_plateau_window
                    ]
                    current_block = stable_losses[-training_plateau_window:]
                    previous_median = float(host_np.median(previous_block))
                    current_median = float(host_np.median(current_block))
                    denominator = max(abs(previous_median), 1e-12)
                    training_plateau_relative_improvement = (
                        previous_median - current_median
                    ) / denominator
                    training_plateau_last_relative_improvement = (
                        training_plateau_relative_improvement
                    )
                    training_plateau_checked = True
                    if (
                        training_plateau_relative_improvement
                        < training_plateau_min_relative_improvement
                    ):
                        training_plateau_checks_without_improvement += 1
                    else:
                        training_plateau_checks_without_improvement = 0

                    training_plateau_stopped = (
                        training_plateau_checks_without_improvement
                        >= training_plateau_patience
                    )

            metrics = {
                "epoch": epoch,
                "batch_size": current_batch_size,
                "training_seconds": training_seconds,
                "optimizer_updates": optimizer_updates,
                "window_rotations": window_rotations,
                "training_loss": float(mean_loss),
                "validation_loss": validation_loss,
                "training_plateau_checked": training_plateau_checked,
                "training_plateau_relative_improvement": (
                    training_plateau_relative_improvement
                ),
                "training_plateau_checks_without_improvement": (
                    training_plateau_checks_without_improvement
                ),
            }
            if epoch_metrics_callback is not None:
                epoch_metrics_callback(metrics)
            if progress_callback is not None:
                progress_callback(completed_epochs, epochs)

            validation_stopped = (
                early_stopping_patience is not None
                and early_checks_without_improvement
                >= early_stopping_patience
            )
            if validation_stopped:
                stopping_reason = "validation_loss_plateau"
                if not quiet:
                    print(
                        "Early stopping: validation loss did not improve for "
                        f"{early_stopping_patience} checks. Stopped after "
                        f"epoch {epoch}."
                    )
                break
            if training_plateau_stopped:
                stopping_reason = "training_loss_plateau"
                if not quiet:
                    print(
                        "Early stopping: median training loss improved by less "
                        f"than {training_plateau_min_relative_improvement:.3%} "
                        f"across {training_plateau_patience} consecutive "
                        f"{training_plateau_window}-epoch blocks. Stopped "
                        f"after epoch {epoch + 1}."
                    )
                break

        self.last_training_summary = {
            "completed_epochs": completed_epochs,
            "best_validation_loss": best_validation_loss,
            "initial_learning_rate": initial_learning_rate,
            "final_learning_rate": self.lr,
            "lr_decay_factor": lr_decay_factor,
            "lr_decay_patience": lr_decay_patience,
            "lr_decay_count": lr_decay_count,
            "lr_checks_without_improvement": lr_checks_without_improvement,
            "early_checks_without_improvement": (
                early_checks_without_improvement
            ),
            "training_plateau_enabled": training_plateau_window is not None,
            "training_plateau_window": training_plateau_window,
            "training_plateau_patience": training_plateau_patience,
            "training_plateau_min_epochs": training_plateau_min_epochs,
            "training_plateau_min_relative_improvement": (
                training_plateau_min_relative_improvement
            ),
            "training_plateau_checks_without_improvement": (
                training_plateau_checks_without_improvement
            ),
            "training_plateau_last_relative_improvement": (
                training_plateau_last_relative_improvement
            ),
            "training_plateau_loss_start_epoch": (
                None if plateau_loss_start is None else plateau_loss_start + 1
            ),
            "training_plateau_stopped": training_plateau_stopped,
            "stopping_reason": stopping_reason,
        }
        return loss_history
