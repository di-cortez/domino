"""Small float32 NumPy/CuPy multilayer perceptron used by domino agents."""

from __future__ import annotations

import os
import time

import numpy as host_np

from agents.network_architecture import (
    DEFAULT_HIDDEN1_SIZE,
    DEFAULT_HIDDEN2_SIZE,
    DEFAULT_NETWORK_ARCHITECTURE,
    policy_layer_names,
)
from utils.myrandom import RandomNamespace, SeedPlan


DEVICES = ("auto", "cpu", "gpu")
NETWORK_DTYPE = host_np.float32
# Both regularizers are opt-in; these values reproduce an unregularized run.
DISABLED_DROPOUT_RATE = 0.0
DISABLED_WEIGHT_DECAY = 0.0
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


def validated_hidden_sizes(hidden_sizes):
    """Return one tuple of hidden-layer widths of any depth from one upwards."""
    sizes = tuple(int(size) for size in hidden_sizes)
    if not sizes:
        raise ValueError("hidden_sizes must describe at least one layer.")
    for position, size in enumerate(sizes, start=1):
        if size < 1:
            raise ValueError(f"hidden{position}_size must be positive.")
    return sizes


def validated_dropout_rate(value):
    """Return one hidden-layer dropout probability inside ``[0, 1)``."""
    rate = float(value)
    if not 0.0 <= rate < 1.0:
        raise ValueError(
            f"dropout_rate must be at least 0 and below 1, got {rate!r}."
        )
    return rate


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


class SupervisedNeuralNetwork:
    """Float32 MLP with a configurable number of hidden layers.

    ``hidden_sizes`` selects both the depth and the width of the hidden stack
    and supersedes the two historical ``hidden1_size``/``hidden2_size``
    arguments, which remain the default 256x128 architecture. Weights are named
    ``W1..W{L}``/``b1..b{L}`` where ``L`` is the hidden-layer count plus the
    output layer, so a two-layer network keeps exactly the historical
    ``W1, b1, W2, b2, W3, b3`` checkpoint keys.

    ``dropout_rate`` enables inverted dropout on every hidden activation. It is
    applied only inside training forward passes (``forward(..., training=True)``);
    validation, diagnostics, and gameplay always evaluate the complete network.
    """

    def __init__(
        self,
        input_size=DEFAULT_NETWORK_ARCHITECTURE.input_size,
        hidden1_size=DEFAULT_HIDDEN1_SIZE,
        hidden2_size=DEFAULT_HIDDEN2_SIZE,
        output_size=DEFAULT_NETWORK_ARCHITECTURE.output_size,
        learning_rate=0.01,
        random_seed=None,
        weight_decay=DISABLED_WEIGHT_DECAY,
        dropout_rate=DISABLED_DROPOUT_RATE,
        device="auto",
        hidden_sizes=None,
        seed_plan=None,
    ):
        self.xp, self.device = resolve_device(device)
        self.lr = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.dropout_rate = validated_dropout_rate(dropout_rate)
        self.hidden_sizes = validated_hidden_sizes(
            (hidden1_size, hidden2_size) if hidden_sizes is None
            else hidden_sizes
        )
        if seed_plan is not None and not isinstance(seed_plan, SeedPlan):
            raise TypeError("seed_plan must be a utils.myrandom.SeedPlan")
        if seed_plan is not None and random_seed is not None:
            raise ValueError("pass either seed_plan or random_seed, not both")
        self.seed_plan = seed_plan
        self._dropout_generator = (
            None
            if seed_plan is None
            else seed_plan.generator(RandomNamespace.SUPERVISED_DROPOUT)
        )
        random_source = None
        if seed_plan is None:
            random_source = (
                self.xp.random
                if random_seed is None
                else self.xp.random.RandomState(random_seed)
            )
        else:
            initialization_generator = seed_plan.generator(
                RandomNamespace.SUPERVISED_INITIALIZATION
            )

        def initialized(shape, scale):
            if seed_plan is None:
                values = random_source.randn(*shape).astype(
                    self.xp.float32,
                    copy=False,
                )
            else:
                host_values = initialization_generator.standard_normal(
                    shape
                ).astype(host_np.float32, copy=False)
                host_values *= host_np.asarray(scale, dtype=host_np.float32)
                values = self.xp.asarray(host_values, dtype=self.xp.float32)
                return values
            return values * self.xp.asarray(scale, dtype=self.xp.float32)

        # Hidden layers keep He initialization and the output layer keeps its
        # smaller Xavier-style scale, drawn in forward order so a fixed seed
        # reproduces the historical two-layer network exactly.
        dimensions = (input_size, *self.hidden_sizes, output_size)
        for index in range(1, len(dimensions)):
            fan_in = dimensions[index - 1]
            fan_out = dimensions[index]
            gain = 1.0 if index == self.layer_count else 2.0
            setattr(
                self,
                f"W{index}",
                initialized((fan_out, fan_in), host_np.sqrt(gain / fan_in)),
            )
            setattr(
                self,
                f"b{index}",
                self.xp.zeros((fan_out, 1), dtype=self.xp.float32),
            )
        self.cache = {}
        self.last_gradient_dtypes = {}
        self.last_training_summary = {}

    @property
    def layer_count(self):
        """Return the number of weight layers, hidden stack plus output."""
        return len(self.hidden_sizes) + 1

    @property
    def weight_names(self):
        return policy_layer_names(len(self.hidden_sizes))

    @property
    def last_hidden_activation_key(self):
        """Return the cache key of the activation feeding the output layer."""
        return f"A{len(self.hidden_sizes)}"

    @property
    def logits_key(self):
        """Return the cache key of the pre-softmax output-layer activation."""
        return f"Z{self.layer_count}"

    def relu(self, z):
        return self.xp.maximum(self.xp.asarray(0, dtype=z.dtype), z)

    def relu_derivative(self, z):
        return (z > 0).astype(z.dtype, copy=False)

    def softmax(self, z):
        exp_z = self.xp.exp(z - self.xp.max(z, axis=0, keepdims=True))
        return exp_z / self.xp.sum(exp_z, axis=0, keepdims=True)

    def _dropout_uniform(self, shape):
        """Return uniform samples used to build one dropout mask."""
        if self._dropout_generator is not None:
            values = self._dropout_generator.random(
                shape,
                dtype=host_np.float32,
            )
            return self.xp.asarray(values, dtype=self.xp.float32)
        return self.xp.random.random(shape)

    def _hidden_dropout(self, activations, training):
        """Return activations and the inverted-dropout scale mask, or ``None``.

        The mask is returned so backpropagation can reuse exactly the units
        that the forward pass kept.
        """
        if not training or self.dropout_rate <= 0.0:
            return activations, None
        keep_probability = 1.0 - self.dropout_rate
        kept = self._dropout_uniform(activations.shape) < keep_probability
        mask = kept.astype(activations.dtype, copy=False) / self.xp.asarray(
            keep_probability,
            dtype=activations.dtype,
        )
        return activations * mask, mask

    def _backpropagate_dropout(self, gradient, mask_name):
        """Scale one hidden gradient by the mask used in the forward pass."""
        mask = self.cache.get(mask_name)
        return gradient if mask is None else gradient * mask

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

    def forward(self, x, training=False):
        x = self._to_backend(x)
        last = self.layer_count
        cache = {"X": x}
        activation = x
        for index in range(1, last):
            pre_activation = self.xp.dot(
                getattr(self, f"W{index}"),
                activation,
            ) + getattr(self, f"b{index}")
            activation, mask = self._hidden_dropout(
                self.relu(pre_activation),
                training,
            )
            cache[f"Z{index}"] = pre_activation
            cache[f"A{index}"] = activation
            # Mask entries exist only while dropout is active, so every cached
            # value stays a float32 activation array for the disabled default.
            if mask is not None:
                cache[f"D{index}"] = mask
        logits = self.xp.dot(
            getattr(self, f"W{last}"),
            activation,
        ) + getattr(self, f"b{last}")
        probabilities = self.softmax(logits)
        cache[f"Z{last}"] = logits
        cache[f"A{last}"] = probabilities
        self.cache = cache
        return probabilities

    def backpropagate_layers(self, delta, inverse_count, hidden_gradient_hook=None):
        """Return every weight and bias gradient from the cached forward pass.

        ``delta`` is the gradient with respect to the output-layer logits.
        ``hidden_gradient_hook`` receives the gradient with respect to the last
        hidden activation before dropout is reapplied, which is how the RL
        value head folds its shared contribution into the policy trunk. The
        returned mapping is ordered ``W1, b1, ..., W{L}, b{L}`` so
        gradient-norm accumulation stays independent of the layer count.
        """
        computed = {}
        for index in range(self.layer_count, 0, -1):
            previous = (
                self.cache[f"A{index - 1}"] if index > 1 else self.cache["X"]
            )
            computed[f"W{index}"] = inverse_count * self.xp.dot(delta, previous.T)
            computed[f"b{index}"] = inverse_count * self.xp.sum(
                delta,
                axis=1,
                keepdims=True,
            )
            if index == 1:
                break
            propagated = self.xp.dot(getattr(self, f"W{index}").T, delta)
            if index == self.layer_count and hidden_gradient_hook is not None:
                propagated = hidden_gradient_hook(propagated)
            propagated = self._backpropagate_dropout(propagated, f"D{index - 1}")
            delta = propagated * self.relu_derivative(self.cache[f"Z{index - 1}"])
        gradients = {}
        for index in range(1, self.layer_count + 1):
            gradients[f"W{index}"] = computed[f"W{index}"]
            gradients[f"b{index}"] = computed[f"b{index}"]
        return gradients

    def backward(self, y_target):
        y_target = self._to_backend(y_target)
        sample_count = y_target.shape[1]
        last = self.layer_count
        probabilities = self.cache[f"A{last}"]
        x = self.cache["X"]
        inverse_count = self.xp.asarray(
            1.0 / sample_count,
            dtype=x.dtype,
        )

        gradients = self.backpropagate_layers(
            probabilities - y_target,
            inverse_count,
        )
        self.last_gradient_dtypes = {
            name: gradient.dtype for name, gradient in gradients.items()
        }
        decay = self.xp.asarray(self.weight_decay, dtype=x.dtype)
        learning_rate = self.xp.asarray(self.lr, dtype=x.dtype)
        for index in range(last, 0, -1):
            weight = getattr(self, f"W{index}")
            bias = getattr(self, f"b{index}")
            weight -= learning_rate * (gradients[f"W{index}"] + decay * weight)
            bias -= learning_rate * gradients[f"b{index}"]

        epsilon = self.xp.asarray(1e-8, dtype=x.dtype)
        return -inverse_count * self.xp.sum(
            y_target * self.xp.log(probabilities + epsilon)
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

    def _run_array_training_epoch(
        self,
        x_train,
        y_train,
        batch_size,
        epoch_index,
    ):
        """Train one complete shuffled epoch from host or backend arrays."""
        sample_count = x_train.shape[1]
        if self.seed_plan is None:
            permutation = host_np.random.permutation(sample_count)
        else:
            permutation = self.seed_plan.generator(
                RandomNamespace.SUPERVISED_SHUFFLE,
                epoch_index,
            ).permutation(sample_count)
        weighted_loss = 0.0
        update_count = 0
        for start in range(0, sample_count, batch_size):
            indices = permutation[start:start + batch_size]
            batch_count = len(indices)
            self.forward(x_train[:, indices], training=True)
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
        plateau_loss_start = 0

        for epoch in range(epochs):
            current_batch_size = batch_size
            self.synchronize()
            training_started = time.perf_counter()
            try:
                if epoch_runner is None:
                    mean_loss, optimizer_updates, window_rotations = (
                        self._run_array_training_epoch(
                            x_train,
                            y_train,
                            current_batch_size,
                            epoch,
                        )
                    )
                else:
                    mean_loss, optimizer_updates, window_rotations = epoch_runner(
                        self,
                        current_batch_size,
                        epoch,
                    )
            except Exception as exc:
                if not self._is_backend_memory_error(exc):
                    raise
                self.release_disposable_cache()
                raise MemoryError(
                    "Supervised training exhausted memory at fixed batch "
                    f"{current_batch_size}."
                ) from exc
            self.synchronize()
            training_seconds = time.perf_counter() - training_started
            loss_history.append(mean_loss)
            completed_epochs += 1

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
            "weight_decay": self.weight_decay,
            "dropout_rate": self.dropout_rate,
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
