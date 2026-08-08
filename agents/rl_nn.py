"""Masked policy network with an optional training-only value head."""

import os
import time

import numpy as np

from agents.network_architecture import architecture_from_weights
from agents.nn import (
    DEVICES,
    DISABLED_DROPOUT_RATE,
    DISABLED_WEIGHT_DECAY,
    SupervisedNeuralNetwork,
)

_VALUE_WEIGHTS = ("Wv", "bv")
_OPTIMIZER_STEP_KEY = "optimizer_step_count"
_ALGORITHM_KEY = "rl_training_algorithm"


class PolicyNetwork(SupervisedNeuralNetwork):
    """Supervised policy architecture with masked PPO/REINFORCE gradients.

    PPO is the default self-play algorithm. ``use_value_head=True`` adds a
    linear ``V(s)`` critic that reads the last hidden activation and is shared
    by PPO and the legacy REINFORCE path.

    The hidden stack is whatever the parent class was built with, so an RL run
    always adopts the depth and widths stored in the checkpoint it starts from.

    ``device`` selects the array backend independently of the parent class:
    ``"auto"`` (default) matches ``GPU_ENABLED`` exactly, reproducing prior
    behavior; ``"cpu"``/``"gpu"`` force one backend regardless of what's
    installed/enabled globally.

    ``dropout_rate`` and ``weight_decay`` are optional regularizers inherited
    from the supervised architecture. Dropout applies only to update forward
    passes, never to rollouts, opponent-pool snapshots, or evaluation, and
    weight decay is applied as a decoupled shrink after gradient clipping.
    """

    def __init__(self, *args, use_value_head=False, device="auto", **kwargs):
        super().__init__(*args, device=device, **kwargs)
        self.use_value_head = use_value_head
        # The current optimizer is plain SGD and therefore has no momentum or
        # adaptive tensors. Its step counter is still checkpointed so resume
        # metadata and PPO optimizer-step accounting remain exact.
        self.optimizer_step_count = 0
        if use_value_head:
            self.Wv = self.xp.zeros(
                (1, self.hidden_sizes[-1]),
                dtype=self.xp.float32,
            )
            self.bv = self.xp.zeros((1, 1), dtype=self.xp.float32)

    @property
    def decayed_weight_names(self):
        """Return the arrays weight decay shrinks; bias vectors stay free.

        ``Wv`` is listed unconditionally because ``_apply_gradient_step``
        shrinks only the names the current update produced a gradient for.
        """
        return tuple(
            f"W{index}" for index in range(1, self.layer_count + 1)
        ) + ("Wv",)

    def _cast_weights_to_device(self):
        """Move the policy weights built by the parent class to ``self.xp``."""
        for name in self.weight_names:
            value = getattr(self, name, None)
            if value is None:
                continue
            if hasattr(value, "get"):
                value = value.get()
            setattr(self, name, self.xp.asarray(value, dtype=self.xp.float32))

    def _dropout_uniform(self, shape):
        """Draw dropout randomness from the checkpointed host RNG.

        RL resume restores the parent NumPy/Python RNG state exactly, while no
        CuPy generator state is persisted. Sampling on the host therefore keeps
        a dropout run resumable on both backends.
        """
        return self.xp.asarray(np.random.random(shape), dtype=self.xp.float32)

    @classmethod
    def _load_npz_weights(
        cls,
        path,
        learning_rate,
        use_value_head=False,
        device="auto",
        weight_decay=DISABLED_WEIGHT_DECAY,
        dropout_rate=DISABLED_DROPOUT_RATE,
    ):
        """Build a network from an ``.npz`` checkpoint.

        The hidden-layer count and every width are read from the archive, so a
        checkpoint always restores the architecture it was trained with.
        """
        with np.load(path, allow_pickle=False) as data:
            architecture = architecture_from_weights(data)

            network = cls(
                input_size=architecture.input_size,
                output_size=architecture.output_size,
                hidden_sizes=architecture.hidden_sizes,
                learning_rate=learning_rate,
                use_value_head=use_value_head,
                device=device,
                weight_decay=weight_decay,
                dropout_rate=dropout_rate,
            )
            for name in network.weight_names:
                setattr(
                    network,
                    name,
                    network.xp.asarray(data[name], dtype=network.xp.float32),
                )
            if use_value_head and all(name in data for name in _VALUE_WEIGHTS):
                for name in _VALUE_WEIGHTS:
                    setattr(
                        network,
                        name,
                        network.xp.asarray(
                            data[name],
                            dtype=network.xp.float32,
                        ),
                    )
            if _OPTIMIZER_STEP_KEY in data:
                network.optimizer_step_count = int(
                    np.asarray(data[_OPTIMIZER_STEP_KEY]).item()
                )
            if _ALGORITHM_KEY in data:
                network.rl_training_algorithm = str(
                    np.asarray(data[_ALGORITHM_KEY]).item()
                )
            return network

    @classmethod
    def load_from_sl(
        cls,
        sl_weights_path="models/domino_sl_weights.npz",
        learning_rate=0.001,
        use_value_head=False,
        device="auto",
        weight_decay=DISABLED_WEIGHT_DECAY,
        dropout_rate=DISABLED_DROPOUT_RATE,
    ):
        """Use a supervised-learning checkpoint as the initial RL policy."""
        return cls._load_npz_weights(
            sl_weights_path,
            learning_rate,
            use_value_head=use_value_head,
            device=device,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
        )

    @classmethod
    def load(
        cls,
        rl_weights_path,
        learning_rate=0.001,
        use_value_head=False,
        device="auto",
        weight_decay=DISABLED_WEIGHT_DECAY,
        dropout_rate=DISABLED_DROPOUT_RATE,
    ):
        """Load policy weights and optionally restore a saved value head."""
        return cls._load_npz_weights(
            rl_weights_path,
            learning_rate,
            use_value_head=use_value_head,
            device=device,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
        )

    def save(self, weights_path):
        """Save policy/value weights and the state of the stateless SGD optimizer."""
        def to_numpy(matrix):
            return matrix.get() if hasattr(matrix, "get") else matrix

        weights_dir = os.path.dirname(weights_path)
        if weights_dir:
            os.makedirs(weights_dir, exist_ok=True)

        weight_names = self.weight_names
        if getattr(self, "use_value_head", False):
            weight_names += _VALUE_WEIGHTS

        arrays = {
            name: to_numpy(getattr(self, name))
            for name in weight_names
        }
        arrays[_OPTIMIZER_STEP_KEY] = np.asarray(
            getattr(self, "optimizer_step_count", 0),
            dtype=np.int64,
        )
        if hasattr(self, "rl_training_algorithm"):
            arrays[_ALGORITHM_KEY] = np.asarray(self.rl_training_algorithm)
        np.savez(weights_path, **arrays)

    def clone(self):
        """Return a frozen copy for the self-play opponent pool."""
        clone = PolicyNetwork(
            input_size=self.W1.shape[1],
            output_size=getattr(self, f"W{self.layer_count}").shape[0],
            hidden_sizes=self.hidden_sizes,
            learning_rate=self.lr,
            use_value_head=getattr(self, "use_value_head", False),
            device=self.device,
            weight_decay=self.weight_decay,
            dropout_rate=self.dropout_rate,
        )
        weight_names = clone.weight_names
        if clone.use_value_head:
            weight_names += _VALUE_WEIGHTS
        for name in weight_names:
            setattr(clone, name, getattr(self, name).copy())
        clone.optimizer_step_count = int(getattr(self, "optimizer_step_count", 0))
        if hasattr(self, "rl_training_algorithm"):
            clone.rl_training_algorithm = self.rl_training_algorithm
        return clone

    def optimizer_state_dict(self):
        """Return the complete state of the current plain-SGD optimizer."""
        return {
            "algorithm": "sgd",
            "learning_rate": float(self.lr),
            "step_count": int(getattr(self, "optimizer_step_count", 0)),
        }

    def load_optimizer_state_dict(self, state):
        """Restore and validate the plain-SGD optimizer state."""
        state = dict(state or {})
        if state.get("algorithm") != "sgd":
            raise ValueError(
                f"Unsupported RL optimizer state: {state.get('algorithm')!r}."
            )
        saved_lr = float(state["learning_rate"])
        if saved_lr != float(self.lr):
            raise ValueError(
                "RL optimizer learning rate does not match the checkpoint: "
                f"checkpoint={saved_lr}, requested={self.lr}."
            )
        self.optimizer_step_count = int(state["step_count"])

    def predict_values(self, x, training=False):
        """Return ``V(s)`` for each state column when the value head is enabled."""
        if not getattr(self, "use_value_head", False):
            raise RuntimeError("The value head is not enabled for this network.")
        self.forward(x, training=training)
        return self.xp.dot(
            self.Wv,
            self.cache[self.last_hidden_activation_key],
        ) + self.bv

    def evaluate_actions(self, x, legal_masks, action_indices, training=False):
        """Evaluate observed actions under the normalized masked policy.

        Returns one log-probability and entropy value per sample while leaving
        the forward cache ready for a subsequent policy-gradient update.
        Whole-buffer PPO metrics keep the default ``training=False`` so reported
        ratios, KL, and clipping describe the complete network.
        """
        xp = self.xp
        self.forward(x, training=training)
        logits = self.cache[self.logits_key]
        sample_count = logits.shape[1]
        action_indices = xp.asarray(action_indices, dtype=xp.int64).reshape(-1)
        legal_masks = xp.asarray(legal_masks, dtype=xp.bool_)
        if action_indices.shape[0] != sample_count:
            raise ValueError(
                "action_indices must contain one action per sample: "
                f"expected {sample_count}, got {action_indices.shape[0]}."
            )
        if legal_masks.shape != logits.shape:
            raise ValueError(
                "legal_masks must have the same shape as policy logits: "
                f"expected {logits.shape}, got {legal_masks.shape}."
            )
        legal_counts = xp.sum(legal_masks, axis=0)
        if self._as_float(xp.any(legal_counts < 2)):
            raise ValueError(
                "Every saved RL decision must have at least two legal policy actions."
            )
        columns = xp.arange(sample_count)
        if self._as_float(xp.any(~legal_masks[action_indices, columns])):
            raise ValueError("An observed PPO action is not legal under its saved mask.")

        masked_logits = xp.where(legal_masks, logits, -xp.inf)
        shifted = masked_logits - xp.max(masked_logits, axis=0, keepdims=True)
        exponentials = xp.exp(shifted)
        policy = exponentials / xp.sum(exponentials, axis=0, keepdims=True)
        probability_floor = xp.asarray(
            np.finfo(np.float32).tiny,
            dtype=policy.dtype,
        )
        log_policy = xp.log(xp.maximum(policy, probability_floor))
        log_probs = log_policy[action_indices, columns]
        entropy = -xp.sum(policy * log_policy, axis=0)
        return log_probs, entropy, policy

    def clipped_value_loss_terms(
        self,
        values,
        returns,
        old_values,
        clip_epsilon,
    ):
        """Return PPO-clipped value losses and their gradient per sample."""
        xp = self.xp
        delta = values - old_values
        clipped_values = old_values + xp.clip(
            delta,
            -float(clip_epsilon),
            float(clip_epsilon),
        )
        value_error = values - returns
        clipped_error = clipped_values - returns
        losses = 0.5 * value_error ** 2
        clipped_losses = 0.5 * clipped_error ** 2
        use_unclipped = losses >= clipped_losses
        inside_clip = (
            (delta >= -float(clip_epsilon))
            & (delta <= float(clip_epsilon))
        )
        gradient = xp.where(
            use_unclipped,
            value_error,
            clipped_error * inside_clip,
        )
        return xp.maximum(losses, clipped_losses), gradient, delta

    def _ppo_value_update_terms(
        self,
        last_hidden,
        returns,
        old_values,
        *,
        sample_count,
        dtype,
        value_coef,
        clip_epsilon,
    ):
        """Build critic metrics and gradients for one PPO minibatch."""
        if not getattr(self, "use_value_head", False):
            if returns is not None or old_values is not None:
                raise ValueError(
                    "PPO value targets were provided, but the value head is disabled."
                )
            return None
        if returns is None or old_values is None:
            raise ValueError(
                "PPO returns and old_values are required with a value head."
            )
        xp = self.xp
        returns = xp.asarray(returns, dtype=dtype).reshape(1, -1)
        old_values = xp.asarray(old_values, dtype=dtype).reshape(1, -1)
        if returns.shape[1] != sample_count or old_values.shape[1] != sample_count:
            raise ValueError("PPO returns and old_values must match the batch size.")
        values = xp.dot(self.Wv, last_hidden) + self.bv
        losses, value_gradient, value_delta = self.clipped_value_loss_terms(
            values,
            returns,
            old_values,
            clip_epsilon,
        )
        finite_values = xp.all(xp.isfinite(values)) & xp.all(xp.isfinite(losses))
        if not bool(self._as_float(finite_values)):
            raise FloatingPointError("PPO value loss produced NaN/Inf.")
        inverse_count = xp.asarray(1.0 / sample_count, dtype=dtype)
        weighted_gradient = float(value_coef) * value_gradient
        return {
            "loss": self._as_float(xp.mean(losses)),
            "clip_fraction": self._as_float(
                xp.mean(xp.abs(value_delta) > float(clip_epsilon))
            ),
            "gradients": {
                "Wv": inverse_count * xp.dot(weighted_gradient, last_hidden.T),
                "bv": inverse_count * xp.sum(
                    weighted_gradient,
                    axis=1,
                    keepdims=True,
                ),
            },
            "shared_hidden_gradient": xp.dot(self.Wv.T, weighted_gradient),
        }

    def _apply_gradient_step(self, gradients, grad_norm, clip_grad_norm, dtype):
        """Clip, apply, decay, and account for one plain-SGD optimizer step.

        Weight decay is decoupled: it shrinks the weight matrices after
        clipping instead of entering the gradient, so the reported gradient
        norms keep describing the policy/value gradient alone.
        """
        xp = self.xp
        grad_clipped = False
        applied_grad_norm = grad_norm
        if clip_grad_norm is not None and grad_norm > clip_grad_norm:
            scale = float(clip_grad_norm) / (grad_norm + 1e-8)
            gradients = {
                name: gradient * scale
                for name, gradient in gradients.items()
            }
            grad_clipped = True
            applied_grad_norm = float(clip_grad_norm)
        learning_rate = xp.asarray(self.lr, dtype=dtype)
        for name, gradient in gradients.items():
            setattr(self, name, getattr(self, name) - learning_rate * gradient)
        if self.weight_decay > 0.0:
            shrink = xp.asarray(
                1.0 - self.lr * self.weight_decay,
                dtype=dtype,
            )
            for name in self.decayed_weight_names:
                if name in gradients:
                    setattr(self, name, getattr(self, name) * shrink)
        self.optimizer_step_count = int(
            getattr(self, "optimizer_step_count", 0)
        ) + 1
        return {
            "grad_norm": grad_norm,
            "grad_clipped": grad_clipped,
            "applied_grad_norm": applied_grad_norm,
            "optimizer_steps": 1,
        }

    def backward_ppo(
        self,
        x,
        action_indices,
        legal_masks,
        old_log_probs,
        advantages,
        *,
        returns=None,
        old_values=None,
        value_coef=0.5,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        clip_grad_norm=5.0,
    ):
        """Apply one masked PPO clipped-surrogate SGD step.

        The returned timing detail uses only synchronization points that the
        optimizer already needed.  It therefore attributes asynchronous GPU
        work to the phase ending at the next existing scalar transfer without
        inserting profiler-only device synchronizations.
        """
        profile_started = time.perf_counter()
        timing = {}

        def finish_phase(name, started):
            timing[name] = timing.get(name, 0.0) + (
                time.perf_counter() - started
            )

        xp = self.xp
        phase_started = time.perf_counter()
        new_log_probs, entropy, masked_policy = self.evaluate_actions(
            x,
            legal_masks,
            action_indices,
            training=True,
        )
        finish_phase("policy_forward_and_action_mask_validation", phase_started)
        phase_started = time.perf_counter()
        sample_count = int(new_log_probs.shape[0])
        action_indices = xp.asarray(action_indices, dtype=xp.int64).reshape(-1)
        old_log_probs = xp.asarray(
            old_log_probs,
            dtype=self.cache[self.logits_key].dtype,
        ).reshape(-1)
        advantages = xp.asarray(
            advantages,
            dtype=self.cache[self.logits_key].dtype,
        ).reshape(-1)
        if old_log_probs.shape[0] != sample_count or advantages.shape[0] != sample_count:
            raise ValueError("PPO old_log_probs and advantages must match the batch size.")
        finish_phase("batch_conversion_and_shape_validation", phase_started)

        phase_started = time.perf_counter()
        log_ratio = new_log_probs - old_log_probs
        ratio = xp.exp(log_ratio)
        lower = 1.0 - float(clip_epsilon)
        upper = 1.0 + float(clip_epsilon)
        clipped_ratio = xp.clip(ratio, lower, upper)
        unclipped = ratio * advantages
        clipped = clipped_ratio * advantages
        surrogate = xp.minimum(unclipped, clipped)
        policy_loss = -xp.mean(surrogate)

        # Where the clipped branch is strictly smaller, its derivative with
        # respect to theta is zero. Else d[-ratio*A]/dlogpi = -ratio*A.
        active_weights = xp.where(unclipped <= clipped, ratio * advantages, 0.0)
        sampled = xp.zeros_like(masked_policy)
        sampled[action_indices, xp.arange(sample_count)] = 1.0
        entropy_row = entropy.reshape(1, -1)
        probability_floor = xp.asarray(
            np.finfo(np.float32).tiny,
            dtype=masked_policy.dtype,
        )
        log_policy = xp.log(xp.maximum(masked_policy, probability_floor))
        dz3_policy = (masked_policy - sampled) * active_weights.reshape(1, -1)
        dz3_entropy = masked_policy * (log_policy + entropy_row)
        dz3 = dz3_policy + float(entropy_coef) * dz3_entropy

        last_hidden = self.cache[self.last_hidden_activation_key]
        inverse_count = xp.asarray(1.0 / sample_count, dtype=dz3.dtype)

        value_terms = self._ppo_value_update_terms(
            last_hidden,
            returns,
            old_values,
            sample_count=sample_count,
            dtype=dz3.dtype,
            value_coef=value_coef,
            clip_epsilon=clip_epsilon,
        )
        gradients = self.backpropagate_layers(
            dz3,
            inverse_count,
            hidden_gradient_hook=(
                None if value_terms is None
                else lambda gradient: gradient + value_terms["shared_hidden_gradient"]
            ),
        )
        if value_terms is not None:
            gradients.update(value_terms["gradients"])
        grad_norm = self._as_float(
            xp.sqrt(sum(xp.sum(gradient ** 2) for gradient in gradients.values()))
        )
        finish_phase(
            "clipped_surrogate_backpropagation_and_gradient_norm",
            phase_started,
        )
        phase_started = time.perf_counter()
        update_metrics = self._apply_gradient_step(
            gradients,
            grad_norm,
            clip_grad_norm,
            dz3.dtype,
        )

        clip_fraction = xp.mean((ratio < lower) | (ratio > upper))
        approx_kl = xp.mean((ratio - 1.0) - log_ratio)
        result = {
            "policy_loss": self._as_float(policy_loss),
            "entropy": self._as_float(xp.mean(entropy)),
            "approx_kl": self._as_float(approx_kl),
            "clip_fraction": self._as_float(clip_fraction),
            "ratio_mean": self._as_float(xp.mean(ratio)),
            "ratio_min": self._as_float(xp.min(ratio)),
            "ratio_max": self._as_float(xp.max(ratio)),
            **update_metrics,
            "value_loss": None if value_terms is None else value_terms["loss"],
            "value_clip_fraction": (
                None if value_terms is None else value_terms["clip_fraction"]
            ),
        }
        finish_phase(
            "gradient_clipping_parameter_update_and_metric_transfers",
            phase_started,
        )
        total_seconds = time.perf_counter() - profile_started
        timing["unaccounted"] = max(0.0, total_seconds - sum(timing.values()))
        result["runtime_profile_detail"] = {
            "calls": 1,
            "execution_seconds": float(total_seconds),
            "gpu_calls": int(self.device == "gpu"),
            "cpu_calls": int(self.device == "cpu"),
            "sections_seconds": {
                name: float(seconds) for name, seconds in timing.items()
            },
            "device": self.device,
        }
        return result

    def backward_policy_gradient(
        self,
        action_indices,
        policy_rewards,
        legal_masks,
        entropy_coef=0.01,
        clip_grad_norm=5.0,
        value_returns=None,
        value_coef=0.5,
    ):
        """Apply masked REINFORCE, optionally updating a value baseline."""
        xp = self.xp
        z3 = self.cache[self.logits_key]
        last_hidden = self.cache[self.last_hidden_activation_key]
        m = z3.shape[1]

        action_indices = xp.asarray(action_indices, dtype=xp.int64).reshape(-1)
        policy_rewards = xp.asarray(
            policy_rewards,
            dtype=z3.dtype,
        ).reshape(1, m)
        legal_masks = (xp.asarray(legal_masks) > 0).astype(z3.dtype)

        if action_indices.shape[0] != m:
            raise ValueError(
                "action_indices must contain one action per cached sample: "
                f"expected {m}, got {action_indices.shape[0]}."
            )

        if legal_masks.shape != z3.shape:
            raise ValueError(
                "legal_masks must have the same shape as the policy logits: "
                f"expected {z3.shape}, got {legal_masks.shape}."
            )

        legal_counts = xp.sum(legal_masks, axis=0)
        if self._as_float(xp.any(legal_counts < 2)):
            raise ValueError(
                "Every saved RL decision must have at least two legal policy actions."
            )

        chosen_action_is_legal = legal_masks[action_indices, xp.arange(m)]
        if self._as_float(xp.any(chosen_action_is_legal < 0.5)):
            raise ValueError(
                "A sampled action is not marked as legal in its action mask."
            )

        masked_logits = xp.where(legal_masks > 0, z3, -xp.inf)
        max_legal_logits = xp.max(masked_logits, axis=0, keepdims=True)
        shifted_logits = masked_logits - max_legal_logits
        exp_logits = xp.exp(shifted_logits)
        masked_policy = exp_logits / xp.sum(exp_logits, axis=0, keepdims=True)

        sampled_y = xp.zeros_like(masked_policy)
        sampled_y[action_indices, xp.arange(m)] = 1.0

        dz3_policy = (masked_policy - sampled_y) * policy_rewards

        log_masked_policy = xp.log(masked_policy + 1e-8)
        entropy = -xp.sum(masked_policy * log_masked_policy, axis=0, keepdims=True)
        dz3_entropy = masked_policy * (log_masked_policy + entropy)
        dz3 = dz3_policy + entropy_coef * dz3_entropy

        value_loss = None
        value_gradients = {}
        shared_hidden_gradient = None
        use_value_head = getattr(self, "use_value_head", False)
        if use_value_head:
            if value_returns is None:
                raise ValueError(
                    "value_returns are required when the value head is enabled."
                )
            value_returns = xp.asarray(
                value_returns,
                dtype=z3.dtype,
            ).reshape(1, m)
            values = xp.dot(self.Wv, last_hidden) + self.bv
            value_error = values - value_returns
            value_loss = float(xp.mean(0.5 * value_error ** 2))

            dzv = value_coef * value_error
            value_gradients = {
                "Wv": (1.0 / m) * xp.dot(dzv, last_hidden.T),
                "bv": (1.0 / m) * xp.sum(dzv, axis=1, keepdims=True),
            }
            shared_hidden_gradient = xp.dot(self.Wv.T, dzv)
        elif value_returns is not None:
            raise ValueError(
                "value_returns were provided, but the value head is disabled."
            )

        gradients = self.backpropagate_layers(
            dz3,
            1.0 / m,
            hidden_gradient_hook=(
                None if shared_hidden_gradient is None
                else lambda gradient: gradient + shared_hidden_gradient
            ),
        )
        gradients.update(value_gradients)

        grad_norm = float(xp.sqrt(sum(xp.sum(grad ** 2) for grad in gradients.values())))
        update_metrics = self._apply_gradient_step(
            gradients,
            grad_norm,
            clip_grad_norm,
            z3.dtype,
        )

        return {
            "entropy": float(xp.mean(entropy)),
            "grad_norm": update_metrics["grad_norm"],
            "grad_clipped": update_metrics["grad_clipped"],
            "applied_grad_norm": update_metrics["applied_grad_norm"],
            "value_loss": value_loss,
        }
