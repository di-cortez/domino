"""Masked policy network with an optional training-only value head."""

import os

import numpy as np

from agents.nn import (
    DEVICES,
    SupervisedNeuralNetwork,
)

_POLICY_WEIGHTS = ("W1", "b1", "W2", "b2", "W3", "b3")
_VALUE_WEIGHTS = ("Wv", "bv")


class PolicyNetwork(SupervisedNeuralNetwork):
    """Supervised policy architecture with masked REINFORCE gradients.

    Direct REINFORCE is the default and keeps exactly the six supervised policy
    weights. ``use_value_head=True`` adds a linear ``V(s)`` head over the second
    hidden layer so the policy can train from reward-minus-value advantages.

    ``device`` selects the array backend independently of the parent class:
    ``"auto"`` (default) matches ``GPU_ENABLED`` exactly, reproducing prior
    behavior; ``"cpu"``/``"gpu"`` force one backend regardless of what's
    installed/enabled globally.
    """

    def __init__(self, *args, use_value_head=False, device="auto", **kwargs):
        super().__init__(*args, device=device, **kwargs)
        self.use_value_head = use_value_head
        if use_value_head:
            hidden2_size = self.W3.shape[1]
            self.Wv = self.xp.zeros(
                (1, hidden2_size),
                dtype=self.xp.float32,
            )
            self.bv = self.xp.zeros((1, 1), dtype=self.xp.float32)

    def _cast_weights_to_device(self):
        """Move the six policy weights (built by the parent class) to ``self.xp``."""
        for name in _POLICY_WEIGHTS:
            value = getattr(self, name, None)
            if value is None:
                continue
            if hasattr(value, "get"):
                value = value.get()
            setattr(self, name, self.xp.asarray(value, dtype=self.xp.float32))

    def forward(self, x):
        """Same three-layer forward pass as the parent class, pinned to ``self.xp``."""
        x = self.xp.asarray(x, dtype=self.xp.float32)
        z1 = self.xp.dot(self.W1, x) + self.b1
        a1 = self.xp.maximum(0, z1)
        z2 = self.xp.dot(self.W2, a1) + self.b2
        a2 = self.xp.maximum(0, z2)
        z3 = self.xp.dot(self.W3, a2) + self.b3
        exp_z = self.xp.exp(z3 - self.xp.max(z3, axis=0, keepdims=True))
        a3 = exp_z / self.xp.sum(exp_z, axis=0, keepdims=True)

        self.cache = {"X": x, "Z1": z1, "A1": a1, "Z2": z2, "A2": a2, "Z3": z3, "A3": a3}
        return a3

    @classmethod
    def _load_npz_weights(cls, path, learning_rate, use_value_head=False, device="auto", data=None):
        """Build a network from an ``.npz`` checkpoint.

        ``data`` accepts an already-loaded mapping of arrays (e.g. a plain
        dict, or an open ``np.load`` result) to skip reading ``path`` from
        disk again -- see ``load_from_sl``'s ``data`` parameter for the
        caller-facing version of this.
        """
        owns_data = data is None
        if owns_data:
            data = np.load(path, allow_pickle=False)
        try:
            hidden1_size, input_size = data["W1"].shape
            hidden2_size, _ = data["W2"].shape
            output_size, _ = data["W3"].shape

            network = cls(
                input_size=input_size,
                hidden1_size=hidden1_size,
                hidden2_size=hidden2_size,
                output_size=output_size,
                learning_rate=learning_rate,
                use_value_head=use_value_head,
                device=device,
            )
            for name in _POLICY_WEIGHTS:
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
            return network
        finally:
            if owns_data:
                data.close()

    @classmethod
    def load_from_sl(
        cls,
        sl_weights_path="models/domino_sl_weights.npz",
        learning_rate=0.001,
        use_value_head=False,
        device="auto",
        data=None,
    ):
        """Use a supervised-learning checkpoint as the initial RL policy.

        Pass a pre-loaded ``data`` mapping (e.g. from
        ``np.load(sl_weights_path)`` or a plain ``{name: array}`` dict) to
        initialize many networks from the same checkpoint without re-reading
        it from disk each time -- useful for a hyperparameter sweep that
        warm-starts every run from one shared SL checkpoint.
        """
        return cls._load_npz_weights(
            sl_weights_path,
            learning_rate,
            use_value_head=use_value_head,
            device=device,
            data=data,
        )

    @classmethod
    def load(cls, rl_weights_path, learning_rate=0.001, use_value_head=False, device="auto"):
        """Load policy weights and optionally restore a saved value head."""
        return cls._load_npz_weights(
            rl_weights_path,
            learning_rate,
            use_value_head=use_value_head,
            device=device,
        )

    def save(self, weights_path):
        """Save only the six policy weights shared with supervised checkpoints."""
        def to_numpy(matrix):
            return matrix.get() if hasattr(matrix, "get") else matrix

        weights_dir = os.path.dirname(weights_path)
        if weights_dir:
            os.makedirs(weights_dir, exist_ok=True)

        weight_names = _POLICY_WEIGHTS
        if getattr(self, "use_value_head", False):
            weight_names += _VALUE_WEIGHTS

        np.savez(weights_path, **{
            name: to_numpy(getattr(self, name))
            for name in weight_names
        })

    def clone(self):
        """Return a frozen copy for the self-play opponent pool."""
        clone = PolicyNetwork(
            input_size=self.W1.shape[1],
            hidden1_size=self.W1.shape[0],
            hidden2_size=self.W2.shape[0],
            output_size=self.W3.shape[0],
            learning_rate=self.lr,
            use_value_head=getattr(self, "use_value_head", False),
            device=self.device,
        )
        weight_names = _POLICY_WEIGHTS
        if clone.use_value_head:
            weight_names += _VALUE_WEIGHTS
        for name in weight_names:
            setattr(clone, name, getattr(self, name).copy())
        return clone

    def predict_values(self, x):
        """Return ``V(s)`` for each state column when the value head is enabled."""
        if not getattr(self, "use_value_head", False):
            raise RuntimeError("The value head is not enabled for this network.")
        self.forward(x)
        return self.xp.dot(self.Wv, self.cache["A2"]) + self.bv

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
        z3 = self.cache["Z3"]
        a2 = self.cache["A2"]
        a1 = self.cache["A1"]
        x = self.cache["X"]
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

        dW3 = (1.0 / m) * xp.dot(dz3, a2.T)
        db3 = (1.0 / m) * xp.sum(dz3, axis=1, keepdims=True)
        da2 = xp.dot(self.W3.T, dz3)

        value_loss = None
        value_gradients = {}
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
            values = xp.dot(self.Wv, a2) + self.bv
            value_error = values - value_returns
            value_loss = float(xp.mean(0.5 * value_error ** 2))

            dzv = value_coef * value_error
            value_gradients = {
                "Wv": (1.0 / m) * xp.dot(dzv, a2.T),
                "bv": (1.0 / m) * xp.sum(dzv, axis=1, keepdims=True),
            }
            da2 = da2 + xp.dot(self.Wv.T, dzv)
        elif value_returns is not None:
            raise ValueError(
                "value_returns were provided, but the value head is disabled."
            )

        dz2 = da2 * self.relu_derivative(self.cache["Z2"])
        dW2 = (1.0 / m) * xp.dot(dz2, a1.T)
        db2 = (1.0 / m) * xp.sum(dz2, axis=1, keepdims=True)

        da1 = xp.dot(self.W2.T, dz2)
        dz1 = da1 * self.relu_derivative(self.cache["Z1"])
        dW1 = (1.0 / m) * xp.dot(dz1, x.T)
        db1 = (1.0 / m) * xp.sum(dz1, axis=1, keepdims=True)

        gradients = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3,
        }
        gradients.update(value_gradients)

        grad_norm = float(xp.sqrt(sum(xp.sum(grad ** 2) for grad in gradients.values())))
        grad_clipped = False
        applied_grad_norm = grad_norm
        if clip_grad_norm is not None and grad_norm > clip_grad_norm:
            scale = clip_grad_norm / (grad_norm + 1e-8)
            gradients = {name: grad * scale for name, grad in gradients.items()}
            grad_clipped = True
            applied_grad_norm = float(clip_grad_norm)

        for name, grad in gradients.items():
            setattr(self, name, getattr(self, name) - self.lr * grad)

        return {
            "entropy": float(xp.mean(entropy)),
            "grad_norm": grad_norm,
            "grad_clipped": grad_clipped,
            "applied_grad_norm": applied_grad_norm,
            "value_loss": value_loss,
        }
