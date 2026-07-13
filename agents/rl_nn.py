"""Policy-only network used by reinforcement-learning self-play."""

import os

import numpy as np

from agents.nn import GPU_ENABLED, SupervisedNeuralNetwork

if GPU_ENABLED:
    import cupy as xp
else:
    import numpy as xp

_POLICY_WEIGHTS = ("W1", "b1", "W2", "b2", "W3", "b3")


class PolicyNetwork(SupervisedNeuralNetwork):
    """Supervised policy architecture updated with masked REINFORCE gradients.

    The network keeps exactly the same weights as the supervised policy:
    ``W1``, ``b1``, ``W2``, ``b2``, ``W3``, and ``b3``. Reinforcement learning
    applies policy-gradient updates directly from the reward assigned to each
    real decision. The checkpoint format contains only the six policy weights.
    """

    @classmethod
    def _load_npz_weights(cls, path, learning_rate):
        data = np.load(path)
        hidden1_size, input_size = data["W1"].shape
        hidden2_size, _ = data["W2"].shape
        output_size, _ = data["W3"].shape

        network = cls(
            input_size=input_size,
            hidden1_size=hidden1_size,
            hidden2_size=hidden2_size,
            output_size=output_size,
            learning_rate=learning_rate,
        )
        for name in _POLICY_WEIGHTS:
            setattr(network, name, xp.array(data[name]))
        return network

    @classmethod
    def load_from_sl(cls, sl_weights_path="models/domino_sl_weights.npz", learning_rate=0.001):
        """Use a supervised-learning checkpoint as the initial RL policy."""
        return cls._load_npz_weights(sl_weights_path, learning_rate)

    @classmethod
    def load(cls, rl_weights_path, learning_rate=0.001):
        """Load a policy checkpoint, ignoring stale extra keys if present."""
        return cls._load_npz_weights(rl_weights_path, learning_rate)

    def save(self, weights_path):
        """Save only the six policy weights shared with supervised checkpoints."""
        def to_numpy(matrix):
            return matrix.get() if hasattr(matrix, "get") else matrix

        weights_dir = os.path.dirname(weights_path)
        if weights_dir:
            os.makedirs(weights_dir, exist_ok=True)

        np.savez(
            weights_path,
            **{name: to_numpy(getattr(self, name)) for name in _POLICY_WEIGHTS},
        )

    def clone(self):
        """Return a frozen copy for the self-play opponent pool."""
        clone = PolicyNetwork(
            input_size=self.W1.shape[1],
            hidden1_size=self.W1.shape[0],
            hidden2_size=self.W2.shape[0],
            output_size=self.W3.shape[0],
            learning_rate=self.lr,
        )
        for name in _POLICY_WEIGHTS:
            setattr(clone, name, getattr(self, name).copy())
        return clone

    def backward_policy_gradient(
        self,
        action_indices,
        policy_rewards,
        legal_masks,
        entropy_coef=0.01,
        clip_grad_norm=5.0,
    ):
        """Apply one masked REINFORCE update using direct reward weights."""
        z3 = self.cache["Z3"]
        a2 = self.cache["A2"]
        a1 = self.cache["A1"]
        x = self.cache["X"]
        m = z3.shape[1]

        action_indices = xp.asarray(action_indices, dtype=xp.int64).reshape(-1)
        policy_rewards = xp.asarray(policy_rewards).reshape(1, m)
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
        }
