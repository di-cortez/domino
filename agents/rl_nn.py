"""Policy network used by reinforcement-learning self-play."""

import os

import numpy as np

from agents.nn import GPU_ENABLED, SupervisedNeuralNetwork

if GPU_ENABLED:
    import cupy as xp
else:
    import numpy as xp

_POLICY_WEIGHTS = ("W1", "b1", "W2", "b2", "W3", "b3")
_VALUE_WEIGHTS = ("Wv", "bv")


class PolicyNetwork(SupervisedNeuralNetwork):
    """
    Supervised policy architecture extended with a linear value head.

    The policy head is the inherited 58-action softmax. The value head predicts
    ``V(s)`` from the second hidden layer and is used as a state-dependent
    baseline for REINFORCE updates.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden2_size = self.W3.shape[1]
        self.Wv = xp.zeros((1, hidden2_size))
        self.bv = xp.zeros((1, 1))

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

        if all(name in data for name in _VALUE_WEIGHTS):
            for name in _VALUE_WEIGHTS:
                setattr(network, name, xp.array(data[name]))
        return network

    @classmethod
    def load_from_sl(cls, sl_weights_path="models/domino_sl_weights.npz", learning_rate=0.001):
        """Use a supervised-learning checkpoint as the initial RL policy."""
        return cls._load_npz_weights(sl_weights_path, learning_rate)

    @classmethod
    def load(cls, rl_weights_path, learning_rate=0.001):
        """Load an RL checkpoint saved by ``save``."""
        return cls._load_npz_weights(rl_weights_path, learning_rate)

    def save(self, weights_path):
        def to_numpy(matrix):
            return matrix.get() if hasattr(matrix, "get") else matrix

        weights_dir = os.path.dirname(weights_path)
        if weights_dir:
            os.makedirs(weights_dir, exist_ok=True)

        np.savez(
            weights_path,
            **{name: to_numpy(getattr(self, name)) for name in _POLICY_WEIGHTS + _VALUE_WEIGHTS},
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
        for name in _POLICY_WEIGHTS + _VALUE_WEIGHTS:
            setattr(clone, name, getattr(self, name).copy())
        return clone

    def predict_values(self, x):
        """Return value estimates for every column in ``x``."""
        self.forward(x)
        return xp.dot(self.Wv, self.cache["A2"]) + self.bv

    def backward_policy_gradient(
        self,
        action_indices,
        advantages,
        returns=None,
        entropy_coef=0.01,
        value_coef=0.5,
        clip_grad_norm=5.0,
    ):
        """Apply one actor-critic policy-gradient update."""
        a3 = self.cache["A3"]
        a2 = self.cache["A2"]
        a1 = self.cache["A1"]
        x = self.cache["X"]
        m = a3.shape[1]

        action_indices = xp.asarray(action_indices)
        advantages = xp.asarray(advantages).reshape(1, m)

        sampled_y = xp.zeros_like(a3)
        sampled_y[action_indices, xp.arange(m)] = 1.0

        dz3_policy = (a3 - sampled_y) * advantages

        log_a3 = xp.log(a3 + 1e-8)
        entropy = -xp.sum(a3 * log_a3, axis=0, keepdims=True)
        dz3_entropy = a3 * (log_a3 + entropy)
        dz3 = dz3_policy + entropy_coef * dz3_entropy

        dW3 = (1.0 / m) * xp.dot(dz3, a2.T)
        db3 = (1.0 / m) * xp.sum(dz3, axis=1, keepdims=True)
        da2 = xp.dot(self.W3.T, dz3)

        value_loss = 0.0
        dWv = xp.zeros_like(self.Wv)
        dbv = xp.zeros_like(self.bv)
        if returns is not None:
            returns = xp.asarray(returns).reshape(1, m)
            values = xp.dot(self.Wv, a2) + self.bv
            value_error = values - returns
            value_loss = float(xp.mean(0.5 * value_error ** 2))

            dzv = value_coef * value_error
            dWv = (1.0 / m) * xp.dot(dzv, a2.T)
            dbv = (1.0 / m) * xp.sum(dzv, axis=1, keepdims=True)
            da2 = da2 + xp.dot(self.Wv.T, dzv)

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
            "Wv": dWv,
            "bv": dbv,
        }

        grad_norm = float(xp.sqrt(sum(xp.sum(grad ** 2) for grad in gradients.values())))
        if clip_grad_norm is not None and grad_norm > clip_grad_norm:
            scale = clip_grad_norm / (grad_norm + 1e-8)
            gradients = {name: grad * scale for name, grad in gradients.items()}

        for name, grad in gradients.items():
            setattr(self, name, getattr(self, name) - self.lr * grad)

        return {
            "entropy": float(xp.mean(entropy)),
            "value_loss": value_loss,
            "grad_norm": grad_norm,
        }
