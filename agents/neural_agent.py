"""Inference agent for the supervised-learning policy network."""

import random

import numpy as np

from agents.agent import Agent
from agents.encoder import DominoEncoder
from agents.nn import GPU_ENABLED, SupervisedNeuralNetwork

if GPU_ENABLED:
    import cupy as xp
else:
    xp = np


class NeuralAgent(Agent):
    """Load an SL checkpoint and choose moves with legal-action masking."""

    def __init__(self, network, epsilon=0.0):
        self.network = network
        self.epsilon = epsilon
        self.encoder = DominoEncoder()

    @classmethod
    def load(cls, weights_path="models/domino_sl_weights.npz", epsilon=0.0):
        """Build an agent from a NumPy ``.npz`` checkpoint."""
        data = np.load(weights_path)

        hidden1_size, input_size = data["W1"].shape
        hidden2_size, _ = data["W2"].shape
        output_size, _ = data["W3"].shape

        encoder = DominoEncoder()
        if input_size != encoder.VECTOR_SIZE:
            raise ValueError(
                f"Checkpoint expects input_size={input_size}, "
                f"but DominoEncoder produces {encoder.VECTOR_SIZE}."
            )
        if output_size != len(encoder.all_actions):
            raise ValueError(
                f"Checkpoint output_size={output_size}, "
                f"but the action space has {len(encoder.all_actions)} actions."
            )

        network = SupervisedNeuralNetwork(
            input_size=input_size,
            hidden1_size=hidden1_size,
            hidden2_size=hidden2_size,
            output_size=output_size,
        )
        for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
            setattr(network, name, xp.array(data[name]))

        return cls(network, epsilon=epsilon)

    def choose_move(self, state, legal_actions):
        if not legal_actions:
            return None

        if self.epsilon > 0.0 and np.random.rand() < self.epsilon:
            return random.choice(legal_actions)

        x = self.encoder.encode_state(state)
        if GPU_ENABLED:
            x = xp.array(x)

        probabilities = self.network.forward(x)
        if hasattr(probabilities, "get"):
            probabilities = probabilities.get()

        print(f"Possible neural moves: {legal_actions}")
        return self.encoder.decode_output(probabilities, legal_actions)
