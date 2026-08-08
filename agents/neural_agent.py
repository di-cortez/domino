"""Inference agent for the supervised-learning policy network."""

import random

import numpy as np

from agents.encoder import DominoEncoder
from agents.network_architecture import architecture_from_weights
from agents.nn import SupervisedNeuralNetwork
from middleware.middleware import Agent
from middleware.opponent_model import ExactOpponentModel


class NeuralAgent(Agent):
    """Choose real tile-play decisions from a supervised policy checkpoint.

    Draw, pass, and single-option tile plays are forced by the rules engine and
    bypass both opponent inference and the neural network.
    """

    def __init__(self, network, epsilon=0.0):
        self.network = network
        self.epsilon = epsilon
        self.encoder = DominoEncoder()
        self.opponent_model = ExactOpponentModel(record_traces=False)

    @classmethod
    def load(
        cls,
        weights_path="models/domino_sl_weights.npz",
        epsilon=0.0,
        device="auto",
    ):
        """Build an agent from a NumPy ``.npz`` checkpoint."""
        with np.load(weights_path, allow_pickle=False) as data:
            # The checkpoint is the single source of truth for depth and
            # widths, so a network trained with --hidden-layers loads here
            # without any agent-side configuration.
            architecture = architecture_from_weights(data)
            input_size = architecture.input_size
            output_size = architecture.output_size

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
                output_size=output_size,
                hidden_sizes=architecture.hidden_sizes,
                device=device,
            )
            network.load_policy_weights(data)

        return cls(network, epsilon=epsilon)

    def choose_move(self, state, legal_actions):
        if not legal_actions:
            return None

        policy_actions = [move for move in legal_actions if self.encoder.is_policy_action(move)]
        if not policy_actions:
            return legal_actions[0]

        if len(policy_actions) == 1:
            return policy_actions[0]

        if self.epsilon > 0.0 and np.random.rand() < self.epsilon:
            return random.choice(policy_actions)

        state["opponent_suit_probabilities"] = self.opponent_model.update(state)
        probabilities = self.network.forward(self.encoder.encode_state(state))
        if hasattr(probabilities, "get"):
            probabilities = probabilities.get()

        return self.encoder.decode_output(probabilities, policy_actions)
