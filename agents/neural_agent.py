"""Inference agent for the supervised-learning policy network."""

import random

import numpy as np

from agents.encoder import DominoEncoder
from agents.network_architecture import architecture_from_weights
from agents.nn import SupervisedNeuralNetwork
from middleware.middleware import Agent
from middleware.opponent_model import ExactOpponentModel
from middleware.rulesets import DEFAULT_RULESET_NAME, resolve_ruleset
from utils.ruleset_paths import default_sl_weights_path


class NeuralAgent(Agent):
    """Choose real tile-play decisions from a supervised policy checkpoint.

    Draw, pass, and single-option tile plays are forced by the rules engine and
    bypass both opponent inference and the neural network.

    ``use_opponent_suit_features=False`` selects the ablated encoder layout and
    skips exact opponent inference entirely, so the checkpoint it loads must
    have been trained with the shortened input.
    """

    def __init__(
        self,
        network,
        epsilon=0.0,
        *,
        ruleset=DEFAULT_RULESET_NAME,
        use_opponent_suit_features=True,
        use_opponent_bucket_features=False,
        opponent_bucket=None,
    ):
        self.ruleset = resolve_ruleset(ruleset)
        self.network = network
        self.epsilon = epsilon
        self.use_opponent_suit_features = bool(use_opponent_suit_features)
        self.use_opponent_bucket_features = bool(use_opponent_bucket_features)
        self.opponent_bucket = opponent_bucket
        self.encoder = DominoEncoder(
            self.ruleset,
            use_opponent_suit_features=self.use_opponent_suit_features,
            use_opponent_bucket_features=self.use_opponent_bucket_features,
            opponent_bucket=opponent_bucket,
        )
        # The model exists only to fill the encoder's trailing block. With that
        # block ablated nothing would read its output, so it is never built.
        self.opponent_model = (
            ExactOpponentModel(ruleset=self.ruleset, record_traces=False)
            if self.use_opponent_suit_features
            else None
        )

    @classmethod
    def load(
        cls,
        weights_path=None,
        epsilon=0.0,
        device="auto",
        ruleset=DEFAULT_RULESET_NAME,
        use_opponent_suit_features=True,
        use_opponent_bucket_features=False,
        opponent_bucket=None,
    ):
        """Build an agent from a NumPy ``.npz`` checkpoint."""
        weights_path = weights_path or default_sl_weights_path(ruleset)
        with np.load(weights_path, allow_pickle=False) as data:
            # The checkpoint is the single source of truth for depth and
            # widths, so a network trained with --hidden-layers loads here
            # without any agent-side configuration.
            architecture = architecture_from_weights(data)
            input_size = architecture.input_size
            output_size = architecture.output_size

            encoder = DominoEncoder(
                ruleset,
                use_opponent_suit_features=use_opponent_suit_features,
                use_opponent_bucket_features=use_opponent_bucket_features,
                opponent_bucket=opponent_bucket,
            )
            if input_size != encoder.vector_size:
                raise ValueError(
                    f"Checkpoint expects input_size={input_size}, "
                    f"but {encoder.ruleset.name} produces {encoder.vector_size}."
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

        return cls(
            network,
            epsilon=epsilon,
            ruleset=ruleset,
            use_opponent_suit_features=use_opponent_suit_features,
            use_opponent_bucket_features=use_opponent_bucket_features,
            opponent_bucket=opponent_bucket,
        )

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

        if self.opponent_model is not None:
            state["opponent_suit_probabilities"] = self.opponent_model.update(state)
        probabilities = self.network.forward(self.encoder.encode_state(state))
        if hasattr(probabilities, "get"):
            probabilities = probabilities.get()

        return self.encoder.decode_output(probabilities, policy_actions)
