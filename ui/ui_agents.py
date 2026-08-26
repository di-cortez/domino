"""
UI-facing agent factory.

The visual interface identifies players with small string IDs:

    "neural", "heuristic", "random", "human", "rl"

This module is the only place that translates those IDs into objects accepted
by `GameManager`. Keeping that mapping here prevents the controller and HUD
from importing every concrete agent implementation directly.
"""

import random

from middleware.rulesets import DEFAULT_RULESET_NAME, resolve_ruleset
from utils.ruleset_paths import default_rl_weights_path, default_sl_weights_path


AGENT_TYPES = ("neural", "heuristic", "random", "human", "rl")


# Bucket handed to a policy trained with --opponent-bucket-features.
#
# The block is one-hot in every vector such a policy ever trained on, so an
# all-zero block is not a neutral state but a pattern it has never seen. A human
# genuinely belongs to no bucket, yet "no bucket" is not a point of the input
# distribution, so the choice is between a pattern that never occurred and a
# label that occurred constantly. The heuristic slot is the closest honest
# stand-in: it is the bucket the supervised dataset is encoded with
# (`training.supervised.training_loop.SUPERVISED_TEACHER_OPPONENT_BUCKET`), so
# every checkpoint has seen it from its first gradient step onward. It is still
# a stand-in -- a human does not play like `StrategicAgent` -- and no bucket
# value makes the conditioning transfer to human play.
UI_OPPONENT_BUCKET = "heuristic"


def _bucket_features_for(input_size, ruleset):
    """Return the bucket-block encoder arguments one input width implies.

    The UI has no run configuration to read the encoder ablations from, so a
    checkpoint's own input width is the only evidence available. Only the
    widened layout is decided here: a vector one bucket block longer than the
    default was trained with the block and has to be given one.

    A default-width vector keeps the historical layout. That reading is
    genuinely ambiguous -- for double-six `--no-opponent-suit-features
    --opponent-bucket-features` is also 168, with seven completely different
    trailing features -- and no width can separate the two, so the layout that
    predates both flags wins. Runs that combine those flags are not loadable
    from the UI menu.
    """
    from agents.encoder import DominoEncoder, OPPONENT_BUCKET_FEATURE_WIDTH

    default_size = DominoEncoder(ruleset).vector_size
    if input_size == default_size + OPPONENT_BUCKET_FEATURE_WIDTH:
        return {
            "use_opponent_bucket_features": True,
            "opponent_bucket": UI_OPPONENT_BUCKET,
        }
    return {}


def _checkpoint_input_size(weights_path):
    """Return the input width one policy checkpoint was trained with."""
    import numpy as np

    from agents.network_architecture import architecture_from_weights

    with np.load(weights_path, allow_pickle=False) as data:
        return architecture_from_weights(data).input_size


class RandomUIAgent:
    """Simple UI-only agent that chooses uniformly among legal actions."""

    def choose_move(self, state, legal_actions):
        return random.choice(legal_actions)


class BlockedHumanAgent:
    """
    Sentinel for human players.

    Human turns are executed directly by `GameController` after keyboard input
    and engine validation. If `GameManager` ever calls this object, the UI flow
    is wrong and should fail loudly.
    """

    def choose_move(self, state, legal_actions):
        raise RuntimeError("Human turns must be handled by the UI controller.")


def agent_type_name(agent_type):
    """Friendly label used by the HUD and notifications."""
    names = {
        "neural": "Neural",
        "heuristic": "Heuristic",
        "random": "Random",
        "human": "Human",
        "rl": "RL (self-play)",
    }
    return names.get(agent_type, agent_type.capitalize())


def create_agent_by_type(agent_type, ruleset=DEFAULT_RULESET_NAME):
    """
    Build the agent instance selected in the UI menu.

    Imports stay inside the function so that unused neural/RL dependencies are
    not loaded when the user chooses a simpler agent type.
    """
    ruleset = resolve_ruleset(ruleset)
    if agent_type == "neural":
        from agents.neural_agent import NeuralAgent

        weights_path = default_sl_weights_path(ruleset)
        return NeuralAgent.load(
            weights_path,
            ruleset=ruleset,
            **_bucket_features_for(
                _checkpoint_input_size(weights_path), ruleset
            ),
        )

    if agent_type == "heuristic":
        from agents.heuristic_agent import StrategicAgent

        return StrategicAgent(ruleset=ruleset)

    if agent_type == "random":
        return RandomUIAgent()

    if agent_type == "human":
        return BlockedHumanAgent()

    if agent_type == "rl":
        from agents.rl_agent import RLAgent
        from agents.rl_nn import PolicyNetwork

        # The UI uses greedy evaluation mode. Stochastic exploration belongs to
        # self-play training in `training/rl/training_loop.py`.
        try:
            network = PolicyNetwork.load(default_rl_weights_path(ruleset))
        except FileNotFoundError:
            # If RL has not been trained yet, warm-start from supervised weights
            # so the menu option remains usable.
            network = PolicyNetwork.load_from_sl(
                default_sl_weights_path(ruleset)
            )

        # Read off the loaded network rather than either weights file, so the
        # supervised warm start above is classified by what it actually built.
        return RLAgent(
            network,
            mode="evaluation",
            ruleset=ruleset,
            **_bucket_features_for(int(network.W1.shape[1]), ruleset),
        )

    raise ValueError(f"Invalid agent type: {agent_type}")
