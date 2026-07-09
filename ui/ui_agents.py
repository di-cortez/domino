"""
UI-facing agent factory.

The visual interface identifies players with small string IDs:

    "neural", "heuristic", "random", "human", "rl"

This module is the only place that translates those IDs into objects accepted
by `GameManager`. Keeping that mapping here prevents the controller and HUD
from importing every concrete agent implementation directly.
"""

import random


AGENT_TYPES = ("neural", "heuristic", "random", "human", "rl")


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


def create_agent_by_type(agent_type):
    """
    Build the agent instance selected in the UI menu.

    Imports stay inside the function so that unused neural/RL dependencies are
    not loaded when the user chooses a simpler agent type.
    """
    if agent_type == "neural":
        from agents.neural_agent import NeuralAgent

        return NeuralAgent.load("models/domino_sl_weights.npz")

    if agent_type == "heuristic":
        from agents.heuristic_agent import StrategicAgent

        return StrategicAgent()

    if agent_type == "random":
        return RandomUIAgent()

    if agent_type == "human":
        return BlockedHumanAgent()

    if agent_type == "rl":
        from agents.rl_agent import RLAgent
        from agents.rl_nn import PolicyNetwork

        # The UI uses greedy evaluation mode. Stochastic exploration belongs to
        # self-play training in `training/self_play.py`.
        try:
            network = PolicyNetwork.load("models/domino_rl_weights.npz")
        except FileNotFoundError:
            # If RL has not been trained yet, warm-start from supervised weights
            # so the menu option remains usable.
            network = PolicyNetwork.load_from_sl("models/domino_sl_weights.npz")

        return RLAgent(network, mode="evaluation")

    raise ValueError(f"Invalid agent type: {agent_type}")
