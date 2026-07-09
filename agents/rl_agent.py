"""Agent wrapper used by reinforcement-learning training and evaluation."""

from agents.agent import Agent
from agents.encoder import DominoEncoder
from agents.nn import GPU_ENABLED
from agents.rl_nn import PolicyNetwork

if GPU_ENABLED:
    import cupy as xp
else:
    import numpy as xp


class RLAgent(Agent):
    """Choose moves from a policy network and record sampled actions in training mode."""

    def __init__(self, network, mode="training"):
        self.network = network
        self.mode = mode
        self.encoder = DominoEncoder()
        self.trajectory = []

    @classmethod
    def load(cls, weights_path="models/domino_rl_weights.npz", mode="evaluation"):
        network = PolicyNetwork.load(weights_path)
        return cls(network, mode=mode)

    def choose_move(self, state, legal_actions):
        if not legal_actions:
            return None

        x = self.encoder.encode_state(state)
        if GPU_ENABLED:
            x = xp.array(x)

        probabilities = self.network.forward(x)
        if hasattr(probabilities, "get"):
            probabilities = probabilities.get()

        if self.mode == "training":
            move, action_index = self.encoder.sample_action(probabilities, legal_actions)
            self.trajectory.append((x, action_index))
            return move

        return self.encoder.decode_output(probabilities, legal_actions)

    def finish_episode(self, final_reward):
        """Attach the terminal reward to every sampled action from this episode."""
        steps = [(x, action_index, final_reward) for x, action_index in self.trajectory]
        self.trajectory = []
        return steps
