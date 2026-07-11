"""Agent wrapper used by reinforcement-learning training and evaluation."""

from agents.agent import Agent
from agents.encoder import DominoEncoder
from agents.nn import GPU_ENABLED
from agents.rl_nn import PolicyNetwork
from middleware.opponent_model import ExactOpponentModel

if GPU_ENABLED:
    import cupy as xp
else:
    import numpy as xp


class RLAgent(Agent):
    """Choose tile plays from a policy network and record sampled decisions.

    Draw, pass, and single-option tile plays are forced by the rules engine in
    the current rule set. They are returned directly and are not stored as
    policy-gradient decisions.
    """

    def __init__(self, network, mode="training"):
        self.network = network
        self.mode = mode
        self.encoder = DominoEncoder()
        self.opponent_model = ExactOpponentModel()
        self.trajectory = []

    @classmethod
    def load(cls, weights_path="models/domino_rl_weights.npz", mode="evaluation"):
        network = PolicyNetwork.load(weights_path)
        return cls(network, mode=mode)

    def choose_move(self, state, legal_actions):
        if not legal_actions:
            return None

        policy_actions = [move for move in legal_actions if self.encoder.is_policy_action(move)]
        if not policy_actions:
            return legal_actions[0]

        if len(policy_actions) == 1:
            return policy_actions[0]

        state["opponent_suit_probabilities"] = self.opponent_model.update(state)
        x = self.encoder.encode_state(state)
        if GPU_ENABLED:
            x = xp.array(x)

        probabilities = self.network.forward(x)
        if hasattr(probabilities, "get"):
            probabilities = probabilities.get()

        legal_mask = self.encoder.policy_action_mask(policy_actions)
        if GPU_ENABLED:
            legal_mask = xp.array(legal_mask)

        if self.mode == "training":
            move, action_index = self.encoder.sample_action(probabilities, policy_actions)
            self.trajectory.append([x, action_index, legal_mask, 0.0])
            return move

        return self.encoder.decode_output(probabilities, policy_actions)

    def add_reward_to_last_decision(self, amount):
        """Add intermediate reward to the most recent learner tile-play decision."""
        if self.trajectory:
            self.trajectory[-1][-1] += amount

    def finish_episode(self, final_reward):
        """Attach terminal reward to every sampled tile-play decision."""
        steps = [
            (x, action_index, legal_mask, final_reward + shaped_reward)
            for x, action_index, legal_mask, shaped_reward in self.trajectory
        ]
        self.trajectory = []
        return steps
