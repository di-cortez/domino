"""Agent wrapper used by reinforcement-learning training and evaluation."""

from dataclasses import dataclass

import numpy as np

from agents.encoder import DominoEncoder
from agents.rl_nn import PolicyNetwork
from middleware.middleware import Agent
from middleware.opponent_model import ExactOpponentModel


@dataclass
class TrajectoryStep:
    """One real learner decision sampled from the frozen rollout policy."""

    x: object
    action_index: int
    legal_mask: object
    decision_turn: int
    old_log_prob: float = 0.0
    local_reward: float = 0.0


@dataclass(frozen=True)
class FinishedTrajectoryStep:
    """A sampled decision after terminal reward has been attached."""

    x: object
    action_index: int
    legal_mask: object
    raw_reward: float
    local_reward: float
    terminal_reward: float
    old_log_prob: float = 0.0


class RLAgent(Agent):
    """Choose tile plays from a policy network and record sampled decisions.

    Draw, pass, and single-option tile plays are forced by the rules engine in
    the current rule set. They bypass the network and are not stored as
    policy-gradient decisions. Real decisions store their turn so self-play can
    apply temporally decayed local rewards outside the agent. Training steps
    also retain the masked-policy log-probability from collection time so PPO
    never has to reconstruct ``pi_old`` after the policy has changed.
    """

    VALID_MODES = {"training", "stochastic_evaluation", "evaluation"}

    def __init__(self, network, mode="training"):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown RLAgent mode {mode!r}; expected one of "
                f"{sorted(self.VALID_MODES)}."
            )
        self.network = network
        self.mode = mode
        self.encoder = DominoEncoder()
        self.opponent_model = ExactOpponentModel(record_traces=False)
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
        # Match the network's own resolved backend (agents/rl_nn.py's
        # `device` toggle), not just whether GPU_ENABLED is true globally.
        x = self.network.xp.asarray(x)

        probabilities = self.network.forward(x)
        if hasattr(probabilities, "get"):
            probabilities = probabilities.get()

        if self.mode in {"training", "stochastic_evaluation"}:
            host_legal_mask = np.zeros(
                self.encoder.ACTION_SIZE,
                dtype=np.bool_,
            )
            for action in policy_actions:
                host_legal_mask[self.encoder._action_index(action)] = True
            logits = getattr(self.network, "cache", {}).get("Z3")
            if logits is None:
                legal_probabilities = np.asarray(
                    probabilities[host_legal_mask, 0],
                    dtype=np.float32,
                ).copy()
            else:
                if hasattr(logits, "get"):
                    logits = logits.get()
                legal_logits = np.asarray(
                    logits[host_legal_mask, 0],
                    dtype=np.float32,
                )
                legal_logits = legal_logits - np.max(legal_logits)
                legal_probabilities = np.exp(legal_logits)
            legal_total = float(legal_probabilities.sum())
            if not np.isfinite(legal_total) or legal_total <= 0.0:
                raise FloatingPointError(
                    "Masked RL rollout policy produced invalid probabilities."
                )
            legal_probabilities /= legal_total
            sampling_probabilities = np.zeros_like(probabilities, dtype=np.float32)
            sampling_probabilities[host_legal_mask, 0] = legal_probabilities
            move, action_index = self.encoder.sample_action(
                sampling_probabilities,
                policy_actions,
            )
            if self.mode == "training":
                old_probability = float(sampling_probabilities[action_index, 0])
                old_log_prob = float(
                    np.log(max(old_probability, np.finfo(np.float32).tiny))
                )
                legal_mask = self.network.xp.asarray(
                    host_legal_mask.reshape(-1, 1),
                    dtype=self.network.xp.bool_,
                )
                self.trajectory.append(
                    TrajectoryStep(
                        x=x,
                        action_index=action_index,
                        legal_mask=legal_mask,
                        old_log_prob=old_log_prob,
                        decision_turn=int(state["turn"]),
                    )
                )
            return move

        return self.encoder.decode_output(probabilities, policy_actions)

    def add_decayed_event_reward(self, event_turn, base_reward, decay_lambda):
        """Distribute one local event reward to every earlier real decision."""
        for step in self.trajectory:
            elapsed_actions = int(event_turn) - step.decision_turn - 1
            if elapsed_actions < 0:
                raise ValueError(
                    "Event reward chronology is invalid: "
                    f"event_turn={event_turn}, decision_turn={step.decision_turn}."
                )
            step.local_reward += float(base_reward) * (float(decay_lambda) ** elapsed_actions)

    def finish_episode(self, final_reward):
        """Attach uniform terminal reward to every sampled tile-play decision."""
        steps = [
            FinishedTrajectoryStep(
                x=step.x,
                action_index=step.action_index,
                legal_mask=step.legal_mask,
                old_log_prob=step.old_log_prob,
                raw_reward=float(final_reward) + step.local_reward,
                local_reward=step.local_reward,
                terminal_reward=float(final_reward),
            )
            for step in self.trajectory
        ]
        self.trajectory = []
        return steps
