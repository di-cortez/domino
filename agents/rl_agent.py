"""Agent wrapper used by reinforcement-learning training and evaluation."""

from dataclasses import dataclass

from agents.agent import Agent
from agents.encoder import DominoEncoder
from agents.rl_nn import PolicyNetwork
from middleware.opponent_model import ExactOpponentModel


@dataclass
class TrajectoryStep:
    """One real learner decision saved for REINFORCE training."""

    x: object
    action_index: int
    legal_mask: object
    decision_turn: int
    option_count: int
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
    option_count: int


class RLAgent(Agent):
    """Choose tile plays from a policy network and record sampled decisions.

    Draw, pass, and single-option tile plays are forced by the rules engine in
    the current rule set. They bypass the network and are not stored as
    policy-gradient decisions. Real decisions store their turn and option count
    so self-play can apply temporally decayed local rewards and option-count
    multipliers outside the agent.
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
            move, action_index = self.encoder.sample_action(probabilities, policy_actions)
            if self.mode == "training":
                legal_mask = self.encoder.policy_action_mask(policy_actions)
                legal_mask = self.network.xp.asarray(legal_mask)
                self.trajectory.append(
                    TrajectoryStep(
                        x=x,
                        action_index=action_index,
                        legal_mask=legal_mask,
                        decision_turn=int(state["turn"]),
                        option_count=len(policy_actions),
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
                raw_reward=float(final_reward) + step.local_reward,
                local_reward=step.local_reward,
                terminal_reward=float(final_reward),
                option_count=step.option_count,
            )
            for step in self.trajectory
        ]
        self.trajectory = []
        return steps
