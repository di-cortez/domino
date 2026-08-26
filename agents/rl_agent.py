"""Agent wrapper used by reinforcement-learning training and evaluation."""

from dataclasses import dataclass
import time

import numpy as np

from agents.encoder import DominoEncoder
from agents.rl_nn import PolicyNetwork
from middleware.middleware import Agent
from middleware.opponent_model import ExactOpponentModel
from middleware.rulesets import DEFAULT_RULESET_NAME, resolve_ruleset
from utils.ruleset_paths import default_rl_weights_path


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
    decision_turn: int
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

    ``use_opponent_suit_features=False`` selects the ablated encoder layout and
    skips exact opponent inference entirely, so the policy it wraps must have
    been trained with the shortened input. The shape check below enforces that.

    ``use_opponent_bucket_features=True`` appends the opponent-bucket one-hot,
    and ``opponent_bucket`` names the bucket *this* agent is playing against.
    Each seat is told about its own adversary, so in a game between the learner
    and a frozen snapshot the two agents receive different buckets. ``None``
    leaves the block at zero for an adversary that belongs to no bucket.
    """

    VALID_MODES = {"training", "stochastic_evaluation", "evaluation"}

    def __init__(
        self,
        network,
        mode="training",
        runtime_profile=None,
        *,
        ruleset=DEFAULT_RULESET_NAME,
        use_opponent_suit_features=True,
        use_opponent_bucket_features=False,
        opponent_bucket=None,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown RLAgent mode {mode!r}; expected one of "
                f"{sorted(self.VALID_MODES)}."
            )
        self.ruleset = resolve_ruleset(ruleset)
        self.network = network
        self.mode = mode
        self.use_opponent_suit_features = bool(use_opponent_suit_features)
        self.use_opponent_bucket_features = bool(use_opponent_bucket_features)
        self.opponent_bucket = opponent_bucket
        self.encoder = DominoEncoder(
            self.ruleset,
            use_opponent_suit_features=self.use_opponent_suit_features,
            use_opponent_bucket_features=self.use_opponent_bucket_features,
            opponent_bucket=opponent_bucket,
        )
        first_weight = getattr(network, "W1", None)
        layer_count = getattr(network, "layer_count", None)
        output_weight = (
            None
            if layer_count is None
            else getattr(network, f"W{layer_count}", None)
        )
        if first_weight is not None and output_weight is not None:
            actual_input = int(first_weight.shape[1])
            actual_output = int(output_weight.shape[0])
            if (
                actual_input != self.encoder.vector_size
                or actual_output != self.encoder.action_size
            ):
                raise ValueError(
                    f"RL policy shape input={actual_input}, output={actual_output} "
                    f"does not match ruleset {self.ruleset.name!r}: "
                    f"input={self.encoder.vector_size}, "
                    f"output={self.encoder.action_size}."
                )
        # The model exists only to fill the encoder's trailing block. With that
        # block ablated nothing would read its output, so it is never built.
        self.opponent_model = (
            ExactOpponentModel(ruleset=self.ruleset, record_traces=False)
            if self.use_opponent_suit_features
            else None
        )
        self.trajectory = []
        # Optional low-overhead accumulator used by rollout/diagnostic workers.
        # Keeping it injectable avoids changing the normal public-agent output.
        self.runtime_profile = runtime_profile

    def _profile_add(self, section, started):
        if self.runtime_profile is None or started is None:
            return
        sections = self.runtime_profile.setdefault("sections_seconds", {})
        sections[section] = sections.get(section, 0.0) + (
            time.perf_counter() - started
        )

    @classmethod
    def load(
        cls,
        weights_path=None,
        mode="evaluation",
        use_value_head=False,
        ruleset=DEFAULT_RULESET_NAME,
        use_opponent_suit_features=True,
        use_opponent_bucket_features=False,
        opponent_bucket=None,
    ):
        """Load an RL policy and optionally restore its persisted value head."""
        weights_path = weights_path or default_rl_weights_path(ruleset)
        network = PolicyNetwork.load(
            weights_path,
            use_value_head=use_value_head,
        )
        return cls(
            network,
            mode=mode,
            ruleset=ruleset,
            use_opponent_suit_features=use_opponent_suit_features,
            use_opponent_bucket_features=use_opponent_bucket_features,
            opponent_bucket=opponent_bucket,
        )

    def choose_move(self, state, legal_actions):
        if self.runtime_profile is None:
            return self._choose_move_unprofiled(state, legal_actions)
        return self._choose_move_profiled(state, legal_actions)

    def _choose_move_unprofiled(self, state, legal_actions):
        """Original hot path, kept free of profiler branches and callbacks."""
        if not legal_actions:
            return None

        policy_actions = [
            move for move in legal_actions if self.encoder.is_policy_action(move)
        ]
        if not policy_actions:
            return legal_actions[0]
        if len(policy_actions) == 1:
            return policy_actions[0]

        if self.opponent_model is not None:
            state["opponent_suit_probabilities"] = self.opponent_model.update(state)
        x = self.encoder.encode_state(state)
        x = self.network.xp.asarray(x)
        probabilities = self.network.forward(x)
        if hasattr(probabilities, "get"):
            probabilities = probabilities.get()

        if self.mode in {"training", "stochastic_evaluation"}:
            host_legal_mask = np.zeros(
                self.encoder.action_size,
                dtype=np.bool_,
            )
            for action in policy_actions:
                host_legal_mask[self.encoder._action_index(action)] = True
            logits = getattr(self.network, "cache", {}).get(
                getattr(self.network, "logits_key", "Z3")
            )
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
            sampling_probabilities = np.zeros_like(
                probabilities,
                dtype=np.float32,
            )
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

    def _choose_move_profiled(self, state, legal_actions):
        profiling = self.runtime_profile is not None
        profile_started = time.perf_counter() if profiling else None
        if profiling:
            self.runtime_profile["calls"] = self.runtime_profile.get("calls", 0) + 1
        try:
            section_started = time.perf_counter() if profiling else None
            if not legal_actions:
                self._profile_add(
                    "action_filtering_and_forced_choice",
                    section_started,
                )
                return None

            policy_actions = [
                move for move in legal_actions if self.encoder.is_policy_action(move)
            ]
            if not policy_actions:
                self._profile_add(
                    "action_filtering_and_forced_choice",
                    section_started,
                )
                return legal_actions[0]

            if len(policy_actions) == 1:
                self._profile_add(
                    "action_filtering_and_forced_choice",
                    section_started,
                )
                return policy_actions[0]
            self._profile_add(
                "action_filtering_and_forced_choice",
                section_started,
            )

            section_started = time.perf_counter() if profiling else None
            if self.opponent_model is not None:
                state["opponent_suit_probabilities"] = (
                    self.opponent_model.update(state)
                )
            self._profile_add(
                "exact_opponent_model_update",
                section_started,
            )

            section_started = time.perf_counter() if profiling else None
            x = self.encoder.encode_state(state)
            # Match the network's own resolved backend (agents/rl_nn.py's
            # `device` toggle), not just whether GPU_ENABLED is true globally.
            x = self.network.xp.asarray(x)
            self._profile_add(
                "state_encoding_and_backend_transfer",
                section_started,
            )

            section_started = time.perf_counter() if profiling else None
            probabilities = self.network.forward(x)
            if hasattr(probabilities, "get"):
                probabilities = probabilities.get()
            self._profile_add(
                "network_forward_and_host_transfer",
                section_started,
            )

            section_started = time.perf_counter() if profiling else None
            if self.mode in {"training", "stochastic_evaluation"}:
                host_legal_mask = np.zeros(
                    self.encoder.action_size,
                    dtype=np.bool_,
                )
                for action in policy_actions:
                    host_legal_mask[self.encoder._action_index(action)] = True
                logits = getattr(self.network, "cache", {}).get(
                    getattr(self.network, "logits_key", "Z3")
                )
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
                sampling_probabilities = np.zeros_like(
                    probabilities,
                    dtype=np.float32,
                )
                sampling_probabilities[host_legal_mask, 0] = legal_probabilities
                move, action_index = self.encoder.sample_action(
                    sampling_probabilities,
                    policy_actions,
                )
                self._profile_add(
                    "legal_mask_and_action_selection",
                    section_started,
                )
                if self.mode == "training":
                    section_started = time.perf_counter() if profiling else None
                    old_probability = float(
                        sampling_probabilities[action_index, 0]
                    )
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
                    self._profile_add(
                        "trajectory_recording",
                        section_started,
                    )
                return move

            move = self.encoder.decode_output(probabilities, policy_actions)
            self._profile_add(
                "legal_mask_and_action_selection",
                section_started,
            )
            return move
        finally:
            if profiling:
                self.runtime_profile["total_seconds"] = (
                    self.runtime_profile.get("total_seconds", 0.0)
                    + time.perf_counter() - profile_started
                )

    def add_decayed_event_reward(
        self,
        event_turn,
        base_reward,
        decay_lambda,
        distance_metric="turn",
    ):
        """Distribute one event reward using turn or real-decision distance."""
        decision_count = len(self.trajectory)
        for index, step in enumerate(self.trajectory):
            if distance_metric == "turn":
                distance = int(event_turn) - step.decision_turn - 1
            elif distance_metric == "decision":
                distance = decision_count - 1 - index
            else:
                raise ValueError(
                    "Local reward distance_metric must be 'turn' or "
                    f"'decision', got {distance_metric!r}."
                )
            if distance < 0:
                raise ValueError(
                    "Event reward chronology is invalid: "
                    f"event_turn={event_turn}, decision_turn={step.decision_turn}."
                )
            step.local_reward += float(base_reward) * (
                float(decay_lambda) ** distance
            )

    def finish_episode(self, final_reward):
        """Attach uniform terminal reward to every sampled tile-play decision."""
        steps = [
            FinishedTrajectoryStep(
                x=step.x,
                action_index=step.action_index,
                legal_mask=step.legal_mask,
                decision_turn=step.decision_turn,
                old_log_prob=step.old_log_prob,
                raw_reward=float(final_reward) + step.local_reward,
                local_reward=step.local_reward,
                terminal_reward=float(final_reward),
            )
            for step in self.trajectory
        ]
        self.trajectory = []
        return steps
