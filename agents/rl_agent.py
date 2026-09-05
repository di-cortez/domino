"""Agent wrapper used by reinforcement-learning training and evaluation."""

from dataclasses import dataclass
import time
import warnings

import numpy as np

from agents.encoder import DominoEncoder
from agents.rl_nn import NonFinitePolicyError, PolicyNetwork
from middleware.middleware import Agent
from middleware.opponent_model import ExactOpponentModel
from middleware.rulesets import DEFAULT_RULESET_NAME, resolve_ruleset
from training.rl.reward_model import DRAW_EVENT, EVENT_KINDS
from utils.ruleset_paths import default_rl_weights_path


@dataclass
class TrajectoryStep:
    """One real learner decision sampled from the frozen rollout policy.

    The immediate return is kept as its two separable halves, ``G_D`` and
    ``G_P`` of ``training.rl.reward_model``, because the periodic diagnostic
    reports their distributions independently. ``local_reward`` remains the
    sum, so every consumer of the mixed objective is untouched by the split.
    """

    x: object
    action_index: int
    legal_mask: object
    decision_turn: int
    old_log_prob: float = 0.0
    draw_return: float = 0.0
    pass_return: float = 0.0
    agent_hand_size: int | None = None
    opponent_hand_size: int | None = None

    @property
    def local_reward(self):
        """Return ``G_I`` before the ``reward_eta`` mixture: ``G_D + G_P``."""
        return self.draw_return + self.pass_return


@dataclass(frozen=True)
class FinishedTrajectoryStep:
    """A sampled decision after terminal reward has been attached."""

    x: object
    action_index: int
    legal_mask: object
    decision_turn: int
    raw_reward: float
    terminal_reward: float
    draw_return: float = 0.0
    pass_return: float = 0.0
    old_log_prob: float = 0.0
    agent_hand_size: int | None = None
    opponent_hand_size: int | None = None

    @property
    def local_reward(self):
        return self.draw_return + self.pass_return


UNDERFLOW_FALLBACK_WARNING = (
    "RL rollout policy left no usable probability mass on the legal actions "
    "of a decision: the full-support softmax underflowed on every one of "
    "them. Sampling uniformly over the legal actions for this decision. The "
    "policy is not diverged -- this is a float32 limit of the full-support "
    "path, which a network publishing its logits cache never takes."
)

# Count of decisions that fell back to a uniform draw, per process. It is a
# module-level counter because the rollout workers that reach this path build
# their agents without a runtime profile, so there is no per-agent channel back
# to the parent. The warning is emitted once per process; the counter keeps the
# full tally for tests and for direct inspection.
underflow_fallback_count = 0
_underflow_fallback_warned = False


def _uniform_over_legal_actions(legal_probabilities):
    """Return a uniform draw over one decision's legal actions.

    Reached only when every legal action underflowed to exactly zero, which
    leaves the full-support array carrying no recoverable ordering over the
    legal subset. A uniform draw is the honest reading of that state, and it
    keeps a run alive that would otherwise die on a single decision.
    """
    global underflow_fallback_count, _underflow_fallback_warned
    underflow_fallback_count += 1
    if not _underflow_fallback_warned:
        _underflow_fallback_warned = True
        warnings.warn(UNDERFLOW_FALLBACK_WARNING, RuntimeWarning, stacklevel=3)
    return np.full(
        legal_probabilities.shape,
        1.0 / legal_probabilities.size,
        dtype=np.float32,
    )


def _masked_rollout_probabilities(probabilities, logits, host_legal_mask):
    """Return the sampling distribution over one decision's legal actions.

    Two numerically different paths reach this, and they fail in different
    ways, so neither may borrow the other's diagnosis.

    With the output logits cached, the softmax is rebuilt over the legal subset
    alone. After the max-subtraction the maximizing entry contributes exactly
    ``exp(0) = 1``, so the total is at least 1.0 for every finite input: a
    non-finite total cannot come from the mask or from the renormalization, it
    can only mean the network returned NaN or infinity. The logits are
    therefore tested directly, before the subtraction, while it is still
    visible which entries were bad -- ``inf - inf`` would erase that.

    Without the logits only the full-support probabilities survive, and there
    the legal mass genuinely can underflow to zero with nothing being
    non-finite. Underflow is a float32 limit rather than a diverged policy, so
    it degrades to a uniform draw; only a non-finite total still raises.
    """
    if logits is None:
        legal_probabilities = np.asarray(
            probabilities[host_legal_mask, 0],
            dtype=np.float32,
        ).copy()
        legal_total = float(legal_probabilities.sum())
        if not np.isfinite(legal_total):
            raise NonFinitePolicyError(
                "RL rollout policy produced a non-finite total over the "
                f"{int(host_legal_mask.sum())} legal actions of this decision "
                f"(total {legal_total!r}). The policy weights have diverged; "
                "the legal mask and the masked softmax are not at fault."
            )
        if legal_total <= 0.0:
            return _uniform_over_legal_actions(legal_probabilities)
    else:
        if hasattr(logits, "get"):
            logits = logits.get()
        legal_logits = np.asarray(
            logits[host_legal_mask, 0],
            dtype=np.float32,
        )
        non_finite = int(np.count_nonzero(~np.isfinite(legal_logits)))
        if non_finite:
            raise NonFinitePolicyError(
                f"RL rollout policy produced non-finite logits: {non_finite} of "
                f"{legal_logits.size} legal logits are NaN or infinite. The "
                "policy weights have diverged; the legal mask and the masked "
                "softmax are not at fault."
            )
        legal_logits = legal_logits - np.max(legal_logits)
        legal_probabilities = np.exp(legal_logits)
        legal_total = float(legal_probabilities.sum())
    legal_probabilities /= legal_total
    return legal_probabilities


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

    @staticmethod
    def _decision_hand_sizes(state):
        """Return learner/opponent hand sizes for one two-player decision."""
        hand_sizes = state.get("hand_sizes")
        current_player = int(state.get("current_player", -1))
        if (
            not isinstance(hand_sizes, (list, tuple))
            or len(hand_sizes) != 2
            or current_player not in (0, 1)
        ):
            raise ValueError(
                "RL training decisions require two ordered hand sizes."
            )
        return int(hand_sizes[current_player]), int(hand_sizes[1 - current_player])

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
            legal_probabilities = _masked_rollout_probabilities(
                probabilities,
                logits,
                host_legal_mask,
            )
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
                agent_hand_size, opponent_hand_size = self._decision_hand_sizes(
                    state
                )
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
                        agent_hand_size=agent_hand_size,
                        opponent_hand_size=opponent_hand_size,
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
                legal_probabilities = _masked_rollout_probabilities(
                    probabilities,
                    logits,
                    host_legal_mask,
                )
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
                    agent_hand_size, opponent_hand_size = (
                        self._decision_hand_sizes(state)
                    )
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
                            agent_hand_size=agent_hand_size,
                            opponent_hand_size=opponent_hand_size,
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
        *,
        event_kind,
    ):
        """Distribute one event reward using turn or real-decision distance.

        ``event_kind`` selects which half of the immediate return the decayed
        value lands in, so ``G_D`` and ``G_P`` stay separable all the way to
        the diagnostic record. It is keyword-only and never inferred: the
        caller detected the event, so only the caller knows its kind.
        """
        if event_kind not in EVENT_KINDS:
            raise ValueError(
                f"Unknown event kind {event_kind!r}; expected one of "
                f"{', '.join(EVENT_KINDS)}."
            )
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
            decayed = float(base_reward) * (float(decay_lambda) ** distance)
            if event_kind == DRAW_EVENT:
                step.draw_return += decayed
            else:
                step.pass_return += decayed

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
                draw_return=step.draw_return,
                pass_return=step.pass_return,
                terminal_reward=float(final_reward),
                agent_hand_size=step.agent_hand_size,
                opponent_hand_size=step.opponent_hand_size,
            )
            for step in self.trajectory
        ]
        self.trajectory = []
        return steps
