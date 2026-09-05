"""Reward finalization and CPU-only game collection for RL rollouts."""

from dataclasses import dataclass, field
import random
import time

from agents.agent import RandomAgent
from agents.heuristic_agent import StrategicAgent
from agents.rl_agent import RLAgent
from middleware.domino_engine import DominoEngine
from middleware.rulesets import DEFAULT_RULESET_NAME
from training.rl.reward_distance import (
    DEFAULT_REWARD_DISTANCE_MODE,
    resolve_reward_distance_mode,
)
from training.rl.reward_model import (
    BLOCKED_WIN_REASONS,
    DEFAULT_GAMMA_F,
    DEFAULT_GAMMA_I,
    DEFAULT_REWARD_ETA,
    DRAW_EVENT,
    PASS_EVENT,
    combine_terminal_components,
    resolved_reward_scales,
    scaled_event_reward,
    terminal_reward_components,
)
from training.rl.restarts import CapturedOpponentDecision
from training.rl.statistics import RunningMoments

REWARD_ZERO_EPSILON = 1e-8

# The bucket a frozen opponent is told it faces. Every opponent in the pool
# plays the current learner, and the learner is by definition the newest policy
# of the ``recent`` band, so this is what "who am I playing against" resolves to
# from the opposite seat. It is deliberately not the learner's own bucket: that
# one names the learner's adversary, not the match.
LEARNER_OPPONENT_BUCKET = "recent"

# The resolved reward definition one run ships unchanged to every rollout
# worker. It carries the four raw weights for provenance and the normalized
# scales the workers actually multiply by, so ``max(a, b)`` is divided out once
# per run rather than once per event. ``training.rl.config`` copies this mapping
# and overrides it with the values resolved from the CLI.
DEFAULT_REWARD_SCHEMA = {
    **resolved_reward_scales(),
    "gamma_f": DEFAULT_GAMMA_F,
    "gamma_i": DEFAULT_GAMMA_I,
    "reward_eta": DEFAULT_REWARD_ETA,
    "reward_distance_mode": DEFAULT_REWARD_DISTANCE_MODE,
}


@dataclass
class EventStats:
    """Raw draw/pass event counts collected during one training game or batch."""

    opponent_draws: int = 0
    opponent_passes: int = 0
    learner_draws: int = 0
    learner_passes: int = 0

    def add(self, other):
        self.opponent_draws += other.opponent_draws
        self.opponent_passes += other.opponent_passes
        self.learner_draws += other.learner_draws
        self.learner_passes += other.learner_passes


@dataclass
class TerminalStats:
    """Per-batch terminal-outcome counts and blocked pip-margin totals.

    The aggregate the reward redesign needs and the return statistics cannot
    supply: ``reward_mean`` cannot say whether the agent is trading ordinary
    wins for a preferred *kind* of win, because both endings land in the same
    scalar. These are sums rather than distributions so one iteration stays one
    flat metrics row; ``blocked_margin_sum`` over the blocked game count is the
    mean margin, and ``blocked_magnitude_sum`` likewise gives mean
    ``m(Delta_p)`` -- the effective blocked/empty-hand ratio before any weight
    is applied.

    ``empty_hand`` and ``blocked`` carry the full distribution of ``R_E`` and
    ``R_B``, which the counts alone cannot supply: a spread cannot be recovered
    from a total. Both hold the **raw** component that
    ``terminal_reward_components`` returns, before ``empty_hand_scale`` and
    ``blocked_scale`` are applied, so the statistic means the same thing across
    runs that weight the pair differently. Exactly one of the two records a
    value per game -- an ending is either an empty hand or a block -- so each
    population counts the games of its own kind, not every game in the batch.
    """

    empty_hand_wins: int = 0
    empty_hand_losses: int = 0
    blocked_wins: int = 0
    blocked_losses: int = 0
    blocked_margin_sum: int = 0
    blocked_magnitude_sum: float = 0.0
    empty_hand: RunningMoments = field(default_factory=RunningMoments)
    blocked: RunningMoments = field(default_factory=RunningMoments)

    def add(self, other):
        self.empty_hand_wins += other.empty_hand_wins
        self.empty_hand_losses += other.empty_hand_losses
        self.blocked_wins += other.blocked_wins
        self.blocked_losses += other.blocked_losses
        self.blocked_margin_sum += other.blocked_margin_sum
        self.blocked_magnitude_sum += other.blocked_magnitude_sum
        self.empty_hand.merge(other.empty_hand)
        self.blocked.merge(other.blocked)


@dataclass(frozen=True)
class TrainingSample:
    """One finalized real decision used by REINFORCE or PPO."""

    x: object
    action_index: int
    legal_mask: object
    policy_reward: float
    raw_reward: float
    local_reward: float
    terminal_reward: float
    old_log_prob: float = 0.0
    # ``G_D`` and ``G_P`` as the reward model defines them: decayed and scaled,
    # but *before* the ``reward_eta`` mixture that ``local_reward`` carries.
    # The diagnostic reports their distributions; training never reads them.
    draw_return: float = 0.0
    pass_return: float = 0.0
    agent_hand_size: int | None = None
    opponent_hand_size: int | None = None


def _tile_play_actions(legal_actions):
    """Return legal tile-play actions, excluding forced draw/pass."""
    return [
        action
        for action in legal_actions
        if action is not None and action != ("DRAW", None)
    ]


def _event_reward_for_action(
    current_player, learner_position, action, event_stats, schema=None
):
    """Return ``(reward, event_kind)`` for a draw/pass, or ``None``.

    Detection and ``EventStats`` accounting live here; the value itself is the
    unit ``r_D``/``r_P`` of ``training.rl.reward_model`` multiplied by the
    pair-normalized scale the run resolved. The event is not decayed yet:
    ``gamma_i`` is applied by the caller once the event's distance to each
    earlier decision is known.

    The kind travels with the reward because the caller has to route it to the
    right half of the immediate return, and this is the only place that knows
    which half it is.
    """
    if schema is None:
        schema = DEFAULT_REWARD_SCHEMA
    by_learner = current_player == learner_position
    if action == ("DRAW", None):
        event_kind = DRAW_EVENT
    elif action is None:
        event_kind = PASS_EVENT
    else:
        return None
    if by_learner:
        if event_kind == DRAW_EVENT:
            event_stats.learner_draws += 1
        else:
            event_stats.learner_passes += 1
    else:
        if event_kind == DRAW_EVENT:
            event_stats.opponent_draws += 1
        else:
            event_stats.opponent_passes += 1
    return scaled_event_reward(
        event_kind,
        by_learner=by_learner,
        draw_scale=schema["draw_scale"],
        pass_scale=schema["pass_scale"],
    ), event_kind


def _terminal_outcome(engine, learner_position, schema):
    """Return ``(U_T, TerminalStats)`` for one finished game.

    An empty-hand ending contributes the binary ``R_E``; a blocked ending
    contributes ``R_B = +/-m(Delta_p)``, whose magnitude is the pip margin
    between the blocked winner and loser saturated at ``2 * max_pip``. The two
    are mutually exclusive and are combined with the run's normalized terminal
    scales. Temporal discounting and the ``reward_eta`` mixture are applied
    later, by ``_finish_episode_with_rewards``.

    The decomposition is already computed here, so the diagnostic counts come
    with it rather than costing a second pass over the terminal state.
    """
    components = terminal_reward_components(engine, learner_position)
    utility = combine_terminal_components(
        components,
        empty_hand_scale=schema["empty_hand_scale"],
        blocked_scale=schema["blocked_scale"],
    )
    stats = TerminalStats()
    won = engine.winner == learner_position
    if components["win_reason"] in BLOCKED_WIN_REASONS:
        if won:
            stats.blocked_wins += 1
        else:
            stats.blocked_losses += 1
        # Margin and magnitude describe the game rather than the seat, so they
        # are accumulated once whichever seat the learner held.
        stats.blocked_margin_sum += int(components["pip_margin"])
        stats.blocked_magnitude_sum += float(components["blocked_magnitude"])
        stats.blocked.add(components["blocked_component"])
    else:
        if won:
            stats.empty_hand_wins += 1
        else:
            stats.empty_hand_losses += 1
        stats.empty_hand.add(components["empty_hand_component"])
    return utility, stats


def _finish_episode_with_rewards(
    learner_agent,
    terminal_utility,
    gamma_f=DEFAULT_GAMMA_F,
    reward_eta=DEFAULT_REWARD_ETA,
    *,
    terminal_turn,
    reward_distance_mode,
):
    """Finalize one learner trajectory into policy-gradient training samples.

    Each decision's total return is the convex combination

        G(t) = (1 - reward_eta) * gamma_f ** k_T(t) * U_T + reward_eta * G_I(t)

    where ``U_T`` is the episode's undiscounted terminal utility, ``k_T(t)`` is
    measured in remaining real decisions or intervening engine turns according
    to ``reward_distance_mode``, and ``G_I(t)`` is the weighted draw/pass
    return already accumulated on the step by ``gamma_i`` decay.
    ``reward_eta`` trades the two subsystems off only after each one is
    complete: ``reward_eta=0`` trains on the terminal outcome alone and
    ``reward_eta=1`` on event shaping alone.

    ``U_T`` is not necessarily binary. An empty-hand ending gives it the sign
    of the outcome scaled by ``empty_hand_scale``; a blocked ending gives it
    ``+/-blocked_scale * m(Delta_p)``.

    The stored ``terminal_reward`` and ``local_reward`` components are the
    mixed halves ``(1 - reward_eta) * G_T`` and ``reward_eta * G_I``, so
    ``policy_reward == terminal_reward + local_reward`` remains an identity for
    the PPO buffer and the reward diagnostics.
    """
    finished_steps = learner_agent.finish_episode(terminal_utility)
    step_count = len(finished_steps)
    _local_metric, terminal_metric = resolve_reward_distance_mode(
        reward_distance_mode
    )
    samples = []
    for index, step in enumerate(finished_steps):
        if terminal_metric == "decision":
            remaining_after = step_count - 1 - index
        else:
            remaining_after = int(terminal_turn) - step.decision_turn - 1
            if remaining_after < 0:
                raise ValueError(
                    "Terminal reward chronology is invalid: "
                    f"terminal_turn={terminal_turn}, "
                    f"decision_turn={step.decision_turn}."
                )
        discounted_terminal = (1.0 - reward_eta) * step.terminal_reward * (
            gamma_f ** remaining_after
        )
        scaled_local = reward_eta * step.local_reward
        raw_reward = discounted_terminal + scaled_local
        samples.append(
            TrainingSample(
                x=step.x,
                action_index=step.action_index,
                legal_mask=step.legal_mask,
                old_log_prob=step.old_log_prob,
                policy_reward=raw_reward,
                raw_reward=raw_reward,
                local_reward=scaled_local,
                terminal_reward=discounted_terminal,
                draw_return=step.draw_return,
                pass_return=step.pass_return,
                agent_hand_size=step.agent_hand_size,
                opponent_hand_size=step.opponent_hand_size,
            )
        )
    return samples


def _profile_worker_section(runtime_profile, section, started):
    """Accumulate one mutually exclusive worker-game phase."""
    if runtime_profile is None or started is None:
        return
    sections = runtime_profile.setdefault("sections_seconds", {})
    sections[section] = sections.get(section, 0.0) + (
        time.perf_counter() - started
    )


def _profile_worker_start(runtime_profile):
    return time.perf_counter() if runtime_profile is not None else None


def _capture_opponent_decision(
    captures,
    engine,
    learner_position,
    current_player,
    tile_actions,
):
    """Capture one exact pre-action opponent choice without consuming RNG."""
    if (
        captures is None
        or current_player == learner_position
        or len(tile_actions) < 2
    ):
        return
    captures.append(CapturedOpponentDecision(
        snapshot_ordinal=len(captures),
        source_turn=int(engine.turn),
        original_learner_position=int(learner_position),
        source_legal_tile_action_count=len(tile_actions),
        engine_state=engine.export_restart_state(),
    ))


def _play_training_game_unprofiled(
    agents,
    learner_position,
    learner_agent,
    schema,
    ruleset_name=DEFAULT_RULESET_NAME,
    *,
    engine=None,
    capture_opponent_decision_restarts=False,
):
    """Profiler-free rollout hot path for non-sampled games."""
    if engine is None:
        engine = DominoEngine(player_count=len(agents), ruleset=ruleset_name)
    event_stats = EventStats()
    captures = [] if capture_opponent_decision_restarts else None
    local_distance_metric, _terminal_distance_metric = (
        resolve_reward_distance_mode(schema["reward_distance_mode"])
    )
    while not engine.game_over:
        state = engine._get_state()
        current_player = state["current_player"]
        legal_actions = engine.valid_actions(current_player)
        tile_actions = _tile_play_actions(legal_actions)
        _capture_opponent_decision(
            captures,
            engine,
            learner_position,
            current_player,
            tile_actions,
        )
        if current_player == learner_position and len(tile_actions) == 1:
            action = tile_actions[0]
        else:
            action = agents[current_player].choose_move(state, legal_actions)
        event = _event_reward_for_action(
            current_player,
            learner_position,
            action,
            event_stats,
            schema,
        )
        if event is not None:
            event_reward, event_kind = event
            learner_agent.add_decayed_event_reward(
                event_turn=state["turn"],
                base_reward=event_reward,
                decay_lambda=schema["gamma_i"],
                distance_metric=local_distance_metric,
                event_kind=event_kind,
            )
        engine.step(
            action,
            return_state=False,
            legal_actions=legal_actions,
        )
    return engine, event_stats, captures


def _play_training_game(
    agents,
    learner_position,
    learner_agent,
    schema,
    runtime_profile=None,
    ruleset_name=DEFAULT_RULESET_NAME,
    *,
    engine=None,
    capture_opponent_decision_restarts=False,
):
    """Play one RL training game and attach decayed local event rewards."""
    if runtime_profile is None:
        return _play_training_game_unprofiled(
            agents,
            learner_position,
            learner_agent,
            schema,
            ruleset_name,
            engine=engine,
            capture_opponent_decision_restarts=(
                capture_opponent_decision_restarts
            ),
        )
    section_started = _profile_worker_start(runtime_profile)
    if engine is None:
        engine = DominoEngine(player_count=len(agents), ruleset=ruleset_name)
    event_stats = EventStats()
    captures = [] if capture_opponent_decision_restarts else None
    local_distance_metric, _terminal_distance_metric = (
        resolve_reward_distance_mode(schema["reward_distance_mode"])
    )
    _profile_worker_section(
        runtime_profile,
        "engine_initialization",
        section_started,
    )

    while not engine.game_over:
        section_started = _profile_worker_start(runtime_profile)
        state = engine._get_state()
        current_player = state["current_player"]
        legal_actions = engine.valid_actions(current_player)
        tile_actions = _tile_play_actions(legal_actions)
        _capture_opponent_decision(
            captures,
            engine,
            learner_position,
            current_player,
            tile_actions,
        )
        _profile_worker_section(
            runtime_profile,
            "state_and_legal_action_generation",
            section_started,
        )

        if current_player == learner_position and len(tile_actions) == 1:
            section_started = _profile_worker_start(runtime_profile)
            action = tile_actions[0]
            _profile_worker_section(
                runtime_profile,
                "forced_learner_action_selection",
                section_started,
            )
        else:
            section_started = _profile_worker_start(runtime_profile)
            action = agents[current_player].choose_move(state, legal_actions)
            _profile_worker_section(
                runtime_profile,
                (
                    "learner_agent_decisions"
                    if current_player == learner_position
                    else "opponent_agent_decisions"
                ),
                section_started,
            )

        section_started = _profile_worker_start(runtime_profile)
        event = _event_reward_for_action(
            current_player,
            learner_position,
            action,
            event_stats,
            schema,
        )
        if event is not None:
            event_reward, event_kind = event
            learner_agent.add_decayed_event_reward(
                event_turn=state["turn"],
                base_reward=event_reward,
                decay_lambda=schema["gamma_i"],
                distance_metric=local_distance_metric,
                event_kind=event_kind,
            )
        _profile_worker_section(
            runtime_profile,
            "reward_shaping",
            section_started,
        )

        section_started = _profile_worker_start(runtime_profile)
        engine.step(
            action,
            return_state=False,
            legal_actions=legal_actions,
        )
        _profile_worker_section(
            runtime_profile,
            "engine_state_transition",
            section_started,
        )

    return engine, event_stats, captures


def _collect_steps_vs_snapshot(
    network,
    opponent_network,
    schema,
    gamma_f,
    runtime_profile=None,
    ruleset_name=DEFAULT_RULESET_NAME,
    capture_opponent_decision_restarts=False,
    use_opponent_suit_features=True,
    use_opponent_bucket_features=False,
    opponent_bucket=None,
):
    """Play against one already-selected frozen neural opponent."""
    section_started = _profile_worker_start(runtime_profile)
    learner_position = random.randint(0, 1)

    learner_policy_profile = (
        runtime_profile.setdefault("learner_policy", {})
        if runtime_profile is not None
        else None
    )
    opponent_policy_profile = (
        runtime_profile.setdefault("opponent_policy", {})
        if runtime_profile is not None
        else None
    )
    learner = RLAgent(
        network,
        mode="training",
        runtime_profile=learner_policy_profile,
        ruleset=ruleset_name,
        use_opponent_suit_features=use_opponent_suit_features,
        use_opponent_bucket_features=use_opponent_bucket_features,
        opponent_bucket=opponent_bucket,
    )
    opponent = RLAgent(
        opponent_network,
        mode="stochastic_evaluation",
        runtime_profile=opponent_policy_profile,
        ruleset=ruleset_name,
        use_opponent_suit_features=use_opponent_suit_features,
        use_opponent_bucket_features=use_opponent_bucket_features,
        opponent_bucket=LEARNER_OPPONENT_BUCKET,
    )
    agents = [None, None]
    agents[learner_position] = learner
    agents[1 - learner_position] = opponent
    _profile_worker_section(runtime_profile, "agent_setup", section_started)

    engine, event_stats, captures = _play_training_game(
        agents,
        learner_position,
        learner,
        schema,
        runtime_profile=runtime_profile,
        ruleset_name=ruleset_name,
        capture_opponent_decision_restarts=capture_opponent_decision_restarts,
    )
    section_started = _profile_worker_start(runtime_profile)
    terminal_utility, terminal_stats = _terminal_outcome(
        engine, learner_position, schema
    )
    samples = _finish_episode_with_rewards(
        learner,
        terminal_utility,
        gamma_f,
        schema["reward_eta"],
        terminal_turn=engine.turn,
        reward_distance_mode=schema["reward_distance_mode"],
    )
    _profile_worker_section(
        runtime_profile,
        "terminal_reward_and_trajectory_finalization",
        section_started,
    )
    result = (
        samples, event_stats, engine.winner, learner_position, terminal_stats
    )
    if capture_opponent_decision_restarts:
        return (*result, tuple(captures))
    return result


def collect_steps_for_assignment(
    learner_network,
    opponent_kind,
    opponent_network,
    schema,
    gamma_f,
    runtime_profile=None,
    ruleset_name=DEFAULT_RULESET_NAME,
    capture_opponent_decision_restarts=False,
    use_opponent_suit_features=True,
    use_opponent_bucket_features=False,
    opponent_bucket=None,
):
    """Dispatch one preselected assignment without knowing pool policy.

    ``opponent_bucket`` is the assignment's bucket name, forwarded verbatim so
    the learner is told which bucket its adversary was drawn from. It stays a
    caller-supplied value rather than being derived from ``opponent_kind``
    because several buckets share one kind: every neural bucket dispatches
    through ``policy_snapshot``, and the champion buckets deliberately overlap
    the chronological bands.
    """
    if opponent_kind == "policy_snapshot":
        if opponent_network is None:
            raise ValueError("A policy-snapshot assignment requires weights")
        return _collect_steps_vs_snapshot(
            learner_network,
            opponent_network,
            schema,
            gamma_f,
            runtime_profile=runtime_profile,
            ruleset_name=ruleset_name,
            capture_opponent_decision_restarts=(
                capture_opponent_decision_restarts
            ),
            use_opponent_suit_features=use_opponent_suit_features,
            use_opponent_bucket_features=use_opponent_bucket_features,
            opponent_bucket=opponent_bucket,
        )
    if opponent_kind == "heuristic":
        if opponent_network is not None:
            raise ValueError("A heuristic assignment must not carry neural weights")
        return _collect_steps_vs_heuristic(
            learner_network,
            schema,
            gamma_f,
            runtime_profile=runtime_profile,
            ruleset_name=ruleset_name,
            capture_opponent_decision_restarts=(
                capture_opponent_decision_restarts
            ),
            use_opponent_suit_features=use_opponent_suit_features,
            use_opponent_bucket_features=use_opponent_bucket_features,
            opponent_bucket=opponent_bucket,
        )
    if opponent_kind == "random":
        if opponent_network is not None:
            raise ValueError("A random assignment must not carry neural weights")
        return _collect_steps_vs_random(
            learner_network,
            schema,
            gamma_f,
            runtime_profile=runtime_profile,
            ruleset_name=ruleset_name,
            capture_opponent_decision_restarts=(
                capture_opponent_decision_restarts
            ),
            use_opponent_suit_features=use_opponent_suit_features,
            use_opponent_bucket_features=use_opponent_bucket_features,
            opponent_bucket=opponent_bucket,
        )
    raise ValueError(f"Unknown RL opponent kind: {opponent_kind!r}")


def _collect_steps_vs_heuristic(
    network,
    schema,
    gamma_f,
    runtime_profile=None,
    ruleset_name=DEFAULT_RULESET_NAME,
    capture_opponent_decision_restarts=False,
    use_opponent_suit_features=True,
    use_opponent_bucket_features=False,
    opponent_bucket=None,
):
    """Play one training game against the fixed heuristic agent."""
    section_started = _profile_worker_start(runtime_profile)
    learner_position = random.randint(0, 1)
    learner_policy_profile = (
        runtime_profile.setdefault("learner_policy", {})
        if runtime_profile is not None
        else None
    )
    learner = RLAgent(
        network,
        mode="training",
        runtime_profile=learner_policy_profile,
        ruleset=ruleset_name,
        use_opponent_suit_features=use_opponent_suit_features,
        use_opponent_bucket_features=use_opponent_bucket_features,
        opponent_bucket=opponent_bucket,
    )
    agents = [None, None]
    agents[learner_position] = learner
    agents[1 - learner_position] = StrategicAgent(ruleset=ruleset_name)
    _profile_worker_section(runtime_profile, "agent_setup", section_started)

    engine, event_stats, captures = _play_training_game(
        agents,
        learner_position,
        learner,
        schema,
        runtime_profile=runtime_profile,
        ruleset_name=ruleset_name,
        capture_opponent_decision_restarts=capture_opponent_decision_restarts,
    )
    section_started = _profile_worker_start(runtime_profile)
    terminal_utility, terminal_stats = _terminal_outcome(
        engine, learner_position, schema
    )
    samples = _finish_episode_with_rewards(
        learner,
        terminal_utility,
        gamma_f,
        schema["reward_eta"],
        terminal_turn=engine.turn,
        reward_distance_mode=schema["reward_distance_mode"],
    )
    _profile_worker_section(
        runtime_profile,
        "terminal_reward_and_trajectory_finalization",
        section_started,
    )
    result = (
        samples, event_stats, engine.winner, learner_position, terminal_stats
    )
    if capture_opponent_decision_restarts:
        return (*result, tuple(captures))
    return result


def _collect_steps_vs_random(
    network,
    schema,
    gamma_f,
    runtime_profile=None,
    ruleset_name=DEFAULT_RULESET_NAME,
    capture_opponent_decision_restarts=False,
    use_opponent_suit_features=True,
    use_opponent_bucket_features=False,
    opponent_bucket=None,
):
    """Play one training game against the fixed uniform-random agent."""
    section_started = _profile_worker_start(runtime_profile)
    learner_position = random.randint(0, 1)
    learner_policy_profile = (
        runtime_profile.setdefault("learner_policy", {})
        if runtime_profile is not None
        else None
    )
    learner = RLAgent(
        network,
        mode="training",
        runtime_profile=learner_policy_profile,
        ruleset=ruleset_name,
        use_opponent_suit_features=use_opponent_suit_features,
        use_opponent_bucket_features=use_opponent_bucket_features,
        opponent_bucket=opponent_bucket,
    )
    agents = [None, None]
    agents[learner_position] = learner
    agents[1 - learner_position] = RandomAgent()
    _profile_worker_section(runtime_profile, "agent_setup", section_started)

    engine, event_stats, captures = _play_training_game(
        agents,
        learner_position,
        learner,
        schema,
        runtime_profile=runtime_profile,
        ruleset_name=ruleset_name,
        capture_opponent_decision_restarts=capture_opponent_decision_restarts,
    )
    section_started = _profile_worker_start(runtime_profile)
    terminal_utility, terminal_stats = _terminal_outcome(
        engine, learner_position, schema
    )
    samples = _finish_episode_with_rewards(
        learner,
        terminal_utility,
        gamma_f,
        schema["reward_eta"],
        terminal_turn=engine.turn,
        reward_distance_mode=schema["reward_distance_mode"],
    )
    _profile_worker_section(
        runtime_profile,
        "terminal_reward_and_trajectory_finalization",
        section_started,
    )
    result = (
        samples, event_stats, engine.winner, learner_position, terminal_stats
    )
    if capture_opponent_decision_restarts:
        return (*result, tuple(captures))
    return result


def collect_steps_from_restart(
    learner_network,
    opponent_kind,
    opponent_network,
    restart,
    schema,
    gamma_f,
    runtime_profile=None,
    ruleset_name=DEFAULT_RULESET_NAME,
    use_opponent_suit_features=True,
    use_opponent_bucket_features=False,
    opponent_bucket=None,
):
    """Continue one captured state with the learner in the opponent's seat."""
    engine = DominoEngine.from_restart_state(restart.engine_state)
    if engine.ruleset.name != ruleset_name:
        raise ValueError("Restart state ruleset does not match the rollout runner.")
    learner_position = restart.restart_learner_position
    if engine.current_player != learner_position:
        raise ValueError("Restart learner must act first from the captured state.")
    if len(_tile_play_actions(engine.valid_actions(learner_position))) < 2:
        raise ValueError("Restart state is not a genuine tile-play decision.")

    learner_policy_profile = (
        runtime_profile.setdefault("learner_policy", {})
        if runtime_profile is not None
        else None
    )
    opponent_policy_profile = (
        runtime_profile.setdefault("opponent_policy", {})
        if runtime_profile is not None
        else None
    )
    learner = RLAgent(
        learner_network,
        mode="training",
        runtime_profile=learner_policy_profile,
        ruleset=ruleset_name,
        use_opponent_suit_features=use_opponent_suit_features,
        use_opponent_bucket_features=use_opponent_bucket_features,
        opponent_bucket=opponent_bucket,
    )
    if opponent_kind == "policy_snapshot":
        if opponent_network is None:
            raise ValueError("A policy restart requires its source bank slot.")
        counterpart = RLAgent(
            opponent_network,
            mode="stochastic_evaluation",
            runtime_profile=opponent_policy_profile,
            ruleset=ruleset_name,
            use_opponent_suit_features=use_opponent_suit_features,
            use_opponent_bucket_features=use_opponent_bucket_features,
            opponent_bucket=LEARNER_OPPONENT_BUCKET,
        )
    elif opponent_kind == "heuristic":
        if opponent_network is not None:
            raise ValueError("A heuristic restart cannot carry neural weights.")
        counterpart = StrategicAgent(ruleset=ruleset_name)
    elif opponent_kind == "random":
        if opponent_network is not None:
            raise ValueError("A random restart cannot carry neural weights.")
        counterpart = RandomAgent()
    else:
        raise ValueError(f"Unknown RL opponent kind: {opponent_kind!r}")

    agents = [None, None]
    agents[learner_position] = learner
    agents[restart.original_learner_position] = counterpart
    engine, event_stats, _captures = _play_training_game(
        agents,
        learner_position,
        learner,
        schema,
        runtime_profile=runtime_profile,
        ruleset_name=ruleset_name,
        engine=engine,
    )
    terminal_utility, terminal_stats = _terminal_outcome(
        engine, learner_position, schema
    )
    samples = _finish_episode_with_rewards(
        learner,
        terminal_utility,
        gamma_f,
        schema["reward_eta"],
        terminal_turn=engine.turn,
        reward_distance_mode=schema["reward_distance_mode"],
    )
    return (
        samples, event_stats, engine.winner, learner_position, terminal_stats
    )
