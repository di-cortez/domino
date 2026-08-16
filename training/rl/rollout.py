"""Reward finalization and CPU-only game collection for RL rollouts."""

from dataclasses import dataclass
import random
import time

from agents.agent import RandomAgent
from agents.heuristic_agent import StrategicAgent
from agents.rl_agent import RLAgent
from middleware.domino_engine import DominoEngine
from middleware.rulesets import DEFAULT_RULESET_NAME
from training.rl.restarts import CapturedOpponentDecision

TERMINAL_WIN_REWARD = 1
TERMINAL_LOSS_REWARD = -TERMINAL_WIN_REWARD
FINAL_PIP_PENALTY = TERMINAL_WIN_REWARD/20

OPPONENT_DRAW_REWARD = TERMINAL_WIN_REWARD/5
LEARNER_DRAW_PENALTY = -OPPONENT_DRAW_REWARD
OPPONENT_PASS_REWARD = OPPONENT_DRAW_REWARD/2
LEARNER_PASS_PENALTY = -OPPONENT_PASS_REWARD

# Per-turn decay applied to a local event reward as it is credited backwards to
# the real decisions that preceded the event.
EVENT_REWARD_DECAY = 0.90

# Terminal-reward discount applied per remaining real decision (1.0 keeps the
# historical undiscounted terminal outcome).
DEFAULT_GAMMA = 1.0

# Convex mixture weight between the two reward components of one decision:
# 0.0 trains on the terminal outcome alone, 1.0 on local event shaping alone.
ALPHA = 0.5

REWARD_ZERO_EPSILON = 1e-8

# The single set of terminal and local event reward constants. The mutable
# mapping shape remains part of the historical self-play compatibility surface:
# it is copied per run, overridden with the tunables resolved from the CLI, and
# shipped unchanged to every rollout worker.
REWARD_SCHEMAS = {
    "terminal_win": TERMINAL_WIN_REWARD,
    "terminal_loss": TERMINAL_LOSS_REWARD,
    "final_pip_penalty": FINAL_PIP_PENALTY,
    "opponent_draw": OPPONENT_DRAW_REWARD,
    "learner_draw": LEARNER_DRAW_PENALTY,
    "opponent_pass": OPPONENT_PASS_REWARD,
    "learner_pass": LEARNER_PASS_PENALTY,
    "event_decay": EVENT_REWARD_DECAY,
    "alpha": ALPHA,
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
    """Return the local event reward for draw/pass actions and update counts."""
    if schema is None:
        schema = REWARD_SCHEMAS
    if current_player != learner_position:
        if action == ("DRAW", None):
            event_stats.opponent_draws += 1
            return schema["opponent_draw"]
        if action is None:
            event_stats.opponent_passes += 1
            return schema["opponent_pass"]
    else:
        if action == ("DRAW", None):
            event_stats.learner_draws += 1
            return schema["learner_draw"]
        if action is None:
            event_stats.learner_passes += 1
            return schema["learner_pass"]
    return None


def _remaining_pips(hand):
    return sum(tile[0] + tile[1] for tile in hand)


def _terminal_reward(engine, learner_position, schema):
    """Return terminal outcome reward plus final remaining-pip penalty."""
    winner = engine.winner
    if winner == learner_position:
        outcome_reward = schema["terminal_win"]
    else:
        outcome_reward = schema["terminal_loss"]

    pip_penalty = schema["final_pip_penalty"] * _remaining_pips(
        engine.hands[learner_position]
    )
    return outcome_reward - pip_penalty


def _finish_episode_with_rewards(
    learner_agent, terminal_reward, gamma=DEFAULT_GAMMA, alpha=ALPHA
):
    """Finalize one learner trajectory into policy-gradient training samples.

    Each decision's total reward is the convex combination

        R_T = (1 - alpha) * gamma ** k * R_f + alpha * R_l

    where ``R_f`` is the episode's terminal reward, ``k`` the number of real
    decisions still remaining after this one, and ``R_l`` the decayed local
    event reward already accumulated on the step. ``gamma`` therefore discounts
    the terminal component per remaining decision, so earlier decisions receive
    a more heavily discounted share of the final outcome than the last one,
    while ``alpha`` trades the two components off against each other:
    ``alpha=0`` trains on the terminal outcome alone and ``alpha=1`` on local
    event shaping alone.

    The stored ``terminal_reward`` and ``local_reward`` components are the
    scaled halves, so ``policy_reward == terminal_reward + local_reward``
    remains an identity for the PPO buffer and the reward diagnostics.
    """
    finished_steps = learner_agent.finish_episode(terminal_reward)
    step_count = len(finished_steps)
    samples = []
    for index, step in enumerate(finished_steps):
        remaining_after = step_count - 1 - index
        discounted_terminal = (1.0 - alpha) * step.terminal_reward * (
            gamma ** remaining_after
        )
        scaled_local = alpha * step.local_reward
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
        event_reward = _event_reward_for_action(
            current_player,
            learner_position,
            action,
            event_stats,
            schema,
        )
        if event_reward is not None:
            learner_agent.add_decayed_event_reward(
                event_turn=state["turn"],
                base_reward=event_reward,
                decay_lambda=schema["event_decay"],
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
        event_reward = _event_reward_for_action(
            current_player,
            learner_position,
            action,
            event_stats,
            schema,
        )
        if event_reward is not None:
            learner_agent.add_decayed_event_reward(
                event_turn=state["turn"],
                base_reward=event_reward,
                decay_lambda=schema["event_decay"],
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
    gamma,
    runtime_profile=None,
    ruleset_name=DEFAULT_RULESET_NAME,
    capture_opponent_decision_restarts=False,
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
    )
    opponent = RLAgent(
        opponent_network,
        mode="stochastic_evaluation",
        runtime_profile=opponent_policy_profile,
        ruleset=ruleset_name,
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
        capture_opponent_decision_restarts=(
            capture_opponent_decision_restarts
        ),
    )
    section_started = _profile_worker_start(runtime_profile)
    reward = _terminal_reward(engine, learner_position, schema)
    samples = _finish_episode_with_rewards(
        learner, reward, gamma, schema["alpha"]
    )
    _profile_worker_section(
        runtime_profile,
        "terminal_reward_and_trajectory_finalization",
        section_started,
    )
    result = (samples, event_stats, engine.winner, learner_position)
    if capture_opponent_decision_restarts:
        return (*result, tuple(captures))
    return result


def collect_steps_for_assignment(
    learner_network,
    opponent_kind,
    opponent_network,
    schema,
    gamma,
    runtime_profile=None,
    ruleset_name=DEFAULT_RULESET_NAME,
    capture_opponent_decision_restarts=False,
):
    """Dispatch one preselected assignment without knowing pool policy."""
    if opponent_kind == "policy_snapshot":
        if opponent_network is None:
            raise ValueError("A policy-snapshot assignment requires weights")
        return _collect_steps_vs_snapshot(
            learner_network,
            opponent_network,
            schema,
            gamma,
            runtime_profile=runtime_profile,
            ruleset_name=ruleset_name,
            capture_opponent_decision_restarts=(
                capture_opponent_decision_restarts
            ),
        )
    if opponent_kind == "heuristic":
        if opponent_network is not None:
            raise ValueError("A heuristic assignment must not carry neural weights")
        return _collect_steps_vs_heuristic(
            learner_network,
            schema,
            gamma,
            runtime_profile=runtime_profile,
            ruleset_name=ruleset_name,
            capture_opponent_decision_restarts=(
                capture_opponent_decision_restarts
            ),
        )
    if opponent_kind == "random":
        if opponent_network is not None:
            raise ValueError("A random assignment must not carry neural weights")
        return _collect_steps_vs_random(
            learner_network,
            schema,
            gamma,
            runtime_profile=runtime_profile,
            ruleset_name=ruleset_name,
            capture_opponent_decision_restarts=(
                capture_opponent_decision_restarts
            ),
        )
    raise ValueError(f"Unknown RL opponent kind: {opponent_kind!r}")


def _collect_steps_vs_heuristic(
    network,
    schema,
    gamma,
    runtime_profile=None,
    ruleset_name=DEFAULT_RULESET_NAME,
    capture_opponent_decision_restarts=False,
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
        capture_opponent_decision_restarts=(
            capture_opponent_decision_restarts
        ),
    )
    section_started = _profile_worker_start(runtime_profile)
    reward = _terminal_reward(engine, learner_position, schema)
    samples = _finish_episode_with_rewards(
        learner, reward, gamma, schema["alpha"]
    )
    _profile_worker_section(
        runtime_profile,
        "terminal_reward_and_trajectory_finalization",
        section_started,
    )
    result = (samples, event_stats, engine.winner, learner_position)
    if capture_opponent_decision_restarts:
        return (*result, tuple(captures))
    return result


def _collect_steps_vs_random(
    network,
    schema,
    gamma,
    runtime_profile=None,
    ruleset_name=DEFAULT_RULESET_NAME,
    capture_opponent_decision_restarts=False,
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
        capture_opponent_decision_restarts=(
            capture_opponent_decision_restarts
        ),
    )
    section_started = _profile_worker_start(runtime_profile)
    reward = _terminal_reward(engine, learner_position, schema)
    samples = _finish_episode_with_rewards(
        learner, reward, gamma, schema["alpha"]
    )
    _profile_worker_section(
        runtime_profile,
        "terminal_reward_and_trajectory_finalization",
        section_started,
    )
    result = (samples, event_stats, engine.winner, learner_position)
    if capture_opponent_decision_restarts:
        return (*result, tuple(captures))
    return result


def collect_steps_from_restart(
    learner_network,
    opponent_kind,
    opponent_network,
    restart,
    schema,
    gamma,
    runtime_profile=None,
    ruleset_name=DEFAULT_RULESET_NAME,
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
    )
    if opponent_kind == "policy_snapshot":
        if opponent_network is None:
            raise ValueError("A policy restart requires its source bank slot.")
        counterpart = RLAgent(
            opponent_network,
            mode="stochastic_evaluation",
            runtime_profile=opponent_policy_profile,
            ruleset=ruleset_name,
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
    reward = _terminal_reward(engine, learner_position, schema)
    samples = _finish_episode_with_rewards(
        learner,
        reward,
        gamma,
        schema["alpha"],
    )
    return samples, event_stats, engine.winner, learner_position
