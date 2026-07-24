"""Reward finalization and CPU-only game collection for RL rollouts."""

from dataclasses import dataclass
import random
import time

from agents.heuristic_agent import StrategicAgent
from agents.rl_agent import RLAgent
from middleware.domino_engine import DominoEngine


TERMINAL_WIN_REWARD = 0.50
TERMINAL_TIE_REWARD = 0.00
TERMINAL_LOSS_REWARD = -0.50
FINAL_PIP_PENALTY = 0.001

OPPONENT_DRAW_REWARD = 0.02
LEARNER_DRAW_PENALTY = -0.02
OPPONENT_PASS_REWARD = 0.10
LEARNER_PASS_PENALTY = -0.10
EVENT_REWARD_DECAY = 0.90

REWARD_ZERO_EPSILON = 1e-8

# Named presets for terminal and local event rewards. The values and mutable
# mapping shape remain part of the historical self-play compatibility surface.
REWARD_SCHEMAS = {
    "default": {
        "terminal_win": TERMINAL_WIN_REWARD,
        "terminal_tie": TERMINAL_TIE_REWARD,
        "terminal_loss": TERMINAL_LOSS_REWARD,
        "final_pip_penalty": FINAL_PIP_PENALTY,
        "opponent_draw": OPPONENT_DRAW_REWARD,
        "learner_draw": LEARNER_DRAW_PENALTY,
        "opponent_pass": OPPONENT_PASS_REWARD,
        "learner_pass": LEARNER_PASS_PENALTY,
        "event_decay": EVENT_REWARD_DECAY,
    },
    "sparse": {
        "terminal_win": TERMINAL_WIN_REWARD,
        "terminal_tie": TERMINAL_TIE_REWARD,
        "terminal_loss": TERMINAL_LOSS_REWARD,
        "final_pip_penalty": 0.0,
        "opponent_draw": 0.0,
        "learner_draw": 0.0,
        "opponent_pass": 0.0,
        "learner_pass": 0.0,
        "event_decay": EVENT_REWARD_DECAY,
    },
    "shaped": {
        "terminal_win": TERMINAL_WIN_REWARD,
        "terminal_tie": TERMINAL_TIE_REWARD,
        "terminal_loss": TERMINAL_LOSS_REWARD,
        "final_pip_penalty": FINAL_PIP_PENALTY,
        "opponent_draw": OPPONENT_DRAW_REWARD * 2.0,
        "learner_draw": LEARNER_DRAW_PENALTY * 2.0,
        "opponent_pass": OPPONENT_PASS_REWARD * 2.0,
        "learner_pass": LEARNER_PASS_PENALTY * 2.0,
        "event_decay": EVENT_REWARD_DECAY,
    },
}
DEFAULT_REWARD_SCHEMA = "default"

# Terminal-reward discount applied per remaining real decision (1.0 keeps the
# historical undiscounted terminal outcome).
DEFAULT_GAMMA = 1.0


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
        schema = REWARD_SCHEMAS[DEFAULT_REWARD_SCHEMA]
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
    if winner == -1:
        outcome_reward = schema["terminal_tie"]
    elif winner == learner_position:
        outcome_reward = schema["terminal_win"]
    else:
        outcome_reward = schema["terminal_loss"]

    pip_penalty = schema["final_pip_penalty"] * _remaining_pips(
        engine.hands[learner_position]
    )
    return outcome_reward - pip_penalty


def _finish_episode_with_rewards(
    learner_agent, terminal_reward, gamma=DEFAULT_GAMMA
):
    """Finalize one learner trajectory into policy-gradient training samples.

    ``gamma`` discounts the terminal-reward component per remaining real
    decision, so earlier decisions in the trajectory receive a more heavily
    discounted share of the final outcome than the last one. Local event
    rewards already carry their own temporal decay and are not affected.
    """
    finished_steps = learner_agent.finish_episode(terminal_reward)
    step_count = len(finished_steps)
    samples = []
    for index, step in enumerate(finished_steps):
        remaining_after = step_count - 1 - index
        discounted_terminal = step.terminal_reward * (gamma ** remaining_after)
        raw_reward = discounted_terminal + step.local_reward
        samples.append(
            TrainingSample(
                x=step.x,
                action_index=step.action_index,
                legal_mask=step.legal_mask,
                old_log_prob=step.old_log_prob,
                policy_reward=raw_reward,
                raw_reward=raw_reward,
                local_reward=step.local_reward,
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


def _play_training_game_unprofiled(
    agents,
    learner_position,
    learner_agent,
    schema,
):
    """Profiler-free rollout hot path for non-sampled games."""
    engine = DominoEngine(player_count=len(agents))
    event_stats = EventStats()
    while not engine.game_over:
        state = engine._get_state()
        current_player = state["current_player"]
        legal_actions = engine.valid_actions(current_player)
        tile_actions = _tile_play_actions(legal_actions)
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
    return engine, event_stats


def _play_training_game(
    agents,
    learner_position,
    learner_agent,
    schema,
    runtime_profile=None,
):
    """Play one RL training game and attach decayed local event rewards."""
    if runtime_profile is None:
        return _play_training_game_unprofiled(
            agents,
            learner_position,
            learner_agent,
            schema,
        )
    section_started = _profile_worker_start(runtime_profile)
    engine = DominoEngine(player_count=len(agents))
    event_stats = EventStats()
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

    return engine, event_stats


def _collect_self_play_steps(
    network, pool, schema, gamma, runtime_profile=None
):
    """Play one game against a frozen policy snapshot and collect learner samples."""
    section_started = _profile_worker_start(runtime_profile)
    learner_position = random.randint(0, 1)
    opponent_network = random.choice(pool) if pool else network

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
    )
    opponent = RLAgent(
        opponent_network,
        mode="stochastic_evaluation",
        runtime_profile=opponent_policy_profile,
    )
    agents = [None, None]
    agents[learner_position] = learner
    agents[1 - learner_position] = opponent
    _profile_worker_section(runtime_profile, "agent_setup", section_started)

    engine, event_stats = _play_training_game(
        agents,
        learner_position,
        learner,
        schema,
        runtime_profile=runtime_profile,
    )
    section_started = _profile_worker_start(runtime_profile)
    reward = _terminal_reward(engine, learner_position, schema)
    samples = _finish_episode_with_rewards(learner, reward, gamma)
    _profile_worker_section(
        runtime_profile,
        "terminal_reward_and_trajectory_finalization",
        section_started,
    )
    return samples, event_stats, engine.winner, learner_position


def _collect_steps_vs_heuristic(
    network, schema, gamma, runtime_profile=None
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
    )
    agents = [None, None]
    agents[learner_position] = learner
    agents[1 - learner_position] = StrategicAgent()
    _profile_worker_section(runtime_profile, "agent_setup", section_started)

    engine, event_stats = _play_training_game(
        agents,
        learner_position,
        learner,
        schema,
        runtime_profile=runtime_profile,
    )
    section_started = _profile_worker_start(runtime_profile)
    reward = _terminal_reward(engine, learner_position, schema)
    samples = _finish_episode_with_rewards(learner, reward, gamma)
    _profile_worker_section(
        runtime_profile,
        "terminal_reward_and_trajectory_finalization",
        section_started,
    )
    return samples, event_stats, engine.winner, learner_position
