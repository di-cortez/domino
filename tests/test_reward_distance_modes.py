"""Deterministic tests for configurable RL reward-discount distances."""

import numpy as np
import pytest

from agents.rl_agent import RLAgent, TrajectoryStep
from training.pipeline import parse_args as parse_pipeline_args
from training.rl.cli import parse_args as parse_rl_args
from training.rl.config import (
    RLExecutionOptions,
    RLResourceOptions,
    RLTrainingOptions,
    resolve_training_options,
)
from training.rl.reward_distance import (
    DEFAULT_REWARD_DISTANCE_MODE,
    REWARD_DISTANCE_MODES,
    resolve_reward_distance_mode,
)
from training.rl.rollout import _finish_episode_with_rewards


def _agent(*decision_turns):
    agent = object.__new__(RLAgent)
    agent.trajectory = [
        TrajectoryStep(
            x=np.asarray([[turn]], dtype=np.float32),
            action_index=index,
            legal_mask=np.asarray([[True], [True]]),
            decision_turn=turn,
        )
        for index, turn in enumerate(decision_turns)
    ]
    return agent


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("turn-turn", ("turn", "turn")),
        ("decision-decision", ("decision", "decision")),
        ("turn-decision", ("turn", "decision")),
        ("decision-turn", ("decision", "turn")),
    ],
)
def test_mode_resolution(mode, expected):
    assert resolve_reward_distance_mode(mode) == expected


def test_mode_resolution_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unknown reward distance mode"):
        resolve_reward_distance_mode("turn-clock")


def test_new_run_defaults_are_exposed_by_both_clis(tmp_path):
    standalone = parse_rl_args([])
    canonical = parse_pipeline_args([
        "small",
        "--artifact-root",
        str(tmp_path),
    ])

    for args in (standalone, canonical):
        assert args.gamma_i == 0.90
        assert args.gamma_f == 0.95
        assert args.reward_eta == 0.50
        assert args.reward_distance_mode == "turn-turn"


def test_resolved_schema_freezes_the_selected_mode():
    resolved = resolve_training_options(
        RLTrainingOptions(reward_distance_mode="decision-decision"),
        RLResourceOptions(),
        RLExecutionOptions(),
    )

    assert resolved.training.reward_distance_mode == "decision-decision"
    assert resolved.schema["reward_distance_mode"] == "decision-decision"


@pytest.mark.parametrize(
    ("metric", "distances"),
    [("turn", (7, 4, 1)), ("decision", (2, 1, 0))],
)
def test_local_reward_distances(metric, distances):
    agent = _agent(2, 5, 8)
    agent.add_decayed_event_reward(
        event_turn=10,
        base_reward=0.2,
        decay_lambda=0.9,
        distance_metric=metric,
    )

    assert [step.local_reward for step in agent.trajectory] == pytest.approx([
        0.2 * 0.9 ** distance for distance in distances
    ])


def test_decision_local_rewards_accumulate_across_events():
    agent = _agent(2, 5)
    agent.add_decayed_event_reward(8, 0.2, 0.9, "decision")
    agent.add_decayed_event_reward(9, -0.1, 0.9, "decision")

    assert [step.local_reward for step in agent.trajectory] == pytest.approx([
        0.2 * 0.9 - 0.1 * 0.9,
        0.2 - 0.1,
    ])


@pytest.mark.parametrize("mode", REWARD_DISTANCE_MODES)
def test_each_mode_controls_both_reward_components(mode):
    local_metric, terminal_metric = resolve_reward_distance_mode(mode)
    local_distances = (4, 1) if local_metric == "turn" else (1, 0)
    terminal_distances = (5, 2) if terminal_metric == "turn" else (1, 0)
    agent = _agent(2, 5)
    agent.add_decayed_event_reward(7, 0.2, 0.9, local_metric)

    samples = _finish_episode_with_rewards(
        agent,
        terminal_utility=0.8,
        gamma_f=0.95,
        reward_eta=0.5,
        terminal_turn=8,
        reward_distance_mode=mode,
    )

    for sample, local_distance, terminal_distance in zip(
        samples,
        local_distances,
        terminal_distances,
        strict=True,
    ):
        expected_local = 0.5 * 0.2 * 0.9 ** local_distance
        expected_terminal = 0.5 * 0.8 * 0.95 ** terminal_distance
        assert sample.local_reward == pytest.approx(expected_local)
        assert sample.terminal_reward == pytest.approx(expected_terminal)
        assert sample.policy_reward == pytest.approx(
            expected_local + expected_terminal
        )


@pytest.mark.parametrize(
    ("mode", "terminal_distances"),
    [
        ("turn-turn", (8, 5, 2)),
        ("decision-decision", (2, 1, 0)),
        ("turn-decision", (2, 1, 0)),
        ("decision-turn", (8, 5, 2)),
    ],
)
def test_terminal_reward_distances_for_all_modes(mode, terminal_distances):
    samples = _finish_episode_with_rewards(
        _agent(2, 5, 8),
        terminal_utility=0.8,
        gamma_f=0.95,
        reward_eta=0.25,
        terminal_turn=11,
        reward_distance_mode=mode,
    )

    expected = [
        0.75 * 0.8 * 0.95 ** distance
        for distance in terminal_distances
    ]
    assert [sample.terminal_reward for sample in samples] == pytest.approx(
        expected
    )
    assert all(sample.raw_reward == sample.policy_reward for sample in samples)


def test_turn_terminal_distance_uses_absolute_restart_timeline():
    samples = _finish_episode_with_rewards(
        _agent(23),
        terminal_utility=1.0,
        gamma_f=0.95,
        reward_eta=0.0,
        terminal_turn=31,
        reward_distance_mode="decision-turn",
    )

    assert samples[0].terminal_reward == pytest.approx(0.95 ** 7)


def test_finalization_requires_explicit_terminal_context():
    with pytest.raises(TypeError, match="terminal_turn"):
        _finish_episode_with_rewards(
            _agent(2),
            1.0,
            reward_distance_mode=DEFAULT_REWARD_DISTANCE_MODE,
        )


def test_chronology_validation_covers_local_and_terminal_turn_metrics():
    agent = _agent(5)
    with pytest.raises(ValueError, match="Event reward chronology"):
        agent.add_decayed_event_reward(5, 0.2, 0.9, "turn")
    with pytest.raises(ValueError, match="Terminal reward chronology"):
        _finish_episode_with_rewards(
            _agent(5),
            1.0,
            terminal_turn=5,
            reward_distance_mode="decision-turn",
        )


def test_public_mode_registry_has_exactly_four_choices():
    assert REWARD_DISTANCE_MODES == (
        "turn-turn",
        "decision-decision",
        "turn-decision",
        "decision-turn",
    )
