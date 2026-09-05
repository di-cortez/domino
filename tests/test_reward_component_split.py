"""The immediate return splits into ``G_D`` and ``G_P`` without moving a number.

Splitting ``local_reward`` is a refactor of the training objective's plumbing,
not a change to it. These tests pin both halves of that claim: the sum is
unchanged, and the two halves really do separate draw events from pass events.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from agents.rl_agent import RLAgent, TrajectoryStep
from training.rl.reward_model import DRAW_EVENT, PASS_EVENT
from training.rl.rollout import (
    DEFAULT_REWARD_SCHEMA,
    EventStats,
    _collect_steps_vs_random,
    _event_reward_for_action,
)


def _agent(*decision_turns):
    agent = object.__new__(RLAgent)
    agent.trajectory = [
        TrajectoryStep(None, 0, None, decision_turn=turn)
        for turn in decision_turns
    ]
    return agent


def test_local_reward_is_exactly_the_sum_of_the_two_halves():
    agent = _agent(2, 5, 8)
    agent.add_decayed_event_reward(10, 0.2, 0.9, "turn", event_kind=DRAW_EVENT)
    agent.add_decayed_event_reward(11, -0.1, 0.9, "turn", event_kind=PASS_EVENT)

    for step in agent.trajectory:
        assert step.local_reward == step.draw_return + step.pass_return


def test_each_kind_lands_in_its_own_half_and_leaves_the_other_at_zero():
    draws_only = _agent(2)
    draws_only.add_decayed_event_reward(4, 0.2, 0.9, "turn", event_kind=DRAW_EVENT)
    assert draws_only.trajectory[0].pass_return == 0.0
    assert draws_only.trajectory[0].draw_return == pytest.approx(0.2 * 0.9)

    passes_only = _agent(2)
    passes_only.add_decayed_event_reward(4, 0.2, 0.9, "turn", event_kind=PASS_EVENT)
    assert passes_only.trajectory[0].draw_return == 0.0
    assert passes_only.trajectory[0].pass_return == pytest.approx(0.2 * 0.9)


def test_the_sum_matches_what_a_single_undivided_accumulator_would_hold():
    """The pre-split behaviour, reproduced by adding both kinds to one total."""
    agent = _agent(1, 4, 7)
    events = [
        (9, 0.20, DRAW_EVENT),
        (10, -0.10, PASS_EVENT),
        (11, 0.05, DRAW_EVENT),
        (12, -0.02, PASS_EVENT),
    ]
    for turn, reward, kind in events:
        agent.add_decayed_event_reward(turn, reward, 0.9, "turn", event_kind=kind)

    for index, step in enumerate(agent.trajectory):
        undivided = sum(
            reward * 0.9 ** (turn - step.decision_turn - 1)
            for turn, reward, _kind in events
        )
        assert step.local_reward == pytest.approx(undivided, abs=1e-15), index


def test_event_reward_reports_the_kind_alongside_the_value():
    stats = EventStats()
    assert _event_reward_for_action(1, 0, ("DRAW", None), stats) == (
        1.0, DRAW_EVENT
    )
    assert _event_reward_for_action(1, 0, None, stats) == (1.0, PASS_EVENT)
    # A tile play is not an event and stays outside the immediate return.
    assert _event_reward_for_action(1, 0, ((3, 4), 0), stats) is None


def test_an_unknown_event_kind_is_refused():
    agent = _agent(1)
    with pytest.raises(ValueError, match="Unknown event kind"):
        agent.add_decayed_event_reward(3, 0.1, 0.9, "turn", event_kind="tile")


class _UniformNetwork:
    """A frozen policy that makes every legal action equally likely."""

    xp = np
    cache = {}
    logits_key = "Z3"

    def __init__(self, action_count=20):
        self.action_count = action_count

    def forward(self, x):
        return np.full((self.action_count, 1), 1.0 / self.action_count)


def test_a_real_rollout_keeps_the_sum_identity_on_every_decision():
    """End to end: play real games and check the identity on each sample."""
    random.seed(20260904)
    np.random.seed(20260904)
    checked = 0
    for _game in range(40):
        samples, *_rest = _collect_steps_vs_random(
            _UniformNetwork(),
            DEFAULT_REWARD_SCHEMA,
            DEFAULT_REWARD_SCHEMA["gamma_f"],
            ruleset_name="double-three",
        )
        for sample in samples:
            # `policy_reward` is the mixed objective PPO consumes; it must stay
            # the exact sum of its two stored halves.
            assert sample.policy_reward == pytest.approx(
                sample.terminal_reward + sample.local_reward, abs=1e-12
            )
            checked += 1
    assert checked > 0, "the rollout produced no trainable decisions"
