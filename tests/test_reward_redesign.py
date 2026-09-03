"""Invariants of the empty-hand/blocked reward decomposition.

The reward is deliberately built in four separable layers -- event semantics,
relative component weights, temporal discounting, terminal/immediate mixture --
so that an ablation on one of them cannot silently move another. These tests
pin the two layers ``training.rl.reward_model`` owns, plus the boundary where
the runtime and the resume identity consume them.
"""

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from middleware.domino_engine import (
    DominoEngine,
    WIN_REASON_BLOCKED_FEWEST_PIPS,
    WIN_REASON_BLOCKED_FEWEST_TILES,
    WIN_REASON_BLOCKED_LAST_VALID_PLAY,
    WIN_REASON_EMPTY_HAND,
)
from middleware.rulesets import RULESET_NAMES, resolve_ruleset
from training.rl.config import (
    RLExecutionOptions,
    RLResourceOptions,
    RLTrainingOptions,
    resolve_training_options,
)
from training.rl.resume import RLTrainingConfiguration
from training.rl.reward_model import (
    BLOCKED_REWARD_CEILING,
    BLOCKED_REWARD_FLOOR,
    BLOCKED_WIN_REASONS,
    DRAW_EVENT,
    PASS_EVENT,
    blocked_reward_magnitude,
    combine_terminal_components,
    normalize_weight_pair,
    resolved_reward_scales,
    scaled_event_reward,
    terminal_reward_components,
    unit_event_reward,
)
from training.rl.reporting import _reward_signal_summary
from training.rl.rollout import (
    DEFAULT_REWARD_SCHEMA,
    TerminalStats,
    TrainingSample,
    _terminal_outcome,
)


def _finished_engine(hands, *, winner, win_reason, ruleset="double-six"):
    """Return an engine parked on one already-decided terminal state."""
    engine = DominoEngine(player_count=2, ruleset=ruleset)
    engine.hands = [list(hand) for hand in hands]
    engine.stock = []
    engine.game_over = True
    engine.winner = winner
    engine.win_reason = win_reason
    return engine


def _blocked_engine(hands, *, winner, ruleset="double-six"):
    return _finished_engine(
        hands,
        winner=winner,
        win_reason=WIN_REASON_BLOCKED_FEWEST_PIPS,
        ruleset=ruleset,
    )


def _empty_hand_engine(loser_hand, *, winner):
    hands = [[], []]
    hands[1 - winner] = list(loser_hand)
    return _finished_engine(
        hands,
        winner=winner,
        win_reason=WIN_REASON_EMPTY_HAND,
    )


# --- Terminal semantics -----------------------------------------------------


@pytest.mark.parametrize("learner", (0, 1))
def test_empty_hand_ending_is_binary_and_leaves_the_blocked_term_at_zero(
    learner,
):
    engine = _empty_hand_engine([(6, 6), (5, 4)], winner=0)

    components = terminal_reward_components(engine, learner)

    assert components["blocked_component"] == 0.0
    assert components["empty_hand_component"] == (1.0 if learner == 0 else -1.0)
    assert components["pip_margin"] is None


def test_empty_hand_reward_ignores_the_learner_own_remaining_pips():
    """The 0.05-per-pip penalty is gone, not merely rescaled.

    Under the old reward a loser holding a heavy hand was punished twice, and
    the pip term applied to blocked and empty-hand endings alike. The blocked
    branch now carries the whole pip story through its margin, so an empty-hand
    loss is worth exactly -1 regardless of what the loser was still holding.
    """
    light = _empty_hand_engine([(0, 1)], winner=0)
    heavy = _empty_hand_engine([(6, 6), (6, 5), (5, 5)], winner=0)

    assert terminal_reward_components(light, 1) == (
        terminal_reward_components(heavy, 1)
    )
    assert terminal_reward_components(heavy, 1)["empty_hand_component"] == -1.0


@pytest.mark.parametrize("learner", (0, 1))
def test_blocked_ending_uses_the_margin_and_leaves_the_empty_hand_term_at_zero(
    learner,
):
    # Winner holds 3 pips, loser holds 9: a margin of 6 on a saturation of 12.
    engine = _blocked_engine([[(1, 2)], [(4, 5)]], winner=0)

    components = terminal_reward_components(engine, learner)

    assert components["empty_hand_component"] == 0.0
    assert components["winner_final_pips"] == 3
    assert components["loser_final_pips"] == 9
    assert components["pip_margin"] == 6
    assert components["blocked_magnitude"] == pytest.approx(0.55)
    expected = 0.55 if learner == 0 else -0.55
    assert components["blocked_component"] == pytest.approx(expected)


def test_blocked_perspectives_are_symmetric():
    engine = _blocked_engine([[(1, 2)], [(4, 5)]], winner=0)

    winner_view = terminal_reward_components(engine, 0)
    loser_view = terminal_reward_components(engine, 1)

    assert winner_view["blocked_component"] == -loser_view["blocked_component"]
    assert winner_view["pip_margin"] == loser_view["pip_margin"]


@pytest.mark.parametrize(
    "win_reason",
    (WIN_REASON_BLOCKED_FEWEST_TILES, WIN_REASON_BLOCKED_LAST_VALID_PLAY),
)
def test_a_tie_broken_blocked_win_is_still_blocked_and_maps_to_the_floor(
    win_reason,
):
    """A zero margin is legal, and both later tie-breakers stay blocked.

    The engine compares pip totals first, so a win by fewest tiles or by the
    last valid play means the totals were equal. That is a real but minimally
    decisive blocked win, which is exactly what the floor expresses.
    """
    engine = _finished_engine(
        [[(1, 2)], [(0, 3), (0, 0)]],
        winner=0,
        win_reason=win_reason,
    )

    components = terminal_reward_components(engine, 0)

    assert win_reason in BLOCKED_WIN_REASONS
    assert components["pip_margin"] == 0
    assert components["blocked_component"] == pytest.approx(
        BLOCKED_REWARD_FLOOR
    )


def test_blocked_component_stays_inside_the_floor_and_ceiling_band():
    for margin, hands in (
        (0, [[(1, 2)], [(0, 3)]]),
        (6, [[(1, 2)], [(4, 5)]]),
        (26, [[(1, 2)], [(6, 6), (6, 5), (6, 0)]]),
    ):
        engine = _blocked_engine(hands, winner=0)
        components = terminal_reward_components(engine, 0)

        assert components["pip_margin"] == margin
        assert BLOCKED_REWARD_FLOOR <= components["blocked_component"] <= (
            BLOCKED_REWARD_CEILING
        )


# --- Blocked magnitude ------------------------------------------------------


@pytest.mark.parametrize("ruleset_name", RULESET_NAMES)
def test_saturation_margin_is_derived_from_the_ruleset(ruleset_name):
    """``S = 2 * max_pip``, never a table keyed by ruleset name."""
    max_pip = resolve_ruleset(ruleset_name).max_pip
    saturation = 2 * max_pip

    assert blocked_reward_magnitude(0, max_pip) == pytest.approx(
        BLOCKED_REWARD_FLOOR
    )
    assert blocked_reward_magnitude(saturation, max_pip) == pytest.approx(
        BLOCKED_REWARD_CEILING
    )
    # Past saturation the magnitude stays clamped rather than growing.
    assert blocked_reward_magnitude(saturation * 3, max_pip) == pytest.approx(
        BLOCKED_REWARD_CEILING
    )
    assert blocked_reward_magnitude(saturation / 2, max_pip) == pytest.approx(
        0.55
    )


def test_blocked_magnitude_is_monotone_in_the_margin():
    values = [blocked_reward_magnitude(margin, 6) for margin in range(0, 13)]

    assert values == sorted(values)
    assert values[0] == pytest.approx(BLOCKED_REWARD_FLOOR)
    assert values[-1] == pytest.approx(BLOCKED_REWARD_CEILING)


def test_a_negative_margin_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        blocked_reward_magnitude(-1, 6)


def test_an_unknown_terminal_state_is_rejected():
    unfinished = DominoEngine(player_count=2)
    with pytest.raises(ValueError, match="unfinished"):
        terminal_reward_components(unfinished, 0)

    unknown = _finished_engine([[(1, 2)], [(3, 4)]], winner=0, win_reason="x")
    with pytest.raises(ValueError, match="Unknown terminal win reason"):
        terminal_reward_components(unknown, 0)


# --- Immediate event semantics ---------------------------------------------


def test_draw_and_pass_events_share_one_unit_scale():
    """Their relative importance lives in the weights, not the magnitudes."""
    for event_kind in (DRAW_EVENT, PASS_EVENT):
        assert unit_event_reward(event_kind, by_learner=False) == 1.0
        assert unit_event_reward(event_kind, by_learner=True) == -1.0


def test_scaled_events_apply_only_their_own_pair_scale():
    assert scaled_event_reward(
        DRAW_EVENT, by_learner=False, draw_scale=1.0, pass_scale=0.25
    ) == 1.0
    assert scaled_event_reward(
        PASS_EVENT, by_learner=False, draw_scale=1.0, pass_scale=0.25
    ) == 0.25
    assert scaled_event_reward(
        PASS_EVENT, by_learner=True, draw_scale=1.0, pass_scale=0.25
    ) == -0.25


# --- Weight normalization ---------------------------------------------------


def test_normalized_scales_keep_the_dominant_component_on_unit_scale():
    for first, second in ((1.0, 1.0), (2.0, 1.0), (1.0, 4.0), (3.0, 0.0)):
        first_scale, second_scale, norm = normalize_weight_pair(
            first, second, first_name="a", second_name="b"
        )

        assert norm == max(first, second)
        assert max(first_scale, second_scale) == 1.0
        assert 0.0 <= min(first_scale, second_scale) <= 1.0


def test_only_the_ratio_of_a_pair_survives_normalization():
    doubled = resolved_reward_scales(
        terminal_empty_hand_weight=2.0,
        terminal_blocked_weight=1.0,
        immediate_draw_weight=6.0,
        immediate_pass_weight=3.0,
    )
    halved = resolved_reward_scales(
        terminal_empty_hand_weight=1.0,
        terminal_blocked_weight=0.5,
        immediate_draw_weight=1.0,
        immediate_pass_weight=0.5,
    )

    for key in ("empty_hand_scale", "blocked_scale", "draw_scale", "pass_scale"):
        assert doubled[key] == pytest.approx(halved[key])
    # The raw weights still record what the experiment actually asked for.
    assert doubled["terminal_empty_hand_weight"] == 2.0


@pytest.mark.parametrize(
    "weights",
    (
        {"terminal_empty_hand_weight": 0.0, "terminal_blocked_weight": 0.0},
        {"immediate_draw_weight": 0.0, "immediate_pass_weight": 0.0},
        {"terminal_blocked_weight": -1.0},
        {"immediate_pass_weight": float("nan")},
        {"immediate_draw_weight": float("inf")},
    ),
)
def test_degenerate_weight_pairs_are_rejected(weights):
    with pytest.raises(ValueError):
        resolved_reward_scales(**weights)


def test_resolve_training_options_validates_and_derives_the_scales():
    resolved = resolve_training_options(
        RLTrainingOptions(
            iterations=1,
            terminal_empty_hand_weight=1.0,
            terminal_blocked_weight=0.5,
            immediate_draw_weight=1.0,
            immediate_pass_weight=0.25,
        ),
        RLResourceOptions(),
        RLExecutionOptions(),
    )

    assert resolved.schema["blocked_scale"] == pytest.approx(0.5)
    assert resolved.schema["pass_scale"] == pytest.approx(0.25)
    # Raw weights travel with the derived scales so a run stays auditable.
    assert resolved.schema["terminal_blocked_weight"] == 0.5
    assert resolved.training.immediate_pass_weight == 0.25


# --- Runtime terminal utility ----------------------------------------------


def test_terminal_utility_applies_the_terminal_pair_scales():
    """``rho_B`` decides importance; ``m`` decides decisiveness. Separately."""
    schema = {
        **DEFAULT_REWARD_SCHEMA,
        "empty_hand_scale": 1.0,
        "blocked_scale": 0.5,
    }
    empty_hand = _empty_hand_engine([(6, 6)], winner=0)
    blocked = _blocked_engine([[(1, 2)], [(4, 5)]], winner=0)

    assert _terminal_outcome(empty_hand, 0, schema)[0] == pytest.approx(1.0)
    assert _terminal_outcome(blocked, 0, schema)[0] == pytest.approx(0.5 * 0.55)
    assert _terminal_outcome(blocked, 1, schema)[0] == pytest.approx(-0.5 * 0.55)


def test_a_zero_blocked_weight_deletes_only_the_blocked_half():
    schema = {
        **DEFAULT_REWARD_SCHEMA,
        "empty_hand_scale": 1.0,
        "blocked_scale": 0.0,
    }
    empty_hand = _empty_hand_engine([(6, 6)], winner=0)
    blocked = _blocked_engine([[(1, 2)], [(4, 5)]], winner=0)

    assert _terminal_outcome(empty_hand, 0, schema)[0] == pytest.approx(1.0)
    assert _terminal_outcome(blocked, 0, schema)[0] == 0.0


def test_combining_components_never_mixes_the_two_terminal_branches():
    blocked = terminal_reward_components(
        _blocked_engine([[(1, 2)], [(4, 5)]], winner=0), 0
    )
    empty_hand = terminal_reward_components(
        _empty_hand_engine([(6, 6)], winner=0), 0
    )

    # Each ending reads exactly one scale, so changing the other cannot move it.
    assert combine_terminal_components(
        blocked, empty_hand_scale=0.0, blocked_scale=1.0
    ) == pytest.approx(blocked["blocked_component"])
    assert combine_terminal_components(
        empty_hand, empty_hand_scale=1.0, blocked_scale=0.0
    ) == pytest.approx(empty_hand["empty_hand_component"])


# --- Run identity -----------------------------------------------------------


def test_a_pre_redesign_run_cannot_resume_under_the_new_objective():
    """Old checkpoints trained a different objective and must be restarted.

    Filling the missing weights with today's defaults would make an old run
    look identical to a new one while its terminal reward had been the binary
    outcome minus a per-pip penalty. The rejection is the point.
    """
    from tests.test_canonical_pipeline import _test_resume_configuration

    saved = _test_resume_configuration().to_dict()
    for name in (
        "terminal_empty_hand_weight",
        "terminal_blocked_weight",
        "immediate_draw_weight",
        "immediate_pass_weight",
    ):
        saved.pop(name)

    with pytest.raises(ValueError, match="before the reward redesign"):
        RLTrainingConfiguration.from_mapping(saved)


# --- Terminal-outcome diagnostics -------------------------------------------


def test_terminal_outcome_counts_one_ending_in_exactly_one_bucket():
    """Every game lands in exactly one of the four outcome counters."""
    empty_win = _terminal_outcome(
        _empty_hand_engine([(6, 6)], winner=0), 0, DEFAULT_REWARD_SCHEMA
    )[1]
    empty_loss = _terminal_outcome(
        _empty_hand_engine([(6, 6)], winner=0), 1, DEFAULT_REWARD_SCHEMA
    )[1]
    blocked_win = _terminal_outcome(
        _blocked_engine([[(1, 2)], [(4, 5)]], winner=0), 0, DEFAULT_REWARD_SCHEMA
    )[1]
    blocked_loss = _terminal_outcome(
        _blocked_engine([[(1, 2)], [(4, 5)]], winner=0), 1, DEFAULT_REWARD_SCHEMA
    )[1]

    assert (empty_win.empty_hand_wins, empty_win.blocked_wins) == (1, 0)
    assert (empty_loss.empty_hand_losses, empty_loss.blocked_losses) == (1, 0)
    assert (blocked_win.blocked_wins, blocked_win.empty_hand_wins) == (1, 0)
    assert (blocked_loss.blocked_losses, blocked_loss.empty_hand_losses) == (1, 0)
    # An empty-hand ending contributes nothing to the blocked margin totals,
    # so the mean margin is over blocked games alone.
    assert empty_win.blocked_margin_sum == 0
    assert empty_win.blocked_magnitude_sum == 0.0


def test_blocked_margin_totals_describe_the_game_not_the_seat():
    """Margin and magnitude are seat-independent; only the win/loss side flips."""
    engine = _blocked_engine([[(1, 2)], [(4, 5)]], winner=0)

    winner_view = _terminal_outcome(engine, 0, DEFAULT_REWARD_SCHEMA)[1]
    loser_view = _terminal_outcome(engine, 1, DEFAULT_REWARD_SCHEMA)[1]

    assert winner_view.blocked_margin_sum == loser_view.blocked_margin_sum == 6
    assert winner_view.blocked_magnitude_sum == pytest.approx(
        loser_view.blocked_magnitude_sum
    )
    assert winner_view.blocked_magnitude_sum == pytest.approx(0.55)


def test_terminal_stats_accumulate_across_games():
    total = TerminalStats()
    for engine, learner in (
        (_empty_hand_engine([(6, 6)], winner=0), 0),
        (_blocked_engine([[(1, 2)], [(4, 5)]], winner=0), 0),
        (_blocked_engine([[(1, 2)], [(0, 3)]], winner=0), 1),
    ):
        total.add(_terminal_outcome(engine, learner, DEFAULT_REWARD_SCHEMA)[1])

    assert total.empty_hand_wins == 1
    assert total.blocked_wins == 1
    assert total.blocked_losses == 1
    assert total.blocked_margin_sum == 6 + 0
    assert total.blocked_magnitude_sum == pytest.approx(0.55 + 0.1)


def test_reward_summary_reports_both_mixed_halves():
    """``terminal_abs_mean`` and ``local_abs_mean`` are E|(1-eta)G_T|, E|eta*G_I|.

    A nominal ``reward_eta`` of 0.5 does not imply the halves carry equal
    influence, so the plan asks every run to log the magnitudes rather than
    assume them equal. These are exactly the stored halves of each sample.
    """
    samples = [
        TrainingSample(
            x=None, action_index=0, legal_mask=None,
            policy_reward=0.7, raw_reward=0.7,
            terminal_reward=0.5, local_reward=0.2,
        ),
        TrainingSample(
            x=None, action_index=0, legal_mask=None,
            policy_reward=-0.5, raw_reward=-0.5,
            terminal_reward=-0.3, local_reward=-0.2,
        ),
    ]

    summary = _reward_signal_summary(samples)

    assert summary["terminal_abs_mean"] == pytest.approx(0.4)
    assert summary["local_abs_mean"] == pytest.approx(0.2)
    assert summary["terminal_mean"] == pytest.approx(0.1)
    assert summary["local_mean"] == pytest.approx(0.0)
