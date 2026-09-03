"""Fixed-artifact and numerical tests for the lookup-table RL baseline."""

from types import SimpleNamespace

import numpy as np
import pytest

from reward_lookup_artifact_state import (
    PACKAGED_RULESETS,
    stale_artifact_reason,
    stale_rulesets,
)
from training.rl.baseline import BaselineSpec
from training.rl.ppo import PPOBuffer
from training.rl.reward_lookup_tables import load_reward_lookup
from training.rl.reward_lookup_tables.lookup import (
    CLOCKS,
    COMPONENTS,
    RewardLookupTable,
)


RULESETS = PACKAGED_RULESETS
# Cells the documented ad hoc fallback ladder must be able to land on.
ANCHOR_CELLS = ("2,3", "2,4", "5,1", "6,2", "6,6")

# Rebuilding is per ruleset, so staleness is checked per ruleset too: a ruleset
# whose artifact has been rebuilt gets its real coverage back immediately
# instead of waiting for the other three.
_STALE_RULESETS = stale_rulesets()


def _skip_if_stale(ruleset):
    if ruleset in _STALE_RULESETS:
        pytest.skip(f"{ruleset}: {stale_artifact_reason()}")


def _schema(*, mode="turn-decision"):
    """Return the reward fields ``resolve_training_options`` puts on a run."""
    return {
        "terminal_empty_hand_weight": 1.0,
        "terminal_blocked_weight": 1.0,
        "empty_hand_scale": 1.0,
        "blocked_scale": 1.0,
        "immediate_draw_weight": 1.0,
        "immediate_pass_weight": 0.5,
        "draw_scale": 1.0,
        "pass_scale": 0.5,
        "gamma_f": 0.5,
        "gamma_i": 0.9,
        "reward_eta": 0.25,
        "reward_distance_mode": mode,
    }


def _table(**cells):
    """Build one lookup whose cells are given as component/clock histograms."""
    tables = {
        component: {clock: {} for clock in CLOCKS} for component in COMPONENTS
    }
    for cell, histograms in cells.items():
        for component in COMPONENTS:
            for clock in CLOCKS:
                tables[component][clock][cell] = list(
                    histograms.get(component, {}).get(clock, [])
                )
    return RewardLookupTable(
        {"ruleset_name": "double-six", "tables": tables},
        artifact_digest="0" * 64,
    )


def _synthetic_lookup():
    return _table(**{
        "2,3": {
            "empty_hand": {"turn": [0.0, 1.0], "decision": [1.0]},
            "blocked": {"turn": [0.0, -0.5], "decision": [-0.5]},
            "pass": {"turn": [1.0], "decision": [0.0, 1.0]},
            "draw": {"turn": [-1.0], "decision": [-1.0]},
        },
    })


def _anchor_lookup():
    return _table(**{cell: {} for cell in ANCHOR_CELLS})


def _sample(agent_size, opponent_size, reward=1.0):
    legal_mask = np.asarray([[True], [True]], dtype=np.bool_)
    return SimpleNamespace(
        x=np.zeros((3, 1), dtype=np.float32),
        action_index=0,
        legal_mask=legal_mask,
        old_log_prob=0.0,
        policy_reward=float(reward),
        local_reward=0.0,
        terminal_reward=float(reward),
        agent_hand_size=agent_size,
        opponent_hand_size=opponent_size,
    )


@pytest.mark.parametrize("ruleset", RULESETS)
def test_packaged_lookups_are_refused_until_they_are_rebuilt(ruleset):
    """A superseded artifact must fail loudly rather than be reinterpreted.

    Its ``final`` column mixes empty-hand and blocked endings into one signed
    outcome, so reading it under the new formula would quietly answer with a
    baseline for a reward the run is not training on.
    """
    if ruleset not in _STALE_RULESETS:
        pytest.skip(f"{ruleset} has already been rebuilt")

    with pytest.raises(ValueError, match="format version"):
        load_reward_lookup(ruleset)


@pytest.mark.parametrize("ruleset", RULESETS)
def test_every_packaged_ruleset_lookup_is_valid_and_identified(ruleset):
    _skip_if_stale(ruleset)
    lookup = load_reward_lookup(ruleset)

    assert lookup.ruleset_name == ruleset
    assert len(lookup.artifact_sha256) == 64
    assert lookup.resolve_cell(1, 99) is None
    assert lookup.resolve_cell(2, 99) == "2,4"
    assert lookup.resolve_cell(99, 1) == "5,1"
    assert lookup.resolve_cell(99, 2) == "6,2"


def test_documented_fallback_ladder_lands_on_the_stored_anchors():
    lookup = _anchor_lookup()

    assert lookup.resolve_cell(1, 99) is None
    assert lookup.resolve_cell(2, 99) == "2,4"
    assert lookup.resolve_cell(99, 1) == "5,1"
    assert lookup.resolve_cell(99, 2) == "6,2"


def test_general_missing_cell_fallback_moves_diagonally_until_stored():
    assert _anchor_lookup().resolve_cell(10, 10) == "6,6"


def test_unit_components_use_runtime_clocks_scales_and_eta():
    lookup = _synthetic_lookup()
    sample = _sample(2, 3)

    # turn-decision means local events use the turn clock while the terminal
    # components use the decision clock:
    #   terminal = 1.0 * 1.0 + 1.0 * (-0.5) = 0.5
    #   local    = 0.5 * 1.0 + 1.0 * (-1.0) = -0.5
    #   0.75 * 0.5 + 0.25 * (-0.5) = 0.25
    assert lookup.baseline_values([sample], _schema())[0] == pytest.approx(0.25)
    # decision-turn swaps both clocks:
    #   terminal = 1.0 * (1.0 * 0.5) + 1.0 * (-0.5 * 0.5) = 0.25
    #   local    = 0.5 * (1.0 * 0.9) + 1.0 * (-1.0) = -0.55
    #   0.75 * 0.25 + 0.25 * (-0.55) = 0.05
    assert lookup.baseline_values(
        [sample],
        _schema(mode="decision-turn"),
    )[0] == pytest.approx(0.05)


def test_terminal_weights_reweight_the_two_endings_independently():
    lookup = _synthetic_lookup()
    sample = _sample(2, 3)
    schema = _schema()
    # Halving the blocked weight renormalizes the pair to (1.0, 0.5), which
    # must move only the blocked half of the terminal estimate.
    schema["terminal_blocked_weight"] = 0.5
    schema["blocked_scale"] = 0.5

    # terminal = 1.0 * 1.0 + 0.5 * (-0.5) = 0.75, local unchanged at -0.5.
    assert lookup.baseline_values([sample], schema)[0] == pytest.approx(
        0.75 * 0.75 + 0.25 * -0.5
    )


def test_one_tile_structural_baseline_is_immediate_empty_hand_win_only():
    lookup = _anchor_lookup()

    # The decision empties the hand, so only ``empty_hand`` is credited and the
    # local half contributes nothing.
    assert lookup.baseline_values([_sample(1, 200)], _schema())[0] == pytest.approx(
        0.75
    )


def test_lookup_rejects_a_reward_pair_that_was_zeroed_out():
    schema = _schema()
    schema["terminal_empty_hand_weight"] = 0.0
    schema["terminal_blocked_weight"] = 0.0

    with pytest.raises(ValueError, match="cannot both be zero"):
        _synthetic_lookup().baseline_values([_sample(2, 3)], schema)


def test_ppo_buffer_subtracts_one_lookup_value_per_decision():
    samples = [_sample(1, 2, 1.0), _sample(2, 3, -1.0)]
    values = np.asarray([0.75, -0.25], dtype=np.float32)
    buffer = PPOBuffer.from_samples(
        samples,
        baseline=BaselineSpec(kind="lookup-table"),
        lookup_values=values,
        normalize=False,
    )

    assert np.allclose(buffer.advantages, [0.25, -0.75])
    assert buffer.baseline_mean == pytest.approx(0.25)


def test_lookup_baseline_rejects_samples_without_hand_sizes():
    sample = _sample(2, 3)
    del sample.agent_hand_size

    with pytest.raises(ValueError, match="needs hand sizes"):
        _anchor_lookup().baseline_values([sample], _schema())
