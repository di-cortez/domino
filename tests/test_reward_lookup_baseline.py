"""Fixed-artifact and numerical tests for the lookup-table RL baseline."""

from types import SimpleNamespace

import numpy as np
import pytest

from training.rl.baseline import BaselineSpec
from training.rl.ppo import PPOBuffer
from training.rl.reward_lookup_tables import load_reward_lookup
from training.rl.reward_lookup_tables.lookup import RewardLookupTable


RULESETS = ("double-three", "double-four", "double-five", "double-six")


def _schema(*, mode="turn-decision"):
    return {
        "terminal_win": 1.0,
        "terminal_loss": -1.0,
        "final_pip_penalty": 0.05,
        "opponent_pass": 0.1,
        "learner_pass": -0.1,
        "opponent_draw": 0.2,
        "learner_draw": -0.2,
        "gamma_f": 0.5,
        "gamma_i": 0.9,
        "reward_eta": 0.25,
        "reward_distance_mode": mode,
    }


def _synthetic_lookup():
    tables = {
        component: {clock: {} for clock in ("turn", "decision")}
        for component in ("final", "pips", "pass", "draw")
    }
    tables["final"]["turn"]["2,3"] = [0.0, 1.0]
    tables["final"]["decision"]["2,3"] = [1.0]
    tables["pips"]["turn"]["2,3"] = [0.0, 2.0]
    tables["pips"]["decision"]["2,3"] = [2.0]
    tables["pass"]["turn"]["2,3"] = [1.0]
    tables["pass"]["decision"]["2,3"] = [0.0, 1.0]
    tables["draw"]["turn"]["2,3"] = [-1.0]
    tables["draw"]["decision"]["2,3"] = [-1.0]
    return RewardLookupTable(
        {"ruleset_name": "double-six", "tables": tables},
        artifact_digest="0" * 64,
    )


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
def test_every_packaged_ruleset_lookup_is_valid_and_identified(ruleset):
    lookup = load_reward_lookup(ruleset)

    assert lookup.ruleset_name == ruleset
    assert len(lookup.artifact_sha256) == 64
    assert lookup.resolve_cell(1, 99) is None
    assert lookup.resolve_cell(2, 99) == "2,4"
    assert lookup.resolve_cell(99, 1) == "5,1"
    assert lookup.resolve_cell(99, 2) == "6,2"


def test_general_missing_cell_fallback_moves_diagonally_until_stored():
    lookup = load_reward_lookup("double-six")

    assert lookup.resolve_cell(10, 10) == "6,6"


def test_unit_components_use_runtime_clocks_magnitudes_and_eta():
    lookup = _synthetic_lookup()
    sample = _sample(2, 3)

    # turn-decision means local events use turn while terminal final/pips use
    # decision: 0.75 * (1 - 0.05*2) + 0.25 * (0.1 - 0.2) = 0.65.
    assert lookup.baseline_values([sample], _schema())[0] == pytest.approx(0.65)
    # decision-turn swaps the clocks: terminal = 0.5 - 0.05*1 and
    # local = 0.1*0.9 - 0.2, followed by the same eta mixture.
    assert lookup.baseline_values(
        [sample],
        _schema(mode="decision-turn"),
    )[0] == pytest.approx(0.31)


def test_one_tile_structural_baseline_is_immediate_win_only():
    lookup = load_reward_lookup("double-three")

    assert lookup.baseline_values([_sample(1, 200)], _schema())[0] == pytest.approx(
        0.75
    )


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
        load_reward_lookup("double-three").baseline_values([sample], _schema())
