"""Tests for deterministic, order-independent stream derivation."""

import json

import numpy as np
import pytest

from utils.myrandom import (
    DERIVATION_SCHEME,
    RandomNamespace,
    SeedPlan,
    supported_bit_generators,
)


def _sample(plan, namespace, *coordinates):
    generator = plan.generator(namespace, *coordinates)
    return generator.integers(0, 2**32, size=16, dtype=np.uint32)


def _numpy_global_states_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_same_identity_recreates_the_same_stream():
    plan = SeedPlan(42)
    first = _sample(plan, RandomNamespace.RL_GAME, 123)
    second = _sample(plan, RandomNamespace.RL_GAME, 123)
    assert np.array_equal(first, second)


def test_stream_creation_order_does_not_change_results():
    plan = SeedPlan(42)
    expected = {
        game_id: _sample(plan, RandomNamespace.RL_GAME, game_id)
        for game_id in range(8)
    }
    observed = {
        game_id: _sample(plan, RandomNamespace.RL_GAME, game_id)
        for game_id in reversed(range(8))
    }
    for game_id, values in expected.items():
        assert np.array_equal(values, observed[game_id])


def test_game_streams_do_not_depend_on_simulated_worker_assignment():
    plan = SeedPlan(42)

    def records(worker_count):
        game_ids_by_worker = [list(range(worker, 40, worker_count))
                              for worker in range(worker_count)]
        return {
            game_id: plan.generator(
                RandomNamespace.RL_GAME,
                game_id,
            ).integers(0, 2**32, size=4, dtype=np.uint32)
            for worker_games in reversed(game_ids_by_worker)
            for game_id in reversed(worker_games)
        }

    one_worker = records(1)
    seven_workers = records(7)
    assert one_worker.keys() == seven_workers.keys()
    for game_id, values in one_worker.items():
        assert np.array_equal(values, seven_workers[game_id])


def test_derivation_contract_has_a_fixed_reference_sequence():
    plan = SeedPlan(42)
    assert plan.uint64_seed(RandomNamespace.RL_GAME, 123) == 15843327384582174627
    generator = plan.generator(RandomNamespace.RL_GAME, 123)
    assert generator.integers(0, 2**32, size=8, dtype=np.uint32).tolist() == [
        519425528,
        2049021683,
        4286558435,
        283444699,
        1194524657,
        76541169,
        3324867694,
        1988051691,
    ]


def test_namespace_root_and_coordinates_separate_streams():
    baseline = _sample(SeedPlan(42), RandomNamespace.RL_GAME, 7)
    comparisons = (
        _sample(SeedPlan(43), RandomNamespace.RL_GAME, 7),
        _sample(SeedPlan(42), RandomNamespace.DATASET_GAME, 7),
        _sample(SeedPlan(42), RandomNamespace.RL_GAME, 8),
        _sample(SeedPlan(42), RandomNamespace.RL_GAME, "7"),
    )
    assert all(not np.array_equal(baseline, item) for item in comparisons)


def test_derivation_does_not_consume_numpy_global_state():
    np.random.seed(20260808)
    before = np.random.get_state()
    plan = SeedPlan(42)
    for index in range(20):
        plan.generator(RandomNamespace.DIAGNOSTIC_GAME, "rl-vs-random", index).random(64)
    after = np.random.get_state()
    assert _numpy_global_states_equal(before, after)


def test_uint64_seed_is_deterministic_and_bounded():
    plan = SeedPlan(42)
    first = plan.uint64_seed(RandomNamespace.PPO_MINIBATCH, 10, 2)
    second = plan.uint64_seed(RandomNamespace.PPO_MINIBATCH, 10, 2)
    assert first == second
    assert 0 <= first < 2**64


def test_manifest_round_trip_and_atomic_file_write(tmp_path):
    plan = SeedPlan(2**100 + 17)
    path = plan.write_manifest(tmp_path / "nested" / "random_manifest.json")
    loaded = SeedPlan.from_manifest_file(path)
    assert loaded == plan
    assert json.loads(path.read_text(encoding="utf-8")) == plan.to_manifest()


def test_manifest_describes_the_complete_derivation_contract():
    manifest = SeedPlan(42).to_manifest()
    assert manifest["schema_version"] == 1
    assert manifest["derivation_scheme"] == DERIVATION_SCHEME
    assert manifest["bit_generator"] == "PCG64"
    assert manifest["registered_namespaces"] == [
        item.value for item in RandomNamespace
    ]


@pytest.mark.parametrize("root_seed", [True, 1.5, "42"])
def test_seed_plan_rejects_non_integer_root_seed(root_seed):
    with pytest.raises(TypeError):
        SeedPlan(root_seed)


def test_seed_plan_rejects_negative_root_seed():
    with pytest.raises(ValueError):
        SeedPlan(-1)


@pytest.mark.parametrize("namespace", ["rl.game", None, 1])
def test_seed_plan_requires_registered_namespace(namespace):
    with pytest.raises(TypeError):
        SeedPlan(42).generator(namespace, 1)


@pytest.mark.parametrize(
    ("coordinate", "expected_error"),
    [
        (True, TypeError),
        (-1, ValueError),
        ("", ValueError),
        (1.5, TypeError),
        (None, TypeError),
    ],
)
def test_seed_plan_rejects_ambiguous_coordinates(coordinate, expected_error):
    with pytest.raises(expected_error):
        SeedPlan(42).generator(RandomNamespace.RL_GAME, coordinate)


def test_only_the_declared_bit_generator_is_supported():
    assert supported_bit_generators() == ("PCG64",)
    with pytest.raises(ValueError):
        SeedPlan(42, bit_generator="MT19937")


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 99),
        ("derivation_scheme", "other"),
        ("registered_namespaces", []),
    ],
)
def test_manifest_rejects_changed_contract_fields(field, value):
    manifest = SeedPlan(42).to_manifest()
    manifest[field] = value
    with pytest.raises(ValueError):
        SeedPlan.from_manifest(manifest)
