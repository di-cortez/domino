"""Tests for exact JSON-safe generator state continuation."""

import json

import numpy as np
import pytest

from utils.myrandom import (
    RandomNamespace,
    SeedPlan,
    generator_from_state,
    generator_state,
    read_generator_state,
    restore_generator_state,
    restore_generators,
    snapshot_generators,
    write_generator_state,
)


def test_snapshot_recreates_exact_continuation():
    generator = SeedPlan(42).generator(RandomNamespace.RL_DROPOUT, 9)
    generator.random(100)
    snapshot = generator_state(generator)
    expected = generator.standard_normal(200)
    restored = generator_from_state(snapshot)
    assert np.array_equal(restored.standard_normal(200), expected)


def test_snapshot_is_detached_from_later_generator_updates():
    generator = SeedPlan(42).generator(RandomNamespace.RL_POLICY, 3)
    snapshot = generator_state(generator)
    original_json = json.dumps(snapshot, sort_keys=True)
    generator.random(1000)
    assert json.dumps(snapshot, sort_keys=True) == original_json


def test_restore_generator_state_rewinds_in_place():
    generator = SeedPlan(42).generator(RandomNamespace.SUPERVISED_DROPOUT, 2)
    snapshot = generator_state(generator)
    expected = generator.random((8, 16))
    restore_generator_state(generator, snapshot)
    assert np.array_equal(generator.random((8, 16)), expected)


def test_state_file_round_trip_is_json_and_exact(tmp_path):
    generator = SeedPlan(42).generator(RandomNamespace.PPO_MINIBATCH, 17)
    generator.permutation(100)
    path = write_generator_state(tmp_path / "state" / "ppo_rng.json", generator)
    payload = read_generator_state(path)
    restored = generator_from_state(payload)
    expected = generator.integers(0, 1_000_000, size=100)
    assert np.array_equal(restored.integers(0, 1_000_000, size=100), expected)
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_named_generator_collection_round_trip():
    generators = {
        "policy": SeedPlan(42).generator(RandomNamespace.RL_POLICY, 1),
        "dropout": SeedPlan(42).generator(RandomNamespace.RL_DROPOUT, 1),
    }
    for generator in generators.values():
        generator.random(50)
    payload = snapshot_generators(generators)
    restored = restore_generators(payload)
    for name, generator in generators.items():
        assert np.array_equal(restored[name].random(100), generator.random(100))


@pytest.mark.parametrize("payload", [None, [], "state"])
def test_generator_state_rejects_non_dictionary_payload(payload):
    with pytest.raises(TypeError):
        generator_from_state(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"schema_version": 99, "bit_generator": "PCG64", "state": {}},
        {"schema_version": 1, "bit_generator": "PCG64"},
    ],
)
def test_generator_state_rejects_incomplete_or_unknown_payload(payload):
    with pytest.raises(ValueError):
        generator_from_state(payload)


def test_named_snapshot_requires_dictionary_and_valid_names():
    with pytest.raises(TypeError):
        snapshot_generators([])
    generator = SeedPlan(42).generator(RandomNamespace.RL_GAME, 1)
    with pytest.raises(ValueError):
        snapshot_generators({"": generator})


def test_named_restore_rejects_invalid_collection():
    with pytest.raises(TypeError):
        restore_generators([])
    with pytest.raises(ValueError):
        restore_generators({"schema_version": 99, "generators": {}})
    with pytest.raises(ValueError):
        restore_generators({"schema_version": 1})
