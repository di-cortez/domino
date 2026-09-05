"""Contracts for the mergeable distribution summaries in the periodic record."""

from __future__ import annotations

import math

import numpy as np

from training.rl.statistics import RunningMoments


def _reference(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def test_single_process_summary_matches_numpy():
    generator = np.random.default_rng(20260904)
    values = generator.normal(0.12, 0.46, size=5_000)
    summary = RunningMoments.from_values(values)
    expected = _reference(values)
    assert summary.count == 5_000
    assert math.isclose(summary.mean, expected["mean"], rel_tol=1e-12)
    assert math.isclose(summary.std, expected["std"], rel_tol=1e-9)
    assert summary.minimum == expected["min"]
    assert summary.maximum == expected["max"]


def test_merge_of_worker_shards_equals_the_single_process_summary():
    """The parallel and serial rollouts must produce the same summary."""
    generator = np.random.default_rng(4242)
    values = generator.normal(-0.3, 1.1, size=9_973)
    # Deliberately uneven shards, including an empty one: a worker that drew no
    # games of a given kind still reports its summary.
    bounds = [0, 1, 1, 500, 4_000, 9_973]
    shards = [values[low:high] for low, high in zip(bounds, bounds[1:])]
    assert sum(len(shard) for shard in shards) == len(values)

    merged = RunningMoments()
    for shard in shards:
        merged.merge(RunningMoments.from_values(shard))

    whole = RunningMoments.from_values(values)
    assert merged.count == whole.count
    assert math.isclose(merged.mean, whole.mean, rel_tol=1e-12)
    assert math.isclose(merged.std, whole.std, rel_tol=1e-9)
    assert merged.minimum == whole.minimum
    assert merged.maximum == whole.maximum


def test_merge_survives_the_transport_round_trip():
    generator = np.random.default_rng(7)
    values = generator.normal(size=1_000)
    original = RunningMoments.from_values(values)
    restored = RunningMoments.from_list(original.to_list())
    assert restored == original


def test_an_empty_population_reports_none_not_zero():
    """Zero is a legitimate value for every one of these statistics."""
    summary = RunningMoments().as_dict("R_B")
    assert summary == {
        "R_B_mean": None,
        "R_B_max": None,
        "R_B_min": None,
        "R_B_std": None,
    }


def test_merging_an_empty_summary_is_a_no_op():
    summary = RunningMoments.from_values([1.0, -1.0])
    before = summary.to_list()
    summary.merge(RunningMoments())
    assert summary.to_list() == before


def test_a_constant_population_has_exactly_zero_deviation():
    """Cancellation must not turn a zero variance into a NaN."""
    summary = RunningMoments.from_values([0.25] * 1_000)
    assert summary.std == 0.0
    assert summary.as_dict("G_D")["G_D_std"] == 0.0


def test_recorded_values_are_rounded_to_six_significant_digits():
    summary = RunningMoments.from_values([1.0 / 3.0])
    recorded = summary.as_dict("G_P")
    assert recorded["G_P_mean"] == 0.333333
    assert recorded["G_P_max"] == 0.333333


def test_a_small_statistic_keeps_its_digits():
    """A PPO trust-region value must survive rounding, not round to zero."""
    summary = RunningMoments.from_values([0.00035123456])
    assert summary.as_dict("max_kl")["max_kl_mean"] == 0.000351235


def test_an_empty_summary_transports_as_json_compliant_nulls():
    """The infinite identity elements must never reach the JSON encoder."""
    import json

    encoded = json.dumps(RunningMoments().to_list())
    assert encoded == "[0, 0.0, 0.0, null, null]"
    assert RunningMoments.from_list(json.loads(encoded)) == RunningMoments()


def test_an_empty_summary_still_merges_as_a_neutral_element_after_transport():
    restored = RunningMoments.from_list(RunningMoments().to_list())
    populated = RunningMoments.from_values([2.0, 4.0])
    populated.merge(restored)
    assert populated.minimum == 2.0
    assert populated.maximum == 4.0
    assert populated.count == 2
