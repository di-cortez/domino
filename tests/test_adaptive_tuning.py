"""Tests for isolated rollout-worker throughput selection."""

from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
from unittest import mock

import numpy as np
import pytest

from agents.rl_nn import PolicyNetwork
from diagnostics.parallel_runner import ParallelSafetyConfig
from training.adaptive_tuning import (
    benchmark_worker_candidates,
    capture_isolation_state,
    hardware_warning,
    policy_sha256,
    restore_isolation_state,
    run_worker_tuning,
    selected_worker_candidate,
)
from training.self_play import REWARD_SCHEMAS


class _RunInfo:
    def __init__(self, workers=1):
        self.requested_workers = workers
        self.initial_workers = workers
        self.final_workers = workers
        self.peak_worker_rss_mb = 0.0
        self.peak_total_children_rss_mb = 0.0
        self.min_available_memory_mb = 1024.0
        self.fallback_count = 0
        self.fallback_history = []
        self.attempted_worker_counts = [workers]
        self.safety_capped = False
        self.memory_monitoring_available = True
        self.workers_cpu_only = True

    def to_dict(self):
        return dict(self.__dict__)


class _Runner:
    def __init__(self):
        self.worker_count = 1
        self.calls = []
        self.closed = False

    def set_workers(self, workers):
        self.worker_count = int(workers)
        return self.worker_count, False, None

    def collect_games(self, first_game, game_count, seed):
        self.calls.append((int(first_game), int(game_count), int(seed), self.worker_count))
        results = [
            {"game_index": int(first_game) + index, "samples": []}
            for index in range(int(game_count))
        ]
        return results, _RunInfo(self.worker_count)

    def close(self):
        self.closed = True


class _FrozenNetwork:
    device = "cpu"

    @staticmethod
    def synchronize():
        return None


def _policy(seed=1):
    return PolicyNetwork(
        input_size=3,
        hidden1_size=4,
        hidden2_size=3,
        output_size=4,
        learning_rate=0.001,
        random_seed=seed,
        device="cpu",
    )


def _pool_snapshot(network):
    return {
        name: np.asarray(getattr(network, name)).copy()
        for name in ("W1", "b1", "W2", "b2", "W3", "b3")
    }


def _numpy_rng_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _tuning_kwargs(network, **overrides):
    values = {
        "gpi": 600,
        "total_training_games": 100_000,
        "workers": "auto",
        "retune_workers": False,
        "saved_tuning": None,
        "worker_benchmark_fraction": 0.01,
        "worker_minimum_gain": 0.10,
        "worker_candidates": (1, 2),
        "base_seed": 42,
        "training_opponent": "self_play",
        "schema": REWARD_SCHEMAS["default"],
        "gamma": 1.0,
        "max_pool_size": 2,
        "safety": ParallelSafetyConfig(memory_reserve_mb=0),
    }
    values.update(overrides)
    return run_worker_tuning(network, **values)


def test_worker_benchmark_uses_exact_one_percent_with_partial_final_block():
    runner = _Runner()
    with mock.patch("training.adaptive_tuning._new_runner", return_value=runner):
        test_games, rows = benchmark_worker_candidates(
            _FrozenNetwork(),
            gpi=600,
            total_training_games=100_000,
            benchmark_fraction=0.01,
            candidates=(1,),
            base_seed=42,
            training_opponent="self_play",
            schema=REWARD_SCHEMAS["default"],
            gamma=1.0,
            max_pool_size=2,
            safety=ParallelSafetyConfig(memory_reserve_mb=0),
        )

    assert runner.closed
    assert test_games == 1000
    assert [call[1] for call in runner.calls] == [600, 400]
    assert rows[0]["actual_games"] == 1000
    assert rows[0]["blocks"] == 2
    assert rows[0]["success"]
    assert rows[0]["accepted"]


def test_worker_benchmark_stops_below_ten_percent_and_keeps_previous():
    runner = _Runner()
    messages = []
    with (
        mock.patch("training.adaptive_tuning._new_runner", return_value=runner),
        mock.patch(
            "training.adaptive_tuning.time.perf_counter",
            side_effect=(0.0, 1.0, 2.0, 2.8, 4.0, 4.75),
        ),
    ):
        test_games, rows = benchmark_worker_candidates(
            _FrozenNetwork(),
            gpi=100,
            total_training_games=10_000,
            benchmark_fraction=0.01,
            minimum_gain=0.10,
            candidates=(1, 2, 4, 6),
            base_seed=42,
            training_opponent="self_play",
            schema=REWARD_SCHEMAS["default"],
            gamma=1.0,
            max_pool_size=2,
            safety=ParallelSafetyConfig(memory_reserve_mb=0),
            status_callback=messages.append,
        )

    assert test_games == 100
    assert [row["requested_workers"] for row in rows] == [1, 2, 4]
    assert [row["accepted"] for row in rows] == [True, True, False]
    assert rows[1]["improvement_over_previous"] == pytest.approx(0.25)
    assert rows[2]["improvement_over_previous"] == pytest.approx(1 / 15)
    assert selected_worker_candidate(rows)["requested_workers"] == 2
    assert any("below 10%" in message for message in messages)


def test_capture_restore_recovers_weights_optimizer_rng_and_pool():
    network = _policy()
    pool = [_pool_snapshot(network)]
    random.seed(71)
    np.random.seed(72)
    snapshot = capture_isolation_state(network, pool)

    network.W1 += 5.0
    network.optimizer_step_count = 99
    pool[0]["W1"] += 7.0
    random.random()
    np.random.random()
    restore_isolation_state(network, snapshot, pool)

    assert policy_sha256(network) == snapshot["weights_sha256"]
    assert network.optimizer_state_dict() == snapshot["optimizer"]
    assert random.getstate() == snapshot["python_rng"]
    assert _numpy_rng_equal(np.random.get_state(), snapshot["numpy_rng"])
    np.testing.assert_array_equal(pool[0]["W1"], snapshot["pool_snapshots"][0]["W1"])


def test_worker_tuning_restores_state_and_writes_fixed_gpi_metadata():
    network = _policy(seed=8)
    pool = [_pool_snapshot(network)]
    initial_hash = policy_sha256(network)
    initial_optimizer = network.optimizer_state_dict()
    random.seed(81)
    np.random.seed(82)
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    def fake_workers(policy, **kwargs):
        policy.W1 += 1.0
        policy.optimizer_step_count += 10
        kwargs["pool_snapshots"][0]["W1"] += 3.0
        random.random()
        np.random.random()
        return 1000, [
            {
                "success": True,
                "accepted": True,
                "requested_workers": 2,
                "games_per_second": 12.0,
                "actual_games": 1000,
            }
        ]

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "adaptive_tuning.json"
        with mock.patch(
            "training.adaptive_tuning.benchmark_worker_candidates",
            side_effect=fake_workers,
        ):
            metadata = _tuning_kwargs(
                network,
                pool_snapshots=pool,
                output_path=path,
            )
        saved = json.loads(path.read_text(encoding="utf-8"))

    assert metadata["gpi"] == 600
    assert metadata["selected_workers"] == 2
    assert metadata["worker_test_games"] == 1000
    assert saved["version"] == 4
    assert saved["worker_minimum_gain"] == 0.10
    assert saved["initial_weights_sha256"] == initial_hash
    assert saved["isolation_verified"]
    assert policy_sha256(network) == initial_hash
    assert network.optimizer_state_dict() == initial_optimizer
    assert random.getstate() == python_state
    assert _numpy_rng_equal(np.random.get_state(), numpy_state)


def test_saved_worker_tuning_is_reused_for_the_same_fixed_gpi():
    network = _policy(seed=9)
    saved = {
        "gpi": 800,
        "selected_workers": 6,
        "worker_results": [{"requested_workers": 6, "success": True}],
        "worker_test_games": 1000,
    }
    with mock.patch(
        "training.adaptive_tuning.benchmark_worker_candidates"
    ) as benchmark:
        result = _tuning_kwargs(network, gpi=800, saved_tuning=saved)

    benchmark.assert_not_called()
    assert result["gpi"] == 800
    assert result["selected_workers"] == 6
    assert result["worker_source"] == "resume"


def test_changing_fixed_gpi_repeats_worker_tuning():
    network = _policy(seed=10)
    saved = {"gpi": 100, "selected_workers": 1}
    rows = [
        {
            "success": True,
            "accepted": True,
            "requested_workers": 2,
            "games_per_second": 12.0,
        }
    ]
    with mock.patch(
        "training.adaptive_tuning.benchmark_worker_candidates",
        return_value=(1000, rows),
    ) as benchmark:
        result = _tuning_kwargs(network, gpi=200, saved_tuning=saved)

    benchmark.assert_called_once()
    assert result["gpi"] == 200
    assert result["selected_workers"] == 2
    assert result["worker_source"] == "autotune"


def test_hardware_change_warning_names_changed_fields():
    warning = hardware_warning(
        {"device": "gpu", "gpu_name": "old", "cpu_count": 8},
        {"device": "gpu", "gpu_name": "new", "cpu_count": 12},
    )
    assert "gpu_name" in warning
    assert "cpu_count" in warning
    assert hardware_warning(
        {"device": "cpu", "gpu_name": None, "cpu_count": 8},
        {"device": "cpu", "gpu_name": None, "cpu_count": 8},
    ) is None
