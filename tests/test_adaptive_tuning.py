"""Tests for isolated GPI/worker throughput selection."""

from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
from unittest import mock

import numpy as np

from agents.rl_nn import PolicyNetwork
from diagnostics.parallel_runner import ParallelSafetyConfig
from training.adaptive_tuning import (
    DEFAULT_GPI_BENCHMARK_WORKERS,
    DEFAULT_GPI_CANDIDATES,
    benchmark_gpi_candidates,
    benchmark_worker_candidates,
    capture_isolation_state,
    gpi_benchmark_iterations,
    hardware_warning,
    policy_sha256,
    restore_isolation_state,
    run_adaptive_tuning,
    select_fastest,
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


def test_gpi_benchmark_iteration_table_is_exact():
    expected = {100: 20, 200: 10, 400: 5, 600: 3, 800: 2, 1000: 2, 2000: 1}
    assert {gpi: gpi_benchmark_iterations(gpi) for gpi in expected} == expected


def test_gpi_benchmark_uses_ten_workers_warmup_and_exact_game_counts():
    runner = _Runner()
    with mock.patch(
        "training.adaptive_tuning._new_runner",
        return_value=runner,
    ):
        rows = benchmark_gpi_candidates(
            _FrozenNetwork(),
            candidates=DEFAULT_GPI_CANDIDATES,
            benchmark_games_target=2000,
            base_seed=42,
            training_opponent="self_play",
            schema=REWARD_SCHEMAS["default"],
            gamma=1.0,
            max_pool_size=2,
            safety=ParallelSafetyConfig(memory_reserve_mb=0),
        )

    assert runner.closed
    assert runner.calls[0][1] == 2  # discarded warm-up
    assert all(call[3] == DEFAULT_GPI_BENCHMARK_WORKERS for call in runner.calls)
    assert [row["benchmark_iterations"] for row in rows] == [20, 10, 5, 3, 2, 2, 1]
    assert [row["actual_games"] for row in rows] == [2000, 2000, 2000, 1800, 1600, 2000, 2000]
    assert all(row["success"] for row in rows)
    assert sum(call[1] for call in runner.calls[1:]) == sum(row["actual_games"] for row in rows)


def test_worker_benchmark_uses_exact_one_percent_with_partial_final_block():
    runner = _Runner()
    with mock.patch(
        "training.adaptive_tuning._new_runner",
        return_value=runner,
    ):
        test_games, rows = benchmark_worker_candidates(
            _FrozenNetwork(),
            selected_gpi=600,
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

    assert test_games == 1000
    assert [call[1] for call in runner.calls] == [600, 400]
    assert rows[0]["actual_games"] == 1000
    assert rows[0]["blocks"] == 2
    assert rows[0]["success"]


def test_tie_rules_prefer_smaller_gpi_or_fewer_workers():
    close = [
        {"success": True, "gpi": 100, "games_per_second": 98.0},
        {"success": True, "gpi": 200, "games_per_second": 100.0},
    ]
    clear = [
        {"success": True, "gpi": 100, "games_per_second": 96.0},
        {"success": True, "gpi": 200, "games_per_second": 100.0},
    ]
    workers = [
        {"success": True, "requested_workers": 1, "games_per_second": 99.0},
        {"success": True, "requested_workers": 2, "games_per_second": 100.0},
    ]

    assert select_fastest(close, key="gpi", tie_fraction=0.03)["gpi"] == 100
    assert select_fastest(clear, key="gpi", tie_fraction=0.03)["gpi"] == 200
    assert select_fastest(workers, key="requested_workers", tie_fraction=0.02)["requested_workers"] == 1


def test_capture_restore_recovers_weights_optimizer_rng_and_pool():
    network = _policy()
    pool = [_pool_snapshot(network)]
    random.seed(71)
    np.random.seed(72)
    snapshot = capture_isolation_state(network, pool)
    expected_python = snapshot["python_rng"]
    expected_numpy = snapshot["numpy_rng"]

    network.W1 += 5.0
    network.optimizer_step_count = 99
    pool[0]["W1"] += 7.0
    random.random()
    np.random.random()
    restore_isolation_state(network, snapshot, pool)

    assert policy_sha256(network) == snapshot["weights_sha256"]
    assert network.optimizer_state_dict() == snapshot["optimizer"]
    assert random.getstate() == expected_python
    assert _numpy_rng_equal(np.random.get_state(), expected_numpy)
    np.testing.assert_array_equal(pool[0]["W1"], snapshot["pool_snapshots"][0]["W1"])


def test_integrated_tuning_restores_state_and_writes_required_metadata():
    network = _policy(seed=8)
    pool = [_pool_snapshot(network)]
    initial_hash = policy_sha256(network)
    initial_optimizer = network.optimizer_state_dict()
    random.seed(81)
    np.random.seed(82)
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    def fake_gpi(policy, **kwargs):
        policy.W1 += 1.0
        policy.optimizer_step_count += 10
        kwargs["pool_snapshots"][0]["W1"] += 3.0
        random.random()
        np.random.random()
        return [
            {"success": True, "gpi": 100, "games_per_second": 10.0},
            {"success": True, "gpi": 200, "games_per_second": 20.0},
        ]

    def fake_workers(policy, **kwargs):
        return 1000, [
            {"success": True, "requested_workers": 1, "games_per_second": 10.0, "actual_games": 1000},
            {"success": True, "requested_workers": 2, "games_per_second": 12.0, "actual_games": 1000},
        ]

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "adaptive_tuning.json"
        with (
            mock.patch("training.adaptive_tuning.benchmark_gpi_candidates", side_effect=fake_gpi),
            mock.patch("training.adaptive_tuning.benchmark_worker_candidates", side_effect=fake_workers),
        ):
            metadata = run_adaptive_tuning(
                network,
                total_training_games=100_000,
                manual_gpi=100,
                adaptive_gpi=True,
                workers="auto",
                retune_gpi=False,
                retune_workers=False,
                saved_tuning=None,
                gpi_candidates=(100, 200),
                gpi_benchmark_games_target=2000,
                worker_benchmark_fraction=0.01,
                worker_candidates=(1, 2),
                base_seed=42,
                training_opponent="self_play",
                schema=REWARD_SCHEMAS["default"],
                gamma=1.0,
                max_pool_size=2,
                safety=ParallelSafetyConfig(memory_reserve_mb=0),
                pool_snapshots=pool,
                output_path=path,
            )
        saved = json.loads(path.read_text(encoding="utf-8"))

    assert metadata["selected_gpi"] == 200
    assert metadata["selected_workers"] == 2
    assert metadata["worker_test_games"] == 1000
    assert metadata["initial_weights_sha256"] == initial_hash
    assert metadata["isolation_verified"]
    assert saved["version"] == 2
    assert saved["gpi_benchmark_workers"] == DEFAULT_GPI_BENCHMARK_WORKERS
    assert saved["base_seed"] == 42
    assert saved["total_training_games"] == 100_000
    assert policy_sha256(network) == initial_hash
    assert network.optimizer_state_dict() == initial_optimizer
    assert random.getstate() == python_state
    assert _numpy_rng_equal(np.random.get_state(), numpy_state)


def test_saved_tuning_is_reused_without_new_benchmarks():
    network = _policy(seed=9)
    saved = {
        "selected_gpi": 800,
        "gpi_results": [{"gpi": 800, "success": True}],
        "selected_workers": 6,
        "worker_results": [{"requested_workers": 6, "success": True}],
        "worker_test_games": 1000,
    }
    with (
        mock.patch("training.adaptive_tuning.benchmark_gpi_candidates") as gpi,
        mock.patch("training.adaptive_tuning.benchmark_worker_candidates") as workers,
    ):
        result = run_adaptive_tuning(
            network,
            total_training_games=100_000,
            manual_gpi=100,
            adaptive_gpi=True,
            workers="auto",
            retune_gpi=False,
            retune_workers=False,
            saved_tuning=saved,
            gpi_candidates=DEFAULT_GPI_CANDIDATES,
            gpi_benchmark_games_target=2000,
            worker_benchmark_fraction=0.01,
            worker_candidates=(1, 2, 4, 6),
            base_seed=42,
            training_opponent="self_play",
            schema=REWARD_SCHEMAS["default"],
            gamma=1.0,
            max_pool_size=2,
            safety=ParallelSafetyConfig(memory_reserve_mb=0),
        )

    gpi.assert_not_called()
    workers.assert_not_called()
    assert result["selected_gpi"] == 800
    assert result["selected_workers"] == 6
    assert result["gpi_source"] == "resume"
    assert result["worker_source"] == "resume"


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
