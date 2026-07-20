"""Correctness and safety tests for parallel reinforcement-learning rollouts."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.rl_nn import PolicyNetwork
from diagnostics.parallel_runner import (
    MAX_PARALLEL_WORKERS,
    DiagnosticMemoryPressure,
    ParallelSafetyConfig,
)
from training.rl_parallel import RLRolloutRunner, worker_count
from training.self_play import (
    REWARD_SCHEMAS,
    load_resume_state,
    numbered_checkpoint_path,
    resume_state_path,
    train,
)
from run_pipeline import parse_args as parse_pipeline_args
from utils.resource_limits import MemorySafetyError


def _rollout_fingerprint(results):
    """Return a comparison-safe representation of arrays and scalar metadata."""
    rows = []
    for result in results:
        samples = []
        for sample in result["samples"]:
            samples.append((
                np.asarray(sample.x).tobytes(),
                sample.action_index,
                np.asarray(sample.legal_mask).tobytes(),
                sample.policy_reward,
                sample.raw_reward,
                sample.local_reward,
                sample.terminal_reward,
                sample.multiplier,
                sample.option_count,
            ))
        rows.append((
            result["game_index"],
            result["game_seed"],
            tuple(samples),
            tuple(sorted(result["event_stats"].items())),
            result["winner"],
            result["learner_position"],
        ))
    return rows


class ParallelRLTests(unittest.TestCase):
    def setUp(self):
        self.safety = ParallelSafetyConfig(
            memory_reserve_mb=0,
            estimated_worker_mb=1,
            max_worker_rss_mb=1024,
        )

    def _network(self):
        return PolicyNetwork.load_from_sl(
            ROOT / "models" / "domino_sl_weights.npz",
            device="cpu",
        )

    def _collect(self, workers, game_count=10, seed=1234):
        network = self._network()
        runner = RLRolloutRunner(
            network,
            training_opponent="self_play",
            schema=REWARD_SCHEMAS["default"],
            gamma=1.0,
            max_pool_size=3,
            safety=self.safety,
        )
        try:
            runner.set_workers(workers)
            runner.sync_current(network)
            return runner.collect_training_iteration(0, game_count, seed)
        finally:
            runner.close()

    def test_worker_parser_enforces_hard_limit(self):
        self.assertEqual(worker_count("auto"), "auto")
        self.assertEqual(worker_count("20"), MAX_PARALLEL_WORKERS)
        with self.assertRaises(ValueError):
            worker_count("21")

        args = parse_pipeline_args([
            "small",
            "--rl-workers",
            "3",
            "--rl-memory-reserve-mb",
            "256",
        ])
        self.assertEqual(args.rl_workers, 3)
        self.assertEqual(args.rl_memory_reserve_mb, 256)

    def test_rollouts_are_identical_with_one_and_multiple_workers(self):
        one_worker, one_info = self._collect(1)
        two_workers, two_info = self._collect(2)
        self.assertEqual(
            _rollout_fingerprint(one_worker),
            _rollout_fingerprint(two_workers),
        )
        self.assertTrue(one_info.workers_cpu_only)
        self.assertTrue(two_info.workers_cpu_only)

    def test_seeded_training_checkpoints_are_bit_identical(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = [Path(temp_dir) / "one.npz", Path(temp_dir) / "two.npz"]
            summaries = []
            for workers, path in zip((1, 2), paths):
                summaries.append(train(
                    iterations=3,
                    games_per_iteration=6,
                    checkpoint_interval=100,
                    evaluation_games=4,
                    pool_interval=1,
                    max_pool_size=3,
                    seed=987,
                    device="cpu",
                    workers=workers,
                    safety_config=self.safety,
                    rl_weights_path=str(path),
                    quiet=True,
                ))
            with np.load(paths[0], allow_pickle=False) as one:
                with np.load(paths[1], allow_pickle=False) as two:
                    self.assertEqual(one.files, two.files)
                    for name in one.files:
                        np.testing.assert_array_equal(one[name], two[name])
            self.assertEqual(summaries[0]["effective_seed"], 987)
            self.assertEqual(summaries[1]["effective_seed"], 987)

    def test_numbered_checkpoint_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full_base = root / "full.npz"
            resumed_base = root / "resumed.npz"
            common = {
                "games_per_iteration": 4,
                "checkpoint_interval": 2,
                "evaluation_games": 2,
                "pool_interval": 1,
                "max_pool_size": 3,
                "seed": 987,
                "device": "cpu",
                "workers": 1,
                "safety_config": self.safety,
                "quiet": True,
                "numbered_checkpoints": True,
            }

            full = train(
                iterations=4,
                rl_weights_path=str(full_base),
                **common,
            )
            train(
                iterations=2,
                rl_weights_path=str(resumed_base),
                **common,
            )
            partial_weights = numbered_checkpoint_path(resumed_base, 2)
            partial_state = resume_state_path(partial_weights)
            metadata, pool = load_resume_state(partial_weights, partial_state)
            self.assertEqual(metadata["completed_iteration"], 2)
            self.assertEqual(len(pool), 3)
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                load_resume_state(full["rl_weights_path"], partial_state)

            with self.assertRaisesRegex(ValueError, "gamma"):
                train(
                    iterations=4,
                    rl_weights_path=str(resumed_base),
                    start_iteration=2,
                    resume_weights_path=str(partial_weights),
                    resume_state_file=str(partial_state),
                    gamma=0.97,
                    **common,
                )

            resumed = train(
                iterations=4,
                rl_weights_path=str(resumed_base),
                start_iteration=2,
                resume_weights_path=str(partial_weights),
                resume_state_file=str(partial_state),
                **common,
            )
            with np.load(full["rl_weights_path"], allow_pickle=False) as left:
                with np.load(resumed["rl_weights_path"], allow_pickle=False) as right:
                    self.assertEqual(left.files, right.files)
                    for name in left.files:
                        np.testing.assert_array_equal(left[name], right[name])

            final_weights = numbered_checkpoint_path(resumed_base, 4)
            self.assertEqual(resumed["rl_weights_path"], str(final_weights))
            self.assertFalse(partial_state.exists())
            self.assertTrue(resume_state_path(final_weights).exists())

    def test_numbered_resume_restores_value_head_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            common = {
                "games_per_iteration": 4,
                "checkpoint_interval": 1,
                "evaluation_games": 2,
                "pool_interval": 1,
                "max_pool_size": 2,
                "use_value_head": True,
                "seed": 654,
                "device": "cpu",
                "workers": 1,
                "safety_config": self.safety,
                "quiet": True,
                "numbered_checkpoints": True,
            }
            full_base = root / "full_critic.npz"
            resumed_base = root / "resumed_critic.npz"
            full = train(iterations=3, rl_weights_path=str(full_base), **common)
            train(iterations=1, rl_weights_path=str(resumed_base), **common)
            partial_weights = numbered_checkpoint_path(resumed_base, 1)
            resumed = train(
                iterations=3,
                rl_weights_path=str(resumed_base),
                start_iteration=1,
                resume_weights_path=str(partial_weights),
                resume_state_file=str(resume_state_path(partial_weights)),
                **common,
            )

            with np.load(full["rl_weights_path"], allow_pickle=False) as left:
                with np.load(resumed["rl_weights_path"], allow_pickle=False) as right:
                    self.assertIn("Wv", left.files)
                    self.assertIn("bv", left.files)
                    for name in left.files:
                        np.testing.assert_array_equal(left[name], right[name])

    def test_autotuning_retains_complete_training_iterations(self):
        messages = []
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = train(
                iterations=4,
                games_per_iteration=4,
                checkpoint_interval=100,
                evaluation_games=4,
                max_pool_size=2,
                seed=44,
                device="cpu",
                workers="auto",
                safety_config=self.safety,
                autotune_fraction=0.25,
                autotune_minimum_gain=1000.0,
                worker_candidates=(1, 2),
                status_callback=messages.append,
                rl_weights_path=str(Path(temp_dir) / "auto.npz"),
                quiet=True,
            )
        tuning = summary["autotune"]
        self.assertEqual(tuning["iterations_per_test"], 1)
        self.assertEqual(tuning["games_per_test"], 4)
        self.assertEqual(tuning["reused_iteration_count"], 2)
        self.assertEqual(tuning["reused_game_count"], 8)
        self.assertEqual(len(tuning["attempts"]), 2)
        self.assertTrue(all(attempt["passed"] for attempt in tuning["attempts"]))
        self.assertTrue(any("games retained" in message for message in messages))

    def test_low_ram_stops_autotuning_before_unsafe_candidate(self):
        constrained_memory = {
            "DOMINO_TEST_AVAILABLE_RAM_MB": "400",
            "DOMINO_TEST_TOTAL_RAM_MB": "4096",
        }
        safety = ParallelSafetyConfig(
            memory_reserve_mb=0,
            estimated_worker_mb=300,
            max_worker_rss_mb=1024,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(os.environ, constrained_memory, clear=False):
                summary = train(
                    iterations=4,
                    games_per_iteration=4,
                    checkpoint_interval=100,
                    evaluation_games=4,
                    max_pool_size=2,
                    seed=55,
                    device="cpu",
                    workers="auto",
                    safety_config=safety,
                    autotune_fraction=0.25,
                    autotune_minimum_gain=0.0,
                    worker_candidates=(1, 2),
                    status_callback=lambda _message: None,
                    rl_weights_path=str(Path(temp_dir) / "limited.npz"),
                    quiet=True,
                )
        self.assertEqual(summary["selected_workers"], 1)
        self.assertEqual(len(summary["autotune"]["attempts"]), 2)
        self.assertTrue(summary["autotune"]["attempts"][0]["passed"])
        self.assertFalse(summary["autotune"]["attempts"][1]["passed"])
        self.assertEqual(
            summary["autotune"]["attempts"][1]["completed_games"],
            0,
        )

    def test_runtime_memory_pressure_retains_games_and_reduces_workers(self):
        safety = ParallelSafetyConfig(
            memory_reserve_mb=512,
            estimated_worker_mb=1,
            max_worker_rss_mb=1024,
            memory_check_interval_s=0.0,
            poll_interval_s=0.01,
        )
        network = self._network()
        runner = RLRolloutRunner(
            network,
            training_opponent="self_play",
            schema=REWARD_SCHEMAS["default"],
            gamma=1.0,
            max_pool_size=2,
            safety=safety,
        )
        try:
            runner.set_workers(4)
            original_run_jobs = runner._run_jobs
            pressure_triggered = False

            def pressure_after_first_result(
                jobs,
                worker_function,
                on_result,
                run_info,
            ):
                nonlocal pressure_triggered
                if pressure_triggered:
                    return original_run_jobs(
                        jobs,
                        worker_function,
                        on_result,
                        run_info,
                    )

                def store_then_fail(result):
                    nonlocal pressure_triggered
                    on_result(result)
                    pressure_triggered = True
                    raise DiagnosticMemoryPressure("simulated runtime pressure")

                return original_run_jobs(
                    jobs,
                    worker_function,
                    store_then_fail,
                    run_info,
                )

            with mock.patch.object(
                runner,
                "_run_jobs",
                side_effect=pressure_after_first_result,
            ):
                recovered, run_info = runner.collect_training_iteration(
                    0,
                    24,
                    4321,
                )
        finally:
            runner.close()

        baseline, _baseline_info = self._collect(1, game_count=24, seed=4321)
        self.assertEqual(
            _rollout_fingerprint(recovered),
            _rollout_fingerprint(baseline),
        )
        self.assertGreaterEqual(run_info.fallback_count, 1)
        self.assertLess(run_info.final_workers, 4)
        self.assertGreater(
            run_info.fallback_history[-1]["completed_games"],
            0,
        )

    def test_rl_preflight_handles_low_host_and_gpu_memory(self):
        low_gpu = {
            "DOMINO_TEST_GPU_FREE_MB": "64",
            "DOMINO_TEST_GPU_TOTAL_MB": "8192",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            cpu_fallback_path = Path(temp_dir) / "cpu_fallback.npz"
            with mock.patch.dict(os.environ, low_gpu, clear=False):
                summary = train(
                    iterations=1,
                    games_per_iteration=2,
                    checkpoint_interval=100,
                    evaluation_games=2,
                    max_pool_size=1,
                    seed=99,
                    device="auto",
                    workers=1,
                    safety_config=self.safety,
                    rl_weights_path=str(cpu_fallback_path),
                    quiet=True,
                )
            self.assertEqual(summary["device"], "cpu")
            self.assertIn("64.0 MiB", summary["device_fallback_reason"])

            low_ram = {
                "DOMINO_TEST_AVAILABLE_RAM_MB": "1",
                "DOMINO_TEST_TOTAL_RAM_MB": "4096",
            }
            rejected_path = Path(temp_dir) / "must_not_exist.npz"
            with mock.patch.dict(os.environ, low_ram, clear=False):
                with self.assertRaises(MemorySafetyError):
                    train(
                        iterations=1,
                        games_per_iteration=40,
                        checkpoint_interval=100,
                        evaluation_games=2,
                        max_pool_size=50,
                        seed=99,
                        device="cpu",
                        workers=1,
                        safety_config=ParallelSafetyConfig(
                            memory_reserve_mb=0,
                            estimated_worker_mb=1,
                        ),
                        rl_weights_path=str(rejected_path),
                        quiet=True,
                    )
            self.assertFalse(rejected_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
