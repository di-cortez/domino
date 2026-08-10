"""Tests for retained, deterministic multiprocessing in dataset generation.

Run from the repository root with::

    python tests/test_parallel_dataset.py
"""

import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.parallel_runner import ParallelSafetyConfig
from training.dataset_generator import (
    dataset_random_manifest_path,
    generate_dataset,
)
from training.dataset_parallel import (
    DatasetExecutionError,
    evaluate_dataset_games,
    generate_dataset_game,
)
from utils.myrandom import SeedPlan


def _numpy_rng_states_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


class ParallelDatasetTests(unittest.TestCase):
    def test_parallel_payloads_equal_single_worker_payloads(self):
        safety = ParallelSafetyConfig(memory_reserve_mb=0, estimated_worker_mb=1)
        seed_plan = SeedPlan(111)
        single, _single_info = evaluate_dataset_games(
            game_indices=range(8),
            seed_plan=seed_plan,
            requested_workers=1,
            safety=safety,
        )
        parallel, parallel_info = evaluate_dataset_games(
            game_indices=reversed(range(8)),
            seed_plan=seed_plan,
            requested_workers=2,
            safety=safety,
        )
        self.assertEqual(single, parallel)
        self.assertTrue(parallel_info.workers_cpu_only)

    def test_dataset_jsonl_is_byte_identical_across_repeats_and_workers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = [
                Path(temp_dir) / "single.jsonl",
                Path(temp_dir) / "parallel.jsonl",
                Path(temp_dir) / "repeat.jsonl",
            ]
            safety = ParallelSafetyConfig(memory_reserve_mb=0, estimated_worker_mb=1)
            for path, workers in zip(paths, (1, 3, 1)):
                generate_dataset(
                    16,
                    path,
                    quiet=True,
                    workers=workers,
                    seed=20260808,
                    safety_config=safety,
                )

            expected = paths[0].read_bytes()
            self.assertGreater(len(expected), 0)
            self.assertEqual(paths[1].read_bytes(), expected)
            self.assertEqual(paths[2].read_bytes(), expected)

    def test_one_game_does_not_consume_process_global_rng_state(self):
        random.seed(12345)
        np.random.seed(54321)
        python_state = random.getstate()
        numpy_state = np.random.get_state()

        result = generate_dataset_game(7, SeedPlan(222))

        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(_numpy_rng_states_equal(np.random.get_state(), numpy_state))
        self.assertEqual(result, generate_dataset_game(7, SeedPlan(222)))

    def test_autotune_games_are_retained_and_output_is_ordered(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "dataset.jsonl"
            summary = generate_dataset(
                20,
                output,
                quiet=True,
                workers="auto",
                seed=222,
                autotune_fraction=0.10,
                autotune_minimum_gain=1000.0,
                safety_config=ParallelSafetyConfig(
                    memory_reserve_mb=0,
                    estimated_worker_mb=1,
                ),
            )

            game_ids = []
            with open(output, "r", encoding="utf-8") as stream:
                for line in stream:
                    game_ids.append(json.loads(line)["state"]["game_id"])

            self.assertEqual(summary["autotune"]["games_per_test"], 2)
            self.assertEqual(summary["autotune"]["reused_game_count"], 4)
            self.assertEqual(len(summary["autotune"]["attempts"]), 2)
            self.assertEqual(game_ids, sorted(game_ids))
            self.assertGreater(summary["saved_turn_count"], 0)
            self.assertFalse(any(output.parent.glob(f".{output.name}.games-*.sqlite3")))
            manifest_path = dataset_random_manifest_path(output)
            self.assertEqual(Path(summary["random_manifest_path"]), manifest_path)
            self.assertEqual(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                SeedPlan(222).to_manifest(),
            )

    def test_low_ram_caps_workers_without_changing_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            normal = Path(temp_dir) / "normal.jsonl"
            constrained = Path(temp_dir) / "constrained.jsonl"
            safety = ParallelSafetyConfig(
                memory_reserve_mb=512,
                estimated_worker_mb=128,
            )
            normal_summary = generate_dataset(
                12,
                normal,
                quiet=True,
                workers=2,
                seed=333,
                safety_config=safety,
            )
            with mock.patch.dict(
                os.environ,
                {
                    "DOMINO_TEST_AVAILABLE_RAM_MB": "768",
                    "DOMINO_TEST_TOTAL_RAM_MB": "4096",
                },
                clear=False,
            ):
                constrained_summary = generate_dataset(
                    12,
                    constrained,
                    quiet=True,
                    workers=20,
                    seed=333,
                    safety_config=safety,
                )

            self.assertEqual(normal.read_bytes(), constrained.read_bytes())
            self.assertEqual(normal_summary["selected_workers"], 2)
            self.assertEqual(constrained_summary["selected_workers"], 2)

    def test_runtime_pressure_retries_and_retains_completed_games(self):
        completed = []
        pressure_triggered = False

        def store(result):
            completed.append(int(result["game_index"]))

        def pressure_after_progress(_executor):
            nonlocal pressure_triggered
            if completed and not pressure_triggered:
                pressure_triggered = True
                return 0.0, 0.0, 511.0
            return 0.0, 0.0, 4096.0

        safety = ParallelSafetyConfig(
            memory_reserve_mb=512,
            estimated_worker_mb=1,
            poll_interval_s=0.01,
            memory_check_interval_s=0.0,
        )
        with mock.patch(
            "training.dataset_parallel.executor_memory_snapshot",
            side_effect=pressure_after_progress,
        ):
            seed_plan = SeedPlan(444)
            recovered, run_info = evaluate_dataset_games(
                game_indices=range(8),
                seed_plan=seed_plan,
                requested_workers=4,
                result_callback=store,
                safety=safety,
            )
        baseline, _baseline_info = evaluate_dataset_games(
            game_indices=range(8),
            seed_plan=seed_plan,
            requested_workers=1,
            safety=ParallelSafetyConfig(
                memory_reserve_mb=0,
                estimated_worker_mb=1,
            ),
        )

        self.assertEqual(recovered, baseline)
        self.assertGreaterEqual(run_info.fallback_count, 1)
        self.assertGreater(run_info.fallback_history[-1]["completed_games"], 0)
        self.assertEqual(len(completed), 8)
        self.assertEqual(len(set(completed)), 8)

    def test_failed_generation_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "dataset.jsonl"
            manifest = dataset_random_manifest_path(output)
            output.write_text("previous valid dataset\n", encoding="utf-8")
            manifest.write_text("previous valid manifest\n", encoding="utf-8")
            with mock.patch.dict(
                os.environ,
                {
                    "DOMINO_TEST_AVAILABLE_RAM_MB": "400",
                    "DOMINO_TEST_TOTAL_RAM_MB": "4096",
                },
                clear=False,
            ):
                with self.assertRaises(DatasetExecutionError):
                    generate_dataset(
                        2,
                        output,
                        quiet=True,
                        workers=1,
                        seed=555,
                        safety_config=ParallelSafetyConfig(
                            memory_reserve_mb=512,
                            estimated_worker_mb=1,
                        ),
                    )
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "previous valid dataset\n",
            )
            self.assertEqual(
                manifest.read_text(encoding="utf-8"),
                "previous valid manifest\n",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
