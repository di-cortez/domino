"""Tests for retained, deterministic multiprocessing in dataset generation.

Run from the repository root with::

    python tests/test_parallel_dataset.py
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.parallel_runner import ParallelSafetyConfig, game_seed
from training.dataset_generator import generate_dataset
from training.dataset_parallel import (
    DatasetExecutionError,
    evaluate_dataset_game_specs,
)


def _game_specs(count, seed):
    return [(index, game_seed(seed, index)) for index in range(count)]


class ParallelDatasetTests(unittest.TestCase):
    def test_parallel_payloads_equal_single_worker_payloads(self):
        safety = ParallelSafetyConfig(memory_reserve_mb=0, estimated_worker_mb=1)
        single, _single_info = evaluate_dataset_game_specs(
            game_specs=_game_specs(8, 111),
            requested_workers=1,
            safety=safety,
        )
        parallel, parallel_info = evaluate_dataset_game_specs(
            game_specs=_game_specs(8, 111),
            requested_workers=2,
            safety=safety,
        )
        self.assertEqual(single, parallel)
        self.assertTrue(parallel_info.workers_cpu_only)

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
            recovered, run_info = evaluate_dataset_game_specs(
                game_specs=_game_specs(8, 444),
                requested_workers=4,
                result_callback=store,
                safety=safety,
            )
        baseline, _baseline_info = evaluate_dataset_game_specs(
            game_specs=_game_specs(8, 444),
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
            output.write_text("previous valid dataset\n", encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
