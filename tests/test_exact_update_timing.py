"""Tests for pipeline-wide ExactOpponentModel.update timing aggregation."""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from middleware.opponent_model import ExactOpponentModel
from training.dataset_generator import generate_dataset
from utils.exact_update_timing import (
    begin_pipeline_stage,
    end_pipeline_stage,
    finish_pipeline_timing,
    start_pipeline_timing,
)


def _initial_observer_state(game_id: int = 1) -> dict:
    """Return a complete initial state accepted by the exact model."""
    hand = [[0, value] for value in range(7)]
    return {
        "game_id": game_id,
        "observer_player": 0,
        "current_player": 0,
        "current_player_initial_hand": hand,
        "current_player_drawn_tiles": [],
        "current_player_hand": hand,
        "hand_sizes": [7, 7],
        "board_history": [],
        "ends": [],
        "turn": 0,
        "game_over": False,
    }


def _spawned_exact_update() -> None:
    """Exercise inherited timing configuration in one spawned worker."""
    ExactOpponentModel().update(_initial_observer_state(game_id=2))


class ExactUpdateTimingTests(unittest.TestCase):
    def test_real_dataset_process_pool_flushes_worker_timing(self):
        """ProcessPoolExecutor workers flush timing before stage aggregation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path = root / "exact_opponent_model_timing.json"
            start_pipeline_timing(report_path, {"pipeline_scale": "pool-test"})
            try:
                token = begin_pipeline_stage("dataset_generation")
                generate_dataset(
                    8,
                    str(root / "dataset.jsonl"),
                    quiet=True,
                    workers=2,
                    seed=123,
                )
                end_pipeline_stage(token, status="completed")
            finally:
                finish_pipeline_timing(status="completed")

            stage = json.loads(report_path.read_text(encoding="utf-8"))[
                "runs"
            ][0]["stages"]["dataset_generation"]
            exact = stage["exact_opponent_model_update"]
            self.assertGreater(exact["calls"], 0)
            self.assertEqual(exact["processes_with_calls"], 2)
            self.assertGreater(exact["cpu_seconds"], 0.0)
            self.assertGreater(stage["child_processes_cpu_seconds"], 0.0)

    def test_pipeline_timing_splits_four_stages_and_collects_worker(self):
        """Parent and worker calls contribute to one non-overlapping split."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "exact_opponent_model_timing.json"
            start_pipeline_timing(report_path, {"pipeline_scale": "test"})
            try:
                dataset_token = begin_pipeline_stage("dataset_generation")
                ExactOpponentModel().update(_initial_observer_state())
                worker = mp.get_context("spawn").Process(
                    target=_spawned_exact_update
                )
                worker.start()
                worker.join(timeout=30)
                self.assertEqual(worker.exitcode, 0)
                end_pipeline_stage(dataset_token, status="completed")

                for stage_name in (
                    "supervised_training",
                    "rl_self_play",
                    "diagnostics",
                ):
                    token = begin_pipeline_stage(stage_name)
                    end_pipeline_stage(token, status="completed")
            finally:
                finish_pipeline_timing(status="completed")

            document = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(document["schema_version"], 1)
            self.assertEqual(len(document["runs"]), 1)
            run = document["runs"][0]
            self.assertEqual(run["status"], "completed")
            self.assertEqual(set(run["stages"]), {
                "dataset_generation",
                "supervised_training",
                "rl_self_play",
                "diagnostics",
            })

            dataset = run["stages"]["dataset_generation"]
            exact = dataset["exact_opponent_model_update"]
            other = dataset["everything_else"]
            self.assertEqual(exact["calls"], 2)
            self.assertEqual(exact["processes_with_calls"], 2)
            self.assertGreater(exact["cpu_seconds"], 0.0)
            self.assertGreater(exact["aggregate_call_wall_seconds"], 0.0)
            self.assertGreater(dataset["child_processes_cpu_seconds"], 0.0)
            self.assertGreaterEqual(
                dataset["aggregate_process_cpu_seconds"],
                exact["cpu_seconds"],
            )
            self.assertGreaterEqual(other["cpu_seconds"], 0.0)
            self.assertAlmostEqual(
                exact["share_of_aggregate_process_cpu_percent"]
                + other["share_of_aggregate_process_cpu_percent"],
                100.0,
            )

            for stage_name in (
                "supervised_training",
                "rl_self_play",
                "diagnostics",
            ):
                stage = run["stages"][stage_name]
                self.assertEqual(
                    stage["exact_opponent_model_update"]["calls"],
                    0,
                )
                self.assertEqual(
                    stage["exact_opponent_model_update"]["cpu_seconds"],
                    0.0,
                )
                self.assertGreaterEqual(
                    stage["everything_else"]["cpu_seconds"],
                    0.0,
                )

    def test_timing_report_appends_runs_instead_of_overwriting(self):
        """Small and regular pipelines can share the requested report file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = Path(temp_dir) / "exact_opponent_model_timing.json"
            for scale in ("small", "default"):
                start_pipeline_timing(report_path, {"pipeline_scale": scale})
                token = begin_pipeline_stage("supervised_training")
                end_pipeline_stage(token, status="completed")
                finish_pipeline_timing(status="completed")

            document = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [
                    run["metadata"]["pipeline_scale"]
                    for run in document["runs"]
                ],
                ["small", "default"],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
