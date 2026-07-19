"""Focused tests for retained, deterministic diagnostic multiprocessing.

Run from the repository root with::

    python tests/test_parallel_diagnostics.py
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.pairwise import evaluate_pair, run_pairwise
from diagnostics.evaluate import run_all_pairs
from diagnostics.parallel_runner import (
    MAX_DIAGNOSTIC_WORKERS,
    ParallelSafetyConfig,
    game_seed,
    safety_cap_workers,
)
from diagnostics.worker_autotune import MatchupSpec, autotune_diagnostic_workers
from diagnostics.plots import (
    WIN_RATE_COLOR_BANDS,
    diagnostic_table_header_lines,
    plot_all_pairs_table,
    win_rate_color_band,
    worst_case_margin_of_error,
)
from run_pipeline import _diagnostic_summary_text, parse_args as parse_pipeline_args
from utils.resource_limits import MemorySafetyError, choose_safe_rl_device


class ParallelDiagnosticsTests(unittest.TestCase):
    def test_pipeline_exposes_and_formats_independent_worker_counts(self):
        args = parse_pipeline_args([
            "small",
            "--dataset-workers",
            "2",
            "--diagnostic-workers",
            "3",
        ])
        self.assertEqual(args.dataset_workers, 2)
        self.assertEqual(args.diagnostic_workers, 3)
        text = _diagnostic_summary_text({
            "evaluated_matchups": 2,
            "game_count_per_matchup": 10,
            "selected_workers_by_matchup": {
                "rl_vs_random": 2,
                "heuristic_vs_random": 4,
            },
        })
        self.assertIn("rl_vs_random=2", text)
        self.assertIn("heuristic_vs_random=4", text)

    def test_margin_of_error_uses_requested_worst_case_formula(self):
        expected = {
            2401: 0.02,
            9604: 0.01,
            38416: 0.005,
            60026: 0.004,
            106712: 0.003,
            240101: 0.002,
            960401: 0.001,
            96040000: 0.0001,
        }
        for game_count, target_margin in expected.items():
            self.assertAlmostEqual(
                worst_case_margin_of_error(game_count),
                target_margin,
                places=6,
            )

    def test_win_rate_colors_change_at_every_five_point_boundary(self):
        probes = (29.9, 30, 35, 40, 45, 50, 55, 60, 65, 70)
        bands = [win_rate_color_band(value) for value in probes]
        self.assertEqual(len({fill for _label, fill, _text in bands}), 10)
        self.assertEqual(bands[0][0], "<30%")
        self.assertEqual(bands[-1][0], "≥70%")
        self.assertEqual(len(WIN_RATE_COLOR_BANDS), 10)

    def test_aggregate_table_renders_one_row_to_png_and_pdf(self):
        agents = ("rl", "neural", "random_nn", "heuristic", "random")
        rates = (0.28, 0.42, 0.52, 0.63, 0.72)
        summaries = [
            {
                "agent": agent,
                "opponent": "random",
                "game_count": 10000,
                "rates": {"win": rate},
            }
            for agent, rate in zip(agents, rates)
        ]
        metadata = {
            "diagnostic_mode": "complete",
            "game_count_per_matchup": 10000,
            "evaluated_matchups": 5,
            "duration_s": 125.0,
            "seed": 42,
            "selected_workers_by_matchup": {
                f"{agent}_vs_random": index + 1
                for index, agent in enumerate(agents)
            },
            "network_metadata": {
                "rl": {
                    "architecture": [168, 256, 128, 56],
                    "total_parameters": 83384,
                    "value_head": False,
                    "checkpoint_name": "rl.npz",
                },
            },
        }

        header = diagnostic_table_header_lines(
            summaries,
            agents,
            report_metadata=metadata,
        )
        self.assertTrue(any("±0.98 percentage points" in line for line in header))
        self.assertTrue(any("168→256→128→56" in line for line in header))

        with tempfile.TemporaryDirectory() as temp_dir:
            png_path = Path(temp_dir) / "table.png"
            pdf_path = Path(temp_dir) / "table.pdf"
            plot_all_pairs_table(summaries, agents, png_path, metadata)
            plot_all_pairs_table(summaries, agents, pdf_path, metadata)
            self.assertEqual(png_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(pdf_path.read_bytes()[:4], b"%PDF")

    def test_per_game_seed_is_stable_and_distinct(self):
        first = [game_seed(1234, index) for index in range(100)]
        second = [game_seed(1234, index) for index in range(100)]
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), len(first))

    def test_parallel_results_equal_single_worker_results(self):
        single_worker = evaluate_pair(
            "random",
            "random",
            game_count=16,
            seed=9876,
            workers=1,
        )
        parallel, metadata = evaluate_pair(
            "random",
            "random",
            game_count=16,
            seed=9876,
            workers=2,
            return_run_info=True,
            safety_config=ParallelSafetyConfig(
                memory_reserve_mb=0,
                estimated_worker_mb=1,
            ),
        )
        self.assertEqual(single_worker, parallel)
        self.assertTrue(metadata["parallel"]["workers_cpu_only"])
        self.assertLessEqual(metadata["parallel"]["initial_workers"], 2)

    def test_hard_worker_limit_and_low_ram_cap(self):
        safety = ParallelSafetyConfig(
            memory_reserve_mb=512,
            estimated_worker_mb=128,
        )
        with mock.patch("diagnostics.parallel_runner.os.cpu_count", return_value=64):
            with mock.patch.dict(
                os.environ,
                {
                    "DOMINO_TEST_AVAILABLE_RAM_MB": "65536",
                    "DOMINO_TEST_TOTAL_RAM_MB": "65536",
                },
                clear=False,
            ):
                capped, was_capped, _reason = safety_cap_workers(999, safety)
                self.assertEqual(capped, MAX_DIAGNOSTIC_WORKERS)
                self.assertTrue(was_capped)

            with mock.patch.dict(
                os.environ,
                {
                    "DOMINO_TEST_AVAILABLE_RAM_MB": "768",
                    "DOMINO_TEST_TOTAL_RAM_MB": "4096",
                },
                clear=False,
            ):
                capped, was_capped, reason = safety_cap_workers(20, safety)
                self.assertEqual(capped, 2)
                self.assertTrue(was_capped)
                self.assertIn("RAM preflight", reason)

    def test_runtime_memory_pressure_retries_and_keeps_determinism(self):
        completed_games = 0
        pressure_triggered = False

        def progress(done, _total):
            nonlocal completed_games
            completed_games = done

        def pressure_once(_executor):
            nonlocal pressure_triggered
            if completed_games > 0 and not pressure_triggered:
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
            "diagnostics.parallel_runner.executor_memory_snapshot",
            side_effect=pressure_once,
        ):
            recovered, metadata = evaluate_pair(
                "random",
                "random",
                game_count=12,
                seed=321,
                workers=4,
                safety_config=safety,
                progress_callback=progress,
                return_run_info=True,
            )
        single_worker = evaluate_pair(
            "random",
            "random",
            game_count=12,
            seed=321,
            workers=1,
        )
        self.assertEqual(recovered, single_worker)
        self.assertGreaterEqual(metadata["parallel"]["fallback_count"], 1)
        self.assertLess(metadata["parallel"]["final_workers"], 4)
        self.assertGreater(
            metadata["parallel"]["fallback_history"][-1]["completed_games"],
            0,
        )

    def test_autotune_retains_every_benchmark_game(self):
        result = autotune_diagnostic_workers(
            matchups=(MatchupSpec("random", "random"),),
            game_count=30,
            base_seed=44,
            safety=ParallelSafetyConfig(
                memory_reserve_mb=0,
                estimated_worker_mb=1,
            ),
            benchmark_fraction=0.10,
            minimum_gain=1000.0,
            candidates=(1, 2),
            status_callback=lambda _message: None,
        )
        records = result["precomputed_games"][("random", "random")]
        self.assertEqual(result["games_per_test"], 3)
        self.assertEqual(result["reused_game_count"], 6)
        self.assertEqual(len(records), 6)
        self.assertEqual(len({record["game"] for record in records}), 6)
        self.assertTrue(all(attempt["passed"] for attempt in result["attempts"]))

    def test_all_pairs_autotunes_each_matchup_independently(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "all_pairs"
            report = run_all_pairs(
                game_count=4,
                output_dir=output_dir,
                seed=88,
                generate_pair_plots=False,
                quiet=True,
                diagnostic_mode="fast",
                workers="auto",
                safety_config=ParallelSafetyConfig(
                    memory_reserve_mb=0,
                    estimated_worker_mb=1,
                ),
                autotune_fraction=0.25,
                autotune_minimum_gain=1000.0,
                status_callback=lambda _message: None,
            )

            self.assertTrue((output_dir / "all_pairs_table.png").exists())
            self.assertTrue((output_dir / "all_pairs_table.pdf").exists())

        expected_matchups = {
            "rl_vs_random",
            "neural_vs_random",
            "random_nn_vs_random",
            "heuristic_vs_random",
            "random_vs_random",
        }
        self.assertEqual(report["autotune"]["scope"], "per_matchup")
        self.assertEqual(set(report["selected_workers_by_matchup"]), expected_matchups)
        self.assertEqual(set(report["autotune"]["matchups"]), expected_matchups)
        self.assertEqual(report["autotune"]["reused_game_count"], 10)
        self.assertEqual(report["comparison_opponent"], "random")
        self.assertEqual(report["report_layout"], "single_row")
        self.assertEqual(set(report["network_metadata"]), {"rl", "neural", "random_nn"})
        for tuning in report["autotune"]["matchups"].values():
            self.assertEqual(tuning["games_per_test"], 1)
            self.assertEqual(tuning["reused_game_count"], 2)
            self.assertEqual(len(tuning["attempts"]), 2)

    def test_low_vram_auto_falls_back_and_explicit_gpu_fails(self):
        healthy_vram = {
            "DOMINO_TEST_GPU_FREE_MB": "4096",
            "DOMINO_TEST_GPU_TOTAL_MB": "8192",
        }
        with mock.patch.dict(os.environ, healthy_vram, clear=False):
            selected, reason = choose_safe_rl_device("auto")
            self.assertEqual(selected, "auto")
            self.assertIsNone(reason)
            selected, reason = choose_safe_rl_device("gpu")
            self.assertEqual(selected, "gpu")
            self.assertIsNone(reason)

        low_vram = {
            "DOMINO_TEST_GPU_FREE_MB": "64",
            "DOMINO_TEST_GPU_TOTAL_MB": "8192",
        }
        with mock.patch.dict(os.environ, low_vram, clear=False):
            selected, reason = choose_safe_rl_device("auto")
            self.assertEqual(selected, "cpu")
            self.assertIn("64.0 MiB", reason)
            with self.assertRaises(MemorySafetyError):
                choose_safe_rl_device("gpu")

    def test_low_ram_preflight_preserves_previous_pairwise_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "diagnostic"
            output_dir.mkdir()
            marker = output_dir / "previous-valid-result.txt"
            marker.write_text("keep me", encoding="utf-8")
            with mock.patch.dict(
                os.environ,
                {
                    "DOMINO_TEST_AVAILABLE_RAM_MB": "512",
                    "DOMINO_TEST_TOTAL_RAM_MB": "4096",
                },
                clear=False,
            ):
                with self.assertRaises(MemorySafetyError):
                    run_pairwise(
                        "random",
                        "random",
                        game_count=2,
                        output_dir=output_dir,
                        generate_plots=False,
                        print_console_summary=False,
                        safety_config=ParallelSafetyConfig(memory_reserve_mb=512),
                    )
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep me")

    def test_supervised_encoder_preallocates_float32_and_honors_ram_guard(self):
        from agents.encoder import DominoEncoder
        from training.dataset_generator import generate_dataset
        from training.training_loop import load_dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "tiny.jsonl"
            generate_dataset(10, str(dataset_path), quiet=True, workers=1, seed=7)
            x, y = load_dataset(str(dataset_path), DominoEncoder(), quiet=True)
            self.assertEqual(x.dtype.name, "float32")
            self.assertEqual(y.dtype.name, "float32")
            self.assertGreater(x.shape[1], 0)

            with mock.patch.dict(
                os.environ,
                {
                    "DOMINO_TEST_AVAILABLE_RAM_MB": "512",
                    "DOMINO_TEST_TOTAL_RAM_MB": "4096",
                },
                clear=False,
            ):
                with self.assertRaises(MemorySafetyError):
                    load_dataset(str(dataset_path), DominoEncoder(), quiet=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
