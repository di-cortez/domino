"""Tests for the compact games-per-iteration sweep-table presentation."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.rl_sweep_table import build_display_rows, build_report


def _raw_row(gpi, win_rate_pct, *, value_coef=0.5):
    """Return one representative flattened model row for pivot testing."""
    return {
        "run_name": f"model_gpi{gpi}",
        "critic": "off",
        "critic_enabled": False,
        "learning_rate": 0.001,
        "gamma": 0.97,
        "games_per_iteration": gpi,
        "value_coef": value_coef,
        "rl_iterations": 2000,
        "seed": 42,
        "win_rate_pct": win_rate_pct,
    }


class RLSweepTableTests(unittest.TestCase):
    def test_empty_report_still_writes_all_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_dir = root / "empty-results"
            output_dir = root / "report"
            results_dir.mkdir()

            rows = build_report(
                results_dir=results_dir,
                output_dir=output_dir,
                quiet=True,
            )

            self.assertEqual(rows, [])
            self.assertTrue((output_dir / "rl_sweep_table.csv").exists())
            self.assertTrue((output_dir / "rl_sweep_table.json").exists())
            self.assertTrue((output_dir / "rl_sweep_table.png").exists())
            self.assertTrue((output_dir / "rl_sweep_table.pdf").exists())

    def test_three_gpi_models_become_one_display_row(self):
        raw_rows = [
            _raw_row(40, 61.0),
            _raw_row(80, 63.5),
            _raw_row(160, 64.2),
        ]

        display_rows, columns = build_display_rows(raw_rows)

        self.assertEqual(columns, (40, 80, 160))
        self.assertEqual(len(display_rows), 1)
        self.assertEqual(display_rows[0]["win_rate_pct_gpi_40"], 61.0)
        self.assertEqual(display_rows[0]["win_rate_pct_gpi_80"], 63.5)
        self.assertEqual(display_rows[0]["win_rate_pct_gpi_160"], 64.2)

    def test_value_coefficient_remains_a_separate_row(self):
        display_rows, _columns = build_display_rows([
            _raw_row(40, 61.0, value_coef=0.25),
            _raw_row(80, 62.0, value_coef=0.25),
            _raw_row(40, 63.0, value_coef=0.75),
        ])

        self.assertEqual(len(display_rows), 2)
        by_value_coef = {row["value_coef"]: row for row in display_rows}
        self.assertEqual(by_value_coef[0.25]["win_rate_pct_gpi_80"], 62.0)
        self.assertEqual(by_value_coef[0.75]["win_rate_pct_gpi_80"], "")

    def test_report_keeps_raw_csv_and_json_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_dir = root / "results"
            output_dir = root / "report"
            for gpi, wins in ((40, 61), (80, 63), (160, 64)):
                run_dir = results_dir / f"model_gpi{gpi}"
                run_dir.mkdir(parents=True)
                (run_dir / "sweep_run.json").write_text(json.dumps({
                    "run_name": f"model_gpi{gpi}",
                    "critic_enabled": False,
                    "varied_parameter": f"gpi{gpi}",
                    "learning_rate": 0.001,
                    "gamma": 0.97,
                    "games_per_iteration": gpi,
                    "value_coef": 0.5,
                    "rl_iterations": 2000,
                    "seed": 42,
                    "model_path": f"models/model_gpi{gpi}.npz",
                }), encoding="utf-8")
                (run_dir / "summary.json").write_text(json.dumps({
                    "rates": {
                        "win": wins / 100,
                        "draw": 0.01,
                        "loss": (99 - wins) / 100,
                    },
                    "counts": {"win": wins, "draw": 1, "loss": 99 - wins},
                    "win_ci95": [0.5, 0.7],
                    "mean_turns": 25.0,
                    "game_count": 100,
                    "duration_s": 1.0,
                }), encoding="utf-8")

            with mock.patch(
                "diagnostics.rl_sweep_table.plot_sweep_comparison_table"
            ) as plot_table:
                returned_rows = build_report(
                    results_dir=results_dir,
                    output_dir=output_dir,
                    quiet=True,
                )

            self.assertEqual(len(returned_rows), 3)
            self.assertEqual(plot_table.call_count, 2)
            persisted_json = json.loads(
                (output_dir / "rl_sweep_table.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(persisted_json), 3)
            with open(
                output_dir / "rl_sweep_table.csv",
                newline="",
                encoding="utf-8",
            ) as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)

            display_rows = plot_table.call_args.args[0]
            self.assertEqual(len(display_rows), 1)
            output_paths = [call.args[1].name for call in plot_table.call_args_list]
            self.assertEqual(
                output_paths,
                ["rl_sweep_table.png", "rl_sweep_table.pdf"],
            )
            self.assertEqual(
                plot_table.call_args.kwargs["games_per_iteration_values"],
                (40, 80, 160),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
