"""Interface regressions for the sequential RL parameter-sweep shell driver."""

import subprocess
import unittest
from pathlib import Path

from train_script import run_rl_parameter_sweep


ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCRIPT = ROOT / "train_script" / "run_rl_parameter_sweep.sh"
PYTHON_SWEEP = ROOT / "train_script" / "run_rl_parameter_sweep.py"
HYPERPARAMETER_SWEEP = ROOT / "diagnostics" / "hyperparameter_sweep.py"


class RLSweepShellTests(unittest.TestCase):
    def test_help_advertises_sequential_outer_and_automatic_inner_workers(self):
        result = subprocess.run(
            ["bash", str(SWEEP_SCRIPT), "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("Outer sweep parallelism is disabled", result.stdout)
        self.assertIn("current sweep point (default: auto)", result.stdout)
        self.assertIn("reuse complete, compatible diagnostics", result.stdout)
        self.assertIn("diagnostic per sweep point (default: 10000)", result.stdout)
        self.assertIn("--compact", SWEEP_SCRIPT.read_text(encoding="utf-8"))
        self.assertNotIn("games_per_iteration", result.stdout)
        self.assertNotIn("GPI", result.stdout)

    def test_removed_jobs_option_is_rejected_as_unknown(self):
        result = subprocess.run(
            ["bash", str(SWEEP_SCRIPT), "--jobs", "1"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Unknown option: --jobs", result.stderr)

    def test_completed_diagnostic_is_checked_before_resume_pool_state(self):
        script = SWEEP_SCRIPT.read_text(encoding="utf-8")
        run_point = script[script.index("run_point() {"):]
        completed_check = run_point.index(
            '"$final_model_path" "$diag_dir"; then'
        )
        resume_scan = run_point.index("find_latest_resume_checkpoint")

        self.assertLess(completed_check, resume_scan)
        self.assertIn(
            "a CPU/GPU selection change cannot",
            run_point[completed_check - 800:completed_check],
        )

    def test_parameter_sweeps_explicitly_start_new_points_from_supervised(self):
        shell_source = SWEEP_SCRIPT.read_text(encoding="utf-8")
        python_source = PYTHON_SWEEP.read_text(encoding="utf-8")
        diagnostic_source = HYPERPARAMETER_SWEEP.read_text(encoding="utf-8")

        self.assertIn("FRESH_START_ARGS=(--fresh-from-sl)", shell_source)
        self.assertIn("fresh_from_sl=True", python_source)
        self.assertIn("fresh_from_sl=True", diagnostic_source)

    def test_sweeps_do_not_expose_or_forward_gpi(self):
        shell_source = SWEEP_SCRIPT.read_text(encoding="utf-8")
        python_source = PYTHON_SWEEP.read_text(encoding="utf-8")
        diagnostic_source = HYPERPARAMETER_SWEEP.read_text(encoding="utf-8")

        self.assertNotIn("--gpi", shell_source)
        self.assertNotIn("games_per_iteration", shell_source)
        self.assertNotIn("GAMES_PER_ITERATION_VALUES", python_source)
        self.assertNotIn("gpi=", python_source)
        self.assertNotIn("--rl-games-per-iteration", diagnostic_source)
        self.assertNotIn("gpi=", diagnostic_source)

    def test_python_sweep_grid_has_no_gpi_axis(self):
        points = list(run_rl_parameter_sweep.iter_sweep_points())

        self.assertEqual(len(points), 18)
        self.assertTrue(all(len(point) == 4 for point in points))
        self.assertTrue(all("gpi" not in point[0].lower() for point in points))


if __name__ == "__main__":
    unittest.main(verbosity=2)
