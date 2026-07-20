"""Interface regressions for the sequential RL parameter-sweep shell driver."""

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCRIPT = ROOT / "train_script" / "run_rl_parameter_sweep.sh"


class RLSweepShellTests(unittest.TestCase):
    def test_help_advertises_sequential_outer_and_automatic_inner_workers(self):
        result = subprocess.run(
            ["bash", str(SWEEP_SCRIPT), "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("outer sweep parallelism is disabled", result.stdout)
        self.assertIn("current sweep point (default: auto)", result.stdout)
        self.assertIn("newest valid numbered checkpoint", result.stdout)

    def test_outer_parallel_jobs_are_rejected_before_training(self):
        result = subprocess.run(
            ["bash", str(SWEEP_SCRIPT), "--jobs", "2"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("--jobs is fixed at 1", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
