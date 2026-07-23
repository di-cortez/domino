"""Focused tests for safe retained supervised CPU/GPU autotuning."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.encoder import DominoEncoder
from agents.neural_agent import NeuralAgent
from agents.nn import GPU_ENABLED, SupervisedNeuralNetwork
from agents.rl_nn import PolicyNetwork
from training.supervised_runtime import (
    RetainedBatchAutotuner,
    SUPERVISED_CPU_BATCH_CANDIDATES,
    SUPERVISED_GPU_BATCH_CANDIDATES,
    SupervisedDataPlan,
    effective_batch_candidates,
    probe_gpu_residency,
)
from training.training_loop import (
    _save_supervised_loss_plot,
    _mmap_cache_paths,
    _supervised_loss_axis_limits,
    load_dataset,
    load_or_build_dataset,
    supervised_loss_plot_path,
    train_supervised,
)
from utils.resource_limits import (
    MemorySafetyError,
    choose_safe_supervised_device,
)
from train_script import run_pipeline


def _safe_preflight(batch_size):
    return {"safe": True, "batch_size": batch_size}


def _dataset_record(game_id):
    state = {
        "game_id": game_id,
        "ends": [4, 6],
        "current_player_hand": [[3, 4], [1, 6], [2, 6]],
        "current_player_initial_hand": [[3, 4], [1, 6], [2, 6]],
        "current_player_drawn_tiles": [],
        "current_player": 0,
        "turn": 2,
        "hand_sizes": [3, 3],
        "board_history": [[[6, 6], 0], [[4, 6], 0]],
        "stock_size": 8,
        "opponent_suit_probabilities": [0.5] * 7,
    }
    return {"state": state, "target_action": [[3, 4], 0]}


def _write_dataset(path, count=12):
    with open(path, "w", encoding="utf-8") as stream:
        for game_id in range(count):
            stream.write(json.dumps(_dataset_record(game_id)) + "\n")


class BatchAutotunerTests(unittest.TestCase):
    def test_candidate_lists_and_full_dataset_caps(self):
        self.assertEqual(SUPERVISED_CPU_BATCH_CANDIDATES[0], 1024)
        self.assertEqual(SUPERVISED_CPU_BATCH_CANDIDATES[-1], 1048576)
        self.assertEqual(SUPERVISED_GPU_BATCH_CANDIDATES[0], 2048)
        self.assertEqual(SUPERVISED_GPU_BATCH_CANDIDATES[-1], 1048576)
        self.assertEqual(effective_batch_candidates("cpu", 900), (900,))
        self.assertEqual(
            effective_batch_candidates("gpu", 5000),
            (2048, 4096, 5000),
        )

    def _run_two_candidates(self, second_gain):
        examples = 4096
        tuner = RetainedBatchAutotuner(
            device="cpu",
            training_examples=examples,
            total_epochs=30,
            preflight=_safe_preflight,
        )
        for epoch in range(10):
            tuner.record_epoch(epoch, 1.0)
        second_duration = 1.0 / (1.0 + second_gain)
        for epoch in range(10, 20):
            tuner.record_epoch(epoch, second_duration)
        return tuner

    def test_exact_ten_percent_is_accepted_but_9_9_percent_is_rejected(self):
        rejected = self._run_two_candidates(0.099)
        self.assertTrue(rejected.finished)
        self.assertEqual(rejected.selected_batch_size, 1024)
        self.assertAlmostEqual(
            rejected.attempts[1]["gain_over_accepted"],
            0.099,
        )
        accepted = self._run_two_candidates(0.10)
        self.assertFalse(accepted.finished)
        self.assertEqual(accepted.selected_batch_size, 2048)
        self.assertGreaterEqual(
            accepted.attempts[1]["gain_over_accepted"],
            0.10,
        )

    def test_status_lines_report_times_throughput_gain_and_selection(self):
        messages = []
        tuner = RetainedBatchAutotuner(
            device="gpu",
            training_examples=4096,
            total_epochs=20,
            preflight=_safe_preflight,
            status_callback=messages.append,
        )
        for epoch in range(10):
            tuner.record_epoch(epoch, 0.5)
        for epoch in range(10, 20):
            tuner.record_epoch(epoch, 0.5)

        combined = "\n".join(messages)
        self.assertIn("optimal supervised batch size on GPU", combined)
        self.assertIn("median epoch 0.500s", combined)
        self.assertIn("test total 5.0s", combined)
        self.assertIn("8,192 examples/s", combined)
        self.assertIn("+0.0% improvement", combined)
        self.assertIn("Marginal gain is below 10%", combined)
        self.assertIn("Optimal supervised batch size: 2,048", combined)

    def test_rejected_candidate_epochs_remain_and_progress_is_exact(self):
        network = SupervisedNeuralNetwork(
            input_size=2,
            hidden1_size=2,
            hidden2_size=2,
            output_size=2,
            random_seed=1,
            device="cpu",
        )
        tuner = RetainedBatchAutotuner(
            device="cpu",
            training_examples=4096,
            total_epochs=25,
            preflight=_safe_preflight,
        )
        initial = network.W1.copy()
        seen_batches = []
        progress = []

        def epoch_runner(current_network, batch_size, _epoch):
            current_network.W1 += np.float32(1.0)
            seen_batches.append(batch_size)
            return 0.0, 1, 0

        timer_values = [float(value) for value in range(50)]
        with mock.patch("agents.nn.time.perf_counter", side_effect=timer_values):
            history = network.train(
                np.zeros((2, 1), dtype=np.float32),
                np.zeros((2, 1), dtype=np.float32),
                epochs=25,
                batch_size=1024,
                quiet=True,
                epoch_runner=epoch_runner,
                batch_controller=tuner,
                progress_callback=lambda done, total: progress.append((done, total)),
            )

        self.assertEqual(len(history), 25)
        np.testing.assert_allclose(network.W1, initial + np.float32(25.0))
        self.assertEqual(seen_batches[:10], [1024] * 10)
        self.assertEqual(seen_batches[10:20], [2048] * 10)
        self.assertEqual(seen_batches[20:], [1024] * 5)
        self.assertEqual(tuner.autotune_epochs_retained, 20)
        self.assertEqual(progress, [(index, 25) for index in range(1, 26)])

    def test_memory_preflight_rejects_before_larger_candidate_updates(self):
        calls = []

        def preflight(batch_size):
            calls.append(batch_size)
            return {
                "safe": batch_size == 1024,
                "reason": "simulated low memory",
            }

        tuner = RetainedBatchAutotuner(
            device="cpu",
            training_examples=4096,
            total_epochs=30,
            preflight=preflight,
        )
        for epoch in range(10):
            tuner.record_epoch(epoch, 1.0)
        self.assertEqual(calls, [1024, 2048])
        self.assertTrue(tuner.finished)
        self.assertEqual(tuner.current_batch_size, 1024)
        self.assertEqual(tuner.attempts[-1]["completed_epochs"], 0)

    def test_runtime_memory_failure_retries_with_last_accepted_batch(self):
        network = SupervisedNeuralNetwork(
            input_size=2,
            hidden1_size=2,
            hidden2_size=2,
            output_size=2,
            random_seed=1,
            device="cpu",
        )
        tuner = RetainedBatchAutotuner(
            device="cpu",
            training_examples=4096,
            total_epochs=25,
            preflight=_safe_preflight,
        )
        successful_updates = []
        failed_once = False

        def epoch_runner(_network, batch_size, epoch):
            nonlocal failed_once
            if epoch == 10 and batch_size == 2048 and not failed_once:
                failed_once = True
                raise MemoryError("simulated allocator pressure")
            successful_updates.append((epoch, batch_size))
            return 0.0, 1, 0

        history = network.train(
            np.zeros((2, 1), dtype=np.float32),
            np.zeros((2, 1), dtype=np.float32),
            epochs=25,
            batch_size=1024,
            quiet=True,
            epoch_runner=epoch_runner,
            batch_controller=tuner,
        )
        self.assertEqual(len(history), 25)
        self.assertEqual(len(successful_updates), 25)
        self.assertEqual(successful_updates[10], (10, 1024))
        self.assertEqual(tuner.selected_batch_size, 1024)
        self.assertIn("MemoryError", tuner.attempts[-1]["failure_reason"])


class SchedulerAndNumericTests(unittest.TestCase):
    def _network_and_arrays(self):
        network = SupervisedNeuralNetwork(
            input_size=2,
            hidden1_size=3,
            hidden2_size=2,
            output_size=2,
            learning_rate=0.01,
            random_seed=4,
            device="cpu",
        )
        x = np.ones((2, 3), dtype=np.float32)
        y = np.zeros((2, 3), dtype=np.float32)
        y[0] = 1.0
        return network, x, y

    def test_scheduler_waits_five_checks_resets_and_decays_twice(self):
        network, x, y = self._network_and_arrays()
        validation_losses = iter(
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        )
        learning_rates_at_checks = []
        output = io.StringIO()
        with redirect_stdout(output):
            network.train(
                x,
                y,
                x_val=x,
                y_val=y,
                epochs=12,
                batch_size=3,
                quiet=False,
                validation_interval=1,
                validation_runner=lambda *_args: next(validation_losses),
                on_validation=lambda *_args: learning_rates_at_checks.append(network.lr),
                lr_decay_factor=0.5,
                lr_decay_patience=5,
            )
        self.assertEqual(learning_rates_at_checks[:6], [0.01] * 6)
        self.assertAlmostEqual(network.lr, 0.0025)
        self.assertEqual(network.last_training_summary["lr_decay_count"], 2)
        message = "learning rate reduced from 0.01000000 to 0.00500000"
        self.assertIn(message, output.getvalue())
        self.assertEqual(output.getvalue().count("consecutive checks"), 2)

    def test_early_stopping_counter_is_independent_and_quiet_is_silent(self):
        network, x, y = self._network_and_arrays()
        output = io.StringIO()
        with redirect_stdout(output):
            history = network.train(
                x,
                y,
                x_val=x,
                y_val=y,
                epochs=20,
                batch_size=3,
                quiet=True,
                validation_interval=1,
                validation_runner=lambda *_args: 1.0,
                early_stopping_patience=3,
                lr_decay_factor=0.5,
                lr_decay_patience=5,
            )
        self.assertEqual(len(history), 4)
        self.assertAlmostEqual(network.lr, 0.01)
        self.assertEqual(network.last_training_summary["lr_checks_without_improvement"], 3)
        self.assertEqual(output.getvalue(), "")

    def test_training_loss_plateau_requires_repeated_complete_blocks(self):
        network, x, y = self._network_and_arrays()
        metrics = []

        history = network.train(
            x,
            y,
            epochs=30,
            batch_size=3,
            quiet=True,
            epoch_runner=lambda *_args: (0.5, 1, 0),
            epoch_metrics_callback=lambda row: metrics.append(row.copy()),
            training_plateau_window=2,
            training_plateau_patience=3,
            training_plateau_min_epochs=4,
            training_plateau_min_relative_improvement=0.001,
        )

        self.assertEqual(len(history), 8)
        self.assertEqual(
            [row["epoch"] + 1 for row in metrics if row["training_plateau_checked"]],
            [4, 6, 8],
        )
        self.assertTrue(network.last_training_summary["training_plateau_stopped"])
        self.assertEqual(
            network.last_training_summary["stopping_reason"],
            "training_loss_plateau",
        )
        self.assertEqual(
            network.last_training_summary[
                "training_plateau_checks_without_improvement"
            ],
            3,
        )

    def test_meaningful_training_loss_improvement_resets_plateau_patience(self):
        network, x, y = self._network_and_arrays()
        block_losses = iter(
            [1.0, 1.0, 0.9995, 0.9995, 0.98, 0.98, 0.9795, 0.9795]
        )

        history = network.train(
            x,
            y,
            epochs=8,
            batch_size=3,
            quiet=True,
            epoch_runner=lambda *_args: (next(block_losses), 1, 0),
            training_plateau_window=2,
            training_plateau_patience=2,
            training_plateau_min_epochs=4,
            training_plateau_min_relative_improvement=0.001,
        )

        self.assertEqual(len(history), 8)
        self.assertFalse(network.last_training_summary["training_plateau_stopped"])
        self.assertEqual(network.last_training_summary["stopping_reason"], "epoch_limit")
        self.assertEqual(
            network.last_training_summary[
                "training_plateau_checks_without_improvement"
            ],
            1,
        )

    def test_training_plateau_excludes_batch_tuning_epochs(self):
        network, x, y = self._network_and_arrays()

        class TwoEpochBatchTuner:
            current_batch_size = 3
            finished = False

            def record_epoch(self, epoch_index, _duration_s):
                if epoch_index == 1:
                    self.finished = True

        history = network.train(
            x,
            y,
            epochs=12,
            batch_size=3,
            quiet=True,
            epoch_runner=lambda *_args: (0.5, 1, 0),
            batch_controller=TwoEpochBatchTuner(),
            training_plateau_window=2,
            training_plateau_patience=1,
            training_plateau_min_epochs=1,
            training_plateau_min_relative_improvement=0.001,
        )

        self.assertEqual(len(history), 6)
        self.assertEqual(
            network.last_training_summary["training_plateau_loss_start_epoch"],
            3,
        )
        self.assertEqual(
            network.last_training_summary["stopping_reason"],
            "training_loss_plateau",
        )

    def test_float32_forward_backward_and_legacy_checkpoint_loading(self):
        network, x, y = self._network_and_arrays()
        network.forward(x.astype(np.float64))
        network.backward(y.astype(np.float64))
        for value in network.cache.values():
            self.assertEqual(value.dtype, np.float32)
        for dtype in network.last_gradient_dtypes.values():
            self.assertEqual(dtype, np.float32)

        legacy = {
            name: np.asarray(getattr(network, name), dtype=np.float64)
            for name in network.weight_names
        }
        network.load_policy_weights(legacy)
        for name in network.weight_names:
            self.assertEqual(getattr(network, name).dtype, np.float32)

    def test_fixed_seed_cpu_training_is_reproducible(self):
        x = np.arange(24, dtype=np.float32).reshape(4, 6) / 24
        y = np.zeros((2, 6), dtype=np.float32)
        y[0, ::2] = 1
        y[1, 1::2] = 1

        final_weights = []
        for _run in range(2):
            np.random.seed(77)
            network = SupervisedNeuralNetwork(
                input_size=4,
                hidden1_size=5,
                hidden2_size=3,
                output_size=2,
                random_seed=77,
                device="cpu",
            )
            network.train(x, y, epochs=3, batch_size=2, quiet=True)
            final_weights.append([getattr(network, name).copy() for name in network.weight_names])
        for first, second in zip(*final_weights):
            np.testing.assert_array_equal(first, second)


class DatasetResidencyTests(unittest.TestCase):
    def test_ram_and_mmap_caches_have_identical_values_and_atomic_rebuild(self):
        encoder = DominoEncoder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "examples.jsonl"
            cache_path = root / "supervised_dataset_encoded.npz"
            _write_dataset(dataset_path)

            x_ram, y_ram = load_dataset(
                dataset_path,
                encoder,
                quiet=True,
                memory_reserve_mb=0,
            )
            with mock.patch.dict(
                os.environ,
                {
                    "DOMINO_TEST_AVAILABLE_RAM_MB": "1",
                    "DOMINO_TEST_TOTAL_RAM_MB": "8",
                },
            ):
                mmap_data = load_or_build_dataset(
                    dataset_path,
                    encoder,
                    cache_path,
                    quiet=True,
                    memory_reserve_mb=2,
                    return_info=True,
                )
                self.assertEqual(mmap_data.storage_mode, "mmap")
                self.assertIsInstance(mmap_data.x, np.memmap)
                np.testing.assert_array_equal(mmap_data.x, x_ram)
                np.testing.assert_array_equal(mmap_data.y, y_ram)

                x_path, _y_path, metadata_path = _mmap_cache_paths(cache_path)
                metadata_path.write_text("{incomplete", encoding="utf-8")
                rebuilt = load_or_build_dataset(
                    dataset_path,
                    encoder,
                    cache_path,
                    quiet=True,
                    memory_reserve_mb=2,
                    return_info=True,
                )
                np.testing.assert_array_equal(rebuilt.x, x_ram)
                self.assertGreater(x_path.stat().st_size, x_ram.nbytes)
                self.assertFalse(list(root.glob(".*.tmp*")))

    def test_ram_cache_split_arrays_are_views(self):
        encoder = DominoEncoder()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "examples.jsonl"
            cache_path = root / "cache.npz"
            _write_dataset(dataset_path)
            data = load_or_build_dataset(
                dataset_path,
                encoder,
                cache_path,
                quiet=True,
                memory_reserve_mb=0,
                return_info=True,
            )
            self.assertEqual(data.storage_mode, "ram")
            split = int(data.x.shape[1] * 0.85)
            self.assertTrue(np.shares_memory(data.x[:, :split], data.x))
            self.assertTrue(np.shares_memory(data.y[:, split:], data.y))

    def test_cpu_mmap_epoch_processes_each_training_index_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x = np.lib.format.open_memmap(
                root / "X.npy",
                mode="w+",
                dtype=np.float32,
                shape=(4, 12),
            )
            y = np.lib.format.open_memmap(
                root / "Y.npy",
                mode="w+",
                dtype=np.float32,
                shape=(2, 12),
            )
            x[:] = np.arange(48, dtype=np.float32).reshape(4, 12)
            y[:] = 0
            y[0] = 1
            observed = []
            plan = SupervisedDataPlan(
                x,
                y,
                train_count=10,
                host_storage_mode="mmap",
                device="cpu",
                index_observer=lambda indices: observed.extend(indices.tolist()),
            )
            network = SupervisedNeuralNetwork(
                input_size=4,
                hidden1_size=5,
                hidden2_size=3,
                output_size=2,
                random_seed=8,
                device="cpu",
            )
            _loss, updates, rotations = plan.train_epoch(network, 3, 0)
            self.assertEqual(sorted(observed), list(range(10)))
            self.assertEqual(len(set(observed)), 10)
            self.assertEqual(updates, 4)
            self.assertEqual(rotations, 0)
            self.assertEqual(plan.storage_mode, "mmap")

    def test_simulated_low_vram_falls_back_or_fails_before_training(self):
        with mock.patch.dict(
            os.environ,
            {"DOMINO_TEST_GPU_FREE_MB": "100", "DOMINO_TEST_GPU_TOTAL_MB": "4096"},
        ):
            selected, reason = choose_safe_supervised_device("auto", 512)
            self.assertEqual(selected, "cpu")
            self.assertIn("100.0 MiB", reason)
            with self.assertRaises(MemorySafetyError):
                choose_safe_supervised_device("gpu", 512)

    def test_loss_plot_failure_preserves_previous_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            weights_path = Path(directory) / "domino_sl_weights.npz"
            plot_path = supervised_loss_plot_path(weights_path)
            plot_path.write_bytes(b"previous graph")

            with mock.patch(
                "matplotlib.figure.Figure.savefig",
                side_effect=RuntimeError("simulated plot failure"),
            ), self.assertRaisesRegex(RuntimeError, "simulated plot failure"):
                _save_supervised_loss_plot(
                    [1.0, 0.75],
                    [
                        {"epoch": 0, "validation_loss": 0.9},
                        {"epoch": 1, "validation_loss": None},
                    ],
                    weights_path,
                )

            self.assertEqual(plot_path.read_bytes(), b"previous graph")
            self.assertEqual(list(plot_path.parent.glob(".*.tmp-*.png")), [])

    def test_loss_plot_limits_follow_observed_terminal_and_maximum_losses(self):
        lower, upper = _supervised_loss_axis_limits(
            [0.344, 0.325, 0.316],
            [(1, 0.342), (11, 0.339)],
        )

        self.assertEqual(upper, 0.344)
        self.assertLess(lower, 0.316)
        self.assertGreater(lower, 0.30)

    def test_quiet_training_hides_autotuner_and_returns_complete_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "examples.jsonl"
            _write_dataset(dataset_path, count=20)
            output = io.StringIO()
            with mock.patch(
                "training.training_loop.CHECKPOINT_DIR",
                str(root / "checkpoints"),
            ), redirect_stdout(output):
                summary = train_supervised(
                    epochs=2,
                    batch_size=8,
                    dataset_file=dataset_path,
                    cache_file=root / "cache.npz",
                    weights_file=root / "weights.npz",
                    quiet=True,
                    device="cpu",
                    seed=5,
                    memory_reserve_mb=0,
                )
            self.assertEqual(output.getvalue(), "")
            self.assertEqual(summary["epochs"], 2)
            self.assertTrue(summary["training_plateau_enabled"])
            self.assertFalse(summary["training_plateau_stopped"])
            self.assertEqual(summary["stopping_reason"], "epoch_limit")
            self.assertEqual(summary["selected_device"], "cpu")
            self.assertEqual(summary["selected_batch_size"], 8)
            loss_plot = root / "weights_loss.png"
            self.assertEqual(summary["loss_plot_file"], str(loss_plot))
            self.assertEqual(loss_plot.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(len(summary["epoch_metrics"]), 2)
            self.assertTrue(all(
                isinstance(metrics["training_loss"], float)
                for metrics in summary["epoch_metrics"]
            ))
            with np.load(root / "weights.npz") as weights:
                self.assertEqual(set(weights.files), {"W1", "b1", "W2", "b2", "W3", "b3"})
                self.assertTrue(all(weights[name].dtype == np.float32 for name in weights.files))
            neural_agent = NeuralAgent.load(root / "weights.npz", device="cpu")
            rl_network = PolicyNetwork.load_from_sl(root / "weights.npz", device="cpu")
            self.assertEqual(neural_agent.network.W1.dtype, np.float32)
            self.assertEqual(rl_network.W1.dtype, np.float32)

    def test_real_cpu_supervised_run_stops_on_configured_training_plateau(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "examples.jsonl"
            _write_dataset(dataset_path, count=20)

            with mock.patch(
                "training.training_loop.CHECKPOINT_DIR",
                str(root / "checkpoints"),
            ):
                summary = train_supervised(
                    epochs=20,
                    batch_size=8,
                    dataset_file=dataset_path,
                    cache_file=root / "cache.npz",
                    weights_file=root / "weights.npz",
                    quiet=True,
                    device="cpu",
                    seed=7,
                    memory_reserve_mb=0,
                    training_plateau_window=2,
                    training_plateau_patience=2,
                    training_plateau_min_epochs=4,
                    training_plateau_min_relative_improvement=1.0,
                )

            self.assertEqual(summary["epochs"], 6)
            self.assertTrue(summary["training_plateau_stopped"])
            self.assertEqual(summary["stopping_reason"], "training_loss_plateau")
            self.assertEqual(summary["training_plateau_loss_start_epoch"], 1)
            self.assertTrue((root / "weights.npz").is_file())
            self.assertTrue((root / "weights_loss.png").is_file())

    def test_compact_pipeline_shows_autotuner_status_without_epoch_chatter(self):
        captured_kwargs = {}

        def fake_train_supervised(**kwargs):
            captured_kwargs.update(kwargs)
            kwargs["status_callback"](
                "Testing the optimal supervised batch size on GPU..."
            )
            kwargs["status_callback"](
                "Supervised batch test with 2,048 passed; median epoch "
                "0.100s; test total 1.0s; 100,000 examples/s (baseline); "
                "10 epochs retained."
            )
            kwargs["status_callback"](
                "Optimal supervised batch size: 2,048."
            )
            kwargs["progress_callback"](2, 2)
            return {
                "epochs": 2,
                "requested_epochs": 2,
                "best_validation_loss": 0.25,
                "total_examples": 20,
                "weights_file": "models/test.npz",
            }

        args = SimpleNamespace(
            sl_batch_size=None,
            weight_decay=0.0,
            early_stopping=None,
            lr_decay=0.5,
            lr_decay_patience=5,
            disable_training_plateau=False,
            sl_training_plateau_window=25,
            sl_training_plateau_patience=4,
            sl_training_plateau_min_epochs=100,
            sl_training_plateau_min_relative_improvement=0.001,
            sl_device="cpu",
            sl_no_batch_autotune=False,
            sl_memory_reserve_mb=512,
            sl_gpu_memory_reserve_mb=512,
            sl_seed=9,
        )
        output = io.StringIO()
        with mock.patch(
            "training.training_loop.train_supervised",
            side_effect=fake_train_supervised,
        ), mock.patch.object(run_pipeline, "tqdm", None), redirect_stdout(output):
            run_pipeline._run_supervised_training(
                SimpleNamespace(supervised_epochs=2),
                args,
            )
        self.assertTrue(captured_kwargs["quiet"])
        self.assertEqual(captured_kwargs["device"], "cpu")
        self.assertTrue(captured_kwargs["training_plateau_enabled"])
        self.assertEqual(captured_kwargs["training_plateau_window"], 25)
        self.assertIn(
            "2/2 epochs, best validation loss 0.2500, 20 examples",
            output.getvalue(),
        )
        self.assertIn("Testing the optimal supervised batch size", output.getvalue())
        self.assertIn("median epoch 0.100s", output.getvalue())
        self.assertIn("Optimal supervised batch size: 2,048", output.getvalue())
        self.assertNotIn("Checkpoint saved", output.getvalue())
        self.assertNotIn("validation loss:", output.getvalue())


@unittest.skipUnless(GPU_ENABLED, "a usable CuPy CUDA device is required")
class RealGPUResidencyTests(unittest.TestCase):
    def setUp(self):
        import cupy as cp

        try:
            probe = cp.zeros(1, dtype=cp.float32)
            cp.cuda.Stream.null.synchronize()
            del probe
        except Exception as exc:
            self.skipTest(f"CUDA is temporarily unavailable: {exc}")

    def test_full_gpu_residency_uploads_once_and_trains_float32(self):
        import cupy as cp

        x = np.arange(48, dtype=np.float32).reshape(4, 12)
        y = np.zeros((2, 12), dtype=np.float32)
        y[0] = 1
        try:
            with mock.patch("cupy.asarray", wraps=cp.asarray) as upload:
                plan = SupervisedDataPlan(
                    x,
                    y,
                    train_count=10,
                    host_storage_mode="ram",
                    device="gpu",
                    resident_capacity=12,
                )
                initial_upload_calls = upload.call_count
        except MemorySafetyError as exc:
            self.skipTest(f"CUDA became unavailable during allocation: {exc}")
        network = SupervisedNeuralNetwork(
            input_size=4,
            hidden1_size=5,
            hidden2_size=3,
            output_size=2,
            random_seed=3,
            device="gpu",
        )
        try:
            x_gpu_identity = id(plan.x_gpu)
            _loss, updates, rotations = plan.train_epoch(network, 4, 0)
            self.assertEqual(id(plan.x_gpu), x_gpu_identity)
            self.assertEqual(updates, 3)
            self.assertEqual(rotations, 0)
            self.assertTrue(plan.full_dataset_on_gpu)
            self.assertEqual(initial_upload_calls, 2)
            self.assertEqual(network.cache["A3"].dtype, cp.float32)
        finally:
            plan.close()

    def test_windowed_gpu_epoch_uses_every_index_once(self):
        x = np.arange(48, dtype=np.float32).reshape(4, 12)
        y = np.zeros((2, 12), dtype=np.float32)
        y[0] = 1
        observed = []
        try:
            plan = SupervisedDataPlan(
                x,
                y,
                train_count=10,
                host_storage_mode="ram",
                device="gpu",
                resident_capacity=4,
                index_observer=lambda indices: observed.extend(indices.tolist()),
            )
        except MemorySafetyError as exc:
            self.skipTest(f"CUDA became unavailable during allocation: {exc}")
        network = SupervisedNeuralNetwork(
            input_size=4,
            hidden1_size=5,
            hidden2_size=3,
            output_size=2,
            random_seed=2,
            device="gpu",
        )
        try:
            _loss, updates, rotations = plan.train_epoch(network, 2, 0)
            self.assertEqual(sorted(observed), list(range(10)))
            self.assertEqual(len(set(observed)), 10)
            self.assertEqual(rotations, 3)
            self.assertEqual(updates, 5)
            self.assertEqual(plan.resident_window_examples, 4)
        finally:
            plan.close()

    def test_real_gpu_residency_probe_changes_no_weights(self):
        x = np.zeros((168, 32), dtype=np.float32)
        y = np.zeros((56, 32), dtype=np.float32)
        y[0] = 1
        try:
            probe = probe_gpu_residency(x, y, reserve_mb=0)
        except Exception as exc:
            self.skipTest(f"CUDA became unavailable during probe: {exc}")
        self.assertEqual(probe.capacity_examples, 32)
        self.assertTrue(probe.full_dataset)
        self.assertTrue(probe.attempts[-1]["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
