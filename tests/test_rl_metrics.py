"""Contracts for the compact, versioned RL training-metrics trace."""

from __future__ import annotations

import json

from training.rl.reporting import (
    TRAINING_METRIC_COLUMNS,
    TRAINING_METRICS_VERSION,
    _prepare_metrics_file,
    _write_metrics_row,
    read_training_metrics,
)


def _row(iteration):
    return {
        "iteration": iteration,
        "total_iterations": 10,
        "games": 2_000,
        "cumulative_games": iteration * 2_000,
        "cumulative_training_games": iteration * 2_000,
        "total_training_games": 20_000,
        "games_per_iteration": 2_000,
        "decision_sample_count": 123,
        "decisions": 123,
        "wins_in_batch": 1_200,
        "batch_win_rate": 0.6,
        "moving_average_win_rate": 0.59,
        "reward_mean": 0.1,
        "reward_std": 0.2,
        "reward_min": -0.5,
        "reward_max": 0.5,
        "good_pct": 60.0,
        "neutral_pct": 0.0,
        "bad_pct": 40.0,
        "entropy": 2.1,
        "grad_norm": 0.75,
        "applied_grad_norm": 0.75,
        "grad_clipped": False,
        "value_loss": None,
        "moving_average_value_loss": None,
        "requested_minibatches": 8,
        "effective_minibatches": 8,
        "minibatch_sizes": [16, 16, 16, 15, 15, 15, 15, 15],
        "epochs_completed": 4,
        "stopped_by_kl": False,
        "optimizer_steps": 32,
        "final_approx_kl": 0.007123456789,
        "max_approx_kl": 0.008123456789,
        "final_clip_fraction": 0.11,
        "final_entropy": 2.1,
        "final_policy_loss": -0.03,
        "gradient_norm_mean": 0.5,
        "gradient_norm_max": 0.75,
        "buffer_location": "ram",
        "buffer_bytes": 10_000,
        "selected_workers": 4,
        "pool_size": 10,
        "rollout_seconds": 1.25,
        "ppo_seconds": 0.75,
        "rollout_duration_s": 1.25,
        "update_duration_s": 0.75,
        "total_iteration_seconds": 2.0,
        "iteration_duration_s": 2.0,
        "checkpoint_written": False,
        "checkpoint_path": None,
        "elapsed_training_s": 2.0 * iteration,
        "rl_training_algorithm": "ppo_v1",
    }


def test_metrics_v2_uses_one_header_and_compact_array_rows(tmp_path):
    path = tmp_path / "training_metrics.jsonl"
    metadata = {
        "run_configuration_sha256": "a" * 64,
        "run_configuration": {"seed": 42, "gpi": 2_000},
    }
    _prepare_metrics_file(path, 0, metadata)
    with open(path, "a", encoding="utf-8") as stream:
        _write_metrics_row(stream, _row(1))

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    assert isinstance(json.loads(raw_lines[0]), dict)
    assert isinstance(json.loads(raw_lines[1]), list)

    header, rows = read_training_metrics(path)
    assert header["version"] == TRAINING_METRICS_VERSION
    assert header["metadata"] == metadata
    assert len(rows) == 1
    assert rows[0]["final_approx_kl"] == 0.007123456789
    assert rows[0]["epochs_completed"] == 4
    assert rows[0]["minibatch_sizes"] == [16, 16, 16, 15, 15, 15, 15, 15]
    assert "cumulative_training_games" not in TRAINING_METRIC_COLUMNS
    assert "decision_sample_count" not in TRAINING_METRIC_COLUMNS
    assert "final_entropy" not in TRAINING_METRIC_COLUMNS


def test_metrics_resume_truncates_after_the_checkpoint_and_checks_hash(tmp_path):
    path = tmp_path / "training_metrics.jsonl"
    metadata = {"run_configuration_sha256": "b" * 64}
    _prepare_metrics_file(path, 0, metadata)
    with open(path, "a", encoding="utf-8") as stream:
        _write_metrics_row(stream, _row(1))
        _write_metrics_row(stream, _row(2))

    _prepare_metrics_file(path, 1, metadata)
    _header, rows = read_training_metrics(path)
    assert [row["iteration"] for row in rows] == [1]

    mismatched = {"run_configuration_sha256": "c" * 64}
    try:
        _prepare_metrics_file(path, 1, mismatched)
    except ValueError as exc:
        assert "configuration hash" in str(exc)
    else:
        raise AssertionError("Expected a configuration-hash mismatch")
