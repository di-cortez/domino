"""Contracts for the compact, versioned RL training-metrics trace."""

from __future__ import annotations

import json
from unittest import mock

from training.rl.reporting import (
    LEGAL_LOGIT_DEFICIT_WARNING,
    RLTrainingReporter,
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
        "opponent_count": 11,
        "unique_neural_opponent_count": 10,
        "bucket_results": [[800, 480, 320], [1_200, 720, 480]],
        "rollout_seconds": 1.25,
        "ppo_seconds": 0.75,
        "rollout_duration_s": 1.25,
        "update_duration_s": 0.75,
        "total_iteration_seconds": 2.0,
        "iteration_duration_s": 2.0,
        "checkpoint_written": False,
        "checkpoint_path": None,
        "elapsed_training_s": 2.0 * iteration,
        "rl_training_algorithm": "ppo_v2_decision_minibatches",
    }


def _metadata(configuration_hash):
    return {
        "run_configuration_sha256": configuration_hash,
        "run_configuration": {"seed": 42, "gpi": 2_000},
        "training": {
            "games_per_iteration": 2_000,
            "difficulty_weight": 0.5,
            "opponent_buckets": ["heuristic", "recent"],
        },
    }


def test_metrics_v5_uses_header_bucket_names_and_numeric_array_rows(tmp_path):
    path = tmp_path / "training_metrics.jsonl"
    metadata = _metadata("a" * 64)
    _prepare_metrics_file(path, 0, metadata)
    with open(path, "a", encoding="utf-8") as stream:
        _write_metrics_row(stream, _row(1))

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    raw_header = json.loads(raw_lines[0])
    raw_row = json.loads(raw_lines[1])
    assert isinstance(raw_header, dict)
    assert isinstance(raw_row, list)
    assert raw_header["bucket_results"] == {
        "bucket_order": ["heuristic", "recent"],
        "columns": ["games", "wins", "losses"],
        "nominal_uniform_budget": 1_000,
        "nominal_difficulty_budget": 1_000,
    }
    assert "heuristic" not in raw_lines[1]
    assert "recent" not in raw_lines[1]
    assert len(raw_lines[1].encode("utf-8")) < 1_000

    header, rows = read_training_metrics(path)
    assert header["version"] == TRAINING_METRICS_VERSION
    assert header["metadata"] == metadata
    assert len(rows) == 1
    assert rows[0]["final_approx_kl"] == 0.007123456789
    assert rows[0]["epochs_completed"] == 4
    assert rows[0]["minibatch_sizes"] == [16, 16, 16, 15, 15, 15, 15, 15]
    assert rows[0]["bucket_results"] == [
        [800, 480, 320],
        [1_200, 720, 480],
    ]
    assert "cumulative_training_games" not in TRAINING_METRIC_COLUMNS
    assert "decision_sample_count" not in TRAINING_METRIC_COLUMNS
    assert "final_entropy" not in TRAINING_METRIC_COLUMNS


def test_metrics_resume_truncates_after_the_checkpoint_and_checks_hash(tmp_path):
    path = tmp_path / "training_metrics.jsonl"
    metadata = _metadata("b" * 64)
    _prepare_metrics_file(path, 0, metadata)
    with open(path, "a", encoding="utf-8") as stream:
        _write_metrics_row(stream, _row(1))
        _write_metrics_row(stream, _row(2))

    with mock.patch(
        "training.rl.reporting.read_training_metrics",
        side_effect=AssertionError("resume must stream instead"),
    ):
        _prepare_metrics_file(path, 1, metadata)
    _header, rows = read_training_metrics(path)
    assert [row["iteration"] for row in rows] == [1]

    mismatched = _metadata("c" * 64)
    try:
        _prepare_metrics_file(path, 1, mismatched)
    except ValueError as exc:
        assert "configuration hash" in str(exc)
    else:
        raise AssertionError("Expected a configuration-hash mismatch")


def _report_iteration(reporter, iteration, ppo_window):
    """Drive one reporter iteration with the minimum a health check needs."""
    reporter.iteration(
        iteration=iteration,
        log_interval=10,
        games=2000,
        completed_games=iteration * 2000,
        total_games=None,
        reward_summary=None,
        wins=0,
        moving_win_rate=0.0,
        win_window_size=0,
        opponent_buckets=("heuristic",),
        difficulty_weight=0.5,
        opponent_count=1,
        unique_neural_opponent_count=0,
        opponent_pool_state=None,
        uniform_rotation_anchors=None,
        bucket_results={},
        gradient_metrics=None,
        use_value_head=False,
        value_loss_window=[],
        ppo_window=ppo_window,
        ppo_max_epochs=16,
        restart_summary=None,
    )


def test_logit_deficit_warning_survives_the_quiet_pipeline_reporter():
    """The rollout sampler's precursor must reach the operator too.

    ``policy_weight_max_abs`` stayed flat at 2.6 for the whole 28M-game run
    that died on an unsamplable decision, so the weight signal alone cannot
    warn about this failure. The deficit is the quantity that actually moved,
    and its threshold is the float32 limit rather than a tuned number.
    """
    captured = []
    reporter = RLTrainingReporter(quiet=True, status_callback=captured.append)
    base = {"rejected_minibatches": 0, "diverged_epoch": None}
    healthy = {**base, "legal_logit_deficit_max": 42.0}
    sharp = {
        **base,
        "legal_logit_deficit_max": LEGAL_LOGIT_DEFICIT_WARNING + 5.0,
    }

    _report_iteration(reporter, 1, [healthy])
    assert captured == []

    _report_iteration(reporter, 2, [healthy, sharp])
    assert len(captured) == 1
    assert "legal logit deficit" in captured[0]
    assert "not diverged" in captured[0]

    # A row without the key at all -- any pre-existing window entry -- is silent
    # rather than an AttributeError.
    _report_iteration(reporter, 3, [healthy, sharp, dict(base)])
    assert len(captured) == 1


def test_numerical_warnings_survive_the_quiet_pipeline_reporter():
    """The canonical pipeline runs the reporter quiet; a NaN must still speak.

    ``training/pipeline.py`` constructs the RL reporter with ``quiet=True``, so
    everything gated on ``log_interval`` is invisible in a ``forever`` run.
    A rejected minibatch or a rolled-back epoch has to reach the operator
    anyway, and it must not wait for the next log interval either.
    """
    captured = []
    reporter = RLTrainingReporter(quiet=True, status_callback=captured.append)
    healthy = {"rejected_minibatches": 0, "diverged_epoch": None}
    diverged = {
        "rejected_minibatches": 2,
        "diverged_epoch": 3,
        "divergence_reason": "PPO full-buffer metrics produced NaN/Inf.",
    }

    _report_iteration(reporter, 1, [healthy])
    assert captured == []

    # Iteration 2 is not a multiple of log_interval, so the ordinary report is
    # suppressed twice over: by quiet and by the interval. The warning is not.
    _report_iteration(reporter, 2, [healthy, diverged])
    assert len(captured) == 2
    assert "2 minibatch(es)" in captured[0]
    assert "rolled back diverged epoch 3" in captured[1]

    # An iteration that produced no update leaves the same row at the tail.
    _report_iteration(reporter, 3, [healthy, diverged])
    assert len(captured) == 2
