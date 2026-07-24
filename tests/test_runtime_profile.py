"""Contracts for low-overhead RL and diagnostic runtime profiling."""

from __future__ import annotations

import json

import pytest

from agents.encoder import DominoEncoder
from agents.rl_nn import PolicyNetwork
from diagnostics.pairwise import run_pairwise
from diagnostics.rl_progress import (
    read_periodic_history,
    run_periodic_diagnostic,
)
from diagnostics.runtime_profile import RuntimeProfileRecorder
from diagnostics.parallel_runner import ParallelSafetyConfig
from training import self_play


def test_runtime_profile_accumulates_sessions_without_estimating_history(tmp_path):
    run_dir = tmp_path / "run"
    first = RuntimeProfileRecorder(
        run_dir,
        pipeline_level="forever",
        seed=42,
        start_rl_games=18_400_000,
    )
    first.record_rl(
        {
            "execution_count": 1,
            "games": 400,
            "iterations": 1,
            "decisions": 1600,
            "optimizer_steps": 12,
            "execution_seconds": 4.0,
            "sections_seconds": {"rollout_game_execution": 2.0, "ppo_update": 1.0},
            "ppo_sections_seconds": {"optimizer_steps": 0.5},
        },
        end_rl_games=18_400_400,
    )
    first.record_diagnostic(
        {
            "execution_count": 1,
            "reused_execution_count": 0,
            "games": 100_000,
            "execution_seconds": 10.0,
            "sections_seconds": {"pairwise_evaluation": 8.0},
            "pairwise_sections_seconds": {"new_game_execution": 7.0},
        },
        end_rl_games=18_400_400,
    )
    first.finish(status="interrupted", end_rl_games=18_400_400)

    second = RuntimeProfileRecorder(
        run_dir,
        pipeline_level="forever",
        seed=42,
        start_rl_games=18_400_400,
    )
    second.record_rl(
        {
            "execution_count": 1,
            "games": 600,
            "iterations": 2,
            "execution_seconds": 6.0,
            "sections_seconds": {"rollout_game_execution": 3.0, "ppo_update": 2.0},
            "ppo_sections_seconds": {"optimizer_steps": 1.0},
        },
        end_rl_games=18_401_000,
    )
    second.finish(status="completed", end_rl_games=18_401_000)

    report = json.loads(second.path.read_text(encoding="utf-8"))
    assert report["coverage"]["unprofiled_rl_games_before_first_profile"] == 18_400_000
    assert report["coverage"]["profiled_rl_games"] == 1000
    assert report["cumulative"]["rl"]["execution_count"] == 2
    assert report["cumulative"]["rl"]["games"] == 1000
    assert report["cumulative"]["rl"]["sections_seconds"]["ppo_update"] == 3.0
    assert report["cumulative"]["rl_vs_random_diagnostics"]["games"] == 100_000
    assert report["derived"]["rl"]["games_per_second"] == 100.0
    assert len(report["sessions"]) == 2
    assert [row["status"] for row in report["sessions"]] == [
        "interrupted",
        "completed",
    ]


def test_pairwise_profile_accounts_for_game_and_artifact_phases(tmp_path):
    result = run_pairwise(
        "random",
        "random",
        game_count=4,
        output_dir=tmp_path / "pair",
        generate_plots=False,
        print_console_summary=False,
        workers=1,
        safety_config=ParallelSafetyConfig(memory_reserve_mb=0),
        seed=7,
    )
    profile = result["runtime_profile_delta"]
    assert profile["games"] == 4
    assert profile["new_games"] == 4
    assert profile["precomputed_games"] == 0
    assert profile["sections_seconds"]["new_game_execution"] > 0.0
    assert profile["sections_seconds"]["games_csv_write"] > 0.0
    worker = profile["game_worker"]
    assert worker["games"] == 4
    assert worker["profiled_games"] == 1
    assert worker["worker_cpu_seconds"] > 0.0
    assert worker["sections_seconds"]["state_and_legal_action_generation"] > 0.0
    assert worker["sections_seconds"]["evaluated_agent_decisions"] > 0.0
    assert sum(worker["sections_seconds"].values()) == pytest.approx(
        worker["profiled_game_cpu_seconds"], rel=1e-6, abs=1e-6
    )
    assert sum(profile["sections_seconds"].values()) == pytest.approx(
        profile["execution_seconds"], rel=1e-6, abs=1e-6
    )


def test_pairwise_can_omit_persisted_game_records(tmp_path):
    output = tmp_path / "compact_pair"
    run_pairwise(
        "random",
        "random",
        game_count=4,
        output_dir=output,
        generate_plots=False,
        print_console_summary=False,
        workers=1,
        safety_config=ParallelSafetyConfig(memory_reserve_mb=0),
        seed=8,
        save_game_records=False,
    )
    assert (output / "summary.json").is_file()
    assert not (output / "games.csv").exists()


def test_self_play_profile_contains_rollout_and_nested_ppo_phases(tmp_path):
    supervised = tmp_path / "supervised.npz"
    PolicyNetwork(
        input_size=DominoEncoder.VECTOR_SIZE,
        hidden1_size=16,
        hidden2_size=8,
        output_size=DominoEncoder.ACTION_SIZE,
        device="cpu",
    ).save(supervised)
    result = self_play.train(
        iterations=1,
        gpi=8,
        workers=1,
        device="cpu",
        seed=123,
        fresh_from_sl=True,
        sl_weights_path=supervised,
        rl_weights_path=tmp_path / "rl.npz",
        metrics_output_path=tmp_path / "metrics.jsonl",
        adaptive_tuning_path=tmp_path / "tuning.json",
        checkpoint_interval=1,
        quiet=True,
    )

    profile = result["runtime_profile_delta"]
    assert profile["games"] == 8
    assert profile["iterations"] == 1
    assert profile["decisions"] > 0
    assert profile["optimizer_steps"] > 0
    assert profile["sections_seconds"]["rollout_game_execution"] > 0.0
    assert profile["sections_seconds"]["ppo_buffer_assembly_and_advantage_normalization"] > 0.0
    assert profile["sections_seconds"]["ppo_update"] > 0.0
    assert profile["ppo_sections_seconds"]["optimizer_steps"] > 0.0
    rollout_worker = profile["rollout_worker"]
    assert rollout_worker["games"] == 8
    assert rollout_worker["profiled_games"] == 1
    assert rollout_worker["worker_cpu_seconds"] > 0.0
    assert rollout_worker["sections_seconds"]["learner_agent_decisions"] > 0.0
    assert rollout_worker["learner_policy"]["sections_seconds"][
        "exact_opponent_model_update"
    ] > 0.0
    assert sum(rollout_worker["sections_seconds"].values()) == pytest.approx(
        rollout_worker["profiled_game_cpu_seconds"], rel=1e-6, abs=1e-6
    )
    optimizer = profile["ppo_optimizer_step"]
    assert optimizer["calls"] == profile["optimizer_steps"]
    assert optimizer["cpu_calls"] == profile["optimizer_steps"]
    assert optimizer["gpu_calls"] == 0
    assert optimizer["sections_seconds"][
        "clipped_surrogate_backpropagation_and_gradient_norm"
    ] > 0.0
    assert sum(optimizer["sections_seconds"].values()) == pytest.approx(
        optimizer["execution_seconds"], rel=1e-6, abs=1e-6
    )
    full_buffer = profile["ppo_full_buffer_evaluation"]
    assert full_buffer["calls"] >= result["completed_iterations_this_run"]
    assert full_buffer["cpu_calls"] == full_buffer["calls"]
    assert full_buffer["sections_seconds"][
        "surrogate_metric_reductions_and_host_transfers"
    ] > 0.0
    assert sum(profile["sections_seconds"].values()) == pytest.approx(
        profile["execution_seconds"], rel=1e-6, abs=1e-6
    )


def test_periodic_profile_separates_reports_from_pairwise_work(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.npz"
    checkpoint.write_bytes(b"profile-test-checkpoint")
    pairwise_options = {}

    def fake_pairwise(*_args, **kwargs):
        pairwise_options.update(kwargs)
        return {
            "summary": {
                "counts": {"win": 2, "draw": 1, "loss": 1},
                "win_ci95": [0.10, 0.90],
            },
            "runtime_profile_delta": {
                "execution_seconds": 0.01,
                "games": 4,
                "new_games": 4,
                "precomputed_games": 0,
                "sections_seconds": {"new_game_execution": 0.008},
            },
        }

    monkeypatch.setattr("diagnostics.rl_progress.run_pairwise", fake_pairwise)
    monkeypatch.setattr(
        "diagnostics.rl_progress.rebuild_progress_plot",
        lambda run_dir, **_kwargs: run_dir / "rl_vs_random_progress.png",
    )
    row, appended = run_periodic_diagnostic(
        run_dir=tmp_path,
        pipeline_level="forever",
        seed=42,
        rl_games=100,
        rl_iterations=1,
        optimizer_steps=12,
        checkpoint_path=checkpoint,
        diagnostic_games=4,
        rl_elapsed_seconds=1.0,
        wall_clock_seconds=2.0,
        workers=1,
        safety_config=ParallelSafetyConfig(memory_reserve_mb=0),
    )

    assert appended
    profile = row["runtime_profile_delta"]
    assert profile["games"] == 4
    assert profile["sections_seconds"]["pairwise_evaluation"] > 0.0
    assert profile["sections_seconds"]["history_jsonl_append_and_fsync"] > 0.0
    assert profile["sections_seconds"]["progress_csv_rebuild"] > 0.0
    assert profile["pairwise_sections_seconds"] == {"new_game_execution": 0.008}
    assert pairwise_options["save_game_records"] is False
    persisted = read_periodic_history(tmp_path / "periodic_diagnostics.jsonl")
    assert "runtime_profile_delta" not in persisted[-1]
