"""Cover the audit script's checkpoint resolution and its skip accounting.

The script's job is to be trusted unattended, so the risk it carries is not a
wrong check but a *missing* one that still prints like a pass. These tests pin
the denominator and the exit code for every way the pool half can be lost.
"""

import json

import pytest

from analysis.check_champion_vs_learner import (
    POOL_CHECK_COUNT,
    audit_pool,
    resolve_checkpoint_pair,
)
from training.rl.pool import CHAMPION_BUCKET_NAMES


def _pool_state():
    """A minimal well-formed resume state, enough to count the pool checks."""
    return {
        "opponent_pool_state": {
            "champion_state_by_bucket": {
                "champion_vs_heuristic": {
                    "completed_event_count": 1,
                    "pending_candidate_ids": [],
                    "heuristic_win_rate_by_opponent_id": {},
                },
                "champion_vs_learner": {
                    "completed_event_count": 1,
                    "pending_candidate_ids": [],
                },
            },
            "buckets": {
                name: {
                    "capacity": 200,
                    "member_ids": [f"snapshot:{index:010d}" for index in range(5)],
                }
                for name in CHAMPION_BUCKET_NAMES
            },
        }
    }


def test_pool_check_count_matches_what_audit_pool_produces():
    """``POOL_CHECK_COUNT`` is what a skipped run subtracts; keep it exact."""
    assert len(audit_pool(_pool_state())) == POOL_CHECK_COUNT


def _write_pair(directory, weights_name, state_name):
    weights = directory / weights_name
    state = directory / state_name
    weights.write_bytes(b"weights")
    state.write_bytes(b"state")
    return weights, state


def test_resolves_the_numbered_checkpoint_convention(tmp_path):
    weights, state = _write_pair(
        tmp_path, "training_iter008427.npz", "training_iter008427.resume.npz"
    )
    assert resolve_checkpoint_pair(weights) == (weights, state)


def test_resolves_the_canonical_generation_convention(tmp_path):
    """``_weights.npz``/``_state.npz`` is what a ``forever`` run publishes.

    It cannot be derived by appending ``.resume``, and skipping it silently
    dropped the pool half of every canonical run.
    """
    weights, state = _write_pair(
        tmp_path,
        "games_0016854000_latest_8befad7b2669_weights.npz",
        "games_0016854000_latest_8befad7b2669_state.npz",
    )
    assert resolve_checkpoint_pair(weights) == (weights, state)


def test_resolves_the_latest_alias_convention(tmp_path):
    weights, state = _write_pair(
        tmp_path, "latest_weights.npz", "latest.resume.npz"
    )
    assert resolve_checkpoint_pair(weights) == (weights, state)


def test_run_directory_resolves_through_training_state(tmp_path):
    """The directory form reads the pair the run itself resumes from."""
    states = tmp_path / "checkpoint_states"
    states.mkdir()
    weights, state = _write_pair(states, "generation_weights.npz", "generation_state.npz")
    (tmp_path / "training_state.json").write_text(
        json.dumps(
            {
                "latest_weights_path": "checkpoint_states/generation_weights.npz",
                "latest_resume_state_path": "checkpoint_states/generation_state.npz",
            }
        ),
        encoding="utf-8",
    )
    assert resolve_checkpoint_pair(tmp_path) == (weights, state)


def test_run_directory_prefers_training_state_over_a_stale_alias(tmp_path):
    """A stale alias must not win over the committed generation."""
    states = tmp_path / "checkpoint_states"
    states.mkdir()
    weights, state = _write_pair(states, "generation_weights.npz", "generation_state.npz")
    _write_pair(tmp_path, "latest_weights.npz", "latest.resume.npz")
    (tmp_path / "training_state.json").write_text(
        json.dumps(
            {
                "latest_weights_path": "checkpoint_states/generation_weights.npz",
                "latest_resume_state_path": "checkpoint_states/generation_state.npz",
            }
        ),
        encoding="utf-8",
    )
    assert resolve_checkpoint_pair(tmp_path) == (weights, state)


def test_directory_without_training_state_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a run directory"):
        resolve_checkpoint_pair(tmp_path)


def test_training_state_pointing_at_missing_files_is_an_error(tmp_path):
    (tmp_path / "training_state.json").write_text(
        json.dumps(
            {
                "latest_weights_path": "checkpoint_states/gone_weights.npz",
                "latest_resume_state_path": "checkpoint_states/gone_state.npz",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="points at files that are gone"):
        resolve_checkpoint_pair(tmp_path)


def test_orphan_weights_file_names_every_convention_it_tried(tmp_path):
    orphan = tmp_path / "orphan.npz"
    orphan.write_bytes(b"weights")
    with pytest.raises(FileNotFoundError) as error:
        resolve_checkpoint_pair(orphan)
    message = str(error.value)
    assert "orphan.resume.npz" in message
    assert "_state.npz" in message
    assert "latest.resume.npz" in message


def test_missing_path_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No such run directory"):
        resolve_checkpoint_pair(tmp_path / "absent.npz")


def _write_metrics(path, iterations=60):
    """A minimal two-champion-bucket trace the audit accepts."""
    from tests.test_rl_metrics import _row
    from training.rl.reporting import _prepare_metrics_file, _write_metrics_row

    buckets = [
        "heuristic",
        "recent",
        "champion_vs_heuristic",
        "champion_vs_learner",
    ]
    metadata = {
        "run_configuration_sha256": "b" * 64,
        "run_configuration": {"seed": 42, "gpi": 2_000},
        "training": {
            "games_per_iteration": 2_000,
            "difficulty_weight": 0.5,
            "opponent_buckets": buckets,
            "champion_evaluation": {
                "selected_targets": [
                    "champion_vs_heuristic",
                    "champion_vs_learner",
                ]
            },
        },
    }
    _prepare_metrics_file(path, 0, metadata)
    with open(path, "a", encoding="utf-8") as stream:
        for iteration in range(1, iterations + 1):
            row = _row(iteration)
            # Champion buckets stay empty until their first event completes.
            if iteration < 51:
                row["bucket_results"] = [
                    [1_000, 500, 500], [1_000, 500, 500], [0, 0, 0], [0, 0, 0],
                ]
            else:
                row["bucket_results"] = [
                    [500, 250, 250], [500, 250, 250],
                    [500, 250, 250], [500, 250, 250],
                ]
            _write_metrics_row(stream, row)


def test_main_reports_a_skipped_pool_half_on_stdout_and_exits_non_zero(
    tmp_path, capsys
):
    """The bug this replaced printed ``11/11 checks passed`` and exited 0."""
    from analysis.check_champion_vs_learner import main

    metrics = tmp_path / "training_metrics.jsonl"
    _write_metrics(metrics)

    exit_code = main(["prog", str(metrics)])
    captured = capsys.readouterr().out

    assert exit_code == 1
    assert f"{POOL_CHECK_COUNT} SKIPPED" in captured
    assert "SKIPPED the pool checks" in captured
    # The denominator must not shrink to hide the missing checks.
    assert "/20 checks passed" in captured
    assert "11/11" not in captured


def test_main_is_clean_and_silent_about_skips_when_the_pool_half_runs(
    tmp_path, capsys, monkeypatch
):
    from analysis import check_champion_vs_learner as module

    metrics = tmp_path / "training_metrics.jsonl"
    _write_metrics(metrics)
    weights, _state = _write_pair(
        tmp_path, "latest_weights.npz", "latest.resume.npz"
    )
    monkeypatch.setattr(
        module, "load_resume_state", lambda *_args: (_pool_state(), None)
    )

    exit_code = module.main(["prog", str(metrics), str(weights)])
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "SKIPPED" not in captured
    assert "20/20 checks passed" in captured
