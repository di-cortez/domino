"""Contracts for canonical supervised assets and game-budgeted RL runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agents.rl_nn import PolicyNetwork
from diagnostics.parallel_runner import ParallelSafetyConfig
from diagnostics.rl_progress import (
    PERIODIC_SUMMARY_RETENTION,
    _rl_elapsed_hours,
    append_periodic_point,
    final_diagnostic_seed,
    periodic_diagnostic_seed,
    prune_periodic_diagnostic_artifacts,
    read_periodic_history,
    rebuild_progress_reports,
)
from training import self_play
from training.canonical_assets import (
    ArtifactCompatibilityError,
    EXPECTED_WEIGHT_SHAPES,
    canonical_asset_paths,
    canonical_generation_config,
    canonical_training_config,
    inspect_canonical_dataset,
    inspect_canonical_weights,
    run_scoped_asset_paths,
    write_dataset_metadata,
    write_weights_metadata,
)
from training.canonical_run import (
    MILESTONE_RESUME_RETENTION,
    _prune_milestone_resume_states,
    canonical_run_dir,
    create_run_config,
    load_resume_point,
    publish_checkpoint,
)
from training.rl_resume import (
    NUMBERED_CHECKPOINT_WEIGHT_RETENTION,
    _prune_numbered_checkpoint_weights,
)
from training.pipeline import (
    PERIODIC_DIAGNOSTIC_TUNING_FILE,
    PIPELINE_LEVELS,
    _cumulative_rl_games_per_second,
    _resolve_periodic_diagnostic_workers,
    _run_periodic_point,
    next_training_stop,
    parse_args,
    validate_args,
)
from training.rl_resume import (
    LEGACY_TRAINING_ALGORITHM,
    PPO_TRAINING_ALGORITHM,
)
from utils.artifacts import file_sha256


ROOT = Path(__file__).resolve().parents[1]


def _write_test_supervised_checkpoint(path, seed=123):
    network = PolicyNetwork(random_seed=seed, device="cpu")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{
            name: np.asarray(getattr(network, name))
            for name in ("W1", "b1", "W2", "b2", "W3", "b3")
        },
    )
    return path


def _generation_config(dataset_games=3):
    return canonical_generation_config(
        dataset_games=dataset_games,
        workers=1,
        tuning={"fraction": 0.01, "minimum_gain": 0.10},
        safety={
            "memory_reserve_mb": 0,
            "estimated_worker_mb": 1,
            "max_worker_rss_mb": 1024,
        },
    )


def _training_config():
    return canonical_training_config(max_epochs=10, batch_size=32)


def _periodic_row(games, checkpoint_hash="a", diagnostic_seed=7):
    return {
        "format_version": 1,
        "pipeline_level": "big",
        "seed": 42,
        "rl_games": games,
        "rl_iterations": games // 100,
        "optimizer_steps": games // 10,
        "checkpoint_path": f"checkpoint-{games}.npz",
        "checkpoint_sha256": checkpoint_hash,
        "opponent": "random",
        "diagnostic_games": 100,
        "wins": 60,
        "draws": 5,
        "losses": 35,
        "win_rate": 0.60,
        "draw_rate": 0.05,
        "loss_rate": 0.35,
        "score": 0.625,
        "ci95_win_rate_low": 0.50,
        "ci95_win_rate_high": 0.69,
        "diagnostic_seed": diagnostic_seed,
        "diagnostic_seconds": 1.25,
        "rl_elapsed_seconds": games / 100.0,
        "wall_clock_seconds": 3.5,
        "selected_workers": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
    }


def test_quick_and_long_run_level_policies_are_distinct(monkeypatch):
    seeds = iter((101, 202))
    tokens = iter(("aaaaaaaa", "bbbbbbbb", "cccccccc"))
    monkeypatch.setattr(
        "training.pipeline.secrets.randbits",
        lambda bits: next(seeds) if bits == 32 else None,
    )
    monkeypatch.setattr(
        "training.pipeline.secrets.token_hex",
        lambda bytes_count: next(tokens) if bytes_count == 4 else None,
    )

    first_quick = parse_args(["default"])
    second_quick = parse_args(["default"])
    explicit_quick = parse_args(["small", "--seed", "7"])
    long_run = parse_args(["big"])

    assert first_quick.seed == 101
    assert second_quick.seed == 202
    assert first_quick.execution_id != second_quick.execution_id
    assert explicit_quick.seed == 7
    assert explicit_quick.execution_id is not None
    assert long_run.seed == 42
    assert long_run.execution_id is None
    assert PIPELINE_LEVELS["small"].dataset_games == 10_000
    assert PIPELINE_LEVELS["default"].dataset_games == 50_000
    assert PIPELINE_LEVELS["big"].dataset_games == 100_000
    assert PIPELINE_LEVELS["huge"].dataset_games == 100_000
    assert PIPELINE_LEVELS["forever"].dataset_games == 100_000
    assert not PIPELINE_LEVELS["small"].reuse_supervised_assets
    assert not PIPELINE_LEVELS["default"].reuse_supervised_assets
    assert PIPELINE_LEVELS["big"].reuse_supervised_assets
    assert PIPELINE_LEVELS["small"].supervised_epochs == 5_000
    assert PIPELINE_LEVELS["default"].supervised_epochs == 5_000
    assert PIPELINE_LEVELS["big"].supervised_epochs == 5_000
    assert PIPELINE_LEVELS["huge"].supervised_epochs == 5_000
    assert PIPELINE_LEVELS["forever"].supervised_epochs == 5_000
    assert PIPELINE_LEVELS["small"].total_rl_games == 100_000
    assert PIPELINE_LEVELS["default"].total_rl_games == 500_000
    assert PIPELINE_LEVELS["big"].total_rl_games == 2_000_000
    assert PIPELINE_LEVELS["huge"].total_rl_games == 10_000_000
    assert PIPELINE_LEVELS["forever"].total_rl_games is None
    assert PIPELINE_LEVELS["big"].diagnostic_games == 1_000_000
    assert PIPELINE_LEVELS["small"].diagnostic_games == 10_000


def test_resume_accepts_default_or_explicit_run_directory():
    automatic = parse_args(["forever", "--resume"])
    assert automatic.resume is True
    assert automatic.resume_from is None

    explicit = parse_args([
        "forever",
        "--resume",
        "models/rl/domino_rl_forever_seed42",
    ])
    assert explicit.resume is False
    assert explicit.resume_from == Path(
        "models/rl/domino_rl_forever_seed42"
    )


def test_canonical_pipeline_accepts_policy_only_reinforce():
    ppo = parse_args(["forever"])
    reinforce = parse_args(["forever", "--no-ppo"])

    validate_args(ppo, PIPELINE_LEVELS["forever"])
    validate_args(reinforce, PIPELINE_LEVELS["forever"])
    assert ppo.ppo_enabled is True
    assert reinforce.ppo_enabled is False

    critic = parse_args(["forever", "--no-ppo", "--value-head"])
    with pytest.raises(ValueError, match="remain policy-only"):
        validate_args(critic, PIPELINE_LEVELS["forever"])


def test_forever_periodic_workers_are_recovered_once_and_then_persisted(tmp_path):
    args = parse_args([
        "forever",
        "--periodic-diagnostic-games",
        "100",
    ])
    history = tmp_path / "periodic_diagnostics.jsonl"
    first = _periodic_row(
        0,
        diagnostic_seed=periodic_diagnostic_seed(args.seed),
    )
    first.update(
        pipeline_level="forever",
        diagnostic_games=100,
        selected_workers=10,
    )
    latest = _periodic_row(
        100_000,
        checkpoint_hash="b",
        diagnostic_seed=periodic_diagnostic_seed(args.seed),
    )
    latest.update(
        pipeline_level="forever",
        diagnostic_games=100,
        selected_workers=8,
    )
    append_periodic_point(history, first)
    append_periodic_point(history, latest)

    workers, source = _resolve_periodic_diagnostic_workers(
        tmp_path,
        "forever",
        args,
    )
    assert workers == 8
    assert source == "recovered forever selection"
    assert (tmp_path / PERIODIC_DIAGNOSTIC_TUNING_FILE).is_file()

    latest["selected_workers"] = 6
    latest["checkpoint_sha256"] = "c"
    append_periodic_point(history, latest)
    workers, source = _resolve_periodic_diagnostic_workers(
        tmp_path,
        "forever",
        args,
    )
    assert workers == 8
    assert source == "saved forever selection"


def test_new_forever_run_autotunes_periodic_workers_only_once(tmp_path, monkeypatch):
    args = parse_args([
        "forever",
        "--periodic-diagnostic-games",
        "100",
    ])
    worker_requests = []

    def fake_diagnostic(**kwargs):
        worker_requests.append(kwargs["workers"])
        row = _periodic_row(
            kwargs["rl_games"],
            checkpoint_hash="checkpoint-hash",
            diagnostic_seed=periodic_diagnostic_seed(args.seed),
        )
        row.update(
            pipeline_level="forever",
            diagnostic_games=100,
            selected_workers=8,
        )
        return row, True

    monkeypatch.setattr(
        "training.pipeline.run_periodic_diagnostic",
        fake_diagnostic,
    )
    common = {
        "args": args,
        "run_dir": tmp_path,
        "level": "forever",
        "checkpoint": tmp_path / "weights.npz",
        "iterations": 0,
        "optimizer_steps": 0,
        "elapsed_rl_seconds": 0.0,
        "pipeline_started": 0.0,
    }
    _run_periodic_point(games=0, **common)
    _run_periodic_point(games=100_000, **common)
    assert worker_requests == ["auto", 8]


def test_rl_throughput_is_cumulative_across_resume_segments():
    assert _cumulative_rl_games_per_second(6_600_000, 6_500.0, 100.0) == 1_000.0


def test_canonical_paths_and_run_directory_include_seed(tmp_path):
    paths = canonical_asset_paths(tmp_path, 42)
    assert paths.dataset.name == "supervised_dataset_standard_seed42.jsonl"
    assert paths.dataset_meta.name == "supervised_dataset_standard_seed42.meta.json"
    assert paths.weights.name == "domino_sl_standard_seed42.npz"
    assert paths.weights_meta.name == "domino_sl_standard_seed42.meta.json"
    assert canonical_run_dir(tmp_path, "big", 42).name == "domino_rl_big_seed42"
    scoped_run = canonical_run_dir(
        tmp_path,
        "small",
        7,
        execution_id="20260723T120000-abcd1234",
    )
    assert scoped_run.name == (
        "domino_rl_small_seed7_run20260723T120000-abcd1234"
    )
    scoped_assets = run_scoped_asset_paths(scoped_run)
    assert scoped_assets.dataset == (
        scoped_run / "supervised" / "supervised_dataset.jsonl"
    )
    assert scoped_assets.weights == scoped_run / "supervised" / "domino_sl.npz"


def test_canonical_asset_hashes_and_metadata_control_reuse(tmp_path):
    paths = canonical_asset_paths(tmp_path, 42)
    paths.dataset.parent.mkdir(parents=True)
    paths.dataset.write_text('{"state": {}, "action": [[0, 0], 0]}\n', encoding="utf-8")
    generation = _generation_config()
    dataset_meta = write_dataset_metadata(
        paths,
        root=tmp_path,
        seed=42,
        dataset_games=3,
        dataset_summary={"saved_turn_count": 1},
        generation_config=generation,
    )
    check = inspect_canonical_dataset(
        paths,
        seed=42,
        dataset_games=3,
        generation_config=generation,
    )
    assert check.compatible

    paths.weights.parent.mkdir(parents=True)
    np.savez(
        paths.weights,
        **{
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in EXPECTED_WEIGHT_SHAPES.items()
        },
    )
    training = _training_config()
    write_weights_metadata(
        paths,
        root=tmp_path,
        seed=42,
        dataset_sha256=dataset_meta["dataset_sha256"],
        training_config=training,
        training_summary={
            "requested_epochs": 10,
            "epochs": 4,
            "best_epoch": 3,
            "best_validation_loss": 0.5,
            "early_stopping_triggered": True,
            "stopping_reason": "training_loss_plateau",
            "final_training_loss": 0.4,
            "final_validation_loss": 0.6,
        },
    )
    assert inspect_canonical_weights(
        paths,
        seed=42,
        dataset_sha256=dataset_meta["dataset_sha256"],
        training_config=training,
    ).compatible

    paths.dataset.write_text("tampered\n", encoding="utf-8")
    incompatible = inspect_canonical_dataset(
        paths,
        seed=42,
        dataset_games=3,
        generation_config=generation,
    )
    assert incompatible.status == "incompatible"
    with pytest.raises(ArtifactCompatibilityError, match="dataset_sha256"):
        incompatible.require_compatible_or_missing(
            rebuild=False,
            label="supervised dataset",
        )
    incompatible.require_compatible_or_missing(
        rebuild=True,
        label="supervised dataset",
    )


def test_exact_milestone_boundary_never_rounds_up():
    assert next_training_stop(99_800, 2_000_000, 100_000, True) == 100_000
    assert next_training_stop(1_999_800, 2_000_000, 100_000, True) == 2_000_000
    assert next_training_stop(12_700_000, None, 100_000, True) == 12_800_000
    assert next_training_stop(0, 500_000, 100_000, False) == 500_000


def test_run_config_is_stable_and_target_extension_must_be_explicit(tmp_path):
    run_dir = canonical_run_dir(tmp_path, "big", 42)
    values = {
        "root": ROOT,
        "pipeline_level": "big",
        "seed": 42,
        "target_rl_games": 2_000_000,
        "supervised_weights_path": "models/sl.npz",
        "supervised_weights_sha256": "abc",
        "ppo_config": {"clip_epsilon": 0.2},
        "rl_config": {"gamma": 1.0},
    }
    first = create_run_config(run_dir, **values)
    second = create_run_config(run_dir, **values)
    assert second["created_at"] == first["created_at"]
    assert first["algorithm"] == PPO_TRAINING_ALGORITHM

    with pytest.raises(ValueError, match="algorithm"):
        create_run_config(
            run_dir,
            **values,
            algorithm=LEGACY_TRAINING_ALGORITHM,
        )

    extended = dict(values)
    extended.update(pipeline_level="huge", target_rl_games=10_000_000)
    with pytest.raises(ValueError, match="target_rl_games"):
        create_run_config(run_dir, **extended)
    updated = create_run_config(
        run_dir,
        **extended,
        allow_target_extension=True,
        lineage=[{"source_run_dir": str(run_dir)}],
    )
    assert updated["created_at"] == first["created_at"]
    assert updated["target_rl_games"] == 10_000_000
    assert updated["pipeline_level"] == "huge"


def test_canonical_reinforce_resume_matches_uninterrupted_training(tmp_path):
    supervised = _write_test_supervised_checkpoint(tmp_path / "sl.npz")
    supervised_hash = file_sha256(supervised)
    safety = ParallelSafetyConfig(
        memory_reserve_mb=0,
        estimated_worker_mb=1,
        max_worker_rss_mb=1024,
    )
    common = {
        "total_training_games": 4,
        "games_per_iteration": 2,
        "adaptive_gpi": False,
        "checkpoint_interval": 1,
        "pool_refresh_games": 2,
        "max_pool_size": 2,
        "sl_weights_path": str(supervised),
        "seed": 321,
        "device": "cpu",
        "workers": 1,
        "safety_config": safety,
        "quiet": True,
        "numbered_checkpoints": True,
        "ppo_enabled": False,
        "use_value_head": False,
    }
    uninterrupted = self_play.train(
        rl_weights_path=str(tmp_path / "uninterrupted" / "training.npz"),
        fresh_from_sl=True,
        **common,
    )
    partial = self_play.train(
        rl_weights_path=str(tmp_path / "split" / "training.npz"),
        stop_after_training_games=2,
        fresh_from_sl=True,
        **common,
    )

    profile = partial["runtime_profile_delta"]
    assert "legacy_policy_update" in profile["sections_seconds"]
    assert "ppo_update" not in profile["sections_seconds"]
    assert profile["ppo_sections_seconds"] == {}

    run_dir = canonical_run_dir(tmp_path, "forever", 321)
    create_run_config(
        run_dir,
        root=ROOT,
        pipeline_level="forever",
        seed=321,
        target_rl_games=None,
        supervised_weights_path=supervised,
        supervised_weights_sha256=supervised_hash,
        ppo_config=partial["ppo_configuration"],
        rl_config={"normalize_advantages": False},
        algorithm=LEGACY_TRAINING_ALGORITHM,
    )
    state = publish_checkpoint(
        run_dir,
        root=ROOT,
        pipeline_level="forever",
        seed=321,
        target_rl_games=None,
        supervised_weights_path=supervised,
        supervised_weights_sha256=supervised_hash,
        summary=partial,
        last_periodic_diagnostic_game=0,
        next_periodic_diagnostic_game=100_000,
    )
    assert state["algorithm"] == LEGACY_TRAINING_ALGORITHM
    assert state["policy_updates_completed"] == 1
    assert state["ppo_updates_completed"] == 0
    assert state["reinforce_updates_completed"] == 1

    point = load_resume_point(
        run_dir,
        seed=321,
        supervised_weights_sha256=supervised_hash,
        ppo_config=partial["ppo_configuration"],
        algorithm=LEGACY_TRAINING_ALGORITHM,
    )
    with pytest.raises(ValueError, match="algorithm"):
        load_resume_point(
            run_dir,
            seed=321,
            supervised_weights_sha256=supervised_hash,
            ppo_config=partial["ppo_configuration"],
            algorithm=PPO_TRAINING_ALGORITHM,
        )

    resumed = self_play.train(
        rl_weights_path=str(tmp_path / "split" / "training.npz"),
        start_iteration=point.completed_iterations,
        resume_weights_path=str(point.weights_path),
        resume_state_file=str(point.resume_state_path),
        **common,
    )
    with np.load(uninterrupted["rl_weights_path"], allow_pickle=False) as left:
        with np.load(resumed["rl_weights_path"], allow_pickle=False) as right:
            assert left.files == right.files
            for name in left.files:
                np.testing.assert_array_equal(left[name], right[name])


def test_periodic_and_final_seed_namespaces_are_separate_and_stable():
    assert periodic_diagnostic_seed(42) == periodic_diagnostic_seed(42)
    assert final_diagnostic_seed(42) == final_diagnostic_seed(42)
    assert periodic_diagnostic_seed(42) != final_diagnostic_seed(42)


def test_jsonl_repairs_partial_tail_deduplicates_and_rebuilds_reports(tmp_path):
    history = tmp_path / "periodic_diagnostics.jsonl"
    first = _periodic_row(0, checkpoint_hash="zero")
    _row, appended = append_periodic_point(history, first)
    assert appended
    _row, appended = append_periodic_point(history, first)
    assert not appended
    with open(history, "a", encoding="utf-8") as stream:
        stream.write('{"partial":')
    assert read_periodic_history(history) == [first]

    second = _periodic_row(100_000, checkpoint_hash="one")
    _row, appended = append_periodic_point(history, second)
    assert appended
    assert read_periodic_history(history) == [first, second]
    csv_path, plot_path, _log_path = rebuild_progress_reports(tmp_path)
    assert csv_path.is_file()
    assert plot_path.is_file()
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 3
    assert "rl_elapsed_hours" in csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert _rl_elapsed_hours(second) == pytest.approx(1000.0 / 3600.0)


def test_periodic_artifact_retention_drops_games_and_keeps_ten_summaries(tmp_path):
    diagnostics_dir = tmp_path / "diagnostics"
    for index in range(PERIODIC_SUMMARY_RETENTION + 3):
        point = diagnostics_dir / f"games_{index * 100_000:010d}"
        point.mkdir(parents=True)
        (point / "games.csv").write_text("game,result\n", encoding="utf-8")
        (point / "summary.json").write_text("{}\n", encoding="utf-8")

    removed = prune_periodic_diagnostic_artifacts(tmp_path)
    remaining = sorted(path.name for path in diagnostics_dir.iterdir())
    assert removed == {
        "games_csv_removed": PERIODIC_SUMMARY_RETENTION + 3,
        "summary_json_removed": 3,
        "directories_removed": 3,
    }
    assert len(remaining) == PERIODIC_SUMMARY_RETENTION
    assert not list(diagnostics_dir.rglob("games.csv"))
    assert len(list(diagnostics_dir.rglob("summary.json"))) == 10


def test_checkpoint_history_retention_keeps_only_five_recent_states(tmp_path):
    state_dir = tmp_path / "checkpoint_states"
    state_dir.mkdir()
    for index in range(MILESTONE_RESUME_RETENTION + 3):
        games = (index + 1) * 100_000
        (state_dir / f"games_{games:010d}_state.npz").write_bytes(b"state")
        (state_dir / f"games_{games:010d}_state.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
    latest_generation = state_dir / "games_0000800000_latest_hash_state.npz"
    latest_generation.write_bytes(b"latest")

    _prune_milestone_resume_states(tmp_path)
    milestone_states = sorted(
        path
        for path in state_dir.glob("games_*_state.npz")
        if "_latest_" not in path.name
    )
    milestone_metadata = sorted(state_dir.glob("games_*_state.json"))
    assert len(milestone_states) == MILESTONE_RESUME_RETENTION
    assert len(milestone_metadata) == MILESTONE_RESUME_RETENTION
    assert latest_generation.is_file()


def test_numbered_policy_checkpoint_retention_keeps_only_five(tmp_path):
    base = tmp_path / "training.npz"
    checkpoints = []
    for iteration in range(1, NUMBERED_CHECKPOINT_WEIGHT_RETENTION + 4):
        path = tmp_path / f"training_iter{iteration:06d}.npz"
        path.write_bytes(b"weights")
        checkpoints.append(path)

    _prune_numbered_checkpoint_weights(base, checkpoints[-1])
    assert sorted(tmp_path.glob("training_iter*.npz")) == checkpoints[-5:]


@pytest.mark.skipif(
    not (ROOT / "models" / "domino_sl_weights.npz").is_file(),
    reason="supervised smoke checkpoint is unavailable",
)
def test_canonical_checkpoint_is_complete_and_alias_damage_does_not_break_resume(
    tmp_path,
):
    supervised = ROOT / "models" / "domino_sl_weights.npz"
    summary = self_play.train(
        total_training_games=3,
        stop_after_training_games=2,
        games_per_iteration=2,
        adaptive_gpi=False,
        checkpoint_interval=1,
        pool_refresh_games=2,
        max_pool_size=2,
        sl_weights_path=str(supervised),
        rl_weights_path=str(tmp_path / "raw" / "training.npz"),
        seed=42,
        device="cpu",
        workers=1,
        safety_config=ParallelSafetyConfig(
            memory_reserve_mb=0,
            estimated_worker_mb=1,
            max_worker_rss_mb=1024,
        ),
        quiet=True,
        numbered_checkpoints=True,
        fresh_from_sl=True,
    )
    run_dir = canonical_run_dir(tmp_path, "big", 42)
    supervised_hash = file_sha256(supervised)
    create_run_config(
        run_dir,
        root=ROOT,
        pipeline_level="big",
        seed=42,
        target_rl_games=3,
        supervised_weights_path=supervised,
        supervised_weights_sha256=supervised_hash,
        ppo_config=summary["ppo_configuration"],
        rl_config={"test": True},
    )
    state = publish_checkpoint(
        run_dir,
        root=ROOT,
        pipeline_level="big",
        seed=42,
        target_rl_games=3,
        supervised_weights_path=supervised,
        supervised_weights_sha256=supervised_hash,
        summary=summary,
        last_periodic_diagnostic_game=0,
        next_periodic_diagnostic_game=100_000,
        milestone=True,
    )
    point = load_resume_point(
        run_dir,
        seed=42,
        supervised_weights_sha256=supervised_hash,
        ppo_config=summary["ppo_configuration"],
    )
    assert point.completed_games == 2
    assert point.completed_iterations == 1
    assert state["ppo_updates_completed"] == 1
    assert Path(run_dir / state["opponent_pool_manifest"]).is_file()
    assert (run_dir / "latest_weights.npz").is_file()
    assert (run_dir / "optimizer_state.npz").is_file()

    # Resume follows the immutable generation named in training_state.json,
    # not the post-commit convenience alias.
    (run_dir / "latest_weights.npz").write_bytes(b"damaged alias")
    assert load_resume_point(
        run_dir,
        seed=42,
        supervised_weights_sha256=supervised_hash,
        ppo_config=summary["ppo_configuration"],
    ).completed_games == 2


@pytest.mark.skipif(
    not (ROOT / "models" / "domino_sl_weights.npz").is_file(),
    reason="supervised smoke checkpoint is unavailable",
)
def test_shutdown_before_first_iteration_still_creates_a_resumable_pair(tmp_path):
    base = tmp_path / "signal" / "training.npz"
    common = {
        "total_training_games": 2,
        "games_per_iteration": 2,
        "adaptive_gpi": False,
        "checkpoint_interval": 1,
        "pool_refresh_games": 2,
        "max_pool_size": 1,
        "sl_weights_path": str(ROOT / "models" / "domino_sl_weights.npz"),
        "rl_weights_path": str(base),
        "seed": 91,
        "device": "cpu",
        "workers": 1,
        "safety_config": ParallelSafetyConfig(
            memory_reserve_mb=0,
            estimated_worker_mb=1,
            max_worker_rss_mb=1024,
        ),
        "quiet": True,
        "numbered_checkpoints": True,
    }
    stopped = self_play.train(
        stop_after_training_games=2,
        shutdown_requested=lambda: True,
        fresh_from_sl=True,
        **common,
    )
    assert stopped["completed_training_games"] == 0
    assert stopped["shutdown_requested"]
    weights = Path(stopped["rl_weights_path"])
    state = Path(stopped["resume_state_path"])
    assert weights.is_file() and state.is_file()

    resumed = self_play.train(
        stop_after_training_games=2,
        start_iteration=0,
        resume_weights_path=str(weights),
        resume_state_file=str(state),
        **common,
    )
    assert resumed["completed_training_games"] == 2
    assert resumed["rl_iterations_completed"] == 1


@pytest.mark.skipif(
    not (ROOT / "models" / "domino_sl_weights.npz").is_file(),
    reason="supervised smoke checkpoint is unavailable",
)
def test_numbered_checkpoint_callback_advances_by_committed_games(tmp_path):
    events = []

    def observe(event):
        metadata, _pool = self_play.load_resume_state(
            event["rl_weights_path"],
            event["resume_state_path"],
        )
        events.append((dict(event), int(metadata["completed_training_games"])))

    summary = self_play.train(
        total_training_games=5,
        games_per_iteration=1,
        adaptive_gpi=False,
        checkpoint_interval=2,
        pool_refresh_games=2,
        max_pool_size=1,
        sl_weights_path=str(ROOT / "models" / "domino_sl_weights.npz"),
        rl_weights_path=str(tmp_path / "callback" / "training.npz"),
        seed=92,
        device="cpu",
        workers=1,
        safety_config=ParallelSafetyConfig(
            memory_reserve_mb=0,
            estimated_worker_mb=1,
            max_worker_rss_mb=1024,
        ),
        quiet=True,
        numbered_checkpoints=True,
        fresh_from_sl=True,
        checkpoint_callback=observe,
    )
    assert [event[0]["completed_training_games"] for event in events] == [2, 4]
    assert [event[0]["rl_iterations_completed"] for event in events] == [2, 4]
    assert [event[1] for event in events] == [2, 4]
    assert summary["completed_training_games"] == 5
