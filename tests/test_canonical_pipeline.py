"""Contracts for canonical supervised assets and game-budgeted RL runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from agents.rl_nn import PolicyNetwork
from agents.network_architecture import architecture_from_hidden_sizes
from diagnostics.parallel_runner import ParallelSafetyConfig
from diagnostics.rl_progress import (
    CSV_FIELDS,
    FORMAT_VERSION,
    HISTORY_DATA_FIELDS,
    HISTORY_RECORD_TYPE,
    PERIODIC_SUMMARY_RETENTION,
    _rl_elapsed_hours,
    _hidden_footer_text,
    _training_footer_line,
    append_periodic_point,
    final_diagnostic_seed,
    periodic_diagnostic_seed,
    prune_periodic_diagnostic_artifacts,
    read_periodic_history,
    rebuild_progress_reports,
)
from training.rl import self_play
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
from training.rl.resume import (
    NUMBERED_CHECKPOINT_WEIGHT_RETENTION,
    RESUME_STATE_VERSION,
    RLTrainingConfiguration,
    _atomic_resume_state_save,
    _prune_numbered_checkpoint_weights,
    _validate_resume_configuration,
    load_resume_state,
)
from training.pipeline import (
    PERIODIC_DIAGNOSTIC_TUNING_FILE,
    PIPELINE_LEVELS,
    _cumulative_rl_games_per_second,
    _locked_run_arguments,
    _network_architecture,
    _ppo_config,
    _rl_config,
    _resolve_periodic_diagnostic_workers,
    _run_periodic_point,
    _write_forever_active_run,
    next_training_stop,
    parse_args,
    validate_args,
)
from training.rl.resume import (
    LEGACY_TRAINING_ALGORITHM,
    PPO_TRAINING_ALGORITHM,
)
from utils.artifacts import file_sha256


ROOT = Path(__file__).resolve().parents[1]


def _write_test_supervised_checkpoint(path, seed=123, hidden_sizes=None):
    network = PolicyNetwork(
        random_seed=seed,
        device="cpu",
        **({} if hidden_sizes is None else {"hidden_sizes": hidden_sizes}),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{
            name: np.asarray(getattr(network, name))
            for name in network.weight_names
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


def _rl_config_from_summary(summary, *, max_pool_size):
    """Return the canonical subset represented by a direct RL smoke run."""
    return {
        "games_per_iteration": summary["games_per_iteration"],
        "training_opponent": summary["training_opponent"],
        "learning_rate": summary["learning_rate"],
        "entropy_coef": summary["entropy_coef"],
        "weight_decay": summary["weight_decay"],
        "dropout_rate": summary["dropout_rate"],
        "max_pool_size": max_pool_size,
        "use_value_head": summary["use_value_head"],
        "value_coef": summary["value_coef"] or 0.5,
        "reward_schema": summary["reward_schema"],
        "gamma": summary["gamma"],
        "clip_grad_norm": summary["clip_grad_norm"],
        "normalize_advantages": summary["normalize_advantages"],
    }


def _test_resume_configuration(**overrides):
    values = {
        "total_training_games": 1000,
        "selected_gpi": 100,
        "selected_workers": 1,
        "rl_training_algorithm": PPO_TRAINING_ALGORITHM,
        "training_opponent": "self_play",
        "learning_rate": 0.001,
        "entropy_coef": 0.01,
        "max_pool_size": 5,
        "use_value_head": False,
        "value_coef": 0.5,
        "gamma": 1.0,
        "reward_schema": "default",
        "clip_grad_norm": 5.0,
        "normalize_advantages": True,
        "weight_decay": 0.0,
        "dropout_rate": 0.0,
        "effective_seed": 3,
        "device": "cpu",
        "sl_weights_sha256": "abc",
        "ppo_clip_epsilon": 0.2,
        "ppo_stop_kl": 0.015,
        "ppo_max_epochs": 4,
        "ppo_games_per_minibatch_scale": 1,
        "ppo_min_decisions_per_minibatch": 1,
        "prefer_gpu_buffer": False,
    }
    values.update(overrides)
    return RLTrainingConfiguration.from_mapping(values)


def _training_config():
    return canonical_training_config(max_epochs=10, batch_size=32)


def _periodic_row(games, checkpoint_hash="a", diagnostic_seed=7):
    return {
        "format_version": FORMAT_VERSION,
        "pipeline_level": "big",
        "seed": 42,
        "rl_games": games,
        "rl_iterations": games // 100,
        "checkpoint_path": f"checkpoint-{games}.npz",
        "checkpoint_sha256": checkpoint_hash,
        "configuration_sha256": None,
        "opponent": "random",
        "diagnostic_games": 100,
        "wins": 60,
        "diagnostic_seed": diagnostic_seed,
        "diagnostic_seed_namespace": "periodic_rl_vs_random",
        "diagnostic_seconds": 1.25,
        "rl_elapsed_seconds": games / 100.0,
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
    assert all(
        PIPELINE_LEVELS[level].ppo_max_epochs == 4
        for level in ("small", "default", "big", "huge")
    )
    assert PIPELINE_LEVELS["forever"].ppo_max_epochs == 16
    assert PIPELINE_LEVELS["small"].total_rl_games == 100_000
    assert PIPELINE_LEVELS["default"].total_rl_games == 500_000
    assert PIPELINE_LEVELS["big"].total_rl_games == 2_000_000
    assert PIPELINE_LEVELS["huge"].total_rl_games == 10_000_000
    assert PIPELINE_LEVELS["forever"].total_rl_games is None
    assert PIPELINE_LEVELS["big"].diagnostic_games == 1_000_000
    assert PIPELINE_LEVELS["small"].diagnostic_games == 10_000


def test_resume_accepts_default_or_explicit_run_directory(tmp_path):
    automatic = parse_args([
        "forever",
        "--resume",
        "--artifact-root",
        str(tmp_path),
    ])
    assert automatic.resume is True
    assert automatic.resume_from is None

    explicit_run = tmp_path / "models" / "rl" / "domino_rl_forever_seed42"
    explicit = parse_args([
        "forever",
        "--resume",
        str(explicit_run),
        "--artifact-root",
        str(tmp_path),
    ])
    assert explicit.resume is False
    assert explicit.resume_from == explicit_run


def test_forever_run_reloads_locked_arguments_from_the_active_pointer(
    tmp_path,
    capsys,
):
    initial = parse_args([
        "forever",
        "--artifact-root",
        str(tmp_path),
        "--seed",
        "7",
        "--run-name",
        "epochs8",
        "--gpi",
        "1000",
        "--ppo-max-epochs",
        "8",
        "--hidden1-size",
        "512",
        "--hidden2-size",
        "192",
    ])
    run_dir = canonical_run_dir(
        tmp_path,
        "forever",
        initial.seed,
        run_name=initial.run_name,
    )
    config = create_run_config(
        run_dir,
        root=tmp_path,
        pipeline_level="forever",
        seed=initial.seed,
        target_rl_games=None,
        supervised_weights_path="supervised.npz",
        supervised_weights_sha256="f" * 64,
        ppo_config=_ppo_config(initial),
        rl_config=_rl_config(initial),
        diagnostic_config={"periodic_games": 100},
        run_name=initial.run_name,
        locked_arguments=_locked_run_arguments(initial),
        network_architecture=_network_architecture(initial),
        machine={
            "cpu_model": "Test CPU",
            "logical_cpu_count": 4,
            "ram_total_bytes": 8 * 1024**3,
            "gpu_name": None,
            "vram_total_bytes": None,
            "rl_device": "cpu",
        },
    )
    _write_forever_active_run(tmp_path, run_dir, config)
    (run_dir / "training_state.json").write_text("{}", encoding="utf-8")

    resumed = parse_args([
        "forever",
        "--artifact-root",
        str(tmp_path),
    ])
    assert resumed.seed == 7
    assert resumed.run_name == "epochs8"
    assert resumed.gpi == 1000
    assert resumed.ppo_max_epochs == 8
    assert resumed.hidden1_size == 512
    assert resumed.hidden2_size == 192
    assert resumed._selected_run_dir == run_dir

    ignored_gpi = parse_args([
        "forever",
        "--artifact-root",
        str(tmp_path),
        "--gpi",
        "2000",
    ])
    assert ignored_gpi.gpi == 1000

    ignored_width = parse_args([
        "forever",
        "--artifact-root",
        str(tmp_path),
        "--hidden1-size",
        "256",
    ])
    assert ignored_width.hidden1_size == 512
    warning = capsys.readouterr().out
    assert "resume accepts no training or asset overrides" in warning
    assert "--gpi=2000" in warning
    assert "--hidden1-size=256" in warning

    separate = parse_args([
        "forever",
        "--artifact-root",
        str(tmp_path),
        "--seed",
        "8",
        "--run-name",
        "new",
        "--gpi",
        "2000",
    ])
    assert separate.seed == 8
    assert separate.run_name == "new"
    assert separate.gpi == 2000


def test_canonical_pipeline_accepts_ppo_and_reinforce_with_optional_critic(tmp_path):
    root_args = ["--artifact-root", str(tmp_path)]
    ppo = parse_args(["forever", *root_args])
    reinforce = parse_args(["forever", "--no-ppo", *root_args])
    finite = parse_args(["huge"])
    explicit = parse_args([
        "forever",
        "--ppo-max-epochs",
        "7",
        *root_args,
    ])

    validate_args(ppo, PIPELINE_LEVELS["forever"])
    validate_args(reinforce, PIPELINE_LEVELS["forever"])
    assert ppo.ppo_enabled is True
    assert ppo.ppo_max_epochs == 16
    assert reinforce.ppo_enabled is False
    assert reinforce.ppo_max_epochs == 4
    assert finite.ppo_max_epochs == 4
    assert explicit.ppo_max_epochs == 7

    critic_ppo = parse_args([
        "forever",
        "--artifact-root",
        str(tmp_path),
        "--run-name",
        "critic_ppo",
        "--value-head",
    ])
    critic_reinforce = parse_args([
        "forever",
        "--artifact-root",
        str(tmp_path),
        "--run-name",
        "critic_reinforce",
        "--no-ppo",
        "--value-head",
    ])
    validate_args(critic_ppo, PIPELINE_LEVELS["forever"])
    validate_args(critic_reinforce, PIPELINE_LEVELS["forever"])
    assert critic_ppo.ppo_enabled is True
    assert critic_ppo.value_head is True
    assert critic_reinforce.ppo_enabled is False
    assert critic_reinforce.value_head is True


def test_pipeline_hidden_sizes_have_one_default_and_are_locked_for_forever(tmp_path):
    defaults = parse_args([
        "forever",
        "--artifact-root",
        str(tmp_path),
        "--run-name",
        "default_architecture",
    ])
    custom = parse_args([
        "forever",
        "--artifact-root",
        str(tmp_path),
        "--run-name",
        "custom_architecture",
        "--hidden1-size",
        "512",
        "--hidden2-size",
        "192",
    ])

    assert (defaults.hidden1_size, defaults.hidden2_size) == (256, 128)
    assert (custom.hidden1_size, custom.hidden2_size) == (512, 192)
    locked = _locked_run_arguments(custom)
    assert locked["hidden1_size"] == 512
    assert locked["hidden2_size"] == 192


def test_forever_periodic_workers_are_recovered_once_and_then_persisted(tmp_path):
    args = parse_args([
        "forever",
        "--periodic-diagnostic-games",
        "100",
        "--artifact-root",
        str(tmp_path),
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
        "--artifact-root",
        str(tmp_path),
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
            losses=40,
            win_rate=0.60,
            ci95_win_rate_low=0.50,
            ci95_win_rate_high=0.69,
            runtime_profile_delta={
                "execution_seconds": 0.0,
                "sections_seconds": {},
            },
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
        "elapsed_rl_seconds": 0.0,
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
    weights_meta = write_weights_metadata(
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
    assert "created_at" not in dataset_meta
    assert "created_at" not in weights_meta
    assert "dataset_path" not in weights_meta
    assert "weights_path" not in weights_meta
    assert inspect_canonical_weights(
        paths,
        seed=42,
        dataset_sha256=dataset_meta["dataset_sha256"],
        training_config=training,
    ).compatible
    architecture_mismatch = inspect_canonical_weights(
        paths,
        seed=42,
        dataset_sha256=dataset_meta["dataset_sha256"],
        training_config=training,
        architecture=architecture_from_hidden_sizes(512, 192),
    )
    assert architecture_mismatch.status == "incompatible"
    assert any(
        "network_architecture" in reason
        for reason in architecture_mismatch.reasons
    )

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


def test_canonical_supervised_metadata_is_location_independent(tmp_path):
    metadata_pairs = []
    for directory_name in ("desktop", "notebook"):
        root = tmp_path / directory_name
        paths = canonical_asset_paths(root, 42)
        paths.dataset.parent.mkdir(parents=True)
        paths.weights.parent.mkdir(parents=True)
        paths.dataset.write_text(
            '{"state": {}, "action": [[0, 0], 0]}\n',
            encoding="utf-8",
        )
        dataset_meta = write_dataset_metadata(
            paths,
            root=root,
            seed=42,
            dataset_games=3,
            dataset_summary={"saved_turn_count": 1},
            generation_config=_generation_config(),
        )
        np.savez(
            paths.weights,
            **{
                name: np.zeros(shape, dtype=np.float32)
                for name, shape in EXPECTED_WEIGHT_SHAPES.items()
            },
        )
        weights_meta = write_weights_metadata(
            paths,
            root=root,
            seed=42,
            dataset_sha256=dataset_meta["dataset_sha256"],
            training_config=_training_config(),
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
        metadata_pairs.append((dataset_meta, weights_meta))

    assert metadata_pairs[0] == metadata_pairs[1]


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
        "ppo_config": {"clip_epsilon": 0.2, "target_kl": 0.01},
        "rl_config": {"gamma": 1.0},
    }
    first = create_run_config(run_dir, **values)
    second = create_run_config(run_dir, **values)
    assert second["created_at"] == first["created_at"]
    assert first["algorithm"] == PPO_TRAINING_ALGORITHM

    reporting_only = dict(values)
    reporting_only["ppo_config"] = {
        "clip_epsilon": 0.2,
        "target_kl": 0.005,
    }
    unchanged = create_run_config(run_dir, **reporting_only)
    assert unchanged["configuration_sha256"] == first["configuration_sha256"]

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


def test_runs_without_stored_regularizers_still_resume(tmp_path):
    """Read a missing dropout/weight-decay field as the disabled default."""
    run_dir = canonical_run_dir(tmp_path, "big", 42)
    legacy_rl_config = {"gamma": 1.0, "learning_rate": 0.001}
    values = {
        "root": ROOT,
        "pipeline_level": "big",
        "seed": 42,
        "target_rl_games": 2_000_000,
        "supervised_weights_path": "models/sl.npz",
        "supervised_weights_sha256": "abc",
        "ppo_config": {"clip_epsilon": 0.2, "target_kl": 0.01},
        "rl_config": legacy_rl_config,
    }
    published = create_run_config(run_dir, **values)
    assert "dropout_rate" not in published["rl_config"]

    disabled = dict(values)
    disabled["rl_config"] = {
        **legacy_rl_config,
        "weight_decay": 0.0,
        "dropout_rate": 0.0,
    }
    reused = create_run_config(run_dir, **disabled)
    assert reused["configuration_sha256"] == published["configuration_sha256"]

    enabled = dict(values)
    enabled["rl_config"] = {**legacy_rl_config, "dropout_rate": 0.2}
    with pytest.raises(ValueError, match="rl_config"):
        create_run_config(run_dir, **enabled)


def test_resume_state_without_stored_regularizers_is_compatible():
    """Continue a pre-regularization exact resume pair without a rebuild."""
    expected = _test_resume_configuration()
    legacy = {
        key: value
        for key, value in expected.to_dict().items()
        if key not in ("weight_decay", "dropout_rate")
    }
    _validate_resume_configuration({"configuration": legacy}, expected)

    enabled = RLTrainingConfiguration.from_mapping({
        **expected.to_dict(),
        "dropout_rate": 0.2,
    })
    with pytest.raises(ValueError, match="dropout_rate"):
        _validate_resume_configuration({"configuration": legacy}, enabled)


def test_commit_change_warns_but_never_blocks_exact_resume(monkeypatch):
    initial_commit = "a" * 40
    current_commit = "b" * 40
    expected = _test_resume_configuration(git_commit=initial_commit)
    messages = []
    monkeypatch.setattr(
        "training.rl.resume.current_git_commit",
        lambda: current_commit,
    )

    expected.warn_if_commit_changed(messages.append)
    _validate_resume_configuration(
        {"configuration": {**expected.to_dict(), "git_commit": current_commit}},
        expected,
        emit_status=messages.append,
    )

    assert expected.git_commit == initial_commit
    assert len(messages) == 2
    assert all("commit" in message.lower() for message in messages)


def test_runs_with_stored_pool_refresh_games_still_resume(tmp_path):
    """Ignore the retired opponent-cadence field stored by older runs.

    The cadence is now one snapshot per iteration, controlled by gpi, so a
    stored ``pool_refresh_games`` no longer affects computation and must not
    make an existing run refuse to continue.
    """
    run_dir = canonical_run_dir(tmp_path, "big", 42)
    legacy_rl_config = {
        "gamma": 1.0,
        "learning_rate": 0.001,
        "pool_refresh_games": 400,
    }
    values = {
        "root": ROOT,
        "pipeline_level": "big",
        "seed": 42,
        "target_rl_games": 2_000_000,
        "supervised_weights_path": "models/sl.npz",
        "supervised_weights_sha256": "abc",
        "ppo_config": {"clip_epsilon": 0.2, "target_kl": 0.01},
        "rl_config": legacy_rl_config,
    }
    published = create_run_config(run_dir, **values)
    assert published["rl_config"]["pool_refresh_games"] == 400

    current = dict(values)
    current["rl_config"] = {"gamma": 1.0, "learning_rate": 0.001}
    reused = create_run_config(run_dir, **current)
    assert reused["configuration_sha256"] == published["configuration_sha256"]
    # The stored configuration is never rewritten by the removal.
    assert reused["rl_config"]["pool_refresh_games"] == 400

    # A field that still affects computation must keep rejecting the resume.
    conflicting = dict(values)
    conflicting["rl_config"] = {"gamma": 0.97, "learning_rate": 0.001}
    with pytest.raises(ValueError, match="rl_config"):
        create_run_config(run_dir, **conflicting)


def test_resume_state_with_stored_pool_refresh_games_is_compatible():
    """Continue an exact resume pair saved while the cadence flag existed."""
    expected = _test_resume_configuration()
    assert "pool_refresh_games" not in expected.to_dict()

    legacy = {**expected.to_dict(), "pool_refresh_games": 400}
    _validate_resume_configuration({"configuration": legacy}, expected)

    # Any stored value is ignored, not merely the historical default.
    other = {**expected.to_dict(), "pool_refresh_games": 1}
    _validate_resume_configuration({"configuration": other}, expected)

    # A field that still affects computation must keep rejecting the resume.
    conflicting = {**legacy, "selected_gpi": 200}
    with pytest.raises(ValueError, match="selected_gpi"):
        _validate_resume_configuration({"configuration": conflicting}, expected)


def test_hidden_layer_count_joins_the_run_identity_and_is_locked(tmp_path):
    """Lock depth and every width for a forever run without touching old runs."""
    defaults = parse_args([
        "forever",
        "--artifact-root",
        str(tmp_path),
        "--run-name",
        "default_depth",
    ])
    deep = parse_args([
        "forever",
        "--artifact-root",
        str(tmp_path),
        "--run-name",
        "deep",
        "--hidden-layers",
        "4",
        "--hidden1-size",
        "512",
        "--hidden3-size",
        "96",
    ])

    assert _network_architecture(defaults).as_list() == [168, 256, 128, 56]
    assert _network_architecture(deep).as_list() == [168, 512, 128, 96, 128, 56]

    locked = _locked_run_arguments(deep)
    assert locked["hidden_layers"] == 4
    assert locked["hidden1_size"] == 512
    assert locked["hidden3_size"] == 96
    assert locked["hidden4_size"] == 128
    # An unused layer is recorded as absent rather than as a stale width.
    assert _locked_run_arguments(defaults)["hidden3_size"] is None


def test_deep_opponent_pool_survives_a_resume_state_round_trip(tmp_path):
    """Save and reload a four-hidden-layer opponent pool without name guesses."""
    network = PolicyNetwork(
        random_seed=5,
        device="cpu",
        hidden_sizes=(64, 48, 32, 16),
    )
    snapshot = {
        name: np.asarray(getattr(network, name))
        for name in network.weight_names
    }
    weights_path = tmp_path / "deep.npz"
    network.save(weights_path)
    state_path = tmp_path / "deep.resume.npz"
    metadata = {
        "version": RESUME_STATE_VERSION,
        "weights_sha256": file_sha256(weights_path),
    }
    _atomic_resume_state_save(state_path, metadata, (snapshot,))

    _loaded_metadata, snapshots = load_resume_state(weights_path, state_path)
    assert len(snapshots) == 1
    assert sorted(snapshots[0]) == sorted(network.weight_names)
    for name, value in snapshot.items():
        assert np.array_equal(snapshots[0][name], value)


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
        "gpi": 2,
        "checkpoint_interval": 1,
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
    run_dir = canonical_run_dir(tmp_path, "forever", 321)
    run_config = create_run_config(
        run_dir,
        root=ROOT,
        pipeline_level="forever",
        seed=321,
        target_rl_games=None,
        supervised_weights_path=supervised,
        supervised_weights_sha256=supervised_hash,
        ppo_config=uninterrupted["ppo_configuration"],
        rl_config=_rl_config_from_summary(uninterrupted, max_pool_size=2),
        algorithm=LEGACY_TRAINING_ALGORITHM,
    )
    partial = self_play.train(
        rl_weights_path=str(tmp_path / "split" / "training.npz"),
        stop_after_training_games=2,
        fresh_from_sl=True,
        run_configuration=run_config,
        **common,
    )

    profile = partial["runtime_profile_delta"]
    assert "legacy_policy_update" in profile["sections_seconds"]
    assert "ppo_update" not in profile["sections_seconds"]
    assert profile["ppo_sections_seconds"] == {}

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

    point = load_resume_point(run_dir)

    resumed = self_play.train(
        rl_weights_path=str(tmp_path / "split" / "training.npz"),
        start_iteration=point.completed_iterations,
        resume_weights_path=str(point.weights_path),
        resume_state_file=str(point.resume_state_path),
        run_configuration=run_config,
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
    first["format_version"] = 2
    first["checkpoint_path"] = str(
        tmp_path / "checkpoints" / "games_0000000000_weights.npz"
    )
    with open(history, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(first) + "\n")
        stream.write('{"partial":')
    legacy_rows = read_periodic_history(history)
    assert len(legacy_rows) == 1
    assert legacy_rows[0]["wins"] == 60

    _row, appended = append_periodic_point(history, first)
    assert not appended

    second = _periodic_row(100_000, checkpoint_hash="one")
    second["checkpoint_path"] = str(
        tmp_path / "checkpoints" / "games_0000100000_weights.npz"
    )
    _row, appended = append_periodic_point(history, second)
    assert appended
    rows = read_periodic_history(history)
    assert len(rows) == 2
    assert rows[0]["losses"] == 40
    assert rows[0]["win_rate"] == pytest.approx(0.60)
    assert rows[0]["progress_elapsed_seconds"] == pytest.approx(1.25)
    assert rows[1]["progress_elapsed_seconds"] == pytest.approx(1002.5)

    raw_lines = history.read_text(encoding="utf-8").splitlines()
    header = json.loads(raw_lines[0])
    assert header["record_type"] == HISTORY_RECORD_TYPE
    assert header["format_version"] == FORMAT_VERSION
    assert header["columns"] == list(HISTORY_DATA_FIELDS)
    assert header["checkpoint_path_base"] == "checkpoints"
    assert len(raw_lines) == 3
    compact_first = json.loads(raw_lines[1])
    checkpoint_index = HISTORY_DATA_FIELDS.index("checkpoint_path")
    assert compact_first[checkpoint_index] == "games_0000000000_weights.npz"
    assert isinstance(compact_first, list)
    assert "ci95" not in history.read_text(encoding="utf-8")
    assert "loss_rate" not in history.read_text(encoding="utf-8")

    csv_path, plot_path, _log_path = rebuild_progress_reports(tmp_path)
    assert csv_path.is_file()
    assert plot_path.is_file()
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 3
    assert _rl_elapsed_hours(rows[1]) == pytest.approx(1002.5 / 3600.0)
    with open(csv_path, newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert list(csv_rows[0]) == list(CSV_FIELDS)
    for field in (
        "rl_elapsed_hours",
        "win_rate_percent",
        "ci95_low_percent",
        "ci95_high_percent",
    ):
        assert all(len(value[field].split(".")[-1]) == 3 for value in csv_rows)
    for removed in (
        "optimizer_steps",
        "diagnostic_games",
        "diagnostic_seconds",
        "checkpoint_path",
        "checkpoint_sha256",
        "configuration_sha256",
    ):
        assert removed not in csv_rows[0]


def test_progress_footer_reports_value_head_hidden_layers_and_regularizers():
    assert _training_footer_line({
        "network_architecture": [168, 512, 192, 56],
        "rl_config": {
            "use_value_head": True,
            "dropout_rate": 0.1,
            "weight_decay": 0.0001,
        },
    }) == (
        "Value head on · hidden 2 layers 512x192 · "
        "dropout 0.1 · weight decay 0.0001"
    )
    assert _training_footer_line({
        "network_architecture": [168, 256, 128, 56],
        "rl_config": {
            "use_value_head": False,
            "dropout_rate": 0.0,
            "weight_decay": 0.0,
        },
    }) == (
        "Value head off · hidden 2 layers 256x128 · "
        "dropout off · weight decay off"
    )
    # A run recorded before the regularizers existed stores neither field.
    assert _training_footer_line({
        "network_architecture": [168, 256, 128, 56],
        "rl_config": {"use_value_head": False},
    }) == (
        "Value head off · hidden 2 layers 256x128 · "
        "dropout off · weight decay off"
    )


def test_progress_footer_lists_every_hidden_layer_at_any_depth():
    """Report the layer count and each width, not just the first two."""
    assert _hidden_footer_text([168, 96, 56]) == "1 layer 96"
    assert _hidden_footer_text([168, 256, 128, 56]) == "2 layers 256x128"
    assert (
        _hidden_footer_text([168, 512, 256, 128, 64, 56])
        == "4 layers 512x256x128x64"
    )
    assert (
        _hidden_footer_text([168, *([128] * 8), 56])
        == "8 layers 128x128x128x128x128x128x128x128"
    )
    # A run configuration without a usable architecture must not invent one.
    assert _hidden_footer_text([]) == "unknown"
    assert _hidden_footer_text(None) == "unknown"


def test_deep_progress_footer_shrinks_instead_of_overlapping(tmp_path):
    """Keep the widest supported footer clear of the left-hand block."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from diagnostics.rl_progress import (
        FOOTER_FONT_SIZE,
        FOOTER_MIN_FONT_SIZE,
        _fitted_footer_fontsize,
    )

    figure = Figure(figsize=(9.5, 5.5))
    FigureCanvasAgg(figure)
    start = figure.text(
        0.01,
        0.06,
        "Start: model_iter000000.npz · sha256 0123456789ab...",
        fontsize=FOOTER_FONT_SIZE,
    )
    reserved = start.get_window_extent(figure.canvas.get_renderer()).width

    default_line = _training_footer_line({
        "network_architecture": [168, 256, 128, 56],
        "rl_config": {},
    })
    widest_line = _training_footer_line({
        "network_architecture": [168, 2048, *([1024] * 7), 56],
        "rl_config": {
            "use_value_head": True,
            "dropout_rate": 0.1,
            "weight_decay": 0.0001,
        },
    })
    # The unchanged default keeps the historical footer size.
    assert _fitted_footer_fontsize(figure, default_line, reserved) == (
        FOOTER_FONT_SIZE
    )
    widest_size = _fitted_footer_fontsize(figure, widest_line, reserved)
    assert FOOTER_MIN_FONT_SIZE < widest_size < FOOTER_FONT_SIZE

    probe = figure.text(0.0, 0.0, widest_line, fontsize=widest_size)
    width = probe.get_window_extent(figure.canvas.get_renderer()).width
    assert reserved + width <= figure.bbox.width


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
    probe_summary = self_play.train(
        total_training_games=3,
        stop_after_training_games=2,
        gpi=2,
        checkpoint_interval=1,
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
    run_config = create_run_config(
        run_dir,
        root=ROOT,
        pipeline_level="big",
        seed=42,
        target_rl_games=3,
        supervised_weights_path=supervised,
        supervised_weights_sha256=supervised_hash,
        ppo_config=probe_summary["ppo_configuration"],
        rl_config=_rl_config_from_summary(probe_summary, max_pool_size=2),
    )
    summary = self_play.train(
        total_training_games=3,
        stop_after_training_games=2,
        gpi=2,
        checkpoint_interval=1,
        max_pool_size=2,
        sl_weights_path=str(supervised),
        rl_weights_path=str(tmp_path / "canonical" / "training.npz"),
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
        run_configuration=run_config,
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
    point = load_resume_point(run_dir)
    assert point.completed_games == 2
    assert point.completed_iterations == 1
    assert state["ppo_updates_completed"] == 1
    assert Path(run_dir / state["opponent_pool_manifest"]).is_file()
    assert (run_dir / "latest_weights.npz").is_file()
    assert (run_dir / "optimizer_state.npz").is_file()

    # Resume follows the immutable generation named in training_state.json,
    # not the post-commit convenience alias.
    (run_dir / "latest_weights.npz").write_bytes(b"damaged alias")
    assert load_resume_point(run_dir).completed_games == 2


@pytest.mark.skipif(
    not (ROOT / "models" / "domino_sl_weights.npz").is_file(),
    reason="supervised smoke checkpoint is unavailable",
)
def test_shutdown_before_first_iteration_still_creates_a_resumable_pair(tmp_path):
    base = tmp_path / "signal" / "training.npz"
    common = {
        "total_training_games": 2,
        "gpi": 2,
        "checkpoint_interval": 1,
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
        gpi=1,
        checkpoint_interval=2,
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
