"""Canonical RL run directories built on exact numbered resume checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from agents.encoder import DominoEncoder
from training.rl_resume import load_resume_state
from utils.artifacts import (
    atomic_copy,
    atomic_savez,
    atomic_write_json,
    file_sha256,
)


RUN_FORMAT_VERSION = 1
NETWORK_ARCHITECTURE = [
    DominoEncoder.VECTOR_SIZE,
    256,
    128,
    DominoEncoder.ACTION_SIZE,
]


@dataclass(frozen=True)
class ResumePoint:
    run_dir: Path
    weights_path: Path
    resume_state_path: Path
    training_state: dict

    @property
    def completed_games(self):
        return int(self.training_state["rl_games_completed"])

    @property
    def completed_iterations(self):
        return int(self.training_state["rl_iterations_completed"])


def canonical_run_dir(root, level, seed):
    return (
        Path(root)
        / "models"
        / "rl"
        / f"domino_rl_{level}_seed{int(seed)}"
    )


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _git_commit(root):
    for candidate in (Path(root), Path(__file__).resolve().parents[1]):
        try:
            return subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=candidate,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except Exception:
            continue
    return None


def _relative(run_dir, path):
    path = Path(path)
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _resolve(run_dir, value):
    path = Path(value)
    return path if path.is_absolute() else Path(run_dir) / path


def _snapshot_digest(snapshot):
    digest = hashlib.sha256()
    for name in sorted(snapshot):
        value = np.asarray(snapshot[name])
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _npz_matches(path, expected):
    """Return whether an existing archive contains exactly the expected arrays."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                return False
            return all(
                np.array_equal(archive[name], np.asarray(value))
                for name, value in expected.items()
            )
    except (OSError, ValueError, KeyError):
        return False


def _prune_superseded_latest_payloads(
    run_dir,
    *,
    current_generation_paths,
    current_pool_paths,
    current_manifest_path,
):
    """Bound latest-state storage after the new commit marker is durable."""
    run_dir = Path(run_dir)
    keep_generation = {Path(path).resolve() for path in current_generation_paths}
    keep_pool = {Path(path).resolve() for path in current_pool_paths}
    for path in (run_dir / "checkpoint_states").glob("games_*_latest_*"):
        if path.resolve() not in keep_generation:
            try:
                path.unlink()
            except OSError:
                pass
    pool_dir = run_dir / "opponent_pool"
    for path in pool_dir.glob("opponent_*.npz"):
        if path.resolve() not in keep_pool:
            try:
                path.unlink()
            except OSError:
                pass
    for path in pool_dir.glob("pool_manifest_games_*.json"):
        if path.resolve() != Path(current_manifest_path).resolve():
            try:
                path.unlink()
            except OSError:
                pass


def create_run_config(
    run_dir,
    *,
    root,
    pipeline_level,
    seed,
    target_rl_games,
    supervised_weights_path,
    supervised_weights_sha256,
    ppo_config,
    rl_config,
    diagnostic_config=None,
    lineage=None,
    allow_target_extension=False,
):
    """Atomically publish the immutable identity and requested target of a run."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for child in ("checkpoints", "checkpoint_states", "opponent_pool", "diagnostics"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    value = {
        "format_version": RUN_FORMAT_VERSION,
        "pipeline_level": pipeline_level,
        "seed": int(seed),
        "target_rl_games": (
            None if target_rl_games is None else int(target_rl_games)
        ),
        "unbounded": target_rl_games is None,
        "encoder_size": DominoEncoder.VECTOR_SIZE,
        "action_count": DominoEncoder.ACTION_SIZE,
        "network_architecture": NETWORK_ARCHITECTURE,
        "algorithm": "ppo_v1",
        "supervised_weights_path": str(supervised_weights_path),
        "supervised_weights_sha256": supervised_weights_sha256,
        "ppo_config": dict(ppo_config),
        "rl_config": dict(rl_config),
        "diagnostic_config": dict(diagnostic_config or {}),
        "lineage": list(lineage or ()),
        "git_commit": _git_commit(root),
        "created_at": _utc_now(),
    }
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as stream:
                existing = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Existing canonical run config cannot be read: {config_path}."
            ) from exc
        immutable_keys = (
            "format_version",
            "seed",
            "encoder_size",
            "action_count",
            "network_architecture",
            "algorithm",
            "supervised_weights_sha256",
            "ppo_config",
            "rl_config",
        )
        differences = [
            key for key in immutable_keys if existing.get(key) != value.get(key)
        ]
        target_changed = existing.get("target_rl_games") != value["target_rl_games"]
        level_changed = existing.get("pipeline_level") != value["pipeline_level"]
        if differences or (
            (target_changed or level_changed) and not allow_target_extension
        ):
            fields = ", ".join(
                differences
                + (["target_rl_games"] if target_changed else [])
                + (["pipeline_level"] if level_changed else [])
            )
            raise ValueError(
                "Existing canonical run_config.json is incompatible in: " + fields
            )
        diagnostic_changed = (
            existing.get("diagnostic_config") != value["diagnostic_config"]
        )
        if target_changed or level_changed or lineage or diagnostic_changed:
            existing.update({
                "pipeline_level": pipeline_level,
                "target_rl_games": value["target_rl_games"],
                "unbounded": value["unbounded"],
                "lineage": list(lineage or existing.get("lineage", ())),
                "updated_at": _utc_now(),
            })
            existing["diagnostic_config"] = value["diagnostic_config"]
            atomic_write_json(config_path, existing)
        return existing
    atomic_write_json(config_path, value)
    return value


def load_resume_point(
    run_dir,
    *,
    seed,
    supervised_weights_sha256,
    ppo_config,
    force_incompatible=False,
):
    """Validate the latest marker and exact weights/resume pair."""
    run_dir = Path(run_dir)
    state_path = run_dir / "training_state.json"
    try:
        with open(state_path, "r", encoding="utf-8") as stream:
            state = json.load(stream)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No canonical RL training state exists at {state_path}."
        ) from exc
    if state.get("format_version") != RUN_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported canonical RL state version: {state.get('format_version')!r}."
        )
    differences = []
    expected = {
        "seed": int(seed),
        "encoder_size": DominoEncoder.VECTOR_SIZE,
        "action_count": DominoEncoder.ACTION_SIZE,
        "network_architecture": NETWORK_ARCHITECTURE,
        "algorithm": "ppo_v1",
        "supervised_weights_sha256": supervised_weights_sha256,
        "ppo_config": dict(ppo_config),
    }
    for key, expected_value in expected.items():
        if state.get(key) != expected_value:
            differences.append(
                f"{key}: checkpoint={state.get(key)!r}, requested={expected_value!r}"
            )
    if differences and not force_incompatible:
        raise ValueError(
            "Canonical RL resume is incompatible: " + "; ".join(differences)
        )
    weights = _resolve(run_dir, state["latest_weights_path"])
    resume_state = _resolve(run_dir, state["latest_resume_state_path"])
    if not weights.is_file() or not resume_state.is_file():
        raise FileNotFoundError(
            "Canonical latest checkpoint is incomplete: "
            f"weights={weights.is_file()}, resume_state={resume_state.is_file()}."
        )
    actual_hash = file_sha256(weights)
    if actual_hash != state.get("latest_weights_sha256"):
        raise ValueError(
            "Canonical latest weights generation does not match "
            "training_state.json."
        )
    metadata, resume_pool = load_resume_state(weights, resume_state)
    if int(metadata["completed_training_games"]) != int(
        state["rl_games_completed"]
    ):
        raise ValueError("Canonical state and exact resume pair disagree on RL games.")
    if int(metadata["completed_iteration"]) != int(
        state["rl_iterations_completed"]
    ):
        raise ValueError(
            "Canonical state and exact resume pair disagree on RL iterations."
        )
    pair_config = metadata.get("configuration", {})
    pair_expected = {
        "effective_seed": int(seed),
        "rl_training_algorithm": "ppo_v1",
        "sl_weights_sha256": supervised_weights_sha256,
        "ppo_clip_epsilon": float(ppo_config["clip_epsilon"]),
        "ppo_target_kl": float(ppo_config["target_kl"]),
        "ppo_stop_kl": float(ppo_config["stop_kl"]),
        "ppo_max_epochs": int(ppo_config["max_epochs"]),
        "ppo_min_minibatches": int(ppo_config["min_minibatches"]),
        "ppo_max_minibatches": int(ppo_config["max_minibatches"]),
        "ppo_games_per_minibatch_scale": int(
            ppo_config["games_per_minibatch_scale"]
        ),
        "ppo_min_decisions_per_minibatch": int(
            ppo_config["min_decisions_per_minibatch"]
        ),
        "prefer_gpu_buffer": bool(ppo_config["prefer_gpu_buffer"]),
        "gpu_buffer_safety_fraction": float(
            ppo_config["gpu_buffer_safety_fraction"]
        ),
    }
    pair_differences = [
        key
        for key, expected_value in pair_expected.items()
        if pair_config.get(key) != expected_value
    ]
    if pair_differences and not force_incompatible:
        raise ValueError(
            "Canonical exact resume pair is incompatible in: "
            + ", ".join(pair_differences)
        )
    resume_hash = file_sha256(resume_state)
    if resume_hash != state.get("latest_resume_state_sha256"):
        raise ValueError(
            "Canonical resume-state file does not match training_state.json."
        )
    optimizer_path = _resolve(run_dir, state["optimizer_state_path"])
    rng_path = _resolve(run_dir, state["rng_state_files"]["parent"])
    manifest_path = _resolve(run_dir, state["opponent_pool_manifest"])
    for label, path in (
        ("optimizer state", optimizer_path),
        ("RNG state", rng_path),
        ("opponent-pool manifest", manifest_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Canonical {label} is missing: {path}.")
    if state.get("optimizer_state_sha256") != file_sha256(optimizer_path):
        raise ValueError("Canonical optimizer state hash is inconsistent.")
    optimizer_arrays = {
        key: np.asarray(value)
        for key, value in metadata["optimizer_state"].items()
    }
    if not _npz_matches(optimizer_path, optimizer_arrays):
        raise ValueError(
            "Canonical optimizer state and exact resume pair disagree."
        )
    if state.get("opponent_pool_manifest_sha256") != file_sha256(manifest_path):
        raise ValueError("Canonical opponent-pool manifest hash is inconsistent.")
    if state.get("rng_state_sha256") != file_sha256(rng_path):
        raise ValueError("Canonical RNG state hash is inconsistent.")
    try:
        rng_value = json.loads(rng_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Canonical RNG/pool metadata cannot be read.") from exc
    for key, expected_value in metadata["rng_state"].items():
        if rng_value.get(key) != expected_value:
            raise ValueError("Canonical RNG state and exact resume pair disagree.")
    if int(manifest.get("snapshot_count", -1)) != len(resume_pool):
        raise ValueError(
            "Canonical opponent-pool manifest and resume pair disagree on size."
        )
    for index, entry in enumerate(manifest.get("snapshots", ())):
        snapshot_path = _resolve(run_dir, entry["path"])
        if not snapshot_path.is_file() or file_sha256(snapshot_path) != entry.get(
            "sha256"
        ) or not _npz_matches(snapshot_path, resume_pool[index]):
            raise ValueError(
                f"Canonical opponent-pool snapshot is incomplete: {snapshot_path}."
            )
    return ResumePoint(run_dir, weights, resume_state, state)


def publish_checkpoint(
    run_dir,
    *,
    root,
    pipeline_level,
    seed,
    target_rl_games,
    supervised_weights_path,
    supervised_weights_sha256,
    summary,
    last_periodic_diagnostic_game,
    next_periodic_diagnostic_game,
    milestone=False,
    reason="training",
):
    """Publish split canonical state and update latest only after all payloads."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    source_weights = Path(summary["rl_weights_path"])
    source_resume = Path(summary["resume_state_path"])
    resume_metadata, pool_snapshots = load_resume_state(
        source_weights,
        source_resume,
    )
    completed_games = int(resume_metadata["completed_training_games"])
    completed_iterations = int(resume_metadata["completed_iteration"])
    training = dict(resume_metadata.get("training_state", {}))
    optimizer = dict(resume_metadata["optimizer_state"])

    source_resume_hash = file_sha256(source_resume)
    generation_prefix = (
        f"games_{completed_games:010d}_latest_{source_resume_hash[:12]}"
    )
    latest_weights = atomic_copy(
        source_weights,
        run_dir / "checkpoint_states" / f"{generation_prefix}_weights.npz",
    )
    latest_resume = atomic_copy(
        source_resume,
        run_dir / "checkpoint_states" / f"{generation_prefix}_state.npz",
    )
    latest_hash = file_sha256(latest_weights)
    latest_resume_hash = file_sha256(latest_resume)
    optimizer_path = (
        run_dir
        / "checkpoint_states"
        / f"{generation_prefix}_optimizer_{latest_resume_hash[:12]}.npz"
    )
    optimizer_arrays = {
        key: np.asarray(value) for key, value in optimizer.items()
    }
    if not _npz_matches(optimizer_path, optimizer_arrays):
        atomic_savez(
            optimizer_path,
            **optimizer_arrays,
        )
    rng_path = run_dir / "checkpoint_states" / (
        f"{generation_prefix}_rng_{latest_resume_hash[:12]}.json"
    )
    atomic_write_json(rng_path, {
        **resume_metadata["rng_state"],
        "derived_streams": {
            "rollout": "stable_seed(base_seed, iteration/game id)",
            "ppo_shuffle": "stable_seed(base_seed, ppo_shuffle, iteration, epoch)",
            "opponent_selection": "per-game seeded Python random",
            "cupy_mutable_rng_used": False,
        },
    })

    pool_dir = run_dir / "opponent_pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    pool_metadata = list(resume_metadata.get("opponent_pool_metadata", ()))
    manifest_entries = []
    for index, snapshot in enumerate(pool_snapshots):
        metadata = dict(pool_metadata[index] if index < len(pool_metadata) else {})
        snapshot_id = int(metadata.get("snapshot_id", index))
        content_digest = _snapshot_digest(snapshot)
        snapshot_path = pool_dir / (
            f"opponent_{snapshot_id:06d}_{content_digest[:16]}.npz"
        )
        if not _npz_matches(snapshot_path, snapshot):
            atomic_savez(snapshot_path, **snapshot)
        manifest_entries.append({
            **metadata,
            "snapshot_id": snapshot_id,
            "logical_order": index,
            "path": _relative(run_dir, snapshot_path),
            "sha256": file_sha256(snapshot_path),
            "sampling_rule": metadata.get("sampling_rule", "uniform_random"),
        })
    pool_manifest_path = pool_dir / (
        f"pool_manifest_games_{completed_games:010d}_"
        f"{latest_resume_hash[:12]}.json"
    )
    manifest_payload = {
        "format_version": 1,
        "sampling_rule": "uniform_random",
        "snapshot_count": len(manifest_entries),
        "snapshots": manifest_entries,
    }
    manifest_matches = False
    if pool_manifest_path.exists():
        try:
            existing_manifest = json.loads(
                pool_manifest_path.read_text(encoding="utf-8")
            )
            manifest_matches = all(
                existing_manifest.get(key) == value
                for key, value in manifest_payload.items()
            )
        except (OSError, json.JSONDecodeError):
            manifest_matches = False
    if not manifest_matches:
        atomic_write_json(pool_manifest_path, {
            **manifest_payload,
            "updated_at": _utc_now(),
        })
    pool_manifest_hash = file_sha256(pool_manifest_path)

    checkpoint_path = None
    checkpoint_state_path = None
    if milestone:
        checkpoint_path = (
            run_dir / "checkpoints" / f"games_{completed_games:010d}_weights.npz"
        )
        checkpoint_state_path = (
            run_dir
            / "checkpoint_states"
            / f"games_{completed_games:010d}_state.npz"
        )
        atomic_copy(source_weights, checkpoint_path)
        atomic_copy(source_resume, checkpoint_state_path)
        atomic_write_json(
            checkpoint_state_path.with_suffix(".json"),
            {
                "format_version": 1,
                "rl_games": completed_games,
                "rl_iterations": completed_iterations,
                "weights_path": _relative(run_dir, checkpoint_path),
                "weights_sha256": file_sha256(checkpoint_path),
                "resume_state_path": _relative(run_dir, checkpoint_state_path),
                "reason": reason,
                "created_at": _utc_now(),
            },
        )

    previous_milestone = None
    previous_state_path = run_dir / "training_state.json"
    if previous_state_path.exists():
        try:
            previous_milestone = json.loads(
                previous_state_path.read_text(encoding="utf-8")
            ).get("latest_milestone_checkpoint")
        except (OSError, json.JSONDecodeError):
            previous_milestone = None
    state = {
        "format_version": RUN_FORMAT_VERSION,
        "algorithm": "ppo_v1",
        "pipeline_level": pipeline_level,
        "seed": int(seed),
        "target_rl_games": (
            None if target_rl_games is None else int(target_rl_games)
        ),
        "unbounded": target_rl_games is None,
        "encoder_size": DominoEncoder.VECTOR_SIZE,
        "action_count": DominoEncoder.ACTION_SIZE,
        "network_architecture": NETWORK_ARCHITECTURE,
        "rl_games_completed": completed_games,
        "rl_iterations_completed": completed_iterations,
        "ppo_updates_completed": int(training.get("ppo_updates_completed", 0)),
        "optimizer_steps_completed": int(optimizer["step_count"]),
        "trainable_decisions_seen": int(
            training.get("trainable_decisions_seen", 0)
        ),
        "selected_gpi": int(summary["games_per_iteration"]),
        "selected_workers": int(summary["final_workers"]),
        "latest_weights_path": _relative(run_dir, latest_weights),
        "latest_weights_sha256": latest_hash,
        "latest_resume_state_path": _relative(run_dir, latest_resume),
        "latest_resume_state_sha256": latest_resume_hash,
        "optimizer_state_path": _relative(run_dir, optimizer_path),
        "optimizer_state_sha256": file_sha256(optimizer_path),
        "opponent_pool_manifest": _relative(run_dir, pool_manifest_path),
        "opponent_pool_manifest_sha256": pool_manifest_hash,
        "supervised_weights_path": str(supervised_weights_path),
        "supervised_weights_sha256": supervised_weights_sha256,
        "ppo_config": dict(summary["ppo_configuration"]),
        "rng_state_files": {"parent": _relative(run_dir, rng_path)},
        "rng_state_sha256": file_sha256(rng_path),
        "last_periodic_diagnostic_game": int(last_periodic_diagnostic_game),
        "next_periodic_diagnostic_game": (
            None
            if next_periodic_diagnostic_game is None
            else int(next_periodic_diagnostic_game)
        ),
        "elapsed_rl_seconds": float(training.get("elapsed_rl_seconds", 0.0)),
        "latest_milestone_checkpoint": (
            previous_milestone
            if checkpoint_path is None
            else _relative(run_dir, checkpoint_path)
        ),
        "checkpoint_reason": reason,
        "shutdown_requested": reason == "shutdown",
        "git_commit": _git_commit(root),
        "updated_at": _utc_now(),
    }
    # This is the commit marker: every path/hash referenced above already exists.
    atomic_write_json(run_dir / "training_state.json", state)
    # Convenience aliases are published after the commit marker. Resume uses
    # the immutable generation paths above, so a killed alias update cannot
    # invalidate the prior or newly committed state.
    atomic_copy(latest_weights, run_dir / "latest_weights.npz")
    atomic_copy(latest_resume, run_dir / "latest.resume.npz")
    atomic_copy(optimizer_path, run_dir / "optimizer_state.npz")
    atomic_copy(rng_path, run_dir / "rng_state.json")
    atomic_copy(pool_manifest_path, pool_dir / "pool_manifest.json")
    atomic_write_json(run_dir / "latest_checkpoint.json", {
        "rl_games": completed_games,
        "rl_iterations": completed_iterations,
        "checkpoint_path": state["latest_weights_path"],
        "checkpoint_sha256": latest_hash,
        "resume_state_path": state["latest_resume_state_path"],
        "reason": reason,
        "updated_at": state["updated_at"],
    })
    _prune_superseded_latest_payloads(
        run_dir,
        current_generation_paths=(
            latest_weights,
            latest_resume,
            optimizer_path,
            rng_path,
        ),
        current_pool_paths=(
            _resolve(run_dir, entry["path"]) for entry in manifest_entries
        ),
        current_manifest_path=pool_manifest_path,
    )
    return state


def update_diagnostic_markers(
    run_dir,
    *,
    last_periodic_diagnostic_game,
    next_periodic_diagnostic_game,
):
    """Commit diagnostic counters without rewriting immutable RL payloads."""
    run_dir = Path(run_dir)
    state_path = run_dir / "training_state.json"
    with open(state_path, "r", encoding="utf-8") as stream:
        state = json.load(stream)
    state["last_periodic_diagnostic_game"] = int(
        last_periodic_diagnostic_game
    )
    state["next_periodic_diagnostic_game"] = (
        None
        if next_periodic_diagnostic_game is None
        else int(next_periodic_diagnostic_game)
    )
    state["updated_at"] = _utc_now()
    atomic_write_json(state_path, state)
    return state
