"""Canonical seed-addressed supervised artifacts and compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from agents.encoder import DominoEncoder
from training.training_loop import ENCODED_FEATURE_VERSION
from utils.artifacts import atomic_write_json, file_sha256


FORMAT_VERSION = 1
DATASET_FORMAT = "jsonl_state_action_v1"
DATASET_GENERATOR_VERSION = "canonical_real_decisions_v1"
RULESET_VERSION = "two_player_domino_v1"
HEURISTIC_VERSION = "strategic_exact_belief_v1"
NETWORK_ARCHITECTURE = {
    "input_size": DominoEncoder.VECTOR_SIZE,
    "hidden1_size": 256,
    "hidden2_size": 128,
    "output_size": DominoEncoder.ACTION_SIZE,
    "dtype": "float32",
}
EXPECTED_WEIGHT_SHAPES = {
    "W1": (256, DominoEncoder.VECTOR_SIZE),
    "b1": (256, 1),
    "W2": (128, 256),
    "b2": (128, 1),
    "W3": (DominoEncoder.ACTION_SIZE, 128),
    "b3": (DominoEncoder.ACTION_SIZE, 1),
}


class ArtifactCompatibilityError(RuntimeError):
    """Raised when an existing canonical artifact cannot be safely reused."""


@dataclass(frozen=True)
class CanonicalAssetPaths:
    dataset: Path
    dataset_meta: Path
    encoded_cache: Path
    weights: Path
    weights_meta: Path
    loss_plot: Path


@dataclass(frozen=True)
class ArtifactCheck:
    compatible: bool
    status: str
    reasons: tuple[str, ...]
    metadata: dict | None
    sha256: str | None

    def require_compatible_or_missing(self, *, rebuild, label):
        if self.status != "incompatible" or rebuild:
            return
        details = "; ".join(self.reasons)
        raise ArtifactCompatibilityError(
            f"Existing canonical {label} is incompatible: {details}. "
            f"Use the explicit rebuild option to replace it."
        )


def canonical_asset_paths(root, seed):
    """Return canonical dataset/cache/weights paths for one seed."""
    root = Path(root)
    suffix = f"standard_seed{int(seed)}"
    dataset = root / "dataset" / f"supervised_dataset_{suffix}.jsonl"
    weights = root / "models" / f"domino_sl_{suffix}.npz"
    return CanonicalAssetPaths(
        dataset=dataset,
        dataset_meta=dataset.with_suffix(".meta.json"),
        encoded_cache=dataset.with_name(f"{dataset.stem}_encoded.npz"),
        weights=weights,
        weights_meta=weights.with_suffix(".meta.json"),
        loss_plot=weights.with_name(f"{weights.stem}_loss.png"),
    )


def run_scoped_asset_paths(run_dir):
    """Return non-reusable supervised paths owned by one pipeline run."""
    asset_dir = Path(run_dir) / "supervised"
    dataset = asset_dir / "supervised_dataset.jsonl"
    weights = asset_dir / "domino_sl.npz"
    return CanonicalAssetPaths(
        dataset=dataset,
        dataset_meta=dataset.with_suffix(".meta.json"),
        encoded_cache=dataset.with_name("supervised_dataset_encoded.npz"),
        weights=weights,
        weights_meta=weights.with_suffix(".meta.json"),
        loss_plot=weights.with_name("domino_sl_loss.png"),
    )


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


def _created_at():
    return datetime.now(timezone.utc).isoformat()


def _json_value(value):
    """Normalize tuples and NumPy scalars to their persisted JSON form."""
    return json.loads(json.dumps(value))


def canonical_generation_config(*, dataset_games, workers, tuning, safety):
    """Build the structural/configuration identity of a canonical dataset."""
    return _json_value({
        "dataset_games": int(dataset_games),
        "workers": workers,
        "autotune_fraction": float(tuning["fraction"]),
        "autotune_minimum_gain": float(tuning["minimum_gain"]),
        "memory_reserve_mb": int(safety["memory_reserve_mb"]),
        "estimated_worker_mb": int(safety["estimated_worker_mb"]),
        "max_worker_rss_mb": int(safety["max_worker_rss_mb"]),
        "teacher": "StrategicAgent_vs_StrategicAgent",
        "real_decisions_only": True,
    })


def canonical_training_config(**values):
    """Return a JSON-stable supervised training configuration mapping."""
    return _json_value(values)


def _load_metadata(path):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            value = json.load(stream)
    except FileNotFoundError:
        return None, "metadata file is missing"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"metadata cannot be read ({type(exc).__name__}: {exc})"
    if not isinstance(value, dict):
        return None, "metadata root is not an object"
    return value, None


def _compare_fields(metadata, expected):
    reasons = []
    for field, expected_value in expected.items():
        actual = metadata.get(field)
        if actual != expected_value:
            reasons.append(
                f"{field} differs (found {actual!r}, expected {expected_value!r})"
            )
    return reasons


def inspect_canonical_dataset(paths, *, seed, dataset_games, generation_config):
    """Validate dataset metadata, structural versions, configuration, and hash."""
    if not paths.dataset.exists():
        reasons = () if not paths.dataset_meta.exists() else (
            "dataset file is missing while metadata exists",
        )
        return ArtifactCheck(False, "missing", reasons, None, None)
    metadata, error = _load_metadata(paths.dataset_meta)
    if error:
        return ArtifactCheck(False, "incompatible", (error,), metadata, None)
    expected = {
        "format_version": FORMAT_VERSION,
        "artifact_type": "supervised_dataset",
        "seed": int(seed),
        "dataset_games": int(dataset_games),
        "encoder_size": DominoEncoder.VECTOR_SIZE,
        "action_count": DominoEncoder.ACTION_SIZE,
        "dataset_format": DATASET_FORMAT,
        "dataset_generator_version": DATASET_GENERATOR_VERSION,
        "ruleset_version": RULESET_VERSION,
        "heuristic_version": HEURISTIC_VERSION,
        "encoded_feature_version": ENCODED_FEATURE_VERSION,
        "generation_config": _json_value(generation_config),
    }
    reasons = _compare_fields(metadata, expected)
    actual_hash = file_sha256(paths.dataset)
    if metadata.get("dataset_sha256") != actual_hash:
        reasons.append(
            "dataset_sha256 differs from the current dataset file "
            f"({metadata.get('dataset_sha256')!r} != {actual_hash!r})"
        )
    return ArtifactCheck(
        not reasons,
        "reused" if not reasons else "incompatible",
        tuple(reasons),
        metadata,
        actual_hash,
    )


def write_dataset_metadata(
    paths,
    *,
    root,
    seed,
    dataset_games,
    dataset_summary,
    generation_config,
):
    """Publish complete metadata for a newly generated canonical dataset."""
    digest = file_sha256(paths.dataset)
    metadata = {
        "format_version": FORMAT_VERSION,
        "artifact_type": "supervised_dataset",
        "seed": int(seed),
        "dataset_games": int(dataset_games),
        "dataset_examples": int(dataset_summary["saved_turn_count"]),
        "encoder_size": DominoEncoder.VECTOR_SIZE,
        "action_count": DominoEncoder.ACTION_SIZE,
        "dataset_format": DATASET_FORMAT,
        "dataset_generator_version": DATASET_GENERATOR_VERSION,
        "ruleset_version": RULESET_VERSION,
        "heuristic_version": HEURISTIC_VERSION,
        "encoded_feature_version": ENCODED_FEATURE_VERSION,
        "git_commit": _git_commit(root),
        "created_at": _created_at(),
        "dataset_sha256": digest,
        "generation_config": _json_value(generation_config),
    }
    atomic_write_json(paths.dataset_meta, metadata)
    return metadata


def _inspect_weight_archive(path):
    reasons = []
    try:
        with np.load(path, allow_pickle=False) as archive:
            for name, expected_shape in EXPECTED_WEIGHT_SHAPES.items():
                if name not in archive:
                    reasons.append(f"weights archive is missing {name}")
                    continue
                if tuple(archive[name].shape) != expected_shape:
                    reasons.append(
                        f"{name} shape differs (found {archive[name].shape}, "
                        f"expected {expected_shape})"
                    )
    except (OSError, ValueError) as exc:
        reasons.append(f"weights archive cannot be read ({type(exc).__name__}: {exc})")
    return reasons


def inspect_canonical_weights(
    paths,
    *,
    seed,
    dataset_sha256,
    training_config,
):
    """Validate supervised weights, origin dataset, architecture, and hash."""
    if not paths.weights.exists():
        reasons = () if not paths.weights_meta.exists() else (
            "weights file is missing while metadata exists",
        )
        return ArtifactCheck(False, "missing", reasons, None, None)
    metadata, error = _load_metadata(paths.weights_meta)
    if error:
        return ArtifactCheck(False, "incompatible", (error,), metadata, None)
    expected = {
        "format_version": FORMAT_VERSION,
        "artifact_type": "supervised_weights",
        "seed": int(seed),
        "dataset_sha256": dataset_sha256,
        "encoder_size": DominoEncoder.VECTOR_SIZE,
        "action_count": DominoEncoder.ACTION_SIZE,
        "network_architecture": NETWORK_ARCHITECTURE,
        "ruleset_version": RULESET_VERSION,
        "encoded_feature_version": ENCODED_FEATURE_VERSION,
        "training_config": _json_value(training_config),
    }
    reasons = _compare_fields(metadata, expected)
    reasons.extend(_inspect_weight_archive(paths.weights))
    actual_hash = file_sha256(paths.weights)
    if metadata.get("weights_sha256") != actual_hash:
        reasons.append(
            "weights_sha256 differs from the current weights file "
            f"({metadata.get('weights_sha256')!r} != {actual_hash!r})"
        )
    return ArtifactCheck(
        not reasons,
        "reused" if not reasons else "incompatible",
        tuple(reasons),
        metadata,
        actual_hash,
    )


def write_weights_metadata(
    paths,
    *,
    root,
    seed,
    dataset_sha256,
    training_config,
    training_summary,
):
    """Publish provenance and convergence metadata for supervised weights."""
    digest = file_sha256(paths.weights)
    metadata = {
        "format_version": FORMAT_VERSION,
        "artifact_type": "supervised_weights",
        "seed": int(seed),
        "dataset_path": str(paths.dataset),
        "dataset_sha256": dataset_sha256,
        "weights_path": str(paths.weights),
        "weights_sha256": digest,
        "encoder_size": DominoEncoder.VECTOR_SIZE,
        "action_count": DominoEncoder.ACTION_SIZE,
        "network_architecture": NETWORK_ARCHITECTURE,
        "ruleset_version": RULESET_VERSION,
        "encoded_feature_version": ENCODED_FEATURE_VERSION,
        "training_config": _json_value(training_config),
        "max_epochs": int(training_summary["requested_epochs"]),
        "epochs_completed": int(training_summary["epochs"]),
        "best_epoch": training_summary.get("best_epoch"),
        "best_validation_loss": float(training_summary["best_validation_loss"]),
        "early_stopping_triggered": bool(
            training_summary.get("early_stopping_triggered")
        ),
        "stopping_reason": training_summary.get("stopping_reason"),
        "final_training_loss": training_summary.get("final_training_loss"),
        "final_validation_loss": training_summary.get("final_validation_loss"),
        "git_commit": _git_commit(root),
        "created_at": _created_at(),
    }
    atomic_write_json(paths.weights_meta, metadata)
    return metadata
