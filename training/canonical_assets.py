"""Canonical seed-addressed supervised artifacts and compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from agents.encoder import DominoEncoder
from agents.network_architecture import DEFAULT_NETWORK_ARCHITECTURE
from agents.nn import (
    TP_MIN_EPOCHS,
    TP_MIN_RELATIVE_IMPROVEMENT,
    TP_PATIENCE_BLOCKS,
    TP_WINDOW_EPOCHS,
)
from training.utils.encoding import ENCODED_FEATURE_VERSION
from utils.artifacts import atomic_write_json, file_sha256
from utils.myrandom import DEFAULT_BIT_GENERATOR, DERIVATION_SCHEME
from utils.repository import current_git_commit
from middleware.rulesets import DEFAULT_RULESET_NAME, resolve_ruleset


FORMAT_VERSION = 1
SUPERVISED_WEIGHTS_FORMAT_VERSION = 2
DATASET_FORMAT = "jsonl_state_action_v1"
DATASET_GENERATOR_VERSION = "canonical_real_decisions_numpy_seed_plan_v2"
RULESET_VERSION = "two_player_domino_v1"
HEURISTIC_VERSION = "strategic_exact_belief_v1"
EXPECTED_WEIGHT_SHAPES = DEFAULT_NETWORK_ARCHITECTURE.policy_weight_shapes()
SUPERVISED_EXECUTION_CONFIG_FIELDS = frozenset({
    "device",
    "gpu_memory_reserve_mb",
    "memory_reserve_mb",
})
SUPERVISED_RANDOM_CONFIG_FIELDS = frozenset({
    "random_bit_generator",
    "random_derivation_scheme",
})


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


def canonical_asset_paths(
    root,
    seed,
    ruleset=DEFAULT_RULESET_NAME,
    *,
    use_opponent_suit_features=True,
):
    """Return canonical dataset/cache/weights paths for one seed.

    Reusable assets are addressed by seed, so two runs that differ only in the
    ablation flag would otherwise fight over one supervised checkpoint: the
    second would refuse to start on an encoder-size mismatch, and retraining it
    would leave the first unresumable. The ablated regime therefore claims its
    own suffix. The enabled path is byte-identical to the historical one, so no
    existing artifact is renamed.
    """
    root = Path(root)
    ruleset = resolve_ruleset(ruleset)
    ruleset_prefix = (
        "" if ruleset.name == DEFAULT_RULESET_NAME else f"{ruleset.name}_"
    )
    suffix = f"{ruleset_prefix}standard_seed{int(seed)}"
    if not use_opponent_suit_features:
        suffix = f"{suffix}_nosuit"
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


def _json_value(value):
    """Normalize tuples and NumPy scalars to their persisted JSON form."""
    return json.loads(json.dumps(value))


def canonical_generation_config(
    *,
    dataset_games,
    workers,
    tuning,
    safety,
    ruleset=DEFAULT_RULESET_NAME,
):
    """Build the structural/configuration identity of a canonical dataset."""
    return _json_value({
        "dataset_games": int(dataset_games),
        "ruleset_name": resolve_ruleset(ruleset).name,
        "workers": workers,
        "autotune_fraction": float(tuning["fraction"]),
        "autotune_minimum_gain": float(tuning["minimum_gain"]),
        "memory_reserve_mb": int(safety["memory_reserve_mb"]),
        "estimated_worker_mb": int(safety["estimated_worker_mb"]),
        "max_worker_rss_mb": int(safety["max_worker_rss_mb"]),
        "teacher": "StrategicAgent_vs_StrategicAgent",
        "real_decisions_only": True,
        "random_bit_generator": DEFAULT_BIT_GENERATOR,
        "random_derivation_scheme": DERIVATION_SCHEME,
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


def _compare_nested_value(metadata, path, expected):
    """Return one mismatch for a dotted metadata path, if it differs."""
    actual = metadata
    for field in path.split("."):
        if not isinstance(actual, dict):
            actual = None
            break
        actual = actual.get(field)
    if actual == expected:
        return []
    return [f"{path} differs (found {actual!r}, expected {expected!r})"]


def _split_supervised_training_config(training_config):
    """Separate learning, randomization, and runtime configuration."""
    normalized = _json_value(dict(training_config or {}))
    execution = {
        field: normalized[field]
        for field in sorted(SUPERVISED_EXECUTION_CONFIG_FIELDS)
        if field in normalized
    }
    randomization = {
        field: normalized[field]
        for field in sorted(SUPERVISED_RANDOM_CONFIG_FIELDS)
        if field in normalized
    }
    excluded = SUPERVISED_EXECUTION_CONFIG_FIELDS | SUPERVISED_RANDOM_CONFIG_FIELDS
    hyperparameters = {
        field: value
        for field, value in normalized.items()
        if field not in excluded
    }
    return hyperparameters, randomization, execution


def _fixed_tp_policy():
    """Return the immutable supervised TP policy recorded in every manifest."""
    return {
        "name": "fixed_block_median_training_plateau_v1",
        "window_epochs": TP_WINDOW_EPOCHS,
        "patience_blocks": TP_PATIENCE_BLOCKS,
        "minimum_epochs": TP_MIN_EPOCHS,
        "minimum_relative_improvement": TP_MIN_RELATIVE_IMPROVEMENT,
    }


def _weights_dataset_section(dataset_metadata):
    """Return dataset origin and creation parameters for a weights manifest."""
    generation_parameters = dict(dataset_metadata["generation_config"])
    generation_parameters.pop("dataset_games", None)
    return {
        "sha256": dataset_metadata["dataset_sha256"],
        "seed": int(dataset_metadata["seed"]),
        "games": int(dataset_metadata["dataset_games"]),
        "examples": int(dataset_metadata["dataset_examples"]),
        "creation": {
            "format": dataset_metadata["dataset_format"],
            "generator_version": dataset_metadata["dataset_generator_version"],
            "heuristic_version": dataset_metadata["heuristic_version"],
            "parameters": _json_value(generation_parameters),
        },
    }


def _resolved_supervised_execution(training_summary):
    """Return concise machine-dependent choices made during one SL run."""
    return {
        "requested_device": training_summary.get("requested_device"),
        "selected_device": training_summary.get("selected_device"),
        "device_fallback_reason": training_summary.get("device_fallback_reason"),
        "host_storage_mode": training_summary.get("host_storage_mode"),
        "storage_mode": training_summary.get("storage_mode"),
        "resident_window_examples": training_summary.get(
            "resident_window_examples"
        ),
        "full_dataset_on_gpu": training_summary.get("full_dataset_on_gpu"),
    }


def inspect_canonical_dataset(
    paths,
    *,
    seed,
    dataset_games,
    generation_config,
    ruleset=DEFAULT_RULESET_NAME,
    use_opponent_suit_features=True,
):
    """Validate dataset metadata, structural versions, configuration, and hash."""
    if not paths.dataset.exists():
        reasons = () if not paths.dataset_meta.exists() else (
            "dataset file is missing while metadata exists",
        )
        return ArtifactCheck(False, "missing", reasons, None, None)
    metadata, error = _load_metadata(paths.dataset_meta)
    if error:
        return ArtifactCheck(False, "incompatible", (error,), metadata, None)
    ruleset = resolve_ruleset(ruleset)
    encoder = DominoEncoder(
        ruleset,
        use_opponent_suit_features=use_opponent_suit_features,
    )
    metadata = dict(metadata)
    if ruleset.name == DEFAULT_RULESET_NAME:
        metadata.setdefault("ruleset_name", DEFAULT_RULESET_NAME)
        stored_generation = dict(metadata.get("generation_config") or {})
        stored_generation.setdefault("ruleset_name", DEFAULT_RULESET_NAME)
        metadata["generation_config"] = stored_generation
    expected = {
        "format_version": FORMAT_VERSION,
        "artifact_type": "supervised_dataset",
        "seed": int(seed),
        "dataset_games": int(dataset_games),
        "ruleset_name": ruleset.name,
        "encoder_size": encoder.vector_size,
        "action_count": encoder.action_size,
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
    ruleset=DEFAULT_RULESET_NAME,
    use_opponent_suit_features=True,
):
    """Publish complete metadata for a newly generated canonical dataset."""
    ruleset = resolve_ruleset(ruleset)
    encoder = DominoEncoder(
        ruleset,
        use_opponent_suit_features=use_opponent_suit_features,
    )
    digest = file_sha256(paths.dataset)
    metadata = {
        "format_version": FORMAT_VERSION,
        "artifact_type": "supervised_dataset",
        "seed": int(seed),
        "dataset_games": int(dataset_games),
        "dataset_examples": int(dataset_summary["saved_turn_count"]),
        "ruleset_name": ruleset.name,
        "encoder_size": encoder.vector_size,
        "action_count": encoder.action_size,
        "dataset_format": DATASET_FORMAT,
        "dataset_generator_version": DATASET_GENERATOR_VERSION,
        "ruleset_version": RULESET_VERSION,
        "heuristic_version": HEURISTIC_VERSION,
        "encoded_feature_version": ENCODED_FEATURE_VERSION,
        "git_commit": current_git_commit(root),
        "dataset_sha256": digest,
        "generation_config": _json_value(generation_config),
    }
    atomic_write_json(paths.dataset_meta, metadata)
    return metadata


def _inspect_weight_archive(path, expected_weight_shapes):
    reasons = []
    try:
        with np.load(path, allow_pickle=False) as archive:
            for name, expected_shape in expected_weight_shapes.items():
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
    dataset_metadata,
    training_config,
    architecture=DEFAULT_NETWORK_ARCHITECTURE,
    ruleset=DEFAULT_RULESET_NAME,
    use_opponent_suit_features=True,
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
    ruleset = resolve_ruleset(ruleset)
    encoder = DominoEncoder(
        ruleset,
        use_opponent_suit_features=use_opponent_suit_features,
    )
    hyperparameters, randomization, execution = (
        _split_supervised_training_config(training_config)
    )
    model_metadata = metadata.setdefault("model", {})
    if ruleset.name == DEFAULT_RULESET_NAME:
        model_metadata.setdefault("ruleset_name", DEFAULT_RULESET_NAME)
    comparisons = {
        "format_version": SUPERVISED_WEIGHTS_FORMAT_VERSION,
        "artifact.type": "supervised_weights",
        "dataset": _weights_dataset_section(dataset_metadata),
        "model.ruleset_name": ruleset.name,
        "model.encoder_size": encoder.vector_size,
        "model.action_count": encoder.action_size,
        "model.network_architecture": architecture.as_dict(),
        "training.hyperparameters": hyperparameters,
        "training.randomization": {"root_seed": int(seed), **randomization},
        "training.fixed_tp_policy": _fixed_tp_policy(),
        "execution.requested": execution,
        "contracts.ruleset_version": RULESET_VERSION,
        "contracts.encoded_feature_version": ENCODED_FEATURE_VERSION,
    }
    reasons = []
    for path, expected in comparisons.items():
        reasons.extend(_compare_nested_value(metadata, path, expected))
    reasons.extend(
        _inspect_weight_archive(
            paths.weights,
            architecture.policy_weight_shapes(),
        )
    )
    actual_hash = file_sha256(paths.weights)
    stored_hash = (metadata.get("artifact") or {}).get("sha256")
    if stored_hash != actual_hash:
        reasons.append(
            "artifact.sha256 differs from the current weights file "
            f"({stored_hash!r} != {actual_hash!r})"
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
    dataset_metadata,
    training_config,
    training_summary,
    architecture=DEFAULT_NETWORK_ARCHITECTURE,
    ruleset=DEFAULT_RULESET_NAME,
    use_opponent_suit_features=True,
):
    """Publish provenance and convergence metadata for supervised weights."""
    ruleset = resolve_ruleset(ruleset)
    encoder = DominoEncoder(
        ruleset,
        use_opponent_suit_features=use_opponent_suit_features,
    )
    digest = file_sha256(paths.weights)
    hyperparameters, randomization, execution = (
        _split_supervised_training_config(training_config)
    )
    metadata = {
        "format_version": SUPERVISED_WEIGHTS_FORMAT_VERSION,
        "artifact": {
            "type": "supervised_weights",
            "sha256": digest,
        },
        "dataset": _weights_dataset_section(dataset_metadata),
        "model": {
            "ruleset_name": ruleset.name,
            "encoder_size": encoder.vector_size,
            "action_count": encoder.action_size,
            "network_architecture": architecture.as_dict(),
        },
        "training": {
            "hyperparameters": hyperparameters,
            "randomization": {"root_seed": int(seed), **randomization},
            "fixed_tp_policy": _fixed_tp_policy(),
            "result": {
                "epochs_completed": int(training_summary["epochs"]),
                "best_epoch": training_summary.get("best_epoch"),
                "best_validation_loss": float(
                    training_summary["best_validation_loss"]
                ),
                "early_stopping_triggered": bool(
                    training_summary.get("early_stopping_triggered")
                ),
                "stopping_reason": training_summary.get("stopping_reason"),
                "final_training_loss": training_summary.get(
                    "final_training_loss"
                ),
                "final_validation_loss": training_summary.get(
                    "final_validation_loss"
                ),
            },
        },
        "execution": {
            "requested": execution,
            "resolved": _resolved_supervised_execution(training_summary),
        },
        "contracts": {
            "ruleset_version": RULESET_VERSION,
            "encoded_feature_version": ENCODED_FEATURE_VERSION,
        },
        "repository": {
            "git_commit": current_git_commit(root),
        },
    }
    atomic_write_json(paths.weights_meta, metadata)
    return metadata
