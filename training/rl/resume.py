"""Atomic policy checkpoints and exact RL resume-state compatibility."""

from collections import deque
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import random
import secrets

import numpy as np

from agents.encoder import DominoEncoder
from agents.network_architecture import (
    architecture_for_ruleset,
    hidden_layer_count_from_weights,
    policy_layer_names,
)
from agents.nn import DISABLED_DROPOUT_RATE
from agents.rl_nn import PolicyNetwork
from middleware.domino_engine import RULESET_VERSION
from middleware.rulesets import DEFAULT_RULESET_NAME, resolve_ruleset
from training.rl.ppo import (
    POLICY_GRADIENT_CLIP_NORM,
    PPO_TRAINING_ALGORITHM,
    REINFORCE_TRAINING_ALGORITHM,
    fixed_ppo_policy,
    ppo_is_enabled,
)
from training.rl import baseline as baselines
from training.rl.checkpoint_archive import archive_policy_manifest
from training.rl.matchmaking import matchmaking_policy_manifest
from training.rl.pool import pool_policy_manifest
from training.rl.reward_distance import (
    HISTORICAL_GAMMA_F,
    HISTORICAL_REWARD_DISTANCE_MODE,
    resolve_reward_distance_mode,
)
from training.rl.rollout import GAMMA_I, REWARD_ETA
from utils.repository import current_git_commit


# Version 12 makes the neural opponent bands chronologically disjoint:
# ``medium_term`` became a delayed archive-backed window and
# ``historical_uniform`` was added. Versions 10 and 11 stored an overlapping
# ``medium_term`` whose members were also ``recent`` members, which cannot be
# reinterpreted under the new semantics without silently changing what the
# saved memberships mean. No converter exists, so they are rejected rather
# than migrated.
# Version 16 persists the independently configurable local/terminal reward
# distance metrics. Version 15 is still unambiguous: absence of the field means
# the historical turn-decision behavior, including the historical numeric
# defaults when an unusually old durable config omitted those too.
RESUME_STATE_VERSION = 16
SUPPORTED_RESUME_STATE_VERSIONS = (15, RESUME_STATE_VERSION)
OVERLAPPING_MEDIUM_TERM_STATE_VERSIONS = (10, 11)
# States written before the uniform member remainder rotated. They carry no
# rotation anchor, so their next match plan cannot be reproduced.
PRE_UNIFORM_ROTATION_STATE_VERSIONS = (12,)
# States written before the pool carried durable champion state.
PRE_CHAMPION_STATE_VERSIONS = (13,)
# States written while a single champion bucket existed, so the pool stored one
# unqualified champion queue, event count, and score map instead of one set per
# champion bucket.
PRE_PER_BUCKET_CHAMPION_STATE_VERSIONS = (14,)
NUMBERED_CHECKPOINT_WEIGHT_RETENTION = 5

# Persisted spellings of the three RL parameters renamed after these runs began.
# Reading both forms keeps every checkpoint and run config resumable; see
# ``training.canonical_run.identity_spelling`` for the identity side.
RENAMED_PARAMETER_KEYS = {
    "gamma_f": "gamma",
    "reward_eta": "alpha",
    "gamma_i": "event_reward_decay",
}
HISTORICAL_REWARD_PARAMETER_DEFAULTS = {
    "gamma_f": HISTORICAL_GAMMA_F,
    "reward_eta": REWARD_ETA,
    "gamma_i": GAMMA_I,
}


def run_config_uses_opponent_bucket_features(run_config):
    """Return whether one saved run appends the opponent-bucket one-hot.

    Read from ``locked_arguments`` rather than ``rl_config`` for the reason
    ``training.canonical_run.run_config_uses_opponent_suit_features`` documents:
    ``rl_config`` is rebuilt from arguments on every invocation and compared as
    an immutable run key, so a new member there would make every run created
    before the flag existed unresumable. It lives here rather than beside its
    sibling in ``training.canonical_run`` because that module imports this one,
    and this module needs the value; ``canonical_run`` re-exports it.
    """
    locked = (run_config or {}).get("locked_arguments") or {}
    return bool(locked.get("opponent_bucket_features", False))


def _renamed(rl_config, current_name):
    """Read one renamed RL parameter from a config written either way."""
    if current_name in rl_config:
        return rl_config[current_name]
    original_name = RENAMED_PARAMETER_KEYS[current_name]
    if original_name in rl_config:
        return rl_config[original_name]
    return HISTORICAL_REWARD_PARAMETER_DEFAULTS[current_name]


@dataclass(frozen=True)
class RLTrainingConfiguration:
    """Typed durable RL resume configuration and source provenance."""

    total_training_games: int
    ruleset_name: str
    selected_gpi: int
    selected_workers: int
    log_interval: int
    checkpoint_interval: int
    moving_average_window: int
    opponent_buckets: tuple[str, ...]
    difficulty_weight: float
    opponent_decision_restarts: bool
    learning_rate: float
    entropy_coef: float
    use_value_head: bool
    value_coef: float
    gamma_f: float
    reward_eta: float
    gamma_i: float
    reward_distance_mode: str
    normalize_advantages: bool
    # Recorded even though the encoder ablations are usually caught by the
    # checkpoint's own input width, because these two are not independent: the
    # suit block and the bucket block differ in width by zero for double-six,
    # so dropping the first while adding the second reproduces the historical
    # 168 inputs with seven completely different trailing features. A shape
    # check cannot see that swap; this field can.
    use_opponent_bucket_features: bool
    # JSON mapping (or None) rather than the typed spec, so one configuration
    # stays directly comparable between a checkpoint and a run config.
    baseline: dict | None
    baseline_artifact_sha256: str | None
    weight_decay: float
    dropout_rate: float
    effective_seed: int
    device: str
    sl_weights_sha256: str | None
    ppo_max_epochs: int
    worker_memory_reserve_mb: int
    worker_estimated_mb: int
    worker_max_rss_mb: int
    opponent_system_policy: dict
    run_configuration_sha256: str | None = None
    git_commit: str | None = None

    @property
    def rl_training_algorithm(self):
        """Return the algorithm implied by the persisted epoch budget."""
        return (
            PPO_TRAINING_ALGORITHM
            if ppo_is_enabled(self.ppo_max_epochs)
            else REINFORCE_TRAINING_ALGORITHM
        )

    @classmethod
    def from_mapping(cls, value):
        """Build the typed configuration from a runtime/checkpoint mapping."""
        data = dict(value)
        data.setdefault("ruleset_name", DEFAULT_RULESET_NAME)
        data["ruleset_name"] = resolve_ruleset(data["ruleset_name"]).name
        data.setdefault("run_configuration_sha256", None)
        data.setdefault("git_commit", None)
        data.setdefault("opponent_decision_restarts", False)
        # Checkpoints written before the flag existed never had the block.
        data.setdefault("use_opponent_bucket_features", False)
        data["use_opponent_bucket_features"] = bool(
            data["use_opponent_bucket_features"]
        )
        # Checkpoints written before --baseline existed record no baseline,
        # which resolves to the one the run already used.
        data.setdefault("baseline", None)
        if data["baseline"] is not None:
            data["baseline"] = baselines.BaselineSpec.from_mapping(
                data["baseline"]
            ).as_mapping()
        data.setdefault("baseline_artifact_sha256", None)
        data.setdefault(
            "reward_distance_mode",
            HISTORICAL_REWARD_DISTANCE_MODE,
        )
        resolve_reward_distance_mode(data["reward_distance_mode"])
        for name, default in HISTORICAL_REWARD_PARAMETER_DEFAULTS.items():
            original_name = RENAMED_PARAMETER_KEYS[name]
            if name not in data and original_name not in data:
                data[name] = default
        # Checkpoints written before the parameter rename spell these three the
        # original way; a resume must keep working across the rename.
        for current_name, original_name in RENAMED_PARAMETER_KEYS.items():
            if current_name not in data and original_name in data:
                data[current_name] = data[original_name]
        data["opponent_buckets"] = tuple(data["opponent_buckets"])
        configuration = cls(**{
            field.name: data[field.name]
            for field in fields(cls)
        })
        recorded_algorithm = data.get("rl_training_algorithm")
        if (
            recorded_algorithm is not None
            and recorded_algorithm != configuration.rl_training_algorithm
        ):
            raise ValueError(
                "RL algorithm disagrees with the persisted ppo_max_epochs."
            )
        return configuration

    @classmethod
    def from_run_config(
        cls,
        run_config,
        *,
        total_training_games,
        selected_workers,
        device,
    ):
        """Build the exact RL identity from one immutable canonical run config."""
        rl = run_config["rl_config"]
        ppo = run_config["ppo_config"]
        ruleset_name = run_config.get("ruleset_name", DEFAULT_RULESET_NAME)
        baseline = baselines.from_run_config(run_config)
        configuration = cls.from_mapping({
            "total_training_games": int(total_training_games),
            "ruleset_name": ruleset_name,
            "selected_gpi": int(rl["games_per_iteration"]),
            "selected_workers": int(selected_workers),
            "log_interval": int(rl["log_interval"]),
            "checkpoint_interval": int(rl["checkpoint_interval"]),
            "moving_average_window": int(rl["moving_average_window"]),
            "opponent_buckets": tuple(rl["opponent_buckets"]),
            "difficulty_weight": float(rl["difficulty_weight"]),
            "opponent_decision_restarts": bool(
                rl.get("opponent_decision_restarts", False)
            ),
            "learning_rate": float(rl["learning_rate"]),
            "entropy_coef": float(rl["entropy_coef"]),
            "use_value_head": bool(rl["use_value_head"]),
            "value_coef": float(rl["value_coef"]),
            "gamma_f": float(_renamed(rl, "gamma_f")),
            "reward_eta": float(_renamed(rl, "reward_eta")),
            "gamma_i": float(_renamed(rl, "gamma_i")),
            "reward_distance_mode": rl.get(
                "reward_distance_mode",
                HISTORICAL_REWARD_DISTANCE_MODE,
            ),
            "normalize_advantages": bool(rl["normalize_advantages"]),
            # From locked_arguments, not rl_config, for the same reason the
            # baseline is; see ``training.canonical_run``.
            "use_opponent_bucket_features": run_config_uses_opponent_bucket_features(
                run_config
            ),
            # From locked_arguments, not rl_config; see
            # ``training.rl.baseline.from_run_config``.
            "baseline": baseline,
            "baseline_artifact_sha256": baselines.artifact_sha256(
                baseline,
                ruleset_name,
            ),
            "weight_decay": float(rl["weight_decay"]),
            "dropout_rate": float(rl["dropout_rate"]),
            "effective_seed": int(run_config["seed"]),
            "device": device,
            "sl_weights_sha256": run_config["supervised_weights_sha256"],
            "ppo_max_epochs": int(ppo["max_epochs"]),
            "worker_memory_reserve_mb": int(rl["worker_memory_reserve_mb"]),
            "worker_estimated_mb": int(rl["worker_estimated_mb"]),
            "worker_max_rss_mb": int(rl["worker_max_rss_mb"]),
            "opponent_system_policy": {
                "pool": pool_policy_manifest(rl["opponent_buckets"]),
                "matchmaking": matchmaking_policy_manifest(),
                "checkpoint_archive": archive_policy_manifest(),
            },
            "run_configuration_sha256": run_config["configuration_sha256"],
            "git_commit": run_config.get("git_commit"),
        })
        if run_config["algorithm"] != configuration.rl_training_algorithm:
            raise ValueError(
                "run_config.json algorithm disagrees with ppo_max_epochs."
            )
        return configuration

    def to_dict(self):
        """Return the durable checkpoint representation, including constants."""
        value = asdict(self)
        value["opponent_buckets"] = list(self.opponent_buckets)
        return {
            **value,
            "clip_grad_norm": POLICY_GRADIENT_CLIP_NORM,
            "rl_training_algorithm": self.rl_training_algorithm,
            "ruleset_version": RULESET_VERSION,
            "ppo_configuration": fixed_ppo_policy(self.ppo_max_epochs),
        }

    def warn_if_commit_changed(self, emit_status):
        """Warn, but never reject, when code changed since the run began."""
        current = current_git_commit()
        if self.git_commit and current and self.git_commit != current:
            emit_status(
                "Warning: the repository commit changed since this RL run "
                f"started ({self.git_commit[:12]} -> {current[:12]}). "
                "Resume will continue with the original saved configuration."
            )


def _checkpoint_shape(network):
    """Return the encoder-facing input and output widths of one network."""
    output_layer = getattr(network, f"W{network.layer_count}")
    return int(network.W1.shape[1]), int(output_layer.shape[0])


def _checkpoint_matches_encoder(
    network,
    ruleset=DEFAULT_RULESET_NAME,
    *,
    use_opponent_suit_features=True,
    use_opponent_bucket_features=False,
):
    """Return True when a loaded checkpoint matches the current encoder shape."""
    encoder = DominoEncoder(
        ruleset,
        use_opponent_suit_features=use_opponent_suit_features,
        use_opponent_bucket_features=use_opponent_bucket_features,
    )
    input_size, output_size = _checkpoint_shape(network)
    return (
        input_size == encoder.vector_size
        and output_size == encoder.action_size
    )


def _load_initial_network(
    learning_rate,
    sl_weights_path,
    rl_weights_path,
    quiet=False,
    use_value_head=False,
    critic_updates_trunk=True,
    critic_owns_network=False,
    device="auto",
    fresh_from_sl=False,
    expected_training_algorithm=None,
    weight_decay=0.0,
    dropout_rate=DISABLED_DROPOUT_RATE,
    ruleset=DEFAULT_RULESET_NAME,
    initialization_seed=None,
    use_opponent_suit_features=True,
    use_opponent_bucket_features=False,
):
    """Load RL/SL weights or create a compatible random policy as fallback.

    ``fresh_from_sl=True`` ignores ``rl_weights_path`` as an initialization
    source while leaving that file intact until the completed new model
    atomically replaces it. If the requested SL checkpoint does not exist, a
    ruleset-default architecture is initialized from ``initialization_seed``.
    """
    if rl_weights_path is not None and not fresh_from_sl:
        try:
            network = PolicyNetwork.load(
                rl_weights_path,
                learning_rate=learning_rate,
                use_value_head=use_value_head,
                critic_updates_trunk=critic_updates_trunk,
                critic_owns_network=critic_owns_network,
                device=device,
                weight_decay=weight_decay,
                dropout_rate=dropout_rate,
            )
            encoder = DominoEncoder(
                ruleset,
                use_opponent_suit_features=use_opponent_suit_features,
                use_opponent_bucket_features=use_opponent_bucket_features,
            )
            if not _checkpoint_matches_encoder(
                network,
                ruleset,
                use_opponent_suit_features=use_opponent_suit_features,
                use_opponent_bucket_features=use_opponent_bucket_features,
            ):
                input_size, output_size = _checkpoint_shape(network)
                raise ValueError(
                    f"RL checkpoint {rl_weights_path} has shape "
                    f"input={input_size}, output={output_size}, "
                    f"but ruleset {encoder.ruleset.name!r} expects "
                    f"input={encoder.vector_size}, output={encoder.action_size}."
                )
            saved_algorithm = getattr(network, "rl_training_algorithm", None)
            if expected_training_algorithm is not None:
                if saved_algorithm is None:
                    if expected_training_algorithm != REINFORCE_TRAINING_ALGORITHM:
                        raise ValueError(
                            f"RL checkpoint {rl_weights_path} predates algorithm "
                            "metadata and cannot be continued as PPO implicitly. "
                            "Use --fresh-from-sl for a new PPO run or "
                            "--ppo-max-epochs 1 to select REINFORCE."
                        )
                elif saved_algorithm != expected_training_algorithm:
                    raise ValueError(
                        "RL checkpoint rl_training_algorithm is "
                        f"{saved_algorithm!r}, but "
                        f"{expected_training_algorithm!r} was requested."
                    )
                network.rl_training_algorithm = expected_training_algorithm
            if not quiet:
                print(f"Resuming RL training from {rl_weights_path}")
            return network
        except FileNotFoundError:
            pass

    encoder = DominoEncoder(
        ruleset,
        use_opponent_suit_features=use_opponent_suit_features,
        use_opponent_bucket_features=use_opponent_bucket_features,
    )
    try:
        network = PolicyNetwork.load_from_sl(
            sl_weights_path,
            learning_rate=learning_rate,
            use_value_head=use_value_head,
            critic_updates_trunk=critic_updates_trunk,
            critic_owns_network=critic_owns_network,
            device=device,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
        )
    except FileNotFoundError:
        architecture = architecture_for_ruleset(
            ruleset,
            use_opponent_suit_features=use_opponent_suit_features,
            use_opponent_bucket_features=use_opponent_bucket_features,
        )
        network = PolicyNetwork(
            input_size=architecture.input_size,
            output_size=architecture.output_size,
            hidden_sizes=architecture.hidden_sizes,
            learning_rate=learning_rate,
            random_seed=(
                None
                if initialization_seed is None
                else int(initialization_seed) & 0xFFFFFFFF
            ),
            use_value_head=use_value_head,
            critic_updates_trunk=critic_updates_trunk,
            critic_owns_network=critic_owns_network,
            device=device,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
        )
        if expected_training_algorithm is not None:
            network.rl_training_algorithm = expected_training_algorithm
        if not quiet:
            print(
                f"Supervised weights not found at {sl_weights_path}; "
                "initializing a random RL policy with default "
                f"{encoder.ruleset.name} architecture "
                f"{architecture.hidden_sizes}."
            )
        return network
    if not _checkpoint_matches_encoder(
        network,
        ruleset,
        use_opponent_suit_features=use_opponent_suit_features,
        use_opponent_bucket_features=use_opponent_bucket_features,
    ):
        input_size, output_size = _checkpoint_shape(network)
        raise ValueError(
            f"SL checkpoint {sl_weights_path} has shape "
            f"input={input_size}, output={output_size}, "
            f"but ruleset {encoder.ruleset.name!r} expects "
            f"input={encoder.vector_size}, output={encoder.action_size}. "
            "Regenerate the supervised dataset and retrain SL first."
        )
    if expected_training_algorithm is not None:
        network.rl_training_algorithm = expected_training_algorithm
    if not quiet:
        print(f"Initializing RL policy from supervised weights: {sl_weights_path}")
    return network


def numbered_checkpoint_path(base_path, iteration):
    """Return an iteration-suffixed checkpoint path derived from ``base_path``."""
    path = Path(base_path)
    suffix = path.suffix or ".npz"
    stem = path.name[:-len(path.suffix)] if path.suffix else path.name
    return path.with_name(f"{stem}_iter{int(iteration):06d}{suffix}")


def resume_state_path(weights_path):
    """Return the auxiliary resume-state path paired with a weights checkpoint."""
    path = Path(weights_path)
    suffix = path.suffix or ".npz"
    stem = path.name[:-len(path.suffix)] if path.suffix else path.name
    return path.with_name(f"{stem}.resume{suffix}")


def _prune_numbered_checkpoint_weights(
    base_path,
    current_weights_path,
    *,
    keep=NUMBERED_CHECKPOINT_WEIGHT_RETENTION,
):
    """Retain a small rolling window of numbered policy checkpoints."""
    keep = int(keep)
    if keep < 1:
        raise ValueError("keep must be positive")
    base = Path(base_path)
    suffix = base.suffix or ".npz"
    stem = base.name[:-len(base.suffix)] if base.suffix else base.name
    prefix = f"{stem}_iter"
    candidates = []
    for path in base.parent.glob(f"{prefix}*{suffix}"):
        iteration_text = path.name[len(prefix):-len(suffix)]
        if iteration_text.isdigit() and path.is_file():
            candidates.append((int(iteration_text), path))

    current = Path(current_weights_path).resolve()
    retained = {current}
    for _iteration, path in sorted(candidates, reverse=True):
        if len(retained) >= keep:
            break
        retained.add(path.resolve())

    removed = []
    for _iteration, path in candidates:
        if path.resolve() not in retained:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            else:
                removed.append(path)
    return removed


def _file_sha256(path):
    """Return a streaming SHA-256 digest without duplicating checkpoint memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_network_save(network, path):
    """Publish a complete network file with one same-directory atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.tmp-{os.getpid()}-{secrets.token_hex(4)}.npz"
    )
    try:
        network.save(str(temporary))
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_resume_state_save(path, metadata, pool_state, pool_weights):
    """Atomically save metadata and one weight set per neural opponent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "pool_weight_count": np.asarray(len(pool_weights), dtype=np.int64),
    }
    serialization = pool_state.get("weight_serialization", ())
    if len(serialization) != len(pool_weights):
        raise ValueError("Pool weight serialization does not match neural weights")
    for item in serialization:
        snapshot_index = int(item["index"])
        opponent_id = item["opponent_id"]
        snapshot = pool_weights[opponent_id]
        for name in policy_layer_names(
            hidden_layer_count_from_weights(snapshot)
        ):
            arrays[f"opponent_{snapshot_index:03d}_{name}"] = np.asarray(
                snapshot[name]
            )
    temporary = path.with_name(
        f".{path.stem}.tmp-{os.getpid()}-{secrets.token_hex(4)}.npz"
    )
    try:
        np.savez_compressed(temporary, **arrays)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _superseded_opponent_band_hint(metadata):
    """Explain why a superseded resume state cannot simply be reinterpreted."""
    version = metadata.get("version")
    if version in PRE_PER_BUCKET_CHAMPION_STATE_VERSIONS:
        return (
            " This state stores one unqualified champion queue, event count, "
            "and score map from when champion_vs_heuristic was the only "
            "champion bucket. Champion state is now recorded per bucket, and "
            "reading the old singular state as one particular bucket's state "
            "would be a guess rather than a continuation, so a new run is "
            "required."
        )
    if version in PRE_CHAMPION_STATE_VERSIONS:
        return (
            " This state predates durable champion_vs_heuristic state and "
            "stores no pending candidate batch, event count, or champion "
            "scores. The opponent-pool schema changed for every bucket "
            "selection, so a new run is required."
        )
    if version in PRE_UNIFORM_ROTATION_STATE_VERSIONS:
        return (
            " This state predates the rotating uniform member allocation and "
            "stores no rotation anchor. Reinterpreting it would silently "
            "restart every bucket from its first ordered member, which is a "
            "policy change and not a bit-identical continuation, so a new run "
            "is required."
        )
    if version not in OVERLAPPING_MEDIUM_TERM_STATE_VERSIONS:
        return ""
    pool_state = metadata.get("opponent_pool_state")
    selected = (
        pool_state.get("selected_buckets", ())
        if isinstance(pool_state, dict)
        else ()
    )
    if "medium_term" in selected:
        return (
            " This state stored the superseded overlapping medium_term bucket, "
            "whose members were also recent members. The delayed band cannot "
            "be derived from it without changing what those memberships mean, "
            "so a new run is required."
        )
    return (
        " The opponent-pool policy changed for every bucket selection and no "
        "migration is implemented, so a new run is required."
    )


def load_resume_state(weights_path, state_path):
    """Validate and load one complete weights/state checkpoint pair."""
    weights_path = Path(weights_path)
    state_path = Path(state_path)
    with np.load(state_path, allow_pickle=False) as state:
        metadata = json.loads(str(state["metadata_json"].item()))
        if metadata.get("version") not in SUPPORTED_RESUME_STATE_VERSIONS:
            raise ValueError(
                f"Unsupported RL resume-state version in {state_path}: "
                f"{metadata.get('version')!r}."
                + _superseded_opponent_band_hint(metadata)
            )
        actual_hash = _file_sha256(weights_path)
        if metadata.get("weights_sha256") != actual_hash:
            raise ValueError(
                f"RL checkpoint pair is inconsistent: {weights_path} does not "
                f"match {state_path}."
            )
        pool_state = metadata.get("opponent_pool_state")
        if not isinstance(pool_state, dict):
            raise ValueError("RL resume state is missing opponent-pool state")
        if not isinstance(metadata.get("uniform_rotation_state"), dict):
            raise ValueError("RL resume state is missing uniform rotation state")
        serialization = pool_state.get("weight_serialization", ())
        pool_count = int(state["pool_weight_count"])
        if pool_count != len(serialization):
            raise ValueError("RL resume pool weight index is inconsistent")
        snapshots = {}
        for item in serialization:
            snapshot_index = int(item["index"])
            opponent_id = item["opponent_id"]
            prefix = f"opponent_{snapshot_index:03d}_"
            stored = {
                key[len(prefix):] for key in state.files if key.startswith(prefix)
            }
            try:
                names = policy_layer_names(
                    hidden_layer_count_from_weights(stored)
                )
            except ValueError as exc:
                raise ValueError(
                    f"RL resume state {state_path} has no usable policy "
                    f"weights for opponent {opponent_id!r}."
                ) from exc
            weights = {}
            for name in names:
                weights[name] = np.asarray(state[prefix + name]).copy()
            snapshots[opponent_id] = weights
    return metadata, snapshots


def _rng_state_metadata():
    """Return JSON-safe parent RNG state for exact checkpoint continuation."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
    }


def _nested_tuple(value):
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value


def _restore_rng_state(metadata):
    """Restore parent RNG state saved by :func:`_rng_state_metadata`."""
    if not metadata:
        raise ValueError("RL resume state is missing parent RNG state.")
    random.setstate(_nested_tuple(metadata["python"]))
    numpy_state = metadata["numpy"]
    np.random.set_state((
        numpy_state["bit_generator"],
        np.asarray(numpy_state["keys"], dtype=np.uint32),
        int(numpy_state["position"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))


def _validate_resume_configuration(
    metadata,
    expected,
    *,
    emit_status=lambda _message: None,
):
    """Reject a resume that would silently continue a different experiment."""
    saved = dict(metadata.get("configuration") or {})
    saved.setdefault("ruleset_name", DEFAULT_RULESET_NAME)
    saved.setdefault("baseline", None)
    saved.setdefault("baseline_artifact_sha256", None)
    saved.setdefault(
        "reward_distance_mode",
        HISTORICAL_REWARD_DISTANCE_MODE,
    )
    for current_name, original_name in RENAMED_PARAMETER_KEYS.items():
        if current_name not in saved and original_name in saved:
            saved[current_name] = saved.pop(original_name)
        elif current_name not in saved:
            saved[current_name] = HISTORICAL_REWARD_PARAMETER_DEFAULTS[
                current_name
            ]
    expected = expected.to_dict()
    if saved.get("git_commit") != expected.get("git_commit"):
        emit_status(
            "Warning: the checkpoint commit differs from the run's initial "
            "commit. Resume will continue with the saved training parameters."
        )
    ignored_keys = {"git_commit"}
    comparable_saved = {
        key: value
        for key, value in saved.items()
        if key not in ignored_keys
    }
    comparable_expected = {
        key: value for key, value in expected.items() if key not in ignored_keys
    }
    if comparable_saved != comparable_expected:
        differences = []
        for key in sorted(set(saved) | set(expected)):
            if key in ignored_keys:
                continue
            if saved.get(key) != expected.get(key):
                differences.append(
                    f"{key}: checkpoint={saved.get(key)!r}, "
                    f"requested={expected.get(key)!r}"
                )
        raise ValueError(
            "RL resume configuration does not match the checkpoint: "
            + "; ".join(differences)
        )


def _save_numbered_resume_checkpoint(
    network,
    runner,
    base_path,
    iteration,
    configuration,
    runtime_workers,
    completed_training_games,
    adaptive_tuning,
    training_state,
):
    """Save one atomic resumable checkpoint and retain only its latest state."""
    weights_path = numbered_checkpoint_path(base_path, iteration)
    state_path = resume_state_path(weights_path)

    # Invalidate an older same-iteration pair before replacing its weights.
    # A sudden interruption then leaves either the previous iteration pair or
    # the newly completed pair, never new weights with stale resume metadata.
    state_path.unlink(missing_ok=True)
    _atomic_network_save(network, weights_path)
    metadata = {
        "version": RESUME_STATE_VERSION,
        "completed_iteration": int(iteration),
        "weights_file": weights_path.name,
        "weights_sha256": _file_sha256(weights_path),
        "runtime_workers": int(runtime_workers),
        "completed_training_games": int(completed_training_games),
        "configuration": configuration.to_dict(),
        "optimizer_state": network.optimizer_state_dict(),
        "rng_state": _rng_state_metadata(),
        "adaptive_tuning": adaptive_tuning,
        "training_state": training_state,
        "opponent_pool_state": runner.opponent_pool.export_state(),
        "opponent_performance_state": (
            runner.performance_tracker.export_state()
        ),
        "uniform_rotation_state": runner.uniform_rotation.export_state(),
        "checkpoint_archive": {
            "path": "checkpoint_archive/manifest.json",
            "policy": archive_policy_manifest(),
        },
    }
    pool_state = runner.opponent_pool.export_state()
    _atomic_resume_state_save(
        state_path,
        metadata,
        pool_state,
        runner.opponent_pool.export_weights(),
    )

    # Pool snapshots are much larger than policy checkpoints. The newest
    # state is sufficient for continuation; older numbered weight files remain
    # available for analysis while their superseded resume states are removed.
    base = Path(base_path)
    base_stem = base.name[:-len(base.suffix)] if base.suffix else base.name
    for older_state in base.parent.glob(f"{base_stem}_iter*.resume.npz"):
        if older_state != state_path:
            older_state.unlink(missing_ok=True)
    _prune_numbered_checkpoint_weights(base, weights_path)
    return weights_path, state_path


def _sl_checkpoint_sha256(path):
    path = Path(path)
    if path.is_file():
        return _file_sha256(path)
    return None


def _training_state_payload(
    *,
    win_rate_window,
    value_loss_window,
    ppo_window,
    total_decision_samples,
    total_normal_decision_samples,
    total_restart_decision_samples,
    total_restart_episodes,
    total_restart_duration_s,
    policy_updates_completed,
    clipped_iteration_count,
    total_rollout_duration_s,
    total_update_duration_s,
    elapsed_rl_seconds,
):
    return {
        "win_rate_window": list(win_rate_window),
        "value_loss_window": list(value_loss_window),
        "ppo_window": list(ppo_window),
        "total_decision_samples": int(total_decision_samples),
        "trainable_decisions_seen": int(total_decision_samples),
        "total_normal_decision_samples": int(total_normal_decision_samples),
        "total_restart_decision_samples": int(total_restart_decision_samples),
        "total_restart_episodes": int(total_restart_episodes),
        "total_restart_duration_s": float(total_restart_duration_s),
        "policy_updates_completed": int(policy_updates_completed),
        "clipped_iteration_count": int(clipped_iteration_count),
        "total_rollout_duration_s": float(total_rollout_duration_s),
        "total_update_duration_s": float(total_update_duration_s),
        "elapsed_rl_seconds": float(elapsed_rl_seconds),
    }


def _restore_training_windows(metadata, moving_average_window):
    state = (metadata or {}).get("training_state", {})
    return (
        deque(state.get("win_rate_window", ()), maxlen=moving_average_window),
        deque(state.get("value_loss_window", ()), maxlen=moving_average_window),
        deque(state.get("ppo_window", ()), maxlen=10),
        state,
    )
