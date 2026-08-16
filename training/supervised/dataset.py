"""Filtering, encoding, and memory-safe caching for supervised datasets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import numpy as np

from agents.network_architecture import DEFAULT_NETWORK_ARCHITECTURE
from training.supervised.runtime import (
    DEFAULT_SUPERVISED_BATCH_SIZE,
    estimate_supervised_workspace_bytes,
)
from training.utils.encoding import ENCODED_FEATURE_VERSION
from utils.artifacts import file_sha256
from utils.resource_limits import MIB, MemorySafetyError, host_allocation_status
from middleware.rulesets import validate_state_ruleset


ENCODED_CACHE_FILE = "dataset/supervised_dataset_encoded.npz"
DATASET_DTYPE = np.float32
DATASET_MEMORY_RESERVE_MB = 512


@dataclass
class EncodedDataset:
    """An encoded dataset and the storage mode that owns its arrays."""

    x: np.ndarray
    y: np.ndarray
    storage_mode: str
    metadata: dict


def _normalize_action(action):
    """Return a normalized tile-play action or ``None`` for draw/pass."""
    if action is None or action == ["DRAW", None] or action == ("DRAW", None):
        return None
    if isinstance(action[0], list):
        return (tuple(action[0]), action[1])
    return action


def _legal_tile_actions_from_state(state):
    """Reconstruct legal tile-play actions from a serialized state."""
    hand = [tuple(tile) for tile in state["current_player_hand"]]
    ends = state.get("ends", [])
    if not ends:
        doubles = [tile for tile in hand if tile[0] == tile[1]]
        if doubles:
            return [(max(doubles, key=lambda tile: tile[0]), 0)]
        return [(tile, 0) for tile in hand]

    left_end, right_end = ends
    actions = []
    for tile in hand:
        if left_end in tile:
            actions.append((tile, 0))
        if right_end in tile:
            actions.append((tile, 1))
    if left_end == right_end:
        actions = [(tile, 0) for tile, _side in actions]
    return list(dict.fromkeys(actions))


def _is_real_decision_state(state):
    """Return whether a player had at least two legal tile-play choices."""
    return len(_legal_tile_actions_from_state(state)) >= 2


def _dataset_metadata(file_path, encoder):
    """Return source and encoder fields used to validate encoded caches."""
    stat = os.stat(file_path)
    return {
        "source_sha256": file_sha256(file_path),
        "source_size": stat.st_size,
        "ruleset_name": encoder.ruleset.name,
        "encoder_vector_size": encoder.vector_size,
        "encoder_action_size": encoder.action_size,
        "feature_version": ENCODED_FEATURE_VERSION,
    }


def _cache_matches(cache_data, expected_metadata):
    """Return whether an ``np.load`` mapping matches current source metadata."""
    for key, expected_value in expected_metadata.items():
        if key not in cache_data or cache_data[key].item() != expected_value:
            return False
    return True


def _mmap_cache_paths(cache_file):
    """Return stable X/Y/metadata paths derived from the compressed cache."""
    cache_path = Path(cache_file)
    stem = cache_path.stem
    if stem.endswith("_encoded"):
        stem = stem[:-len("_encoded")]
    parent = cache_path.parent
    return (
        parent / f"{stem}_X.npy",
        parent / f"{stem}_Y.npy",
        parent / f"{stem}_metadata.json",
    )


def _scan_dataset(file_path, encoder):
    """Count usable examples without retaining decoded JSON records."""
    counts = {
        "example_count": 0,
        "skipped_draw_pass": 0,
        "skipped_single_option": 0,
    }
    with open(file_path, "r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            validate_state_ruleset(record["state"], encoder.ruleset)
            action = _normalize_action(record["target_action"])
            if action is None:
                counts["skipped_draw_pass"] += 1
            elif not _is_real_decision_state(record["state"]):
                counts["skipped_single_option"] += 1
            else:
                counts["example_count"] += 1
    if counts["example_count"] < 1:
        raise ValueError(
            "The dataset contains no real tile-play decisions after filtering "
            "draw/pass and single-option tile-play actions."
        )
    return counts


def _fill_encoded_arrays(file_path, encoder, x, y, expected_count):
    """Fill preallocated RAM or mmap arrays during one streaming JSONL pass."""
    column = 0
    with open(file_path, "r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            state = record["state"]
            action = _normalize_action(record["target_action"])
            if action is None or not _is_real_decision_state(state):
                continue
            x[:, column] = encoder.encode_state(state)[:, 0]
            y[:, column] = 0.0
            y[encoder._action_index(action), column] = 1.0
            column += 1
    if column != expected_count:
        raise RuntimeError(
            "dataset changed while it was being encoded: expected "
            f"{expected_count} examples, read {column}"
        )


def _encoded_bytes(example_count, encoder):
    return int(
        example_count
        * (encoder.vector_size + encoder.action_size)
        * np.dtype(DATASET_DTYPE).itemsize
    )


def _host_dataset_working_set_bytes(
    example_count,
    encoder,
    architecture=DEFAULT_NETWORK_ARCHITECTURE,
):
    """Estimate dataset, permutation, and minimum CPU training workspace."""
    train_count = max(1, int(example_count * 0.85))
    return (
        _encoded_bytes(example_count, encoder)
        + train_count * np.dtype(np.int64).itemsize
        + estimate_supervised_workspace_bytes(
            min(DEFAULT_SUPERVISED_BATCH_SIZE, train_count),
            encoder.vector_size,
            architecture.hidden_sizes,
            encoder.action_size,
        )
    )


def _save_encoded_cache(cache_file, x, y, metadata, quiet=False):
    """Persist the RAM-resident compressed cache through an atomic replace."""
    cache_path = Path(cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            X=np.asarray(x, dtype=DATASET_DTYPE),
            Y=np.asarray(y, dtype=DATASET_DTYPE),
            encoded_example_count=x.shape[1],
            encoded_bytes=x.nbytes + y.nbytes,
            **metadata,
        )
        os.replace(temporary, cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    if not quiet:
        print(f"Encoded dataset cache saved to {cache_file}.")


def _mmap_metadata_matches(metadata_path, x_path, y_path, expected):
    """Validate mmap metadata, shapes, dtypes, sizes, and completed files."""
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False, None
        if metadata.get("dtype") != np.dtype(DATASET_DTYPE).name:
            return False, None
        if not x_path.is_file() or not y_path.is_file():
            return False, None
        if metadata.get("x_file_size") != x_path.stat().st_size:
            return False, None
        if metadata.get("y_file_size") != y_path.stat().st_size:
            return False, None
        x = np.load(x_path, mmap_mode="r", allow_pickle=False)
        y = np.load(y_path, mmap_mode="r", allow_pickle=False)
        if list(x.shape) != metadata.get("x_shape"):
            return False, None
        if list(y.shape) != metadata.get("y_shape"):
            return False, None
        if x.dtype != DATASET_DTYPE or y.dtype != DATASET_DTYPE:
            return False, None
        if x.shape[1] != y.shape[1] or x.shape[1] != metadata.get("example_count"):
            return False, None
        return True, (x, y, metadata)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False, None


def _build_mmap_cache(file_path, encoder, cache_file, source_metadata, counts):
    """Build complete disk-backed arrays and publish metadata last."""
    x_path, y_path, metadata_path = _mmap_cache_paths(cache_file)
    x_path.parent.mkdir(parents=True, exist_ok=True)
    example_count = counts["example_count"]
    token = f"{os.getpid()}-{time.time_ns()}"
    temporary_x = x_path.with_name(f".{x_path.name}.{token}.tmp.npy")
    temporary_y = y_path.with_name(f".{y_path.name}.{token}.tmp.npy")
    temporary_metadata = metadata_path.with_name(
        f".{metadata_path.name}.{token}.tmp"
    )
    try:
        x = np.lib.format.open_memmap(
            temporary_x,
            mode="w+",
            dtype=DATASET_DTYPE,
            shape=(encoder.vector_size, example_count),
        )
        y = np.lib.format.open_memmap(
            temporary_y,
            mode="w+",
            dtype=DATASET_DTYPE,
            shape=(encoder.action_size, example_count),
        )
        _fill_encoded_arrays(file_path, encoder, x, y, example_count)
        x.flush()
        y.flush()
        del x, y
        os.replace(temporary_x, x_path)
        os.replace(temporary_y, y_path)
        metadata = {
            **source_metadata,
            "example_count": example_count,
            "dtype": np.dtype(DATASET_DTYPE).name,
            "x_shape": [encoder.vector_size, example_count],
            "y_shape": [encoder.action_size, example_count],
            "x_file_size": x_path.stat().st_size,
            "y_file_size": y_path.stat().st_size,
        }
        with open(temporary_metadata, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_metadata, metadata_path)
        return (
            np.load(x_path, mmap_mode="r", allow_pickle=False),
            np.load(y_path, mmap_mode="r", allow_pickle=False),
            metadata,
        )
    finally:
        for path in (temporary_x, temporary_y, temporary_metadata):
            if path.exists():
                path.unlink()


def load_dataset(
    file_path,
    encoder,
    quiet=False,
    memory_reserve_mb=DATASET_MEMORY_RESERVE_MB,
    architecture=DEFAULT_NETWORK_ARCHITECTURE,
):
    """Encode a JSONL dataset directly into preallocated float32 RAM arrays."""
    if not quiet:
        print(f"Scanning dataset from {file_path}...")
    counts = _scan_dataset(file_path, encoder)
    required = _host_dataset_working_set_bytes(
        counts["example_count"],
        encoder,
        architecture,
    )
    safe, status = host_allocation_status(required, memory_reserve_mb)
    if not safe:
        raise MemorySafetyError(
            "RAM-resident supervised encoding is unsafe: "
            f"needs about {required / MIB:.1f} MiB plus the "
            f"{memory_reserve_mb} MiB reserve, while "
            f"{status['available_bytes'] / MIB:.1f} MiB is available."
        )
    count = counts["example_count"]
    x = np.empty((encoder.vector_size, count), dtype=DATASET_DTYPE)
    y = np.empty((encoder.action_size, count), dtype=DATASET_DTYPE)
    _fill_encoded_arrays(file_path, encoder, x, y, count)
    if not quiet:
        print(f"Dataset loaded. X: {x.shape}, Y: {y.shape}")
        print(f"Skipped forced draw/pass examples: {counts['skipped_draw_pass']}")
        print(
            "Skipped single-option tile-play examples: "
            f"{counts['skipped_single_option']}"
        )
    return x, y


def load_or_build_dataset(
    file_path,
    encoder,
    cache_file=ENCODED_CACHE_FILE,
    quiet=False,
    *,
    memory_reserve_mb=DATASET_MEMORY_RESERVE_MB,
    return_info=False,
    architecture=DEFAULT_NETWORK_ARCHITECTURE,
):
    """Return a validated RAM or mmap encoded cache without unsafe allocation."""
    source_metadata = _dataset_metadata(file_path, encoder)
    counts = _scan_dataset(file_path, encoder)
    example_count = counts["example_count"]
    required = _host_dataset_working_set_bytes(
        example_count,
        encoder,
        architecture,
    )
    safe_in_ram, _status = host_allocation_status(required, memory_reserve_mb)

    if safe_in_ram and os.path.exists(cache_file):
        try:
            with np.load(cache_file, allow_pickle=False) as cache_data:
                if (
                    _cache_matches(cache_data, source_metadata)
                    and int(cache_data["encoded_example_count"].item())
                    == example_count
                ):
                    x = np.asarray(cache_data["X"], dtype=DATASET_DTYPE)
                    y = np.asarray(cache_data["Y"], dtype=DATASET_DTYPE)
                    result = EncodedDataset(x, y, "ram", source_metadata)
                    if not quiet:
                        print(f"Loaded encoded dataset cache from {cache_file}.")
                    return result if return_info else (x, y)
        except (OSError, KeyError, ValueError):
            pass

    x_path, y_path, metadata_path = _mmap_cache_paths(cache_file)
    valid_mmap, mmap_payload = _mmap_metadata_matches(
        metadata_path,
        x_path,
        y_path,
        source_metadata,
    )
    if not safe_in_ram:
        if valid_mmap:
            x, y, metadata = mmap_payload
        else:
            x, y, metadata = _build_mmap_cache(
                file_path,
                encoder,
                cache_file,
                source_metadata,
                counts,
            )
        result = EncodedDataset(x, y, "mmap", metadata)
        if not quiet:
            print(
                "Encoded dataset uses disk-backed mmap storage: "
                f"{x_path} and {y_path}."
            )
        return result if return_info else (x, y)

    x, y = load_dataset(
        file_path,
        encoder,
        quiet=quiet,
        memory_reserve_mb=memory_reserve_mb,
        architecture=architecture,
    )
    _save_encoded_cache(cache_file, x, y, source_metadata, quiet=quiet)
    result = EncodedDataset(x, y, "ram", source_metadata)
    return result if return_info else (x, y)
