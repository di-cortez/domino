"""Train the supervised domino policy with safe CPU/GPU autotuning.

The encoded dataset stays in host RAM when that is safe, falls back to a
disk-backed ``.npy`` cache when it is not, and may be kept fully or partially
resident on a selected GPU. Every completed autotuning epoch updates the live
network and counts toward the requested maximum epoch total. Once tuning is
complete, repeated low-improvement training-loss blocks can stop a saturated
run early.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import numpy as np

from agents.encoder import DominoEncoder
from agents.nn import SupervisedNeuralNetwork
from training.supervised_runtime import (
    DEFAULT_SUPERVISED_CPU_BATCH_SIZE,
    DEFAULT_SUPERVISED_GPU_BATCH_SIZE,
    RetainedBatchAutotuner,
    SUPERVISED_BATCH_AUTOTUNE_EPOCHS,
    SUPERVISED_GPU_MEMORY_RESERVE_MB,
    SupervisedDataPlan,
    SupervisedResourceTracker,
    estimate_supervised_workspace_bytes,
    probe_gpu_residency,
)
from utils.resource_limits import (
    MIB,
    MemorySafetyError,
    choose_safe_supervised_device,
    effective_gpu_available_bytes,
    gpu_memory_info,
    host_allocation_status,
)
from utils.runtime_status import format_duration, memory_report


EPOCHS = 2000
BATCH_SIZE = DEFAULT_SUPERVISED_CPU_BATCH_SIZE
INITIAL_SUPERVISED_LEARNING_RATE = 0.005
DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_SUPERVISED_LR_DECAY_PATIENCE = 5
DEFAULT_SUPERVISED_LR_DECAY_FACTOR = 0.5
SUPERVISED_VALIDATION_INTERVAL_EPOCHS = 10
DEFAULT_TRAINING_PLATEAU_WINDOW = 25
DEFAULT_TRAINING_PLATEAU_PATIENCE = 4
DEFAULT_TRAINING_PLATEAU_MIN_EPOCHS = 100
DEFAULT_TRAINING_PLATEAU_MIN_RELATIVE_IMPROVEMENT = 0.001

CHECKPOINT_EVERY = SUPERVISED_VALIDATION_INTERVAL_EPOCHS
CHECKPOINT_DIR = "models/supervised_checkpoints"
MAX_SUPERVISED_CHECKPOINTS = 10
ENCODED_CACHE_FILE = "dataset/supervised_dataset_encoded.npz"
ENCODED_FEATURE_VERSION = "opponent_suit_presence_float32_v2"
DATASET_DTYPE = np.float32
DATASET_MEMORY_RESERVE_MB = 512


def _format_optional_mib(byte_count):
    """Format an optional byte measurement for detailed resource logs."""
    if byte_count is None:
        return "unavailable"
    return f"{byte_count / MIB:.1f} MiB"


def supervised_loss_plot_path(weights_file):
    """Return the loss-plot path beside one supervised checkpoint."""
    weights_path = Path(weights_file)
    stem = weights_path.stem
    if stem.endswith("_weights"):
        stem = stem.removesuffix("_weights")
    return weights_path.with_name(f"{stem}_loss.png")


def _supervised_loss_axis_limits(loss_history, validation_points):
    """Frame the visible loss range around the observed supervised curves."""
    training_losses = [
        float(loss) for loss in loss_history if np.isfinite(float(loss))
    ]
    validation_losses = [
        float(loss)
        for _epoch, loss in validation_points
        if np.isfinite(float(loss))
    ]
    if not training_losses:
        raise ValueError("Cannot scale a graph without finite training losses.")

    final_training_loss = training_losses[-1]
    maximum_loss = max(training_losses + validation_losses)
    observed_drop = max(0.0, maximum_loss - final_training_loss)
    scale = max(abs(maximum_loss), abs(final_training_loss), 1e-6)
    lower_padding = max(observed_drop * 0.10, scale * 0.005, 1e-6)
    lower_limit = final_training_loss - lower_padding
    if maximum_loss <= lower_limit:
        lower_limit = maximum_loss - max(scale * 0.01, 1e-6)
    return lower_limit, maximum_loss


def _save_supervised_loss_plot(loss_history, epoch_metrics, weights_file):
    """Atomically plot current-run training and validation cross-entropy loss."""
    if not loss_history:
        raise ValueError("Cannot plot an empty supervised loss history.")

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output_path = supervised_loss_plot_path(weights_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.tmp-{os.getpid()}-{time.time_ns()}.png"
    )

    figure = Figure(figsize=(9.0, 5.25), facecolor="#263b34")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.set_facecolor("#263b34")

    epochs = list(range(1, len(loss_history) + 1))
    axis.plot(
        epochs,
        [float(loss) for loss in loss_history],
        color="#f1d36b",
        linewidth=2.2,
        label="Training loss",
    )
    validation_points = [
        (int(metrics["epoch"]) + 1, float(metrics["validation_loss"]))
        for metrics in epoch_metrics
        if metrics.get("validation_loss") is not None
        and np.isfinite(float(metrics["validation_loss"]))
    ]
    if validation_points:
        validation_epochs, validation_losses = zip(*validation_points)
        axis.plot(
            validation_epochs,
            validation_losses,
            color="#d7eee4",
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            label="Validation loss",
        )

    axis.set_title("Supervised Training Loss", color="#f4f0df", pad=12)
    axis.set_xlabel("Epoch", color="#f4f0df")
    axis.set_ylabel("Cross-entropy loss", color="#f4f0df")
    lower_limit, upper_limit = _supervised_loss_axis_limits(
        loss_history,
        validation_points,
    )
    axis.set_ylim(lower_limit, upper_limit)
    axis.grid(color="#81978d", alpha=0.25, linewidth=0.8)
    axis.tick_params(colors="#f4f0df")
    for spine in axis.spines.values():
        spine.set_color("#b7c5bd")
    legend = axis.legend(frameon=False)
    if legend is not None:
        for text in legend.get_texts():
            text.set_color("#f4f0df")
    figure.tight_layout()

    try:
        figure.savefig(
            temporary_path,
            format="png",
            dpi=150,
            facecolor=figure.get_facecolor(),
        )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
        figure.clear()
    return output_path


@dataclass
class EncodedDataset:
    """An encoded dataset and the storage mode that owns its arrays."""

    x: np.ndarray
    y: np.ndarray
    storage_mode: str
    metadata: dict


def _prune_supervised_checkpoints(
    checkpoint_dir=CHECKPOINT_DIR,
    keep_count=MAX_SUPERVISED_CHECKPOINTS,
):
    """Delete older archival checkpoints and return the removed paths."""
    if keep_count < 1:
        raise ValueError("keep_count must be at least one.")
    checkpoint_paths = list(Path(checkpoint_dir).glob("domino_sl_epoch_*.npz"))
    checkpoint_paths.sort(
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )
    removed_paths = checkpoint_paths[keep_count:]
    for path in removed_paths:
        path.unlink()
    return removed_paths


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
        "source_path": os.path.abspath(file_path),
        "source_mtime_ns": stat.st_mtime_ns,
        "source_size": stat.st_size,
        "encoder_vector_size": encoder.VECTOR_SIZE,
        "encoder_action_size": len(encoder.all_actions),
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


def _scan_dataset(file_path):
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
        * (encoder.VECTOR_SIZE + len(encoder.all_actions))
        * np.dtype(DATASET_DTYPE).itemsize
    )


def _host_dataset_working_set_bytes(example_count, encoder):
    """Estimate dataset, permutation, and minimum CPU training workspace."""
    train_count = max(1, int(example_count * 0.85))
    return (
        _encoded_bytes(example_count, encoder)
        + train_count * np.dtype(np.int64).itemsize
        + estimate_supervised_workspace_bytes(
            min(DEFAULT_SUPERVISED_CPU_BATCH_SIZE, train_count),
            encoder.VECTOR_SIZE,
            256,
            128,
            len(encoder.all_actions),
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
            shape=(encoder.VECTOR_SIZE, example_count),
        )
        y = np.lib.format.open_memmap(
            temporary_y,
            mode="w+",
            dtype=DATASET_DTYPE,
            shape=(len(encoder.all_actions), example_count),
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
            "x_shape": [encoder.VECTOR_SIZE, example_count],
            "y_shape": [len(encoder.all_actions), example_count],
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
):
    """Encode a JSONL dataset directly into preallocated float32 RAM arrays."""
    if not quiet:
        print(f"Scanning dataset from {file_path}...")
    counts = _scan_dataset(file_path)
    required = _host_dataset_working_set_bytes(counts["example_count"], encoder)
    safe, status = host_allocation_status(required, memory_reserve_mb)
    if not safe:
        raise MemorySafetyError(
            "RAM-resident supervised encoding is unsafe: "
            f"needs about {required / MIB:.1f} MiB plus the "
            f"{memory_reserve_mb} MiB reserve, while "
            f"{status['available_bytes'] / MIB:.1f} MiB is available."
        )
    count = counts["example_count"]
    x = np.empty((encoder.VECTOR_SIZE, count), dtype=DATASET_DTYPE)
    y = np.empty((len(encoder.all_actions), count), dtype=DATASET_DTYPE)
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
):
    """Return a validated RAM or mmap encoded cache without unsafe allocation."""
    source_metadata = _dataset_metadata(file_path, encoder)
    counts = _scan_dataset(file_path)
    example_count = counts["example_count"]
    required = _host_dataset_working_set_bytes(example_count, encoder)
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
    )
    _save_encoded_cache(cache_file, x, y, source_metadata, quiet=quiet)
    result = EncodedDataset(x, y, "ram", source_metadata)
    return result if return_info else (x, y)


def _network_weight_payload(network, weights=None):
    """Return six compatible host float32 arrays for checkpoint writing."""
    source = weights or {name: getattr(network, name) for name in network.weight_names}
    return {
        name: network.to_host(source[name]).astype(np.float32, copy=False)
        for name in network.weight_names
    }


def _load_existing_weights(network, weights_file):
    """Validate and load a legacy or current supervised checkpoint."""
    with np.load(weights_file, allow_pickle=False) as weights:
        for name in network.weight_names:
            expected_shape = getattr(network, name).shape
            if weights[name].shape != expected_shape:
                raise ValueError(
                    f"Cannot resume from {weights_file}: {name} has shape "
                    f"{weights[name].shape}, but expected {expected_shape}. "
                    "Delete or move the old checkpoint and retrain from scratch."
                )
        network.load_policy_weights(weights)


def _create_network(*, device, weight_decay, seed):
    return SupervisedNeuralNetwork(
        input_size=DominoEncoder.VECTOR_SIZE,
        hidden1_size=256,
        hidden2_size=128,
        output_size=DominoEncoder.ACTION_SIZE,
        learning_rate=INITIAL_SUPERVISED_LEARNING_RATE,
        weight_decay=weight_decay,
        random_seed=seed,
        device=device,
    )


def _is_gpu_startup_failure(exc):
    """Return whether an exception came from CuPy/CUDA initialization."""
    module_name = type(exc).__module__
    return module_name.startswith("cupy") or module_name.startswith(
        "cupy_backends"
    )


def train_supervised(
    epochs=EPOCHS,
    batch_size=None,
    dataset_file="dataset/supervised_dataset.jsonl",
    weights_file="models/domino_sl_weights.npz",
    cache_file=ENCODED_CACHE_FILE,
    quiet=False,
    progress_callback=None,
    status_callback=None,
    weight_decay=0.0,
    early_stopping_patience=None,
    lr_decay_factor=DEFAULT_SUPERVISED_LR_DECAY_FACTOR,
    lr_decay_patience=DEFAULT_SUPERVISED_LR_DECAY_PATIENCE,
    training_plateau_enabled=True,
    training_plateau_window=DEFAULT_TRAINING_PLATEAU_WINDOW,
    training_plateau_patience=DEFAULT_TRAINING_PLATEAU_PATIENCE,
    training_plateau_min_epochs=DEFAULT_TRAINING_PLATEAU_MIN_EPOCHS,
    training_plateau_min_relative_improvement=(
        DEFAULT_TRAINING_PLATEAU_MIN_RELATIVE_IMPROVEMENT
    ),
    device="auto",
    autotune_batch_size=True,
    memory_reserve_mb=DATASET_MEMORY_RESERVE_MB,
    gpu_memory_reserve_mb=SUPERVISED_GPU_MEMORY_RESERVE_MB,
    seed=None,
):
    """Train the policy and return scheduler, storage, tuning, and memory data."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    started = time.time()
    if seed is not None:
        np.random.seed(seed)
    if not quiet:
        print(f"Supervised training startup memory: {memory_report()}")

    encoder = DominoEncoder()
    dataset = load_or_build_dataset(
        dataset_file,
        encoder,
        cache_file,
        quiet=quiet,
        memory_reserve_mb=memory_reserve_mb,
        return_info=True,
    )
    total_examples = dataset.x.shape[1]
    train_count = int(total_examples * 0.85)
    if total_examples == 1:
        train_count = 1
    else:
        train_count = max(1, min(train_count, total_examples - 1))
    validation_count = total_examples - train_count

    requested_device = device
    selected_device, fallback_reason = choose_safe_supervised_device(
        requested_device,
        gpu_memory_reserve_mb,
    )
    try:
        network = _create_network(
            device=selected_device,
            weight_decay=weight_decay,
            seed=seed,
        )
    except Exception as exc:
        if selected_device != "gpu" or not _is_gpu_startup_failure(exc):
            raise
        reason = f"GPU network initialization failed ({type(exc).__name__}: {exc})"
        if requested_device == "gpu":
            raise MemorySafetyError(
                f"Cannot honor device='gpu': {reason}."
            ) from exc
        selected_device = "cpu"
        fallback_reason = reason
        network = _create_network(
            device="cpu",
            weight_decay=weight_decay,
            seed=seed,
        )
    residency_probe = None
    resident_capacity = None

    if selected_device == "gpu":
        if seed is not None:
            network.xp.random.seed(seed)
        residency_probe = probe_gpu_residency(
            dataset.x,
            dataset.y,
            reserve_mb=gpu_memory_reserve_mb,
        )
        if residency_probe.capacity_examples < 1:
            reason = (
                "the first GPU dataset-residency candidate could not preserve "
                "the configured VRAM reserve"
            )
            network.release_disposable_cache()
            if requested_device == "gpu":
                raise MemorySafetyError(f"Cannot honor device='gpu': {reason}.")
            fallback_reason = reason
            selected_device = "cpu"
            network = _create_network(
                device="cpu",
                weight_decay=weight_decay,
                seed=seed,
            )
        else:
            resident_capacity = residency_probe.capacity_examples

    try:
        data_plan = SupervisedDataPlan(
            dataset.x,
            dataset.y,
            train_count=train_count,
            host_storage_mode=dataset.storage_mode,
            device=selected_device,
            resident_capacity=resident_capacity,
        )
    except MemorySafetyError:
        if requested_device != "auto" or selected_device != "gpu":
            raise
        fallback_reason = "the final GPU residency allocation was unsafe"
        network.release_disposable_cache()
        selected_device = "cpu"
        network = _create_network(
            device="cpu",
            weight_decay=weight_decay,
            seed=seed,
        )
        data_plan = SupervisedDataPlan(
            dataset.x,
            dataset.y,
            train_count=train_count,
            host_storage_mode=dataset.storage_mode,
            device="cpu",
        )

    if not quiet:
        print(f"Supervised device: {selected_device}")
        if fallback_reason:
            print(f"Automatic supervised CPU fallback: {fallback_reason}.")
        if selected_device == "gpu":
            gpu_info = gpu_memory_info()
            effective_free = effective_gpu_available_bytes()
            if gpu_info is not None:
                effective_text = (
                    "unknown"
                    if effective_free is None
                    else f"{effective_free / MIB:.1f} MiB"
                )
                print(
                    "GPU VRAM before supervised residency: "
                    f"{gpu_info.available / MIB:.1f} MiB free / "
                    f"{gpu_info.total / MIB:.1f} MiB total; "
                    f"{effective_text} effective free."
                )
            print(
                "GPU dataset residency: "
                f"{data_plan.resident_window_examples:,} examples; "
                f"mode={data_plan.storage_mode}; "
                f"reserve={gpu_memory_reserve_mb} MiB."
            )
            if data_plan.full_upload_seconds is not None:
                print(
                    "One-time full-dataset GPU upload: "
                    f"{data_plan.full_upload_seconds:.3f}s."
                )
        print(
            f"Split complete: {train_count} train | "
            f"{validation_count} validation"
        )

    if os.path.exists(weights_file):
        _load_existing_weights(network, weights_file)
        if not quiet:
            print(f"Existing supervised model found at {weights_file}. Resuming training.")
    elif not quiet:
        print("No existing supervised model found. Training from scratch.")

    fixed_batch = None
    if batch_size is not None:
        fixed_batch = int(batch_size)
    elif not autotune_batch_size:
        fixed_batch = (
            DEFAULT_SUPERVISED_GPU_BATCH_SIZE
            if selected_device == "gpu"
            else DEFAULT_SUPERVISED_CPU_BATCH_SIZE
        )

    emit_status = status_callback
    if emit_status is None:
        emit_status = (lambda _message: None) if quiet else print
    autotuner = RetainedBatchAutotuner(
        device=selected_device,
        training_examples=train_count,
        total_epochs=epochs,
        preflight=lambda candidate: data_plan.batch_memory_preflight(
            network,
            candidate,
            gpu_memory_reserve_mb
            if selected_device == "gpu"
            else memory_reserve_mb,
        ),
        enabled=autotune_batch_size and batch_size is None,
        fixed_batch_size=fixed_batch,
        epochs_per_candidate=SUPERVISED_BATCH_AUTOTUNE_EPOCHS,
        status_callback=emit_status,
    )

    tracker = SupervisedResourceTracker(selected_device)
    best_state = {"validation_loss": float("inf"), "weights": None}
    last_checkpoint_time = {"value": started}

    def save_checkpoint(current_network, epoch, validation_loss):
        checkpoint_dir = Path(CHECKPOINT_DIR)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        weights_path = Path(weights_file)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / (
            f"domino_sl_epoch_{epoch:04d}_val_{validation_loss:.4f}.npz"
        )
        payload = _network_weight_payload(current_network)
        np.savez(checkpoint_file, **payload)
        np.savez(weights_path, **payload)
        removed = _prune_supervised_checkpoints(checkpoint_dir)
        now = time.time()
        checkpoint_elapsed = now - last_checkpoint_time["value"]
        last_checkpoint_time["value"] = now
        if not quiet:
            print(
                f"  -> Checkpoint saved to {checkpoint_file} "
                f"(time since previous checkpoint: "
                f"{format_duration(checkpoint_elapsed)})."
            )
            if removed:
                print(
                    f"  -> Removed {len(removed)} older supervised "
                    f"checkpoint(s); keeping the latest "
                    f"{MAX_SUPERVISED_CHECKPOINTS}."
                )
            print(f"  -> Active supervised model updated at {weights_file}.")

    def save_if_best(epoch, validation_loss, current_network):
        if epoch % CHECKPOINT_EVERY == 0:
            save_checkpoint(current_network, epoch, validation_loss)
        if validation_loss < best_state["validation_loss"]:
            best_state["validation_loss"] = validation_loss
            best_state["weights"] = {
                name: getattr(current_network, name).copy()
                for name in current_network.weight_names
            }
            if not quiet:
                print(
                    f"  -> New best validation loss "
                    f"{validation_loss:.4f} at epoch {epoch}."
                )

    x_train = dataset.x[:, :train_count]
    y_train = dataset.y[:, :train_count]
    x_val = dataset.x[:, train_count:] if validation_count else None
    y_val = dataset.y[:, train_count:] if validation_count else None
    if not quiet:
        print("\nStarting supervised training...")

    epoch_metrics = []

    def record_epoch_metrics(metrics):
        epoch_metrics.append(metrics.copy())
        tracker.observe()

    try:
        loss_history = network.train(
            x_train,
            y_train,
            x_val=x_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=autotuner.current_batch_size,
            on_validation=save_if_best,
            progress_callback=progress_callback,
            quiet=quiet,
            early_stopping_patience=early_stopping_patience,
            lr_decay_factor=lr_decay_factor,
            lr_decay_patience=lr_decay_patience,
            validation_interval=SUPERVISED_VALIDATION_INTERVAL_EPOCHS,
            epoch_runner=data_plan.train_epoch,
            validation_runner=data_plan.validation_loss,
            batch_controller=autotuner,
            epoch_metrics_callback=record_epoch_metrics,
            training_plateau_window=(
                training_plateau_window if training_plateau_enabled else None
            ),
            training_plateau_patience=training_plateau_patience,
            training_plateau_min_epochs=training_plateau_min_epochs,
            training_plateau_min_relative_improvement=(
                training_plateau_min_relative_improvement
            ),
        )
        weights_path = Path(weights_file)
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            weights_path,
            **_network_weight_payload(network, best_state["weights"]),
        )
    finally:
        tracker.observe()
        data_plan.close()

    loss_plot_path = _save_supervised_loss_plot(
        loss_history,
        epoch_metrics,
        weights_file,
    )

    elapsed = time.time() - started
    tuning_summary = autotuner.to_dict()
    training_summary = network.last_training_summary
    resource_summary = tracker.to_dict()
    if data_plan.peak_gpu_pool_used_bytes:
        resource_summary["peak_gpu_pool_used_bytes"] = max(
            resource_summary.get("peak_gpu_pool_used_bytes") or 0,
            data_plan.peak_gpu_pool_used_bytes,
        )
    if data_plan.minimum_effective_free_vram_bytes is not None:
        prior = resource_summary.get("minimum_effective_free_vram_bytes")
        resource_summary["minimum_effective_free_vram_bytes"] = (
            data_plan.minimum_effective_free_vram_bytes
            if prior is None
            else min(prior, data_plan.minimum_effective_free_vram_bytes)
        )

    if not quiet:
        best_text = (
            "unavailable"
            if best_state["validation_loss"] == float("inf")
            else f"{best_state['validation_loss']:.4f}"
        )
        print(f"Model saved to {weights_file} (best validation loss: {best_text}).")
        print(f"Loss graph saved to {loss_plot_path}.")
        if data_plan.storage_mode == "gpu_windowed":
            rotations = [
                metrics["window_rotations"] for metrics in epoch_metrics
            ]
            print(
                "GPU window rotations: "
                f"{sum(rotations)} total across {len(rotations)} epochs "
                f"({min(rotations)}-{max(rotations)} per epoch)."
            )
        print(
            "Supervised resource bounds: "
            f"peak host RSS="
            f"{_format_optional_mib(resource_summary['peak_host_rss_bytes'])}; "
            f"minimum available host RAM="
            f"{_format_optional_mib(resource_summary['minimum_available_host_ram_bytes'])}; "
            f"peak CuPy pool="
            f"{_format_optional_mib(resource_summary['peak_gpu_pool_used_bytes'])}; "
            f"minimum effective free VRAM="
            f"{_format_optional_mib(resource_summary['minimum_effective_free_vram_bytes'])}."
        )
        print(f"Total elapsed time: {format_duration(elapsed)}.")

    return {
        "epochs": len(loss_history),
        "requested_epochs": epochs,
        "batch_size": tuning_summary["selected_batch_size"],
        "selected_batch_size": tuning_summary["selected_batch_size"],
        "total_examples": total_examples,
        "train_examples": train_count,
        "validation_examples": validation_count,
        "best_validation_loss": best_state["validation_loss"],
        "weight_decay": weight_decay,
        "early_stopping_patience": early_stopping_patience,
        "training_plateau_enabled": training_summary[
            "training_plateau_enabled"
        ],
        "training_plateau_window": training_summary[
            "training_plateau_window"
        ],
        "training_plateau_patience": training_summary[
            "training_plateau_patience"
        ],
        "training_plateau_min_epochs": training_summary[
            "training_plateau_min_epochs"
        ],
        "training_plateau_min_relative_improvement": training_summary[
            "training_plateau_min_relative_improvement"
        ],
        "training_plateau_checks_without_improvement": training_summary[
            "training_plateau_checks_without_improvement"
        ],
        "training_plateau_last_relative_improvement": training_summary[
            "training_plateau_last_relative_improvement"
        ],
        "training_plateau_loss_start_epoch": training_summary[
            "training_plateau_loss_start_epoch"
        ],
        "training_plateau_stopped": training_summary[
            "training_plateau_stopped"
        ],
        "stopping_reason": training_summary["stopping_reason"],
        "requested_device": requested_device,
        "selected_device": selected_device,
        "device_fallback_reason": fallback_reason,
        "host_storage_mode": dataset.storage_mode,
        "storage_mode": data_plan.storage_mode,
        "resident_window_examples": data_plan.resident_window_examples,
        "full_dataset_on_gpu": data_plan.full_dataset_on_gpu,
        "full_dataset_upload_seconds": data_plan.full_upload_seconds,
        "batch_autotune_attempts": tuning_summary["attempts"],
        "autotune_epochs_retained": tuning_summary["autotune_epochs_retained"],
        "initial_learning_rate": training_summary["initial_learning_rate"],
        "final_learning_rate": training_summary["final_learning_rate"],
        "lr_decay_factor": lr_decay_factor,
        "lr_decay_patience": lr_decay_patience,
        "lr_decay_count": training_summary["lr_decay_count"],
        **resource_summary,
        "resource_usage": resource_summary,
        "gpu_residency_probe": (
            None
            if residency_probe is None
            else {
                "capacity_examples": residency_probe.capacity_examples,
                "full_dataset": residency_probe.full_dataset,
                "attempts": residency_probe.attempts,
                "minimum_effective_free_vram_bytes": (
                    residency_probe.minimum_effective_free_vram_bytes
                ),
                "peak_pool_used_bytes": residency_probe.peak_pool_used_bytes,
            }
        ),
        "epoch_metrics": epoch_metrics,
        "weights_file": weights_file,
        "loss_plot_file": str(loss_plot_path),
        "duration_s": elapsed,
    }


def _nonnegative_float(value):
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _decay_factor(value):
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be greater than 0 and less than 1")
    return parsed


def add_optional_training_arguments(parser, *, include_device_alias=False):
    """Add supervised regularization, device, memory, and tuning controls."""
    group = parser.add_argument_group("supervised-training controls")
    group.add_argument(
        "--weight-decay",
        nargs="?",
        type=_nonnegative_float,
        const=DEFAULT_WEIGHT_DECAY,
        default=0.0,
        metavar="COEFFICIENT",
        help=f"Enable L2 weight decay (shortcut value: {DEFAULT_WEIGHT_DECAY}).",
    )
    group.add_argument(
        "--early-stopping",
        nargs="?",
        type=_positive_int,
        const=DEFAULT_EARLY_STOPPING_PATIENCE,
        default=None,
        metavar="PATIENCE",
        help="Stop after this many validation checks without improvement.",
    )
    decay = group.add_mutually_exclusive_group()
    decay.add_argument(
        "--lr-decay",
        nargs="?",
        type=_decay_factor,
        const=DEFAULT_SUPERVISED_LR_DECAY_FACTOR,
        default=DEFAULT_SUPERVISED_LR_DECAY_FACTOR,
        metavar="FACTOR",
        help="Plateau LR multiplier; enabled by default.",
    )
    decay.add_argument(
        "--no-lr-decay",
        action="store_const",
        const=None,
        dest="lr_decay",
        help="Disable supervised plateau LR decay.",
    )
    group.add_argument(
        "--lr-decay-patience",
        type=_positive_int,
        default=DEFAULT_SUPERVISED_LR_DECAY_PATIENCE,
        help="Failed validation checks required before each LR reduction.",
    )
    group.add_argument(
        "--sl-no-training-plateau-stop",
        dest="disable_training_plateau",
        action="store_true",
        default=False,
        help="Disable automatic stopping when median training loss saturates.",
    )
    group.add_argument(
        "--sl-training-plateau-window",
        type=_positive_int,
        default=DEFAULT_TRAINING_PLATEAU_WINDOW,
        metavar="EPOCHS",
        help="Non-overlapping epoch-block size for training-loss plateau checks.",
    )
    group.add_argument(
        "--sl-training-plateau-patience",
        type=_positive_int,
        default=DEFAULT_TRAINING_PLATEAU_PATIENCE,
        metavar="BLOCKS",
        help="Consecutive low-improvement blocks required before stopping.",
    )
    group.add_argument(
        "--sl-training-plateau-min-epochs",
        type=_positive_int,
        default=DEFAULT_TRAINING_PLATEAU_MIN_EPOCHS,
        metavar="EPOCHS",
        help="Minimum total epochs before training-loss stopping is allowed.",
    )
    group.add_argument(
        "--sl-training-plateau-min-relative-improvement",
        type=_nonnegative_float,
        default=DEFAULT_TRAINING_PLATEAU_MIN_RELATIVE_IMPROVEMENT,
        metavar="FRACTION",
        help="Minimum median-loss improvement that resets plateau patience.",
    )
    group.add_argument(
        "--sl-device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Supervised array backend.",
    )
    if include_device_alias:
        group.add_argument(
            "--device",
            choices=("auto", "cpu", "gpu"),
            dest="sl_device",
            help="Standalone alias for --sl-device.",
        )
    group.add_argument(
        "--sl-batch-size",
        type=_positive_int,
        default=None,
        help="Use a fixed supervised mini-batch and disable autotuning.",
    )
    group.add_argument(
        "--sl-no-batch-autotune",
        action="store_true",
        help="Use the device default batch without retained autotuning.",
    )
    group.add_argument(
        "--sl-memory-reserve-mb",
        type=_nonnegative_int,
        default=DATASET_MEMORY_RESERVE_MB,
        help="Host RAM reserve for supervised data and CPU training.",
    )
    group.add_argument(
        "--sl-gpu-memory-reserve-mb",
        type=_nonnegative_int,
        default=SUPERVISED_GPU_MEMORY_RESERVE_MB,
        help="VRAM reserve for supervised GPU training.",
    )
    group.add_argument(
        "--sl-seed",
        type=int,
        default=None,
        help="Fix supervised initialization and shuffle randomness.",
    )
    return parser


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the supervised-learning domino policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_optional_training_arguments(parser, include_device_alias=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train_supervised(
        batch_size=args.sl_batch_size,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping,
        lr_decay_factor=args.lr_decay,
        lr_decay_patience=args.lr_decay_patience,
        training_plateau_enabled=not args.disable_training_plateau,
        training_plateau_window=args.sl_training_plateau_window,
        training_plateau_patience=args.sl_training_plateau_patience,
        training_plateau_min_epochs=args.sl_training_plateau_min_epochs,
        training_plateau_min_relative_improvement=(
            args.sl_training_plateau_min_relative_improvement
        ),
        device=args.sl_device,
        autotune_batch_size=not args.sl_no_batch_autotune,
        memory_reserve_mb=args.sl_memory_reserve_mb,
        gpu_memory_reserve_mb=args.sl_gpu_memory_reserve_mb,
        seed=args.sl_seed,
    )


if __name__ == "__main__":
    main()
