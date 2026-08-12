"""Train the supervised domino policy with a fixed, memory-checked batch.

The encoded dataset stays in host RAM when that is safe, falls back to a
disk-backed ``.npy`` cache when it is not, and may be kept fully or partially
resident on a selected GPU. Repeated low-improvement training-loss blocks can
stop a saturated run early.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from agents.encoder import DominoEncoder
from agents.network_architecture import (
    DEFAULT_HIDDEN_SIZES,
    architecture_from_hidden_sizes,
)
from agents.nn import (
    DISABLED_DROPOUT_RATE,
    DISABLED_WEIGHT_DECAY,
    SupervisedNeuralNetwork,
)
from training.supervised.dataset import (
    DATASET_MEMORY_RESERVE_MB,
    ENCODED_CACHE_FILE,
    load_or_build_dataset,
)
from training.supervised.plotting import save_supervised_loss_plot
from training.supervised.runtime import (
    DEFAULT_SUPERVISED_BATCH_SIZE,
    SUPERVISED_GPU_MEMORY_RESERVE_MB,
    SupervisedDataPlan,
    SupervisedResourceTracker,
    probe_gpu_residency,
)
from utils.resource_limits import (
    MIB,
    MemorySafetyError,
    choose_safe_supervised_device,
    effective_gpu_available_bytes,
    gpu_memory_info,
)
from utils.artifacts import atomic_savez
from utils.myrandom import (
    DEFAULT_BIT_GENERATOR,
    DERIVATION_SCHEME,
    SeedPlan,
    fresh_root_seed,
)
from utils.runtime_status import format_duration, memory_report


EPOCHS = 2000
BATCH_SIZE = DEFAULT_SUPERVISED_BATCH_SIZE
INITIAL_SUPERVISED_LEARNING_RATE = 0.005
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_SUPERVISED_LR_DECAY_PATIENCE = 5
DEFAULT_SUPERVISED_LR_DECAY_FACTOR = 0.5
SUPERVISED_VALIDATION_INTERVAL_EPOCHS = 10
DEFAULT_TRAINING_PLATEAU_WINDOW = 25
DEFAULT_TRAINING_PLATEAU_PATIENCE = 4
DEFAULT_TRAINING_PLATEAU_MIN_EPOCHS = 100
DEFAULT_TRAINING_PLATEAU_MIN_RELATIVE_IMPROVEMENT = 0.001

def _format_optional_mib(byte_count):
    """Format an optional byte measurement for detailed resource logs."""
    if byte_count is None:
        return "unavailable"
    return f"{byte_count / MIB:.1f} MiB"


def supervised_random_manifest_path(weights_file):
    """Return the seed-plan manifest path paired with supervised weights."""
    weights_path = Path(weights_file)
    return weights_path.with_name(
        f"{weights_path.stem}.random_manifest.json"
    )


def _network_weight_payload(network, weights=None):
    """Return six compatible host float32 arrays for checkpoint writing."""
    source = weights or {name: getattr(network, name) for name in network.weight_names}
    return {
        name: network.to_host(source[name]).astype(np.float32, copy=False)
        for name in network.weight_names
    }


def _create_network(
    *,
    device,
    weight_decay,
    dropout_rate,
    seed,
    architecture,
):
    return SupervisedNeuralNetwork(
        input_size=architecture.input_size,
        output_size=architecture.output_size,
        hidden_sizes=architecture.hidden_sizes,
        learning_rate=INITIAL_SUPERVISED_LEARNING_RATE,
        weight_decay=weight_decay,
        dropout_rate=dropout_rate,
        seed_plan=SeedPlan(seed),
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
    batch_size=DEFAULT_SUPERVISED_BATCH_SIZE,
    hidden_sizes=DEFAULT_HIDDEN_SIZES,
    dataset_file="dataset/supervised_dataset.jsonl",
    weights_file="models/domino_sl_weights.npz",
    cache_file=ENCODED_CACHE_FILE,
    quiet=False,
    progress_callback=None,
    weight_decay=DISABLED_WEIGHT_DECAY,
    dropout_rate=DISABLED_DROPOUT_RATE,
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
    memory_reserve_mb=DATASET_MEMORY_RESERVE_MB,
    gpu_memory_reserve_mb=SUPERVISED_GPU_MEMORY_RESERVE_MB,
    seed=None,
):
    """Train a fresh policy and return scheduler, storage, batch, and memory data."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    architecture = architecture_from_hidden_sizes(hidden_sizes)
    started = time.time()
    seed = int(seed) if seed is not None else fresh_root_seed()
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
        architecture=architecture,
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
            dropout_rate=dropout_rate,
            seed=seed,
            architecture=architecture,
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
            dropout_rate=dropout_rate,
            seed=seed,
            architecture=architecture,
        )
    residency_probe = None
    resident_capacity = None

    if selected_device == "gpu":
        residency_probe = probe_gpu_residency(
            dataset.x,
            dataset.y,
            reserve_mb=gpu_memory_reserve_mb,
            hidden_sizes=architecture.hidden_sizes,
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
                dropout_rate=dropout_rate,
                seed=seed,
                architecture=architecture,
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
            seed_plan=network.seed_plan,
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
            dropout_rate=dropout_rate,
            seed=seed,
            architecture=architecture,
        )
        data_plan = SupervisedDataPlan(
            dataset.x,
            dataset.y,
            train_count=train_count,
            host_storage_mode=dataset.storage_mode,
            device="cpu",
            seed_plan=network.seed_plan,
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

    if not quiet:
        print("Supervised training starts from fresh random weights.")

    requested_batch_size = int(batch_size)
    if requested_batch_size < 1:
        raise ValueError("batch_size must be positive")
    selected_batch_size = min(requested_batch_size, train_count)
    batch_memory_preflight = dict(
        data_plan.batch_memory_preflight(
            network,
            selected_batch_size,
            (
                gpu_memory_reserve_mb
                if selected_device == "gpu"
                else memory_reserve_mb
            ),
        )
    )
    if not batch_memory_preflight.get("safe", False):
        reason = batch_memory_preflight.get(
            "reason",
            "memory preflight rejected the fixed batch",
        )
        raise MemorySafetyError(
            f"Fixed supervised batch {selected_batch_size:,} is unsafe: {reason}."
        )
    if not quiet:
        print(
            f"Supervised batch size: {selected_batch_size:,}"
            + (
                f" (requested {requested_batch_size:,}; capped to training set)."
                if selected_batch_size != requested_batch_size
                else "."
            )
        )

    tracker = SupervisedResourceTracker(selected_device)
    best_state = {
        "validation_loss": float("inf"),
        "epoch": None,
        "weights": None,
    }
    def save_if_best(epoch, validation_loss, current_network):
        if validation_loss < best_state["validation_loss"]:
            best_state["validation_loss"] = validation_loss
            best_state["epoch"] = int(epoch) + 1
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
            batch_size=selected_batch_size,
            on_validation=save_if_best,
            progress_callback=progress_callback,
            quiet=quiet,
            early_stopping_patience=early_stopping_patience,
            lr_decay_factor=lr_decay_factor,
            lr_decay_patience=lr_decay_patience,
            validation_interval=SUPERVISED_VALIDATION_INTERVAL_EPOCHS,
            epoch_runner=data_plan.train_epoch,
            validation_runner=data_plan.validation_loss,
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
        atomic_savez(
            weights_path,
            **_network_weight_payload(network, best_state["weights"]),
        )
        network.seed_plan.write_manifest(
            supervised_random_manifest_path(weights_path)
        )
    finally:
        tracker.observe()
        data_plan.close()

    loss_plot_path = save_supervised_loss_plot(
        loss_history,
        epoch_metrics,
        weights_file,
    )

    elapsed = time.time() - started
    training_summary = network.last_training_summary
    final_validation_loss = next(
        (
            float(metrics["validation_loss"])
            for metrics in reversed(epoch_metrics)
            if metrics.get("validation_loss") is not None
        ),
        None,
    )
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
        print(
            "Random manifest saved to "
            f"{supervised_random_manifest_path(weights_file)}."
        )
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
        "best_epoch": best_state["epoch"],
        "early_stopping_triggered": (
            training_summary["stopping_reason"] != "epoch_limit"
        ),
        "final_training_loss": float(loss_history[-1]),
        "final_validation_loss": final_validation_loss,
        "batch_size": selected_batch_size,
        "requested_batch_size": requested_batch_size,
        "selected_batch_size": selected_batch_size,
        "network_architecture": architecture.as_dict(),
        "effective_seed": seed,
        "random_bit_generator": DEFAULT_BIT_GENERATOR,
        "random_derivation_scheme": DERIVATION_SCHEME,
        "random_manifest_path": str(
            supervised_random_manifest_path(weights_file)
        ),
        "random_manifest": network.seed_plan.to_manifest(),
        "total_examples": total_examples,
        "train_examples": train_count,
        "validation_examples": validation_count,
        "best_validation_loss": best_state["validation_loss"],
        "weight_decay": weight_decay,
        "dropout_rate": dropout_rate,
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
        "batch_memory_preflight": batch_memory_preflight,
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
