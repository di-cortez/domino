"""Train the supervised-learning domino policy from a JSONL dataset.

The supervised policy only learns real voluntary tile-play decisions. Forced
draw, pass, and single-option tile-play records are skipped because those turns
do not require a neural decision.
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from agents.encoder import DominoEncoder
from agents.nn import (
    GPU_ENABLED,
    GPU_UNAVAILABLE_REASON,
    SupervisedNeuralNetwork,
)
from utils.resource_limits import ensure_ram_available
from utils.runtime_status import format_duration, print_memory_report

if GPU_ENABLED:
    import cupy as cp

    USE_GPU = True
    print("CuPy available. Training on GPU.")
else:
    import numpy as cp

    USE_GPU = False
    print(
        "GPU backend unavailable; training on CPU"
        f" ({GPU_UNAVAILABLE_REASON or 'CuPy is not available'})."
    )

EPOCHS = 2000
BATCH_SIZE = 1024
DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_EARLY_STOPPING_PATIENCE = 5
DEFAULT_LR_DECAY_FACTOR = 0.5

CHECKPOINT_EVERY = 10
CHECKPOINT_DIR = "models/supervised_checkpoints"
MAX_SUPERVISED_CHECKPOINTS = 10
ENCODED_CACHE_FILE = "dataset/supervised_dataset_encoded.npz"
ENCODED_FEATURE_VERSION = "opponent_suit_presence_float32_v2"
DATASET_DTYPE = np.float32
DATASET_MEMORY_RESERVE_MB = 512


def to_backend_array(matrix):
    """Convert a NumPy array loaded from disk to the active backend."""
    return cp.array(matrix)


def _prune_supervised_checkpoints(
    checkpoint_dir=CHECKPOINT_DIR,
    keep_count=MAX_SUPERVISED_CHECKPOINTS,
):
    """Delete older archival checkpoints and return the removed paths."""
    if keep_count < 1:
        raise ValueError("keep_count must be at least one.")

    checkpoint_paths = list(
        Path(checkpoint_dir).glob("domino_sl_epoch_*.npz")
    )
    checkpoint_paths.sort(
        key=lambda path: (path.stat().st_mtime_ns, path.name),
        reverse=True,
    )

    removed_paths = checkpoint_paths[keep_count:]
    for path in removed_paths:
        path.unlink()
    return removed_paths


def _normalize_action(action):
    """Return a normalized tile-play action or None for draw/pass."""
    if action is None:
        return None
    if action == ["DRAW", None] or action == ("DRAW", None):
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
            opening_double = max(doubles, key=lambda tile: tile[0])
            return [(opening_double, 0)]
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
    """Return True when the player had at least two legal tile-play choices."""
    return len(_legal_tile_actions_from_state(state)) >= 2


def _dataset_metadata(file_path, encoder):
    """Return metadata used to decide whether an encoded cache is reusable."""
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
    """Return True when ``cache_data`` was built from the current dataset/encoder."""
    for key, expected_value in expected_metadata.items():
        if key not in cache_data:
            return False

        cached_value = cache_data[key].item()
        if cached_value != expected_value:
            return False

    return True


def _save_encoded_cache(cache_file, x, y, metadata, quiet=False):
    """Persist encoded supervised arrays for faster future training runs."""
    cache_dir = os.path.dirname(cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    np.savez_compressed(
        cache_file,
        X=x,
        Y=y,
        encoded_example_count=x.shape[1],
        encoded_bytes=x.nbytes + y.nbytes,
        **metadata,
    )
    if not quiet:
        print(f"Encoded dataset cache saved to {cache_file}.")


def load_or_build_dataset(file_path, encoder, cache_file=ENCODED_CACHE_FILE, quiet=False):
    """Load encoded ``X/Y`` arrays from cache, rebuilding when the cache is stale."""
    metadata = _dataset_metadata(file_path, encoder)

    if os.path.exists(cache_file):
        try:
            with np.load(cache_file, allow_pickle=False) as cache_data:
                if _cache_matches(cache_data, metadata):
                    if "encoded_bytes" in cache_data:
                        ensure_ram_available(
                            int(cache_data["encoded_bytes"].item()),
                            DATASET_MEMORY_RESERVE_MB,
                            "loading the encoded supervised dataset",
                        )
                    x = cache_data["X"]
                    y = cache_data["Y"]
                    if not quiet:
                        print(f"Loaded encoded dataset cache from {cache_file}.")
                        print(f"Dataset loaded. X: {x.shape}, Y: {y.shape}")
                    return x, y

            if not quiet:
                print(f"Encoded dataset cache is stale: {cache_file}. Rebuilding.")
        except (OSError, KeyError, ValueError) as exc:
            if not quiet:
                print(f"Could not read encoded dataset cache {cache_file}: {exc}. Rebuilding.")

    x, y = load_dataset(file_path, encoder, quiet=quiet)
    _save_encoded_cache(cache_file, x, y, metadata, quiet=quiet)
    return x, y


def load_dataset(file_path, encoder, quiet=False):
    """Load JSONL examples with a two-pass, preallocated float32 encoder.

    The previous list-of-column-arrays approach retained one Python/NumPy
    object per example and then duplicated all data during ``hstack``.  A
    counting pass lets us validate RAM headroom and allocate the two final
    matrices exactly once.
    """
    skipped_draw_pass = 0
    skipped_single_option = 0
    example_count = 0

    if not quiet:
        print(f"Scanning dataset from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            state = record["state"]
            target_action = _normalize_action(record["target_action"])

            if target_action is None:
                skipped_draw_pass += 1
                continue

            if not _is_real_decision_state(state):
                skipped_single_option += 1
                continue

            example_count += 1

    if not example_count:
        raise ValueError(
            "The dataset contains no real tile-play decisions after filtering "
            "draw/pass and single-option tile-play actions."
        )

    required_bytes = example_count * (
        encoder.VECTOR_SIZE + len(encoder.all_actions)
    ) * np.dtype(DATASET_DTYPE).itemsize
    ensure_ram_available(
        required_bytes,
        DATASET_MEMORY_RESERVE_MB,
        "encoding the supervised dataset",
    )
    x = np.empty((encoder.VECTOR_SIZE, example_count), dtype=DATASET_DTYPE)
    y = np.zeros((len(encoder.all_actions), example_count), dtype=DATASET_DTYPE)

    if not quiet:
        print(
            f"Encoding {example_count} examples into one "
            f"{required_bytes / (1024 * 1024):.1f} MiB allocation..."
        )
    column = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            state = record["state"]
            target_action = _normalize_action(record["target_action"])
            if target_action is None or not _is_real_decision_state(state):
                continue

            action_index = encoder._action_index(target_action)
            x[:, column] = encoder.encode_state(state)[:, 0]
            y[action_index, column] = 1.0
            column += 1

    if column != example_count:
        raise RuntimeError(
            f"dataset changed while it was being encoded: expected "
            f"{example_count} examples, read {column}"
        )
    if not quiet:
        print(f"Dataset loaded. X: {x.shape}, Y: {y.shape}")
        print(f"Skipped forced draw/pass examples: {skipped_draw_pass}")
        print(f"Skipped single-option tile-play examples: {skipped_single_option}")
    return x, y


def train_supervised(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    dataset_file="dataset/supervised_dataset.jsonl",
    weights_file="models/domino_sl_weights.npz",
    cache_file=ENCODED_CACHE_FILE,
    quiet=False,
    progress_callback=None,
    weight_decay=0.0,
    early_stopping_patience=None,
    lr_decay_factor=None,
):
    """Train the supervised policy and return a compact run summary.

    Regularization, early stopping, and learning-rate decay are opt-in. The
    default call keeps a fixed learning rate and runs every requested epoch.
    """
    start_time = time.time()

    if not quiet:
        print_memory_report("Supervised training startup memory")

    encoder = DominoEncoder()
    x_full, y_full = load_or_build_dataset(dataset_file, encoder, cache_file, quiet=quiet)

    total_examples = x_full.shape[1]
    train_count = int(total_examples * 0.85)
    # Dataset games and actions are already generated in randomized order.
    # Contiguous views avoid duplicating all four split matrices at peak RAM;
    # the network still shuffles the training columns independently each epoch.
    x_train = x_full[:, :train_count]
    y_train = y_full[:, :train_count]
    x_val = x_full[:, train_count:]
    y_val = y_full[:, train_count:]
    if not quiet:
        print(f"Split complete: {x_train.shape[1]} train | {x_val.shape[1]} validation")

    network = SupervisedNeuralNetwork(
        input_size=DominoEncoder.VECTOR_SIZE,
        hidden1_size=256,
        hidden2_size=128,
        output_size=len(encoder.all_actions),
        learning_rate=0.005,
        weight_decay=weight_decay,
    )

    if os.path.exists(weights_file):
        if not quiet:
            print(f"Existing supervised model found at {weights_file}. Resuming training.")

        with np.load(weights_file, allow_pickle=False) as weights:
            expected_shapes = {
                "W1": network.W1.shape,
                "b1": network.b1.shape,
                "W2": network.W2.shape,
                "b2": network.b2.shape,
                "W3": network.W3.shape,
                "b3": network.b3.shape,
            }

            for name, expected_shape in expected_shapes.items():
                if weights[name].shape != expected_shape:
                    raise ValueError(
                        f"Cannot resume from {weights_file}: {name} has shape "
                        f"{weights[name].shape}, but expected {expected_shape}. "
                        "Delete or move the old checkpoint and retrain from scratch."
                    )

            network.W1 = to_backend_array(weights["W1"])
            network.b1 = to_backend_array(weights["b1"])
            network.W2 = to_backend_array(weights["W2"])
            network.b2 = to_backend_array(weights["b2"])
            network.W3 = to_backend_array(weights["W3"])
            network.b3 = to_backend_array(weights["b3"])
    else:
        if not quiet:
            print("No existing supervised model found. Training from scratch.")
        
    best_state = {"validation_loss": float("inf"), "weights": None}
    last_checkpoint_time = {"value": start_time}


    def to_numpy(matrix):
        return cp.asnumpy(matrix) if USE_GPU else matrix


    def save_checkpoint(current_network, epoch, validation_loss):
        """Save both an archival checkpoint and the active model used by UI/self-play."""
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(weights_file), exist_ok=True)

        checkpoint_file = os.path.join(
            CHECKPOINT_DIR,
            f"domino_sl_epoch_{epoch:04d}_val_{validation_loss:.4f}.npz",
        )

        weights_payload = {
            "W1": to_numpy(current_network.W1),
            "b1": to_numpy(current_network.b1),
            "W2": to_numpy(current_network.W2),
            "b2": to_numpy(current_network.b2),
            "W3": to_numpy(current_network.W3),
            "b3": to_numpy(current_network.b3),
        }

        np.savez(checkpoint_file, **weights_payload)
        np.savez(weights_file, **weights_payload)
        removed_checkpoints = _prune_supervised_checkpoints(CHECKPOINT_DIR)

        now = time.time()
        checkpoint_elapsed = now - last_checkpoint_time["value"]
        last_checkpoint_time["value"] = now

        if not quiet:
            print(
                f"  -> Checkpoint saved to {checkpoint_file} "
                f"(time since previous checkpoint: {format_duration(checkpoint_elapsed)})."
            )
            if removed_checkpoints:
                print(
                    "  -> Removed "
                    f"{len(removed_checkpoints)} older supervised checkpoint(s); "
                    f"keeping the latest {MAX_SUPERVISED_CHECKPOINTS}."
                )
            print(f"  -> Active supervised model updated at {weights_file}.")
        
    def save_if_best(epoch, validation_loss, current_network):
        if epoch % CHECKPOINT_EVERY == 0:
            save_checkpoint(current_network, epoch, validation_loss)

        if validation_loss < best_state["validation_loss"]:
            best_state["validation_loss"] = validation_loss
            best_state["weights"] = {
                "W1": current_network.W1.copy(),
                "b1": current_network.b1.copy(),
                "W2": current_network.W2.copy(),
                "b2": current_network.b2.copy(),
                "W3": current_network.W3.copy(),
                "b3": current_network.b3.copy(),
            }
            if not quiet:
                print(f"  -> New best validation loss {validation_loss:.4f} at epoch {epoch}.")

    if not quiet:
        print("\nStarting supervised training...")
    loss_history = network.train(
        x_train,
        y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        on_validation=save_if_best,
        progress_callback=progress_callback,
        quiet=quiet,
        early_stopping_patience=early_stopping_patience,
        lr_decay_factor=lr_decay_factor,
    )

    os.makedirs(os.path.dirname(weights_file), exist_ok=True)
    weights_to_save = best_state["weights"] or {
        "W1": network.W1,
        "b1": network.b1,
        "W2": network.W2,
        "b2": network.b2,
        "W3": network.W3,
        "b3": network.b3,
    }

    np.savez(
        weights_file,
        W1=to_numpy(weights_to_save["W1"]),
        b1=to_numpy(weights_to_save["b1"]),
        W2=to_numpy(weights_to_save["W2"]),
        b2=to_numpy(weights_to_save["b2"]),
        W3=to_numpy(weights_to_save["W3"]),
        b3=to_numpy(weights_to_save["b3"]),
    )
    if not quiet:
        print(
            f"Model saved to {weights_file} "
            f"(best validation loss: {best_state['validation_loss']:.4f})."
        )
    elapsed_time = time.time() - start_time
    if not quiet:
        print(f"Total elapsed time: {format_duration(elapsed_time)}.")

    return {
        "epochs": len(loss_history),
        "requested_epochs": epochs,
        "batch_size": batch_size,
        "total_examples": total_examples,
        "train_examples": x_train.shape[1],
        "validation_examples": x_val.shape[1],
        "best_validation_loss": best_state["validation_loss"],
        "weight_decay": weight_decay,
        "early_stopping_patience": early_stopping_patience,
        "lr_decay_factor": lr_decay_factor,
        "final_learning_rate": network.lr,
        "weights_file": weights_file,
        "duration_s": elapsed_time,
    }


def _nonnegative_float(value):
    """Parse a non-negative command-line floating-point value."""
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_int(value):
    """Parse a positive command-line integer value."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _decay_factor(value):
    """Parse a multiplicative decay factor strictly between zero and one."""
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be greater than 0 and less than 1")
    return parsed


def add_optional_training_arguments(parser):
    """Add opt-in SL regularization and scheduling flags to ``parser``."""
    group = parser.add_argument_group("optional supervised-training controls")
    group.add_argument(
        "--weight-decay",
        nargs="?",
        type=_nonnegative_float,
        const=DEFAULT_WEIGHT_DECAY,
        default=0.0,
        metavar="COEFFICIENT",
        help=(
            "Enable L2 weight decay. When passed without a value, use "
            f"{DEFAULT_WEIGHT_DECAY}."
        ),
    )
    group.add_argument(
        "--early-stopping",
        nargs="?",
        type=_positive_int,
        const=DEFAULT_EARLY_STOPPING_PATIENCE,
        default=None,
        metavar="PATIENCE",
        help=(
            "Stop after this many validation checks without improvement. "
            f"When passed without a value, use {DEFAULT_EARLY_STOPPING_PATIENCE}."
        ),
    )
    group.add_argument(
        "--lr-decay",
        nargs="?",
        type=_decay_factor,
        const=DEFAULT_LR_DECAY_FACTOR,
        default=None,
        metavar="FACTOR",
        help=(
            "Multiply the learning rate by this factor after each failed "
            "validation check. When omitted, keep the learning rate fixed."
        ),
    )
    return parser


def parse_args(argv=None):
    """Return optional supervised-training controls from the command line."""
    parser = argparse.ArgumentParser(
        description="Train the supervised-learning domino policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_optional_training_arguments(parser)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train_supervised(
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping,
        lr_decay_factor=args.lr_decay,
    )


if __name__ == "__main__":
    main()
