"""Train the supervised-learning domino policy from a JSONL dataset.

The supervised policy only learns real voluntary tile-play decisions. Forced
draw, pass, and single-option tile-play records are skipped because those turns
do not require a neural decision.
"""

import argparse
import json
import os
import time

import numpy as np

from agents.encoder import DominoEncoder
from agents.nn import SupervisedNeuralNetwork
from utils.runtime_status import format_duration, print_memory_report

try:
    import cupy as cp

    USE_GPU = True
    print("CuPy available. Training on GPU.")
except ImportError:
    import numpy as cp

    USE_GPU = False
    print("CuPy not found. Training on CPU.")

EPOCHS = 1000
BATCH_SIZE = 1024
EARLY_STOPPING_PATIENCE = 5
WEIGHT_DECAY = 0.0001
LR_DECAY_FACTOR = 0.5

CHECKPOINT_EVERY = 10
CHECKPOINT_DIR = "models/supervised_checkpoints"
ENCODED_CACHE_FILE = "dataset/supervised_dataset_encoded.npz"
ENCODED_FEATURE_VERSION = "opponent_suit_presence_v1"


def to_backend_array(matrix):
    """Convert a NumPy array loaded from disk to the active backend."""
    return cp.array(matrix)


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


def _save_encoded_cache(cache_file, x, y, metadata):
    """Persist encoded supervised arrays for faster future training runs."""
    cache_dir = os.path.dirname(cache_file)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    np.savez_compressed(
        cache_file,
        X=x,
        Y=y,
        **metadata,
    )
    print(f"Encoded dataset cache saved to {cache_file}.")


def load_or_build_dataset(file_path, encoder, cache_file=ENCODED_CACHE_FILE):
    """Load encoded ``X/Y`` arrays from cache, rebuilding when the cache is stale."""
    metadata = _dataset_metadata(file_path, encoder)

    if os.path.exists(cache_file):
        try:
            with np.load(cache_file, allow_pickle=False) as cache_data:
                if _cache_matches(cache_data, metadata):
                    x = cache_data["X"]
                    y = cache_data["Y"]
                    print(f"Loaded encoded dataset cache from {cache_file}.")
                    print(f"Dataset loaded. X: {x.shape}, Y: {y.shape}")
                    return x, y

            print(f"Encoded dataset cache is stale: {cache_file}. Rebuilding.")
        except (OSError, KeyError, ValueError) as exc:
            print(f"Could not read encoded dataset cache {cache_file}: {exc}. Rebuilding.")

    x, y = load_dataset(file_path, encoder)
    _save_encoded_cache(cache_file, x, y, metadata)
    return x, y


def load_dataset(file_path, encoder):
    """Load JSONL tile-play examples into ``X`` and one-hot ``Y`` matrices."""
    x_rows = []
    y_rows = []
    skipped_draw_pass = 0
    skipped_single_option = 0

    print(f"Loading dataset from {file_path}...")
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

            action_index = encoder._action_index(target_action)

            x_rows.append(encoder.encode_state(state))
            y = np.zeros((len(encoder.all_actions), 1))
            y[action_index, 0] = 1.0
            y_rows.append(y)

    if not x_rows:
        raise ValueError(
            "The dataset contains no real tile-play decisions after filtering "
            "draw/pass and single-option tile-play actions."
        )

    x = np.hstack(x_rows)
    y = np.hstack(y_rows)
    print(f"Dataset loaded. X: {x.shape}, Y: {y.shape}")
    print(f"Skipped forced draw/pass examples: {skipped_draw_pass}")
    print(f"Skipped single-option tile-play examples: {skipped_single_option}")
    return x, y


def main(
    dataset_file="dataset/supervised_dataset.jsonl",
    weights_file="models/domino_sl_weights.npz",
    cache_file=ENCODED_CACHE_FILE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=0.005,
    checkpoint_every=CHECKPOINT_EVERY,
    checkpoint_dir=CHECKPOINT_DIR,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    weight_decay=WEIGHT_DECAY,
    lr_decay_factor=LR_DECAY_FACTOR,
):
    start_time = time.time()

    print_memory_report("Supervised training startup memory")

    encoder = DominoEncoder()
    x_full, y_full = load_or_build_dataset(dataset_file, encoder, cache_file)

    total_examples = x_full.shape[1]
    train_count = int(total_examples * 0.85)
    indices = np.random.permutation(total_examples)
    train_indices = indices[:train_count]
    validation_indices = indices[train_count:]

    # Keep the full split in host (NumPy) memory. SupervisedNeuralNetwork.train()
    # moves one mini-batch at a time onto the GPU, so GPU memory stays
    # proportional to batch_size instead of the whole dataset, which can be
    # far larger than available VRAM once the dataset has millions of rows.
    x_train = x_full[:, train_indices]
    y_train = y_full[:, train_indices]
    x_val = x_full[:, validation_indices]
    y_val = y_full[:, validation_indices]
    print(f"Split complete: {x_train.shape[1]} train | {x_val.shape[1]} validation")

    network = SupervisedNeuralNetwork(
        input_size=DominoEncoder.VECTOR_SIZE,
        hidden1_size=256,
        hidden2_size=128,
        output_size=len(encoder.all_actions),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    if os.path.exists(weights_file):
        print(f"Existing supervised model found at {weights_file}. Resuming training.")

        weights = np.load(weights_file)

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
        print("No existing supervised model found. Training from scratch.")
        
    best_state = {"validation_loss": float("inf"), "weights": None}
    last_checkpoint_time = {"value": start_time}


    def to_numpy(matrix):
        return cp.asnumpy(matrix) if USE_GPU else matrix


    def save_checkpoint(current_network, epoch, validation_loss):
        """Save both an archival checkpoint and the active model used by UI/self-play."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(weights_file), exist_ok=True)

        checkpoint_file = os.path.join(
            checkpoint_dir,
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

        now = time.time()
        checkpoint_elapsed = now - last_checkpoint_time["value"]
        last_checkpoint_time["value"] = now

        print(
            f"  -> Checkpoint saved to {checkpoint_file} "
            f"(time since previous checkpoint: {format_duration(checkpoint_elapsed)})."
        )
        print(f"  -> Active supervised model updated at {weights_file}.")
        
    def save_if_best(epoch, validation_loss, current_network):
        if epoch % checkpoint_every == 0:
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
            print(f"  -> New best validation loss {validation_loss:.4f} at epoch {epoch}.")

    print("\nStarting supervised training...")
    network.train(
        x_train,
        y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        on_validation=save_if_best,
        early_stopping_patience=early_stopping_patience,
        lr_decay_factor=(
            lr_decay_factor if lr_decay_factor and 0 < lr_decay_factor < 1 else None
        ),
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
    print(
        f"Model saved to {weights_file} "
        f"(best validation loss: {best_state['validation_loss']:.4f})."
    )
    elapsed_time = time.time() - start_time
    print(f"Total elapsed time: {format_duration(elapsed_time)}.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the supervised-learning domino policy from a JSONL dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-file", type=str, default="dataset/supervised_dataset.jsonl")
    parser.add_argument("--weights-file", type=str, default="models/domino_sl_weights.npz")
    parser.add_argument("--cache-file", type=str, default=ENCODED_CACHE_FILE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help=(
            "Validation checks (every 10 epochs) without improvement before "
            "stopping early. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=WEIGHT_DECAY,
        help="L2 penalty applied to W1/W2/W3 during updates. Use 0 to disable.",
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=LR_DECAY_FACTOR,
        help=(
            "Multiply the learning rate by this factor on each validation "
            "check without improvement. Use 1 to disable."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_file=args.dataset_file,
        weights_file=args.weights_file,
        cache_file=args.cache_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_patience=(
            args.early_stopping_patience if args.early_stopping_patience > 0 else None
        ),
        weight_decay=args.weight_decay,
        lr_decay_factor=args.lr_decay_factor,
    )
