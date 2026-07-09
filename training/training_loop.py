"""Train the supervised-learning domino policy from a JSONL dataset."""

import json
import os

import numpy as np

from agents.encoder import DominoEncoder
from agents.nn import SupervisedNeuralNetwork

try:
    import cupy as cp

    USE_GPU = True
    print("CuPy available. Training on GPU.")
except ImportError:
    import numpy as cp

    USE_GPU = False
    print("CuPy not found. Training on CPU.")

EPOCHS = 1000
BATCH_SIZE = 128


def load_dataset(file_path, encoder):
    """Load JSONL state/action examples into ``X`` and one-hot ``Y`` matrices."""
    x_rows = []
    y_rows = []

    print(f"Loading dataset from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            state = record["state"]
            target_action = record["target_action"]

            x_rows.append(encoder.encode_state(state))

            if target_action is None:
                action_index = 57
            elif target_action == ["DRAW", None] or target_action == ("DRAW", None):
                action_index = 56
            else:
                if isinstance(target_action[0], list):
                    target_action = (tuple(target_action[0]), target_action[1])
                elif isinstance(target_action, list):
                    target_action = tuple(target_action)
                action_index = encoder.action_to_index[target_action]

            y = np.zeros((58, 1))
            y[action_index, 0] = 1.0
            y_rows.append(y)

    x = np.hstack(x_rows)
    y = np.hstack(y_rows)
    print(f"Dataset loaded. X: {x.shape}, Y: {y.shape}")
    return x, y


def main():
    dataset_file = "dataset/supervised_dataset.jsonl"
    weights_file = "models/domino_sl_weights.npz"

    encoder = DominoEncoder()
    x_full, y_full = load_dataset(dataset_file, encoder)

    total_examples = x_full.shape[1]
    train_count = int(total_examples * 0.85)
    indices = np.random.permutation(total_examples)
    train_indices = indices[:train_count]
    validation_indices = indices[train_count:]

    x_train = cp.array(x_full[:, train_indices])
    y_train = cp.array(y_full[:, train_indices])
    x_val = cp.array(x_full[:, validation_indices])
    y_val = cp.array(y_full[:, validation_indices])
    print(f"Split complete: {x_train.shape[1]} train | {x_val.shape[1]} validation")

    network = SupervisedNeuralNetwork(
        input_size=DominoEncoder.VECTOR_SIZE,
        hidden1_size=256,
        hidden2_size=128,
        output_size=58,
        learning_rate=0.005,
    )

    best_state = {"validation_loss": float("inf"), "weights": None}

    def save_if_best(epoch, validation_loss, current_network):
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
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        on_validation=save_if_best,
    )

    def to_numpy(matrix):
        return cp.asnumpy(matrix) if USE_GPU else matrix

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


if __name__ == "__main__":
    main()
