"""Small NumPy/CuPy multilayer perceptron used by the domino agents."""

try:
    import cupy as np

    GPU_ENABLED = True
except ImportError:
    import numpy as np

    GPU_ENABLED = False


class SupervisedNeuralNetwork:
    """Three-layer MLP: 86 -> 256 -> 128 -> 58."""

    def __init__(
        self,
        input_size=86,
        hidden1_size=256,
        hidden2_size=128,
        output_size=58,
        learning_rate=0.01,
    ):
        self.lr = learning_rate

        self.W1 = np.random.randn(hidden1_size, input_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((hidden1_size, 1))
        self.W2 = np.random.randn(hidden2_size, hidden1_size) * np.sqrt(2.0 / hidden1_size)
        self.b2 = np.zeros((hidden2_size, 1))
        self.W3 = np.random.randn(output_size, hidden2_size) * np.sqrt(1.0 / hidden2_size)
        self.b3 = np.zeros((output_size, 1))
        self.cache = {}

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward(self, x):
        z1 = np.dot(self.W1, x) + self.b1
        a1 = self.relu(z1)
        z2 = np.dot(self.W2, a1) + self.b2
        a2 = self.relu(z2)
        z3 = np.dot(self.W3, a2) + self.b3
        a3 = self.softmax(z3)

        self.cache = {"X": x, "Z1": z1, "A1": a1, "Z2": z2, "A2": a2, "Z3": z3, "A3": a3}
        return a3

    def backward(self, y_target):
        m = y_target.shape[1]
        a3 = self.cache["A3"]
        a2 = self.cache["A2"]
        a1 = self.cache["A1"]
        x = self.cache["X"]

        dz3 = a3 - y_target
        dW3 = (1.0 / m) * np.dot(dz3, a2.T)
        db3 = (1.0 / m) * np.sum(dz3, axis=1, keepdims=True)

        da2 = np.dot(self.W3.T, dz3)
        dz2 = da2 * self.relu_derivative(self.cache["Z2"])
        dW2 = (1.0 / m) * np.dot(dz2, a1.T)
        db2 = (1.0 / m) * np.sum(dz2, axis=1, keepdims=True)

        da1 = np.dot(self.W2.T, dz2)
        dz1 = da1 * self.relu_derivative(self.cache["Z1"])
        dW1 = (1.0 / m) * np.dot(dz1, x.T)
        db1 = (1.0 / m) * np.sum(dz1, axis=1, keepdims=True)

        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return -(1.0 / m) * np.sum(y_target * np.log(a3 + 1e-8))

    def _as_float(self, value):
        if hasattr(value, "get"):
            value = value.get()
        return float(value)

    def _release_gpu_cache(self):
        """Release temporary CuPy blocks while keeping live arrays and weights."""
        self.cache = {}
        if GPU_ENABLED:
            np.get_default_memory_pool().free_all_blocks()

    def _batched_validation_loss(self, x_val, y_val, batch_size):
        total_loss = 0.0
        sample_count = x_val.shape[1]

        for i in range(0, sample_count, batch_size):
            x_batch = x_val[:, i:i + batch_size]
            y_batch = y_val[:, i:i + batch_size]
            batch_count = x_batch.shape[1]
            probabilities = self.forward(x_batch)
            batch_loss = -(1.0 / batch_count) * np.sum(
                y_batch * np.log(probabilities + 1e-8)
            )
            total_loss += self._as_float(batch_loss) * batch_count

        self._release_gpu_cache()
        return total_loss / sample_count

    def train(self, x_train, y_train, x_val=None, y_val=None, epochs=1500, batch_size=128, on_validation=None):
        """
        Train with mini-batch SGD.

        The shuffle step keeps only index arrays in memory and materializes one
        batch at a time. This avoids duplicating the full dataset in GPU memory.
        """
        loss_history = []
        sample_count = x_train.shape[1]

        for epoch in range(epochs):
            permutation = np.random.permutation(sample_count)
            epoch_loss = 0.0
            batch_counter = 0

            for i in range(0, sample_count, batch_size):
                batch_indices = permutation[i:i + batch_size]
                x_batch = x_train[:, batch_indices]
                y_batch = y_train[:, batch_indices]

                self.forward(x_batch)
                loss = self.backward(y_batch)
                epoch_loss += self._as_float(loss)
                batch_counter += 1

            mean_loss = epoch_loss / batch_counter
            loss_history.append(mean_loss)

            if epoch % 10 == 0:
                validation_text = ""
                if x_val is not None and y_val is not None:
                    validation_loss = self._batched_validation_loss(
                        x_val,
                        y_val,
                        batch_size=max(batch_size, 4096),
                    )
                    validation_text = f" | validation loss: {validation_loss:.4f}"
                    if on_validation is not None:
                        on_validation(epoch, validation_loss, self)

                print(f"Epoch {epoch} | training loss: {mean_loss:.4f}{validation_text}")

        return loss_history
