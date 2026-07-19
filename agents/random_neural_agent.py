"""Untrained neural-policy baseline for reproducible agent diagnostics."""

from agents.neural_agent import NeuralAgent
from agents.nn import SupervisedNeuralNetwork


class RandomNeuralAgent(NeuralAgent):
    """Play with the supervised architecture before any training updates.

    A fixed initialization seed makes this a stable benchmark: every matchup
    evaluates the same randomly initialized policy, while the network still
    starts from the exact initialization used by supervised training.
    """

    DEFAULT_SEED = 0

    @classmethod
    def create(cls, seed=DEFAULT_SEED, device="auto"):
        """Create an untrained policy without loading or saving a checkpoint."""
        network = SupervisedNeuralNetwork(random_seed=seed, device=device)
        return cls(network)
