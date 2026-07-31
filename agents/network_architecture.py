"""Single source of truth for the domino policy-network architecture."""

from __future__ import annotations

from dataclasses import dataclass

from agents.encoder import DominoEncoder


DEFAULT_HIDDEN1_SIZE = 256
DEFAULT_HIDDEN2_SIZE = 128


@dataclass(frozen=True)
class NetworkArchitecture:
    """Serializable dimensions and derived policy-weight shapes."""

    hidden1_size: int = DEFAULT_HIDDEN1_SIZE
    hidden2_size: int = DEFAULT_HIDDEN2_SIZE
    input_size: int = DominoEncoder.VECTOR_SIZE
    output_size: int = DominoEncoder.ACTION_SIZE
    dtype: str = "float32"

    def __post_init__(self):
        for name in ("input_size", "hidden1_size", "hidden2_size", "output_size"):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

    def as_dict(self):
        """Return the stable metadata representation used by SL artifacts."""
        return {
            "input_size": self.input_size,
            "hidden1_size": self.hidden1_size,
            "hidden2_size": self.hidden2_size,
            "output_size": self.output_size,
            "dtype": self.dtype,
        }

    def as_list(self):
        """Return the compact representation used by canonical RL state."""
        return [
            self.input_size,
            self.hidden1_size,
            self.hidden2_size,
            self.output_size,
        ]

    def policy_weight_shapes(self):
        """Return the six expected supervised policy-array shapes."""
        return {
            "W1": (self.hidden1_size, self.input_size),
            "b1": (self.hidden1_size, 1),
            "W2": (self.hidden2_size, self.hidden1_size),
            "b2": (self.hidden2_size, 1),
            "W3": (self.output_size, self.hidden2_size),
            "b3": (self.output_size, 1),
        }


DEFAULT_NETWORK_ARCHITECTURE = NetworkArchitecture()


def architecture_from_hidden_sizes(hidden1_size, hidden2_size):
    """Build a policy architecture with the fixed encoder/action dimensions."""
    return NetworkArchitecture(
        hidden1_size=hidden1_size,
        hidden2_size=hidden2_size,
    )
