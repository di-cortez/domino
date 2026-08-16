"""Single source of truth for the domino policy-network architecture."""

from __future__ import annotations

from dataclasses import dataclass

from agents.encoder import DominoEncoder
from middleware.rulesets import DEFAULT_RULESET_NAME, resolve_ruleset


DEFAULT_HIDDEN1_SIZE = 256
DEFAULT_HIDDEN2_SIZE = 128
# The historical two-layer 256x128 network remains the default architecture.
DEFAULT_HIDDEN_SIZES = (DEFAULT_HIDDEN1_SIZE, DEFAULT_HIDDEN2_SIZE)
DEFAULT_HIDDEN_LAYER_COUNT = len(DEFAULT_HIDDEN_SIZES)
_DEFAULT_HIDDEN_SIZES_BY_RULESET = {
    "double-six": (256, 128),
    "double-five": (192, 96),
    "double-four": (128, 64),
    "double-three": (96, 48),
}
# Depth is not limited by the network implementation: every forward pass,
# gradient, checkpoint key, and metadata field is generated from the requested
# hidden stack, so ``NetworkArchitecture`` and both network classes accept any
# ``n >= 1``. This constant bounds only the command line, where one
# ``--hidden<n>-size`` option has to exist per layer.
MAX_HIDDEN_LAYER_COUNT = 8
# Width of any requested hidden layer the historical default does not describe.
FALLBACK_HIDDEN_SIZE = 128


def default_hidden_sizes(ruleset=DEFAULT_RULESET_NAME):
    """Return the compact two-layer default for one named ruleset."""
    return _DEFAULT_HIDDEN_SIZES_BY_RULESET[resolve_ruleset(ruleset).name]


def default_hidden_size(position, ruleset=DEFAULT_RULESET_NAME):
    """Return the width used when ``--hidden<position>-size`` is omitted."""
    if 1 <= position <= DEFAULT_HIDDEN_LAYER_COUNT:
        return default_hidden_sizes(ruleset)[position - 1]
    return FALLBACK_HIDDEN_SIZE


def validated_hidden_layer_count(value, maximum=None):
    """Return one supported hidden-layer count.

    ``maximum`` bounds the depth for callers that need one, such as the CLI
    with its fixed set of per-layer width options. Library callers leave it
    unset and may build a network of any depth.
    """
    count = int(value)
    if count < 1:
        raise ValueError(
            f"hidden_layer_count must be at least 1, got {value!r}."
        )
    if maximum is not None and count > int(maximum):
        raise ValueError(
            "hidden_layer_count must be between 1 and "
            f"{int(maximum)}, got {value!r}."
        )
    return count


def resolve_hidden_sizes(
    hidden_layer_count,
    requested_sizes=(),
    maximum=None,
    *,
    ruleset=DEFAULT_RULESET_NAME,
):
    """Return the widths of ``hidden_layer_count`` hidden layers.

    ``requested_sizes`` is indexed by layer position: entry ``i`` is the
    explicit ``--hidden<i+1>-size`` value or ``None`` when it was omitted. An
    omitted width falls back to :func:`default_hidden_size`, which keeps the
    unchanged 256x128 default for a two-layer network and uses
    ``FALLBACK_HIDDEN_SIZE`` for every deeper layer. Sizing a layer the
    requested architecture does not have is an error rather than a silently
    ignored flag. ``maximum`` is forwarded to
    :func:`validated_hidden_layer_count`.
    """
    count = validated_hidden_layer_count(hidden_layer_count, maximum=maximum)
    requested = tuple(requested_sizes)
    for position, size in enumerate(requested, start=1):
        if size is not None and position > count:
            raise ValueError(
                f"hidden{position}_size was requested, but --hidden-layers "
                f"{count} has no layer {position}."
            )
    sizes = []
    for position in range(1, count + 1):
        size = requested[position - 1] if position <= len(requested) else None
        sizes.append(
            default_hidden_size(position, ruleset) if size is None else int(size)
        )
    return tuple(sizes)


def policy_layer_names(hidden_layer_count):
    """Return the ordered weight/bias names of one policy architecture.

    The output layer keeps the index directly after the last hidden layer, so
    a two-layer network is still exactly ``W1, b1, W2, b2, W3, b3``.
    """
    names = []
    for index in range(1, int(hidden_layer_count) + 2):
        names.extend((f"W{index}", f"b{index}"))
    return tuple(names)


def hidden_layer_count_from_weights(weights):
    """Return the hidden-layer count stored in one policy weight mapping."""
    layer_count = 0
    while f"W{layer_count + 1}" in weights and f"b{layer_count + 1}" in weights:
        layer_count += 1
    if layer_count < 2:
        raise ValueError(
            "Policy weights must contain at least one hidden layer and one "
            "output layer named W1/b1 and W2/b2."
        )
    return layer_count - 1


@dataclass(frozen=True)
class NetworkArchitecture:
    """Serializable dimensions and derived policy-weight shapes.

    ``hidden_sizes`` may describe any depth from one layer upwards; only the
    command line bounds it, at :data:`MAX_HIDDEN_LAYER_COUNT`.
    """

    hidden_sizes: tuple = DEFAULT_HIDDEN_SIZES
    input_size: int = DominoEncoder.VECTOR_SIZE
    output_size: int = DominoEncoder.ACTION_SIZE
    dtype: str = "float32"

    def __post_init__(self):
        hidden_sizes = tuple(int(size) for size in self.hidden_sizes)
        if not hidden_sizes:
            raise ValueError("hidden_sizes must describe at least one layer")
        for position, size in enumerate(hidden_sizes, start=1):
            if size < 1:
                raise ValueError(f"hidden{position}_size must be positive")
        object.__setattr__(self, "hidden_sizes", hidden_sizes)
        for name in ("input_size", "output_size"):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

    @property
    def hidden_layer_count(self):
        """Return the number of hidden layers."""
        return len(self.hidden_sizes)

    @property
    def layer_dimensions(self):
        """Return input, hidden, and output widths in forward-pass order."""
        return (self.input_size, *self.hidden_sizes, self.output_size)

    def as_dict(self):
        """Return the stable metadata representation used by SL artifacts."""
        value = {"input_size": self.input_size}
        for position, size in enumerate(self.hidden_sizes, start=1):
            value[f"hidden{position}_size"] = size
        value["output_size"] = self.output_size
        value["dtype"] = self.dtype
        return value

    def as_list(self):
        """Return the compact representation used by canonical RL state."""
        return list(self.layer_dimensions)

    def policy_weight_shapes(self):
        """Return the expected supervised policy-array shapes."""
        dimensions = self.layer_dimensions
        shapes = {}
        for index in range(1, len(dimensions)):
            shapes[f"W{index}"] = (dimensions[index], dimensions[index - 1])
            shapes[f"b{index}"] = (dimensions[index], 1)
        return shapes


DEFAULT_NETWORK_ARCHITECTURE = NetworkArchitecture()


def architecture_from_hidden_sizes(
    *hidden_sizes,
    ruleset=DEFAULT_RULESET_NAME,
):
    """Build a policy architecture with one ruleset's policy dimensions."""
    if len(hidden_sizes) == 1 and not isinstance(hidden_sizes[0], int):
        hidden_sizes = tuple(hidden_sizes[0])
    encoder = DominoEncoder(ruleset)
    return NetworkArchitecture(
        hidden_sizes=hidden_sizes,
        input_size=encoder.vector_size,
        output_size=encoder.action_size,
    )


def architecture_for_ruleset(ruleset=DEFAULT_RULESET_NAME, hidden_sizes=None):
    """Return the compact default or an explicit hidden stack for a ruleset."""
    resolved = resolve_ruleset(ruleset)
    if hidden_sizes is None:
        hidden_sizes = default_hidden_sizes(resolved)
    return architecture_from_hidden_sizes(hidden_sizes, ruleset=resolved)


def architecture_from_weights(weights):
    """Build the architecture described by one policy weight mapping."""
    hidden_layer_count = hidden_layer_count_from_weights(weights)
    output_index = hidden_layer_count + 1
    return NetworkArchitecture(
        hidden_sizes=tuple(
            int(weights[f"W{index}"].shape[0])
            for index in range(1, output_index)
        ),
        input_size=int(weights["W1"].shape[1]),
        output_size=int(weights[f"W{output_index}"].shape[0]),
    )
