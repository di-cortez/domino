"""Operating-system entropy kept separate from reproducible random streams."""

import secrets


DEFAULT_ROOT_SEED_BITS = 128
MINIMUM_ROOT_SEED_BITS = 32
MAXIMUM_ROOT_SEED_BITS = 256


def _validated_positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def fresh_root_seed(bit_count=DEFAULT_ROOT_SEED_BITS):
    """Return a non-deterministic root seed sourced from the operating system."""
    bit_count = _validated_positive_integer(bit_count, "bit_count")
    if not MINIMUM_ROOT_SEED_BITS <= bit_count <= MAXIMUM_ROOT_SEED_BITS:
        raise ValueError(
            "bit_count must be between "
            f"{MINIMUM_ROOT_SEED_BITS} and {MAXIMUM_ROOT_SEED_BITS}"
        )
    if bit_count % 8:
        raise ValueError("bit_count must be a multiple of 8")
    return secrets.randbits(bit_count)


def unique_token(byte_count=8):
    """Return an operating-system-backed hexadecimal token for technical IDs."""
    byte_count = _validated_positive_integer(byte_count, "byte_count")
    return secrets.token_hex(byte_count)
