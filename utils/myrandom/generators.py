"""Construction and validation of the package's NumPy generators."""

import numpy as np


DEFAULT_BIT_GENERATOR = "PCG64"
_BIT_GENERATOR_TYPES = {
    DEFAULT_BIT_GENERATOR: np.random.PCG64,
}


def supported_bit_generators():
    """Return the stable names accepted by this package."""
    return tuple(_BIT_GENERATOR_TYPES)


def validate_bit_generator_name(name):
    """Return a supported bit-generator name or raise a useful error."""
    if not isinstance(name, str):
        raise TypeError("bit_generator must be a string")
    if name not in _BIT_GENERATOR_TYPES:
        supported = ", ".join(supported_bit_generators())
        raise ValueError(
            f"unsupported bit generator {name!r}; supported: {supported}"
        )
    return name


def create_bit_generator(name=DEFAULT_BIT_GENERATOR, seed_sequence=None):
    """Create one supported NumPy bit generator."""
    name = validate_bit_generator_name(name)
    generator_type = _BIT_GENERATOR_TYPES[name]
    return generator_type(seed_sequence) if seed_sequence is not None else generator_type()


def create_generator(seed_sequence, bit_generator=DEFAULT_BIT_GENERATOR):
    """Create a NumPy Generator from an explicit SeedSequence."""
    if not isinstance(seed_sequence, np.random.SeedSequence):
        raise TypeError("seed_sequence must be numpy.random.SeedSequence")
    return np.random.Generator(create_bit_generator(bit_generator, seed_sequence))


def bit_generator_name(generator):
    """Return and validate the stable name of a NumPy Generator backend."""
    if not isinstance(generator, np.random.Generator):
        raise TypeError("generator must be numpy.random.Generator")
    return validate_bit_generator_name(type(generator.bit_generator).__name__)
