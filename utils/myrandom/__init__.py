"""Central authority for deterministic and non-deterministic randomness.

The package is intentionally isolated from the rest of the project for now.
Future migrations should obtain every reproducible NumPy generator from a
``SeedPlan`` instead of using module-global random state.
"""

from .entropy import fresh_root_seed, unique_token
from .generators import DEFAULT_BIT_GENERATOR, supported_bit_generators
from .seed_plan import (
    DERIVATION_SCHEME,
    MANIFEST_SCHEMA_VERSION,
    RandomNamespace,
    SeedPlan,
)
from .serialization import (
    GENERATOR_STATE_SCHEMA_VERSION,
    generator_from_state,
    generator_state,
    read_generator_state,
    restore_generator_state,
    restore_generators,
    snapshot_generators,
    write_generator_state,
)

__all__ = [
    "DEFAULT_BIT_GENERATOR",
    "DERIVATION_SCHEME",
    "GENERATOR_STATE_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "RandomNamespace",
    "SeedPlan",
    "fresh_root_seed",
    "generator_from_state",
    "generator_state",
    "read_generator_state",
    "restore_generator_state",
    "restore_generators",
    "snapshot_generators",
    "supported_bit_generators",
    "unique_token",
    "write_generator_state",
]
