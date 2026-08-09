"""Stable namespace-and-coordinate derivation of independent NumPy streams."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from .generators import (
    DEFAULT_BIT_GENERATOR,
    create_generator,
    validate_bit_generator_name,
)


MANIFEST_SCHEMA_VERSION = 1
DERIVATION_SCHEME = "blake2b-128-namespace-coordinates-v1"
_DERIVATION_PERSON = b"domino-rng-v1"


class RandomNamespace(str, Enum):
    """Stable stream names for every current randomness responsibility."""

    DATASET_GAME = "dataset.game"
    SUPERVISED_INITIALIZATION = "supervised.initialization"
    SUPERVISED_SHUFFLE = "supervised.shuffle"
    SUPERVISED_DROPOUT = "supervised.dropout"
    RL_GAME = "rl.game"
    RL_POSITION = "rl.position"
    RL_OPPONENT = "rl.opponent"
    RL_POLICY = "rl.policy"
    RL_DROPOUT = "rl.dropout"
    PPO_MINIBATCH = "ppo.minibatch"
    DIAGNOSTIC_GAME = "diagnostic.game"
    AUTOTUNE_DATASET = "autotune.dataset"
    AUTOTUNE_SUPERVISED = "autotune.supervised"
    AUTOTUNE_RL = "autotune.rl"
    AUTOTUNE_DIAGNOSTIC = "autotune.diagnostic"
    BENCHMARK_GAME = "benchmark.game"
    UI_GAME = "ui.game"


def _validated_root_seed(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("root_seed must be a non-negative integer")
    if value < 0:
        raise ValueError("root_seed must be a non-negative integer")
    return value


def _validated_namespace(namespace):
    if not isinstance(namespace, RandomNamespace):
        raise TypeError("namespace must be a RandomNamespace")
    return namespace


def _coordinate_record(coordinate):
    if isinstance(coordinate, bool):
        raise TypeError("seed coordinates cannot be booleans")
    if isinstance(coordinate, int):
        if coordinate < 0:
            raise ValueError("integer seed coordinates must be non-negative")
        return {"type": "integer", "value": str(coordinate)}
    if isinstance(coordinate, str):
        if not coordinate:
            raise ValueError("string seed coordinates cannot be empty")
        return {"type": "string", "value": coordinate}
    raise TypeError("seed coordinates must be non-negative integers or strings")


def _spawn_key(namespace, coordinates):
    payload = {
        "coordinates": [_coordinate_record(value) for value in coordinates],
        "namespace": _validated_namespace(namespace).value,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.blake2b(
        encoded,
        digest_size=16,
        person=_DERIVATION_PERSON,
    ).digest()
    return struct.unpack("<4I", digest)


@dataclass(frozen=True, slots=True)
class SeedPlan:
    """Immutable source of order-independent NumPy random streams."""

    root_seed: int
    bit_generator: str = DEFAULT_BIT_GENERATOR

    def __post_init__(self):
        object.__setattr__(self, "root_seed", _validated_root_seed(self.root_seed))
        object.__setattr__(
            self,
            "bit_generator",
            validate_bit_generator_name(self.bit_generator),
        )

    def seed_sequence(self, namespace, *coordinates):
        """Return the deterministic SeedSequence for one logical stream."""
        return np.random.SeedSequence(
            entropy=self.root_seed,
            spawn_key=_spawn_key(namespace, coordinates),
        )

    def generator(self, namespace, *coordinates):
        """Return a new independent Generator for the logical stream."""
        return create_generator(
            self.seed_sequence(namespace, *coordinates),
            bit_generator=self.bit_generator,
        )

    def uint64_seed(self, namespace, *coordinates):
        """Return a deterministic scalar seed for process or API boundaries."""
        state = self.seed_sequence(namespace, *coordinates).generate_state(
            2,
            dtype=np.uint32,
        )
        return int(state[0]) | (int(state[1]) << 32)

    def to_manifest(self):
        """Return the small static manifest that identifies this seed plan."""
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "root_seed": self.root_seed,
            "bit_generator": self.bit_generator,
            "derivation_scheme": DERIVATION_SCHEME,
            "numpy_version": np.__version__,
            "registered_namespaces": [item.value for item in RandomNamespace],
        }

    @classmethod
    def from_manifest(cls, manifest):
        """Validate a manifest and rebuild its SeedPlan."""
        if not isinstance(manifest, dict):
            raise TypeError("manifest must be a dictionary")
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported random manifest schema version")
        if manifest.get("derivation_scheme") != DERIVATION_SCHEME:
            raise ValueError("unsupported random seed derivation scheme")
        registered = manifest.get("registered_namespaces")
        expected = [item.value for item in RandomNamespace]
        if registered != expected:
            raise ValueError("random manifest namespace registry does not match")
        return cls(
            root_seed=manifest.get("root_seed"),
            bit_generator=manifest.get("bit_generator"),
        )

    def write_manifest(self, path):
        """Atomically write the static seed manifest and return its Path."""
        from .serialization import atomic_write_json

        return atomic_write_json(path, self.to_manifest())

    @classmethod
    def from_manifest_file(cls, path):
        """Load and validate a seed manifest from disk."""
        with open(Path(path), "r", encoding="utf-8") as stream:
            return cls.from_manifest(json.load(stream))
