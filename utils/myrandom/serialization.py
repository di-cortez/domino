"""JSON-safe persistence for NumPy generator states and seed manifests."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np

from .entropy import unique_token
from .generators import bit_generator_name, create_bit_generator


GENERATOR_STATE_SCHEMA_VERSION = 1


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_write_json(path, value):
    """Atomically publish one JSON document beside a unique temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{unique_token(4)}"
    )
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def generator_state(generator):
    """Return a detached, JSON-safe snapshot of one supported Generator."""
    return {
        "schema_version": GENERATOR_STATE_SCHEMA_VERSION,
        "bit_generator": bit_generator_name(generator),
        "state": _json_safe(deepcopy(generator.bit_generator.state)),
    }


def _validated_state_payload(payload):
    if not isinstance(payload, dict):
        raise TypeError("generator state payload must be a dictionary")
    if payload.get("schema_version") != GENERATOR_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported generator state schema version")
    if "bit_generator" not in payload or "state" not in payload:
        raise ValueError("generator state payload is incomplete")
    return payload


def generator_from_state(payload):
    """Create a Generator positioned at a previously captured state."""
    payload = _validated_state_payload(payload)
    bit_generator = create_bit_generator(payload["bit_generator"])
    bit_generator.state = deepcopy(payload["state"])
    return np.random.Generator(bit_generator)


def restore_generator_state(generator, payload):
    """Restore a compatible Generator in place and return it."""
    payload = _validated_state_payload(payload)
    if bit_generator_name(generator) != payload["bit_generator"]:
        raise ValueError("generator state uses a different bit generator")
    generator.bit_generator.state = deepcopy(payload["state"])
    return generator


def write_generator_state(path, generator):
    """Atomically write one Generator state as JSON."""
    return atomic_write_json(path, generator_state(generator))


def read_generator_state(path):
    """Read and validate one Generator state document."""
    with open(Path(path), "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    _validated_state_payload(payload)
    return payload


def snapshot_generators(generators):
    """Return named snapshots for a mapping of independent generators."""
    if not isinstance(generators, dict):
        raise TypeError("generators must be a dictionary")
    snapshots = {}
    for name, generator in generators.items():
        if not isinstance(name, str) or not name:
            raise ValueError("generator names must be non-empty strings")
        snapshots[name] = generator_state(generator)
    return {
        "schema_version": GENERATOR_STATE_SCHEMA_VERSION,
        "generators": snapshots,
    }


def restore_generators(payload):
    """Recreate a named generator mapping from snapshot_generators output."""
    if not isinstance(payload, dict):
        raise TypeError("generator collection payload must be a dictionary")
    if payload.get("schema_version") != GENERATOR_STATE_SCHEMA_VERSION:
        raise ValueError("unsupported generator collection schema version")
    generators = payload.get("generators")
    if not isinstance(generators, dict):
        raise ValueError("generator collection payload is incomplete")
    return {
        name: generator_from_state(state)
        for name, state in generators.items()
    }
