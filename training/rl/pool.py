"""Opponent-pool admission, retention, persistence, and rollout selection.

The current policy is intentionally simple: every updated learner is admitted,
the oldest snapshot is replaced when a fixed-size FIFO ring is full, and every
retained opponent is eligible for uniform sampling. These decisions live behind
``OpponentPool`` so future minipools can change admission, retention, and
matchmaking without leaking those policies into the RL training loop or worker
executor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from multiprocessing import shared_memory

import numpy as np


@dataclass(frozen=True)
class SharedPolicyDescriptor:
    """Pickle-friendly description of one policy stored in shared memory."""

    name: str
    weight_names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    element_count: int
    dtype: str


class _SnapshotPolicy:
    """Minimal attribute-backed policy used while restoring saved snapshots."""

    def __init__(self, weights):
        for name, value in weights.items():
            setattr(self, name, value)


class SharedPolicyBank:
    """Own the learner slot and shared-memory slots used by the opponent pool."""

    def __init__(self, network, max_pool_size):
        if max_pool_size < 0:
            raise ValueError("max_pool_size must be non-negative")
        self.weight_names = tuple(network.weight_names)
        self.shapes = tuple(
            tuple(int(value) for value in getattr(network, name).shape)
            for name in self.weight_names
        )
        first_weight = getattr(network, self.weight_names[0])
        if hasattr(first_weight, "get"):
            first_weight = first_weight.get()
        self.dtype = np.asarray(first_weight).dtype
        self.element_count = sum(math.prod(shape) for shape in self.shapes)
        self._segments = []
        self._descriptors = []
        self._closed = False
        try:
            for _slot in range(1 + max_pool_size):
                segment = shared_memory.SharedMemory(
                    create=True,
                    size=self.element_count * self.dtype.itemsize,
                )
                self._segments.append(segment)
                self._descriptors.append(SharedPolicyDescriptor(
                    name=segment.name,
                    weight_names=self.weight_names,
                    shapes=self.shapes,
                    element_count=self.element_count,
                    dtype=self.dtype.str,
                ))
        except BaseException:
            self.close()
            raise

        self.max_pool_size = int(max_pool_size)
        self.pool_slots = []
        self._pool_metadata = {}
        self._next_snapshot_id = 0
        self._next_pool_slot = 0
        self.write_current(network)

    @property
    def current_descriptor(self):
        return self._descriptors[0]

    @property
    def pool_descriptors(self):
        return tuple(self._descriptors[1:])

    @property
    def allocated_bytes(self):
        return len(self._segments) * self.element_count * self.dtype.itemsize

    def _write_slot(self, slot_index, network):
        flat = np.ndarray(
            (self.element_count,),
            dtype=self.dtype,
            buffer=self._segments[slot_index].buf,
        )
        offset = 0
        for name, shape in zip(self.weight_names, self.shapes):
            value = getattr(network, name)
            if hasattr(value, "get"):
                value = value.get()
            value = np.asarray(value, dtype=self.dtype)
            if value.shape != shape:
                raise ValueError(
                    f"Policy weight {name} changed shape from {shape} to "
                    f"{value.shape}."
                )
            size = value.size
            np.copyto(flat[offset:offset + size].reshape(shape), value)
            offset += size

    def write_current(self, network):
        """Publish the learner policy after the previous gradient update."""
        self._write_slot(0, network)

    def append_snapshot(self, network, metadata=None):
        """Append an admitted opponent, overwriting the oldest ring slot."""
        if self.max_pool_size == 0:
            return
        slot = self._next_pool_slot
        self._write_slot(slot + 1, network)
        if slot in self.pool_slots:
            self.pool_slots.remove(slot)
        self.pool_slots.append(slot)
        metadata = dict(metadata or {})
        snapshot_id = int(metadata.get("snapshot_id", self._next_snapshot_id))
        metadata.update({
            "snapshot_id": snapshot_id,
            "logical_order": len(self.pool_slots) - 1,
            "sampling_rule": "uniform_random",
        })
        self._pool_metadata[slot] = metadata
        self._next_snapshot_id = max(self._next_snapshot_id, snapshot_id + 1)
        self._next_pool_slot = (slot + 1) % self.max_pool_size

    def export_snapshots(self):
        """Copy snapshots in logical oldest-to-newest order."""
        snapshots = []
        for slot in self.pool_slots:
            flat = np.ndarray(
                (self.element_count,),
                dtype=self.dtype,
                buffer=self._segments[slot + 1].buf,
            )
            weights = {}
            offset = 0
            for name, shape in zip(self.weight_names, self.shapes):
                size = math.prod(shape)
                weights[name] = flat[offset:offset + size].reshape(shape).copy()
                offset += size
            snapshots.append(weights)
        return tuple(snapshots)

    def export_metadata(self):
        """Return JSON-safe metadata aligned with exported snapshots."""
        values = []
        for logical_order, slot in enumerate(self.pool_slots):
            metadata = dict(self._pool_metadata.get(slot, {}))
            metadata["logical_order"] = logical_order
            metadata.setdefault("sampling_rule", "uniform_random")
            values.append(metadata)
        return tuple(values)

    def restore(self, snapshots, metadata=None):
        """Replace the ring with serialized snapshots from a resume state."""
        snapshots = tuple(snapshots)
        metadata = (
            tuple({} for _snapshot in snapshots)
            if metadata is None
            else tuple(metadata)
        )
        if len(metadata) != len(snapshots):
            raise ValueError("Opponent-pool metadata count does not match snapshots.")
        if len(snapshots) > self.max_pool_size:
            raise ValueError(
                f"Resume state contains {len(snapshots)} opponent snapshots, "
                f"but max_pool_size is {self.max_pool_size}."
            )
        self.pool_slots = []
        self._pool_metadata = {}
        self._next_snapshot_id = 0
        self._next_pool_slot = 0
        for weights, snapshot_metadata in zip(snapshots, metadata):
            missing = [name for name in self.weight_names if name not in weights]
            if missing:
                raise ValueError(
                    "Resume opponent snapshot is missing policy weights: "
                    + ", ".join(missing)
                )
            self.append_snapshot(
                _SnapshotPolicy(weights),
                metadata=snapshot_metadata,
            )

    def close(self):
        """Release every shared segment, even after a failed training run."""
        if self._closed:
            return
        self._closed = True
        for segment in self._segments:
            try:
                segment.close()
            finally:
                try:
                    segment.unlink()
                except FileNotFoundError:
                    pass


class OpponentPool:
    """Own opponent admission, retention, persistence, and rollout eligibility."""

    ADMISSION_RULE = "every_updated_iteration"
    RETENTION_RULE = "fifo_ring"
    SAMPLING_RULE = "uniform_random"

    def __init__(self, bank, *, enabled, initial_network):
        self.bank = bank
        self.enabled = bool(enabled)
        if self.enabled:
            self.bank.append_snapshot(initial_network, metadata={
                "origin": "initial_policy",
                "introduced_at_rl_games": 0,
            })

    @property
    def size(self):
        return len(self.bank.pool_slots)

    def eligible_slots(self):
        """Return pool slots eligible for the current uniform matchmaking rule."""
        return tuple(self.bank.pool_slots) if self.enabled else ()

    def consider_updated_policy(self, network, *, completed_games, has_samples):
        """Apply the current admission rule to one freshly updated learner."""
        if not self.enabled or not has_samples:
            return False
        self.bank.append_snapshot(network, metadata={
            "origin": "training_update",
            "introduced_at_rl_games": int(completed_games),
        })
        return True

    def restore(self, snapshots, metadata=None):
        """Restore exact pool contents without changing selection policy."""
        self.bank.restore(snapshots, metadata=metadata)

    def export_snapshots(self):
        return self.bank.export_snapshots()

    def export_metadata(self):
        return self.bank.export_metadata()

    def manifest(self):
        """Return the fixed current policy for run/checkpoint provenance."""
        return {
            "enabled": self.enabled,
            "maximum_size": self.bank.max_pool_size,
            "admission_rule": self.ADMISSION_RULE,
            "retention_rule": self.RETENTION_RULE,
            "sampling_rule": self.SAMPLING_RULE,
        }
