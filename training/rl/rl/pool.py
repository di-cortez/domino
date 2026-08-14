"""Logical RL opponents and physical shared-memory policy storage.

``OpponentPool`` owns durable opponent identity, bucket membership, admission,
and retention. ``SharedPolicyBank`` only owns reusable physical storage.  A
bank slot is deliberately absent from :class:`OpponentRecord`: exact resume
reconstructs the runtime mapping instead of mistaking a reusable address for
an opponent identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from multiprocessing import shared_memory

import numpy as np


POOL_SCHEMA_VERSION = 2
POOL_POLICY_VERSION = 3
K_RECENT = 200
MEDIUM_TERM_CAPACITY = 200
MEDIUM_TERM_INTERVAL_ITERATIONS = 10
HEURISTIC_OPPONENT_ID = "heuristic:strategic_v1"
HEURISTIC_KIND = "heuristic"
RANDOM_OPPONENT_ID = "random:uniform_v1"
RANDOM_KIND = "random"
SNAPSHOT_KIND = "policy_snapshot"


@dataclass(frozen=True)
class BucketSpecification:
    """Internal immutable definition of one logical opponent bucket."""

    name: str
    capacity: int | None
    admission_rule: str
    retention_rule: str
    neural: bool
    admission_interval_iterations: int | None = None


BUCKET_REGISTRY = (
    BucketSpecification(
        name="heuristic",
        capacity=1,
        admission_rule="fixed_programmatic_opponent",
        retention_rule="fixed",
        neural=False,
    ),
    BucketSpecification(
        name="random",
        capacity=1,
        admission_rule="fixed_programmatic_opponent",
        retention_rule="fixed",
        neural=False,
    ),
    BucketSpecification(
        name="recent",
        capacity=K_RECENT,
        admission_rule="every_updated_iteration",
        retention_rule="fifo",
        neural=True,
    ),
    BucketSpecification(
        name="medium_term",
        capacity=MEDIUM_TERM_CAPACITY,
        admission_rule="successful_updates_at_fixed_iteration_interval",
        retention_rule="fifo",
        neural=True,
        admission_interval_iterations=MEDIUM_TERM_INTERVAL_ITERATIONS,
    ),
)
BUCKET_SPECIFICATIONS = {item.name: item for item in BUCKET_REGISTRY}
DEFAULT_OPPONENT_BUCKETS = ("heuristic", "recent")


def canonicalize_bucket_names(value):
    """Validate names and return an immutable tuple in registry order."""
    if isinstance(value, str):
        names = tuple(part.strip() for part in value.split(","))
    else:
        names = tuple(str(part).strip() for part in value)
    if not names or any(not name for name in names):
        raise ValueError("opponent_buckets must select at least one bucket")
    if len(set(names)) != len(names):
        raise ValueError("opponent_buckets cannot contain duplicate names")
    unknown = sorted(set(names) - set(BUCKET_SPECIFICATIONS))
    if unknown:
        raise ValueError(
            "Unknown opponent bucket(s): " + ", ".join(unknown)
        )
    selected = set(names)
    return tuple(
        specification.name
        for specification in BUCKET_REGISTRY
        if specification.name in selected
    )


def unique_neural_capacity(bucket_names):
    """Return the maximum unique active neural policies for selected buckets."""
    names = canonicalize_bucket_names(bucket_names)
    return sum(
        int(BUCKET_SPECIFICATIONS[name].capacity or 0)
        for name in names
        if BUCKET_SPECIFICATIONS[name].neural
    )


def pool_policy_manifest(bucket_names=DEFAULT_OPPONENT_BUCKETS):
    """Return the versioned internal pool policy used for provenance/resume."""
    names = canonicalize_bucket_names(bucket_names)
    return {
        "schema_version": POOL_SCHEMA_VERSION,
        "policy_version": POOL_POLICY_VERSION,
        "selected_buckets": list(names),
        "bucket_registry_order": [item.name for item in BUCKET_REGISTRY],
        "bucket_definitions": {
            name: {
                "capacity": BUCKET_SPECIFICATIONS[name].capacity,
                "admission_rule": BUCKET_SPECIFICATIONS[name].admission_rule,
                "retention_rule": BUCKET_SPECIFICATIONS[name].retention_rule,
                "neural": BUCKET_SPECIFICATIONS[name].neural,
                "admission_interval_iterations": (
                    BUCKET_SPECIFICATIONS[name].admission_interval_iterations
                ),
            }
            for name in names
        },
    }


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
    """Own one learner slot and a fixed number of reusable opponent slots."""

    def __init__(self, network, opponent_capacity):
        opponent_capacity = int(opponent_capacity)
        if opponent_capacity < 0:
            raise ValueError("opponent_capacity must be non-negative")
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
        self.opponent_capacity = opponent_capacity
        self._segments = []
        self._descriptors = []
        self._free_slots = list(range(opponent_capacity))
        self._allocated_slots = set()
        self._closed = False
        try:
            for _slot in range(1 + opponent_capacity):
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
        self.write_current(network)

    @property
    def current_descriptor(self):
        return self._descriptors[0]

    @property
    def opponent_descriptors(self):
        return tuple(self._descriptors[1:])

    @property
    def allocated_bytes(self):
        return len(self._segments) * self.element_count * self.dtype.itemsize

    @property
    def allocated_opponent_count(self):
        return len(self._allocated_slots)

    def descriptor(self, slot):
        self._validate_opponent_slot(slot)
        return self._descriptors[int(slot) + 1]

    def allocate_slot(self):
        """Allocate the lowest free physical slot deterministically."""
        if not self._free_slots:
            raise RuntimeError("Shared policy bank has no free opponent slot")
        slot = self._free_slots.pop(0)
        self._allocated_slots.add(slot)
        return slot

    def release_slot(self, slot):
        """Release a physical slot after logical retention removes its owner."""
        self._validate_opponent_slot(slot)
        slot = int(slot)
        if slot not in self._allocated_slots:
            raise ValueError(f"Shared policy slot {slot} is not allocated")
        self._allocated_slots.remove(slot)
        self._free_slots.append(slot)
        self._free_slots.sort()

    def _validate_opponent_slot(self, slot):
        slot = int(slot)
        if not 0 <= slot < self.opponent_capacity:
            raise ValueError(f"Invalid shared policy slot: {slot}")

    def _write_segment(self, segment_index, network):
        flat = np.ndarray(
            (self.element_count,),
            dtype=self.dtype,
            buffer=self._segments[segment_index].buf,
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
        """Publish the current frozen learner policy."""
        self._write_segment(0, network)

    def write_policy(self, slot, network):
        """Write one frozen opponent into an already allocated slot."""
        self._validate_opponent_slot(slot)
        if int(slot) not in self._allocated_slots:
            raise ValueError(f"Shared policy slot {slot} is not allocated")
        self._write_segment(int(slot) + 1, network)

    def read_policy(self, slot):
        """Copy one opponent policy from shared memory to ordinary arrays."""
        self._validate_opponent_slot(slot)
        if int(slot) not in self._allocated_slots:
            raise ValueError(f"Shared policy slot {slot} is not allocated")
        flat = np.ndarray(
            (self.element_count,),
            dtype=self.dtype,
            buffer=self._segments[int(slot) + 1].buf,
        )
        weights = {}
        offset = 0
        for name, shape in zip(self.weight_names, self.shapes):
            size = math.prod(shape)
            weights[name] = flat[offset:offset + size].reshape(shape).copy()
            offset += size
        return weights

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


@dataclass(frozen=True)
class OpponentRecord:
    """Durable logical identity of one programmatic or neural opponent."""

    opponent_id: str
    kind: str
    checkpoint_id: str | None
    introduced_iteration: int
    introduced_at_rl_games: int
    origin: str


@dataclass
class OpponentBucket:
    """Ordered membership and fixed retention policy for one bucket."""

    name: str
    member_ids: list[str]
    capacity: int | None
    admission_rule: str
    retention_rule: str
    admission_interval_iterations: int | None


class OpponentPool:
    """Own logical records, bucket memberships, retention, and bank mapping."""

    def __init__(self, bank, *, selected_buckets, initial_network=None):
        self.bank = bank
        self.selected_buckets = canonicalize_bucket_names(selected_buckets)
        self.opponents_by_id = {}
        self.bank_slot_by_opponent_id = {}
        self.opponent_id_by_checkpoint_id = {}
        self.buckets_by_name = {}
        self._next_snapshot_id = 0
        self.last_completed_rl_iteration = 0
        self._initial_snapshot_record = None
        self._admission_counts = {name: 0 for name in self.selected_buckets}
        self._eviction_counts = {name: 0 for name in self.selected_buckets}
        for name in self.selected_buckets:
            specification = BUCKET_SPECIFICATIONS[name]
            self.buckets_by_name[name] = OpponentBucket(
                name=name,
                member_ids=[],
                capacity=specification.capacity,
                admission_rule=specification.admission_rule,
                retention_rule=specification.retention_rule,
                admission_interval_iterations=(
                    specification.admission_interval_iterations
                ),
            )
        if "heuristic" in self.buckets_by_name:
            record = OpponentRecord(
                opponent_id=HEURISTIC_OPPONENT_ID,
                kind=HEURISTIC_KIND,
                checkpoint_id=None,
                introduced_iteration=0,
                introduced_at_rl_games=0,
                origin="fixed_programmatic_opponent",
            )
            self.opponents_by_id[record.opponent_id] = record
            self.buckets_by_name["heuristic"].member_ids.append(
                record.opponent_id
            )
        if "random" in self.buckets_by_name:
            record = OpponentRecord(
                opponent_id=RANDOM_OPPONENT_ID,
                kind=RANDOM_KIND,
                checkpoint_id=None,
                introduced_iteration=0,
                introduced_at_rl_games=0,
                origin="fixed_programmatic_opponent",
            )
            self.opponents_by_id[record.opponent_id] = record
            self.buckets_by_name["random"].member_ids.append(
                record.opponent_id
            )
        if initial_network is not None:
            self._initial_snapshot_record = self.add_initial_snapshot(
                initial_network,
                iteration=0,
                completed_games=0,
            )

    @property
    def size(self):
        """Return the number of active logical opponents."""
        return len(self.opponents_by_id)

    @property
    def unique_neural_opponent_count(self):
        return len(self.bank_slot_by_opponent_id)

    def bucket_sizes(self):
        return {
            name: len(self.buckets_by_name[name].member_ids)
            for name in self.selected_buckets
        }

    def bucket_members(self, name):
        try:
            member_ids = self.buckets_by_name[name].member_ids
        except KeyError as exc:
            raise KeyError(f"Opponent bucket {name!r} is not active") from exc
        return tuple(member_ids)

    def active_opponents(self):
        return tuple(
            self.opponents_by_id[opponent_id]
            for opponent_id in sorted(self.opponents_by_id)
        )

    def opponent(self, opponent_id):
        return self.opponents_by_id[opponent_id]

    def bank_slot(self, opponent_id):
        return self.bank_slot_by_opponent_id.get(opponent_id)

    @property
    def initial_snapshot_record(self):
        """Return the iteration-zero identity created for a fresh run."""
        return self._initial_snapshot_record

    def _new_snapshot_record(self, *, iteration, completed_games, origin):
        snapshot_number = self._next_snapshot_id
        self._next_snapshot_id += 1
        suffix = f"{snapshot_number:010d}"
        return OpponentRecord(
            opponent_id=f"snapshot:{suffix}",
            kind=SNAPSHOT_KIND,
            checkpoint_id=f"checkpoint:{suffix}",
            introduced_iteration=int(iteration),
            introduced_at_rl_games=int(completed_games),
            origin=origin,
        )

    def _make_room_for_memberships(self, bucket_names):
        for bucket_name in bucket_names:
            bucket = self.buckets_by_name[bucket_name]
            if bucket.capacity is None:
                continue
            while len(bucket.member_ids) >= bucket.capacity:
                removed_id = bucket.member_ids.pop(0)
                self._eviction_counts[bucket_name] += 1
                self._remove_if_unreferenced(removed_id)

    def _add_snapshot(
        self,
        network,
        *,
        iteration,
        completed_games,
        origin,
        bucket_names,
    ):
        bucket_names = tuple(bucket_names)
        record = self._new_snapshot_record(
            iteration=iteration,
            completed_games=completed_games,
            origin=origin,
        )
        if not bucket_names:
            return record
        self._make_room_for_memberships(bucket_names)
        slot = self.bank.allocate_slot()
        try:
            self.bank.write_policy(slot, network)
        except BaseException:
            self.bank.release_slot(slot)
            raise
        self.opponents_by_id[record.opponent_id] = record
        self.bank_slot_by_opponent_id[record.opponent_id] = slot
        self.opponent_id_by_checkpoint_id[record.checkpoint_id] = record.opponent_id
        for bucket_name in bucket_names:
            self.buckets_by_name[bucket_name].member_ids.append(record.opponent_id)
            self._admission_counts[bucket_name] += 1
            self._apply_retention(bucket_name)
        return record

    def add_initial_snapshot(self, network, iteration=0, completed_games=0):
        """Freeze and admit the initial learner before the first iteration."""
        bucket_names = tuple(
            name
            for name in ("recent", "medium_term")
            if name in self.buckets_by_name
        )
        if any(self.buckets_by_name[name].member_ids for name in bucket_names):
            raise ValueError("A neural bucket already has an initial snapshot")
        return self._add_snapshot(
            network,
            iteration=iteration,
            completed_games=completed_games,
            origin="initial_policy",
            bucket_names=bucket_names,
        )

    def consider_updated_policy(
        self,
        network,
        *,
        iteration,
        completed_games,
        has_samples,
    ):
        """Admit one successful learner update according to bucket policies."""
        iteration = int(iteration)
        if iteration <= self.last_completed_rl_iteration:
            raise ValueError(
                "Completed RL iterations must advance monotonically: "
                f"{iteration} <= {self.last_completed_rl_iteration}"
            )
        self.last_completed_rl_iteration = iteration
        if not has_samples:
            return None
        bucket_names = []
        if "recent" in self.buckets_by_name:
            bucket_names.append("recent")
        if (
            "medium_term" in self.buckets_by_name
            and iteration % MEDIUM_TERM_INTERVAL_ITERATIONS == 0
        ):
            bucket_names.append("medium_term")
        archive_milestone = iteration % MEDIUM_TERM_INTERVAL_ITERATIONS == 0
        if not bucket_names and not archive_milestone:
            return None
        return self._add_snapshot(
            network,
            iteration=iteration,
            completed_games=completed_games,
            origin=(
                "training_update"
                if bucket_names
                else "training_update_archive_only"
            ),
            bucket_names=bucket_names,
        )

    def _apply_retention(self, bucket_name):
        bucket = self.buckets_by_name[bucket_name]
        if bucket.capacity is None:
            return
        while len(bucket.member_ids) > bucket.capacity:
            removed_id = bucket.member_ids.pop(0)
            self._eviction_counts[bucket_name] += 1
            self._remove_if_unreferenced(removed_id)

    def _remove_if_unreferenced(self, opponent_id):
        if any(
            opponent_id in bucket.member_ids
            for bucket in self.buckets_by_name.values()
        ):
            return
        record = self.opponents_by_id.pop(opponent_id)
        if record.checkpoint_id is not None:
            self.opponent_id_by_checkpoint_id.pop(record.checkpoint_id, None)
        slot = self.bank_slot_by_opponent_id.pop(opponent_id, None)
        if slot is not None:
            self.bank.release_slot(slot)

    def add_membership(self, bucket_name, opponent_id):
        """Add a deduplicated logical reference for future bucket policies."""
        bucket = self.buckets_by_name[bucket_name]
        if opponent_id not in self.opponents_by_id:
            raise KeyError(f"Unknown opponent: {opponent_id}")
        if opponent_id not in bucket.member_ids:
            self._make_room_for_memberships((bucket_name,))
            bucket.member_ids.append(opponent_id)
            self._admission_counts[bucket_name] += 1
            self._apply_retention(bucket_name)

    def checkpoint_ids_for_bucket(self, bucket_name):
        """Return checkpoint identities referenced by one active bucket."""
        if bucket_name not in self.buckets_by_name:
            return ()
        return tuple(
            self.opponents_by_id[opponent_id].checkpoint_id
            for opponent_id in self.buckets_by_name[bucket_name].member_ids
            if self.opponents_by_id[opponent_id].checkpoint_id is not None
        )

    def observability(self, *, games_per_iteration=None):
        """Return compact mutable pool state for logs and final manifests."""
        bucket_state = {}
        for name in self.selected_buckets:
            bucket = self.buckets_by_name[name]
            neural_records = [
                self.opponents_by_id[opponent_id]
                for opponent_id in bucket.member_ids
                if self.opponents_by_id[opponent_id].kind == SNAPSHOT_KIND
            ]
            oldest = neural_records[0] if neural_records else None
            newest = neural_records[-1] if neural_records else None
            exact_span_games = (
                None
                if oldest is None or newest is None
                else newest.introduced_at_rl_games
                - oldest.introduced_at_rl_games
            )
            nominal_span_games = None
            if (
                games_per_iteration is not None
                and bucket.admission_interval_iterations is not None
                and bucket.capacity is not None
            ):
                nominal_span_games = (
                    int(games_per_iteration)
                    * bucket.admission_interval_iterations
                    * bucket.capacity
                )
            bucket_state[name] = {
                "membership_count": len(bucket.member_ids),
                "capacity": bucket.capacity,
                "admission_interval_iterations": (
                    bucket.admission_interval_iterations
                ),
                "oldest_iteration": (
                    None if oldest is None else oldest.introduced_iteration
                ),
                "newest_iteration": (
                    None if newest is None else newest.introduced_iteration
                ),
                "oldest_completed_games": (
                    None if oldest is None else oldest.introduced_at_rl_games
                ),
                "newest_completed_games": (
                    None if newest is None else newest.introduced_at_rl_games
                ),
                "exact_historical_span_games": exact_span_games,
                "nominal_historical_span_games": nominal_span_games,
                "admissions": self._admission_counts[name],
                "fifo_evictions": self._eviction_counts[name],
            }
        recent = set(
            self.buckets_by_name["recent"].member_ids
            if "recent" in self.buckets_by_name
            else ()
        )
        medium_term = set(
            self.buckets_by_name["medium_term"].member_ids
            if "medium_term" in self.buckets_by_name
            else ()
        )
        return {
            "last_completed_rl_iteration": self.last_completed_rl_iteration,
            "membership_count": sum(
                len(bucket.member_ids) for bucket in self.buckets_by_name.values()
            ),
            "unique_opponent_count": self.size,
            "unique_neural_opponent_count": self.unique_neural_opponent_count,
            "recent_medium_term_overlap_count": len(recent & medium_term),
            "buckets": bucket_state,
        }

    def export_weights(self):
        """Return one copied weight set per unique active neural opponent."""
        return {
            opponent_id: self.bank.read_policy(slot)
            for opponent_id, slot in sorted(self.bank_slot_by_opponent_id.items())
        }

    def export_state(self):
        """Return JSON-safe logical state with no durable physical slot IDs."""
        neural_ids = sorted(self.bank_slot_by_opponent_id)
        return {
            "schema_version": POOL_SCHEMA_VERSION,
            "policy_manifest": self.manifest(),
            "selected_buckets": list(self.selected_buckets),
            "next_snapshot_id": int(self._next_snapshot_id),
            "last_completed_rl_iteration": int(
                self.last_completed_rl_iteration
            ),
            "lifecycle_counters": {
                "admissions": dict(self._admission_counts),
                "fifo_evictions": dict(self._eviction_counts),
            },
            "opponents": [
                asdict(self.opponents_by_id[opponent_id])
                for opponent_id in sorted(self.opponents_by_id)
            ],
            "buckets": {
                name: {
                    "member_ids": list(self.buckets_by_name[name].member_ids),
                    "capacity": self.buckets_by_name[name].capacity,
                    "admission_rule": self.buckets_by_name[name].admission_rule,
                    "retention_rule": self.buckets_by_name[name].retention_rule,
                    "admission_interval_iterations": (
                        self.buckets_by_name[name].admission_interval_iterations
                    ),
                }
                for name in self.selected_buckets
            },
            "weight_serialization": [
                {"index": index, "opponent_id": opponent_id}
                for index, opponent_id in enumerate(neural_ids)
            ],
        }

    def restore_state(self, state, weights):
        """Restore identities/memberships and allocate fresh runtime bank slots."""
        if int(state.get("schema_version", -1)) != POOL_SCHEMA_VERSION:
            raise ValueError("Unsupported opponent-pool state version")
        selected = canonicalize_bucket_names(state.get("selected_buckets", ()))
        if selected != self.selected_buckets:
            raise ValueError(
                "Resume opponent buckets do not match the active pool: "
                f"{selected!r} != {self.selected_buckets!r}"
            )
        if state.get("policy_manifest") != self.manifest():
            raise ValueError("Resume opponent-pool policy manifest changed")

        for slot in tuple(self.bank_slot_by_opponent_id.values()):
            self.bank.release_slot(slot)
        self.opponents_by_id = {}
        self.bank_slot_by_opponent_id = {}
        self.opponent_id_by_checkpoint_id = {}
        records = [OpponentRecord(**value) for value in state.get("opponents", ())]
        for record in records:
            if record.opponent_id in self.opponents_by_id:
                raise ValueError(f"Duplicate opponent ID: {record.opponent_id}")
            self.opponents_by_id[record.opponent_id] = record
            if record.checkpoint_id is not None:
                self.opponent_id_by_checkpoint_id[record.checkpoint_id] = (
                    record.opponent_id
                )

        for name in self.selected_buckets:
            saved = state.get("buckets", {}).get(name)
            if saved is None:
                raise ValueError(f"Resume state is missing bucket {name!r}")
            bucket = self.buckets_by_name[name]
            expected = BUCKET_SPECIFICATIONS[name]
            if (
                saved.get("capacity") != expected.capacity
                or saved.get("admission_rule") != expected.admission_rule
                or saved.get("retention_rule") != expected.retention_rule
                or saved.get("admission_interval_iterations")
                != expected.admission_interval_iterations
            ):
                raise ValueError(f"Resume policy changed for bucket {name!r}")
            bucket.member_ids = list(saved.get("member_ids", ()))
            if name == "medium_term" and not bucket.member_ids:
                raise ValueError(
                    "Resume selected medium_term but restored it empty; the "
                    "iteration-zero milestone is required"
                )
            if any(item not in self.opponents_by_id for item in bucket.member_ids):
                raise ValueError(f"Bucket {name!r} references an unknown opponent")
            if bucket.capacity is not None and len(bucket.member_ids) > bucket.capacity:
                raise ValueError(f"Bucket {name!r} exceeds its capacity")

        expected_neural = {
            record.opponent_id
            for record in records
            if record.kind == SNAPSHOT_KIND
        }
        if set(weights) != expected_neural:
            raise ValueError(
                "Resume opponent weights do not match active neural identities"
            )
        try:
            for opponent_id in sorted(expected_neural):
                slot = self.bank.allocate_slot()
                self.bank.write_policy(slot, _SnapshotPolicy(weights[opponent_id]))
                self.bank_slot_by_opponent_id[opponent_id] = slot
        except BaseException:
            for slot in tuple(self.bank_slot_by_opponent_id.values()):
                self.bank.release_slot(slot)
            self.bank_slot_by_opponent_id = {}
            raise
        self._next_snapshot_id = int(state.get("next_snapshot_id", 0))
        self.last_completed_rl_iteration = int(
            state.get("last_completed_rl_iteration", -1)
        )
        if self.last_completed_rl_iteration < 0:
            raise ValueError(
                "Resume opponent-pool state is missing the completed iteration"
            )
        counters = state.get("lifecycle_counters", {})
        admissions = counters.get("admissions", {})
        evictions = counters.get("fifo_evictions", {})
        if set(admissions) != set(self.selected_buckets) or set(evictions) != set(
            self.selected_buckets
        ):
            raise ValueError("Resume opponent-pool lifecycle counters are incomplete")
        self._admission_counts = {
            name: int(admissions[name]) for name in self.selected_buckets
        }
        self._eviction_counts = {
            name: int(evictions[name]) for name in self.selected_buckets
        }

    def manifest(self):
        """Return fixed policy details, not mutable pool contents."""
        return pool_policy_manifest(self.selected_buckets)
