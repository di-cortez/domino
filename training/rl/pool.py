"""Logical RL opponents and physical shared-memory policy storage.

``OpponentPool`` owns durable opponent identity, bucket membership, admission,
and retention. ``SharedPolicyBank`` only owns reusable physical storage.  A
bank slot is deliberately absent from :class:`OpponentRecord`: exact resume
reconstructs the runtime mapping instead of mistaking a reusable address for
an opponent identity.
"""

from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass
import math
from multiprocessing import shared_memory

import numpy as np


POOL_SCHEMA_VERSION = 3
POOL_POLICY_VERSION = 4
K_RECENT = 200
MEDIUM_TERM_CAPACITY = 200
MEDIUM_TERM_INTERVAL_ITERATIONS = 10
HISTORICAL_UNIFORM_CAPACITY = 200
# The neural buckets cover disjoint chronological regions. ``recent`` owns the
# newest ``K_RECENT`` snapshots, ``medium_term`` the archive milestones behind
# that band, and ``historical_uniform`` everything older than both.
RECENT_BAND_WIDTH_ITERATIONS = K_RECENT
MEDIUM_TERM_BAND_WIDTH_ITERATIONS = (
    MEDIUM_TERM_CAPACITY * MEDIUM_TERM_INTERVAL_ITERATIONS
)
HISTORICAL_MINIMUM_AGE_ITERATIONS = (
    RECENT_BAND_WIDTH_ITERATIONS + MEDIUM_TERM_BAND_WIDTH_ITERATIONS
)
MEDIUM_TERM_RETENTION_RULE = "delayed_fifo_archive_window"
HISTORICAL_UNIFORM_RETENTION_RULE = "deterministic_uniform_historical_rebalance"
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
        admission_rule="latest_archive_milestones_older_than_recent_band",
        retention_rule=MEDIUM_TERM_RETENTION_RULE,
        neural=True,
        admission_interval_iterations=MEDIUM_TERM_INTERVAL_ITERATIONS,
    ),
    BucketSpecification(
        name="historical_uniform",
        capacity=HISTORICAL_UNIFORM_CAPACITY,
        admission_rule="archive_records_older_than_recent_and_medium_bands",
        retention_rule=HISTORICAL_UNIFORM_RETENTION_RULE,
        neural=True,
        admission_interval_iterations=MEDIUM_TERM_INTERVAL_ITERATIONS,
    ),
)
BUCKET_SPECIFICATIONS = {item.name: item for item in BUCKET_REGISTRY}
DEFAULT_OPPONENT_BUCKETS = ("heuristic", "recent")
# The delayed archive-backed bands are genuinely empty during warm-up, so a
# selection made only of them could never play its own first iteration.
BOOTSTRAP_CAPABLE_BUCKETS = ("heuristic", "random", "recent")
NEURAL_BUCKET_NAMES = tuple(
    item.name for item in BUCKET_REGISTRY if item.neural
)
ARCHIVE_BACKED_BUCKET_NAMES = ("medium_term", "historical_uniform")


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
        "band_policy": {
            "recent_band_width_iterations": RECENT_BAND_WIDTH_ITERATIONS,
            "medium_term_band_width_iterations": (
                MEDIUM_TERM_BAND_WIDTH_ITERATIONS
            ),
            "historical_minimum_age_iterations": (
                HISTORICAL_MINIMUM_AGE_ITERATIONS
            ),
            "band_eligibility_coordinate": "completed_iteration",
            "historical_spacing_coordinate": "completed_rl_games",
            "historical_selection_tie_breaking": (
                "absolute_target_error_then_older_iteration_then_checkpoint_id"
            ),
        },
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


def medium_term_cutoff_iteration(completed_iteration):
    """Return the newest absolute iteration still behind the ``recent`` band."""
    return int(completed_iteration) - RECENT_BAND_WIDTH_ITERATIONS


def historical_uniform_cutoff_iteration(completed_iteration):
    """Return the newest absolute iteration older than recent and medium."""
    return int(completed_iteration) - HISTORICAL_MINIMUM_AGE_ITERATIONS


def sorted_archive_records(archive_records):
    """Order archive metadata deterministically for every band computation.

    The band selectors accept any object exposing ``completed_iteration``,
    ``completed_rl_games``, ``opponent_id``, and ``checkpoint_id``. The archive
    owns that record type; this module only owns the selection mathematics.
    """
    records = tuple(archive_records)
    checkpoint_ids = [record.checkpoint_id for record in records]
    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise ValueError("Archive metadata contains duplicate checkpoint IDs")
    return tuple(sorted(
        records,
        key=lambda record: (
            int(record.completed_iteration),
            record.checkpoint_id,
        ),
    ))


def _eligible_archive_records(archive_records, cutoff, excluded_opponent_ids):
    excluded = set(excluded_opponent_ids)
    return tuple(
        record
        for record in sorted_archive_records(archive_records)
        if int(record.completed_iteration) <= cutoff
        and record.opponent_id not in excluded
    )


def select_medium_term_records(
    archive_records,
    *,
    completed_iteration,
    excluded_opponent_ids=(),
):
    """Return the newest archive milestones strictly behind the recent band.

    Identities still held by ``recent`` are removed before the newest-first cut
    instead of after it, so an iteration without trainable samples widens the
    search backward rather than shrinking the band or creating an overlap.
    """
    eligible = _eligible_archive_records(
        archive_records,
        medium_term_cutoff_iteration(completed_iteration),
        excluded_opponent_ids,
    )
    if not eligible:
        return ()
    return eligible[-MEDIUM_TERM_CAPACITY:]


def select_medium_term_staging_records(
    archive_records,
    *,
    completed_iteration,
    pending_opponent_ids=(),
):
    """Return milestones that will enter ``medium_term`` but have not yet.

    These files are not band members yet, so nothing else would keep them
    alive. Without an explicit pin a tight byte limit could thin a milestone
    during the exact window between its archive write and its delayed
    admission.

    The nominal window is one recent-band width. Iterations without trainable
    decisions stretch that window in absolute iterations, so identities still
    held by ``recent`` are staged too: they are older than the cutoff yet
    barred from the band precisely because ``recent`` still owns them.
    """
    completed_iteration = int(completed_iteration)
    cutoff = medium_term_cutoff_iteration(completed_iteration)
    pending = set(pending_opponent_ids)
    return tuple(
        record
        for record in sorted_archive_records(archive_records)
        if int(record.completed_iteration) <= completed_iteration
        and (
            int(record.completed_iteration) > cutoff
            or record.opponent_id in pending
        )
    )


def _nearest_record_index(records, *, lower_index, upper_index, target, scale):
    """Return the feasible index closest to one integer grid target."""
    position = bisect.bisect_left(
        records,
        target,
        lower_index,
        upper_index + 1,
        key=lambda record: scale * int(record.completed_rl_games),
    )
    candidates = [
        index
        for index in (position - 1, position)
        if lower_index <= index <= upper_index
    ]
    return min(
        candidates,
        key=lambda index: (
            abs(scale * int(records[index].completed_rl_games) - target),
            int(records[index].completed_iteration),
            records[index].checkpoint_id,
        ),
    )


def select_historical_uniform_records(
    archive_records,
    *,
    completed_iteration,
    excluded_opponent_ids=(),
):
    """Return deterministic uniform representatives of the oldest history.

    Targets are spaced uniformly in completed-game coordinates rather than in
    record rank, so thinned regions widen their own gaps instead of borrowing
    resolution from a denser era. Comparisons stay integral to keep the result
    independent of platform floating-point behavior.
    """
    eligible = _eligible_archive_records(
        archive_records,
        historical_uniform_cutoff_iteration(completed_iteration),
        excluded_opponent_ids,
    )
    available = len(eligible)
    capacity = HISTORICAL_UNIFORM_CAPACITY
    if available <= capacity:
        return eligible
    scale = capacity - 1
    oldest_games = int(eligible[0].completed_rl_games)
    newest_games = int(eligible[-1].completed_rl_games)
    selected_indices = [0]
    for step in range(1, capacity - 1):
        target = (scale - step) * oldest_games + step * newest_games
        selected_indices.append(_nearest_record_index(
            eligible,
            lower_index=selected_indices[-1] + 1,
            upper_index=available - (capacity - step),
            target=target,
            scale=scale,
        ))
    selected_indices.append(available - 1)
    return tuple(eligible[index] for index in selected_indices)


def historical_uniform_selection_diagnostics(selected_records, *, eligible_records):
    """Report how closely one selection approximates ideal uniform spacing.

    The ideal grid is defined over the selected span in both the partial and
    the capped case, so the reported error always answers the same question:
    how uneven is the history this bucket actually represents. Once the
    archive has been thinned in the represented region the selection is only
    the closest available approximation, so that fact is reported alongside
    the error rather than left implicit.
    """
    records = tuple(selected_records)
    eligible = sorted_archive_records(eligible_records)
    games = [int(record.completed_rl_games) for record in records]
    gaps = [later - earlier for earlier, later in zip(games, games[1:])]
    diagnostics = {
        "selected_count": len(records),
        "capacity": HISTORICAL_UNIFORM_CAPACITY,
        "eligible_record_count": len(eligible),
        "archive_thinned_in_region": any(
            int(later.completed_iteration) - int(earlier.completed_iteration)
            > MEDIUM_TERM_INTERVAL_ITERATIONS
            for earlier, later in zip(eligible, eligible[1:])
        ),
        "oldest_completed_games": games[0] if games else None,
        "newest_completed_games": games[-1] if games else None,
        "span_games": (games[-1] - games[0]) if gaps else None,
        "ideal_gap_games": None,
        "minimum_gap_games": min(gaps) if gaps else None,
        "maximum_gap_games": max(gaps) if gaps else None,
        "mean_absolute_target_error_games": None,
        "maximum_absolute_target_error_games": None,
    }
    if not gaps:
        return diagnostics
    scale = len(records) - 1
    span = games[-1] - games[0]
    errors = [
        abs(scale * games[step] - ((scale - step) * games[0] + step * games[-1]))
        / scale
        for step in range(len(records))
    ]
    diagnostics["ideal_gap_games"] = span / scale
    diagnostics["mean_absolute_target_error_games"] = math.fsum(errors) / len(errors)
    diagnostics["maximum_absolute_target_error_games"] = max(errors)
    return diagnostics


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
        self._band_removal_counts = {name: 0 for name in self.selected_buckets}
        self._band_rebalance_counts = {name: 0 for name in self.selected_buckets}
        self._historical_diagnostics = None
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

    def available_bucket_names(self):
        """Return the configured buckets that currently hold at least one member.

        A configured bucket with no member is not an error during warm-up. It
        simply receives no games until the archive makes its band real.
        """
        return tuple(
            name
            for name in self.selected_buckets
            if self.buckets_by_name[name].member_ids
        )

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
        """Freeze and admit the initial learner before the first iteration.

        The delayed bands never receive a fresh policy directly. The baseline
        reaches them only once the archive makes it old enough, which is what
        keeps the neural bands chronologically disjoint.
        """
        bucket_names = tuple(
            name for name in ("recent",) if name in self.buckets_by_name
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
        # A milestone still needs a stable identity so the archive can store
        # it, but it joins no delayed band until its age qualifies.
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

    def archive_backed_bucket_names(self):
        """Return the selected buckets whose membership comes from the archive."""
        return tuple(
            name
            for name in ARCHIVE_BACKED_BUCKET_NAMES
            if name in self.buckets_by_name
        )

    def desired_archive_backed_memberships(
        self,
        archive_records,
        *,
        completed_iteration,
    ):
        """Return the target archive records for every delayed band.

        The bands are derived together and in registry order so the older band
        can subtract the newer one's identities. Chronology alone already
        separates them; the explicit subtraction is the invariant that survives
        iterations which produced no trainable decisions.
        """
        excluded = set(
            self.buckets_by_name["recent"].member_ids
            if "recent" in self.buckets_by_name
            else ()
        )
        targets = {}
        if "medium_term" in self.buckets_by_name:
            targets["medium_term"] = select_medium_term_records(
                archive_records,
                completed_iteration=completed_iteration,
                excluded_opponent_ids=excluded,
            )
            excluded |= {
                record.opponent_id for record in targets["medium_term"]
            }
        if "historical_uniform" in self.buckets_by_name:
            targets["historical_uniform"] = select_historical_uniform_records(
                archive_records,
                completed_iteration=completed_iteration,
                excluded_opponent_ids=excluded,
            )
        self._validate_band_targets(targets)
        return targets

    def _validate_band_targets(self, targets):
        memberships = {}
        if "recent" in self.buckets_by_name:
            memberships["recent"] = tuple(
                self.buckets_by_name["recent"].member_ids
            )
        for name, records in targets.items():
            opponent_ids = tuple(record.opponent_id for record in records)
            if len(set(opponent_ids)) != len(opponent_ids):
                raise ValueError(f"Band target for {name!r} repeats an identity")
            iterations = [int(record.completed_iteration) for record in records]
            if iterations != sorted(iterations):
                raise ValueError(f"Band target for {name!r} is not chronological")
            capacity = self.buckets_by_name[name].capacity
            if capacity is not None and len(opponent_ids) > capacity:
                raise ValueError(f"Band target for {name!r} exceeds its capacity")
            memberships[name] = opponent_ids
        names = tuple(memberships)
        for index, first in enumerate(names):
            for second in names[index + 1:]:
                overlap = set(memberships[first]) & set(memberships[second])
                if overlap:
                    raise ValueError(
                        f"Buckets {first!r} and {second!r} would share "
                        + ", ".join(sorted(overlap))
                    )

    def _validate_bank_compatible_weights(self, weights, checkpoint_id):
        expected = dict(zip(self.bank.weight_names, self.bank.shapes))
        if set(weights) != set(expected):
            raise ValueError(
                f"Archived checkpoint {checkpoint_id!r} does not carry the "
                "active policy weight names"
            )
        for name, shape in expected.items():
            if tuple(np.shape(weights[name])) != tuple(shape):
                raise ValueError(
                    f"Archived checkpoint {checkpoint_id!r} weight {name!r} "
                    f"has shape {tuple(np.shape(weights[name]))}, expected "
                    f"{tuple(shape)}"
                )

    def reconcile_archive_backed_buckets(
        self,
        archive_records,
        *,
        completed_iteration,
        load_weights,
    ):
        """Rebuild both delayed bands as one logical transaction.

        Every incoming weight set is loaded and validated into ordinary host
        memory before any logical state changes, and slots are released only
        after the complete target union is known. A checkpoint leaving
        ``medium_term`` in the same refresh that promotes it to
        ``historical_uniform`` therefore keeps its identity, slot, and
        accumulated performance evidence.
        """
        bucket_names = self.archive_backed_bucket_names()
        if not bucket_names:
            return {}
        targets = self.desired_archive_backed_memberships(
            archive_records,
            completed_iteration=completed_iteration,
        )
        if "historical_uniform" in targets:
            self._historical_diagnostics = (
                historical_uniform_selection_diagnostics(
                    targets["historical_uniform"],
                    eligible_records=_eligible_archive_records(
                        archive_records,
                        historical_uniform_cutoff_iteration(completed_iteration),
                        (),
                    ),
                )
            )
        previous = {
            name: tuple(self.buckets_by_name[name].member_ids)
            for name in bucket_names
        }
        previous_union = {
            opponent_id
            for members in previous.values()
            for opponent_id in members
        }
        target_union = {
            record.opponent_id
            for records in targets.values()
            for record in records
        }
        incoming = []
        seen = set()
        for name in bucket_names:
            for record in targets[name]:
                if record.opponent_id in self.opponents_by_id:
                    continue
                if record.opponent_id in seen:
                    continue
                seen.add(record.opponent_id)
                incoming.append(record)
        released = sorted(previous_union - target_union)

        loaded = {}
        for record in incoming:
            weights = load_weights(record.checkpoint_id)
            self._validate_bank_compatible_weights(weights, record.checkpoint_id)
            loaded[record.opponent_id] = weights
        reclaimable = sum(
            1
            for opponent_id in released
            if self.bank_slot_by_opponent_id.get(opponent_id) is not None
            and not self._referenced_outside(opponent_id, bucket_names)
        )
        free_slots = (
            self.bank.opponent_capacity - self.bank.allocated_opponent_count
        )
        if len(incoming) > free_slots + reclaimable:
            raise RuntimeError(
                "Archive-backed rebalance needs "
                f"{len(incoming)} slots but only {free_slots + reclaimable} "
                "can be made free"
            )

        for name in bucket_names:
            self.buckets_by_name[name].member_ids = []
        for opponent_id in released:
            self._remove_if_unreferenced(opponent_id)
        for record in incoming:
            slot = self.bank.allocate_slot()
            self.bank.write_policy(
                slot,
                _SnapshotPolicy(loaded[record.opponent_id]),
            )
            identity = OpponentRecord(
                opponent_id=record.opponent_id,
                kind=SNAPSHOT_KIND,
                checkpoint_id=record.checkpoint_id,
                introduced_iteration=int(record.completed_iteration),
                introduced_at_rl_games=int(record.completed_rl_games),
                origin="archive_rehydration",
            )
            self.opponents_by_id[identity.opponent_id] = identity
            self.bank_slot_by_opponent_id[identity.opponent_id] = slot
            self.opponent_id_by_checkpoint_id[identity.checkpoint_id] = (
                identity.opponent_id
            )
        summary = {}
        for name in bucket_names:
            members = tuple(record.opponent_id for record in targets[name])
            self.buckets_by_name[name].member_ids = list(members)
            added = tuple(
                opponent_id
                for opponent_id in members
                if opponent_id not in previous[name]
            )
            removed = tuple(
                opponent_id
                for opponent_id in previous[name]
                if opponent_id not in members
            )
            self._admission_counts[name] += len(added)
            self._band_removal_counts[name] += len(removed)
            if added or removed:
                self._band_rebalance_counts[name] += 1
            summary[name] = {
                "membership_count": len(members),
                "added": len(added),
                "removed": len(removed),
                "rehydrated": sum(
                    1 for record in incoming if record.opponent_id in members
                ),
            }
        return summary

    def _referenced_outside(self, opponent_id, bucket_names):
        excluded = set(bucket_names)
        return any(
            opponent_id in bucket.member_ids
            for name, bucket in self.buckets_by_name.items()
            if name not in excluded
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
            # A uniform historical band spans the whole old history, so its
            # width grows with the run and no fixed-cadence formula applies.
            nominal_span_games = None
            if (
                games_per_iteration is not None
                and bucket.admission_interval_iterations is not None
                and bucket.capacity is not None
                and bucket.retention_rule != HISTORICAL_UNIFORM_RETENTION_RULE
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
                "band_removals": self._band_removal_counts[name],
                "band_rebalances": self._band_rebalance_counts[name],
                "band_cutoff_iteration": self._band_cutoff_iteration(name),
                "oldest_age_iterations": (
                    None
                    if oldest is None
                    else self.last_completed_rl_iteration
                    - oldest.introduced_iteration
                ),
                "newest_age_iterations": (
                    None
                    if newest is None
                    else self.last_completed_rl_iteration
                    - newest.introduced_iteration
                ),
            }
            if name == "historical_uniform":
                bucket_state[name]["selection_diagnostics"] = (
                    self._historical_diagnostics
                )
        overlaps = self.bucket_overlap_counts()
        return {
            "last_completed_rl_iteration": self.last_completed_rl_iteration,
            "membership_count": sum(
                len(bucket.member_ids) for bucket in self.buckets_by_name.values()
            ),
            "unique_opponent_count": self.size,
            "unique_neural_opponent_count": self.unique_neural_opponent_count,
            "configured_buckets": list(self.selected_buckets),
            "available_buckets": list(self.available_bucket_names()),
            "bucket_overlap_counts": overlaps,
            "total_bucket_overlap_count": sum(overlaps.values()),
            "buckets": bucket_state,
        }

    def _band_cutoff_iteration(self, bucket_name):
        """Return the theoretical age boundary a delayed band admits behind."""
        if bucket_name == "medium_term":
            return medium_term_cutoff_iteration(self.last_completed_rl_iteration)
        if bucket_name == "historical_uniform":
            return historical_uniform_cutoff_iteration(
                self.last_completed_rl_iteration
            )
        return None

    def bucket_overlap_counts(self):
        """Return every neural membership intersection, named by its pair.

        Reporting one aggregate count would hide which disjointness invariant
        failed, so each selected pair keeps its own entry.
        """
        selected = [
            name for name in NEURAL_BUCKET_NAMES if name in self.buckets_by_name
        ]
        return {
            f"{first}|{second}": len(
                set(self.buckets_by_name[first].member_ids)
                & set(self.buckets_by_name[second].member_ids)
            )
            for index, first in enumerate(selected)
            for second in selected[index + 1:]
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
                "band_removals": dict(self._band_removal_counts),
                "band_rebalances": dict(self._band_rebalance_counts),
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
            # A delayed band is legitimately empty until the archive makes its
            # region old enough, so an empty restored list is not a defect.
            bucket.member_ids = list(saved.get("member_ids", ()))
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
        groups = (
            "admissions",
            "fifo_evictions",
            "band_removals",
            "band_rebalances",
        )
        if any(
            set(counters.get(group, {})) != set(self.selected_buckets)
            for group in groups
        ):
            raise ValueError("Resume opponent-pool lifecycle counters are incomplete")
        (
            self._admission_counts,
            self._eviction_counts,
            self._band_removal_counts,
            self._band_rebalance_counts,
        ) = (
            {name: int(counters[group][name]) for name in self.selected_buckets}
            for group in groups
        )

    def manifest(self):
        """Return fixed policy details, not mutable pool contents."""
        return pool_policy_manifest(self.selected_buckets)
