"""Deterministic exact-budget matchmaking for RL opponent buckets."""

from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math

import numpy as np

from training.rl.pool import (
    HEURISTIC_KIND,
    RANDOM_KIND,
    SNAPSHOT_KIND,
    canonicalize_bucket_names,
)
from training.utils.seeding import stable_seed


MATCHMAKING_POLICY_VERSION = 4
DIFFICULTY_WIN_RATE_LOWER = 0.375
DIFFICULTY_WIN_RATE_UPPER = 0.625
DIFFICULTY_PRIOR_GAMES = 20.0
DIFFICULTY_EVIDENCE_HALF_LIFE_ITERATIONS = 10.0
_EVIDENCE_DECAY = 2 ** (-1 / DIFFICULTY_EVIDENCE_HALF_LIFE_ITERATIONS)


def matchmaking_policy_manifest():
    """Return all internal computation-affecting matchmaking constants."""
    return {
        "policy_version": MATCHMAKING_POLICY_VERSION,
        "allocation": "hierarchical_largest_remainder",
        "difficulty_win_rate_lower": DIFFICULTY_WIN_RATE_LOWER,
        "difficulty_win_rate_upper": DIFFICULTY_WIN_RATE_UPPER,
        "difficulty_prior_games": DIFFICULTY_PRIOR_GAMES,
        "difficulty_evidence_half_life_iterations": (
            DIFFICULTY_EVIDENCE_HALF_LIFE_ITERATIONS
        ),
        "difficulty_evidence_decay": _EVIDENCE_DECAY,
        "new_opponent_win_rate": 0.5,
        "game_result_outcomes": ["learner_win", "learner_loss"],
        "integer_rounding": "round_half_up_then_largest_remainder",
        "integer_allocation_tie_breaking": (
            "bucket_registry_order_then_opponent_id"
        ),
        "bucket_allocation": "largest_remainder_over_available_buckets",
        "uniform_member_allocation": "cyclic_remainder_by_persisted_anchor",
        "uniform_rotation_anchor": (
            "last_extra_game_opponent_id_then_bisect_right_successor"
        ),
        "difficulty_member_allocation": "largest_remainder_by_difficulty",
        "component_allocation_scope": "available_buckets_only",
        "empty_configured_bucket_games": 0,
        "assignment_order": "stable_seeded_shuffle",
    }


def difficulty_from_win_rate(win_rate):
    """Map learner win rate to the calibrated [0, 1] opponent difficulty."""
    value = (
        DIFFICULTY_WIN_RATE_UPPER - float(win_rate)
    ) / (
        DIFFICULTY_WIN_RATE_UPPER - DIFFICULTY_WIN_RATE_LOWER
    )
    return min(1.0, max(0.0, value))


@dataclass
class OpponentPerformance:
    """Smoothed win/loss evidence plus auditable lifetime outcome counters."""

    decayed_wins: float = 0.0
    decayed_losses: float = 0.0
    lifetime_wins: int = 0
    lifetime_losses: int = 0

    @property
    def estimated_win_rate(self):
        return (
            self.decayed_wins + 0.5 * DIFFICULTY_PRIOR_GAMES
        ) / (
            self.decayed_wins
            + self.decayed_losses
            + DIFFICULTY_PRIOR_GAMES
        )

    @property
    def difficulty(self):
        return difficulty_from_win_rate(self.estimated_win_rate)

    def decay(self):
        self.decayed_wins *= _EVIDENCE_DECAY
        self.decayed_losses *= _EVIDENCE_DECAY

    def add(self, *, wins=0, losses=0):
        wins = int(wins)
        losses = int(losses)
        if min(wins, losses) < 0:
            raise ValueError("Opponent outcomes cannot be negative")
        self.decayed_wins += wins
        self.decayed_losses += losses
        self.lifetime_wins += wins
        self.lifetime_losses += losses


class OpponentPerformanceTracker:
    """Own one shared performance estimate per durable opponent identity."""

    def __init__(self):
        self._values = {}

    def ensure(self, opponent_ids):
        for opponent_id in opponent_ids:
            self._values.setdefault(opponent_id, OpponentPerformance())

    def performance(self, opponent_id):
        self.ensure((opponent_id,))
        return self._values[opponent_id]

    def estimated_win_rate(self, opponent_id):
        return self.performance(opponent_id).estimated_win_rate

    def difficulty(self, opponent_id):
        return self.performance(opponent_id).difficulty

    def update(self, active_opponent_ids, outcomes):
        """Decay every active opponent once, then add this iteration's evidence."""
        active = tuple(sorted(set(active_opponent_ids)))
        self.ensure(active)
        for opponent_id in active:
            self._values[opponent_id].decay()
        for opponent_id, result in outcomes.items():
            if opponent_id not in self._values:
                raise ValueError(
                    f"Iteration returned inactive opponent {opponent_id!r}"
                )
            unexpected = set(result) - {"games", "wins", "losses"}
            if unexpected:
                raise ValueError(
                    "Opponent result contains unsupported outcome fields: "
                    + ", ".join(sorted(unexpected))
                )
            wins = int(result.get("wins", 0))
            losses = int(result.get("losses", 0))
            if "games" in result and int(result["games"]) != wins + losses:
                raise ValueError(
                    "Opponent games must equal wins plus losses; game-result "
                    "draws do not exist in this ruleset"
                )
            self._values[opponent_id].add(
                wins=wins,
                losses=losses,
            )

    def retain_only(self, opponent_ids):
        active = set(opponent_ids)
        self._values = {
            opponent_id: value
            for opponent_id, value in self._values.items()
            if opponent_id in active
        }

    def export_state(self):
        return {
            "policy_manifest": matchmaking_policy_manifest(),
            "opponents": {
                opponent_id: asdict(value)
                for opponent_id, value in sorted(self._values.items())
            },
        }

    def restore_state(self, state, active_opponent_ids):
        if state.get("policy_manifest") != matchmaking_policy_manifest():
            raise ValueError("Resume matchmaking policy manifest changed")
        active = set(active_opponent_ids)
        stored = state.get("opponents", {})
        if set(stored) != active:
            raise ValueError(
                "Resume performance identities do not match active opponents"
            )
        self._values = {
            opponent_id: OpponentPerformance(**value)
            for opponent_id, value in stored.items()
        }


def _rotation_start_index(members, anchor):
    """Return the first member index that receives a rotating extra game.

    ``bisect_right`` covers both a live and a departed anchor with one lookup:
    an anchor that is still a member sits immediately before its own insertion
    point, so the result is the next identity, and an anchor that has left the
    bucket resolves to the first surviving identity that sorts after it.
    """
    if anchor is None:
        return 0
    return bisect.bisect_right(members, anchor) % len(members)


def cyclic_remainder_allocation(total, members, *, anchor):
    """Split one bucket budget equally and rotate the remainder by identity.

    Every member receives ``total // len(members)`` games. The
    ``total % len(members)`` leftover games go to consecutive members starting
    after ``anchor``, so the identities carrying the remainder move forward
    across iterations instead of always being the same stable-order prefix.
    Returns the per-member counts and the anchor the caller should commit.
    """
    total = int(total)
    if total < 0:
        raise ValueError("Uniform member allocation total cannot be negative")
    members = tuple(members)
    if not members:
        if total:
            raise ValueError(
                "Cannot allocate a non-zero uniform budget to no members"
            )
        return {}, anchor
    count = len(members)
    base, extra = divmod(total, count)
    counts = {opponent_id: base for opponent_id in members}
    # No remainder means no bias to rotate away from, so the anchor keeps its
    # meaning for the next iteration that actually has one.
    if extra == 0:
        return counts, anchor
    start = _rotation_start_index(members, anchor)
    for step in range(extra):
        counts[members[(start + step) % count]] += 1
    return counts, members[(start + extra - 1) % count]


class UniformRotationState:
    """Own one logical uniform-remainder anchor per configured bucket.

    The anchor is a durable opponent ID and not an integer cursor. Bucket
    membership changes constantly: ``recent`` evicts and admits on every
    successful update and the archive-backed bands rebalance, so an index would
    silently change meaning while an ID stays interpretable through
    ``bisect_right`` continuation.
    """

    def __init__(self, bucket_names=()):
        self._anchors = {}
        self.ensure(bucket_names)

    @property
    def configured_buckets(self):
        """Return the registered bucket names in canonical registry order."""
        if not self._anchors:
            return ()
        return canonicalize_bucket_names(tuple(self._anchors))

    def ensure(self, bucket_names):
        """Register every configured bucket, including the ones still empty.

        A warm-up bucket keeps an anchor so it starts from the first canonical
        member the moment it becomes available, rather than inheriting one.
        """
        names = tuple(bucket_names)
        if not names:
            return
        for name in canonicalize_bucket_names(names):
            self._anchors.setdefault(name, None)

    def anchor(self, bucket_name):
        """Return the last member ID that received a rotating extra game."""
        try:
            return self._anchors[bucket_name]
        except KeyError as exc:
            raise KeyError(
                f"Uniform rotation bucket {bucket_name!r} is not configured"
            ) from exc

    def anchors(self):
        """Return every current anchor in canonical registry order."""
        return {name: self._anchors[name] for name in self.configured_buckets}

    def plan_uniform_allocation(self, bucket_budgets, bucket_members):
        """Return exact member counts plus the anchors a plan would commit.

        This never mutates durable state. The caller commits the returned
        anchors only after the complete match plan validates, so a rejected
        plan cannot advance the rotation.
        """
        counts = {}
        next_anchors = {}
        for bucket_name, budget in bucket_budgets.items():
            members = tuple(bucket_members.get(bucket_name, ()))
            if list(members) != sorted(set(members)):
                raise ValueError(
                    f"Uniform rotation members for {bucket_name!r} are not "
                    "sorted and unique"
                )
            member_counts, next_anchor = cyclic_remainder_allocation(
                budget,
                members,
                anchor=self.anchor(bucket_name),
            )
            counts.update({
                (bucket_name, opponent_id): value
                for opponent_id, value in member_counts.items()
            })
            next_anchors[bucket_name] = next_anchor
        return counts, next_anchors

    def commit(self, next_anchors):
        """Advance the durable anchors after a complete plan has validated."""
        unknown = sorted(set(next_anchors) - set(self._anchors))
        if unknown:
            raise KeyError(
                "Uniform rotation cannot commit unconfigured bucket(s): "
                + ", ".join(unknown)
            )
        self._anchors.update({
            name: (None if value is None else str(value))
            for name, value in next_anchors.items()
        })

    def export_state(self):
        """Return JSON-safe rotation state for exact resume."""
        return {
            "policy_manifest": matchmaking_policy_manifest(),
            "anchors": self.anchors(),
        }

    def restore_state(self, state, bucket_names):
        """Restore anchors verbatim instead of deriving them from membership."""
        if state.get("policy_manifest") != matchmaking_policy_manifest():
            raise ValueError("Resume matchmaking policy manifest changed")
        names = canonicalize_bucket_names(bucket_names)
        stored = state.get("anchors", {})
        if set(stored) != set(names):
            raise ValueError(
                "Resume uniform rotation buckets do not match the configured "
                "selection"
            )
        self._anchors = {
            name: (None if stored[name] is None else str(stored[name]))
            for name in names
        }


@dataclass(frozen=True)
class MatchAllocation:
    """Exact component counts for one bucket membership."""

    bucket_name: str
    opponent_id: str
    opponent_kind: str
    bank_slot: int | None
    game_count: int
    estimated_win_rate: float
    difficulty: float
    uniform_games: int
    difficulty_games: int


@dataclass(frozen=True)
class GameAssignment:
    """One immutable absolute-game assignment executed by a worker."""

    game_index: int
    bucket_name: str
    opponent_id: str
    opponent_kind: str
    bank_slot: int | None


@dataclass(frozen=True)
class MatchPlan:
    """Complete exact-budget allocation for one RL iteration."""

    iteration: int
    first_absolute_game: int
    game_count: int
    difficulty_weight: float
    uniform_budget: int
    difficulty_budget: int
    configured_buckets: tuple[str, ...]
    available_buckets: tuple[str, ...]
    # Rotation anchors are ordered pairs rather than a mapping so the plan stays
    # immutable like every other field, and so the transition itself is part of
    # durable plan identity.
    uniform_rotation_before: tuple[tuple[str, str | None], ...]
    uniform_rotation_after: tuple[tuple[str, str | None], ...]
    allocations: tuple[MatchAllocation, ...]
    assignments: tuple[GameAssignment, ...]
    plan_sha256: str


def _round_half_up(value):
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def matchmaking_component_budgets(game_count, difficulty_weight):
    """Return exact uniform/difficulty budgets for one iteration game count."""
    game_count = int(game_count)
    difficulty_weight = float(difficulty_weight)
    if game_count < 0:
        raise ValueError("Matchmaking game count cannot be negative")
    if not 0.0 <= difficulty_weight <= 1.0:
        raise ValueError("difficulty_weight must be between 0 and 1")
    difficulty_budget = _round_half_up(difficulty_weight * game_count)
    return game_count - difficulty_budget, difficulty_budget


def _largest_remainder(total, keys, weights):
    """Allocate an exact non-negative integer total with stable key ties."""
    total = int(total)
    keys = tuple(keys)
    if total < 0:
        raise ValueError("Allocation total cannot be negative")
    if not keys:
        if total:
            raise ValueError("Cannot allocate a non-zero total to no keys")
        return {}
    normalized = [max(0.0, float(weights[key])) for key in keys]
    weight_sum = math.fsum(normalized)
    if weight_sum <= 0:
        normalized = [1.0] * len(keys)
        weight_sum = float(len(keys))
    quotas = [total * weight / weight_sum for weight in normalized]
    floors = [math.floor(value) for value in quotas]
    remaining = total - sum(floors)
    ranked = sorted(
        range(len(keys)),
        key=lambda index: (-(quotas[index] - floors[index]), index),
    )
    for index in ranked[:remaining]:
        floors[index] += 1
    return {key: int(value) for key, value in zip(keys, floors)}


def _canonical_members(opponent_pool, buckets):
    """Return the stable logical member order both components allocate over."""
    return {
        bucket: tuple(sorted(opponent_pool.bucket_members(bucket)))
        for bucket in buckets
    }


def _component_bucket_budgets(
    opponent_pool,
    tracker,
    buckets,
    budget,
    difficulty,
):
    """Split one component budget across available buckets.

    Bucket-level allocation is identical for both components and is deliberately
    unchanged by the member-level rotation: only the identities that receive a
    bucket's remainder games rotate, never the size of a bucket's share.
    """
    bucket_keys = tuple(buckets)
    # Only available buckets reach this point, so no weight divides by zero.
    empty = [
        bucket for bucket in bucket_keys
        if not opponent_pool.bucket_members(bucket)
    ]
    if empty:
        raise ValueError(
            "Component allocation received empty bucket(s): " + ", ".join(empty)
        )
    if difficulty:
        bucket_weights = {
            bucket: math.fsum(
                tracker.difficulty(opponent_id)
                for opponent_id in opponent_pool.bucket_members(bucket)
            ) / len(opponent_pool.bucket_members(bucket))
            for bucket in bucket_keys
        }
    else:
        bucket_weights = {bucket: 1.0 for bucket in bucket_keys}
    return _largest_remainder(budget, bucket_keys, bucket_weights)


def _difficulty_member_allocation(tracker, members_by_bucket, bucket_counts):
    """Allocate each bucket's difficulty games by individual member difficulty.

    This stays exactly meritocratic. The uniform rotation cursor is never
    applied here, so a hard opponent keeps concentrating games regardless of
    where the uniform remainder currently sits.
    """
    result = {}
    for bucket, count in bucket_counts.items():
        members = members_by_bucket[bucket]
        member_counts = _largest_remainder(
            count,
            members,
            {opponent_id: tracker.difficulty(opponent_id) for opponent_id in members},
        )
        result.update({
            (bucket, opponent_id): value
            for opponent_id, value in member_counts.items()
        })
    return result


def _plan_hash_payload(
    *,
    iteration,
    first_absolute_game,
    game_count,
    difficulty_weight,
    uniform_budget,
    difficulty_budget,
    configured_buckets,
    available_buckets,
    uniform_rotation_before,
    uniform_rotation_after,
    allocations,
    assignments,
):
    # Physical bank slots are deliberately excluded. They are runtime-only
    # storage addresses and may be compacted differently after exact restore;
    # durable plan identity is the logical bucket/opponent/game assignment.
    durable_allocations = []
    for value in allocations:
        serialized = asdict(value)
        serialized.pop("bank_slot")
        durable_allocations.append(serialized)
    durable_assignments = []
    for value in assignments:
        serialized = asdict(value)
        serialized.pop("bank_slot")
        durable_assignments.append(serialized)
    return {
        "policy_version": MATCHMAKING_POLICY_VERSION,
        "iteration": int(iteration),
        "first_absolute_game": int(first_absolute_game),
        "game_count": int(game_count),
        "difficulty_weight": float(difficulty_weight),
        "uniform_budget": int(uniform_budget),
        "difficulty_budget": int(difficulty_budget),
        # Availability is part of durable plan identity: a warm-up plan must
        # never hash identically to the same allocation under a different
        # availability state.
        "configured_buckets": list(configured_buckets),
        "available_buckets": list(available_buckets),
        # Exact resume now owns the rotation, so the transition is hashed
        # directly instead of being inferred from the resulting assignments.
        "uniform_rotation_before": [
            [name, anchor] for name, anchor in uniform_rotation_before
        ],
        "uniform_rotation_after": [
            [name, anchor] for name, anchor in uniform_rotation_after
        ],
        "allocations": durable_allocations,
        "assignments": durable_assignments,
    }


def _sha256(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_match_plan(
    *,
    opponent_pool,
    performance_tracker,
    uniform_rotation,
    selected_buckets,
    difficulty_weight,
    iteration,
    first_absolute_game,
    game_count,
    base_seed,
):
    """Build and validate one deterministic plan totaling exactly ``game_count``.

    ``uniform_rotation`` is advanced only after the complete plan validates, so
    a rejected plan can never consume a rotation step. It is a required
    argument: defaulting to a throwaway state would silently restore the
    stable-prefix member bias this rotation exists to remove.
    """
    buckets = canonicalize_bucket_names(selected_buckets)
    if buckets != tuple(opponent_pool.selected_buckets):
        raise ValueError("Matchmaker buckets do not match the active opponent pool")
    game_count = int(game_count)
    if game_count < 1:
        raise ValueError("Match plan game_count must be positive")
    alpha = float(difficulty_weight)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("difficulty_weight must be between 0 and 1")
    available = opponent_pool.available_bucket_names()
    if not available:
        raise ValueError(
            "Every configured opponent bucket is empty: "
            + ", ".join(buckets)
        )
    performance_tracker.ensure(
        record.opponent_id for record in opponent_pool.active_opponents()
    )
    # Registering a configured bucket only creates a ``None`` anchor; it can
    # never advance one, so this is safe before the plan validates.
    uniform_rotation.ensure(buckets)
    if uniform_rotation.configured_buckets != buckets:
        raise ValueError(
            "Uniform rotation buckets do not match the active opponent pool"
        )
    rotation_before = uniform_rotation.anchors()

    uniform_budget, difficulty_budget = matchmaking_component_budgets(
        game_count,
        alpha,
    )
    members_by_bucket = _canonical_members(opponent_pool, available)
    uniform, next_anchors = uniform_rotation.plan_uniform_allocation(
        _component_bucket_budgets(
            opponent_pool,
            performance_tracker,
            available,
            uniform_budget,
            difficulty=False,
        ),
        members_by_bucket,
    )
    # An unavailable bucket is absent from the budget mapping, so its anchor
    # carries over untouched instead of resetting when it becomes available.
    rotation_after = {**rotation_before, **next_anchors}
    difficult = _difficulty_member_allocation(
        performance_tracker,
        members_by_bucket,
        _component_bucket_budgets(
            opponent_pool,
            performance_tracker,
            available,
            difficulty_budget,
            difficulty=True,
        ),
    )
    allocation_keys = tuple(
        (bucket, opponent_id)
        for bucket in available
        for opponent_id in members_by_bucket[bucket]
    )
    allocations = tuple(
        MatchAllocation(
            bucket_name=bucket,
            opponent_id=opponent_id,
            opponent_kind=opponent_pool.opponent(opponent_id).kind,
            bank_slot=opponent_pool.bank_slot(opponent_id),
            game_count=uniform.get((bucket, opponent_id), 0)
            + difficult.get((bucket, opponent_id), 0),
            estimated_win_rate=performance_tracker.estimated_win_rate(opponent_id),
            difficulty=performance_tracker.difficulty(opponent_id),
            uniform_games=uniform.get((bucket, opponent_id), 0),
            difficulty_games=difficult.get((bucket, opponent_id), 0),
        )
        for bucket, opponent_id in allocation_keys
    )

    ordered_memberships = []
    for allocation in allocations:
        ordered_memberships.extend(
            (allocation.bucket_name, allocation.opponent_id)
            for _game in range(allocation.game_count)
        )
    generator = np.random.default_rng(stable_seed(
        base_seed,
        "match_plan",
        int(iteration),
        int(first_absolute_game),
        game_count,
    ))
    if ordered_memberships:
        permutation = generator.permutation(len(ordered_memberships))
        ordered_memberships = [ordered_memberships[index] for index in permutation]
    assignments = tuple(
        GameAssignment(
            game_index=int(first_absolute_game) + local_index,
            bucket_name=bucket,
            opponent_id=opponent_id,
            opponent_kind=opponent_pool.opponent(opponent_id).kind,
            bank_slot=opponent_pool.bank_slot(opponent_id),
        )
        for local_index, (bucket, opponent_id) in enumerate(ordered_memberships)
    )
    rotation_pairs_before = tuple(
        (name, rotation_before[name]) for name in buckets
    )
    rotation_pairs_after = tuple(
        (name, rotation_after[name]) for name in buckets
    )
    payload = _plan_hash_payload(
        iteration=iteration,
        first_absolute_game=first_absolute_game,
        game_count=game_count,
        difficulty_weight=alpha,
        uniform_budget=uniform_budget,
        difficulty_budget=difficulty_budget,
        configured_buckets=buckets,
        available_buckets=available,
        uniform_rotation_before=rotation_pairs_before,
        uniform_rotation_after=rotation_pairs_after,
        allocations=allocations,
        assignments=assignments,
    )
    plan = MatchPlan(
        iteration=int(iteration),
        first_absolute_game=int(first_absolute_game),
        game_count=game_count,
        difficulty_weight=alpha,
        uniform_budget=uniform_budget,
        difficulty_budget=difficulty_budget,
        configured_buckets=buckets,
        available_buckets=available,
        uniform_rotation_before=rotation_pairs_before,
        uniform_rotation_after=rotation_pairs_after,
        allocations=allocations,
        assignments=assignments,
        plan_sha256=_sha256(payload),
    )
    validate_match_plan(plan, opponent_pool)
    # The rotation is consumed only here, once nothing can still reject the
    # plan the extra games were allocated for.
    uniform_rotation.commit(next_anchors)
    return plan


def validate_match_plan(plan, opponent_pool):
    """Raise if any exact-budget, identity, or storage invariant is violated."""
    if len(plan.assignments) != plan.game_count:
        raise ValueError("Match plan assignment count differs from game_count")
    if sum(value.game_count for value in plan.allocations) != plan.game_count:
        raise ValueError("Match plan allocations do not sum to game_count")
    if plan.uniform_budget + plan.difficulty_budget != plan.game_count:
        raise ValueError("Match plan component budgets do not sum to game_count")
    if plan.configured_buckets != tuple(opponent_pool.selected_buckets):
        raise ValueError("Match plan configured buckets changed")
    if plan.available_buckets != opponent_pool.available_bucket_names():
        raise ValueError("Match plan availability does not match the pool")
    if not set(plan.available_buckets) <= set(plan.configured_buckets):
        raise ValueError("Match plan availability is not a configured subset")
    _validate_uniform_rotation(plan, opponent_pool)
    expected_games = set(range(
        plan.first_absolute_game,
        plan.first_absolute_game + plan.game_count,
    ))
    actual_games = {value.game_index for value in plan.assignments}
    if actual_games != expected_games or len(actual_games) != len(plan.assignments):
        raise ValueError("Match plan game IDs are not complete and unique")
    available = set(plan.available_buckets)
    for assignment in plan.assignments:
        if assignment.bucket_name not in available:
            raise ValueError("Match plan references an inactive bucket")
        if assignment.opponent_id not in opponent_pool.bucket_members(
            assignment.bucket_name
        ):
            raise ValueError("Match plan references a non-member opponent")
        if assignment.opponent_kind == SNAPSHOT_KIND:
            if assignment.bank_slot is None:
                raise ValueError("Neural match assignment has no bank slot")
            if opponent_pool.bank_slot(assignment.opponent_id) != assignment.bank_slot:
                raise ValueError("Neural match assignment has a stale bank slot")
        elif assignment.opponent_kind in (HEURISTIC_KIND, RANDOM_KIND):
            if assignment.bank_slot is not None:
                raise ValueError(
                    "Programmatic match assignment allocated a bank slot"
                )
        else:
            raise ValueError(
                f"Unknown opponent kind: {assignment.opponent_kind!r}"
            )


def _validate_uniform_rotation(plan, opponent_pool):
    """Raise if the recorded rotation transition is not the one a plan earned."""
    before = dict(plan.uniform_rotation_before)
    after = dict(plan.uniform_rotation_after)
    configured = tuple(plan.configured_buckets)
    if tuple(name for name, _anchor in plan.uniform_rotation_before) != configured:
        raise ValueError("Match plan rotation anchors are not in configured order")
    if tuple(name for name, _anchor in plan.uniform_rotation_after) != configured:
        raise ValueError("Match plan rotation anchors are not in configured order")
    available = set(plan.available_buckets)
    uniform_by_bucket = {name: 0 for name in configured}
    for allocation in plan.allocations:
        uniform_by_bucket[allocation.bucket_name] += allocation.uniform_games
    if sum(uniform_by_bucket.values()) != plan.uniform_budget:
        raise ValueError("Match plan uniform games do not sum to the uniform budget")
    for name in configured:
        member_count = len(opponent_pool.bucket_members(name))
        # An unavailable bucket receives no games, and a bucket whose uniform
        # share divides evenly has no remainder to rotate. Both must carry
        # their anchor forward untouched.
        rotated = (
            name in available
            and member_count
            and uniform_by_bucket[name] % member_count
        )
        if not rotated:
            if after[name] != before[name]:
                raise ValueError(
                    f"Match plan advanced the {name!r} rotation without a remainder"
                )
        elif after[name] not in opponent_pool.bucket_members(name):
            raise ValueError(
                f"Match plan rotation anchor for {name!r} is not a current member"
            )


def aggregate_match_results(plan, rollout_results):
    """Validate returned identities and aggregate learner outcomes by ID/bucket."""
    expected = {value.game_index: value for value in plan.assignments}
    if len(rollout_results) != len(expected):
        raise ValueError("Rollout result count does not match the MatchPlan")
    by_opponent = {
        allocation.opponent_id: {
            "games": 0,
            "wins": 0,
            "losses": 0,
        }
        for allocation in plan.allocations
    }
    # Every configured bucket keeps a row, including the ones still empty
    # during warm-up. The compact metrics header depends on a fixed shape.
    by_bucket = {
        bucket_name: {
            "games": 0,
            "wins": 0,
            "losses": 0,
        }
        for bucket_name in plan.configured_buckets
    }
    seen = set()
    for result in rollout_results:
        game_index = int(result["game_index"])
        if game_index in seen or game_index not in expected:
            raise ValueError("Rollout returned a duplicate or unexpected game ID")
        seen.add(game_index)
        assignment = expected[game_index]
        for field in ("bucket_name", "opponent_id", "opponent_kind", "bank_slot"):
            if result.get(field) != getattr(assignment, field):
                raise ValueError(
                    f"Rollout result changed assignment field {field!r} for "
                    f"game {game_index}"
                )
        winner = result["winner"]
        learner_position = result["learner_position"]
        if winner not in (0, 1):
            raise ValueError(
                f"Game {game_index} returned invalid winner {winner!r}; "
                "game-result draws do not exist in this ruleset"
            )
        if learner_position not in (0, 1):
            raise ValueError(
                f"Game {game_index} returned invalid learner position "
                f"{learner_position!r}"
            )
        outcome = "wins" if winner == learner_position else "losses"
        opponent_row = by_opponent[assignment.opponent_id]
        bucket_row = by_bucket[assignment.bucket_name]
        for row in (opponent_row, bucket_row):
            row["games"] += 1
            row[outcome] += 1
    if seen != set(expected):
        raise ValueError("Rollout did not complete every MatchPlan game")
    return by_opponent, by_bucket
