"""Deterministic exact-budget matchmaking for RL opponent buckets."""

from __future__ import annotations

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


MATCHMAKING_POLICY_VERSION = 1
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
        "draw_points": 0.5,
        "integer_rounding": "round_half_up_then_largest_remainder",
        "tie_breaking": "bucket_registry_order_then_opponent_id",
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
    """Smoothed recent evidence plus auditable lifetime outcome counters."""

    decayed_learner_points: float = 0.0
    decayed_games: float = 0.0
    lifetime_games: int = 0
    lifetime_wins: int = 0
    lifetime_losses: int = 0
    lifetime_draws: int = 0

    @property
    def estimated_win_rate(self):
        return (
            self.decayed_learner_points + 0.5 * DIFFICULTY_PRIOR_GAMES
        ) / (
            self.decayed_games + DIFFICULTY_PRIOR_GAMES
        )

    @property
    def difficulty(self):
        return difficulty_from_win_rate(self.estimated_win_rate)

    def decay(self):
        self.decayed_learner_points *= _EVIDENCE_DECAY
        self.decayed_games *= _EVIDENCE_DECAY

    def add(self, *, wins=0, losses=0, draws=0):
        wins = int(wins)
        losses = int(losses)
        draws = int(draws)
        if min(wins, losses, draws) < 0:
            raise ValueError("Opponent outcomes cannot be negative")
        games = wins + losses + draws
        self.decayed_learner_points += wins + 0.5 * draws
        self.decayed_games += games
        self.lifetime_games += games
        self.lifetime_wins += wins
        self.lifetime_losses += losses
        self.lifetime_draws += draws


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
            self._values[opponent_id].add(
                wins=result.get("wins", 0),
                losses=result.get("losses", 0),
                draws=result.get("draws", 0),
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
    allocations: tuple[MatchAllocation, ...]
    assignments: tuple[GameAssignment, ...]
    plan_sha256: str


def _round_half_up(value):
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


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


def _component_allocation(opponent_pool, tracker, buckets, budget, difficulty):
    bucket_keys = tuple(buckets)
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
    bucket_counts = _largest_remainder(budget, bucket_keys, bucket_weights)
    result = {}
    for bucket in bucket_keys:
        members = tuple(sorted(opponent_pool.bucket_members(bucket)))
        if not members:
            raise ValueError(f"Selected opponent bucket {bucket!r} is empty")
        member_weights = (
            {opponent_id: tracker.difficulty(opponent_id) for opponent_id in members}
            if difficulty
            else {opponent_id: 1.0 for opponent_id in members}
        )
        member_counts = _largest_remainder(
            bucket_counts[bucket],
            members,
            member_weights,
        )
        result.update({(bucket, opponent_id): count for opponent_id, count in member_counts.items()})
    return result


def _plan_hash_payload(
    *,
    iteration,
    first_absolute_game,
    game_count,
    difficulty_weight,
    uniform_budget,
    difficulty_budget,
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
    selected_buckets,
    difficulty_weight,
    iteration,
    first_absolute_game,
    game_count,
    base_seed,
):
    """Build and validate one deterministic plan totaling exactly ``game_count``."""
    buckets = canonicalize_bucket_names(selected_buckets)
    if buckets != tuple(opponent_pool.selected_buckets):
        raise ValueError("Matchmaker buckets do not match the active opponent pool")
    game_count = int(game_count)
    if game_count < 1:
        raise ValueError("Match plan game_count must be positive")
    alpha = float(difficulty_weight)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("difficulty_weight must be between 0 and 1")
    for bucket in buckets:
        if not opponent_pool.bucket_members(bucket):
            raise ValueError(f"Selected opponent bucket {bucket!r} is empty")
    performance_tracker.ensure(
        record.opponent_id for record in opponent_pool.active_opponents()
    )

    difficulty_budget = _round_half_up(alpha * game_count)
    uniform_budget = game_count - difficulty_budget
    uniform = _component_allocation(
        opponent_pool,
        performance_tracker,
        buckets,
        uniform_budget,
        difficulty=False,
    )
    difficult = _component_allocation(
        opponent_pool,
        performance_tracker,
        buckets,
        difficulty_budget,
        difficulty=True,
    )
    allocation_keys = tuple(
        (bucket, opponent_id)
        for bucket in buckets
        for opponent_id in sorted(opponent_pool.bucket_members(bucket))
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
    payload = _plan_hash_payload(
        iteration=iteration,
        first_absolute_game=first_absolute_game,
        game_count=game_count,
        difficulty_weight=alpha,
        uniform_budget=uniform_budget,
        difficulty_budget=difficulty_budget,
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
        allocations=allocations,
        assignments=assignments,
        plan_sha256=_sha256(payload),
    )
    validate_match_plan(plan, opponent_pool)
    return plan


def validate_match_plan(plan, opponent_pool):
    """Raise if any exact-budget, identity, or storage invariant is violated."""
    if len(plan.assignments) != plan.game_count:
        raise ValueError("Match plan assignment count differs from game_count")
    if sum(value.game_count for value in plan.allocations) != plan.game_count:
        raise ValueError("Match plan allocations do not sum to game_count")
    if plan.uniform_budget + plan.difficulty_budget != plan.game_count:
        raise ValueError("Match plan component budgets do not sum to game_count")
    expected_games = set(range(
        plan.first_absolute_game,
        plan.first_absolute_game + plan.game_count,
    ))
    actual_games = {value.game_index for value in plan.assignments}
    if actual_games != expected_games or len(actual_games) != len(plan.assignments):
        raise ValueError("Match plan game IDs are not complete and unique")
    selected = set(opponent_pool.selected_buckets)
    for assignment in plan.assignments:
        if assignment.bucket_name not in selected:
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
            "draws": 0,
        }
        for allocation in plan.allocations
    }
    by_bucket = {
        allocation.bucket_name: {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
        }
        for allocation in plan.allocations
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
        winner = int(result["winner"])
        learner_position = int(result["learner_position"])
        outcome = (
            "draws" if winner not in (0, 1)
            else "wins" if winner == learner_position
            else "losses"
        )
        opponent_row = by_opponent[assignment.opponent_id]
        bucket_row = by_bucket[assignment.bucket_name]
        for row in (opponent_row, bucket_row):
            row["games"] += 1
            row[outcome] += 1
    if seen != set(expected):
        raise ValueError("Rollout did not complete every MatchPlan game")
    return by_opponent, by_bucket


def matchmaking_metrics(plan, by_opponent, by_bucket):
    """Return compact auditable allocation and outcome provenance."""
    bucket_payload = {
        bucket_name: {
            **dict(value),
            "allocations": [],
        }
        for bucket_name, value in by_bucket.items()
    }
    opponent_payload = {}
    for allocation in plan.allocations:
        membership = {
            "opponent_id": allocation.opponent_id,
            "opponent_kind": allocation.opponent_kind,
            "game_count": allocation.game_count,
            "uniform_games": allocation.uniform_games,
            "difficulty_games": allocation.difficulty_games,
            "estimated_win_rate": allocation.estimated_win_rate,
            "difficulty": allocation.difficulty,
        }
        bucket_payload[allocation.bucket_name]["allocations"].append(membership)
        opponent = opponent_payload.setdefault(allocation.opponent_id, {
            "opponent_id": allocation.opponent_id,
            "opponent_kind": allocation.opponent_kind,
            "game_count": 0,
            "uniform_games": 0,
            "difficulty_games": 0,
            "estimated_win_rate": allocation.estimated_win_rate,
            "difficulty": allocation.difficulty,
            "outcomes": dict(by_opponent[allocation.opponent_id]),
        })
        opponent["game_count"] += allocation.game_count
        opponent["uniform_games"] += allocation.uniform_games
        opponent["difficulty_games"] += allocation.difficulty_games
    return {
        "plan_sha256": plan.plan_sha256,
        "difficulty_weight": plan.difficulty_weight,
        "uniform_budget": plan.uniform_budget,
        "difficulty_budget": plan.difficulty_budget,
        "buckets": bucket_payload,
        "opponents": list(opponent_payload.values()),
    }
