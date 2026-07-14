"""Exact two-player opponent inference with slot and hand-weight beliefs.

The public model starts with an exact temporal-slot representation. Each slot
keeps the restrictions that were known when that tile entered the opponent's
hand. At the end of the first non-terminal public turn where ``comb(|U|, h)``
is at most ``SWITCH_TO_MU_MAX_HANDS``, the model converts once to an exact
``mu(H)`` distribution over hidden hands and remains there for the game.

The exported probabilities have direct presence semantics:

    p[s] = P(the opponent currently holds at least one tile containing suit s)

Thus ``0.0`` means known absence and ``1.0`` means known presence. The exact
path never creates particles or silently changes to an approximate posterior.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from math import comb
from time import perf_counter
from typing import Iterable, Sequence


SWITCH_TO_MU_MAX_HANDS = 500
MODEL_VERSION = "slots-mu-exact-v1"

ALL_TILES = [(i, j) for i in range(7) for j in range(i, 7)]
SUIT_COUNT = 7
INITIAL_HAND_SIZE = 7

TILE_TO_INDEX = {tile: index for index, tile in enumerate(ALL_TILES)}
INDEX_TO_TILE = {index: tile for tile, index in TILE_TO_INDEX.items()}
ALL_MASK = (1 << len(ALL_TILES)) - 1

SUIT_MASKS: list[int] = []
for suit in range(SUIT_COUNT):
    mask = 0
    for tile, index in TILE_TO_INDEX.items():
        if suit in tile:
            mask |= 1 << index
    SUIT_MASKS.append(mask)


class ProbabilityStage(Enum):
    """Named stages exposed while a public turn is being processed."""

    AFTER_NEGATIVE_EVIDENCE = "after_negative_evidence"
    AFTER_DRAW = "after_draw"
    END_TURN = "end_turn"


@dataclass(frozen=True)
class PublicAction:
    """A board-history action with reconstructed actor, turn, and board ends."""

    actor: int
    action: object
    ends_before: tuple[int, int] | None
    ends_after: tuple[int, int] | None
    history_action_index: int
    public_turn: int


@dataclass(frozen=True)
class ProbabilitySnapshot:
    """One exact probability vector at a named stage of a public turn."""

    public_turn: int
    history_action_index: int
    actor: int
    stage: ProbabilityStage
    probabilities: tuple[float, ...]
    mode: str
    ends: tuple[int, int] | None
    opponent_hand_size: int
    unknown_count: int
    state_count: int
    profile_count: int | None
    mu_hand_count: int | None
    total_weight: int
    raw_hand_upper_bound: int
    action: object
    same_as_previous: bool = False


@dataclass
class OpponentTurnTrace:
    """Labelled probability stages belonging to one reconstructed public turn."""

    public_turn: int
    actor: int
    after_negative_evidence: ProbabilitySnapshot | None = None
    after_draw: ProbabilitySnapshot | None = None
    end_turn: ProbabilitySnapshot | None = None


@dataclass(frozen=True)
class OpponentModelUpdate:
    """Rich result returned by :meth:`HybridExactOpponentModel.update_detailed`."""

    probabilities: tuple[float, ...]
    new_snapshots: tuple[ProbabilitySnapshot, ...]
    completed_turn_traces: tuple[OpponentTurnTrace, ...]
    mode: str
    switched_this_update: bool


def _normalize_tile(tile: Sequence[int]) -> tuple[int, int]:
    """Return the canonical unordered representation used by ``ALL_TILES``."""
    left, right = int(tile[0]), int(tile[1])
    return (left, right) if left <= right else (right, left)


def _normalize_action(action):
    """Convert JSON-list actions to tuple-based internal actions."""
    if action is None:
        return None
    if action == ["DRAW", None] or action == ("DRAW", None):
        return ("DRAW", None)
    tile, side = action
    return (_normalize_tile(tile), int(side))


def _is_draw(action) -> bool:
    return action is not None and action[0] == "DRAW"


def _is_pass(action) -> bool:
    return action is None


def _is_tile_play(action) -> bool:
    return action is not None and action[0] != "DRAW"


def _tile_bit(tile: Sequence[int]) -> int:
    return 1 << TILE_TO_INDEX[_normalize_tile(tile)]


def _indices_from_mask(mask: int) -> Iterable[int]:
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _bits_from_mask(mask: int) -> Iterable[int]:
    while mask:
        bit = mask & -mask
        yield bit
        mask ^= bit


def _mask_from_indices(indices: Iterable[int]) -> int:
    mask = 0
    for index in indices:
        mask |= 1 << index
    return mask


def mask_from_tiles(tiles: Iterable[Sequence[int]]) -> int:
    """Return a bit mask containing ``tiles``."""
    mask = 0
    for tile in tiles:
        mask |= _tile_bit(tile)
    return mask


def _legal_mask(left_end: int, right_end: int) -> int:
    return SUIT_MASKS[int(left_end)] | SUIT_MASKS[int(right_end)]


def _raw_hand_upper_bound(unknown_mask: int, hand_size: int) -> int:
    unknown_count = int(unknown_mask).bit_count()
    if hand_size < 0 or hand_size > unknown_count:
        return 0
    return comb(unknown_count, hand_size)


def reconstruct_public_actions(state: dict) -> list[PublicAction]:
    """Reconstruct actors, public turns, and board ends for action history.

    A draw keeps the actor and public-turn number. A pass or tile play closes
    the current public turn and advances both values.
    """
    board_history = [
        _normalize_action(action)
        for action in state.get("board_history", [])
    ]
    player_count = len(state.get("hand_sizes", [])) or 2
    current_player = int(
        state.get("history_current_player", state.get("current_player", 0))
    )

    # DominoEngine keeps the terminal actor as ``current_player`` because it
    # does not advance after a winning play or the final blocked-game pass.
    # Actor reconstruction needs the player who would have acted next.
    if state.get("game_over") and board_history and not _is_draw(board_history[-1]):
        current_player = (current_player + 1) % player_count

    advancing_actions = sum(1 for action in board_history if not _is_draw(action))
    actor = (current_player - advancing_actions) % player_count
    public_turn = 1
    ends: list[int] | None = None
    annotated: list[PublicAction] = []

    for action_index, action in enumerate(board_history, start=1):
        ends_before = tuple(ends) if ends is not None else None

        if _is_tile_play(action):
            tile, side = action
            if ends is None:
                ends = [tile[0], tile[1]]
            else:
                connected_value = ends[side]
                ends[side] = tile[1] if tile[0] == connected_value else tile[0]

        ends_after = tuple(ends) if ends is not None else None
        annotated.append(
            PublicAction(
                actor=actor,
                action=action,
                ends_before=ends_before,
                ends_after=ends_after,
                history_action_index=action_index,
                public_turn=public_turn,
            )
        )

        if not _is_draw(action):
            actor = (actor + 1) % player_count
            public_turn += 1

    return annotated


class MuOpponentBelief:
    """Exact integer-weight posterior ``hand_mask -> mu(H)``."""

    mode = "mu_exact"

    def __init__(
        self,
        *,
        unknown_mask: int,
        opponent_hand_size: int,
        weights: dict[int, int],
    ):
        self.unknown_mask = int(unknown_mask)
        self.opponent_hand_size = int(opponent_hand_size)
        self.weights = dict(weights)
        self._probability_cache: tuple[float, ...] | None = None
        self.assert_consistent()

    @classmethod
    def from_initial(
        cls,
        observer_initial_hand: Sequence[Sequence[int]],
        opponent_hand_size: int = INITIAL_HAND_SIZE,
    ) -> "MuOpponentBelief":
        """Enumerate the independent uniform initial hand distribution."""
        unknown_mask = ALL_MASK & ~mask_from_tiles(observer_initial_hand)
        unknown_indices = list(_indices_from_mask(unknown_mask))
        weights = {
            _mask_from_indices(indices): 1
            for indices in combinations(unknown_indices, int(opponent_hand_size))
        }
        return cls(
            unknown_mask=unknown_mask,
            opponent_hand_size=opponent_hand_size,
            weights=weights,
        )

    @classmethod
    def from_weights(
        cls,
        *,
        unknown_mask: int,
        opponent_hand_size: int,
        weights: dict[int, int],
    ) -> "MuOpponentBelief":
        """Build a belief from exact slot-conversion weights without re-enumeration."""
        return cls(
            unknown_mask=unknown_mask,
            opponent_hand_size=opponent_hand_size,
            weights=dict(weights),
        )

    def _invalidate_cache(self) -> None:
        self._probability_cache = None

    def assert_consistent(self) -> None:
        """Validate all exact hand-weight invariants."""
        if not self.weights:
            raise ValueError("Mu belief has no compatible hidden hands.")
        if self.opponent_hand_size < 0:
            raise ValueError("Opponent hand size cannot be negative.")
        for hand_mask, weight in self.weights.items():
            if not isinstance(weight, int) or weight <= 0:
                raise ValueError("Mu weights must be positive integers.")
            if hand_mask & ~self.unknown_mask:
                raise ValueError("A mu hand contains a tile outside the unknown pool.")
            if hand_mask.bit_count() != self.opponent_hand_size:
                raise ValueError("A mu hand has an incompatible tile count.")

    def condition_no_legal(self, left_end: int, right_end: int) -> None:
        legal_mask = _legal_mask(left_end, right_end)
        self.weights = {
            hand_mask: weight
            for hand_mask, weight in self.weights.items()
            if not hand_mask & legal_mask
        }
        self._invalidate_cache()
        self.assert_consistent()

    def _observer_known_tile(self, tile: Sequence[int]) -> None:
        bit = _tile_bit(tile)
        if not self.unknown_mask & bit:
            return
        self.weights = {
            hand_mask: weight
            for hand_mask, weight in self.weights.items()
            if not hand_mask & bit
        }
        self.unknown_mask &= ~bit
        self._invalidate_cache()
        self.assert_consistent()

    def observer_known_draw(self, tile: Sequence[int]) -> None:
        self._observer_known_tile(tile)

    def observer_known_play(self, tile: Sequence[int]) -> None:
        self._observer_known_tile(tile)

    def opponent_reveals_and_plays(self, tile: Sequence[int]) -> None:
        bit = _tile_bit(tile)
        if not self.unknown_mask & bit:
            raise ValueError(
                f"Revealed opponent tile {_normalize_tile(tile)} is not unknown."
            )

        new_weights: dict[int, int] = defaultdict(int)
        for hand_mask, weight in self.weights.items():
            if hand_mask & bit:
                new_weights[hand_mask ^ bit] += weight

        self.weights = dict(new_weights)
        self.unknown_mask &= ~bit
        self.opponent_hand_size -= 1
        self._invalidate_cache()
        self.assert_consistent()

    def opponent_hidden_draw(self) -> "MuOpponentBelief":
        stock_size = self.unknown_mask.bit_count() - self.opponent_hand_size
        if stock_size <= 0:
            raise ValueError("Opponent cannot draw because the hidden stock is empty.")

        new_weights: dict[int, int] = defaultdict(int)
        for hand_mask, weight in self.weights.items():
            stock_mask = self.unknown_mask & ~hand_mask
            for bit in _bits_from_mask(stock_mask):
                new_weights[hand_mask | bit] += weight

        self.weights = dict(new_weights)
        self.opponent_hand_size += 1
        self._invalidate_cache()
        self.assert_consistent()
        return self

    def suit_probabilities(self) -> list[float]:
        if self._probability_cache is None:
            total_weight = self.total_weight
            self._probability_cache = tuple(
                sum(
                    weight
                    for hand_mask, weight in self.weights.items()
                    if hand_mask & suit_mask
                )
                / total_weight
                for suit_mask in SUIT_MASKS
            )
        return list(self._probability_cache)

    def probability_can_play(self, ends: Sequence[int]) -> float:
        if not ends:
            return 1.0
        legal_mask = _legal_mask(ends[0], ends[1])
        playable_weight = sum(
            weight
            for hand_mask, weight in self.weights.items()
            if hand_mask & legal_mask
        )
        return playable_weight / self.total_weight

    @property
    def total_weight(self) -> int:
        return sum(self.weights.values())

    @property
    def state_count(self) -> int:
        return len(self.weights)

    @property
    def raw_hand_upper_bound(self) -> int:
        return _raw_hand_upper_bound(self.unknown_mask, self.opponent_hand_size)


class SlotOpponentBelief:
    """Exact posterior over canonical temporal-slot domain profiles.

    ``profiles`` maps a sorted tuple of allowed tile masks to an integer history
    weight. Slots are injective: two slots can never receive the same tile.
    Assignment totals use a slot-occupancy DP, while conversion to ``mu(H)``
    uses the required partial-hand-mask DP and merges equal hands after every
    slot.
    """

    mode = "slots_exact"

    def __init__(
        self,
        observer_initial_hand: Sequence[Sequence[int]],
        opponent_hand_size: int = INITIAL_HAND_SIZE,
    ):
        own_initial_mask = mask_from_tiles(observer_initial_hand)
        self.unknown_mask = ALL_MASK & ~own_initial_mask
        self.opponent_hand_size = int(opponent_hand_size)
        initial_profile = tuple([self.unknown_mask] * self.opponent_hand_size)
        self.profiles: dict[tuple[int, ...], int] = {initial_profile: 1}
        self._assignment_count_cache: dict[tuple[int, ...], int] = {}
        self._hand_weights_cache: dict[tuple[int, ...], dict[int, int]] = {}
        self._probability_cache: tuple[float, ...] | None = None
        self.assert_consistent()

    @classmethod
    def from_profiles(
        cls,
        *,
        unknown_mask: int,
        opponent_hand_size: int,
        profiles: dict[tuple[int, ...], int],
    ) -> "SlotOpponentBelief":
        """Build a small exact slot state for tests and controlled conversions."""
        belief = cls.__new__(cls)
        belief.unknown_mask = int(unknown_mask)
        belief.opponent_hand_size = int(opponent_hand_size)
        canonical_profiles: dict[tuple[int, ...], int] = defaultdict(int)
        for profile, weight in profiles.items():
            canonical = tuple(sorted(int(mask) for mask in profile))
            canonical_profiles[canonical] += int(weight)
        belief.profiles = dict(canonical_profiles)
        belief._assignment_count_cache = {}
        belief._hand_weights_cache = {}
        belief._probability_cache = None
        belief.assert_consistent()
        return belief

    def _invalidate_caches(self) -> None:
        self._assignment_count_cache.clear()
        self._hand_weights_cache.clear()
        self._probability_cache = None

    def _count_profile_assignments(self, profile: tuple[int, ...]) -> int:
        """Count injective assignments without materializing hidden hands.

        The DP scans unknown tiles and tracks which labelled slot positions are
        occupied. Its state space is ``2**h`` rather than the set of possible
        hand masks, which keeps slot-mode probability queries compact.
        """
        profile = tuple(sorted(profile))
        cached = self._assignment_count_cache.get(profile)
        if cached is not None:
            return cached
        if not profile:
            return 1
        if any(mask == 0 for mask in profile):
            return 0

        slot_count = len(profile)
        full_slot_mask = (1 << slot_count) - 1
        assignment_counts = [0] * (1 << slot_count)
        assignment_counts[0] = 1

        tile_union = 0
        for domain in profile:
            tile_union |= domain

        for tile_bit in _bits_from_mask(tile_union):
            eligible_slots = 0
            for slot_index, domain in enumerate(profile):
                if domain & tile_bit:
                    eligible_slots |= 1 << slot_index

            next_counts = assignment_counts.copy()
            for occupied_slots, count in enumerate(assignment_counts):
                if not count:
                    continue
                available_slots = eligible_slots & ~occupied_slots
                while available_slots:
                    slot_bit = available_slots & -available_slots
                    next_counts[occupied_slots | slot_bit] += count
                    available_slots ^= slot_bit
            assignment_counts = next_counts

        total = assignment_counts[full_slot_mask]
        self._assignment_count_cache[profile] = total
        return total

    def _profile_hand_weights(self, profile: tuple[int, ...]) -> dict[int, int]:
        """Return hand weights for one profile using incremental DP merging."""
        profile = tuple(sorted(profile, key=lambda mask: (mask.bit_count(), mask)))
        cached = self._hand_weights_cache.get(profile)
        if cached is not None:
            return dict(cached)

        partial: dict[int, int] = {0: 1}
        for allowed_mask in profile:
            next_partial: dict[int, int] = defaultdict(int)
            for hand_mask, count in partial.items():
                available = allowed_mask & self.unknown_mask & ~hand_mask
                for tile_bit in _bits_from_mask(available):
                    next_partial[hand_mask | tile_bit] += count
            partial = dict(next_partial)
            if not partial:
                break

        self._hand_weights_cache[profile] = dict(partial)
        return partial

    def assert_consistent(self) -> None:
        """Validate profile shape, weights, domains, and injective feasibility."""
        if not self.profiles:
            raise ValueError("Slot belief has no compatible profiles.")
        if self.opponent_hand_size < 0:
            raise ValueError("Opponent hand size cannot be negative.")

        for profile, weight in self.profiles.items():
            if tuple(sorted(profile)) != profile:
                raise ValueError("Slot profiles must be canonical sorted tuples.")
            if len(profile) != self.opponent_hand_size:
                raise ValueError("A slot profile has an incompatible slot count.")
            if not isinstance(weight, int) or weight <= 0:
                raise ValueError("Slot profile weights must be positive integers.")
            if any(mask == 0 for mask in profile):
                raise ValueError("A slot domain cannot be empty.")
            if any(mask & ~self.unknown_mask for mask in profile):
                raise ValueError("A slot domain contains a tile outside the unknown pool.")
            if self._count_profile_assignments(profile) <= 0:
                raise ValueError("A slot profile has no injective assignment.")

    def _replace_profiles(
        self,
        profiles: dict[tuple[int, ...], int],
        *,
        unknown_mask: int | None = None,
        opponent_hand_size: int | None = None,
    ) -> None:
        if unknown_mask is not None:
            self.unknown_mask = int(unknown_mask)
        if opponent_hand_size is not None:
            self.opponent_hand_size = int(opponent_hand_size)
        self.profiles = dict(profiles)
        self._invalidate_caches()
        self.assert_consistent()

    def condition_no_legal(self, left_end: int, right_end: int) -> None:
        legal_mask = _legal_mask(left_end, right_end)
        new_profiles: dict[tuple[int, ...], int] = defaultdict(int)

        for profile, weight in self.profiles.items():
            restricted = tuple(sorted(domain & ~legal_mask for domain in profile))
            if any(domain == 0 for domain in restricted):
                continue
            if self._count_profile_assignments(restricted) > 0:
                new_profiles[restricted] += weight

        self._replace_profiles(dict(new_profiles))

    def _observer_known_tile(self, tile: Sequence[int]) -> None:
        bit = _tile_bit(tile)
        if not self.unknown_mask & bit:
            return

        new_unknown_mask = self.unknown_mask & ~bit
        new_profiles: dict[tuple[int, ...], int] = defaultdict(int)
        for profile, weight in self.profiles.items():
            restricted = tuple(sorted(domain & ~bit for domain in profile))
            if any(domain == 0 for domain in restricted):
                continue
            if self._count_profile_assignments(restricted) > 0:
                new_profiles[restricted] += weight

        self._replace_profiles(
            dict(new_profiles),
            unknown_mask=new_unknown_mask,
        )

    def observer_known_draw(self, tile: Sequence[int]) -> None:
        self._observer_known_tile(tile)

    def observer_known_play(self, tile: Sequence[int]) -> None:
        self._observer_known_tile(tile)

    def opponent_reveals_and_plays(self, tile: Sequence[int]) -> None:
        bit = _tile_bit(tile)
        if not self.unknown_mask & bit:
            raise ValueError(
                f"Revealed opponent tile {_normalize_tile(tile)} is not unknown."
            )

        new_unknown_mask = self.unknown_mask & ~bit
        new_profiles: dict[tuple[int, ...], int] = defaultdict(int)

        for profile, profile_weight in self.profiles.items():
            domain_counts = Counter(profile)
            for domain, multiplicity in domain_counts.items():
                if not domain & bit:
                    continue
                remaining = list(profile)
                remaining.remove(domain)
                remaining = [allowed & ~bit for allowed in remaining]
                new_profile = tuple(sorted(remaining))
                if any(allowed == 0 for allowed in new_profile):
                    continue
                if self._count_profile_assignments(new_profile) > 0:
                    new_profiles[new_profile] += profile_weight * multiplicity

        self._replace_profiles(
            dict(new_profiles),
            unknown_mask=new_unknown_mask,
            opponent_hand_size=self.opponent_hand_size - 1,
        )

    def opponent_hidden_draw(self) -> "SlotOpponentBelief":
        stock_size = self.unknown_mask.bit_count() - self.opponent_hand_size
        if stock_size <= 0:
            raise ValueError("Opponent cannot draw because the hidden stock is empty.")

        new_profiles: dict[tuple[int, ...], int] = defaultdict(int)
        for profile, weight in self.profiles.items():
            new_profile = tuple(sorted((*profile, self.unknown_mask)))
            new_profiles[new_profile] += weight

        self._replace_profiles(
            dict(new_profiles),
            opponent_hand_size=self.opponent_hand_size + 1,
        )
        return self

    def suit_probabilities(self) -> list[float]:
        if self._probability_cache is None:
            denominator = self.assignment_weight
            numerators = [0] * SUIT_COUNT

            for profile, profile_weight in self.profiles.items():
                total_assignments = self._count_profile_assignments(profile)
                for suit, suit_mask in enumerate(SUIT_MASKS):
                    without_suit = tuple(
                        sorted(domain & ~suit_mask for domain in profile)
                    )
                    no_suit_assignments = self._count_profile_assignments(without_suit)
                    numerators[suit] += profile_weight * (
                        total_assignments - no_suit_assignments
                    )

            self._probability_cache = tuple(
                numerator / denominator for numerator in numerators
            )
        return list(self._probability_cache)

    def probability_can_play(self, ends: Sequence[int]) -> float:
        if not ends:
            return 1.0
        legal_mask = _legal_mask(ends[0], ends[1])
        denominator = self.assignment_weight
        playable_weight = 0

        for profile, profile_weight in self.profiles.items():
            total_assignments = self._count_profile_assignments(profile)
            without_legal = tuple(
                sorted(domain & ~legal_mask for domain in profile)
            )
            no_legal_assignments = self._count_profile_assignments(without_legal)
            playable_weight += profile_weight * (
                total_assignments - no_legal_assignments
            )

        return playable_weight / denominator

    def to_hand_weights_dp(self) -> dict[int, int]:
        """Convert profiles to exact ``mu(H)`` weights with incremental merging."""
        result: dict[int, int] = defaultdict(int)
        for profile, profile_weight in self.profiles.items():
            for hand_mask, assignment_count in self._profile_hand_weights(profile).items():
                result[hand_mask] += profile_weight * assignment_count
        if not result:
            raise ValueError("Slot-to-mu conversion produced no compatible hands.")
        return dict(result)

    @property
    def profile_count(self) -> int:
        return len(self.profiles)

    @property
    def state_count(self) -> int:
        return self.profile_count

    @property
    def assignment_weight(self) -> int:
        return sum(
            profile_weight * self._count_profile_assignments(profile)
            for profile, profile_weight in self.profiles.items()
        )

    @property
    def total_weight(self) -> int:
        return self.assignment_weight

    @property
    def raw_hand_upper_bound(self) -> int:
        return _raw_hand_upper_bound(self.unknown_mask, self.opponent_hand_size)


class HybridExactOpponentModel:
    """Persistent exact controller that switches once from slots to ``mu(H)``."""

    def __init__(
        self,
        *,
        switch_to_mu_max_hands: int = SWITCH_TO_MU_MAX_HANDS,
        trace_history_limit: int = 256,
        **legacy_options,
    ):
        if switch_to_mu_max_hands <= 0:
            raise ValueError("switch_to_mu_max_hands must be positive.")
        if trace_history_limit <= 0:
            raise ValueError("trace_history_limit must be positive.")

        # Old constructor knobs are accepted so external callers do not fail,
        # but they cannot alter the exact path or enable particle fallback.
        unsupported = set(legacy_options) - {
            "max_enumerated_hands",
            "particle_count",
            "seed",
        }
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise TypeError(f"Unexpected opponent-model options: {names}.")

        self.switch_to_mu_max_hands = int(switch_to_mu_max_hands)
        self.trace_history_limit = int(trace_history_limit)
        self._belief: SlotOpponentBelief | MuOpponentBelief | None = None
        self._game_id = None
        self._observer_player: int | None = None
        self._processed_history_length = 0
        self._own_draws_consumed = 0
        self._pending_trace: OpponentTurnTrace | None = None
        self._last_snapshot: ProbabilitySnapshot | None = None
        self._last_completed_turn_trace: OpponentTurnTrace | None = None
        self._turn_trace_history: deque[OpponentTurnTrace] = deque(
            maxlen=self.trace_history_limit
        )
        self._new_snapshots: deque[ProbabilitySnapshot] = deque(
            maxlen=self.trace_history_limit * 3
        )
        self._current_update_snapshots: list[ProbabilitySnapshot] = []
        self._current_update_traces: list[OpponentTurnTrace] = []
        self._switched_to_mu = False
        self._switched_this_update = False
        self._switch_turn: int | None = None
        self._switch_upper_bound: int | None = None
        self._switch_mu_state_count: int | None = None
        self._switch_conversion_time_ms: float | None = None

    def reset(self) -> None:
        """Clear game identity, belief state, traces, and switch metadata."""
        self._belief = None
        self._game_id = None
        self._observer_player = None
        self._processed_history_length = 0
        self._own_draws_consumed = 0
        self._pending_trace = None
        self._last_snapshot = None
        self._last_completed_turn_trace = None
        self._turn_trace_history.clear()
        self._new_snapshots.clear()
        self._current_update_snapshots = []
        self._current_update_traces = []
        self._switched_to_mu = False
        self._switched_this_update = False
        self._switch_turn = None
        self._switch_upper_bound = None
        self._switch_mu_state_count = None
        self._switch_conversion_time_ms = None

    def update(self, state: dict) -> list[float]:
        """Process new history and return the current seven probabilities."""
        return list(self.update_detailed(state).probabilities)

    def update_detailed(self, state: dict) -> OpponentModelUpdate:
        """Process new history and return probabilities plus labelled new traces."""
        self._validate_state(state)
        game_id = state.get("game_id")
        observer_player = int(
            state.get("observer_player", state.get("current_player", 0))
        )
        history = reconstruct_public_actions(state)

        if (
            self._belief is None
            or self._game_id != game_id
            or self._observer_player != observer_player
            or len(history) < self._processed_history_length
        ):
            self._reset_from_state(state)
            history = reconstruct_public_actions(state)

        own_draws = [
            _normalize_tile(tile)
            for tile in state.get("current_player_drawn_tiles", [])
        ]
        hand_sizes = [int(size) for size in state.get("hand_sizes", [])]
        history_is_terminal = bool(state.get("game_over")) or 0 in hand_sizes

        self._current_update_snapshots = []
        self._current_update_traces = []
        self._switched_this_update = False

        for history_index in range(self._processed_history_length, len(history)):
            entry = history[history_index]
            is_terminal_entry = (
                history_is_terminal
                and history_index == len(history) - 1
                and not _is_draw(entry.action)
            )
            self._process_entry(
                entry,
                observer_player,
                own_draws,
                terminal_turn=is_terminal_entry,
            )
            self._processed_history_length += 1

        probabilities = tuple(self.suit_probabilities())
        state["opponent_suit_probabilities"] = list(probabilities)
        state["opponent_model_mode"] = self.mode
        state["opponent_model_state_count"] = self.state_count
        state["opponent_model_metadata"] = {
            "model_version": MODEL_VERSION,
            "game_id": self._game_id,
            "observer_player": self._observer_player,
            "processed_history_length": self._processed_history_length,
        }

        return OpponentModelUpdate(
            probabilities=probabilities,
            new_snapshots=tuple(self._current_update_snapshots),
            completed_turn_traces=tuple(self._current_update_traces),
            mode=self.mode,
            switched_this_update=self._switched_this_update,
        )

    def _validate_state(self, state: dict) -> None:
        missing: list[str] = []
        if len(state.get("hand_sizes", [])) != 2:
            missing.append("exactly two hand sizes")
        if state.get("current_player_initial_hand") is None:
            missing.append("current_player_initial_hand")
        if state.get("current_player_drawn_tiles") is None:
            missing.append("current_player_drawn_tiles")
        if missing:
            raise ValueError(
                "Opponent model requires a complete two-player observer state. "
                "Missing or invalid: " + ", ".join(missing)
            )

    def _reset_from_state(self, state: dict) -> None:
        self.reset()
        observer_player = int(
            state.get("observer_player", state.get("current_player", 0))
        )
        self._belief = SlotOpponentBelief(
            state["current_player_initial_hand"],
            opponent_hand_size=int(
                state.get("initial_hand_size", INITIAL_HAND_SIZE)
            ),
        )
        self._game_id = state.get("game_id")
        self._observer_player = observer_player

    def _start_or_get_trace(self, entry: PublicAction) -> OpponentTurnTrace:
        if self._pending_trace is None:
            self._pending_trace = OpponentTurnTrace(
                public_turn=entry.public_turn,
                actor=entry.actor,
            )
        elif (
            self._pending_trace.public_turn != entry.public_turn
            or self._pending_trace.actor != entry.actor
        ):
            raise ValueError(
                "Public history advanced before a pending draw turn was completed."
            )
        return self._pending_trace

    def _process_entry(
        self,
        entry: PublicAction,
        observer_player: int,
        own_draws: Sequence[tuple[int, int]],
        *,
        terminal_turn: bool,
    ) -> None:
        if self._belief is None:
            raise RuntimeError("Opponent belief is not initialized.")

        trace = self._start_or_get_trace(entry)
        action = entry.action
        actor_is_opponent = entry.actor != observer_player

        if _is_draw(action):
            if actor_is_opponent:
                if entry.ends_before is not None:
                    self._belief.condition_no_legal(*entry.ends_before)
                    trace.after_negative_evidence = self._record_snapshot(
                        entry,
                        ProbabilityStage.AFTER_NEGATIVE_EVIDENCE,
                        entry.ends_before,
                    )

                self._belief.opponent_hidden_draw()
                trace.after_draw = self._record_snapshot(
                    entry,
                    ProbabilityStage.AFTER_DRAW,
                    entry.ends_after,
                )
            else:
                if self._own_draws_consumed >= len(own_draws):
                    raise ValueError("Missing private identity for an observer draw.")
                self._belief.observer_known_draw(
                    own_draws[self._own_draws_consumed]
                )
                self._own_draws_consumed += 1
            return

        same_as_previous = False
        if actor_is_opponent:
            if _is_pass(action):
                if entry.ends_before is not None:
                    self._belief.condition_no_legal(*entry.ends_before)
                    if trace.after_negative_evidence is None:
                        trace.after_negative_evidence = self._record_snapshot(
                            entry,
                            ProbabilityStage.AFTER_NEGATIVE_EVIDENCE,
                            entry.ends_before,
                        )
                        same_as_previous = True
            elif _is_tile_play(action):
                self._belief.opponent_reveals_and_plays(action[0])
        elif _is_tile_play(action):
            self._belief.observer_known_play(action[0])

        trace.end_turn = self._record_snapshot(
            entry,
            ProbabilityStage.END_TURN,
            entry.ends_after,
            same_as_previous=same_as_previous,
        )
        self._complete_turn_trace(trace, terminal_turn=terminal_turn)

    def _record_snapshot(
        self,
        entry: PublicAction,
        stage: ProbabilityStage,
        ends: tuple[int, int] | None,
        *,
        same_as_previous: bool = False,
    ) -> ProbabilitySnapshot:
        if self._belief is None:
            raise RuntimeError("Opponent belief is not initialized.")

        snapshot = ProbabilitySnapshot(
            public_turn=entry.public_turn,
            history_action_index=entry.history_action_index,
            actor=entry.actor,
            stage=stage,
            probabilities=tuple(self._belief.suit_probabilities()),
            mode=self.mode,
            ends=ends,
            opponent_hand_size=self.opponent_hand_size,
            unknown_count=self.unknown_count,
            state_count=self.state_count,
            profile_count=self.profile_count,
            mu_hand_count=self.mu_hand_count,
            total_weight=self.total_weight,
            raw_hand_upper_bound=self._belief.raw_hand_upper_bound,
            action=entry.action,
            same_as_previous=same_as_previous,
        )
        self._last_snapshot = snapshot
        self._new_snapshots.append(snapshot)
        self._current_update_snapshots.append(snapshot)
        return snapshot

    def _complete_turn_trace(
        self,
        trace: OpponentTurnTrace,
        *,
        terminal_turn: bool,
    ) -> None:
        if trace.end_turn is None:
            raise RuntimeError("Cannot complete a public turn without END_TURN.")
        self._last_completed_turn_trace = trace
        self._turn_trace_history.append(trace)
        self._current_update_traces.append(trace)
        self._pending_trace = None
        self._maybe_switch_to_mu(trace.public_turn, terminal_turn=terminal_turn)

    def _maybe_switch_to_mu(self, public_turn: int, *, terminal_turn: bool) -> None:
        if (
            terminal_turn
            or self._switched_to_mu
            or not isinstance(self._belief, SlotOpponentBelief)
        ):
            return

        upper_bound = self._belief.raw_hand_upper_bound
        if upper_bound > self.switch_to_mu_max_hands:
            return

        slot_probabilities = self._belief.suit_probabilities()
        started = perf_counter()
        weights = self._belief.to_hand_weights_dp()
        mu_belief = MuOpponentBelief.from_weights(
            unknown_mask=self._belief.unknown_mask,
            opponent_hand_size=self._belief.opponent_hand_size,
            weights=weights,
        )
        conversion_time_ms = (perf_counter() - started) * 1000.0

        mu_probabilities = mu_belief.suit_probabilities()
        if any(
            abs(slot_value - mu_value) > 1e-12
            for slot_value, mu_value in zip(slot_probabilities, mu_probabilities)
        ):
            raise AssertionError("Slot-to-mu conversion changed exact probabilities.")

        self._belief = mu_belief
        self._switched_to_mu = True
        self._switched_this_update = True
        self._switch_turn = int(public_turn)
        self._switch_upper_bound = int(upper_bound)
        self._switch_mu_state_count = len(weights)
        self._switch_conversion_time_ms = conversion_time_ms

    def suit_probabilities(self) -> list[float]:
        if self._belief is None:
            return [0.0] * SUIT_COUNT
        return self._belief.suit_probabilities()

    def probability_can_play(self, ends: Sequence[int]) -> float:
        """Return exact response probability from the joint hand posterior."""
        if self._belief is None:
            return 1.0
        return self._belief.probability_can_play(ends)

    def consume_new_snapshots(self) -> list[ProbabilitySnapshot]:
        """Return and clear snapshots not previously consumed by a UI/logger."""
        snapshots = list(self._new_snapshots)
        self._new_snapshots.clear()
        return snapshots

    @property
    def mode(self) -> str:
        return "uninitialized" if self._belief is None else self._belief.mode

    @property
    def state_count(self) -> int:
        return 0 if self._belief is None else self._belief.state_count

    @property
    def profile_count(self) -> int | None:
        if isinstance(self._belief, SlotOpponentBelief):
            return self._belief.profile_count
        return None

    @property
    def mu_hand_count(self) -> int | None:
        if isinstance(self._belief, MuOpponentBelief):
            return self._belief.state_count
        return None

    @property
    def unknown_count(self) -> int:
        return 0 if self._belief is None else self._belief.unknown_mask.bit_count()

    @property
    def opponent_hand_size(self) -> int:
        return 0 if self._belief is None else self._belief.opponent_hand_size

    @property
    def total_weight(self) -> int:
        return 0 if self._belief is None else self._belief.total_weight

    @property
    def switched_to_mu(self) -> bool:
        return self._switched_to_mu

    @property
    def switch_turn(self) -> int | None:
        return self._switch_turn

    @property
    def switch_upper_bound(self) -> int | None:
        return self._switch_upper_bound

    @property
    def switch_mu_state_count(self) -> int | None:
        return self._switch_mu_state_count

    @property
    def switch_conversion_time_ms(self) -> float | None:
        return self._switch_conversion_time_ms

    @property
    def last_snapshot(self) -> ProbabilitySnapshot | None:
        return self._last_snapshot

    @property
    def last_completed_turn_trace(self) -> OpponentTurnTrace | None:
        return self._last_completed_turn_trace

    @property
    def turn_trace_history(self) -> list[OpponentTurnTrace]:
        return list(self._turn_trace_history)

# Stable public names used by agents, UI code, and older imports.
ExactOpponentModel = HybridExactOpponentModel
HybridOpponentModel = HybridExactOpponentModel
EnumeratedOpponentBelief = MuOpponentBelief
CompactOpponentBelief = SlotOpponentBelief


def compute_opponent_suit_probabilities(state: dict) -> list[float]:
    """Replay one observer state through a fresh exact hybrid model.

    Persistent agents should keep one ``ExactOpponentModel`` per observer and
    game. This one-shot wrapper intentionally ignores probability values already
    stored in ``state`` so stale output cannot suppress new history processing.
    """
    model = HybridExactOpponentModel()
    return model.update(state)


def approximate_response_probability_from_marginals(
    probabilities: Sequence[float],
    ends: Sequence[int],
) -> float:
    """Approximate response chance assuming independent suit marginals."""
    if not ends:
        return 1.0

    left, right = int(ends[0]), int(ends[1])
    if left == right:
        return float(probabilities[left])

    return (
        1.0
        - (1.0 - float(probabilities[left]))
        * (1.0 - float(probabilities[right]))
    )


def response_probability_from_marginals(
    probabilities: Sequence[float],
    ends: Sequence[int],
) -> float:
    """Deprecated wrapper for the explicitly approximate marginal formula."""
    return approximate_response_probability_from_marginals(probabilities, ends)
