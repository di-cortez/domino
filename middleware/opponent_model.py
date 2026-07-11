"""Hybrid two-player opponent-suit probabilities.

The model exports seven presence probabilities:

    p[j] = P(the opponent currently has at least one tile containing suit j).

Semantics are direct:

    0.0 = known absence;
    1.0 = known presence.

The implementation has three internal modes:

* ``compact_exact``: exact uniform posterior over all h-subsets of a candidate
  tile mask. No hands are enumerated.
* ``enumerated_exact``: exact weighted posterior over at most
  ``MAX_ENUMERATED_HANDS`` explicit hands.
* ``particle_approximate``: fixed-size particle approximation when an exact
  hidden-draw expansion would exceed the configured hand limit.

The model is strict: it supports temporal reconstruction only for two-player
observer states containing the observer's initial hand and private draw history.
There is no snapshot fallback.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import blake2b
from itertools import combinations
from math import comb
import random
from typing import Iterable, Sequence


MAX_ENUMERATED_HANDS = 1000
PARTICLE_COUNT = 1000
PARTICLE_SEED = 0

MODEL_VERSION = "hybrid-particles-rejuvenation-v2"

ALL_TILES = [(i, j) for i in range(7) for j in range(i, 7)]
SUIT_COUNT = 7
INITIAL_HAND_SIZE = 7

TILE_TO_INDEX = {tile: index for index, tile in enumerate(ALL_TILES)}
INDEX_TO_TILE = {index: tile for tile, index in TILE_TO_INDEX.items()}
TILE_BITS = [1 << index for index in range(len(ALL_TILES))]
ALL_MASK = (1 << len(ALL_TILES)) - 1

SUIT_MASKS: list[int] = []
for suit in range(SUIT_COUNT):
    mask = 0
    for tile, index in TILE_TO_INDEX.items():
        if suit in tile:
            mask |= 1 << index
    SUIT_MASKS.append(mask)


class ParticleDepletionError(RuntimeError):
    """Raised when no particle survives a logically required observation."""


@dataclass(frozen=True)
class PublicAction:
    """A board-history action annotated with its reconstructed actor."""

    actor: int
    action: object
    ends_before: tuple[int, int] | None


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


def reconstruct_public_actions(state: dict) -> list[PublicAction]:
    """Reconstruct actors and pre-action ends for public board history.

    The engine history stores actions but not actors. Tile plays and passes
    advance the actor, while a draw keeps the same actor.
    """
    board_history = [
        _normalize_action(action)
        for action in state.get("board_history", [])
    ]
    player_count = len(state.get("hand_sizes", [])) or 2
    current_player = int(
        state.get("history_current_player", state.get("current_player", 0))
    )

    advancing_actions = sum(
        1 for action in board_history if not _is_draw(action)
    )
    actor = (current_player - advancing_actions) % player_count

    ends: list[int] | None = None
    annotated: list[PublicAction] = []

    for action in board_history:
        ends_before = tuple(ends) if ends is not None else None
        annotated.append(
            PublicAction(actor=actor, action=action, ends_before=ends_before)
        )

        if _is_draw(action):
            continue

        if _is_pass(action):
            actor = (actor + 1) % player_count
            continue

        tile, side = action
        if ends is None:
            ends = [tile[0], tile[1]]
        else:
            connected_value = ends[side]
            ends[side] = tile[1] if tile[0] == connected_value else tile[0]
        actor = (actor + 1) % player_count

    return annotated


def _stable_seed(
    base_seed: int,
    game_id,
    observer_player: int,
    observer_initial_hand: Sequence[Sequence[int]],
) -> int:
    """Build a reproducible per-game, per-observer RNG seed."""
    initial_mask = mask_from_tiles(observer_initial_hand)
    payload = (
        f"{int(base_seed)}|{game_id!r}|{int(observer_player)}|{initial_mask}"
    ).encode("utf-8")
    return int.from_bytes(
        blake2b(payload, digest_size=8).digest(),
        byteorder="big",
        signed=False,
    )


def _presence_probabilities_from_uniform_subsets(
    candidate_mask: int,
    hand_size: int,
) -> list[float]:
    """Exact suit probabilities for a uniform h-subset of candidate_mask."""
    candidate_count = candidate_mask.bit_count()
    hand_size = int(hand_size)

    if hand_size <= 0:
        return [0.0] * SUIT_COUNT
    if candidate_count < hand_size:
        raise ValueError("Candidate pool is smaller than the opponent hand.")

    denominator = comb(candidate_count, hand_size)
    probabilities: list[float] = []

    for suit_mask in SUIT_MASKS:
        suit_count = (candidate_mask & suit_mask).bit_count()
        non_suit_count = candidate_count - suit_count
        if non_suit_count < hand_size:
            probability_no_suit = 0.0
        else:
            probability_no_suit = comb(non_suit_count, hand_size) / denominator
        probabilities.append(1.0 - probability_no_suit)

    return probabilities


def _probability_can_play_uniform(
    candidate_mask: int,
    hand_size: int,
    ends: Sequence[int],
) -> float:
    """Exact play probability for a uniform h-subset of candidate_mask."""
    if not ends:
        return 1.0

    legal_mask = SUIT_MASKS[int(ends[0])] | SUIT_MASKS[int(ends[1])]
    candidate_count = candidate_mask.bit_count()
    playable_count = (candidate_mask & legal_mask).bit_count()
    non_playable_count = candidate_count - playable_count

    if hand_size <= 0:
        return 0.0
    if candidate_count < hand_size:
        raise ValueError("Candidate pool is smaller than the opponent hand.")

    denominator = comb(candidate_count, hand_size)
    if non_playable_count < hand_size:
        probability_no_play = 0.0
    else:
        probability_no_play = comb(non_playable_count, hand_size) / denominator
    return 1.0 - probability_no_play


class CompactOpponentBelief:
    """Exact uniform posterior over h-subsets of ``candidate_mask``."""

    mode = "compact_exact"

    def __init__(
        self,
        observer_initial_hand: Sequence[Sequence[int]],
        opponent_hand_size: int = INITIAL_HAND_SIZE,
        *,
        rng: random.Random,
        max_enumerated_hands: int,
        particle_count: int,
    ):
        own_initial_mask = mask_from_tiles(observer_initial_hand)
        self.unknown_mask = ALL_MASK ^ own_initial_mask
        self.candidate_mask = self.unknown_mask
        self.opponent_hand_size = int(opponent_hand_size)
        self.rng = rng
        self.max_enumerated_hands = int(max_enumerated_hands)
        self.particle_count = int(particle_count)
        self._probability_cache: list[float] | None = None
        self._assert_consistent()

    def _invalidate_cache(self) -> None:
        self._probability_cache = None

    def _assert_consistent(self) -> None:
        if self.candidate_mask & ~self.unknown_mask:
            raise ValueError("Candidate mask is not contained in the unknown mask.")
        if self.opponent_hand_size < 0:
            raise ValueError("Opponent hand size cannot be negative.")
        if self.candidate_mask.bit_count() < self.opponent_hand_size:
            raise ValueError("Not enough candidate tiles for the opponent hand.")

    def condition_no_legal(self, left_end: int, right_end: int) -> None:
        legal_mask = SUIT_MASKS[int(left_end)] | SUIT_MASKS[int(right_end)]
        self.candidate_mask &= ~legal_mask
        self._invalidate_cache()
        self._assert_consistent()

    def observer_known_draw(self, tile: Sequence[int]) -> None:
        bit = _tile_bit(tile)
        if not (self.unknown_mask & bit):
            return
        self.candidate_mask &= ~bit
        self.unknown_mask &= ~bit
        self._invalidate_cache()
        self._assert_consistent()

    def observer_known_play(self, tile: Sequence[int]) -> None:
        bit = _tile_bit(tile)
        if not (self.unknown_mask & bit):
            return
        self.candidate_mask &= ~bit
        self.unknown_mask &= ~bit
        self._invalidate_cache()
        self._assert_consistent()

    def opponent_reveals_and_plays(self, tile: Sequence[int]) -> None:
        bit = _tile_bit(tile)
        if not (self.candidate_mask & bit):
            raise ValueError(
                f"Opponent revealed impossible tile {_normalize_tile(tile)}."
            )
        self.candidate_mask &= ~bit
        self.unknown_mask &= ~bit
        self.opponent_hand_size -= 1
        self._invalidate_cache()
        self._assert_consistent()

    def opponent_hidden_draw(self):
        """Return an exact-enumerated or particle belief after one hidden draw."""
        candidate_count = self.candidate_mask.bit_count()
        unknown_count = self.unknown_mask.bit_count()
        hand_size = self.opponent_hand_size
        outside_count = unknown_count - candidate_count

        if unknown_count - hand_size <= 0:
            raise ValueError("Opponent cannot draw because the hidden stock is empty.")

        inside_only_count = (
            comb(candidate_count, hand_size + 1)
            if candidate_count >= hand_size + 1
            else 0
        )
        one_outside_count = comb(candidate_count, hand_size) * outside_count
        final_state_count = inside_only_count + one_outside_count

        if final_state_count <= self.max_enumerated_hands:
            return EnumeratedOpponentBelief.from_compact_hidden_draw(
                self,
                final_state_count=final_state_count,
            )
        return ParticleOpponentBelief.from_compact_hidden_draw(
            self,
            final_state_count=final_state_count,
        )

    def suit_probabilities(self) -> list[float]:
        if self._probability_cache is None:
            self._probability_cache = _presence_probabilities_from_uniform_subsets(
                self.candidate_mask, self.opponent_hand_size
            )
        return list(self._probability_cache)

    def probability_can_play(self, ends: Sequence[int]) -> float:
        return _probability_can_play_uniform(
            self.candidate_mask, self.opponent_hand_size, ends
        )

    @property
    def state_count(self) -> int:
        return comb(self.candidate_mask.bit_count(), self.opponent_hand_size)


class EnumeratedOpponentBelief:
    """Exact weighted posterior over explicitly represented hidden hands."""

    mode = "enumerated_exact"

    def __init__(
        self,
        *,
        unknown_mask: int,
        opponent_hand_size: int,
        weights: dict[int, int],
        rng: random.Random,
        max_enumerated_hands: int,
        particle_count: int,
        compact_hidden_draw_final_state_count: int | None = None,
    ):
        self.unknown_mask = int(unknown_mask)
        self.opponent_hand_size = int(opponent_hand_size)
        self.weights = dict(weights)
        self.rng = rng
        self.max_enumerated_hands = int(max_enumerated_hands)
        self.particle_count = int(particle_count)
        self.compact_hidden_draw_final_state_count = compact_hidden_draw_final_state_count
        self._probability_cache: list[float] | None = None
        self._assert_consistent()

    @classmethod
    def from_compact_hidden_draw(
        cls,
        compact: CompactOpponentBelief,
        *,
        final_state_count: int | None = None,
    ):
        candidate_indices = list(_indices_from_mask(compact.candidate_mask))
        outside_mask = compact.unknown_mask & ~compact.candidate_mask
        outside_indices = list(_indices_from_mask(outside_mask))
        hand_size = compact.opponent_hand_size
        weights: dict[int, int] = defaultdict(int)

        # If the drawn tile was also in candidate_mask, each final hand of
        # size h+1 has h+1 compatible histories.
        if len(candidate_indices) >= hand_size + 1:
            for combo in combinations(candidate_indices, hand_size + 1):
                final_hand = _mask_from_indices(combo)
                weights[final_hand] += hand_size + 1

        # If the drawn tile was known to be in the stock, the history is unique.
        if len(candidate_indices) >= hand_size:
            for combo in combinations(candidate_indices, hand_size):
                original_hand = _mask_from_indices(combo)
                for outside_index in outside_indices:
                    weights[original_hand | (1 << outside_index)] += 1

        return cls(
            unknown_mask=compact.unknown_mask,
            opponent_hand_size=hand_size + 1,
            weights=dict(weights),
            rng=compact.rng,
            max_enumerated_hands=compact.max_enumerated_hands,
            particle_count=compact.particle_count,
            compact_hidden_draw_final_state_count=final_state_count,
        )

    def _invalidate_cache(self) -> None:
        self._probability_cache = None

    def _assert_consistent(self) -> None:
        if not self.weights:
            raise ValueError("Enumerated model has no compatible hidden hands.")
        for hand_mask, weight in self.weights.items():
            if not isinstance(weight, int) or weight <= 0:
                raise ValueError("Enumerated weights must be positive integers.")
            if hand_mask & ~self.unknown_mask:
                raise ValueError("A hidden hand contains a tile outside the unknown pool.")
            if hand_mask.bit_count() != self.opponent_hand_size:
                raise ValueError("A hidden hand has an incompatible tile count.")

    def _filter(self, predicate) -> None:
        self.weights = {
            hand_mask: weight
            for hand_mask, weight in self.weights.items()
            if predicate(hand_mask)
        }
        self._invalidate_cache()
        self._assert_consistent()

    def condition_no_legal(self, left_end: int, right_end: int) -> None:
        legal_mask = SUIT_MASKS[int(left_end)] | SUIT_MASKS[int(right_end)]
        self._filter(lambda hand_mask: (hand_mask & legal_mask) == 0)

    def observer_known_draw(self, tile: Sequence[int]) -> None:
        bit = _tile_bit(tile)
        if not (self.unknown_mask & bit):
            return
        self._filter(lambda hand_mask: (hand_mask & bit) == 0)
        self.unknown_mask &= ~bit
        self._invalidate_cache()
        self._assert_consistent()

    def observer_known_play(self, tile: Sequence[int]) -> None:
        bit = _tile_bit(tile)
        if not (self.unknown_mask & bit):
            return
        self._filter(lambda hand_mask: (hand_mask & bit) == 0)
        self.unknown_mask &= ~bit
        self._invalidate_cache()
        self._assert_consistent()

    def opponent_reveals_and_plays(self, tile: Sequence[int]) -> None:
        bit = _tile_bit(tile)
        if not (self.unknown_mask & bit):
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
        self._assert_consistent()

    def opponent_hidden_draw(self):
        stock_size = self.unknown_mask.bit_count() - self.opponent_hand_size
        if stock_size <= 0:
            raise ValueError("Opponent cannot draw because the hidden stock is empty.")

        new_weights: dict[int, int] = defaultdict(int)
        for hand_mask, weight in self.weights.items():
            stock_mask = self.unknown_mask & ~hand_mask
            for index in _indices_from_mask(stock_mask):
                new_weights[hand_mask | (1 << index)] += weight
                if len(new_weights) > self.max_enumerated_hands:
                    particle_belief = ParticleOpponentBelief.from_enumerated(self)
                    return particle_belief.opponent_hidden_draw()

        self.weights = dict(new_weights)
        self.opponent_hand_size += 1
        self._invalidate_cache()
        self._assert_consistent()
        return self

    def suit_probabilities(self) -> list[float]:
        if self._probability_cache is not None:
            return list(self._probability_cache)
        total_weight = sum(self.weights.values())
        probabilities = []
        for suit_mask in SUIT_MASKS:
            numerator = sum(
                weight
                for hand_mask, weight in self.weights.items()
                if hand_mask & suit_mask
            )
            probabilities.append(numerator / total_weight)
        self._probability_cache = probabilities
        return list(probabilities)

    def probability_can_play(self, ends: Sequence[int]) -> float:
        if not ends:
            return 1.0
        legal_mask = SUIT_MASKS[int(ends[0])] | SUIT_MASKS[int(ends[1])]
        total_weight = sum(self.weights.values())
        playable_weight = sum(
            weight
            for hand_mask, weight in self.weights.items()
            if hand_mask & legal_mask
        )
        return playable_weight / total_weight

    @property
    def state_count(self) -> int:
        return len(self.weights)


class ParticleOpponentBelief:
    """Fixed-size sequential Monte Carlo approximation of hidden hands."""

    mode = "particle_approximate"

    def __init__(
        self,
        *,
        unknown_mask: int,
        opponent_hand_size: int,
        particles: Sequence[int],
        rng: random.Random,
        max_enumerated_hands: int,
        particle_count: int,
        compact_hidden_draw_final_state_count: int | None = None,
    ):
        self.unknown_mask = int(unknown_mask)
        self.opponent_hand_size = int(opponent_hand_size)
        self.particles = list(particles)
        self.rng = rng
        self.max_enumerated_hands = int(max_enumerated_hands)
        self.particle_count = int(particle_count)
        self.compact_hidden_draw_final_state_count = compact_hidden_draw_final_state_count
        self._probability_cache: list[float] | None = None
        self._assert_consistent()

    @classmethod
    def from_compact_hidden_draw(
        cls,
        compact: CompactOpponentBelief,
        *,
        final_state_count: int | None = None,
    ):
        candidate_indices = list(_indices_from_mask(compact.candidate_mask))
        particles: list[int] = []

        for _ in range(compact.particle_count):
            sampled_indices = compact.rng.sample(
                candidate_indices, compact.opponent_hand_size
            )
            hand_mask = _mask_from_indices(sampled_indices)
            stock_indices = list(
                _indices_from_mask(compact.unknown_mask & ~hand_mask)
            )
            if not stock_indices:
                raise ValueError("Opponent cannot draw because the hidden stock is empty.")
            drawn_index = compact.rng.choice(stock_indices)
            particles.append(hand_mask | (1 << drawn_index))

        return cls(
            unknown_mask=compact.unknown_mask,
            opponent_hand_size=compact.opponent_hand_size + 1,
            particles=particles,
            rng=compact.rng,
            max_enumerated_hands=compact.max_enumerated_hands,
            particle_count=compact.particle_count,
            compact_hidden_draw_final_state_count=final_state_count,
        )

    @classmethod
    def from_enumerated(cls, enumerated: EnumeratedOpponentBelief):
        hands = list(enumerated.weights)
        weights = list(enumerated.weights.values())
        particles = enumerated.rng.choices(
            hands, weights=weights, k=enumerated.particle_count
        )
        return cls(
            unknown_mask=enumerated.unknown_mask,
            opponent_hand_size=enumerated.opponent_hand_size,
            particles=particles,
            rng=enumerated.rng,
            max_enumerated_hands=enumerated.max_enumerated_hands,
            particle_count=enumerated.particle_count,
        )

    def _invalidate_cache(self) -> None:
        self._probability_cache = None

    def _assert_consistent(self) -> None:
        if not self.particles:
            raise ParticleDepletionError("Particle model has no particles.")
        if len(self.particles) != self.particle_count:
            raise ValueError("Particle population has an unexpected size.")
        for hand_mask in self.particles:
            if hand_mask & ~self.unknown_mask:
                raise ValueError("A particle contains a tile outside the unknown pool.")
            if hand_mask.bit_count() != self.opponent_hand_size:
                raise ValueError("A particle has an incompatible tile count.")

    def _filter_and_resample(
        self,
        predicate,
        transform=None,
        *,
        observation: str,
        repair=None,
    ) -> None:
        """Condition particles on an observation without needless collapse.

        Every surviving particle is retained once. Sampling with replacement is
        used only to fill the population back to ``particle_count``. If finite
        particle support is accidentally exhausted, ``repair`` performs a local
        rejuvenation move that creates hands compatible with the hard public
        observation instead of aborting the game.
        """
        survivors: list[int] = []
        for hand_mask in self.particles:
            if predicate(hand_mask):
                survivors.append(
                    transform(hand_mask) if transform is not None else hand_mask
                )

        if not survivors:
            if repair is None:
                raise ParticleDepletionError(
                    f"No particle survived observation {observation!r}. "
                    "No compatible rejuvenation rule is available."
                )
            survivors = [repair(hand_mask) for hand_mask in self.particles]

        # Preserve all support that survived. The previous implementation drew
        # all particles afresh from ``survivors`` and therefore discarded much
        # of the remaining diversity after every observation.
        if len(survivors) >= self.particle_count:
            self.particles = self.rng.sample(
                survivors,
                self.particle_count,
            )
        else:
            self.particles = list(survivors)
            self.particles.extend(
                self.rng.choices(
                    survivors,
                    k=self.particle_count - len(survivors),
                )
            )
            self.rng.shuffle(self.particles)

        self._invalidate_cache()
        self._assert_consistent()

    def condition_no_legal(self, left_end: int, right_end: int) -> None:
        legal_mask = SUIT_MASKS[int(left_end)] | SUIT_MASKS[int(right_end)]

        def repair(hand_mask: int) -> int:
            repaired = hand_mask & ~legal_mask
            missing_count = self.opponent_hand_size - repaired.bit_count()
            replacement_mask = self.unknown_mask & ~legal_mask & ~repaired
            replacement_indices = list(_indices_from_mask(replacement_mask))
            if len(replacement_indices) < missing_count:
                raise ParticleDepletionError(
                    "The public no-legal observation is incompatible with the "
                    "remaining unknown tiles."
                )
            additions = self.rng.sample(replacement_indices, missing_count)
            return repaired | _mask_from_indices(additions)

        self._filter_and_resample(
            lambda hand_mask: (hand_mask & legal_mask) == 0,
            observation=f"no legal tile on ({left_end}, {right_end})",
            repair=repair,
        )

    def observer_known_draw(self, tile: Sequence[int]) -> None:
        bit = _tile_bit(tile)
        if not (self.unknown_mask & bit):
            return
        normalized = _normalize_tile(tile)

        def repair(hand_mask: int) -> int:
            if not (hand_mask & bit):
                return hand_mask
            stock_mask = self.unknown_mask & ~hand_mask
            stock_indices = list(_indices_from_mask(stock_mask))
            if not stock_indices:
                raise ParticleDepletionError(
                    f"Cannot condition on observer draw {normalized}: hidden "
                    "stock is empty in every particle."
                )
            replacement_index = self.rng.choice(stock_indices)
            return (hand_mask ^ bit) | (1 << replacement_index)

        self._filter_and_resample(
            lambda hand_mask: (hand_mask & bit) == 0,
            observation=f"observer drew {normalized}",
            repair=repair,
        )
        self.unknown_mask &= ~bit
        self._invalidate_cache()
        self._assert_consistent()

    def observer_known_play(self, tile: Sequence[int]) -> None:
        bit = _tile_bit(tile)
        if not (self.unknown_mask & bit):
            return
        normalized = _normalize_tile(tile)

        def repair(hand_mask: int) -> int:
            if not (hand_mask & bit):
                return hand_mask
            stock_mask = self.unknown_mask & ~hand_mask
            stock_indices = list(_indices_from_mask(stock_mask))
            if not stock_indices:
                raise ParticleDepletionError(
                    f"Cannot condition on observer play {normalized}: no "
                    "replacement stock tile is available."
                )
            replacement_index = self.rng.choice(stock_indices)
            return (hand_mask ^ bit) | (1 << replacement_index)

        self._filter_and_resample(
            lambda hand_mask: (hand_mask & bit) == 0,
            observation=f"observer played {normalized}",
            repair=repair,
        )
        self.unknown_mask &= ~bit
        self._invalidate_cache()
        self._assert_consistent()

    def opponent_reveals_and_plays(self, tile: Sequence[int]) -> None:
        bit = _tile_bit(tile)
        normalized = _normalize_tile(tile)
        if not (self.unknown_mask & bit):
            raise ValueError(f"Revealed opponent tile {normalized} is not unknown.")

        # Set the new size before checking transformed particles.
        self.opponent_hand_size -= 1

        def repair(hand_mask: int) -> int:
            # The particle had x in its stock although the public observation
            # proves x was in the opponent hand. Swap x into that prior hand and
            # then remove it as the played tile. The resulting post-play hand is
            # obtained by removing one uniformly chosen former hand tile.
            hand_indices = list(_indices_from_mask(hand_mask))
            if not hand_indices:
                raise ParticleDepletionError(
                    f"Cannot condition on opponent play {normalized}: particle "
                    "hand is empty."
                )
            removed_index = self.rng.choice(hand_indices)
            return hand_mask & ~(1 << removed_index)

        self._filter_and_resample(
            lambda hand_mask: bool(hand_mask & bit),
            transform=lambda hand_mask: hand_mask ^ bit,
            observation=f"opponent played {normalized}",
            repair=repair,
        )
        self.unknown_mask &= ~bit
        self._invalidate_cache()
        self._assert_consistent()

    def opponent_hidden_draw(self):
        new_particles: list[int] = []
        for hand_mask in self.particles:
            stock_indices = list(
                _indices_from_mask(self.unknown_mask & ~hand_mask)
            )
            if not stock_indices:
                raise ValueError("Opponent cannot draw because the hidden stock is empty.")
            drawn_index = self.rng.choice(stock_indices)
            new_particles.append(hand_mask | (1 << drawn_index))

        self.particles = new_particles
        self.opponent_hand_size += 1
        self._invalidate_cache()
        self._assert_consistent()
        return self

    def suit_probabilities(self) -> list[float]:
        if self._probability_cache is not None:
            return list(self._probability_cache)
        total = len(self.particles)
        probabilities = [
            sum(
                1 for hand_mask in self.particles if hand_mask & suit_mask
            ) / total
            for suit_mask in SUIT_MASKS
        ]
        self._probability_cache = probabilities
        return list(probabilities)

    def probability_can_play(self, ends: Sequence[int]) -> float:
        if not ends:
            return 1.0
        legal_mask = SUIT_MASKS[int(ends[0])] | SUIT_MASKS[int(ends[1])]
        return sum(
            1 for hand_mask in self.particles if hand_mask & legal_mask
        ) / len(self.particles)

    @property
    def state_count(self) -> int:
        return len(self.particles)

    @property
    def unique_state_count(self) -> int:
        return len(set(self.particles))


class ExactOpponentModel:
    """Persistent hybrid model for one observer in one two-player game.

    The public class name is retained for compatibility. The model is exact in
    compact and enumerated modes and approximate in particle mode.
    """

    def __init__(
        self,
        *,
        max_enumerated_hands: int = MAX_ENUMERATED_HANDS,
        particle_count: int = PARTICLE_COUNT,
        seed: int = PARTICLE_SEED,
    ):
        if max_enumerated_hands <= 0:
            raise ValueError("max_enumerated_hands must be positive.")
        if particle_count <= 0:
            raise ValueError("particle_count must be positive.")

        self.max_enumerated_hands = int(max_enumerated_hands)
        self.particle_count = int(particle_count)
        self.seed = int(seed)
        self._belief = None
        self._game_id = None
        self._observer_player = None
        self._processed_history_length = 0
        self._own_draws_consumed = 0
        self._last_action_by_player: dict[int, object] = {}
        self._compact_to_enumerated_state_counts: list[int] = []
        self._compact_hidden_draw_state_records: list[dict] = []

    def reset(self) -> None:
        self._belief = None
        self._game_id = None
        self._observer_player = None
        self._processed_history_length = 0
        self._own_draws_consumed = 0
        self._last_action_by_player = {}
        self._compact_to_enumerated_state_counts = []
        self._compact_hidden_draw_state_records = []

    def update(self, state: dict) -> list[float]:
        """Process new actions and return opponent suit-presence probabilities."""
        cached = state.get("opponent_suit_probabilities")
        if cached is not None:
            return [float(value) for value in cached]

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

        for entry in history[self._processed_history_length:]:
            action_turn = self._processed_history_length + 1
            self._process_entry(entry, observer_player, own_draws, action_turn)
            self._processed_history_length += 1

        probabilities = self.suit_probabilities()
        state["opponent_suit_probabilities"] = probabilities
        state["opponent_model_mode"] = self.mode
        state["opponent_model_state_count"] = self.state_count
        return probabilities

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
        observer_player = int(
            state.get("observer_player", state.get("current_player", 0))
        )
        observer_initial_hand = state["current_player_initial_hand"]
        initial_opponent_hand_size = int(
            state.get("initial_hand_size", INITIAL_HAND_SIZE)
        )
        rng = random.Random(
            _stable_seed(
                self.seed,
                state.get("game_id"),
                observer_player,
                observer_initial_hand,
            )
        )

        self._belief = CompactOpponentBelief(
            observer_initial_hand,
            opponent_hand_size=initial_opponent_hand_size,
            rng=rng,
            max_enumerated_hands=self.max_enumerated_hands,
            particle_count=self.particle_count,
        )
        self._game_id = state.get("game_id")
        self._observer_player = observer_player
        self._processed_history_length = 0
        self._own_draws_consumed = 0
        self._last_action_by_player = {}
        self._compact_to_enumerated_state_counts = []
        self._compact_hidden_draw_state_records = []

    def _process_entry(
        self,
        entry: PublicAction,
        observer_player: int,
        own_draws: Sequence[tuple[int, int]],
        action_turn: int,
    ) -> None:
        if self._belief is None:
            raise RuntimeError("Opponent belief is not initialized.")

        actor = entry.actor
        action = entry.action
        previous_actor_action = self._last_action_by_player.get(actor)

        if actor == observer_player:
            if _is_draw(action):
                if self._own_draws_consumed >= len(own_draws):
                    raise ValueError("Missing private identity for an observer draw.")
                self._belief.observer_known_draw(
                    own_draws[self._own_draws_consumed]
                )
                self._own_draws_consumed += 1
            elif _is_tile_play(action):
                self._belief.observer_known_play(action[0])
        else:
            if _is_draw(action):
                if entry.ends_before is not None:
                    self._belief.condition_no_legal(*entry.ends_before)
                previous_mode = self._belief.mode
                self._belief = self._belief.opponent_hidden_draw()
                transition_count = getattr(
                    self._belief,
                    "compact_hidden_draw_final_state_count",
                    None,
                )
                if previous_mode == "compact_exact" and transition_count is not None:
                    transition_count = int(transition_count)
                    self._compact_hidden_draw_state_records.append({
                        "turn": int(action_turn),
                        "final_state_count": transition_count,
                        "resulting_mode": self._belief.mode,
                    })
                    if self._belief.mode == "enumerated_exact":
                        self._compact_to_enumerated_state_counts.append(transition_count)
            elif _is_pass(action):
                if (
                    previous_actor_action != ("DRAW", None)
                    and entry.ends_before is not None
                ):
                    self._belief.condition_no_legal(*entry.ends_before)
            elif _is_tile_play(action):
                self._belief.opponent_reveals_and_plays(action[0])

        self._last_action_by_player[actor] = action

    def suit_probabilities(self) -> list[float]:
        if self._belief is None:
            return [0.0] * SUIT_COUNT
        return self._belief.suit_probabilities()

    def probability_can_play(self, ends: Sequence[int]) -> float:
        if self._belief is None:
            return 1.0
        return self._belief.probability_can_play(ends)

    @property
    def compact_to_enumerated_state_counts(self) -> list[int]:
        """Return final-state counts for compact-to-enumerated transitions."""
        return list(self._compact_to_enumerated_state_counts)

    @property
    def compact_hidden_draw_state_records(self) -> list[dict]:
        """Return compact hidden-draw expansion records with turn metadata."""
        return [dict(record) for record in self._compact_hidden_draw_state_records]

    @property
    def mode(self) -> str:
        return "uninitialized" if self._belief is None else self._belief.mode

    @property
    def state_count(self) -> int:
        return 0 if self._belief is None else self._belief.state_count


HybridOpponentModel = ExactOpponentModel


def compute_opponent_suit_probabilities(state: dict) -> list[float]:
    """Strict one-shot compatibility wrapper.

    Prefer a persistent ``ExactOpponentModel`` instance in agents. This wrapper
    has no snapshot fallback and raises when required private observer fields are
    absent. Cached probabilities in ``state`` are returned immediately.
    """
    model = ExactOpponentModel()
    return model.update(state)


def response_probability_from_marginals(
    probabilities: Sequence[float],
    ends: Sequence[int],
) -> float:
    """Approximate the chance that the opponent can answer the two board ends."""
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
