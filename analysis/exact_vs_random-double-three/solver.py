"""Exact optimal-vs-random solver backed by the repository's domino engine.

The solver owns hidden-world enumeration and exact belief recursion, but it
does not duplicate game rules. Initial deals, opening selection, legal actions,
draw/pass behavior, terminal detection, and blocked-game tie-breaking all run
through :class:`middleware.domino_engine.DominoEngine`.

Hidden information is handled as an exact belief over concrete worlds.  The
solver never lets the controlled player condition an action on a hidden world.
All probabilities are fractions.Fraction objects.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from itertools import combinations
from math import gcd, lcm
from pathlib import Path
import sys
from typing import Iterable, Iterator, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from middleware.domino_engine import (  # pylint: disable=wrong-import-position
    DominoEngine,
    DominoRestartState,
    RESTART_STATE_FORMAT_VERSION,
    RULESET_VERSION,
)
from middleware.rulesets import (  # pylint: disable=wrong-import-position
    RULESET_NAMES,
    resolve_ruleset,
)


RULESETS = {
    name: (resolve_ruleset(name).max_pip, resolve_ruleset(name).hand_size)
    for name in RULESET_NAMES
}

# Action = (tile_id, side).  side is 0 for left and 1 for right.
# Negative tile ids encode the two non-tile engine actions.
DRAW_ACTION = (-1, -1)
PASS_ACTION = (-2, -1)


@dataclass(frozen=True, order=True, slots=True)
class World:
    """Compact immutable projection of one engine continuation state."""

    hand0: int
    hand1: int
    stock_mask: int
    left: int               # -1 means empty board
    right: int              # -1 means empty board
    current_player: int
    required_opening: int   # tile id, or -1
    consecutive_passes: int
    drew_mask: int          # bit p == player p already drew this turn
    last_valid_player: int  # -1 before first valid tile play
    winner: int             # -1 for a non-terminal world


Belief = tuple[tuple[World, int], ...]
Action = tuple[int, int]


@dataclass(frozen=True, slots=True)
class SolveStats:
    value_calls: int
    cache_hits: int
    cache_entries: int
    max_belief_worlds: int


@dataclass(frozen=True, slots=True)
class InitialHandResult:
    value: Fraction
    initial_world_count: int
    initial_observation_groups: tuple[tuple[int, int, Fraction], ...]
    stats: SolveStats


class _FixedDealRng:
    """Supply one exact deal while still letting ``DominoEngine.reset`` own it."""

    def __init__(self, ordered_tiles):
        self.ordered_tiles = list(ordered_tiles)

    def shuffle(self, values):
        if len(values) != len(self.ordered_tiles):
            raise ValueError("Fixed deal has the wrong number of tiles.")
        values[:] = self.ordered_tiles


class ExactVsRandomSolver:
    """Exact POMDP solver for one controlled seat against uniform random."""

    def __init__(self, ruleset: str = "double-three", hero_seat: int = 0):
        if ruleset not in RULESETS:
            raise ValueError(f"Unknown ruleset {ruleset!r}; choose from {tuple(RULESETS)}")
        if hero_seat not in (0, 1):
            raise ValueError("hero_seat must be 0 or 1")

        self.ruleset = resolve_ruleset(ruleset)
        self.ruleset_name = self.ruleset.name
        self.max_pip = self.ruleset.max_pip
        self.hand_size = self.ruleset.hand_size
        self.hero_seat = hero_seat
        self.random_seat = 1 - hero_seat
        self.tiles = self.ruleset.all_tiles
        self.tile_to_id = {tile: tile_id for tile_id, tile in enumerate(self.tiles)}
        self.double_tile_ids = frozenset(
            tile_id
            for tile_id, (left, right) in enumerate(self.tiles)
            if left == right
        )
        self.tile_count = len(self.tiles)
        self.full_mask = (1 << self.tile_count) - 1
        needed = 2 * self.hand_size
        if needed > self.tile_count:
            raise ValueError("Ruleset cannot deal both two-player hands")
        self.initial_stock_size = self.tile_count - needed

        self._cache: dict[Belief, Fraction] = {}
        self._value_calls = 0
        self._cache_hits = 0
        self._max_belief_worlds = 0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def tile_ids_to_tiles(self, tile_ids: Sequence[int]) -> tuple[tuple[int, int], ...]:
        return tuple(self.tiles[i] for i in tile_ids)

    def mask_to_ids(self, mask: int) -> tuple[int, ...]:
        return tuple(i for i in range(self.tile_count) if mask & (1 << i))

    def ids_to_mask(self, ids: Iterable[int]) -> int:
        mask = 0
        for tile_id in ids:
            tile_id = int(tile_id)
            if not 0 <= tile_id < self.tile_count:
                raise ValueError(f"Invalid tile id: {tile_id}")
            bit = 1 << tile_id
            if mask & bit:
                raise ValueError(f"Duplicate tile id: {tile_id}")
            mask |= bit
        return mask

    def all_initial_hands(self) -> Iterator[tuple[int, ...]]:
        return combinations(range(self.tile_count), self.hand_size)

    def initial_hand_has_guaranteed_double_opener(
        self,
        hero_hand_ids: Sequence[int],
    ) -> bool:
        """Return whether every compatible deal opens from a held double.

        When true, swapping player labels preserves every game and permits an
        exact seat-0 result to be reused for seat 1. The only asymmetric rule
        is player 0 opening when every double is in the stock.
        """
        hero_ids = frozenset(int(tile_id) for tile_id in hero_hand_ids)
        if hero_ids & self.double_tile_ids:
            return True
        unseen_doubles = len(self.double_tile_ids - hero_ids)
        return unseen_doubles > self.initial_stock_size

    def clear_cache(self) -> None:
        self._cache.clear()
        self._value_calls = 0
        self._cache_hits = 0
        self._max_belief_worlds = 0

    def stats(self) -> SolveStats:
        return SolveStats(
            value_calls=self._value_calls,
            cache_hits=self._cache_hits,
            cache_entries=len(self._cache),
            max_belief_worlds=self._max_belief_worlds,
        )

    def solve_initial_hand(self, hero_hand_ids: Sequence[int]) -> InitialHandResult:
        """Return exact P(win) conditional on this unordered initial hero hand.

        The hero knows their own hand and the publicly visible current_player at
        reset.  Therefore the  hidden worlds are partitioned by current_player
        before optimization.  This is essential: taking max across worlds with
        different initial observations would be information leakage.
        """

        self.clear_cache()
        hero_hand = tuple(sorted(int(x) for x in hero_hand_ids))
        if len(hero_hand) != self.hand_size:
            raise ValueError(
                f"{self.ruleset_name} requires {self.hand_size} tiles in the initial hand"
            )
        hero_mask = self.ids_to_mask(hero_hand)

        worlds = list(self.enumerate_initial_worlds(hero_mask))
        if not worlds:
            raise RuntimeError("No initial worlds were generated")

        groups: dict[int, dict[World, Fraction]] = defaultdict(lambda: defaultdict(Fraction))
        for world in worlds:
            # current_player is part of the compact public state returned by the engine.
            groups[world.current_player][world] += Fraction(1)

        total = len(worlds)
        result = Fraction(0)
        group_details: list[tuple[int, int, Fraction]] = []
        for observed_current in sorted(groups):
            raw_group = groups[observed_current]
            count = sum(raw_group.values(), Fraction(0))
            belief = self._normalize_belief(raw_group)
            group_value = self._value(belief)
            result += Fraction(count, total) * group_value
            group_details.append((observed_current, int(count), group_value))

        return InitialHandResult(
            value=result,
            initial_world_count=total,
            initial_observation_groups=tuple(group_details),
            stats=self.stats(),
        )

    def best_action(self, belief: Belief) -> tuple[Fraction, tuple[tuple[Action, Fraction], ...]]:
        """Return value and exact Q-values when it is the hero's turn.

        This is mainly useful for interactive use/testing.  ``belief`` must be a
        valid normalized belief with a common hero observation.
        """

        if not belief:
            raise ValueError("Belief cannot be empty")
        current = belief[0][0].current_player
        if current != self.hero_seat:
            raise ValueError("best_action() requires the hero to be on turn")
        terminal = self._terminal_belief_value(belief)
        if terminal is not None:
            return terminal, tuple()

        actions = self._common_hero_actions(belief)
        q_values = tuple((action, self._hero_action_value(belief, action)) for action in actions)
        return max(v for _, v in q_values), q_values

    # ------------------------------------------------------------------
    # Initial hidden worlds
    # ------------------------------------------------------------------

    def enumerate_initial_worlds(self, hero_mask: int) -> Iterator[World]:
        """Enumerate equiprobable worlds conditional on the hero's unordered hand.

        A hidden world consists of the opponent's unordered initial hand and
        unordered stock. Stock permutations are integrated exactly as uniform
        chance branches at draw time rather than materialized in advance.
        """

        if hero_mask.bit_count() != self.hand_size:
            raise ValueError("hero_mask has the wrong number of tiles")

        remaining_ids = self.mask_to_ids(self.full_mask ^ hero_mask)
        for opponent_ids in combinations(remaining_ids, self.hand_size):
            opponent_mask = self.ids_to_mask(opponent_ids)
            stock_ids = tuple(i for i in remaining_ids if not (opponent_mask & (1 << i)))
            if self.hero_seat == 0:
                hand0_ids = self.mask_to_ids(hero_mask)
                hand1_ids = self.mask_to_ids(opponent_mask)
            else:
                hand0_ids = self.mask_to_ids(opponent_mask)
                hand1_ids = self.mask_to_ids(hero_mask)

            # Canonical stock order is only an engine-construction detail. The
            # World immediately converts it to a mask, and every future draw
            # branches uniformly over that mask.
            ordered_deal = (
                self.tile_ids_to_tiles(hand0_ids)
                + self.tile_ids_to_tiles(hand1_ids)
                + self.tile_ids_to_tiles(stock_ids)
            )
            engine = DominoEngine(
                player_count=2,
                rng=_FixedDealRng(ordered_deal),
                ruleset=self.ruleset,
            )
            yield self._world_from_engine(engine)

    # ------------------------------------------------------------------
    # Exact recursion on beliefs
    # ------------------------------------------------------------------

    def _value(self, belief: Belief) -> Fraction:
        cached = self._cache.get(belief)
        if cached is not None:
            self._cache_hits += 1
            return cached

        self._value_calls += 1
        self._max_belief_worlds = max(self._max_belief_worlds, len(belief))

        terminal = self._terminal_belief_value(belief)
        if terminal is not None:
            self._cache[belief] = terminal
            return terminal

        current_players = {world.current_player for world, _ in belief}
        if len(current_players) != 1:
            raise AssertionError(
                "Belief mixes different public current_player observations; "
                "the caller must partition observable branches first"
            )
        current = next(iter(current_players))

        if current == self.hero_seat:
            actions = self._common_hero_actions(belief)
            value = max(self._hero_action_value(belief, action) for action in actions)
        else:
            value = self._random_turn_value(belief)

        self._cache[belief] = value
        return value

    def _common_hero_actions(self, belief: Belief) -> tuple[Action, ...]:
        first_actions = tuple(self.legal_actions(belief[0][0]))
        first_set = set(first_actions)
        for world, _ in belief[1:]:
            other = set(self.legal_actions(world))
            if other != first_set:
                raise AssertionError(
                    "Hero legal actions differ across hidden worlds.  This indicates "
                    "an observable branch was not partitioned correctly."
                )
        return tuple(sorted(first_set))

    def _hero_action_value(self, belief: Belief, action: Action) -> Fraction:
        total_weight = sum(weight for _, weight in belief)

        if action == DRAW_ACTION:
            # The hero observes the identity of their own drawn tile, so this
            # chance event must branch before any later max decision.
            groups: dict[int, dict[World, Fraction]] = defaultdict(lambda: defaultdict(Fraction))
            for world, weight in belief:
                for next_world, drawn, probability in self.action_outcomes(
                    world, action
                ):
                    if drawn is None:
                        raise AssertionError("DRAW did not return a drawn tile")
                    groups[drawn][next_world] += Fraction(weight) * probability

            value = Fraction(0)
            for raw_group in groups.values():
                branch_mass = sum(raw_group.values(), Fraction(0))
                next_belief = self._normalize_belief(raw_group)
                value += Fraction(branch_mass, total_weight) * self._value(next_belief)
            return value

        raw_next: dict[World, Fraction] = defaultdict(Fraction)
        for world, weight in belief:
            outcomes = self.action_outcomes(world, action)
            if len(outcomes) != 1:
                raise AssertionError("A non-draw action produced chance branches")
            next_world, _drawn, probability = outcomes[0]
            raw_next[next_world] += Fraction(weight) * probability
        return self._value(self._normalize_belief(raw_next))

    def _random_turn_value(self, belief: Belief) -> Fraction:
        """Average exactly over the random opponent's legal-action distribution.

        Observing which action the random opponent chose updates the posterior.
        A world with k legal actions contributes likelihood 1/k to each action.
        """

        total_prior_weight = sum(weight for _, weight in belief)
        by_observed_action: dict[Action, dict[World, Fraction]] = defaultdict(
            lambda: defaultdict(Fraction)
        )

        for world, weight in belief:
            actions = self.legal_actions(world)
            probability_factor = Fraction(weight, len(actions))
            for action in actions:
                for next_world, _drawn, draw_probability in self.action_outcomes(
                    world, action
                ):
                    by_observed_action[action][next_world] += (
                        probability_factor * draw_probability
                    )

        value = Fraction(0)
        branch_mass_check = Fraction(0)
        for raw_group in by_observed_action.values():
            branch_mass = sum(raw_group.values(), Fraction(0))
            branch_probability = Fraction(branch_mass, total_prior_weight)
            branch_mass_check += branch_probability
            next_belief = self._normalize_belief(raw_group)
            value += branch_probability * self._value(next_belief)

        if branch_mass_check != 1:
            raise AssertionError(f"Random branches sum to {branch_mass_check}, not 1")
        return value

    # ------------------------------------------------------------------
    # DominoEngine adapter
    # ------------------------------------------------------------------

    def _world_from_engine(self, engine: DominoEngine) -> World:
        ends = tuple(engine.ends)
        return World(
            hand0=self.ids_to_mask(self.tile_to_id[tile] for tile in engine.hands[0]),
            hand1=self.ids_to_mask(self.tile_to_id[tile] for tile in engine.hands[1]),
            stock_mask=self.ids_to_mask(
                self.tile_to_id[tile] for tile in engine.stock
            ),
            left=-1 if not ends else int(ends[0]),
            right=-1 if not ends else int(ends[1]),
            current_player=int(engine.current_player),
            required_opening=(
                -1
                if engine.required_opening_tile is None
                else self.tile_to_id[tuple(engine.required_opening_tile)]
            ),
            consecutive_passes=int(engine.consecutive_passes),
            drew_mask=sum(
                1 << player
                for player in range(engine.player_count)
                if engine.drew_this_turn[player]
            ),
            last_valid_player=(
                -1
                if engine.last_valid_tile_player is None
                else int(engine.last_valid_tile_player)
            ),
            winner=-1 if engine.winner is None else int(engine.winner),
        )

    def _engine_from_world(
        self,
        world: World,
        *,
        first_stock_tile: int | None = None,
    ) -> DominoEngine:
        if world.winner >= 0:
            raise ValueError("A terminal world cannot be continued.")

        hands = (
            self.tile_ids_to_tiles(self.mask_to_ids(world.hand0)),
            self.tile_ids_to_tiles(self.mask_to_ids(world.hand1)),
        )
        last_valid = (
            None if world.last_valid_player < 0 else world.last_valid_player
        )
        last_turns = [-1, -1]
        if last_valid is not None:
            last_turns[last_valid] = 0
        stock_ids = list(self.mask_to_ids(world.stock_mask))
        if first_stock_tile is not None:
            if first_stock_tile not in stock_ids:
                raise ValueError("Requested draw tile is absent from the stock.")
            stock_ids.remove(first_stock_tile)
            stock_ids.insert(0, first_stock_tile)
        state = DominoRestartState(
            format_version=RESTART_STATE_FORMAT_VERSION,
            ruleset_name=self.ruleset_name,
            ruleset_version=RULESET_VERSION,
            game_id=0,
            player_count=2,
            current_player=world.current_player,
            hands=hands,
            initial_hands=hands,
            drawn_tiles_by_player=(tuple(), tuple()),
            stock=self.tile_ids_to_tiles(stock_ids),
            board_history=tuple(),
            ends=(tuple() if world.left < 0 else (world.left, world.right)),
            turn=1,
            game_over=False,
            winner=None,
            win_reason=None,
            consecutive_passes=world.consecutive_passes,
            drew_this_turn=tuple(
                bool(world.drew_mask & (1 << player)) for player in range(2)
            ),
            last_valid_tile_player=last_valid,
            last_valid_tile_turn_by_player=tuple(last_turns),
            required_opening_tile=(
                None
                if world.required_opening < 0
                else self.tiles[world.required_opening]
            ),
            horizontal_direction=(-1, 1),
        )
        return DominoEngine.from_restart_state(state)

    def _encode_engine_action(self, action) -> Action:
        if action == ("DRAW", None):
            return DRAW_ACTION
        if action is None:
            return PASS_ACTION
        tile, side = action
        return self.tile_to_id[tuple(tile)], int(side)

    def _decode_action(self, action: Action):
        if action == DRAW_ACTION:
            return ("DRAW", None)
        if action == PASS_ACTION:
            return None
        tile_id, side = action
        return self.tiles[tile_id], side

    def legal_actions(self, world: World) -> tuple[Action, ...]:
        engine = self._engine_from_world(world)
        return tuple(sorted(
            self._encode_engine_action(action)
            for action in engine.valid_actions()
        ))

    def _apply_engine_action(
        self,
        world: World,
        action: Action,
        *,
        drawn_tile: int | None = None,
    ) -> World:
        engine = self._engine_from_world(world, first_stock_tile=drawn_tile)
        legal_actions = engine.valid_actions()
        engine_action = self._decode_action(action)
        if engine_action not in legal_actions:
            raise ValueError(f"Illegal action {action} in world {world}")
        engine.step(
            engine_action,
            return_state=False,
            legal_actions=legal_actions,
        )
        return self._world_from_engine(engine)

    def action_outcomes(
        self,
        world: World,
        action: Action,
    ) -> tuple[tuple[World, int | None, Fraction], ...]:
        """Return exact engine transitions, integrating unknown stock order."""
        if action != DRAW_ACTION:
            return ((self._apply_engine_action(world, action), None, Fraction(1)),)

        stock_ids = self.mask_to_ids(world.stock_mask)
        if not stock_ids:
            raise ValueError("DRAW requested with an empty stock.")
        probability = Fraction(1, len(stock_ids))
        return tuple(
            (
                self._apply_engine_action(
                    world,
                    action,
                    drawn_tile=tile_id,
                ),
                tile_id,
                probability,
            )
            for tile_id in stock_ids
        )

    def apply_action(self, world: World, action: Action) -> tuple[World, int | None]:
        """Apply a deterministic action; use ``action_outcomes`` for a draw."""
        outcomes = self.action_outcomes(world, action)
        if len(outcomes) != 1:
            raise ValueError(
                "DRAW has multiple exact chance outcomes; use action_outcomes()."
            )
        next_world, drawn_tile, _probability = outcomes[0]
        return next_world, drawn_tile

    def terminal_winner(self, world: World) -> int | None:
        return None if world.winner < 0 else world.winner

    # ------------------------------------------------------------------
    # Belief canonicalization / terminal handling
    # ------------------------------------------------------------------

    def _terminal_belief_value(self, belief: Belief) -> Fraction | None:
        winners = [(self.terminal_winner(world), weight) for world, weight in belief]
        terminal_flags = [winner is not None for winner, _ in winners]
        if not any(terminal_flags):
            return None
        if not all(terminal_flags):
            raise AssertionError(
                "A single observation branch mixes terminal and non-terminal worlds"
            )
        total = sum(weight for _, weight in winners)
        wins = sum(weight for winner, weight in winners if winner == self.hero_seat)
        return Fraction(wins, total)

    def _normalize_belief(self, raw: Mapping[World, Fraction | int]) -> Belief:
        """Canonicalize positive relative exact weights as coprime integers."""

        nonzero: list[tuple[World, Fraction]] = []
        for world, weight in raw.items():
            f = Fraction(weight)
            if f < 0:
                raise ValueError("Belief weights must be non-negative")
            if f:
                nonzero.append((world, f))
        if not nonzero:
            raise ValueError("Cannot normalize an empty/zero belief")

        common_denominator = 1
        for _, weight in nonzero:
            common_denominator = lcm(common_denominator, weight.denominator)

        integer_weights = [
            (world, weight.numerator * (common_denominator // weight.denominator))
            for world, weight in nonzero
        ]
        common_gcd = reduce(gcd, (weight for _, weight in integer_weights))
        canonical = tuple(
            sorted((world, weight // common_gcd) for world, weight in integer_weights)
        )
        return canonical


class CheaterVsRandomSolver(ExactVsRandomSolver):
    """Exact solver for an optimal player that observes the complete world.

    The cheater continuously sees both hands and the membership of the stock.
    The future order of the stock remains random: every draw branches uniformly
    over the tiles still present.  Unlike :class:`ExactVsRandomSolver`, this
    solver therefore recurses directly on one ``World`` and never constructs a
    hidden-state belief.
    """

    def solve_initial_hand(self, hero_hand_ids: Sequence[int]) -> InitialHandResult:
        """Return exact P(win) when every compatible world is observable."""

        self.clear_cache()
        hero_hand = tuple(sorted(int(x) for x in hero_hand_ids))
        if len(hero_hand) != self.hand_size:
            raise ValueError(
                f"{self.ruleset_name} requires {self.hand_size} tiles in the initial hand"
            )
        hero_mask = self.ids_to_mask(hero_hand)
        worlds = list(self.enumerate_initial_worlds(hero_mask))
        if not worlds:
            raise RuntimeError("No initial worlds were generated")

        groups: dict[int, list[World]] = defaultdict(list)
        for world in worlds:
            groups[world.current_player].append(world)

        total = len(worlds)
        result = Fraction(0)
        group_details: list[tuple[int, int, Fraction]] = []
        for observed_current in sorted(groups):
            group_worlds = groups[observed_current]
            group_value = sum(
                (self._perfect_information_value(world) for world in group_worlds),
                Fraction(0),
            ) / len(group_worlds)
            result += Fraction(len(group_worlds), total) * group_value
            group_details.append(
                (observed_current, len(group_worlds), group_value)
            )

        return InitialHandResult(
            value=result,
            initial_world_count=total,
            initial_observation_groups=tuple(group_details),
            stats=self.stats(),
        )

    def _perfect_information_value(self, world: World) -> Fraction:
        cached = self._cache.get(world)
        if cached is not None:
            self._cache_hits += 1
            return cached

        self._value_calls += 1
        self._max_belief_worlds = max(self._max_belief_worlds, 1)
        winner = self.terminal_winner(world)
        if winner is not None:
            value = Fraction(int(winner == self.hero_seat))
        elif world.current_player == self.hero_seat:
            value = max(
                self._perfect_information_action_value(world, action)
                for action in self.legal_actions(world)
            )
        else:
            actions = self.legal_actions(world)
            value = sum(
                (
                    self._perfect_information_action_value(world, action)
                    for action in actions
                ),
                Fraction(0),
            ) / len(actions)

        self._cache[world] = value
        return value

    def _perfect_information_action_value(
        self,
        world: World,
        action: Action,
    ) -> Fraction:
        return sum(
            (
                probability * self._perfect_information_value(next_world)
                for next_world, _drawn, probability in self.action_outcomes(
                    world,
                    action,
                )
            ),
            Fraction(0),
        )


def fraction_to_decimal(value: Fraction, digits: int = 15) -> str:
    """Stable human-readable decimal without changing the exact stored fraction."""

    return f"{float(value):.{digits}g}"


def action_to_string(action: Action, tiles: Sequence[tuple[int, int]]) -> str:
    if action == DRAW_ACTION:
        return "DRAW"
    if action == PASS_ACTION:
        return "PASS"
    tile_id, side = action
    side_name = "L" if side == 0 else "R"
    return f"{tiles[tile_id]}@{side_name}"
