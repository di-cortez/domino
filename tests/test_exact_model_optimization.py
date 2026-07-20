"""Correctness tests for exact opponent-model performance optimizations.

Run from the repository root with::

    python tests/test_exact_model_optimization.py
"""

from __future__ import annotations

import copy
import random
import sys
import unittest
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.agent import RandomAgent
from middleware.domino_engine import DominoEngine
from middleware.middleware import GameManager
from middleware.opponent_model import (
    ExactOpponentModel,
    PROFILE_ASSIGNMENT_CACHE_SIZE,
    MuOpponentBelief,
    SlotOpponentBelief,
    _annotate_public_action_suffix,
    _bits_from_mask,
    _count_profile_assignments_cached,
    mask_from_tiles,
    reconstruct_public_actions,
)


def _tile_by_tile_assignment_count(profile: tuple[int, ...]) -> int:
    """Independent reference implementation of the former scalar DP."""
    if not profile:
        return 1
    if any(domain == 0 for domain in profile):
        return 0
    slot_count = len(profile)
    full_slot_mask = (1 << slot_count) - 1
    counts = [0] * (1 << slot_count)
    counts[0] = 1
    tile_union = 0
    for domain in profile:
        tile_union |= domain
    for tile_bit in _bits_from_mask(tile_union):
        eligible_slots = 0
        for slot_index, domain in enumerate(profile):
            if domain & tile_bit:
                eligible_slots |= 1 << slot_index
        next_counts = counts.copy()
        for occupied_slots, count in enumerate(counts):
            if not count:
                continue
            available = eligible_slots & ~occupied_slots
            while available:
                slot_bit = available & -available
                next_counts[occupied_slots | slot_bit] += count
                available ^= slot_bit
        counts = next_counts
    return counts[full_slot_mask]


class GroupedAssignmentDPTests(unittest.TestCase):
    def test_grouped_counts_match_independent_dp_and_hand_multiplicities(self):
        """Equal, overlapping, restrictive, and infeasible profiles stay exact."""
        profiles = (
            (0b1111, 0b1111, 0b1111),
            (0b0011, 0b0110, 0b1100),
            (0b0001, 0b0011, 0b0111),
            (0b10001, 0b00111, 0b11100, 0b11010),
            (0b0011, 0b0011, 0b0011),
        )
        for raw_profile in profiles:
            profile = tuple(sorted(raw_profile))
            grouped = _count_profile_assignments_cached(profile)
            reference = _tile_by_tile_assignment_count(profile)
            self.assertEqual(grouped, reference, profile)

            if grouped:
                unknown_mask = 0
                for domain in profile:
                    unknown_mask |= domain
                belief = SlotOpponentBelief.from_profiles(
                    unknown_mask=unknown_mask,
                    opponent_hand_size=len(profile),
                    profiles={profile: 1},
                )
                self.assertEqual(
                    sum(belief._profile_hand_weights(profile).values()),
                    grouped,
                )

    def test_random_small_profiles_match_tile_by_tile_reference(self):
        """Deterministic asymmetric coverage exercises many eligibility groups."""
        generator = random.Random(20260719)
        for _case in range(500):
            slot_count = generator.randint(1, 6)
            tile_count = generator.randint(1, 10)
            full_tile_mask = (1 << tile_count) - 1
            profile = tuple(sorted(
                generator.randint(1, full_tile_mask)
                for _slot in range(slot_count)
            ))
            self.assertEqual(
                _count_profile_assignments_cached(profile),
                _tile_by_tile_assignment_count(profile),
                profile,
            )

    def test_uniform_domain_closed_form_and_bounded_process_cache(self):
        """Shared domains use exact falling factorials and the cache is bounded."""
        _count_profile_assignments_cached.cache_clear()
        self.assertEqual(
            _count_profile_assignments_cached((0b11111,) * 3),
            5 * 4 * 3,
        )
        self.assertEqual(
            _count_profile_assignments_cached((0b11,) * 3),
            0,
        )
        self.assertEqual(
            _count_profile_assignments_cached.cache_info().maxsize,
            PROFILE_ASSIGNMENT_CACHE_SIZE,
        )
        for index in range(PROFILE_ASSIGNMENT_CACHE_SIZE + 37):
            domain = ((1 << 7) - 1) | ((index + 1) << 7)
            _count_profile_assignments_cached((domain,) * 7)
        self.assertEqual(
            _count_profile_assignments_cached.cache_info().currsize,
            PROFILE_ASSIGNMENT_CACHE_SIZE,
        )


def _small_mu_belief() -> MuOpponentBelief:
    """Return a mutable exact belief with both hand and stock possibilities."""
    tiles = [(value, value) for value in range(5)]
    bits = [mask_from_tiles([tile]) for tile in tiles]
    return MuOpponentBelief.from_weights(
        unknown_mask=mask_from_tiles(tiles),
        opponent_hand_size=2,
        weights={left | right: 1 for left, right in combinations(bits, 2)},
    )


def _small_slot_belief() -> SlotOpponentBelief:
    """Return the slot equivalent of the small mu belief."""
    unknown_mask = mask_from_tiles([(value, value) for value in range(5)])
    return SlotOpponentBelief.from_profiles(
        unknown_mask=unknown_mask,
        opponent_hand_size=2,
        profiles={(unknown_mask, unknown_mask): 1},
    )


class BeliefQueryCacheTests(unittest.TestCase):
    def _prime_mu(self, belief):
        belief.suit_probabilities()
        belief.probability_can_play((0, 1))
        _ = belief.total_weight
        self.assertIsNotNone(belief._probability_cache)
        self.assertTrue(belief._response_probability_cache)
        self.assertIsNotNone(belief._total_weight_cache)

    def _assert_mu_cleared(self, belief):
        self.assertIsNone(belief._probability_cache)
        self.assertEqual(belief._response_probability_cache, {})
        self.assertIsNone(belief._total_weight_cache)

    def _prime_slot(self, belief):
        belief.suit_probabilities()
        belief.probability_can_play((0, 1))
        profile = next(iter(belief.profiles))
        belief._profile_hand_weights(profile)
        _ = belief.assignment_weight
        self.assertIsNotNone(belief._probability_cache)
        self.assertTrue(belief._response_probability_cache)
        self.assertTrue(belief._hand_weights_cache)
        self.assertIsNotNone(belief._assignment_weight_cache)

    def _assert_slot_query_caches_cleared(self, belief):
        self.assertIsNone(belief._probability_cache)
        self.assertEqual(belief._response_probability_cache, {})
        self.assertEqual(belief._hand_weights_cache, {})
        self.assertIsNone(belief._assignment_weight_cache)

    def test_mu_constructor_caches_zero_response_and_reversed_ends(self):
        """Alternate construction initializes bounded current-state query caches."""
        belief = _small_mu_belief()
        self.assertEqual(belief._response_probability_cache, {})
        self.assertIsNone(belief._total_weight_cache)

        no_low_suits = MuOpponentBelief.from_weights(
            unknown_mask=mask_from_tiles([(2, 2), (3, 3), (4, 4)]),
            opponent_hand_size=2,
            weights={
                mask_from_tiles([(2, 2), (3, 3)]): 1,
                mask_from_tiles([(2, 2), (4, 4)]): 1,
            },
        )
        self.assertEqual(no_low_suits.probability_can_play((0, 1)), 0.0)
        self.assertEqual(no_low_suits.probability_can_play((1, 0)), 0.0)
        self.assertEqual(len(no_low_suits._response_probability_cache), 1)

    def test_mu_mutations_invalidate_every_query_cache(self):
        """Condition, known tile, reveal, and hidden draw clear mu caches."""
        mutations = (
            lambda belief: belief.condition_no_legal(0, 0),
            lambda belief: belief.observer_known_draw((4, 4)),
            lambda belief: belief.observer_known_play((4, 4)),
            lambda belief: belief.opponent_reveals_and_plays((0, 0)),
            lambda belief: belief.opponent_hidden_draw(),
        )
        for mutation in mutations:
            belief = _small_mu_belief()
            self._prime_mu(belief)
            mutation(belief)
            self._assert_mu_cleared(belief)

            uncached_probabilities = belief.suit_probabilities()
            uncached_response = belief.probability_can_play((1, 2))
            self.assertEqual(uncached_probabilities, belief.suit_probabilities())
            self.assertEqual(
                uncached_response,
                belief.probability_can_play((2, 1)),
            )

    def test_slot_constructor_caches_zero_response_and_reversed_ends(self):
        """from_profiles initializes every slot cache and preserves cached zero."""
        belief = _small_slot_belief()
        self.assertEqual(belief._response_probability_cache, {})
        self.assertIsNone(belief._assignment_weight_cache)

        high_mask = mask_from_tiles([(2, 2), (3, 3), (4, 4)])
        no_low_suits = SlotOpponentBelief.from_profiles(
            unknown_mask=high_mask,
            opponent_hand_size=2,
            profiles={(high_mask, high_mask): 1},
        )
        self.assertEqual(no_low_suits.probability_can_play((0, 1)), 0.0)
        self.assertEqual(no_low_suits.probability_can_play((1, 0)), 0.0)
        self.assertEqual(len(no_low_suits._response_probability_cache), 1)

    def test_slot_mutations_invalidate_every_dependent_query_cache(self):
        """All profile replacements discard old scalar and probability results."""
        mutations = (
            lambda belief: belief.condition_no_legal(0, 0),
            lambda belief: belief.observer_known_draw((4, 4)),
            lambda belief: belief.observer_known_play((4, 4)),
            lambda belief: belief.opponent_reveals_and_plays((0, 0)),
            lambda belief: belief.opponent_hidden_draw(),
        )
        for mutation in mutations:
            belief = _small_slot_belief()
            self._prime_slot(belief)
            mutation(belief)
            self._assert_slot_query_caches_cleared(belief)
            # The post-mutation invariant check legitimately repopulates the
            # scalar cache, but only for profiles in the new belief state.
            self.assertTrue(
                set(belief._assignment_count_cache).issubset(belief.profiles)
            )

            uncached_probabilities = belief.suit_probabilities()
            uncached_response = belief.probability_can_play((1, 2))
            self.assertEqual(uncached_probabilities, belief.suit_probabilities())
            self.assertEqual(
                uncached_response,
                belief.probability_can_play((2, 1)),
            )


class TraceDisabledModelTests(unittest.TestCase):
    def test_default_keeps_traces_and_explicit_fast_mode_keeps_exact_state(self):
        """Trace allocation is optional while evidence and transition stay exact."""
        random.seed(20260719)
        engine = DominoEngine(player_count=2)
        manager = GameManager(engine, [RandomAgent(), RandomAgent()])
        _info, history = manager.play_full_game()

        terminal_state = engine._get_state()
        terminal_state["game_over"] = engine.game_over
        states = [row["state"] for row in history] + [terminal_state]
        model_pairs = {
            player: (
                ExactOpponentModel(),
                ExactOpponentModel(record_traces=False),
            )
            for player in range(2)
        }
        saw_default_trace = False

        for state in states:
            player = int(state["current_player"])
            detailed_model, fast_model = model_pairs[player]
            detailed_result = detailed_model.update_detailed(copy.deepcopy(state))
            fast_result = fast_model.update_detailed(copy.deepcopy(state))

            self.assertEqual(detailed_result.probabilities, fast_result.probabilities)
            self.assertEqual(detailed_result.mode, fast_result.mode)
            self.assertEqual(
                detailed_result.switched_this_update,
                fast_result.switched_this_update,
            )
            self.assertEqual(detailed_model.state_count, fast_model.state_count)
            self.assertEqual(detailed_model.profile_count, fast_model.profile_count)
            self.assertEqual(detailed_model.mu_hand_count, fast_model.mu_hand_count)
            self.assertEqual(detailed_model.total_weight, fast_model.total_weight)
            self.assertEqual(detailed_model.unknown_count, fast_model.unknown_count)
            self.assertEqual(
                detailed_model.opponent_hand_size,
                fast_model.opponent_hand_size,
            )
            self.assertEqual(detailed_model.switch_turn, fast_model.switch_turn)
            self.assertEqual(
                detailed_model.switch_upper_bound,
                fast_model.switch_upper_bound,
            )
            self.assertEqual(
                detailed_model.switch_mu_state_count,
                fast_model.switch_mu_state_count,
            )

            for left_end in range(7):
                for right_end in range(7):
                    self.assertEqual(
                        detailed_model.probability_can_play((left_end, right_end)),
                        fast_model.probability_can_play((left_end, right_end)),
                    )

            if detailed_result.new_snapshots or detailed_result.completed_turn_traces:
                saw_default_trace = True
            self.assertEqual(fast_result.new_snapshots, ())
            self.assertEqual(fast_result.completed_turn_traces, ())
            self.assertIsNone(fast_model.last_snapshot)
            self.assertIsNone(fast_model.last_completed_turn_trace)
            self.assertEqual(fast_model.turn_trace_history, [])
            self.assertEqual(fast_model.consume_new_snapshots(), [])
            self.assertIsNone(fast_model._pending_trace)

            if detailed_model._belief.mode == "slots_exact":
                self.assertEqual(
                    detailed_model._belief.profiles,
                    fast_model._belief.profiles,
                )
            else:
                self.assertEqual(
                    detailed_model._belief.weights,
                    fast_model._belief.weights,
                )

        self.assertTrue(saw_default_trace)


def _reconstruction_state(actions, *, terminal=False):
    """Build actor metadata for one syntactically valid annotated history."""
    actor = 0
    last_actor = actor
    for action in actions:
        last_actor = actor
        if action is None or action[0] != "DRAW":
            actor = (actor + 1) % 2
    current_player = last_actor if terminal and actions else actor
    return {
        "board_history": list(actions),
        "hand_sizes": [7, 7],
        "current_player": current_player,
        "game_over": terminal,
    }


def _observer_state(engine, observer_player):
    """Return a complete fixed-observer state after any real engine action."""
    return {
        "game_id": engine.game_id,
        "ends": list(engine.ends),
        "current_player_hand": [
            list(tile) for tile in engine.hands[observer_player]
        ],
        "current_player_initial_hand": [
            list(tile) for tile in engine.initial_hands[observer_player]
        ],
        "current_player_drawn_tiles": [
            list(tile) for tile in engine.drawn_tiles_by_player[observer_player]
        ],
        "current_player": engine.current_player,
        "observer_player": observer_player,
        "turn": engine.turn,
        "hand_sizes": [len(hand) for hand in engine.hands],
        "board_history": [
            engine._serialize_action(action) for action in engine.board_history
        ],
        "stock_size": len(engine.stock),
        "game_over": engine.game_over,
    }


class IncrementalPublicHistoryTests(unittest.TestCase):
    def test_suffix_annotation_matches_fresh_reconstruction_for_edge_sequences(self):
        """Draw chains, pass turns, and terminal actor handling remain identical."""
        sequences = (
            ([], False),
            ([((6, 6), 0), ((5, 6), 0), ((4, 5), 0)], False),
            ([((6, 6), 0), ("DRAW", None), ("DRAW", None), ((5, 6), 0)], False),
            ([((6, 6), 0), ("DRAW", None), None], False),
            ([((6, 6), 0), None, None], True),
            ([((6, 6), 0)], True),
        )
        for actions, terminal in sequences:
            annotated = []
            self.assertEqual(
                annotated,
                reconstruct_public_actions(_reconstruction_state([])),
            )
            for index, action in enumerate(actions):
                if annotated:
                    last = annotated[-1]
                    if last.action is not None and last.action[0] == "DRAW":
                        actor = last.actor
                        public_turn = last.public_turn
                    else:
                        actor = (last.actor + 1) % 2
                        public_turn = last.public_turn + 1
                    ends = last.ends_after
                else:
                    actor = 0
                    public_turn = 1
                    ends = None
                annotated = _annotate_public_action_suffix(
                    annotated,
                    [action],
                    actor,
                    public_turn,
                    ends,
                    2,
                )
                is_terminal_prefix = terminal and index == len(actions) - 1
                fresh = reconstruct_public_actions(_reconstruction_state(
                    actions[:index + 1],
                    terminal=is_terminal_prefix,
                ))
                self.assertEqual(annotated, fresh)

    def test_persistent_cache_matches_full_reconstruction_after_every_action(self):
        """Real seeded histories extend incrementally and reset safely."""
        observer_player = 0
        model = ExactOpponentModel(record_traces=False)
        saved_states = []
        draw_actors = set()

        for seed in (17, 29, 41, 53):
            random.seed(seed)
            engine = DominoEngine(player_count=2)
            initial_state = _observer_state(engine, observer_player)
            model.update(copy.deepcopy(initial_state))
            self.assertEqual(
                model._public_history,
                reconstruct_public_actions(initial_state),
            )
            while not engine.game_over:
                legal_actions = engine.valid_actions(engine.current_player)
                action = random.choice(legal_actions)
                if action == ("DRAW", None):
                    draw_actors.add(engine.current_player)
                engine.step(
                    action,
                    return_state=False,
                    legal_actions=legal_actions,
                )
                state = _observer_state(engine, observer_player)
                saved_states.append(copy.deepcopy(state))
                model.update(copy.deepcopy(state))
                expected = reconstruct_public_actions(state)
                self.assertEqual(model._public_history, expected)
                self.assertEqual(model._processed_history_length, len(expected))

                # Repeating an unchanged state is idempotent and does not
                # rebuild or extend the cached annotations.
                cached_object = model._public_history
                repeated = model.update(copy.deepcopy(state))
                self.assertIs(model._public_history, cached_object)
                self.assertEqual(repeated, model.suit_probabilities())

        self.assertEqual(draw_actors, {0, 1})

        # A shorter history with the same game id takes the safe reset path.
        earlier_state = saved_states[-3]
        model.update(copy.deepcopy(saved_states[-1]))
        model.update(copy.deepcopy(earlier_state))
        self.assertEqual(
            model._public_history,
            reconstruct_public_actions(earlier_state),
        )

        # A different game identity also reconstructs exactly once from scratch.
        random.seed(101)
        new_engine = DominoEngine(player_count=2)
        new_state = _observer_state(new_engine, observer_player)
        model.update(copy.deepcopy(new_state))
        self.assertEqual(model._game_id, new_engine.game_id)
        self.assertEqual(model._public_history, reconstruct_public_actions(new_state))

if __name__ == "__main__":
    unittest.main(verbosity=2)
