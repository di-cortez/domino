"""Focused compatibility tests for the headless ``DominoEngine.step`` path.

Run from the repository root with::

    python tests/test_headless_engine_step.py
"""

from __future__ import annotations

import copy
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.agent import RandomAgent
from agents.rl_nn import PolicyNetwork
from diagnostics.pairwise import play_game
from middleware.domino_engine import DominoEngine
from middleware.middleware import GameManager
from training.dataset_parallel import generate_dataset_game
from training.self_play import (
    DEFAULT_GAMMA,
    DEFAULT_REWARD_SCHEMA,
    REWARD_SCHEMAS,
    _collect_self_play_steps,
)


class FirstLegalAgent:
    """Deterministic test agent that leaves the supplied collection unchanged."""

    def choose_move(self, state, legal_actions):
        return legal_actions[0]


def _configure_draw_pass_game() -> DominoEngine:
    """Return a game that must draw, pass twice, and terminate as blocked."""
    engine = DominoEngine(player_count=2)
    engine.ends = [1, 1]
    engine.hands = [[(4, 4)], [(2, 2)]]
    engine.initial_hands = [hand.copy() for hand in engine.hands]
    engine.drawn_tiles_by_player = [[], []]
    engine.stock = [(3, 3)]
    engine.board_history = []
    engine.current_player = 0
    engine.required_opening_tile = None
    engine.consecutive_passes = 0
    engine.drew_this_turn = {0: False, 1: False}
    engine.turn = 0
    engine.game_over = False
    engine.winner = None
    return engine


def _configure_left_right_win_game() -> DominoEngine:
    """Return a game with left play, right play, and an empty-hand win."""
    engine = DominoEngine(player_count=2)
    engine.ends = [1, 2]
    engine.hands = [
        [(1, 3), (4, 5)],
        [(2, 4), (6, 6)],
    ]
    engine.initial_hands = [hand.copy() for hand in engine.hands]
    engine.drawn_tiles_by_player = [[], []]
    engine.stock = []
    engine.board_history = []
    engine.current_player = 0
    engine.required_opening_tile = None
    engine.consecutive_passes = 0
    engine.drew_this_turn = {0: False, 1: False}
    engine.turn = 0
    engine.game_over = False
    engine.winner = None
    return engine


class HeadlessEngineStepTests(unittest.TestCase):
    def _legacy_step_patch(self):
        """Return a patch that makes optimized call sites use default step work."""
        original_step = DominoEngine.step

        def legacy_step(engine, action, *args, **kwargs):
            return original_step(engine, action)

        return mock.patch.object(DominoEngine, "step", legacy_step)

    def test_default_step_remains_backward_compatible(self):
        """The default path still validates and returns a post-action state."""
        engine = DominoEngine(player_count=2)
        action = engine.valid_actions(engine.current_player)[0]

        state, game_over, info = engine.step(action)

        self.assertIsNotNone(state)
        self.assertEqual(state, engine._get_state())
        self.assertEqual(game_over, engine.game_over)
        self.assertEqual(info, {"winner": engine.winner})

    def test_headless_step_returns_stable_tuple_without_building_state(self):
        """return_state=False returns None and never calls _get_state."""
        engine = DominoEngine(player_count=2)
        legal_actions = engine.valid_actions(engine.current_player)
        action = legal_actions[0]

        with mock.patch.object(
            engine,
            "_get_state",
            wraps=engine._get_state,
        ) as get_state:
            result = engine.step(
                action,
                return_state=False,
                legal_actions=legal_actions,
            )

        self.assertEqual(len(result), 3)
        self.assertIsNone(result[0])
        self.assertEqual(result[1], engine.game_over)
        self.assertEqual(result[2], {"winner": engine.winner})
        get_state.assert_not_called()

    def test_supplied_actions_skip_generation_but_still_validate_membership(self):
        """The trusted path reuses actions without accepting an absent action."""
        engine = DominoEngine(player_count=2)
        legal_actions = engine.valid_actions(engine.current_player)
        action = legal_actions[0]

        with mock.patch.object(
            engine,
            "valid_actions",
            wraps=engine.valid_actions,
        ) as valid_actions:
            engine.step(
                action,
                return_state=False,
                legal_actions=legal_actions,
            )
        valid_actions.assert_not_called()

        rejected_engine = DominoEngine(player_count=2)
        legal_action = rejected_engine.valid_actions(
            rejected_engine.current_player
        )[0]
        original_state = copy.deepcopy(rejected_engine.__dict__)
        with self.assertRaises(ValueError):
            rejected_engine.step(
                legal_action,
                return_state=False,
                legal_actions=[],
            )
        self.assertEqual(rejected_engine.__dict__, original_state)

    def test_none_actions_compute_and_validate_inside_engine(self):
        """legal_actions=None retains the original internal validation path."""
        engine = DominoEngine(player_count=2)
        action = engine.valid_actions(engine.current_player)[0]
        with mock.patch.object(
            engine,
            "valid_actions",
            wraps=engine.valid_actions,
        ) as valid_actions:
            engine.step(action, return_state=False)
        valid_actions.assert_called_once()

    def test_precomputed_draw_and_pass_actions_preserve_special_rules(self):
        """Fresh trusted collections work for both DRAW and PASS membership."""
        engine = _configure_draw_pass_game()

        draw_actions = engine.valid_actions(engine.current_player)
        self.assertEqual(draw_actions, [("DRAW", None)])
        state, done, _info = engine.step(
            ("DRAW", None),
            return_state=False,
            legal_actions=draw_actions,
        )
        self.assertIsNone(state)
        self.assertFalse(done)

        pass_actions = engine.valid_actions(engine.current_player)
        self.assertEqual(pass_actions, [None])
        state, done, _info = engine.step(
            None,
            return_state=False,
            legal_actions=pass_actions,
        )
        self.assertIsNone(state)
        self.assertFalse(done)

        with self.assertRaises(ValueError):
            engine.step(None, return_state=False, legal_actions=[])

    def test_default_and_headless_mutations_match_across_complete_sequences(self):
        """Draw/pass and left/right/win sequences produce identical engines."""
        scenarios = (
            (
                _configure_draw_pass_game(),
                [("DRAW", None), None, None],
            ),
            (
                _configure_left_right_win_game(),
                [((1, 3), 0), ((2, 4), 1), ((4, 5), 1)],
            ),
        )

        for initial_engine, actions in scenarios:
            default_engine = copy.deepcopy(initial_engine)
            headless_engine = copy.deepcopy(initial_engine)
            for action in actions:
                default_state, default_done, default_info = default_engine.step(
                    action
                )
                legal_actions = headless_engine.valid_actions(
                    headless_engine.current_player
                )
                headless_state, headless_done, headless_info = (
                    headless_engine.step(
                        action,
                        return_state=False,
                        legal_actions=legal_actions,
                    )
                )

                self.assertIsNone(headless_state)
                self.assertEqual(default_done, headless_done)
                self.assertEqual(default_info, headless_info)
                self.assertEqual(default_engine.__dict__, headless_engine.__dict__)
                self.assertEqual(default_state, headless_engine._get_state())

            self.assertTrue(default_engine.game_over)
            self.assertTrue(headless_engine.game_over)

    def test_game_manager_performs_one_state_and_legal_action_scan_per_turn(self):
        """One automatic turn has no repeated legal scan or discarded snapshot."""
        engine = DominoEngine(player_count=2)
        manager = GameManager(engine, [FirstLegalAgent(), FirstLegalAgent()])

        with mock.patch.object(
            engine,
            "valid_actions",
            wraps=engine.valid_actions,
        ) as valid_actions, mock.patch.object(
            engine,
            "_get_state",
            wraps=engine._get_state,
        ) as get_state:
            manager.play_turn()

        self.assertEqual(valid_actions.call_count, 1)
        self.assertEqual(get_state.call_count, 1)

    def test_fixed_seed_dataset_payload_matches_legacy_equivalent_path(self):
        """The optimized manager preserves serialized supervised examples."""
        with self._legacy_step_patch():
            baseline = generate_dataset_game(game_index=0, seed=20260719)
        optimized = generate_dataset_game(game_index=0, seed=20260719)

        self.assertEqual(baseline, optimized)

    def test_fixed_seed_diagnostic_record_matches_legacy_equivalent_path(self):
        """The optimized pairwise loop preserves the complete game record."""
        random.seed(12345)
        np.random.seed(12345)
        with self._legacy_step_patch():
            baseline = play_game(
                RandomAgent(),
                RandomAgent(),
                agent_position=0,
            )

        random.seed(12345)
        np.random.seed(12345)
        optimized = play_game(
            RandomAgent(),
            RandomAgent(),
            agent_position=0,
        )
        self.assertEqual(baseline, optimized)

    def test_fixed_seed_rl_trajectory_matches_legacy_equivalent_path(self):
        """The optimized training loop preserves rewards and trajectory arrays."""
        network = PolicyNetwork(
            input_size=168,
            hidden1_size=8,
            hidden2_size=4,
            output_size=56,
            random_seed=7,
            device="cpu",
        )
        schema = REWARD_SCHEMAS[DEFAULT_REWARD_SCHEMA]

        random.seed(54321)
        np.random.seed(54321)
        with self._legacy_step_patch():
            baseline = _collect_self_play_steps(
                network,
                [],
                schema,
                DEFAULT_GAMMA,
            )

        random.seed(54321)
        np.random.seed(54321)
        optimized = _collect_self_play_steps(
            network,
            [],
            schema,
            DEFAULT_GAMMA,
        )

        baseline_samples, baseline_events, baseline_winner, baseline_position = baseline
        optimized_samples, optimized_events, optimized_winner, optimized_position = (
            optimized
        )
        self.assertEqual(baseline_events, optimized_events)
        self.assertEqual(baseline_winner, optimized_winner)
        self.assertEqual(baseline_position, optimized_position)
        self.assertEqual(len(baseline_samples), len(optimized_samples))
        for baseline_sample, optimized_sample in zip(
            baseline_samples,
            optimized_samples,
        ):
            self.assertTrue(np.array_equal(baseline_sample.x, optimized_sample.x))
            self.assertTrue(np.array_equal(
                baseline_sample.legal_mask,
                optimized_sample.legal_mask,
            ))
            self.assertEqual(
                (
                    baseline_sample.action_index,
                    baseline_sample.policy_reward,
                    baseline_sample.raw_reward,
                    baseline_sample.local_reward,
                    baseline_sample.terminal_reward,
                    baseline_sample.multiplier,
                    baseline_sample.option_count,
                ),
                (
                    optimized_sample.action_index,
                    optimized_sample.policy_reward,
                    optimized_sample.raw_reward,
                    optimized_sample.local_reward,
                    optimized_sample.terminal_reward,
                    optimized_sample.multiplier,
                    optimized_sample.option_count,
                ),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
