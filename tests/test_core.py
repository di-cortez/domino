"""
Sequential core tests for the engine, encoder, and training history.

Run from the repository root with:

    python tests/test_core.py
"""

import csv
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as host_np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.encoder import DominoEncoder
from agents.heuristic_agent import StrategicAgent
from agents.nn import GPU_ENABLED
from agents.rl_agent import RLAgent
from agents.rl_nn import PolicyNetwork
from diagnostics.pairwise import (
    save_csv,
    summarize_first_stock_draw_expansions,
    summarize_first_stock_draw_turns,
)
from middleware.domino_engine import DominoEngine, infer_dead_suits
from middleware.middleware import GameManager
from middleware.opponent_model import (
    ALL_TILES,
    CompactOpponentBelief,
    compute_opponent_suit_probabilities,
)

from math import comb

if GPU_ENABLED:
    import cupy as xp
else:
    import numpy as xp


class FirstLegalAgent:
    def choose_move(self, state, legal_actions):
        return legal_actions[0]


class NetworkThatMustNotRun:
    def forward(self, x):
        raise AssertionError("The network must not run for forced actions.")


class UniformPolicyNetwork:
    def forward(self, x):
        return host_np.ones((DominoEncoder.ACTION_SIZE, 1), dtype=float) / DominoEncoder.ACTION_SIZE


def _to_numpy(value):
    return value.get() if hasattr(value, "get") else value


def _run(name, fn):
    fn()
    print(f"OK - {name}")


def _base_probability_state():
    initial_hand = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (2, 2)]
    return {
        "game_id": 1,
        "ends": [],
        "current_player_hand": [list(tile) for tile in initial_hand],
        "current_player_initial_hand": [list(tile) for tile in initial_hand],
        "current_player_drawn_tiles": [],
        "current_player": 0,
        "turn": 0,
        "hand_sizes": [7, 7],
        "board_history": [],
        "stock_size": 14,
    }


def test_encoder_action_space_excludes_forced_actions():
    encoder = DominoEncoder()

    assert len(encoder.all_actions) == 56
    assert ("DRAW", None) not in encoder.all_actions
    assert None not in encoder.all_actions
    assert not encoder.is_policy_action(("DRAW", None))
    assert not encoder.is_policy_action(None)


def test_encoder_accepts_list_tiles_from_json():
    encoder = DominoEncoder()

    assert encoder._action_index(([0, 6], 1)) == encoder._action_index(((0, 6), 1))


def test_engine_requires_highest_opening_double_when_present():
    engine = DominoEngine(player_count=2)
    player = engine.current_player

    engine.ends = []
    engine.hands[player] = [(0, 0), (6, 6), (1, 2)]
    engine.required_opening_tile = (6, 6)

    assert engine.valid_actions(player) == [((6, 6), 0)]


def test_engine_game_ids_are_unique_across_instances():
    first = DominoEngine(player_count=2)
    second = DominoEngine(player_count=2)

    assert first.game_id != second.game_id


def test_infer_dead_suits_from_draw_and_pass_history():
    board_history = [((2, 3), 0), ("DRAW", None), None]

    dead_suits = infer_dead_suits(
        board_history=board_history,
        hand_sizes=[7, 7],
        current_player=0,
    )

    assert dead_suits[1] == {2, 3}
    assert dead_suits[0] == set()


def test_game_manager_training_history_uses_compact_engine_state():
    engine = DominoEngine(player_count=2)
    manager = GameManager(engine, [FirstLegalAgent(), FirstLegalAgent()])

    manager.play_turn()

    assert len(manager.training_history) == 1
    row = manager.training_history[0]

    assert "state" in row
    assert "target_action" in row
    assert "visual_chain" not in row["state"]
    assert "current_player_initial_hand" in row["state"]
    assert "current_player_drawn_tiles" in row["state"]


def test_exact_opponent_probabilities_match_initial_hypergeometric_formula():
    state = _base_probability_state()
    probabilities = compute_opponent_suit_probabilities(state)

    known_tiles = {tuple(tile) for tile in state["current_player_initial_hand"]}
    unknown_tiles = [tile for tile in ALL_TILES if tile not in known_tiles]
    unknown_count = len(unknown_tiles)
    denominator = comb(unknown_count, 7)

    for suit in range(7):
        suit_count = sum(1 for tile in unknown_tiles if suit in tile)
        non_suit_count = unknown_count - suit_count
        expected = 1.0
        if non_suit_count >= 7:
            expected = 1.0 - comb(non_suit_count, 7) / denominator
        assert abs(probabilities[suit] - expected) < 1e-12


def test_exact_opponent_pass_sets_playable_suit_probabilities_to_zero():
    state = _base_probability_state()
    state["current_player_hand"] = [
        tile for tile in state["current_player_initial_hand"] if tuple(tile) != (1, 2)
    ]
    state["ends"] = [1, 2]
    state["turn"] = 2
    state["hand_sizes"] = [6, 7]
    state["board_history"] = [
        [[1, 2], 0],
        None,
    ]

    probabilities = compute_opponent_suit_probabilities(state)

    assert probabilities[1] == 0.0
    assert probabilities[2] == 0.0


def test_exact_opponent_draw_reopens_suit_probabilities_after_no_legal_condition():
    state = _base_probability_state()
    state["current_player_hand"] = [
        tile for tile in state["current_player_initial_hand"] if tuple(tile) != (1, 2)
    ]
    state["ends"] = [1, 2]
    state["turn"] = 3
    state["hand_sizes"] = [6, 8]
    state["board_history"] = [
        [[1, 2], 0],
        ["DRAW", None],
        None,
    ]
    state["stock_size"] = 13

    probabilities = compute_opponent_suit_probabilities(state)

    assert probabilities[1] > 0.0
    assert probabilities[2] > 0.0


def test_exact_observer_draw_removes_private_tile_from_unknown_pool():
    before_state = _base_probability_state()
    before_probabilities = compute_opponent_suit_probabilities(before_state)

    after_state = _base_probability_state()
    drawn_tile = (6, 6)
    after_state["current_player_hand"] = (
        after_state["current_player_initial_hand"] + [list(drawn_tile)]
    )
    after_state["current_player_drawn_tiles"] = [list(drawn_tile)]
    after_state["turn"] = 1
    after_state["hand_sizes"] = [8, 7]
    after_state["board_history"] = [["DRAW", None]]
    after_state["stock_size"] = 13

    after_probabilities = compute_opponent_suit_probabilities(after_state)

    assert after_probabilities[6] < before_probabilities[6]


def test_strategic_agent_uses_response_then_mobility_then_pip_sum_filters():
    agent = StrategicAgent()
    state = {
        "opponent_suit_probabilities": [0.00, 0.20, 0.27, 0.30, 0.45, 0.70, 0.00],
        "ends": [0, 6],
        "current_player_hand": [[0, 1], [0, 2], [0, 3], [0, 4]],
        "current_player": 0,
        "hand_sizes": [4, 7],
        "board_history": [],
        "stock_size": 14,
    }
    legal_actions = [
        ((0, 1), 0),
        ((0, 2), 0),
        ((0, 3), 0),
        ((0, 4), 0),
    ]

    assert agent.choose_move(state, legal_actions) == ((0, 3), 0)


def test_rl_agent_skips_network_for_forced_actions():
    forced_cases = [
        ([("DRAW", None)], ("DRAW", None)),
        ([None], None),
        ([((6, 6), 0)], ((6, 6), 0)),
    ]

    for legal_actions, expected_action in forced_cases:
        agent = RLAgent(NetworkThatMustNotRun(), mode="training")

        chosen = agent.choose_move(state={}, legal_actions=legal_actions)

        assert chosen == expected_action
        assert agent.trajectory == []


def test_rl_agent_saves_legal_mask_for_real_decision():
    encoder = DominoEncoder()
    agent = RLAgent(UniformPolicyNetwork(), mode="training")
    state = _base_probability_state()
    legal_actions = [((0, 0), 0), ((0, 1), 0)]

    chosen = agent.choose_move(state=state, legal_actions=legal_actions)

    assert chosen in legal_actions
    assert len(agent.trajectory) == 1

    x, action_index, legal_mask, reward = agent.trajectory[0]
    legal_mask = _to_numpy(legal_mask)

    assert x.shape == (encoder.VECTOR_SIZE, 1)
    assert legal_mask.shape == (encoder.ACTION_SIZE, 1)
    assert legal_mask.sum() == 2.0
    assert legal_mask[action_index, 0] == 1.0
    assert reward == 0.0


def test_policy_gradient_updates_only_legal_policy_biases():
    network = PolicyNetwork(
        input_size=4,
        hidden1_size=5,
        hidden2_size=3,
        output_size=DominoEncoder.ACTION_SIZE,
        learning_rate=0.1,
    )
    x_batch = xp.ones((4, 1))
    legal_mask = xp.zeros((DominoEncoder.ACTION_SIZE, 1))
    legal_mask[3, 0] = 1.0
    legal_mask[8, 0] = 1.0

    network.forward(x_batch)
    b3_before = _to_numpy(network.b3).copy()

    network.backward_policy_gradient(
        action_indices=[3],
        advantages=xp.ones((1, 1)),
        legal_masks=legal_mask,
        returns=None,
        entropy_coef=0.0,
        value_coef=0.0,
        clip_grad_norm=None,
    )

    b3_after = _to_numpy(network.b3)
    for index in range(DominoEncoder.ACTION_SIZE):
        if index not in (3, 8):
            assert b3_after[index, 0] == b3_before[index, 0]

    assert not host_np.allclose(b3_after[[3, 8], 0], b3_before[[3, 8], 0])


def test_policy_gradient_rejects_single_action_mask():
    network = PolicyNetwork(
        input_size=4,
        hidden1_size=5,
        hidden2_size=3,
        output_size=DominoEncoder.ACTION_SIZE,
        learning_rate=0.1,
    )
    x_batch = xp.ones((4, 1))
    legal_mask = xp.zeros((DominoEncoder.ACTION_SIZE, 1))
    legal_mask[3, 0] = 1.0

    network.forward(x_batch)

    try:
        network.backward_policy_gradient(
            action_indices=[3],
            advantages=xp.ones((1, 1)),
            legal_masks=legal_mask,
            returns=None,
            entropy_coef=0.0,
            value_coef=0.0,
            clip_grad_norm=None,
        )
    except ValueError as exc:
        assert "at least two legal policy actions" in str(exc)
    else:
        raise AssertionError("Expected ValueError for a single-action legal mask.")


def test_first_stock_draw_summary_ignores_games_without_draws():
    games = [
        {"first_stock_draw_turn": None},
        {"first_stock_draw_turn": 2},
        {"first_stock_draw_turn": 5},
        {"first_stock_draw_turn": 5},
    ]

    summary = summarize_first_stock_draw_turns(games)

    assert summary["games"] == 4
    assert summary["games_with_stock_draw"] == 3
    assert summary["games_without_stock_draw"] == 1
    assert summary["stock_draw_rate"] == 0.75
    assert summary["mean_turn"] == 4.0
    assert summary["median_turn"] == 5.0
    assert summary["min_turn"] == 2
    assert summary["max_turn"] == 5
    assert summary["turn_histogram"] == {"2": 1, "5": 2}


def test_compact_hidden_draw_records_expansion_count():
    belief = CompactOpponentBelief(
        observer_initial_hand=ALL_TILES[:7],
        opponent_hand_size=1,
        rng=random.Random(0),
        max_enumerated_hands=1000,
        particle_count=100,
    )

    next_belief = belief.opponent_hidden_draw()

    assert next_belief.mode == "enumerated_exact"
    assert next_belief.compact_hidden_draw_final_state_count == 210


def test_compact_hidden_draw_records_particle_expansion_count():
    belief = CompactOpponentBelief(
        observer_initial_hand=ALL_TILES[:7],
        opponent_hand_size=1,
        rng=random.Random(0),
        max_enumerated_hands=1,
        particle_count=100,
    )

    next_belief = belief.opponent_hidden_draw()

    assert next_belief.mode == "particle_approximate"
    assert next_belief.compact_hidden_draw_final_state_count == 210


def test_first_stock_draw_expansion_summary_ignores_games_without_counts():
    games = [
        {"first_stock_draw_final_state_count": None},
        {"first_stock_draw_final_state_count": 210},
        {"first_stock_draw_final_state_count": 210},
        {"first_stock_draw_final_state_count": 840},
    ]

    summary = summarize_first_stock_draw_expansions(games)

    assert summary["games"] == 4
    assert summary["games_with_count"] == 3
    assert summary["games_without_count"] == 1
    assert summary["count_rate"] == 0.75
    assert summary["mean_final_state_count"] == 420.0
    assert summary["median_final_state_count"] == 210.0
    assert summary["min_final_state_count"] == 210
    assert summary["max_final_state_count"] == 840
    assert summary["final_state_count_histogram"] == {"210": 2, "840": 1}


def test_pairwise_csv_writes_initial_hands_as_json_arrays():
    games = [
        {
            "game": 1,
            "agent_position": 0,
            "result": "win",
            "turns": 12,
            "first_stock_draw_turn": 4,
            "first_stock_draw_final_state_count": 210,
            "agent_initial_hand": [[6, 6], [0, 1]],
            "opponent_initial_hand": [[5, 5], [2, 3]],
            "agent_remaining_pips": 0,
            "opponent_remaining_pips": 10,
        }
    ]

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "games.csv"
        save_csv(games, path)

        with open(path, newline="") as f:
            row = next(csv.DictReader(f))

    assert json.loads(row["agent_initial_hand"]) == [[6, 6], [0, 1]]
    assert json.loads(row["opponent_initial_hand"]) == [[5, 5], [2, 3]]
    assert row["first_stock_draw_final_state_count"] == "210"


def main():
    tests = [
        ("encoder action space", test_encoder_action_space_excludes_forced_actions),
        ("encoder JSON tile actions", test_encoder_accepts_list_tiles_from_json),
        ("opening double rule", test_engine_requires_highest_opening_double_when_present),
        ("unique game ids", test_engine_game_ids_are_unique_across_instances),
        ("dead suit inference", test_infer_dead_suits_from_draw_and_pass_history),
        ("training history shape", test_game_manager_training_history_uses_compact_engine_state),
        (
            "exact probability initialization",
            test_exact_opponent_probabilities_match_initial_hypergeometric_formula,
        ),
        ("exact probability pass", test_exact_opponent_pass_sets_playable_suit_probabilities_to_zero),
        (
            "exact probability draw",
            test_exact_opponent_draw_reopens_suit_probabilities_after_no_legal_condition,
        ),
        (
            "exact private draw",
            test_exact_observer_draw_removes_private_tile_from_unknown_pool,
        ),
        (
            "strategic probability filters",
            test_strategic_agent_uses_response_then_mobility_then_pip_sum_filters,
        ),
        ("RL forced actions skip network", test_rl_agent_skips_network_for_forced_actions),
        ("RL trajectory legal mask", test_rl_agent_saves_legal_mask_for_real_decision),
        ("masked policy gradient", test_policy_gradient_updates_only_legal_policy_biases),
        ("invalid policy mask", test_policy_gradient_rejects_single_action_mask),
        ("first stock draw summary", test_first_stock_draw_summary_ignores_games_without_draws),
        (
            "compact hidden draw expansion",
            test_compact_hidden_draw_records_expansion_count,
        ),
        (
            "compact hidden draw particle expansion",
            test_compact_hidden_draw_records_particle_expansion_count,
        ),
        (
            "first stock draw expansion summary",
            test_first_stock_draw_expansion_summary_ignores_games_without_counts,
        ),
        ("pairwise CSV initial hands", test_pairwise_csv_writes_initial_hands_as_json_arrays),
    ]

    for name, fn in tests:
        _run(name, fn)

    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
