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
from agents.rl_agent import RLAgent, TrajectoryStep
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
from training.self_play import (
    EVENT_REWARD_DECAY,
    LEARNER_DRAW_PENALTY,
    LEARNER_PASS_PENALTY,
    OPPONENT_DRAW_REWARD,
    OPPONENT_PASS_REWARD,
    EventStats,
    TrainingSample,
    _choice_multiplier,
    _event_reward_for_action,
    _finish_episode_with_rewards,
    _reward_signal_summary,
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


def _masked_action_probability(network, x_batch, legal_mask, action_index):
    network.forward(x_batch)
    logits = network.cache["Z3"]
    masked_logits = xp.where(legal_mask > 0, logits, -xp.inf)
    shifted = masked_logits - xp.max(masked_logits, axis=0, keepdims=True)
    masked_policy = xp.exp(shifted) / xp.sum(xp.exp(shifted), axis=0, keepdims=True)
    return float(_to_numpy(masked_policy[action_index, 0]))


def _small_policy_network(input_size=4, hidden1_size=5, hidden2_size=3, output_size=56, learning_rate=0.1):
    """Build a deterministic tiny policy network without invoking backend RNG."""
    network = PolicyNetwork.__new__(PolicyNetwork)
    network.lr = learning_rate
    network.W1 = xp.zeros((hidden1_size, input_size))
    network.b1 = xp.zeros((hidden1_size, 1))
    network.W2 = xp.zeros((hidden2_size, hidden1_size))
    network.b2 = xp.zeros((hidden2_size, 1))
    network.W3 = xp.zeros((output_size, hidden2_size))
    network.b3 = xp.zeros((output_size, 1))
    network.cache = {}
    return network


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

    step = agent.trajectory[0]
    legal_mask = _to_numpy(step.legal_mask)

    assert step.x.shape == (encoder.VECTOR_SIZE, 1)
    assert legal_mask.shape == (encoder.ACTION_SIZE, 1)
    assert legal_mask.sum() == 2.0
    assert legal_mask[step.action_index, 0] == 1.0
    assert step.decision_turn == state["turn"]
    assert step.option_count == 2
    assert step.local_reward == 0.0


def test_policy_gradient_updates_only_legal_policy_biases():
    network = _small_policy_network(output_size=DominoEncoder.ACTION_SIZE)
    x_batch = xp.ones((4, 1))
    legal_mask = xp.zeros((DominoEncoder.ACTION_SIZE, 1))
    legal_mask[3, 0] = 1.0
    legal_mask[8, 0] = 1.0

    network.forward(x_batch)
    b3_before = _to_numpy(network.b3).copy()

    network.backward_policy_gradient(
        action_indices=[3],
        policy_rewards=xp.ones((1, 1)),
        legal_masks=legal_mask,
        entropy_coef=0.0,
        clip_grad_norm=None,
    )

    b3_after = _to_numpy(network.b3)
    for index in range(DominoEncoder.ACTION_SIZE):
        if index not in (3, 8):
            assert b3_after[index, 0] == b3_before[index, 0]

    assert not host_np.allclose(b3_after[[3, 8], 0], b3_before[[3, 8], 0])


def test_policy_gradient_rejects_single_action_mask():
    network = _small_policy_network(output_size=DominoEncoder.ACTION_SIZE)
    x_batch = xp.ones((4, 1))
    legal_mask = xp.zeros((DominoEncoder.ACTION_SIZE, 1))
    legal_mask[3, 0] = 1.0

    network.forward(x_batch)

    try:
        network.backward_policy_gradient(
            action_indices=[3],
            policy_rewards=xp.ones((1, 1)),
            legal_masks=legal_mask,
            entropy_coef=0.0,
            clip_grad_norm=None,
        )
    except ValueError as exc:
        assert "at least two legal policy actions" in str(exc)
    else:
        raise AssertionError("Expected ValueError for a single-action legal mask.")


def test_decayed_event_reward_exponents():
    cases = [(11, 0.10), (12, 0.09), (13, 0.081)]

    for event_turn, expected in cases:
        agent = RLAgent(UniformPolicyNetwork(), mode="training")
        agent.trajectory = [
            TrajectoryStep(None, 0, None, decision_turn=10, option_count=2),
        ]

        agent.add_decayed_event_reward(event_turn, 0.10, EVENT_REWARD_DECAY)

        assert abs(agent.trajectory[0].local_reward - expected) < 1e-12


def test_event_reward_signs_and_counts():
    stats = EventStats()

    assert _event_reward_for_action(1, 0, ("DRAW", None), stats) == OPPONENT_DRAW_REWARD
    assert _event_reward_for_action(1, 0, None, stats) == OPPONENT_PASS_REWARD
    assert _event_reward_for_action(0, 0, ("DRAW", None), stats) == LEARNER_DRAW_PENALTY
    assert _event_reward_for_action(0, 0, None, stats) == LEARNER_PASS_PENALTY

    assert stats.opponent_draws == 1
    assert stats.opponent_passes == 1
    assert stats.learner_draws == 1
    assert stats.learner_passes == 1


def test_multiple_events_and_all_previous_decisions_receive_rewards():
    agent = RLAgent(UniformPolicyNetwork(), mode="training")
    agent.trajectory = [
        TrajectoryStep(None, 0, None, decision_turn=10, option_count=2),
        TrajectoryStep(None, 0, None, decision_turn=12, option_count=2),
    ]

    agent.add_decayed_event_reward(13, 0.10, EVENT_REWARD_DECAY)
    agent.add_decayed_event_reward(14, -0.02, EVENT_REWARD_DECAY)

    assert abs(agent.trajectory[0].local_reward - (0.081 - 0.01458)) < 1e-12
    assert abs(agent.trajectory[1].local_reward - (0.10 - 0.018)) < 1e-12


def test_event_reward_without_decisions_is_noop():
    agent = RLAgent(UniformPolicyNetwork(), mode="training")

    agent.add_decayed_event_reward(3, 0.10, EVENT_REWARD_DECAY)

    assert agent.trajectory == []


def test_terminal_reward_is_uniform_before_local_shaping():
    agent = RLAgent(UniformPolicyNetwork(), mode="training")
    agent.trajectory = [
        TrajectoryStep(None, 0, None, decision_turn=1, option_count=2, local_reward=0.10),
        TrajectoryStep(None, 0, None, decision_turn=3, option_count=2, local_reward=-0.05),
    ]

    steps = agent.finish_episode(0.50)

    assert steps[0].terminal_reward == 0.50
    assert steps[1].terminal_reward == 0.50
    assert abs(steps[0].raw_reward - 0.60) < 1e-12
    assert abs(steps[1].raw_reward - 0.45) < 1e-12


def test_option_multipliers_apply_after_terminal_and_local_rewards():
    agent = RLAgent(UniformPolicyNetwork(), mode="training")
    agent.trajectory = [
        TrajectoryStep(None, 0, None, decision_turn=1, option_count=2, local_reward=0.10),
        TrajectoryStep(None, 0, None, decision_turn=1, option_count=3, local_reward=0.10),
        TrajectoryStep(None, 0, None, decision_turn=1, option_count=4, local_reward=0.10),
        TrajectoryStep(None, 0, None, decision_turn=1, option_count=5, local_reward=0.10),
        TrajectoryStep(None, 0, None, decision_turn=1, option_count=6, local_reward=0.10),
    ]

    samples = _finish_episode_with_rewards(agent, 0.50)

    assert _choice_multiplier(2) == 1.0
    assert _choice_multiplier(3) == 2.0
    assert _choice_multiplier(4) == 5.0
    assert _choice_multiplier(5) == 10.0
    assert _choice_multiplier(6) == 10.0
    assert [sample.policy_reward for sample in samples] == [0.60, 1.20, 3.00, 6.00, 6.00]


def test_positive_reward_increases_chosen_masked_probability():
    network = _small_policy_network()
    x_batch = xp.ones((4, 1))
    legal_mask = xp.zeros((56, 1))
    legal_mask[3, 0] = 1.0
    legal_mask[8, 0] = 1.0
    network.W1 = xp.zeros_like(network.W1)
    network.W2 = xp.zeros_like(network.W2)
    network.W3 = xp.zeros_like(network.W3)

    before = _masked_action_probability(network, x_batch, legal_mask, 3)
    network.backward_policy_gradient(
        action_indices=[3],
        policy_rewards=xp.ones((1, 1)),
        legal_masks=legal_mask,
        entropy_coef=0.0,
        clip_grad_norm=None,
    )
    after = _masked_action_probability(network, x_batch, legal_mask, 3)

    assert after > before


def test_negative_reward_decreases_chosen_masked_probability():
    network = _small_policy_network()
    x_batch = xp.ones((4, 1))
    legal_mask = xp.zeros((56, 1))
    legal_mask[3, 0] = 1.0
    legal_mask[8, 0] = 1.0
    network.W1 = xp.zeros_like(network.W1)
    network.W2 = xp.zeros_like(network.W2)
    network.W3 = xp.zeros_like(network.W3)

    before = _masked_action_probability(network, x_batch, legal_mask, 3)
    network.backward_policy_gradient(
        action_indices=[3],
        policy_rewards=-xp.ones((1, 1)),
        legal_masks=legal_mask,
        entropy_coef=0.0,
        clip_grad_norm=None,
    )
    after = _masked_action_probability(network, x_batch, legal_mask, 3)

    assert after < before


def test_policy_checkpoint_saves_policy_weights_and_loads_legacy_value_keys():
    network = _small_policy_network(learning_rate=0.01)

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "policy.npz"
        network.save(path)
        saved = host_np.load(path)
        assert set(saved.files) == {"W1", "b1", "W2", "b2", "W3", "b3"}

        legacy_path = Path(folder) / "legacy.npz"
        host_np.savez(
            legacy_path,
            W1=_to_numpy(network.W1),
            b1=_to_numpy(network.b1),
            W2=_to_numpy(network.W2),
            b2=_to_numpy(network.b2),
            W3=_to_numpy(network.W3),
            b3=_to_numpy(network.b3),
            Wv=host_np.zeros((1, 3)),
            bv=host_np.zeros((1, 1)),
        )
        loaded = PolicyNetwork.load(legacy_path)

    assert not hasattr(loaded, "Wv")
    assert loaded.W1.shape == network.W1.shape


def test_reward_signal_summary_classifies_rewards():
    samples = [
        TrainingSample(None, 0, None, 1.0, 1.0, 0.20, 0.80, 1.0, 2),
        TrainingSample(None, 0, None, 0.0, 0.0, 0.00, 0.00, 1.0, 2),
        TrainingSample(None, 0, None, -1.0, -1.0, -0.10, -0.90, 1.0, 2),
    ]

    summary = _reward_signal_summary(samples)

    assert abs(summary["good_pct"] - (100.0 / 3.0)) < 1e-12
    assert abs(summary["neutral_pct"] - (100.0 / 3.0)) < 1e-12
    assert abs(summary["bad_pct"] - (100.0 / 3.0)) < 1e-12
    assert abs(summary["local_mean"] - (0.10 / 3.0)) < 1e-12


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
        ("decayed event reward exponents", test_decayed_event_reward_exponents),
        ("event reward signs", test_event_reward_signs_and_counts),
        (
            "multiple decayed events",
            test_multiple_events_and_all_previous_decisions_receive_rewards,
        ),
        ("event reward no decisions", test_event_reward_without_decisions_is_noop),
        ("uniform terminal reward", test_terminal_reward_is_uniform_before_local_shaping),
        ("option reward multipliers", test_option_multipliers_apply_after_terminal_and_local_rewards),
        (
            "positive reward gradient",
            test_positive_reward_increases_chosen_masked_probability,
        ),
        (
            "negative reward gradient",
            test_negative_reward_decreases_chosen_masked_probability,
        ),
        ("policy checkpoint keys", test_policy_checkpoint_saves_policy_weights_and_loads_legacy_value_keys),
        ("reward signal summary", test_reward_signal_summary_classifies_rewards),
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
