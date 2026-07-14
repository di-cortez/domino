"""
Sequential core tests for the engine, encoder, and training history.

Run from the repository root with:

    python tests/test_core.py
"""

import csv
import json
import sys
import tempfile
from itertools import combinations
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
    SUIT_MASKS,
    ExactOpponentModel,
    MuOpponentBelief,
    ProbabilityStage,
    SlotOpponentBelief,
    compute_opponent_suit_probabilities,
    mask_from_tiles,
    reconstruct_public_actions,
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

from math import comb, factorial

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


class FixedStrategicOpponentModel:
    """Small exact-model stand-in used to isolate heuristic tie-break tests."""

    def __init__(self, probabilities):
        self.probabilities = list(probabilities)

    def update(self, state):
        return list(self.probabilities)

    def probability_can_play(self, ends):
        left, right = ends
        if left == right:
            return self.probabilities[left]
        return 1.0 - (
            (1.0 - self.probabilities[left])
            * (1.0 - self.probabilities[right])
        )


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


def _uniform_mu_belief(tiles, hand_size):
    """Return a small uniform mu belief over all hands from ``tiles``."""
    unknown_mask = mask_from_tiles(tiles)
    indices = [
        index
        for index, tile in enumerate(ALL_TILES)
        if tile in set(tiles)
    ]
    weights = {}
    for selected in combinations(indices, hand_size):
        hand_mask = sum(1 << index for index in selected)
        weights[hand_mask] = 1
    return MuOpponentBelief.from_weights(
        unknown_mask=unknown_mask,
        opponent_hand_size=hand_size,
        weights=weights,
    )


def test_mu_belief_exact_integer_operations():
    tiles = [(0, 0), (0, 1), (1, 1), (2, 2)]

    initial = _uniform_mu_belief(tiles, 2)
    assert initial.state_count == comb(4, 2)
    assert all(isinstance(weight, int) and weight == 1 for weight in initial.weights.values())

    conditioned = _uniform_mu_belief(tiles, 2)
    conditioned.condition_no_legal(0, 0)
    expected_hand = mask_from_tiles([(1, 1), (2, 2)])
    assert conditioned.weights == {expected_hand: 1}

    observer_conditioned = _uniform_mu_belief(tiles, 2)
    observer_conditioned.observer_known_draw((0, 0))
    assert not observer_conditioned.unknown_mask & mask_from_tiles([(0, 0)])
    assert observer_conditioned.state_count == comb(3, 2)

    revealed = _uniform_mu_belief(tiles, 2)
    revealed.opponent_reveals_and_plays((0, 0))
    assert revealed.opponent_hand_size == 1
    assert revealed.state_count == 3
    assert set(revealed.weights.values()) == {1}

    drawn = _uniform_mu_belief(tiles, 1)
    drawn.opponent_hidden_draw()
    assert drawn.opponent_hand_size == 2
    assert drawn.state_count == comb(4, 2)
    assert set(drawn.weights.values()) == {2}


def test_mu_probability_can_play_uses_joint_distribution():
    tile_00 = mask_from_tiles([(0, 0)])
    tile_11 = mask_from_tiles([(1, 1)])
    belief = MuOpponentBelief.from_weights(
        unknown_mask=tile_00 | tile_11,
        opponent_hand_size=1,
        weights={tile_00: 1, tile_11: 1},
    )

    assert belief.suit_probabilities()[0] == 0.5
    assert belief.suit_probabilities()[1] == 0.5
    assert belief.probability_can_play((0, 1)) == 1.0


def test_slot_initial_count_and_dp_conversion_match_mu():
    observer_hand = ALL_TILES[:7]
    slot = SlotOpponentBelief(observer_hand)
    mu = MuOpponentBelief.from_initial(observer_hand)

    assert slot.mode == "slots_exact"
    assert slot.profile_count == 1
    assert slot.opponent_hand_size == 7
    assert slot.assignment_weight == factorial(21) // factorial(14)

    converted = slot.to_hand_weights_dp()
    assert len(converted) == comb(21, 7)
    assert set(converted.values()) == {factorial(7)}
    assert slot.suit_probabilities() == mu.suit_probabilities()


def test_slot_cohorts_preserve_temporal_draw_restrictions():
    tiles = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (3, 3)]
    unknown_mask = mask_from_tiles(tiles)
    slot = SlotOpponentBelief.from_profiles(
        unknown_mask=unknown_mask,
        opponent_hand_size=1,
        profiles={(unknown_mask,): 1},
    )

    slot.condition_no_legal(0, 0)
    first_cohort_domain = next(iter(slot.profiles))[0]
    assert first_cohort_domain & SUIT_MASKS[0] == 0

    slot.opponent_hidden_draw()
    assert slot.suit_probabilities()[0] > 0.0

    slot.condition_no_legal(1, 1)
    slot.opponent_hidden_draw()
    profile = next(iter(slot.profiles))
    expected_domains = sorted((
        unknown_mask & ~SUIT_MASKS[0] & ~SUIT_MASKS[1],
        unknown_mask & ~SUIT_MASKS[1],
        unknown_mask,
    ))
    assert list(profile) == expected_domains

    weights = slot.to_hand_weights_dp()
    mu = MuOpponentBelief.from_weights(
        unknown_mask=slot.unknown_mask,
        opponent_hand_size=slot.opponent_hand_size,
        weights=weights,
    )
    assert slot.suit_probabilities() == mu.suit_probabilities()
    assert slot.probability_can_play((2, 3)) == mu.probability_can_play((2, 3))


def test_slot_play_branch_multiplicity_matches_mu():
    tiles = [(0, 0), (1, 1), (2, 2), (3, 3)]
    unknown_mask = mask_from_tiles(tiles)
    slot = SlotOpponentBelief.from_profiles(
        unknown_mask=unknown_mask,
        opponent_hand_size=2,
        profiles={(unknown_mask, unknown_mask): 1},
    )
    mu = MuOpponentBelief.from_weights(
        unknown_mask=unknown_mask,
        opponent_hand_size=2,
        weights=slot.to_hand_weights_dp(),
    )

    slot.opponent_reveals_and_plays((0, 0))
    mu.opponent_reveals_and_plays((0, 0))

    assert slot.to_hand_weights_dp() == mu.weights
    assert next(iter(slot.profiles.values())) == 2


def test_slot_known_tile_removes_hall_infeasible_profiles():
    tile_a = mask_from_tiles([(0, 0)])
    tile_b = mask_from_tiles([(1, 1)])
    tile_c = mask_from_tiles([(2, 2)])
    unknown_mask = tile_a | tile_b | tile_c
    slot = SlotOpponentBelief.from_profiles(
        unknown_mask=unknown_mask,
        opponent_hand_size=2,
        profiles={
            tuple(sorted((tile_a | tile_b, tile_a | tile_b))): 1,
            tuple(sorted((tile_a | tile_c, tile_b | tile_c))): 1,
        },
    )

    slot.observer_known_draw((1, 1))

    assert slot.profile_count == 1
    assert slot.assignment_weight > 0


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

    model = ExactOpponentModel()
    result = model.update_detailed(state)
    probabilities = result.probabilities
    trace = result.completed_turn_traces[-1]

    assert probabilities[1] == 0.0
    assert probabilities[2] == 0.0
    assert trace.after_negative_evidence is not None
    assert trace.after_draw is None
    assert trace.end_turn is not None
    assert trace.end_turn.same_as_previous


_NO_FINAL_DRAW_ACTION = object()


def _draw_turn_state(include_final_action=_NO_FINAL_DRAW_ACTION):
    """Return an observer state ending during or after one opponent draw turn."""
    state = _base_probability_state()
    state["current_player_hand"] = [
        tile for tile in state["current_player_initial_hand"] if tuple(tile) != (1, 2)
    ]
    state["ends"] = [1, 2]
    state["observer_player"] = 0
    state["history_current_player"] = 1
    state["turn"] = 2
    state["hand_sizes"] = [6, 8]
    state["board_history"] = [
        [[1, 2], 0],
        ["DRAW", None],
    ]
    state["stock_size"] = 13

    if include_final_action is not _NO_FINAL_DRAW_ACTION:
        state["history_current_player"] = 0
        state["turn"] = 3
        action = None if include_final_action is False else include_final_action
        state["board_history"].append(action)
        if action is not None:
            state["ends"] = [3, 2]
            state["hand_sizes"] = [6, 7]
    return state


def test_draw_pass_exposes_negative_draw_and_end_turn_probabilities():
    model = ExactOpponentModel()
    partial_state = _draw_turn_state()

    partial = model.update_detailed(partial_state)
    repeated = model.update_detailed(partial_state)

    assert [snapshot.stage for snapshot in partial.new_snapshots] == [
        ProbabilityStage.END_TURN,
        ProbabilityStage.AFTER_NEGATIVE_EVIDENCE,
        ProbabilityStage.AFTER_DRAW,
    ]
    assert repeated.new_snapshots == ()
    assert repeated.completed_turn_traces == ()

    full_state = _draw_turn_state(include_final_action=False)
    completed = model.update_detailed(full_state)
    trace = completed.completed_turn_traces[0]

    assert trace.public_turn == 2
    assert trace.after_negative_evidence.probabilities[1] == 0.0
    assert trace.after_negative_evidence.probabilities[2] == 0.0
    assert trace.after_draw.probabilities[1] > 0.0
    assert trace.after_draw.probabilities[2] > 0.0
    assert trace.end_turn.probabilities[1] == 0.0
    assert trace.end_turn.probabilities[2] == 0.0
    assert completed.probabilities[1] == 0.0
    assert completed.probabilities[2] == 0.0

    snapshots = model.consume_new_snapshots()
    assert len(snapshots) == 4
    assert model.consume_new_snapshots() == []
    model.reset()
    assert model.last_snapshot is None
    assert model.last_completed_turn_trace is None
    assert model.turn_trace_history == []
    assert not model.switched_to_mu


def test_draw_play_exposes_three_stages_and_reveals_drawn_tile():
    state = _draw_turn_state(include_final_action=[[1, 3], 0])
    model = ExactOpponentModel()

    result = model.update_detailed(state)
    trace = result.completed_turn_traces[-1]

    assert trace.after_negative_evidence is not None
    assert trace.after_draw is not None
    assert trace.end_turn is not None
    assert trace.after_negative_evidence.stage is ProbabilityStage.AFTER_NEGATIVE_EVIDENCE
    assert trace.after_draw.stage is ProbabilityStage.AFTER_DRAW
    assert trace.end_turn.stage is ProbabilityStage.END_TURN
    assert trace.after_negative_evidence.probabilities[1] == 0.0
    assert trace.after_draw.probabilities[1] > 0.0


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
    fixed_probabilities = [0.00, 0.20, 0.27, 0.30, 0.45, 0.70, 0.00]
    agent.opponent_model = FixedStrategicOpponentModel(fixed_probabilities)
    state = {
        "opponent_suit_probabilities": fixed_probabilities,
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


def test_hybrid_switches_once_at_threshold_and_never_returns_to_slots():
    tiles = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    unknown_mask = mask_from_tiles(tiles)
    slot = SlotOpponentBelief.from_profiles(
        unknown_mask=unknown_mask,
        opponent_hand_size=2,
        profiles={(unknown_mask, unknown_mask): 1},
    )
    model = ExactOpponentModel(switch_to_mu_max_hands=10)
    model._belief = slot

    model._maybe_switch_to_mu(public_turn=4, terminal_turn=False)

    assert model.mode == "mu_exact"
    assert model.switched_to_mu
    assert model.switch_turn == 4
    assert model.switch_upper_bound == comb(5, 2)
    assert model.switch_mu_state_count == comb(5, 2)
    first_switch_time = model.switch_conversion_time_ms

    model._belief.opponent_hidden_draw()
    model._maybe_switch_to_mu(public_turn=5, terminal_turn=False)

    assert model.mode == "mu_exact"
    assert model.switch_turn == 4
    assert model.switch_conversion_time_ms == first_switch_time


def test_hybrid_does_not_switch_above_threshold_or_on_terminal_turn():
    tiles = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    unknown_mask = mask_from_tiles(tiles)

    above_threshold = ExactOpponentModel(switch_to_mu_max_hands=9)
    above_threshold._belief = SlotOpponentBelief.from_profiles(
        unknown_mask=unknown_mask,
        opponent_hand_size=2,
        profiles={(unknown_mask, unknown_mask): 1},
    )
    above_threshold._maybe_switch_to_mu(public_turn=1, terminal_turn=False)
    assert above_threshold.mode == "slots_exact"

    terminal = ExactOpponentModel(switch_to_mu_max_hands=10)
    terminal._belief = SlotOpponentBelief.from_profiles(
        unknown_mask=unknown_mask,
        opponent_hand_size=2,
        profiles={(unknown_mask, unknown_mask): 1},
    )
    terminal._maybe_switch_to_mu(public_turn=1, terminal_turn=True)
    assert terminal.mode == "slots_exact"


def test_opponent_model_does_not_trust_stale_state_probability_output():
    state = _base_probability_state()
    model = ExactOpponentModel()
    initial = model.update_detailed(state)
    assert initial.new_snapshots == ()

    state["current_player_hand"] = [
        tile for tile in state["current_player_initial_hand"] if tuple(tile) != (1, 2)
    ]
    state["ends"] = [1, 2]
    state["history_current_player"] = 1
    state["current_player"] = 1
    state["observer_player"] = 0
    state["turn"] = 1
    state["hand_sizes"] = [6, 7]
    state["board_history"] = [[[1, 2], 0]]
    state["opponent_suit_probabilities"] = [0.123] * 7

    updated = model.update_detailed(state)

    assert len(updated.new_snapshots) == 1
    assert updated.probabilities != tuple([0.123] * 7)
    assert state["opponent_model_metadata"]["processed_history_length"] == 1


def test_terminal_history_reconstructs_the_non_advanced_final_actor():
    state = {
        "game_over": True,
        "history_current_player": 1,
        "current_player": 1,
        "hand_sizes": [3, 0],
        "board_history": [
            [[6, 6], 0],
            [[3, 6], 0],
        ],
    }

    actions = reconstruct_public_actions(state)

    assert actions[0].actor == 0
    assert actions[1].actor == 1


def test_hidden_draw_record_exposes_exact_raw_hand_upper_bound():
    model = ExactOpponentModel()
    model.update_detailed(_draw_turn_state())

    record = model.hidden_draw_state_records[0]
    assert record["turn"] == 2
    assert record["raw_hand_upper_bound"] == comb(21, 8)
    assert record["final_state_count"] == record["raw_hand_upper_bound"]
    assert record["resulting_mode"] == "slots_exact"


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
        ("mu exact operations", test_mu_belief_exact_integer_operations),
        ("mu joint play probability", test_mu_probability_can_play_uses_joint_distribution),
        ("slot initial conversion", test_slot_initial_count_and_dp_conversion_match_mu),
        ("slot temporal cohorts", test_slot_cohorts_preserve_temporal_draw_restrictions),
        ("slot play multiplicity", test_slot_play_branch_multiplicity_matches_mu),
        ("slot infeasible profile filter", test_slot_known_tile_removes_hall_infeasible_profiles),
        ("exact probability pass", test_exact_opponent_pass_sets_playable_suit_probabilities_to_zero),
        (
            "draw-pass probability stages",
            test_draw_pass_exposes_negative_draw_and_end_turn_probabilities,
        ),
        (
            "draw-play probability stages",
            test_draw_play_exposes_three_stages_and_reveals_drawn_tile,
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
            "hybrid one-way threshold switch",
            test_hybrid_switches_once_at_threshold_and_never_returns_to_slots,
        ),
        (
            "hybrid switch guards",
            test_hybrid_does_not_switch_above_threshold_or_on_terminal_turn,
        ),
        (
            "opponent cache invalidation",
            test_opponent_model_does_not_trust_stale_state_probability_output,
        ),
        (
            "terminal actor reconstruction",
            test_terminal_history_reconstructs_the_non_advanced_final_actor,
        ),
        ("hidden draw upper bound", test_hidden_draw_record_exposes_exact_raw_hand_upper_bound),
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
