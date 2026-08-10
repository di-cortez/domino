"""
Sequential core tests for the engine, encoder, and training history.

Run from the repository root with:

    python tests/test_core.py
"""

import csv
import json
import os
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as host_np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.encoder import DominoEncoder
from agents.network_architecture import (
    MAX_HIDDEN_LAYER_COUNT,
    architecture_from_hidden_sizes,
)
from agents.heuristic_agent import StrategicAgent
from agents.neural_agent import NeuralAgent
from agents.nn import (
    DISABLED_DROPOUT_RATE,
    DISABLED_WEIGHT_DECAY,
    GPU_ENABLED,
    GPU_UNAVAILABLE_REASON,
    SupervisedNeuralNetwork,
)
from agents.rl_agent import RLAgent, TrajectoryStep
from agents.rl_nn import PolicyNetwork
from diagnostics.pairwise import (
    CANONICAL_AGENTS,
    _atomic_replace_directory,
    save_csv,
)
from diagnostics.evaluate import diagnostic_plan
from middleware.domino_engine import (
    DominoEngine,
    WIN_REASON_BLOCKED_FEWEST_PIPS,
    WIN_REASON_BLOCKED_FEWEST_TILES,
    WIN_REASON_BLOCKED_LAST_VALID_PLAY,
    WIN_REASON_EMPTY_HAND,
    infer_dead_suits,
)
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
    DEFAULT_GPI,
    EVENT_REWARD_DECAY,
    LEARNER_DRAW_PENALTY,
    LEARNER_PASS_PENALTY,
    OPPONENT_DRAW_REWARD,
    OPPONENT_PASS_REWARD,
    EventStats,
    TrainingSample,
    _event_reward_for_action,
    _finish_episode_with_rewards,
    _legacy_policy_update,
    _reward_signal_summary,
    parse_args as parse_self_play_args,
)
from training.pipeline import (
    _rl_config as _canonical_rl_config,
    parse_args as parse_canonical_pipeline_args,
)
from training.rl_cli import _training_kwargs_from_args
from training.training_loop import (
    DEFAULT_DROPOUT_RATE,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_SUPERVISED_LR_DECAY_FACTOR,
    DEFAULT_TRAINING_PLATEAU_MIN_EPOCHS,
    DEFAULT_TRAINING_PLATEAU_MIN_RELATIVE_IMPROVEMENT,
    DEFAULT_TRAINING_PLATEAU_PATIENCE,
    DEFAULT_TRAINING_PLATEAU_WINDOW,
    DEFAULT_WEIGHT_DECAY,
    hidden_sizes_from_args,
    parse_args as parse_supervised_args,
)
from train_script.run_pipeline import _build_config, parse_args as parse_pipeline_args
from utils.myrandom import SeedPlan
from utils.runtime_status import pipeline_compute_report

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
    xp = host_np

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


class StrategicOpponentModelThatMustNotRun:
    """Fail if a forced heuristic action performs exact-model work."""

    def update(self, state):
        raise AssertionError("The exact opponent model must not run for forced actions.")


def _to_numpy(value):
    return value.get() if hasattr(value, "get") else value


def _masked_action_probability(network, x_batch, legal_mask, action_index):
    network.forward(x_batch)
    logits = network.cache["Z3"]
    masked_logits = xp.where(legal_mask > 0, logits, -xp.inf)
    shifted = masked_logits - xp.max(masked_logits, axis=0, keepdims=True)
    masked_policy = xp.exp(shifted) / xp.sum(xp.exp(shifted), axis=0, keepdims=True)
    return float(_to_numpy(masked_policy[action_index, 0]))


def _small_policy_network(
    input_size=4,
    hidden1_size=5,
    hidden2_size=3,
    output_size=56,
    learning_rate=0.1,
    use_value_head=False,
    hidden_sizes=None,
):
    """Build a deterministic tiny policy network without invoking backend RNG."""
    if hidden_sizes is None:
        hidden_sizes = (hidden1_size, hidden2_size)
    hidden_sizes = tuple(int(size) for size in hidden_sizes)
    network = PolicyNetwork.__new__(PolicyNetwork)
    network.xp = xp
    network.device = "gpu" if GPU_ENABLED else "cpu"
    network.lr = learning_rate
    network.weight_decay = 0.0
    network.dropout_rate = 0.0
    network.hidden_sizes = hidden_sizes
    dimensions = (input_size, *hidden_sizes, output_size)
    for index in range(1, len(dimensions)):
        setattr(
            network,
            f"W{index}",
            xp.zeros((dimensions[index], dimensions[index - 1])),
        )
        setattr(network, f"b{index}", xp.zeros((dimensions[index], 1)))
    network.use_value_head = use_value_head
    if use_value_head:
        network.Wv = xp.zeros((1, hidden_sizes[-1]))
        network.bv = xp.zeros((1, 1))
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


def test_neural_agent_skips_network_for_single_option_tile_play():
    """Forced tile plays must bypass inference for the supervised policy."""
    only_action = ((6, 6), 0)

    agent = NeuralAgent(NetworkThatMustNotRun())
    assert agent.choose_move({}, [only_action]) == only_action


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


def _blocked_engine(first_hand, second_hand, *, last_player=None):
    engine = DominoEngine(player_count=2)
    engine.ends = [6, 6]
    engine.hands = [list(first_hand), list(second_hand)]
    engine.initial_hands = [list(first_hand), list(second_hand)]
    engine.drawn_tiles_by_player = [[], []]
    engine.stock = []
    engine.board_history = []
    engine.current_player = 0
    engine.required_opening_tile = None
    engine.consecutive_passes = 0
    engine.drew_this_turn = {0: False, 1: False}
    engine.turn = 10
    engine.game_over = False
    engine.winner = None
    engine.win_reason = None
    engine.last_valid_tile_player = last_player
    engine._last_valid_tile_turn_by_player = (
        [4, 8] if last_player == 1 else [8, 4]
    )
    return engine


def test_engine_empty_hand_win_has_an_explicit_reason():
    engine = DominoEngine(player_count=2)
    player = engine.current_player
    engine.ends = [1, 2]
    engine.hands[player] = [(1, 3)]
    engine.hands[1 - player] = [(4, 4)]
    engine.stock = []
    engine.required_opening_tile = None

    _state, done, info = engine.step(((1, 3), 0))

    assert done
    assert engine.winner == player
    assert engine.win_reason == WIN_REASON_EMPTY_HAND
    assert info["win_reason"] == WIN_REASON_EMPTY_HAND


def test_blocked_game_uses_fewest_pips_before_other_tiebreakers():
    engine = _blocked_engine([(0, 1)], [(2, 2)], last_player=1)

    engine.step(None)
    _state, done, _info = engine.step(None)

    assert done
    assert engine.winner == 0
    assert engine.win_reason == WIN_REASON_BLOCKED_FEWEST_PIPS


def test_blocked_game_uses_fewest_tiles_when_pips_are_tied():
    engine = _blocked_engine([(0, 2)], [(0, 0), (1, 1)], last_player=1)

    engine.step(None)
    _state, done, _info = engine.step(None)

    assert done
    assert engine.winner == 0
    assert engine.win_reason == WIN_REASON_BLOCKED_FEWEST_TILES


def test_blocked_game_uses_last_valid_play_as_final_tiebreaker():
    engine = _blocked_engine([(0, 3)], [(1, 2)], last_player=1)

    engine.step(None)
    _state, done, info = engine.step(None)

    assert done
    assert engine.winner == 1
    assert engine.win_reason == WIN_REASON_BLOCKED_LAST_VALID_PLAY
    assert info["winner"] in (0, 1)


def test_engine_final_stock_draw_unplayable_tile_requires_pass_before_blocked_game():
    """Drawing the last, unplayable stock tile must not end the game immediately.

    consecutive_passes is already at the blocked-game threshold before the
    draw. The draw empties the stock, but the current player must still be
    offered the forced PASS before the blocked-game outcome is decided (see
    domino_final_stock_draw_bug_report.txt).
    """
    engine = DominoEngine(player_count=2)
    engine.ends = [1, 1]
    engine.hands = [[(4, 4), (2, 5)], [(0, 0)]]
    engine.stock = [(3, 5)]
    engine.current_player = 0
    engine.consecutive_passes = 2
    engine.drew_this_turn = {0: False, 1: False}

    _state, game_over, _info = engine.step(("DRAW", None))
    assert game_over is False
    assert engine.game_over is False
    assert engine.current_player == 0
    assert engine.valid_actions(0) == [None]

    _state, game_over, _info = engine.step(None)
    assert game_over is True
    assert engine.game_over is True
    assert engine.winner is not None


def test_engine_final_stock_draw_playable_tile_can_be_played_immediately():
    """Drawing a playable final stock tile must let the same player play it.

    consecutive_passes is already at the blocked-game threshold before the
    draw, so the pre-fix engine would end the game the instant the stock
    emptied instead of offering the drawn tile as a legal play.
    """
    engine = DominoEngine(player_count=2)
    engine.ends = [6, 5]
    engine.hands = [[(4, 4), (1, 1)], [(0, 0)]]
    engine.stock = [(5, 6)]
    engine.current_player = 0
    engine.consecutive_passes = 2
    engine.drew_this_turn = {0: False, 1: False}

    _state, game_over, _info = engine.step(("DRAW", None))
    assert game_over is False
    assert engine.game_over is False

    legal_actions = engine.valid_actions(0)
    assert ((5, 6), 0) in legal_actions or ((5, 6), 1) in legal_actions

    engine.step(((5, 6), 0))
    assert engine.game_over is False
    assert engine.consecutive_passes == 0


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


def test_supervised_training_transfers_only_host_minibatches_to_backend():
    """Keep full supervised arrays on the host and transfer bounded batches."""
    network = SupervisedNeuralNetwork(
        input_size=4,
        hidden1_size=5,
        hidden2_size=3,
        output_size=2,
        learning_rate=0.01,
        random_seed=7,
        device="cpu",
    )
    x_train = host_np.ones((4, 5), dtype=float)
    y_train = host_np.zeros((2, 5), dtype=float)
    y_train[0, :] = 1.0

    transferred_shapes = []
    original_to_backend = network._to_backend

    def track_transfer(array):
        transferred_shapes.append(array.shape)
        return original_to_backend(array)

    network._to_backend = track_transfer
    network.train(
        x_train,
        y_train,
        epochs=1,
        batch_size=2,
        quiet=True,
    )

    assert isinstance(x_train, host_np.ndarray)
    assert isinstance(y_train, host_np.ndarray)
    assert transferred_shapes
    assert max(shape[1] for shape in transferred_shapes) == 2
    assert network.cache["X"].shape == (4, 1)
    assert isinstance(network.cache["X"], host_np.ndarray)


def test_supervised_weight_decay_regularizes_weights_but_not_biases():
    """Apply the configured L2 term only to trainable weight matrices."""
    common_args = {
        "input_size": 4,
        "hidden1_size": 5,
        "hidden2_size": 3,
        "output_size": 2,
        "learning_rate": 0.01,
        "random_seed": 11,
        "device": "cpu",
    }
    plain = SupervisedNeuralNetwork(**common_args)
    regularized = SupervisedNeuralNetwork(**common_args, weight_decay=0.2)
    x_batch = host_np.ones((4, 3), dtype=float)
    y_batch = host_np.zeros((2, 3), dtype=float)
    y_batch[0, :] = 1.0

    initial_weights = {
        name: _to_numpy(getattr(regularized, name)).copy()
        for name in ("W1", "W2", "W3")
    }
    plain.forward(x_batch)
    plain.backward(y_batch)
    regularized.forward(x_batch)
    regularized.backward(y_batch)

    for name in ("W1", "W2", "W3"):
        expected = (
            _to_numpy(getattr(plain, name))
            - common_args["learning_rate"] * 0.2 * initial_weights[name]
        )
        assert host_np.allclose(_to_numpy(getattr(regularized, name)), expected)

    for name in ("b1", "b2", "b3"):
        assert host_np.allclose(
            _to_numpy(getattr(regularized, name)),
            _to_numpy(getattr(plain, name)),
        )


def test_supervised_dropout_applies_only_to_training_forward_passes():
    """Drop and rescale hidden units while updating, never while evaluating."""
    common_args = {
        "input_size": 4,
        "hidden1_size": 64,
        "hidden2_size": 48,
        "output_size": 2,
        "learning_rate": 0.01,
        "seed_plan": SeedPlan(5),
        "device": "cpu",
    }
    network = SupervisedNeuralNetwork(**common_args, dropout_rate=0.5)
    x_batch = host_np.ones((4, 6), dtype=float)

    network.forward(x_batch, training=True)
    for activation, mask in (("A1", "D1"), ("A2", "D2")):
        scale = _to_numpy(network.cache[mask])
        kept = scale > 0
        assert kept.any() and not kept.all()
        # Inverted dropout keeps the expected activation magnitude constant.
        assert host_np.allclose(scale[kept], 2.0)
        assert host_np.all(_to_numpy(network.cache[activation])[~kept] == 0.0)

    network.forward(x_batch)
    assert "D1" not in network.cache
    assert "D2" not in network.cache

    disabled = SupervisedNeuralNetwork(**common_args)
    assert disabled.dropout_rate == DISABLED_DROPOUT_RATE
    disabled.forward(x_batch, training=True)
    assert "D1" not in disabled.cache


def test_supervised_dropout_backpropagates_through_the_forward_mask():
    """Zero the gradient of every unit the forward pass dropped."""
    network = SupervisedNeuralNetwork(
        input_size=4,
        hidden1_size=32,
        hidden2_size=16,
        output_size=2,
        learning_rate=0.05,
        seed_plan=SeedPlan(7),
        device="cpu",
        dropout_rate=0.5,
    )
    # One column keeps the mask per hidden unit, so a dropped unit is dropped
    # for the complete update.
    x_batch = host_np.ones((4, 1), dtype=float)
    y_batch = host_np.zeros((2, 1), dtype=float)
    y_batch[0, :] = 1.0

    w2_before = _to_numpy(network.W2).copy()
    network.forward(x_batch, training=True)
    dropped_first_layer = _to_numpy(network.cache["D1"])[:, 0] == 0.0
    assert dropped_first_layer.any()
    network.backward(y_batch)

    w2_after = _to_numpy(network.W2)
    # W2 columns read the first hidden layer, so a dropped unit cannot
    # contribute to its update.
    assert host_np.allclose(
        w2_after[:, dropped_first_layer],
        w2_before[:, dropped_first_layer],
    )
    assert not host_np.allclose(
        w2_after[:, ~dropped_first_layer],
        w2_before[:, ~dropped_first_layer],
    )


def test_rl_weight_decay_shrinks_weight_matrices_but_not_biases():
    """Apply the decoupled RL shrink after the clipped gradient step."""
    decay = 0.25
    learning_rate = 0.1
    plain = _small_policy_network(
        output_size=DominoEncoder.ACTION_SIZE,
        learning_rate=learning_rate,
    )
    regularized = _small_policy_network(
        output_size=DominoEncoder.ACTION_SIZE,
        learning_rate=learning_rate,
    )
    regularized.weight_decay = decay
    for name in ("W1", "W2", "W3"):
        filled = xp.ones_like(getattr(plain, name))
        setattr(plain, name, filled)
        setattr(regularized, name, filled.copy())

    legal_mask = xp.zeros((DominoEncoder.ACTION_SIZE, 1))
    legal_mask[3, 0] = 1.0
    legal_mask[8, 0] = 1.0
    x_batch = xp.ones((4, 1))

    for network in (plain, regularized):
        network.forward(x_batch)
        network.backward_policy_gradient(
            action_indices=[3],
            policy_rewards=xp.ones((1, 1)),
            legal_masks=legal_mask,
            entropy_coef=0.0,
            clip_grad_norm=None,
        )

    shrink = 1.0 - learning_rate * decay
    for name in ("W1", "W2", "W3"):
        assert host_np.allclose(
            _to_numpy(getattr(regularized, name)),
            _to_numpy(getattr(plain, name)) * shrink,
        )
    for name in ("b1", "b2", "b3"):
        assert host_np.allclose(
            _to_numpy(getattr(regularized, name)),
            _to_numpy(getattr(plain, name)),
        )


def test_rl_dropout_is_absent_from_rollout_and_evaluation_forward_passes():
    """Keep opponent rollouts and PPO metrics on the complete network."""
    network = _small_policy_network(output_size=DominoEncoder.ACTION_SIZE)
    network.dropout_rate = 0.5
    x_batch = xp.ones((4, 4))
    legal_mask = xp.zeros((DominoEncoder.ACTION_SIZE, 4))
    legal_mask[3, :] = 1.0
    legal_mask[8, :] = 1.0

    network.forward(x_batch)
    assert "D1" not in network.cache
    network.evaluate_actions(x_batch, legal_mask, [3, 3, 3, 3])
    assert "D1" not in network.cache
    network.evaluate_actions(x_batch, legal_mask, [3, 3, 3, 3], training=True)
    assert "D1" in network.cache and "D2" in network.cache

    # The legacy --no-ppo update differentiates the cache built by
    # _legacy_policy_update, so that pass must be a dropout pass too.
    samples = [
        TrainingSample(
            x=host_np.ones((4, 1)),
            action_index=3,
            legal_mask=host_np.asarray(_to_numpy(legal_mask)[:, :1] > 0),
            policy_reward=1.0,
            raw_reward=1.0,
            local_reward=0.0,
            terminal_reward=1.0,
            old_log_prob=-1.0,
        )
    ]
    network.forward(x_batch)
    _legacy_policy_update(
        network,
        samples,
        entropy_coef=0.0,
        clip_grad_norm=None,
        normalize_advantages=False,
        use_value_head=False,
        value_coef=0.5,
    )
    assert "D1" in network.cache and "D2" in network.cache


def test_regularization_is_disabled_unless_its_flag_is_passed():
    """Keep every entry point unregularized until a flag requests otherwise."""
    parsers = (
        (parse_supervised_args, []),
        (parse_self_play_args, []),
        (parse_pipeline_args, []),
        (parse_canonical_pipeline_args, ["small"]),
    )
    for parse, base in parsers:
        defaults = parse(list(base))
        assert defaults.weight_decay == DISABLED_WEIGHT_DECAY
        assert defaults.dropout == DISABLED_DROPOUT_RATE
        # A bare flag falls back to its default coefficient.
        shortcut = parse(base + ["--weight-decay", "--dropout"])
        assert shortcut.weight_decay == DEFAULT_WEIGHT_DECAY
        assert shortcut.dropout == DEFAULT_DROPOUT_RATE
        # An explicit coefficient wins over that fallback.
        explicit = parse(base + ["--weight-decay", "0.002", "--dropout", "0.35"])
        assert explicit.weight_decay == 0.002
        assert explicit.dropout == 0.35
        # Requesting one regularizer must never enable the other.
        only_decay = parse(base + ["--weight-decay"])
        assert only_decay.dropout == DISABLED_DROPOUT_RATE
        only_dropout = parse(base + ["--dropout"])
        assert only_dropout.weight_decay == DISABLED_WEIGHT_DECAY

    # The same single coefficient reaches the supervised and the RL network.
    rl_kwargs = _training_kwargs_from_args(
        parse_self_play_args(["--weight-decay", "0.002", "--dropout", "0.35"])
    )
    assert rl_kwargs["weight_decay"] == 0.002
    assert rl_kwargs["dropout_rate"] == 0.35
    disabled_kwargs = _training_kwargs_from_args(parse_self_play_args([]))
    assert disabled_kwargs["weight_decay"] == DISABLED_WEIGHT_DECAY
    assert disabled_kwargs["dropout_rate"] == DISABLED_DROPOUT_RATE

    canonical = parse_canonical_pipeline_args(
        ["small", "--weight-decay", "0.002", "--dropout", "0.35"]
    )
    assert _canonical_rl_config(canonical)["weight_decay"] == 0.002
    assert _canonical_rl_config(canonical)["dropout_rate"] == 0.35


def test_networks_are_unregularized_without_explicit_coefficients():
    """Construct both networks with regularization off by default."""
    supervised = SupervisedNeuralNetwork(
        input_size=4,
        hidden1_size=5,
        hidden2_size=3,
        output_size=2,
        device="cpu",
    )
    assert supervised.weight_decay == DISABLED_WEIGHT_DECAY
    assert supervised.dropout_rate == DISABLED_DROPOUT_RATE

    policy = PolicyNetwork(
        input_size=4,
        hidden1_size=5,
        hidden2_size=3,
        output_size=2,
        random_seed=1,
        device="cpu",
    )
    assert policy.weight_decay == DISABLED_WEIGHT_DECAY
    assert policy.dropout_rate == DISABLED_DROPOUT_RATE

    # An enabled network keeps both settings through a pool clone.
    regularized = PolicyNetwork(
        input_size=4,
        hidden1_size=5,
        hidden2_size=3,
        output_size=2,
        random_seed=1,
        device="cpu",
        weight_decay=0.002,
        dropout_rate=0.35,
    )
    clone = regularized.clone()
    assert clone.weight_decay == 0.002
    assert clone.dropout_rate == 0.35


def test_hidden_layer_flags_keep_the_default_and_size_deeper_stacks():
    """Resolve --hidden-layers and every --hiddenN-size on every entry point."""
    parsers = (
        (parse_supervised_args, []),
        (parse_pipeline_args, []),
        (parse_canonical_pipeline_args, ["small"]),
    )
    for parse, base in parsers:
        # The unchanged default architecture stays two layers of 256 and 128.
        defaults = parse(list(base))
        assert defaults.hidden_layers == 2
        assert hidden_sizes_from_args(defaults) == (256, 128)
        assert defaults.hidden3_size is None
        assert defaults.hidden4_size is None

        # An omitted width falls back to its documented default: the historical
        # 256 and 128 for the first two layers and 128 for every deeper one.
        deep = parse(base + ["--hidden-layers", "4"])
        assert hidden_sizes_from_args(deep) == (256, 128, 128, 128)

        # Explicit widths win, and an omitted one still falls back to 128.
        selected = parse(base + [
            "--hidden-layers", "4",
            "--hidden1-size", "512",
            "--hidden2-size", "256",
            "--hidden4-size", "64",
        ])
        assert hidden_sizes_from_args(selected) == (512, 256, 128, 64)

        # Every layer up to the command-line maximum has its own width flag.
        deepest = parse(base + [
            "--hidden-layers", str(MAX_HIDDEN_LAYER_COUNT),
        ] + [
            argument
            for position in range(1, MAX_HIDDEN_LAYER_COUNT + 1)
            for argument in (f"--hidden{position}-size", str(32 * position))
        ])
        assert hidden_sizes_from_args(deepest) == tuple(
            32 * position
            for position in range(1, MAX_HIDDEN_LAYER_COUNT + 1)
        )

        # Depth beyond the available width flags is refused on the CLI.
        try:
            parse(base + ["--hidden-layers", str(MAX_HIDDEN_LAYER_COUNT + 1)])
        except SystemExit:
            pass
        else:
            raise AssertionError(
                f"--hidden-layers must stop at {MAX_HIDDEN_LAYER_COUNT}"
            )

        # A single hidden layer is supported and drops the unused widths.
        shallow = parse(base + ["--hidden-layers", "1", "--hidden1-size", "96"])
        assert hidden_sizes_from_args(shallow) == (96,)
        assert shallow.hidden2_size is None

        # Sizing a layer the architecture does not have is refused rather than
        # silently ignored.
        try:
            parse(base + ["--hidden-layers", "2", "--hidden3-size", "64"])
        except SystemExit:
            pass
        else:
            raise AssertionError("--hidden3-size must require --hidden-layers 3")

    architecture = architecture_from_hidden_sizes(512, 256, 128, 64)
    assert architecture.hidden_layer_count == 4
    assert architecture.as_list() == [168, 512, 256, 128, 64, 56]
    assert architecture.as_dict()["hidden4_size"] == 64
    assert list(architecture.policy_weight_shapes()) == [
        "W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "W5", "b5",
    ]
    # The two-layer metadata representation is unchanged, so existing seed-addressed
    # supervised assets keep matching without a rebuild.
    assert architecture_from_hidden_sizes(256, 128).as_dict() == {
        "input_size": 168,
        "hidden1_size": 256,
        "hidden2_size": 128,
        "output_size": 56,
        "dtype": "float32",
    }


def test_networks_accept_any_depth_beyond_the_command_line_maximum():
    """Build, train, and reload a network deeper than the CLI can express.

    Depth is bounded only by the number of ``--hidden<n>-size`` options, so a
    programmatic caller may use any ``n >= 1``.
    """
    hidden_sizes = tuple(range(12, 0, -1))
    assert len(hidden_sizes) > MAX_HIDDEN_LAYER_COUNT
    network = SupervisedNeuralNetwork(
        input_size=6,
        output_size=4,
        hidden_sizes=hidden_sizes,
        learning_rate=0.05,
        random_seed=19,
        dropout_rate=0.2,
        device="cpu",
    )
    assert network.layer_count == len(hidden_sizes) + 1
    assert network.weight_names[-2:] == (
        f"W{len(hidden_sizes) + 1}",
        f"b{len(hidden_sizes) + 1}",
    )

    x = host_np.asarray(
        host_np.random.RandomState(8).rand(6, 16),
        dtype=host_np.float32,
    )
    y = host_np.zeros((4, 16), dtype=host_np.float32)
    y[0] = 1.0
    losses = network.train(
        x,
        y,
        epochs=5,
        batch_size=8,
        quiet=True,
        validation_interval=100,
    )
    assert len(losses) == 5
    assert all(host_np.isfinite(float(loss)) for loss in losses)

    architecture = architecture_from_hidden_sizes(hidden_sizes)
    assert architecture.hidden_layer_count == len(hidden_sizes)
    assert architecture.as_list() == [168, *hidden_sizes, 56]


def test_deep_networks_train_and_survive_a_checkpoint_round_trip():
    """Train, save, and reload several supported hidden-layer counts."""
    for hidden_sizes in ((7,), (7, 5), (7, 5, 4), (7, 5, 4, 3), (8, 7, 6, 5, 4, 3, 3, 2)):
        layer_count = len(hidden_sizes) + 1
        network = PolicyNetwork(
            input_size=6,
            output_size=5,
            hidden_sizes=hidden_sizes,
            learning_rate=0.05,
            random_seed=21,
            use_value_head=True,
            device="cpu",
        )
        assert network.layer_count == layer_count
        assert network.weight_names == tuple(
            name
            for index in range(1, layer_count + 1)
            for name in (f"W{index}", f"b{index}")
        )
        # The critic reads the last hidden activation whatever the depth.
        assert network.last_hidden_activation_key == f"A{len(hidden_sizes)}"
        assert network.Wv.shape == (1, hidden_sizes[-1])

        x = host_np.asarray(
            host_np.random.RandomState(4).rand(6, 12),
            dtype=host_np.float32,
        )
        legal_masks = host_np.zeros((5, 12), dtype=bool)
        legal_masks[1] = True
        legal_masks[3] = True
        actions = host_np.full(12, 1)
        network.backward_ppo(
            x,
            actions,
            legal_masks,
            host_np.zeros(12, dtype=host_np.float32),
            host_np.ones(12, dtype=host_np.float32),
            returns=host_np.zeros(12, dtype=host_np.float32),
            old_values=host_np.zeros(12, dtype=host_np.float32),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "deep.npz")
            network.save(path)
            reloaded = PolicyNetwork.load(
                path,
                learning_rate=0.05,
                use_value_head=True,
                device="cpu",
            )
        assert reloaded.hidden_sizes == hidden_sizes
        for name in network.weight_names + ("Wv", "bv"):
            assert host_np.array_equal(
                _to_numpy(getattr(reloaded, name)),
                _to_numpy(getattr(network, name)),
            )
        assert host_np.allclose(
            _to_numpy(reloaded.forward(x)),
            _to_numpy(network.forward(x)),
        )


def test_supervised_early_stopping_and_lr_decay_use_independent_counters():
    """Early stopping may occur before the independent LR counter reaches five."""
    network = SupervisedNeuralNetwork(
        input_size=4,
        hidden1_size=5,
        hidden2_size=3,
        output_size=2,
        learning_rate=0.01,
        random_seed=13,
        device="cpu",
    )
    x = host_np.ones((4, 5), dtype=float)
    y = host_np.zeros((2, 5), dtype=float)
    y[0, :] = 1.0
    network._batched_validation_loss = lambda *_args, **_kwargs: 1.0

    history = network.train(
        x,
        y,
        x_val=x,
        y_val=y,
        epochs=50,
        batch_size=2,
        quiet=True,
        early_stopping_patience=2,
        lr_decay_factor=0.5,
    )

    assert len(history) == 21
    assert abs(network.lr - 0.01) < 1e-12


def test_supervised_regularization_cli_defaults_and_shortcuts():
    """Enable plateau decay by default while keeping other controls optional."""
    defaults = parse_supervised_args([])
    assert defaults.weight_decay == DISABLED_WEIGHT_DECAY
    assert defaults.dropout == DISABLED_DROPOUT_RATE
    assert defaults.early_stopping is None
    assert defaults.lr_decay == DEFAULT_SUPERVISED_LR_DECAY_FACTOR
    assert defaults.lr_decay_patience == 5
    assert not defaults.disable_training_plateau
    assert defaults.sl_training_plateau_window == DEFAULT_TRAINING_PLATEAU_WINDOW
    assert (
        defaults.sl_training_plateau_patience
        == DEFAULT_TRAINING_PLATEAU_PATIENCE
    )
    assert (
        defaults.sl_training_plateau_min_epochs
        == DEFAULT_TRAINING_PLATEAU_MIN_EPOCHS
    )
    assert (
        defaults.sl_training_plateau_min_relative_improvement
        == DEFAULT_TRAINING_PLATEAU_MIN_RELATIVE_IMPROVEMENT
    )
    assert defaults.sl_device == "auto"

    enabled = parse_supervised_args([
        "--weight-decay",
        "--dropout",
        "--early-stopping",
        "--lr-decay",
    ])
    assert enabled.weight_decay == DEFAULT_WEIGHT_DECAY
    assert enabled.dropout == DEFAULT_DROPOUT_RATE
    assert enabled.early_stopping == DEFAULT_EARLY_STOPPING_PATIENCE
    assert enabled.lr_decay == DEFAULT_SUPERVISED_LR_DECAY_FACTOR

    custom = parse_supervised_args([
        "--weight-decay",
        "0.0005",
        "--dropout",
        "0.3",
        "--early-stopping",
        "8",
        "--lr-decay",
        "0.8",
    ])
    assert custom.weight_decay == 0.0005
    assert custom.dropout == 0.3
    assert custom.early_stopping == 8
    assert custom.lr_decay == 0.8

    disabled = parse_supervised_args([
        "--no-lr-decay",
        "--sl-no-training-plateau-stop",
        "--device",
        "cpu",
    ])
    assert disabled.lr_decay is None
    assert disabled.disable_training_plateau
    assert disabled.sl_device == "cpu"

    pipeline = parse_pipeline_args([
        "small",
        "--weight-decay",
        "--dropout",
        "0.15",
        "--early-stopping",
        "7",
        "--lr-decay",
        "0.6",
        "--value-head",
    ])
    assert pipeline.scale == "small"
    # One flag per regularizer drives both the supervised and the RL network.
    assert pipeline.weight_decay == DEFAULT_WEIGHT_DECAY
    assert pipeline.dropout == 0.15
    assert pipeline.early_stopping == 7
    assert pipeline.lr_decay == 0.6
    assert pipeline.value_head


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


def test_strategic_agent_skips_exact_model_for_forced_actions():
    agent = StrategicAgent()
    agent.opponent_model = StrategicOpponentModelThatMustNotRun()
    forced_cases = [
        ([((6, 6), 0)], ((6, 6), 0)),
        ([("DRAW", None)], ("DRAW", None)),
        ([None], None),
    ]

    for legal_actions, expected_action in forced_cases:
        assert agent.choose_move({}, legal_actions) == expected_action


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
    assert host_np.isclose(step.old_log_prob, host_np.log(0.5))
    assert step.decision_turn == state["turn"]
    assert step.local_reward == 0.0


def test_rl_evaluation_modes_separate_sampling_from_trajectory_storage():
    state = _base_probability_state()
    legal_actions = [((0, 0), 0), ((0, 1), 0)]
    stochastic = RLAgent(UniformPolicyNetwork(), mode="stochastic_evaluation")
    stochastic.opponent_model = FixedStrategicOpponentModel([0.5] * 7)

    def choose_second(_probabilities, actions):
        action = actions[1]
        return action, stochastic.encoder._action_index(action)

    def trajectory_mask_must_not_run(_actions):
        raise AssertionError("Evaluation must not build a trajectory mask.")

    stochastic.encoder.sample_action = choose_second
    stochastic.encoder.policy_action_mask = trajectory_mask_must_not_run

    assert stochastic.choose_move(state, legal_actions) == legal_actions[1]
    assert stochastic.trajectory == []

    deterministic = RLAgent(UniformPolicyNetwork(), mode="evaluation")
    deterministic.opponent_model = FixedStrategicOpponentModel([0.5] * 7)
    deterministic.encoder.sample_action = lambda *_args: (_ for _ in ()).throw(
        AssertionError("Deterministic evaluation must not sample.")
    )
    deterministic.encoder.decode_output = lambda _probabilities, actions: actions[0]

    assert deterministic.choose_move(
        _base_probability_state(), legal_actions
    ) == legal_actions[0]
    assert deterministic.trajectory == []

    try:
        RLAgent(UniformPolicyNetwork(), mode="unknown")
    except ValueError as exc:
        assert "Unknown RLAgent mode" in str(exc)
    else:
        raise AssertionError("Expected invalid RLAgent modes to be rejected.")


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
            TrajectoryStep(None, 0, None, decision_turn=10),
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
        TrajectoryStep(None, 0, None, decision_turn=10),
        TrajectoryStep(None, 0, None, decision_turn=12),
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
        TrajectoryStep(None, 0, None, decision_turn=1, local_reward=0.10),
        TrajectoryStep(None, 0, None, decision_turn=3, local_reward=-0.05),
    ]

    steps = agent.finish_episode(0.50)

    assert steps[0].terminal_reward == 0.50
    assert steps[1].terminal_reward == 0.50
    assert abs(steps[0].raw_reward - 0.60) < 1e-12
    assert abs(steps[1].raw_reward - 0.45) < 1e-12


def test_choice_count_does_not_weight_terminal_or_local_rewards():
    agent = RLAgent(UniformPolicyNetwork(), mode="training")
    agent.trajectory = [
        TrajectoryStep(None, 0, None, decision_turn=1, local_reward=0.10),
        TrajectoryStep(None, 0, None, decision_turn=1, local_reward=0.10),
    ]

    samples = _finish_episode_with_rewards(agent, 0.50)

    assert [sample.policy_reward for sample in samples] == [0.60, 0.60]
    assert all(not hasattr(sample, "multiplier") for sample in samples)
    assert all(not hasattr(sample, "option_count") for sample in samples)


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


def test_optional_value_head_learns_reward_baseline():
    network = _small_policy_network(use_value_head=True)
    network.b1 = xp.ones_like(network.b1)
    network.b2 = xp.ones_like(network.b2)
    x_batch = xp.ones((4, 1))
    legal_mask = xp.zeros((56, 1))
    legal_mask[3, 0] = 1.0
    legal_mask[8, 0] = 1.0
    returns = xp.ones((1, 1))

    values_before = network.predict_values(x_batch)
    advantages = returns - values_before
    metrics = network.backward_policy_gradient(
        action_indices=[3],
        policy_rewards=advantages,
        legal_masks=legal_mask,
        entropy_coef=0.0,
        clip_grad_norm=None,
        value_returns=returns,
        value_coef=0.5,
    )
    values_after = network.predict_values(x_batch)

    assert float(_to_numpy(values_before[0, 0])) == 0.0
    assert float(_to_numpy(values_after[0, 0])) > 0.0
    assert abs(metrics["value_loss"] - 0.5) < 1e-12
    assert host_np.any(_to_numpy(network.Wv) != 0.0)


def test_legacy_value_head_update_reports_pre_update_predictions():
    network = PolicyNetwork(
        input_size=4,
        hidden1_size=5,
        hidden2_size=3,
        output_size=56,
        random_seed=123,
        use_value_head=True,
        device="cpu",
    )
    network.W1.fill(0.0)
    network.b1.fill(1.0)
    network.W2.fill(0.0)
    network.b2.fill(1.0)
    network.Wv[:] = 0.25
    network.bv[:] = -0.10
    legal_mask = host_np.zeros((56, 1), dtype=host_np.bool_)
    legal_mask[3, 0] = True
    legal_mask[8, 0] = True
    samples = [
        TrainingSample(
            x=host_np.ones((4, 1), dtype=host_np.float32),
            action_index=3,
            legal_mask=legal_mask,
            policy_reward=reward,
            raw_reward=reward,
            local_reward=0.0,
            terminal_reward=reward,
        )
        for reward in (1.0, -1.0)
    ]

    metrics = _legacy_policy_update(
        network,
        samples,
        entropy_coef=0.0,
        clip_grad_norm=None,
        normalize_advantages=False,
        use_value_head=True,
        value_coef=0.5,
        collect_value_predictions=True,
    )

    values = metrics["value_predictions_before_update"]
    assert values["sample_count"] == 2
    assert abs(values["mean"] - 0.65) < 1e-6
    assert values["std"] == 0.0
    assert abs(values["min"] - 0.65) < 1e-6
    assert abs(values["max"] - 0.65) < 1e-6


def test_policy_checkpoint_saves_policy_weights_and_loads_legacy_value_keys():
    network = _small_policy_network(learning_rate=0.01)

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "policy.npz"
        network.save(path)
        with host_np.load(path) as saved:
            assert set(saved.files) == {
                "W1", "b1", "W2", "b2", "W3", "b3",
                "optimizer_step_count",
            }
            assert int(saved["optimizer_step_count"]) == 0

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

        value_network = _small_policy_network(
            learning_rate=0.01,
            use_value_head=True,
        )
        value_network.Wv[:] = 0.25
        value_network.bv[:] = -0.10
        value_path = Path(folder) / "value_policy.npz"
        value_network.save(value_path)
        with host_np.load(value_path) as value_saved:
            assert set(value_saved.files) == {
                "W1", "b1", "W2", "b2", "W3", "b3", "Wv", "bv",
                "optimizer_step_count",
            }

        value_loaded = PolicyNetwork.load(value_path, use_value_head=True)
        policy_only_loaded = PolicyNetwork.load(value_path)

    assert not hasattr(loaded, "Wv")
    assert loaded.W1.shape == network.W1.shape
    assert value_loaded.use_value_head
    assert host_np.allclose(_to_numpy(value_loaded.Wv), 0.25)
    assert host_np.allclose(_to_numpy(value_loaded.bv), -0.10)
    assert not policy_only_loaded.use_value_head
    assert not hasattr(policy_only_loaded, "Wv")


def test_value_head_cli_is_disabled_by_default():
    assert not parse_self_play_args([]).value_head
    assert parse_self_play_args(["--value-head"]).value_head


def test_rl_initialization_cli_defaults_are_context_specific():
    assert not parse_self_play_args([]).fresh_from_sl
    assert parse_self_play_args(["--fresh-from-sl"]).fresh_from_sl
    assert parse_pipeline_args([]).fresh_from_sl
    assert not parse_pipeline_args(["--continue-existing-rl"]).fresh_from_sl


def test_rl_workload_and_pool_defaults_use_games():
    standalone = parse_self_play_args([])
    pipeline = _build_config("default")

    # The standalone and canonical entry points share the fixed GPI default;
    # the canonical default pipeline owns a 500,000-game RL budget.
    assert standalone.iterations is None
    assert standalone.total_training_games is None
    assert standalone.gpi == DEFAULT_GPI
    assert standalone.ppo_enabled
    assert not hasattr(standalone, "evaluation_games")
    assert not hasattr(standalone, "pool_interval")
    assert not hasattr(standalone, "pool_refresh_games")
    assert pipeline.total_rl_games == 500_000
    assert pipeline.rl_iterations * pipeline.rl_games_per_iteration == 500_000


def test_rl_gpi_is_fixed_explicit_and_positive():
    assert parse_self_play_args(["--gpi", "1000"]).gpi == 1000
    for invalid_arguments in (
        ["--gpi", "0"],
        ["--gpi", "40"],
        ["--adaptive-gpi"],
    ):
        try:
            parse_self_play_args(invalid_arguments)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"Expected rejection for {invalid_arguments!r}")

    assert parse_pipeline_args([]).gpi == DEFAULT_GPI
    assert parse_pipeline_args(["--gpi", "1000"]).gpi == 1000


def test_reward_signal_summary_classifies_rewards():
    samples = [
        TrainingSample(None, 0, None, 1.0, 1.0, 0.20, 0.80),
        TrainingSample(None, 0, None, 0.0, 0.0, 0.00, 0.00),
        TrainingSample(None, 0, None, -1.0, -1.0, -0.10, -0.90),
    ]

    summary = _reward_signal_summary(samples)

    assert abs(summary["good_pct"] - (100.0 / 3.0)) < 1e-12
    assert abs(summary["neutral_pct"] - (100.0 / 3.0)) < 1e-12
    assert abs(summary["bad_pct"] - (100.0 / 3.0)) < 1e-12
    assert abs(summary["local_mean"] - (0.10 / 3.0)) < 1e-12


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


def test_pairwise_csv_writes_initial_hands_as_json_arrays():
    games = [
        {
            "game": 1,
            "agent_position": 0,
            "result": "win",
            "turns": 12,
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
    assert "first_stock_draw_turn" not in row
    assert "first_stock_draw_final_state_count" not in row


def test_diagnostics_atomic_directory_replacement_drops_stale_files():
    with tempfile.TemporaryDirectory() as folder:
        base_dir = Path(folder)
        output_dir = base_dir / "pair"
        staging_dir = base_dir / ".pair.tmp"
        output_dir.mkdir()
        staging_dir.mkdir()
        (output_dir / "stale.png").write_text("old", encoding="utf-8")
        (staging_dir / "summary.json").write_text("{}", encoding="utf-8")

        _atomic_replace_directory(staging_dir, output_dir)

        assert not (output_dir / "stale.png").exists()
        assert (output_dir / "summary.json").read_text(encoding="utf-8") == "{}"
        assert not staging_dir.exists()


def test_diagnostic_plan_selects_canonical_random_matchups():
    agents, matchups = diagnostic_plan()
    expected_matchups = tuple((agent, "random") for agent in CANONICAL_AGENTS)
    assert CANONICAL_AGENTS == ("rl", "neural", "heuristic", "random")
    assert agents == CANONICAL_AGENTS
    assert matchups == expected_matchups


def test_pipeline_scales_set_explicit_diagnostic_game_counts():
    assert _build_config("small").dataset_games == 10_000
    assert _build_config("default").dataset_games == 50_000
    assert all(
        _build_config(level).dataset_games == 100_000
        for level in ("big", "huge", "forever")
    )
    assert all(
        _build_config(level).supervised_epochs == 5_000
        for level in ("small", "default", "big", "huge", "forever")
    )
    assert _build_config("small").diagnostic_games == 10000
    assert _build_config("default").diagnostic_games == 10000
    assert _build_config("big").diagnostic_games == 1_000_000
    assert _build_config("huge").diagnostic_games == 1_000_000
    assert _build_config("forever").diagnostic_games == 0


def test_pipeline_compute_report_names_backends_and_memory():
    report = pipeline_compute_report("auto")

    assert report.startswith("Pipeline compute resources: ")
    if GPU_ENABLED:
        assert (
            "supervised=GPU" in report
            or "supervised=CPU (automatic fallback" in report
        )
        assert GPU_UNAVAILABLE_REASON is None
    else:
        assert "supervised=CPU" in report
        assert GPU_UNAVAILABLE_REASON
    assert "RL parent=" in report
    assert "dataset/RL rollout/diagnostic workers=CPU-only" in report
    assert "system RAM" in report
    assert "GPU VRAM" in report


def main():
    tests = [
        ("encoder action space", test_encoder_action_space_excludes_forced_actions),
        ("encoder JSON tile actions", test_encoder_accepts_list_tiles_from_json),
        (
            "neural forced tile skips network",
            test_neural_agent_skips_network_for_single_option_tile_play,
        ),
        ("opening double rule", test_engine_requires_highest_opening_double_when_present),
        ("unique game ids", test_engine_game_ids_are_unique_across_instances),
        ("empty-hand winner reason", test_engine_empty_hand_win_has_an_explicit_reason),
        (
            "blocked fewer-pips winner",
            test_blocked_game_uses_fewest_pips_before_other_tiebreakers,
        ),
        (
            "blocked fewer-tiles winner",
            test_blocked_game_uses_fewest_tiles_when_pips_are_tied,
        ),
        (
            "blocked last-play winner",
            test_blocked_game_uses_last_valid_play_as_final_tiebreaker,
        ),
        (
            "final stock draw unplayable tile requires pass",
            test_engine_final_stock_draw_unplayable_tile_requires_pass_before_blocked_game,
        ),
        (
            "final stock draw playable tile can be played",
            test_engine_final_stock_draw_playable_tile_can_be_played_immediately,
        ),
        ("dead suit inference", test_infer_dead_suits_from_draw_and_pass_history),
        ("training history shape", test_game_manager_training_history_uses_compact_engine_state),
        (
            "supervised host minibatch transfers",
            test_supervised_training_transfers_only_host_minibatches_to_backend,
        ),
        (
            "supervised weight decay",
            test_supervised_weight_decay_regularizes_weights_but_not_biases,
        ),
        (
            "supervised dropout training-only",
            test_supervised_dropout_applies_only_to_training_forward_passes,
        ),
        (
            "supervised dropout backpropagation",
            test_supervised_dropout_backpropagates_through_the_forward_mask,
        ),
        (
            "RL decoupled weight decay",
            test_rl_weight_decay_shrinks_weight_matrices_but_not_biases,
        ),
        (
            "RL dropout excluded from evaluation",
            test_rl_dropout_is_absent_from_rollout_and_evaluation_forward_passes,
        ),
        (
            "supervised early stopping and LR decay",
            test_supervised_early_stopping_and_lr_decay_use_independent_counters,
        ),
        (
            "supervised optional CLI controls",
            test_supervised_regularization_cli_defaults_and_shortcuts,
        ),
        (
            "regularization opt-in defaults",
            test_regularization_is_disabled_unless_its_flag_is_passed,
        ),
        (
            "unregularized network defaults",
            test_networks_are_unregularized_without_explicit_coefficients,
        ),
        (
            "hidden-layer depth and width flags",
            test_hidden_layer_flags_keep_the_default_and_size_deeper_stacks,
        ),
        (
            "deep network checkpoint round trip",
            test_deep_networks_train_and_survive_a_checkpoint_round_trip,
        ),
        (
            "unbounded network depth",
            test_networks_accept_any_depth_beyond_the_command_line_maximum,
        ),
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
        (
            "strategic forced actions skip exact model",
            test_strategic_agent_skips_exact_model_for_forced_actions,
        ),
        ("RL forced actions skip network", test_rl_agent_skips_network_for_forced_actions),
        ("RL trajectory legal mask", test_rl_agent_saves_legal_mask_for_real_decision),
        (
            "RL stochastic and deterministic evaluation",
            test_rl_evaluation_modes_separate_sampling_from_trajectory_storage,
        ),
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
        ("uniform choice reward", test_choice_count_does_not_weight_terminal_or_local_rewards),
        (
            "positive reward gradient",
            test_positive_reward_increases_chosen_masked_probability,
        ),
        (
            "negative reward gradient",
            test_negative_reward_decreases_chosen_masked_probability,
        ),
        ("optional value baseline", test_optional_value_head_learns_reward_baseline),
        (
            "value head prediction logging",
            test_legacy_value_head_update_reports_pre_update_predictions,
        ),
        ("policy checkpoint keys", test_policy_checkpoint_saves_policy_weights_and_loads_legacy_value_keys),
        ("value head CLI", test_value_head_cli_is_disabled_by_default),
        (
            "RL initialization CLI defaults",
            test_rl_initialization_cli_defaults_are_context_specific,
        ),
        ("RL workload defaults", test_rl_workload_and_pool_defaults_use_games),
        ("reward signal summary", test_reward_signal_summary_classifies_rewards),
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
        ("pairwise CSV initial hands", test_pairwise_csv_writes_initial_hands_as_json_arrays),
        (
            "atomic diagnostic replacement",
            test_diagnostics_atomic_directory_replacement_drops_stale_files,
        ),
        (
            "diagnostic plan matchups",
            test_diagnostic_plan_selects_canonical_random_matchups,
        ),
        (
            "pipeline diagnostic game counts",
            test_pipeline_scales_set_explicit_diagnostic_game_counts,
        ),
        (
            "pipeline compute resource report",
            test_pipeline_compute_report_names_backends_and_memory,
        ),
    ]

    for name, fn in tests:
        _run(name, fn)

    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
