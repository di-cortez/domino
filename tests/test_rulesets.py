"""Focused contracts for named compact domino rulesets."""

from itertools import combinations
import json
import random

import numpy as np
import pytest

from agents.encoder import DominoEncoder
from agents.heuristic_agent import StrategicAgent
from agents.network_architecture import (
    architecture_for_ruleset,
    default_hidden_sizes,
)
from agents.neural_agent import NeuralAgent
from agents.rl_agent import RLAgent
from middleware.domino_engine import DominoEngine
from middleware.opponent_model import HybridExactOpponentModel
from middleware.rulesets import (
    DEFAULT_RULESET_NAME,
    RULESETS,
    RULESET_NAMES,
    resolve_ruleset,
    validate_state_ruleset,
)
from training.canonical_assets import canonical_asset_paths
from training.canonical_run import (
    canonical_run_dir,
    configuration_sha256,
    load_run_config,
)
from training.pipeline import _forever_active_pointer
from diagnostics.parallel_runner import ParallelSafetyConfig
from training.rl.config import (
    RLExecutionOptions,
    RLResourceOptions,
    RLTrainingOptions,
)
from training.rl.training_loop import train as train_rl
from utils.ruleset_paths import (
    default_dataset_path,
    default_encoded_dataset_path,
    default_rl_weights_path,
    default_sl_weights_path,
)


EXPECTED_RULESETS = {
    "double-six": (6, 7, 28, 14),
    "double-five": (5, 6, 21, 9),
    "double-four": (4, 5, 15, 5),
    "double-three": (3, 4, 10, 2),
}


def test_ruleset_registry_is_closed_and_has_expected_geometry():
    assert DEFAULT_RULESET_NAME == "double-six"
    assert RULESET_NAMES == (
        "double-six",
        "double-five",
        "double-four",
        "double-three",
    )
    assert tuple(RULESETS) == RULESET_NAMES
    for name, (max_pip, hand_size, tile_count, stock_size) in (
        EXPECTED_RULESETS.items()
    ):
        ruleset = resolve_ruleset(name)
        assert ruleset.name == name
        assert ruleset.max_pip == max_pip
        assert ruleset.pip_count == max_pip + 1
        assert ruleset.hand_size == hand_size
        assert ruleset.tile_count == tile_count
        assert ruleset.initial_stock_size(2) == stock_size
        assert len(ruleset.all_tiles) == len(set(ruleset.all_tiles)) == tile_count
        assert ruleset.all_tiles[0] == (0, 0)
        assert ruleset.all_tiles[-1] == (max_pip, max_pip)
        assert resolve_ruleset(ruleset) is ruleset


def test_ruleset_registry_rejects_unknown_names_and_unregistered_objects():
    with pytest.raises(ValueError, match="Unknown ruleset"):
        resolve_ruleset("double-seven")
    with pytest.raises(ValueError, match="Unknown ruleset"):
        resolve_ruleset(6)


@pytest.mark.parametrize("ruleset_name", RULESET_NAMES)
def test_engine_deal_and_state_follow_ruleset(ruleset_name):
    ruleset = resolve_ruleset(ruleset_name)
    first = DominoEngine(rng=random.Random(12345), ruleset=ruleset_name)
    second = DominoEngine(rng=random.Random(12345), ruleset=ruleset_name)

    assert first.hands == second.hands
    assert first.stock == second.stock
    assert all(len(hand) == ruleset.hand_size for hand in first.hands)
    assert len(first.stock) == ruleset.initial_stock_size(2)
    dealt = [tile for hand in first.hands for tile in hand] + first.stock
    assert len(dealt) == len(set(dealt)) == ruleset.tile_count
    assert set(dealt) == set(ruleset.all_tiles)
    assert max(max(tile) for tile in dealt) == ruleset.max_pip

    state = first._get_state()
    assert state["ruleset_name"] == ruleset_name
    assert state["initial_hand_size"] == ruleset.hand_size
    assert first.to_dict()["ruleset_name"] == ruleset_name
    assert first.reset()["ruleset_name"] == ruleset_name


def test_double_six_fixed_seed_deal_matches_pre_ruleset_engine():
    engine = DominoEngine(rng=random.Random(12345))
    assert engine.hands == [
        [(0, 1), (2, 3), (1, 6), (2, 5), (5, 5), (4, 4), (1, 1)],
        [(3, 5), (2, 6), (2, 4), (4, 6), (0, 2), (1, 4), (3, 6)],
    ]
    assert engine.stock == [
        (0, 4), (3, 4), (0, 3), (5, 6), (0, 5), (6, 6), (3, 3),
        (1, 2), (0, 6), (1, 5), (1, 3), (0, 0), (4, 5), (2, 2),
    ]
    assert engine.current_player == 0
    assert engine.required_opening_tile == (5, 5)


def test_engine_rejects_deals_larger_than_ruleset_deck():
    with pytest.raises(ValueError, match="Cannot deal"):
        DominoEngine(player_count=3, ruleset="double-three")


def test_state_ruleset_validation_only_defaults_legacy_state_to_double_six():
    validate_state_ruleset({}, "double-six")
    validate_state_ruleset({"ruleset_name": "double-four"}, "double-four")
    with pytest.raises(ValueError, match="has no ruleset_name"):
        validate_state_ruleset({}, "double-four")
    with pytest.raises(ValueError, match="does not match"):
        validate_state_ruleset(
            {"ruleset_name": "double-three"},
            "double-six",
        )


@pytest.mark.parametrize(
    ("name", "vector_size", "action_size", "hidden_sizes", "parameter_count"),
    [
        ("double-six", 168, 56, (256, 128), 83_384),
        ("double-five", 130, 42, (192, 96), 47_754),
        ("double-four", 97, 30, (128, 64), 22_750),
        ("double-three", 69, 20, (96, 48), 12_356),
    ],
)
def test_compact_encoder_and_default_architecture(
    name,
    vector_size,
    action_size,
    hidden_sizes,
    parameter_count,
):
    encoder = DominoEncoder(name)
    architecture = architecture_for_ruleset(name)

    assert encoder.vector_size == vector_size
    assert encoder.action_size == action_size
    assert architecture.layer_dimensions == (
        vector_size,
        *hidden_sizes,
        action_size,
    )
    assert sum(
        shape[0] * shape[1]
        for shape in architecture.policy_weight_shapes().values()
    ) == parameter_count


@pytest.mark.parametrize(
    ("name", "expected_offsets"),
    [
        ("double-six", (0, 28, 56, 84, 112, 140, 147, 154, 156, 157, 159, 161, 168)),
        ("double-five", (0, 21, 42, 63, 84, 105, 111, 117, 119, 120, 122, 124, 130)),
        ("double-four", (0, 15, 30, 45, 60, 75, 80, 85, 87, 88, 90, 92, 97)),
        ("double-three", (0, 10, 20, 30, 40, 50, 54, 58, 60, 61, 63, 65, 69)),
    ],
)
def test_encoder_offsets_are_exact(name, expected_offsets):
    layout = DominoEncoder(name).layout
    assert (
        layout.hand,
        layout.played,
        layout.played_turn,
        layout.played_by_me,
        layout.played_by_opponent,
        layout.left_end,
        layout.right_end,
        layout.hand_size,
        layout.stock_size,
        layout.draw_count,
        layout.pass_count,
        layout.opponent_suit_probability,
        layout.vector_size,
    ) == expected_offsets


@pytest.mark.parametrize("ruleset_name", RULESET_NAMES)
def test_encoder_normalization_and_end_widths_follow_ruleset(ruleset_name):
    ruleset = resolve_ruleset(ruleset_name)
    encoder = DominoEncoder(ruleset)
    state = {
        "ruleset_name": ruleset.name,
        "initial_hand_size": ruleset.hand_size,
        "current_player": 0,
        "current_player_hand": [[0, 0]],
        "hand_sizes": [ruleset.hand_size, ruleset.hand_size - 1],
        "stock_size": ruleset.initial_stock_size(2),
        "ends": [0, ruleset.max_pip],
        "board_history": [],
        "opponent_suit_probabilities": [0.25] * ruleset.pip_count,
    }
    vector = encoder.encode_state(state)[:, 0]

    assert vector.shape == (encoder.vector_size,)
    assert vector[encoder.HAND_OFFSET] == 1.0
    assert vector[encoder.HAND_SIZE_OFFSET] == 1.0
    assert vector[encoder.HAND_SIZE_OFFSET + 1] == pytest.approx(
        (ruleset.hand_size - 1) / ruleset.hand_size
    )
    assert vector[encoder.STOCK_SIZE_OFFSET] == 1.0
    assert vector[encoder.LEFT_END_OFFSET:encoder.RIGHT_END_OFFSET].tolist() == (
        [1.0] + [0.0] * (ruleset.pip_count - 1)
    )
    assert vector[
        encoder.RIGHT_END_OFFSET:encoder.HAND_SIZE_OFFSET
    ].tolist() == ([0.0] * (ruleset.pip_count - 1) + [1.0])
    assert vector[encoder.OPPONENT_SUIT_PROBABILITY_OFFSET:].tolist() == (
        [0.25] * ruleset.pip_count
    )


def test_double_six_action_order_and_legacy_state_remain_exact():
    encoder = DominoEncoder()
    tiles = list(resolve_ruleset("double-six").all_tiles)
    assert encoder.all_actions == (
        [(tile, 0) for tile in tiles] + [(tile, 1) for tile in tiles]
    )
    legacy_state = {
        "current_player": 0,
        "current_player_hand": [[0, 0]],
        "hand_sizes": [7, 6],
        "stock_size": 14,
        "ends": [0, 6],
        "board_history": [],
        "opponent_suit_probabilities": [0.5] * 7,
    }
    vector = encoder.encode_state(legacy_state)[:, 0]
    expected = np.zeros(168, dtype=np.float32)
    expected[0] = 1.0
    expected[140] = 1.0
    expected[153] = 1.0
    expected[154] = 1.0
    expected[155] = 6 / 7
    expected[156] = 1.0
    expected[161:168] = 0.5
    np.testing.assert_array_equal(vector, expected)


def test_compact_encoder_rejects_invalid_tiles_and_probability_width():
    encoder = DominoEncoder("double-three")
    state = {
        "ruleset_name": "double-three",
        "initial_hand_size": 4,
        "current_player": 0,
        "current_player_hand": [[4, 4]],
        "hand_sizes": [4, 4],
        "stock_size": 2,
        "ends": [],
        "board_history": [],
        "opponent_suit_probabilities": [0.0] * 4,
    }
    with pytest.raises(KeyError):
        encoder.encode_state(state)
    state["current_player_hand"] = [[0, 0]]
    state["opponent_suit_probabilities"] = [0.0] * 7
    with pytest.raises(ValueError, match="expected 4"):
        encoder.encode_state(state)


@pytest.mark.parametrize(
    ("name", "expected_upper_bound", "probability_count"),
    [
        ("double-six", 116_280, 7),
        ("double-five", 5_005, 6),
        ("double-four", 252, 5),
        ("double-three", 15, 4),
    ],
)
def test_opponent_model_uses_ruleset_local_domain(
    name,
    expected_upper_bound,
    probability_count,
):
    engine = DominoEngine(rng=random.Random(99), ruleset=name)
    state = engine._get_state()
    model = HybridExactOpponentModel(ruleset=name, record_traces=False)

    probabilities = model.update(state)

    assert len(probabilities) == probability_count
    assert model._belief.raw_hand_upper_bound == expected_upper_bound
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)


def test_double_three_initial_probabilities_match_brute_force_hands():
    ruleset = resolve_ruleset("double-three")
    engine = DominoEngine(rng=random.Random(913), ruleset=ruleset)
    state = engine._get_state()
    model = HybridExactOpponentModel(ruleset=ruleset, record_traces=False)

    actual = model.update(state)
    own_tiles = {tuple(tile) for tile in state["current_player_initial_hand"]}
    unknown = [tile for tile in ruleset.all_tiles if tile not in own_tiles]
    hands = list(combinations(unknown, ruleset.hand_size))
    expected = [
        sum(any(pip in tile for tile in hand) for hand in hands) / len(hands)
        for pip in range(ruleset.pip_count)
    ]
    assert actual == pytest.approx(expected)


def test_opponent_models_for_different_rulesets_are_isolated():
    small_engine = DominoEngine(rng=random.Random(5), ruleset="double-three")
    large_engine = DominoEngine(rng=random.Random(5), ruleset="double-six")
    small = HybridExactOpponentModel(
        ruleset="double-three",
        record_traces=False,
    )
    large = HybridExactOpponentModel(ruleset="double-six", record_traces=False)

    assert len(small.update(small_engine._get_state())) == 4
    assert len(large.update(large_engine._get_state())) == 7
    assert len(small.update(small_engine._get_state())) == 4
    assert small.domain.all_mask.bit_count() == 10
    assert large.domain.all_mask.bit_count() == 28


@pytest.mark.parametrize("ruleset_name", RULESET_NAMES)
def test_strategic_agents_finish_a_ruleset_game(ruleset_name):
    engine = DominoEngine(rng=random.Random(81), ruleset=ruleset_name)
    agents = [
        StrategicAgent(ruleset=ruleset_name),
        StrategicAgent(ruleset=ruleset_name),
    ]
    for _step in range(200):
        if engine.game_over:
            break
        state = engine._get_state()
        legal_actions = engine.valid_actions()
        move = agents[engine.current_player].choose_move(state, legal_actions)
        engine.step(move, legal_actions=legal_actions)
    assert engine.game_over
    assert engine.winner in (0, 1)


class _ShapedPolicy:
    """Minimum network surface needed by RLAgent's early shape validation."""

    def __init__(self, input_size, output_size):
        self.W1 = np.zeros((2, input_size), dtype=np.float32)
        self.W3 = np.zeros((output_size, 2), dtype=np.float32)
        self.layer_count = 3


@pytest.mark.parametrize("ruleset_name", RULESET_NAMES)
def test_rl_agent_accepts_only_matching_ruleset_shape(ruleset_name):
    encoder = DominoEncoder(ruleset_name)
    RLAgent(
        _ShapedPolicy(encoder.vector_size, encoder.action_size),
        ruleset=ruleset_name,
    )
    wrong_shape = (
        (69, 20)
        if ruleset_name == "double-six"
        else (DominoEncoder.VECTOR_SIZE, DominoEncoder.ACTION_SIZE)
    )
    with pytest.raises(ValueError, match="does not match ruleset"):
        RLAgent(
            _ShapedPolicy(*wrong_shape),
            ruleset=ruleset_name,
        )


def test_neural_agent_load_rejects_cross_ruleset_checkpoint(tmp_path):
    architecture = architecture_for_ruleset("double-three")
    weights = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in architecture.policy_weight_shapes().items()
    }
    checkpoint = tmp_path / "double-three.npz"
    np.savez(checkpoint, **weights)

    agent = NeuralAgent.load(
        checkpoint,
        device="cpu",
        ruleset="double-three",
    )
    assert agent.ruleset.name == "double-three"
    assert agent.encoder.vector_size == 69
    with pytest.raises(ValueError, match="double-six produces 168"):
        NeuralAgent.load(checkpoint, device="cpu", ruleset="double-six")


def test_default_and_compact_artifact_names_do_not_collide(tmp_path):
    default_assets = canonical_asset_paths(tmp_path, 42)
    compact_assets = canonical_asset_paths(tmp_path, 42, "double-four")
    assert default_assets.dataset.name == (
        "supervised_dataset_standard_seed42.jsonl"
    )
    assert default_assets.weights.name == "domino_sl_standard_seed42.npz"
    assert compact_assets.dataset.name == (
        "supervised_dataset_double-four_standard_seed42.jsonl"
    )
    assert compact_assets.weights.name == (
        "domino_sl_double-four_standard_seed42.npz"
    )
    assert default_dataset_path().as_posix() == (
        "dataset/supervised_dataset.jsonl"
    )
    assert default_encoded_dataset_path().as_posix() == (
        "dataset/supervised_dataset_encoded.npz"
    )
    assert default_sl_weights_path().as_posix() == (
        "models/domino_sl_weights.npz"
    )
    assert default_rl_weights_path().as_posix() == (
        "models/domino_rl_weights.npz"
    )
    assert default_dataset_path("double-four").name == (
        "supervised_dataset_double-four.jsonl"
    )
    assert default_sl_weights_path("double-four").name == (
        "domino_sl_double-four_weights.npz"
    )
    assert default_rl_weights_path("double-four").name == (
        "domino_rl_double-four_weights.npz"
    )


def test_compact_run_directories_and_forever_pointers_are_namespaced(tmp_path):
    assert canonical_run_dir(tmp_path, "forever", 42).name == (
        "domino_rl_forever_seed42"
    )
    assert canonical_run_dir(
        tmp_path,
        "forever",
        42,
        ruleset="double-four",
    ).name == "domino_rl_double-four_forever_seed42"
    assert _forever_active_pointer(tmp_path).name == "active_forever_run.json"
    assert _forever_active_pointer(tmp_path, "double-four").name == (
        "active_forever_run_double-four.json"
    )


def test_ruleset_participates_in_new_config_hash_but_not_legacy_v3_hash():
    common = {
        "pipeline_level": "forever",
        "run_name": None,
        "seed": 42,
        "target_rl_games": None,
        "ruleset_version": 2,
        "encoder_size": 168,
        "action_count": 56,
        "network_architecture": [168, 256, 128, 56],
        "algorithm": "ppo",
        "supervised_weights_sha256": "abc",
        "ppo_config": {},
        "rl_config": {},
        "diagnostic_config": {},
        "locked_arguments": {},
    }
    default_v4 = {
        **common,
        "config_hash_version": 4,
        "ruleset_name": "double-six",
    }
    compact_v4 = {
        **default_v4,
        "ruleset_name": "double-four",
    }
    assert configuration_sha256(default_v4) != configuration_sha256(compact_v4)

    legacy_v3 = {**common, "config_hash_version": 3}
    legacy_with_ignored_name = {**legacy_v3, "ruleset_name": "double-four"}
    assert configuration_sha256(legacy_v3) == configuration_sha256(
        legacy_with_ignored_name
    )


def test_legacy_v3_run_config_without_ruleset_loads_as_double_six(tmp_path):
    run_config = {
        "format_version": 4,
        "config_hash_version": 3,
        "pipeline_level": "forever",
        "run_name": None,
        "seed": 42,
        "target_rl_games": None,
        "ruleset_version": 2,
        "encoder_size": 168,
        "action_count": 56,
        "network_architecture": [168, 256, 128, 56],
        "algorithm": "ppo",
        "supervised_weights_sha256": "abc",
        "ppo_config": {},
        "rl_config": {},
        "diagnostic_config": {},
        "locked_arguments": {},
    }
    run_config["configuration_sha256"] = configuration_sha256(run_config)
    (tmp_path / "run_config.json").write_text(
        json.dumps(run_config),
        encoding="utf-8",
    )
    loaded = load_run_config(tmp_path)
    assert loaded["ruleset_name"] == "double-six"
    assert loaded["configuration_sha256"] == run_config["configuration_sha256"]


def test_double_three_numbered_checkpoint_resumes_exactly(tmp_path):
    architecture = architecture_for_ruleset("double-three")
    supervised = tmp_path / "sl.npz"
    np.savez(
        supervised,
        **{
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in architecture.policy_weight_shapes().items()
        },
    )
    training = RLTrainingOptions(
        ruleset_name="double-three",
        total_training_games=20,
        gpi=10,
        seed=2468,
        ppo_max_epochs=2,
        # Pinned: the default `lookup-table` baseline has no packaged artifact
        # for double-three, and this test is about the ruleset, not the
        # baseline.
        baseline=("batch-mean",),
    )
    resources = RLResourceOptions(
        sl_weights_path=supervised,
        rl_weights_path=tmp_path / "rl.npz",
        device="cpu",
        workers=1,
        safety_config=ParallelSafetyConfig(
            memory_reserve_mb=0,
            estimated_worker_mb=1,
        ),
    )
    first = train_rl(
        training,
        resources,
        RLExecutionOptions(
            quiet=True,
            numbered_checkpoints=True,
            fresh_from_sl=True,
            checkpoint_interval=1,
            log_interval=1,
            stop_after_training_games=10,
        ),
    )
    second = train_rl(
        training,
        resources,
        RLExecutionOptions(
            quiet=True,
            numbered_checkpoints=True,
            checkpoint_interval=1,
            log_interval=1,
            resume_weights_path=first["rl_weights_path"],
            resume_state_file=first["resume_state_path"],
        ),
    )

    assert first["completed_training_games"] == 10
    assert second["ruleset_name"] == "double-three"
    assert second["start_iteration"] == 1
    assert second["completed_training_games"] == 20
    assert second["rl_iterations_completed"] == 2


def test_double_three_resume_uses_saved_ruleset_instead_of_cli_override(tmp_path):
    architecture = architecture_for_ruleset("double-three")
    supervised = tmp_path / "sl.npz"
    np.savez(
        supervised,
        **{
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in architecture.policy_weight_shapes().items()
        },
    )
    compact_training = RLTrainingOptions(
        ruleset_name="double-three",
        total_training_games=20,
        gpi=10,
        seed=9753,
        ppo_max_epochs=2,
        baseline=("batch-mean",),
    )
    resources = RLResourceOptions(
        sl_weights_path=supervised,
        rl_weights_path=tmp_path / "rl.npz",
        device="cpu",
        workers=1,
        safety_config=ParallelSafetyConfig(
            memory_reserve_mb=0,
            estimated_worker_mb=1,
        ),
    )
    first = train_rl(
        compact_training,
        resources,
        RLExecutionOptions(
            quiet=True,
            numbered_checkpoints=True,
            fresh_from_sl=True,
            checkpoint_interval=1,
            log_interval=1,
            stop_after_training_games=10,
        ),
    )
    resumed = train_rl(
        RLTrainingOptions(
            ruleset_name="double-six",
            total_training_games=20,
            gpi=10,
            seed=9753,
            ppo_max_epochs=2,
            baseline=("batch-mean",),
        ),
        resources,
        RLExecutionOptions(
            quiet=True,
            numbered_checkpoints=True,
            checkpoint_interval=1,
            log_interval=1,
            resume_weights_path=first["rl_weights_path"],
            resume_state_file=first["resume_state_path"],
        ),
    )
    assert resumed["ruleset_name"] == "double-three"
    assert resumed["completed_training_games"] == 20
