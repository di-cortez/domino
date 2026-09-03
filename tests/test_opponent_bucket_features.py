"""Contracts for the opponent-bucket one-hot appended to the policy input.

The block exists so the agent can condition on *who* it is facing: one feature
per logical opponent bucket, set on the bucket that seat's adversary was drawn
from. Every test here protects one property the experiment depends on: the
fixed width, the untouched prefix, the per-seat semantics that give the two
players different buckets in the same game, and the run-identity plumbing that
keeps a run created before the flag existed resumable.
"""

import argparse
import random

import numpy as np
import pytest

from agents.encoder import (
    OPPONENT_BUCKET_FEATURE_INDEX,
    OPPONENT_BUCKET_FEATURE_ORDER,
    OPPONENT_BUCKET_FEATURE_WIDTH,
    DominoEncoder,
    opponent_bucket_feature_index,
)
from agents.network_architecture import architecture_for_ruleset
from agents.rl_agent import RLAgent
from agents.rl_nn import PolicyNetwork
from diagnostics.gameplay import opponent_bucket_for_agent
from middleware.domino_engine import DominoEngine
from middleware.rulesets import RULESET_NAMES, resolve_ruleset
from training.canonical_assets import canonical_asset_paths
from training.canonical_run import run_config_uses_opponent_bucket_features
from training.pipeline import (
    _locked_run_arguments,
    _network_architecture,
    _rl_config,
    _use_opponent_bucket_features,
    parse_args,
)
from training.rl import rollout as rollout_module
from training.rl.cli import (
    add_optional_rl_arguments,
    training_options_from_args,
)
from training.rl.pool import BUCKET_REGISTRY
from training.supervised.training_loop import (
    SUPERVISED_TEACHER_OPPONENT_BUCKET,
)
from ui.ui_agents import (
    UI_OPPONENT_BUCKET,
    _bucket_features_for,
    _checkpoint_input_size,
)
from training.rl.resume import (
    RLTrainingConfiguration,
    _checkpoint_matches_encoder,
)
from training.rl.rollout import (
    LEARNER_OPPONENT_BUCKET,
    DEFAULT_REWARD_SCHEMA,
    collect_steps_for_assignment,
)
from training.supervised.training_loop import (
    SUPERVISED_TEACHER_OPPONENT_BUCKET,
)


def _first_state():
    """Return one real double-six state, enough to exercise every block."""
    return DominoEngine(player_count=2)._get_state()


def _host(array):
    """Return one encoded input as NumPy, whichever backend produced it."""
    return array.get() if hasattr(array, "get") else np.asarray(array)


def _bucket_block(encoder, vector):
    return _host(vector)[encoder.OPPONENT_BUCKET_OFFSET:, 0]


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------


def test_feature_order_is_exactly_the_bucket_registry():
    """The encoder's vocabulary must not drift from the pool's registry.

    ``agents`` cannot import ``training``, so the order is written out in the
    encoder. This is the check that keeps the copy honest: a new bucket added
    to the registry without a matching feature would silently encode as the
    all-zero no-bucket state.
    """
    assert OPPONENT_BUCKET_FEATURE_ORDER == tuple(
        specification.name for specification in BUCKET_REGISTRY
    )


def test_width_covers_every_bucket_not_just_the_selected_ones():
    assert OPPONENT_BUCKET_FEATURE_WIDTH == len(BUCKET_REGISTRY)
    assert OPPONENT_BUCKET_FEATURE_WIDTH == 7


def test_none_is_the_no_bucket_state_and_unknown_names_are_rejected():
    assert opponent_bucket_feature_index(None) is None
    for name in OPPONENT_BUCKET_FEATURE_ORDER:
        assert opponent_bucket_feature_index(name) == (
            OPPONENT_BUCKET_FEATURE_INDEX[name]
        )
    with pytest.raises(ValueError, match="Unknown opponent bucket"):
        opponent_bucket_feature_index("recent_band")


def test_a_misspelled_bucket_fails_when_the_encoder_is_built():
    """Not once per decision, and not silently as an all-zero block."""
    with pytest.raises(ValueError, match="Unknown opponent bucket"):
        DominoEncoder(
            use_opponent_bucket_features=True,
            opponent_bucket="champion",
        )


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ruleset_name", RULESET_NAMES)
def test_block_lengthens_the_vector_by_its_own_width(ruleset_name):
    without = DominoEncoder(ruleset_name)
    with_block = DominoEncoder(
        ruleset_name,
        use_opponent_bucket_features=True,
    )
    assert with_block.vector_size == (
        without.vector_size + OPPONENT_BUCKET_FEATURE_WIDTH
    )
    assert with_block.action_size == without.action_size
    assert with_block.OPPONENT_BUCKET_OFFSET == without.vector_size


def test_double_six_grows_from_168_to_175():
    assert DominoEncoder().vector_size == 168
    assert DominoEncoder(use_opponent_bucket_features=True).vector_size == 175


def test_omitting_the_block_reproduces_the_historical_layout_exactly():
    """The block is trailing, so leaving it off must move nothing at all."""
    historical = DominoEncoder()
    default = DominoEncoder(use_opponent_bucket_features=False)
    assert default.vector_size == historical.vector_size
    assert default.OPPONENT_BUCKET_OFFSET == historical.vector_size
    state = _first_state()
    assert np.array_equal(
        default.encode_state(state),
        historical.encode_state(state),
    )


def test_the_block_never_disturbs_the_features_before_it():
    state = _first_state()
    baseline = DominoEncoder().encode_state(state)
    for name in OPPONENT_BUCKET_FEATURE_ORDER:
        encoder = DominoEncoder(
            use_opponent_bucket_features=True,
            opponent_bucket=name,
        )
        vector = encoder.encode_state(state)
        assert np.array_equal(
            vector[:encoder.OPPONENT_BUCKET_OFFSET],
            baseline,
        )


def test_the_two_encoder_flags_are_independent():
    """Dropping the suit block while adding this one restores 168 features.

    The collision is the reason the run configuration records the flag instead
    of trusting the checkpoint's input width to imply it.
    """
    swapped = DominoEncoder(
        use_opponent_suit_features=False,
        use_opponent_bucket_features=True,
    )
    assert swapped.vector_size == DominoEncoder().vector_size == 168
    assert swapped.OPPONENT_BUCKET_OFFSET == 161


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def test_each_bucket_sets_exactly_its_own_feature():
    state = _first_state()
    for position, name in enumerate(OPPONENT_BUCKET_FEATURE_ORDER):
        encoder = DominoEncoder(
            use_opponent_bucket_features=True,
            opponent_bucket=name,
        )
        block = _bucket_block(encoder, encoder.encode_state(state))
        expected = np.zeros(OPPONENT_BUCKET_FEATURE_WIDTH, dtype=np.float32)
        expected[position] = 1.0
        assert np.array_equal(block, expected)


def test_no_bucket_encodes_as_an_all_zero_block():
    """The human seat in the UI belongs to no bucket and must say so."""
    encoder = DominoEncoder(
        use_opponent_bucket_features=True,
        opponent_bucket=None,
    )
    block = _bucket_block(encoder, encoder.encode_state(state=_first_state()))
    assert not block.any()


def test_the_bucket_is_constant_across_every_turn_of_one_game():
    """It describes the adversary, which cannot change inside a game."""
    encoder = DominoEncoder(
        use_opponent_bucket_features=True,
        opponent_bucket="medium_term",
    )
    engine = DominoEngine(player_count=2)
    blocks = set()
    while not engine.game_over:
        state = engine._get_state()
        blocks.add(tuple(_bucket_block(encoder, encoder.encode_state(state))))
        actions = engine.valid_actions(state["current_player"])
        engine.step(actions[0], return_state=False, legal_actions=actions)
    assert len(blocks) == 1


# --------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ruleset_name", RULESET_NAMES)
def test_architecture_input_matches_the_encoder(ruleset_name):
    architecture = architecture_for_ruleset(
        ruleset_name,
        use_opponent_bucket_features=True,
    )
    encoder = DominoEncoder(
        ruleset_name,
        use_opponent_bucket_features=True,
    )
    assert architecture.input_size == encoder.vector_size
    assert architecture.output_size == encoder.action_size


def test_an_agent_rejects_a_policy_shaped_for_the_other_layout():
    architecture = architecture_for_ruleset()
    network = PolicyNetwork(
        input_size=architecture.input_size,
        output_size=architecture.output_size,
        hidden_sizes=architecture.hidden_sizes,
        random_seed=11,
    )
    with pytest.raises(ValueError, match="does not match ruleset"):
        RLAgent(
            network,
            use_opponent_bucket_features=True,
            opponent_bucket="recent",
        )


def test_checkpoint_shape_check_follows_the_flag():
    architecture = architecture_for_ruleset(
        use_opponent_bucket_features=True,
    )
    network = PolicyNetwork(
        input_size=architecture.input_size,
        output_size=architecture.output_size,
        hidden_sizes=architecture.hidden_sizes,
        random_seed=12,
    )
    assert _checkpoint_matches_encoder(
        network,
        use_opponent_bucket_features=True,
    )
    assert not _checkpoint_matches_encoder(
        network,
        use_opponent_bucket_features=False,
    )


# --------------------------------------------------------------------------
# Rollout: each seat is told about its own adversary
# --------------------------------------------------------------------------


def _rollout_network(seed):
    architecture = architecture_for_ruleset(
        use_opponent_bucket_features=True,
    )
    return PolicyNetwork(
        input_size=architecture.input_size,
        output_size=architecture.output_size,
        hidden_sizes=architecture.hidden_sizes,
        random_seed=seed,
    )


@pytest.mark.parametrize(
    ("opponent_kind", "bucket"),
    [
        ("heuristic", "heuristic"),
        ("random", "random"),
        ("policy_snapshot", "recent"),
        ("policy_snapshot", "medium_term"),
        ("policy_snapshot", "historical_uniform"),
        ("policy_snapshot", "champion_vs_heuristic"),
        ("policy_snapshot", "champion_vs_learner"),
    ],
)
def test_the_learner_sees_the_bucket_its_assignment_named(
    opponent_kind, bucket
):
    """Including both champion buckets, which share ``recent``'s members.

    The bucket therefore cannot be derived from the opponent's kind or even its
    identity; only the assignment knows it.
    """
    random.seed(5)
    np.random.seed(5)
    network = _rollout_network(21)
    opponent = (
        _rollout_network(22) if opponent_kind == "policy_snapshot" else None
    )
    samples, *_ = collect_steps_for_assignment(
        network,
        opponent_kind,
        opponent,
        DEFAULT_REWARD_SCHEMA,
        1.0,
        use_opponent_bucket_features=True,
        opponent_bucket=bucket,
    )
    assert samples
    expected = np.zeros(OPPONENT_BUCKET_FEATURE_WIDTH, dtype=np.float32)
    expected[OPPONENT_BUCKET_FEATURE_INDEX[bucket]] = 1.0
    for sample in samples:
        assert np.array_equal(_host(sample.x)[168:, 0], expected)


def test_the_frozen_opponent_is_told_it_faces_the_recent_learner(monkeypatch):
    """The two seats of one game receive different buckets, by design.

    The learner is told which bucket its adversary came from; the adversary is
    told ``recent``, because what it faces is the current learner.
    """
    built = []
    original = rollout_module.RLAgent

    def record(*args, **kwargs):
        built.append((kwargs.get("mode"), kwargs.get("opponent_bucket")))
        return original(*args, **kwargs)

    monkeypatch.setattr(rollout_module, "RLAgent", record)
    random.seed(6)
    np.random.seed(6)
    collect_steps_for_assignment(
        _rollout_network(31),
        "policy_snapshot",
        _rollout_network(32),
        DEFAULT_REWARD_SCHEMA,
        1.0,
        use_opponent_bucket_features=True,
        opponent_bucket="historical_uniform",
    )
    assert built == [
        ("training", "historical_uniform"),
        ("stochastic_evaluation", LEARNER_OPPONENT_BUCKET),
    ]
    assert LEARNER_OPPONENT_BUCKET == "recent"


def test_the_block_stays_empty_while_the_flag_is_off():
    """The default rollout must produce the historical 168-wide input."""
    random.seed(7)
    np.random.seed(7)
    architecture = architecture_for_ruleset()
    network = PolicyNetwork(
        input_size=architecture.input_size,
        output_size=architecture.output_size,
        hidden_sizes=architecture.hidden_sizes,
        random_seed=41,
    )
    samples, *_ = collect_steps_for_assignment(
        network,
        "heuristic",
        None,
        DEFAULT_REWARD_SCHEMA,
        1.0,
        opponent_bucket="heuristic",
    )
    assert samples
    for sample in samples:
        assert _host(sample.x).shape[0] == 168


# --------------------------------------------------------------------------
# Contexts outside the RL rollout
# --------------------------------------------------------------------------


def test_the_supervised_teacher_is_a_real_bucket_the_rollout_reuses():
    """The dataset is StrategicAgent against StrategicAgent.

    Pretraining therefore has to encode ``heuristic``, not the no-bucket state,
    or it would teach a pattern the RL stage never shows again.
    """
    assert SUPERVISED_TEACHER_OPPONENT_BUCKET == "heuristic"
    assert SUPERVISED_TEACHER_OPPONENT_BUCKET in OPPONENT_BUCKET_FEATURE_ORDER


@pytest.mark.parametrize(
    ("agent_name", "bucket"),
    [
        ("heuristic", "heuristic"),
        ("random", "random"),
        ("rl", "recent"),
        ("neural", "recent"),
    ],
)
def test_diagnostic_agents_map_onto_the_bucket_they_stand_for(
    agent_name, bucket
):
    assert opponent_bucket_for_agent(agent_name) == bucket
    assert bucket in OPPONENT_BUCKET_FEATURE_ORDER


def test_supervised_assets_of_the_two_regimes_do_not_collide(tmp_path):
    """Both flags claim their own suffix, and neither renames the default."""
    default = canonical_asset_paths(tmp_path, 42)
    with_block = canonical_asset_paths(
        tmp_path,
        42,
        use_opponent_bucket_features=True,
    )
    swapped = canonical_asset_paths(
        tmp_path,
        42,
        use_opponent_suit_features=False,
        use_opponent_bucket_features=True,
    )
    assert default.dataset.name == "supervised_dataset_standard_seed42.jsonl"
    assert "_bucket" in with_block.dataset.name
    assert "_nosuit_bucket" in swapped.dataset.name
    names = {default.weights, with_block.weights, swapped.weights}
    assert len(names) == 3


# --------------------------------------------------------------------------
# Command line and run identity
# --------------------------------------------------------------------------


def _standalone(argv):
    parser = argparse.ArgumentParser()
    add_optional_rl_arguments(parser)
    return parser.parse_args(argv)


ENTRY_POINTS = pytest.mark.parametrize(
    "parse",
    [
        lambda argv: parse_args(["small", *argv]),
        _standalone,
    ],
    ids=["pipeline", "rl-cli"],
)


@ENTRY_POINTS
def test_the_flag_is_off_unless_it_is_passed(parse):
    assert parse([]).opponent_bucket_features is False
    assert parse(["--opponent-bucket-features"]).opponent_bucket_features


def test_the_rl_cli_carries_the_flag_into_training_options():
    options = training_options_from_args(
        _standalone(["--opponent-bucket-features"])
    )
    assert options[0].use_opponent_bucket_features is True
    assert training_options_from_args(
        _standalone([])
    )[0].use_opponent_bucket_features is False


def test_the_pipeline_sizes_its_network_from_the_flag():
    enabled = parse_args(["small", "--opponent-bucket-features"])
    assert _use_opponent_bucket_features(enabled) is True
    assert _network_architecture(enabled).input_size == 175
    default = parse_args(["small"])
    assert _use_opponent_bucket_features(default) is False
    assert _network_architecture(default).input_size == 168


def test_the_flag_is_locked_rather_than_part_of_rl_config():
    """``rl_config`` is rebuilt and compared as an immutable run key.

    A new member there would make every run created before the flag existed
    unresumable, which is why the value travels in ``locked_arguments``.
    """
    args = parse_args(["small", "--opponent-bucket-features"])
    assert "opponent_bucket_features" not in _rl_config(args)
    locked = _locked_run_arguments(args)
    assert locked["opponent_bucket_features"] is True
    assert run_config_uses_opponent_bucket_features(
        {"locked_arguments": locked}
    )


def test_a_run_predating_the_flag_reads_as_disabled():
    assert run_config_uses_opponent_bucket_features({}) is False
    assert run_config_uses_opponent_bucket_features(
        {"locked_arguments": {}}
    ) is False
    assert run_config_uses_opponent_bucket_features(None) is False


def _configuration_mapping(**overrides):
    mapping = {
        "total_training_games": 10,
        "ruleset_name": "double-six",
        "selected_gpi": 4,
        "selected_workers": 1,
        "log_interval": 1,
        "checkpoint_interval": 1,
        "moving_average_window": 2,
        "opponent_buckets": ("heuristic", "recent"),
        "difficulty_weight": 0.5,
        "opponent_decision_restarts": False,
        "learning_rate": 0.001,
        "entropy_coef": 0.01,
        "use_value_head": False,
        "value_coef": 0.5,
        "gamma_f": 1.0,
        "terminal_empty_hand_weight": 1.0,
        "terminal_blocked_weight": 1.0,
        "immediate_draw_weight": 1.0,
        "immediate_pass_weight": 1.0,
        "reward_eta": 0.5,
        "gamma_i": 0.9,
        "normalize_advantages": True,
        "baseline": None,
        "weight_decay": 0.0,
        "dropout_rate": 0.0,
        "effective_seed": 1,
        "device": "cpu",
        "sl_weights_sha256": None,
        "ppo_max_epochs": 4,
        "worker_memory_reserve_mb": 1,
        "worker_estimated_mb": 1,
        "worker_max_rss_mb": 1,
        "opponent_system_policy": {},
    }
    mapping.update(overrides)
    return mapping


def test_a_checkpoint_predating_the_flag_resumes_as_disabled():
    configuration = RLTrainingConfiguration.from_mapping(
        _configuration_mapping()
    )
    assert configuration.use_opponent_bucket_features is False


def test_the_flag_is_part_of_the_durable_resume_configuration():
    """A shape check cannot see a suit-for-bucket swap; this field can."""
    enabled = RLTrainingConfiguration.from_mapping(
        _configuration_mapping(use_opponent_bucket_features=True)
    )
    disabled = RLTrainingConfiguration.from_mapping(
        _configuration_mapping(use_opponent_bucket_features=False)
    )
    assert enabled.use_opponent_bucket_features is True
    assert enabled != disabled


# ---------------------------------------------------------------------------
# The UI seat
# ---------------------------------------------------------------------------
#
# A human belongs to no bucket, but the block is one-hot in every vector the
# policy trained on, so an all-zero block is a pattern it has never seen rather
# than a neutral input. The UI hands the heuristic slot instead. These tests
# pin that choice, and pin that it never reaches a checkpoint trained without
# the block.


def _policy_checkpoint(tmp_path, *, use_opponent_bucket_features):
    """Write a minimal policy .npz shaped for one encoder layout."""
    encoder = DominoEncoder(
        "double-six",
        use_opponent_bucket_features=use_opponent_bucket_features,
    )
    hidden = 8
    outputs = len(encoder.all_actions)
    path = tmp_path / "policy.npz"
    np.savez(
        path,
        W1=np.zeros((hidden, encoder.vector_size)),
        b1=np.zeros((hidden, 1)),
        W2=np.zeros((outputs, hidden)),
        b2=np.zeros((outputs, 1)),
    )
    return path


def test_the_ui_bucket_is_a_real_slot_in_the_one_hot():
    """A stand-in the encoder cannot place would silently encode all zeros."""
    assert UI_OPPONENT_BUCKET in OPPONENT_BUCKET_FEATURE_ORDER
    assert opponent_bucket_feature_index(UI_OPPONENT_BUCKET) is not None


def test_the_ui_names_the_bucket_supervised_pretraining_uses():
    """The slot is chosen because every checkpoint has always seen it."""
    assert UI_OPPONENT_BUCKET == SUPERVISED_TEACHER_OPPONENT_BUCKET


def test_the_widened_layout_is_recognized_and_given_the_heuristic_slot():
    arguments = _bucket_features_for(
        DominoEncoder(
            "double-six", use_opponent_bucket_features=True
        ).vector_size,
        "double-six",
    )
    assert arguments == {
        "use_opponent_bucket_features": True,
        "opponent_bucket": UI_OPPONENT_BUCKET,
    }


@pytest.mark.parametrize("input_size", [161, 168])
def test_a_default_width_checkpoint_keeps_the_historical_layout(input_size):
    """168 is ambiguous, so the layout that predates both flags wins."""
    assert _bucket_features_for(input_size, "double-six") == {}


def test_the_ui_gives_a_bucket_trained_policy_the_heuristic_bit(tmp_path):
    path = _policy_checkpoint(tmp_path, use_opponent_bucket_features=True)
    network = PolicyNetwork.load(path)
    agent = RLAgent(
        network,
        mode="evaluation",
        ruleset="double-six",
        **_bucket_features_for(int(network.W1.shape[1]), "double-six"),
    )
    assert agent.use_opponent_bucket_features is True
    assert agent.opponent_bucket == UI_OPPONENT_BUCKET
    assert agent.encoder.opponent_bucket_index == (
        OPPONENT_BUCKET_FEATURE_INDEX[UI_OPPONENT_BUCKET]
    )


def test_the_ui_leaves_a_policy_trained_without_the_block_untouched(tmp_path):
    """The detection must never widen the input a checkpoint never had."""
    path = _policy_checkpoint(tmp_path, use_opponent_bucket_features=False)
    network = PolicyNetwork.load(path)
    agent = RLAgent(
        network,
        mode="evaluation",
        ruleset="double-six",
        **_bucket_features_for(int(network.W1.shape[1]), "double-six"),
    )
    assert agent.use_opponent_bucket_features is False
    assert agent.opponent_bucket is None
    assert agent.encoder.vector_size == int(network.W1.shape[1])


def test_the_checkpoint_peek_reports_the_trained_input_width(tmp_path):
    wide = _policy_checkpoint(tmp_path, use_opponent_bucket_features=True)
    assert _checkpoint_input_size(wide) == DominoEncoder(
        "double-six", use_opponent_bucket_features=True
    ).vector_size
