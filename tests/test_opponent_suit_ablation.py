"""Contracts for the ablated encoder layout that drops the exact-model block.

The ablation exists so one experiment can measure whether the policy learns
better or worse without the opponent suit-presence features. Every test here
protects one property that experiment depends on: the shortened vector, the
untouched prefix, and the guarantee that the exact model is never consulted
when the block is absent.
"""

import argparse
import importlib
import inspect
import random

import numpy as np
import pytest

import agents.encoder as encoder_module
from agents.encoder import DominoEncoder
from agents.heuristic_agent import StrategicAgent
from agents.network_architecture import (
    architecture_for_ruleset,
    architecture_from_hidden_sizes,
)
from agents.neural_agent import NeuralAgent
from agents.rl_agent import RLAgent
from agents.rl_nn import PolicyNetwork
from middleware.domino_engine import DominoEngine
from middleware.middleware import GameManager
from middleware.rulesets import RULESET_NAMES, resolve_ruleset
from diagnostics.gameplay import create_agent
from diagnostics.parallel_runner import ParallelSafetyConfig
from diagnostics.worker_autotune import MatchupSpec
from training.canonical_assets import canonical_asset_paths
from training.canonical_run import (
    create_run_config,
    run_config_uses_opponent_suit_features,
)
from training.pipeline import (
    _RESUME_OPERATIONAL_ARGUMENTS,
    _locked_run_arguments,
    _network_architecture,
    _rl_config,
    _use_opponent_suit_features,
    parse_args,
)
from training.rl.adaptive_tuning import _new_runner
from training.rl.config import RLTrainingOptions
from training.rl.cli import (
    add_optional_rl_arguments,
    training_options_from_args,
)
from training.rl.parallel import RLRolloutRunner
from training.rl.ppo import PPO_TRAINING_ALGORITHM
from training.rl.rollout import REWARD_SCHEMAS, collect_steps_for_assignment
from training.rl.resume import _checkpoint_matches_encoder
from training.datagen.parallel import (
    _is_real_decision_state,
    _normalize_action,
)


def _ablated(ruleset):
    return DominoEncoder(ruleset, use_opponent_suit_features=False)


@pytest.mark.parametrize(
    ("name", "enabled_size", "ablated_size", "action_size"),
    [
        ("double-six", 168, 161, 56),
        ("double-five", 130, 124, 42),
        ("double-four", 97, 92, 30),
        ("double-three", 69, 65, 20),
    ],
)
def test_ablated_encoder_drops_one_suit_block(
    name,
    enabled_size,
    ablated_size,
    action_size,
):
    """The vector loses ``S`` features and the action space is untouched."""
    enabled = DominoEncoder(name)
    ablated = _ablated(name)

    assert enabled.vector_size == enabled_size
    assert ablated.vector_size == ablated_size
    assert ablated_size == enabled_size - resolve_ruleset(name).pip_count
    assert enabled.action_size == ablated.action_size == action_size


@pytest.mark.parametrize("name", RULESET_NAMES)
def test_ablation_moves_no_other_offset(name):
    """Only the trailing block disappears, so every prior offset is stable."""
    enabled = DominoEncoder(name).layout
    ablated = _ablated(name).layout
    stable_fields = (
        "hand",
        "played",
        "played_turn",
        "played_by_me",
        "played_by_opponent",
        "left_end",
        "right_end",
        "hand_size",
        "stock_size",
        "draw_count",
        "pass_count",
        "opponent_suit_probability",
    )

    for field in stable_fields:
        assert getattr(ablated, field) == getattr(enabled, field), field
    # An empty trailing block starts exactly where the vector ends.
    assert ablated.opponent_suit_probability == ablated.vector_size


def test_encoder_defaults_to_the_historical_layout():
    """Omitting the flag must reproduce the pre-ablation encoder exactly."""
    encoder = DominoEncoder("double-six")
    assert encoder.use_opponent_suit_features is True
    assert encoder.vector_size == 168


def _real_decision_states(game_count=5, ruleset="double-six"):
    """Return the states the supervised generator would actually save."""
    states = []
    for game_index in range(game_count):
        engine = DominoEngine(
            player_count=2,
            rng=random.Random(4200 + game_index),
            ruleset=ruleset,
        )
        manager = GameManager(
            engine,
            [StrategicAgent(ruleset=ruleset), StrategicAgent(ruleset=ruleset)],
        )
        _, history = manager.play_full_game()
        for turn in history:
            if _normalize_action(turn["target_action"]) is None:
                continue
            if not _is_real_decision_state(turn["state"]):
                continue
            states.append(turn["state"])
    return states


def test_saved_dataset_states_always_carry_the_probabilities():
    """The heuristic writes the key into the very dict the dataset stores.

    This is why the ablation has to cut inside ``encode_state``: an encoder that
    merely stopped writing the key would still find one waiting for it.
    """
    states = _real_decision_states()

    assert states
    assert all("opponent_suit_probabilities" in state for state in states)


def test_ablated_prefix_is_identical_to_the_enabled_vector():
    """Dropping the block must not perturb a single earlier feature."""
    states = _real_decision_states()
    enabled = DominoEncoder("double-six")
    ablated = _ablated("double-six")

    enabled_batch = np.hstack([enabled.encode_state(state) for state in states])
    ablated_batch = np.hstack([ablated.encode_state(state) for state in states])

    assert enabled_batch.shape == (168, len(states))
    assert ablated_batch.shape == (161, len(states))
    assert ablated_batch.dtype == np.float32
    np.testing.assert_array_equal(ablated_batch, enabled_batch[:161])
    # The block being removed really did carry information in these states.
    assert np.any(enabled_batch[161:] != 0.0)


def test_ablated_encoder_ignores_a_probability_value_present_in_the_state():
    """A stored value must not leak back in through any path."""
    states = _real_decision_states()
    ablated = _ablated("double-six")
    state = states[0]

    poisoned = dict(state)
    poisoned["opponent_suit_probabilities"] = [1.0] * 7
    without_key = {
        key: value
        for key, value in state.items()
        if key != "opponent_suit_probabilities"
    }

    np.testing.assert_array_equal(
        ablated.encode_state(poisoned),
        ablated.encode_state(without_key),
    )


def test_ablated_encoder_never_consults_the_exact_model(monkeypatch):
    """No fallback reconstruction may run when the block is absent."""
    states = _real_decision_states()
    ablated = _ablated("double-six")
    enabled = DominoEncoder("double-six")

    calls = []

    def _record(state):
        calls.append(state)
        return [0.0] * 7

    monkeypatch.setattr(
        encoder_module,
        "compute_opponent_suit_probabilities",
        _record,
    )
    stripped = [
        {
            key: value
            for key, value in state.items()
            if key != "opponent_suit_probabilities"
        }
        for state in states
    ]

    for state in stripped:
        ablated.encode_state(state)
    assert calls == []

    # The enabled encoder still reconstructs, so the guard is the flag and not
    # a change in how missing probabilities are handled.
    enabled.encode_state(stripped[0])
    assert len(calls) == 1


def test_ablated_encoder_skips_the_probability_width_check():
    """With no block to fill there is nothing to validate."""
    state = {
        "ruleset_name": "double-three",
        "initial_hand_size": 4,
        "current_player": 0,
        "current_player_hand": [[0, 0]],
        "hand_sizes": [4, 4],
        "stock_size": 2,
        "ends": [],
        "board_history": [],
        "opponent_suit_probabilities": [0.0] * 7,
    }

    with pytest.raises(ValueError, match="expected 4"):
        DominoEncoder("double-three").encode_state(state)

    vector = _ablated("double-three").encode_state(state)
    assert vector.shape == (65, 1)


class _ShapedPolicy:
    """Minimal policy stand-in whose only contract is its weight shapes."""

    xp = np

    def __init__(self, input_size, output_size):
        self.W1 = np.zeros((2, input_size), dtype=np.float32)
        self.W3 = np.zeros((output_size, 2), dtype=np.float32)
        self.layer_count = 3
        self.output_size = output_size

    def forward(self, x):
        return np.ones((self.output_size, 1), dtype=np.float32) / self.output_size


@pytest.mark.parametrize("agent_factory", ["rl", "neural"])
def test_ablated_agents_never_build_the_exact_model(agent_factory):
    """The model only fills the trailing block, so it must not be constructed."""
    encoder = _ablated("double-six")
    if agent_factory == "rl":
        enabled = RLAgent(_ShapedPolicy(168, 56), mode="evaluation")
        ablated = RLAgent(
            _ShapedPolicy(encoder.vector_size, encoder.action_size),
            mode="evaluation",
            use_opponent_suit_features=False,
        )
    else:
        enabled = NeuralAgent(_ShapedPolicy(168, 56))
        ablated = NeuralAgent(
            _ShapedPolicy(encoder.vector_size, encoder.action_size),
            use_opponent_suit_features=False,
        )

    assert enabled.opponent_model is not None
    assert ablated.opponent_model is None
    assert ablated.encoder.vector_size == 161


def test_ablated_rl_agent_rejects_a_full_width_policy():
    """The shape guard makes cross-regime checkpoint reuse impossible."""
    with pytest.raises(ValueError, match="does not match ruleset"):
        RLAgent(
            _ShapedPolicy(168, 56),
            mode="evaluation",
            use_opponent_suit_features=False,
        )
    with pytest.raises(ValueError, match="does not match ruleset"):
        RLAgent(_ShapedPolicy(161, 56), mode="evaluation")


def test_neural_agent_load_rejects_a_cross_regime_checkpoint(tmp_path):
    """A 168-input checkpoint cannot be loaded into an ablated agent."""
    from agents.network_architecture import architecture_for_ruleset

    architecture = architecture_for_ruleset("double-six")
    checkpoint = tmp_path / "enabled.npz"
    np.savez(
        checkpoint,
        **{
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in architecture.policy_weight_shapes().items()
        },
    )

    NeuralAgent.load(checkpoint, ruleset="double-six")
    with pytest.raises(ValueError, match="produces 161"):
        NeuralAgent.load(
            checkpoint,
            ruleset="double-six",
            use_opponent_suit_features=False,
        )


@pytest.mark.parametrize("agent_factory", ["rl", "neural"])
def test_ablated_agents_leave_the_state_untouched_while_choosing(agent_factory):
    """No probability key is written, and no exact inference runs."""
    states = _real_decision_states()
    state = dict(states[0])
    state.pop("opponent_suit_probabilities", None)
    hand = [tuple(tile) for tile in state["current_player_hand"]]
    left, right = state["ends"]
    legal = [
        (tile, side)
        for tile in hand
        for side, end in ((0, left), (1, right))
        if end in tile
    ]
    assert len(legal) >= 2

    if agent_factory == "rl":
        agent = RLAgent(
            _ShapedPolicy(161, 56),
            mode="evaluation",
            use_opponent_suit_features=False,
        )
    else:
        agent = NeuralAgent(
            _ShapedPolicy(161, 56),
            use_opponent_suit_features=False,
        )

    chosen = agent.choose_move(state, legal)

    assert chosen in legal
    assert "opponent_suit_probabilities" not in state


def test_heuristic_agent_always_keeps_the_exact_model():
    """The reference opponent is deliberately outside the ablation's reach."""
    agent = StrategicAgent(ruleset="double-six")

    assert agent.opponent_model is not None
    assert not hasattr(agent, "use_opponent_suit_features")


def test_ablated_rl_agent_plays_a_full_game_against_the_heuristic():
    """End-to-end: one side is ablated, the reference opponent is not."""
    ablated = RLAgent(
        _ShapedPolicy(161, 56),
        mode="evaluation",
        use_opponent_suit_features=False,
    )
    heuristic = StrategicAgent(ruleset="double-six")
    engine = DominoEngine(
        player_count=2,
        rng=random.Random(31415),
        ruleset="double-six",
    )
    manager = GameManager(engine, [ablated, heuristic])

    info, history = manager.play_full_game()

    assert history
    assert info
    assert ablated.opponent_model is None
    assert heuristic.opponent_model is not None
    # The heuristic still publishes its inference on the turns it owns, and the
    # ablated agent never does on the turns it owns.
    learner_turns = [
        turn["state"]
        for turn in history
        if turn["state"]["current_player"] == 0
    ]
    assert learner_turns
    assert all(
        "opponent_suit_probabilities" not in state for state in learner_turns
    )


@pytest.mark.parametrize(
    ("name", "enabled_input", "ablated_input", "action_size"),
    [
        ("double-six", 168, 161, 56),
        ("double-five", 130, 124, 42),
        ("double-four", 97, 92, 30),
        ("double-three", 69, 65, 20),
    ],
)
def test_architecture_builders_follow_the_ablated_input(
    name,
    enabled_input,
    ablated_input,
    action_size,
):
    """A run must never build a network wider than its encoder's output."""
    enabled = architecture_for_ruleset(name)
    ablated = architecture_for_ruleset(name, use_opponent_suit_features=False)

    assert enabled.input_size == enabled_input
    assert ablated.input_size == ablated_input
    assert enabled.output_size == ablated.output_size == action_size
    assert ablated.hidden_sizes == enabled.hidden_sizes

    explicit = architecture_from_hidden_sizes(
        (64, 32),
        ruleset=name,
        use_opponent_suit_features=False,
    )
    assert explicit.layer_dimensions == (ablated_input, 64, 32, action_size)


def test_resume_shape_guard_follows_the_ablation():
    """F7: nightly resumes of the ablated run must accept its own checkpoint.

    Without the flag reaching this guard, a 161-input checkpoint would be
    rejected against a 168-wide encoder and a long ``forever`` run would die on
    its first resume.
    """
    ablated_policy = _ShapedPolicy(161, 56)
    enabled_policy = _ShapedPolicy(168, 56)

    assert _checkpoint_matches_encoder(enabled_policy, "double-six")
    assert not _checkpoint_matches_encoder(ablated_policy, "double-six")
    assert _checkpoint_matches_encoder(
        ablated_policy,
        "double-six",
        use_opponent_suit_features=False,
    )
    assert not _checkpoint_matches_encoder(
        enabled_policy,
        "double-six",
        use_opponent_suit_features=False,
    )


def test_run_config_identity_separates_the_two_regimes(tmp_path):
    """The two runs must never collide on ``configuration_sha256``."""
    def _build(run_dir, use_features):
        architecture = architecture_for_ruleset(
            "double-six",
            use_opponent_suit_features=use_features,
        )
        return create_run_config(
            run_dir,
            root=tmp_path,
            pipeline_level="small",
            seed=7,
            target_rl_games=10,
            supervised_weights_path=tmp_path / "sl.npz",
            supervised_weights_sha256="0" * 64,
            ppo_config={},
            rl_config={},
            algorithm=PPO_TRAINING_ALGORITHM,
            diagnostic_config={},
            network_architecture=architecture,
            ruleset="double-six",
            use_opponent_suit_features=use_features,
        )

    enabled = _build(tmp_path / "enabled", True)
    ablated = _build(tmp_path / "ablated", False)

    assert enabled["encoder_size"] == 168
    assert ablated["encoder_size"] == 161
    assert enabled["action_count"] == ablated["action_count"] == 56
    assert (
        enabled["configuration_sha256"] != ablated["configuration_sha256"]
    )


def test_cli_flag_defaults_to_keeping_the_features():
    """Omitting the flag must leave the historical training options intact."""
    parser = argparse.ArgumentParser()
    add_optional_rl_arguments(parser)
    args = parser.parse_args([])

    assert args.no_opponent_suit_features is False
    training, _resources, _execution = training_options_from_args(args)
    assert training.use_opponent_suit_features is True


def test_cli_flag_turns_the_features_off():
    """``--no-opponent-suit-features`` reaches ``RLTrainingOptions``."""
    parser = argparse.ArgumentParser()
    add_optional_rl_arguments(parser)
    args = parser.parse_args(["--no-opponent-suit-features"])

    training, _resources, _execution = training_options_from_args(args)
    assert training.use_opponent_suit_features is False


def test_ablated_rollout_collects_real_decisions_end_to_end():
    """A full ablated rollout runs through rollout.py without the exact model.

    This is the integration that proves Etapa 4's plumbing: a 161-input policy
    plays the heuristic, and every collected decision carries the shortened
    encoding.
    """
    network = PolicyNetwork(
        input_size=161,
        output_size=56,
        hidden_sizes=(8,),
        learning_rate=0.01,
        random_seed=17,
        device="cpu",
    )
    random.seed(2026)

    samples, events, winner, learner_position = collect_steps_for_assignment(
        network,
        "heuristic",
        None,
        REWARD_SCHEMAS,
        1.0,
        ruleset_name="double-six",
        use_opponent_suit_features=False,
    )[:4]

    assert winner in (0, 1)
    assert learner_position in (0, 1)
    assert events.opponent_draws >= 0
    assert events.learner_draws >= 0
    assert samples
    for sample in samples:
        assert sample.x.shape == (161, 1)
        assert sample.legal_mask.shape[0] == 56


def test_rollout_runner_forwards_the_flag_to_its_workers():
    """The runner stores the flag so ``initargs`` can hand it to workers."""
    network = PolicyNetwork(
        input_size=161,
        output_size=56,
        hidden_sizes=(8,),
        learning_rate=0.01,
        random_seed=5,
        device="cpu",
    )
    runner = RLRolloutRunner(
        network,
        opponent_buckets=("heuristic",),
        schema=REWARD_SCHEMAS,
        gamma_f=1.0,
        ruleset_name="double-six",
        use_opponent_suit_features=False,
    )
    try:
        assert runner.use_opponent_suit_features is False
    finally:
        runner.close()


def test_pipeline_flag_selects_the_network_width(tmp_path):
    """One CLI switch has to reach the architecture both stages are built from.

    ``forever`` hydrates its arguments from whatever run the artifact root
    already points at, so the root is redirected to keep the test hermetic.
    """
    enabled = parse_args(["forever", "--artifact-root", str(tmp_path)])
    ablated = parse_args([
        "forever",
        "--artifact-root",
        str(tmp_path),
        "--no-opponent-suit-features",
    ])

    assert _use_opponent_suit_features(enabled) is True
    assert _use_opponent_suit_features(ablated) is False
    assert _network_architecture(enabled).input_size == 168
    assert _network_architecture(ablated).input_size == 161
    assert _network_architecture(ablated).output_size == 56


def test_pipeline_flag_is_locked_but_stays_out_of_rl_config(tmp_path):
    """F6: ``rl_config`` is rebuilt on every resume and is an immutable key.

    Recording the flag there would make every canonical run predating the flag
    unresumable. ``locked_arguments`` is reused from the saved run config
    instead, so it carries the flag without that hazard.
    """
    enabled = parse_args(["forever", "--artifact-root", str(tmp_path)])
    ablated = parse_args([
        "forever",
        "--artifact-root",
        str(tmp_path),
        "--no-opponent-suit-features",
    ])

    enabled_locked = _locked_run_arguments(enabled)
    ablated_locked = _locked_run_arguments(ablated)

    assert enabled_locked["no_opponent_suit_features"] is False
    assert ablated_locked["no_opponent_suit_features"] is True
    assert [
        key
        for key in enabled_locked
        if enabled_locked[key] != ablated_locked.get(key)
    ] == ["no_opponent_suit_features"]
    assert "artifact_root" in _RESUME_OPERATIONAL_ARGUMENTS

    for args in (enabled, ablated):
        assert not any("suit" in key for key in _rl_config(args))


def test_pipeline_flag_is_not_a_resume_operational_argument():
    """A locked argument is what separates the two runs' identities."""
    assert "no_opponent_suit_features" not in _RESUME_OPERATIONAL_ARGUMENTS


def test_canonical_asset_paths_isolate_the_two_regimes(tmp_path):
    """D1: reusable assets are seed-addressed, so the regimes need own paths.

    Sharing them would make the second run refuse to start on an encoder-size
    mismatch, and retraining past that would leave the first unresumable.
    """
    enabled = canonical_asset_paths(tmp_path, 42)
    ablated = canonical_asset_paths(tmp_path, 42, use_opponent_suit_features=False)

    for field in ("dataset", "dataset_meta", "encoded_cache", "weights", "weights_meta"):
        assert getattr(enabled, field) != getattr(ablated, field), field
    assert ablated.weights.name == "domino_sl_standard_seed42_nosuit.npz"
    # The enabled path must stay byte-identical to the historical one.
    assert enabled.weights.name == "domino_sl_standard_seed42.npz"
    assert enabled.dataset.name == "supervised_dataset_standard_seed42.jsonl"


def test_canonical_asset_paths_keep_the_ruleset_prefix(tmp_path):
    """The two suffix rules compose instead of overwriting each other."""
    paths = canonical_asset_paths(
        tmp_path,
        7,
        "double-four",
        use_opponent_suit_features=False,
    )
    assert paths.weights.name == "domino_sl_double-four_standard_seed7_nosuit.npz"


def test_run_config_reader_recovers_the_regime_from_locked_arguments():
    """Diagnostics must read the regime the run recorded, not current args."""
    assert run_config_uses_opponent_suit_features({}) is True
    assert run_config_uses_opponent_suit_features(
        {"locked_arguments": {}}
    ) is True
    assert run_config_uses_opponent_suit_features(
        {"locked_arguments": {"no_opponent_suit_features": False}}
    ) is True
    assert run_config_uses_opponent_suit_features(
        {"locked_arguments": {"no_opponent_suit_features": True}}
    ) is False


def test_diagnostic_agent_factory_spares_only_the_reference_opponents(tmp_path):
    """RL and neural agents follow the regime; heuristic and random do not."""
    architecture = architecture_for_ruleset(
        "double-six",
        use_opponent_suit_features=False,
    )
    checkpoint = tmp_path / "ablated_rl.npz"
    np.savez(
        checkpoint,
        **{
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in architecture.policy_weight_shapes().items()
        },
    )

    agent = create_agent(
        "rl",
        checkpoint,
        "double-six",
        use_opponent_suit_features=False,
    )
    assert agent.encoder.vector_size == 161
    assert agent.opponent_model is None

    heuristic = create_agent(
        "heuristic",
        None,
        "double-six",
        use_opponent_suit_features=False,
    )
    assert heuristic.opponent_model is not None


# Every entry point that can build an encoder-shaped object. A member missing
# the flag silently defaults to the full-width layout, which surfaces only as a
# shape error deep inside a worker process — the failure mode that let the
# adaptive-tuning path ship broken. Keeping the roster executable turns that
# into a test failure instead.
_FLAG_AWARE_CALLABLES = (
    ("agents.encoder", "DominoEncoder.__init__"),
    ("agents.rl_agent", "RLAgent.__init__"),
    ("agents.rl_agent", "RLAgent.load"),
    ("agents.neural_agent", "NeuralAgent.__init__"),
    ("agents.neural_agent", "NeuralAgent.load"),
    ("agents.network_architecture", "architecture_for_ruleset"),
    ("agents.network_architecture", "architecture_from_hidden_sizes"),
    ("training.rl.parallel", "RLRolloutRunner.__init__"),
    ("training.rl.parallel", "_worker_initializer"),
    ("training.rl.rollout", "collect_steps_for_assignment"),
    ("training.rl.rollout", "collect_steps_from_restart"),
    ("training.rl.champion_evaluation", "play_champion_game"),
    ("training.rl.adaptive_tuning", "run_worker_tuning"),
    ("training.rl.adaptive_tuning", "benchmark_worker_candidates"),
    ("training.rl.adaptive_tuning", "_new_runner"),
    ("training.rl.resume", "_load_initial_network"),
    ("training.rl.resume", "_checkpoint_matches_encoder"),
    ("training.canonical_assets", "canonical_asset_paths"),
    ("training.canonical_assets", "inspect_canonical_dataset"),
    ("training.canonical_assets", "write_dataset_metadata"),
    ("training.canonical_assets", "inspect_canonical_weights"),
    ("training.canonical_assets", "write_weights_metadata"),
    ("training.canonical_run", "create_run_config"),
    ("training.supervised.training_loop", "train_supervised"),
    ("diagnostics.gameplay", "create_agent"),
    ("diagnostics.pairwise", "evaluate_pair"),
    ("diagnostics.pairwise", "run_pairwise"),
    ("diagnostics.evaluate", "run_all_pairs"),
    ("diagnostics.parallel_runner", "evaluate_game_specs"),
    ("diagnostics.parallel_runner", "_worker_initializer"),
)


@pytest.mark.parametrize(("module_name", "dotted"), _FLAG_AWARE_CALLABLES)
def test_every_encoder_shaped_entry_point_exposes_the_flag(module_name, dotted):
    """No path may build a network or agent without being told the regime."""
    target = importlib.import_module(module_name)
    for part in dotted.split("."):
        target = getattr(target, part)

    parameters = inspect.signature(target).parameters
    assert "use_opponent_suit_features" in parameters, dotted
    assert parameters["use_opponent_suit_features"].default is True, dotted


@pytest.mark.parametrize(
    "dataclass_type",
    [RLTrainingOptions, MatchupSpec],
    ids=["RLTrainingOptions", "MatchupSpec"],
)
def test_flag_carriers_default_to_keeping_the_features(dataclass_type):
    """The dataclasses that ferry the regime across process boundaries."""
    field = dataclass_type.__dataclass_fields__["use_opponent_suit_features"]
    assert field.default is True


def test_adaptive_tuning_runner_inherits_the_regime():
    """The worker-count benchmark builds its own runner, and it must match.

    A full-width runner here fails only once a rollout worker tries to feed a
    168-wide vector to a 161-wide policy, and the parent sees an opaque
    ``RLRolloutExecutionError`` instead of the real cause.
    """
    network = PolicyNetwork(
        input_size=161,
        output_size=56,
        hidden_sizes=(8,),
        learning_rate=0.01,
        random_seed=11,
        device="cpu",
    )
    runner = _new_runner(
        network,
        opponent_buckets=("heuristic",),
        schema=REWARD_SCHEMAS,
        gamma_f=1.0,
        ruleset_name="double-six",
        use_opponent_suit_features=False,
        safety=ParallelSafetyConfig(),
        pool_state=None,
        pool_weights={},
        performance_state=None,
    )
    try:
        assert runner.use_opponent_suit_features is False
    finally:
        runner.close()
