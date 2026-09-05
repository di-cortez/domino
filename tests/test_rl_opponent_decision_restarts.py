"""Exact-state and deterministic same-iteration RL restart tests."""

import random
from dataclasses import replace
from pathlib import Path

import numpy as np

from agents.encoder import DominoEncoder
from agents.network_architecture import default_hidden_sizes
from agents.rl_nn import PolicyNetwork
from diagnostics.parallel_runner import ParallelSafetyConfig
from middleware.domino_engine import DominoEngine
from training.rl.matchmaking import UniformRotationState, build_match_plan
from training.rl.parallel import RLRolloutRunner
from training.rl.config import (
    RLExecutionOptions,
    RLResourceOptions,
    RLTrainingOptions,
)
from training.rl.ppo import PPO_TRAINING_ALGORITHM
from training.rl.resume import _load_initial_network, resume_state_path
from training.rl.restarts import OpponentDecisionRestart
from training.rl.rollout import (
    DEFAULT_REWARD_SCHEMA,
    _collect_steps_vs_random,
    _tile_play_actions,
    collect_steps_from_restart,
)
from training.rl.training_loop import train
from training.rl.cli import parse_args as parse_rl_args
from training.pipeline import parse_args as parse_pipeline_args


RULESET = "double-three"


def test_restart_flag_is_optional_in_standalone_and_canonical_clis(tmp_path):
    assert parse_rl_args([]).opponent_decision_restarts is False
    assert parse_rl_args([
        "--opponent-decision-restarts"
    ]).opponent_decision_restarts is True
    # An isolated artifact root keeps this parsing contract independent of any
    # real canonical run on the machine. Without it the parser finds
    # models/rl/active_forever_run.json and rejects the flag as conflicting
    # with that run's saved configuration.
    assert parse_pipeline_args([
        "forever",
        "--opponent-decision-restarts",
        "--artifact-root",
        str(tmp_path),
    ]).opponent_decision_restarts is True


def test_missing_supervised_checkpoint_builds_seeded_ruleset_default_policy(
    tmp_path,
):
    missing = tmp_path / "missing_supervised.npz"
    first = _load_initial_network(
        0.001,
        missing,
        None,
        quiet=True,
        fresh_from_sl=True,
        device="cpu",
        ruleset=RULESET,
        initialization_seed=123,
        expected_training_algorithm=PPO_TRAINING_ALGORITHM,
    )
    second = _load_initial_network(
        0.001,
        missing,
        None,
        quiet=True,
        fresh_from_sl=True,
        device="cpu",
        ruleset=RULESET,
        initialization_seed=123,
        expected_training_algorithm=PPO_TRAINING_ALGORITHM,
    )
    different = _load_initial_network(
        0.001,
        missing,
        None,
        quiet=True,
        fresh_from_sl=True,
        device="cpu",
        ruleset=RULESET,
        initialization_seed=124,
        expected_training_algorithm=PPO_TRAINING_ALGORITHM,
    )
    assert first.hidden_sizes == default_hidden_sizes(RULESET)
    assert first.W1.shape[1] == DominoEncoder(RULESET).vector_size
    assert getattr(first, f"W{first.layer_count}").shape[0] == (
        DominoEncoder(RULESET).action_size
    )
    for name in first.weight_names:
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert any(
        not np.array_equal(getattr(first, name), getattr(different, name))
        for name in first.weight_names
    )
    assert first.rl_training_algorithm == PPO_TRAINING_ALGORITHM


def _network():
    encoder = DominoEncoder(RULESET)
    return PolicyNetwork(
        input_size=encoder.vector_size,
        hidden1_size=8,
        hidden2_size=4,
        output_size=encoder.action_size,
        random_seed=7,
        device="cpu",
    )


def _canonical_action(engine):
    actions = engine.valid_actions()
    return sorted(actions, key=repr)[0]


def _restart_from_capture(capture, *, opponent_kind="random"):
    return OpponentDecisionRestart(
        restart_index=0,
        source_iteration=1,
        source_game_index=17,
        snapshot_ordinal=capture.snapshot_ordinal,
        source_turn=capture.source_turn,
        original_learner_position=capture.original_learner_position,
        source_legal_tile_action_count=capture.source_legal_tile_action_count,
        opponent_kind=opponent_kind,
        opponent_id=None,
        bucket_name=opponent_kind,
        bank_slot=None,
        engine_state=capture.engine_state,
    )


def _result_fingerprint(results, index_key):
    rows = []
    for result in results:
        rows.append((
            result[index_key],
            result["winner"],
            result["learner_position"],
            tuple(
                (
                    np.asarray(sample.x).tobytes(),
                    sample.action_index,
                    np.asarray(sample.legal_mask).tobytes(),
                    sample.policy_reward,
                    sample.old_log_prob,
                )
                for sample in result["samples"]
            ),
        ))
    return rows


def test_restart_state_is_deeply_immutable_and_replays_exact_engine_suffix():
    random.seed(41)
    engine = DominoEngine(ruleset=RULESET)
    for _turn in range(4):
        engine.step(_canonical_action(engine))
    state = engine.export_restart_state()
    left = DominoEngine.from_restart_state(state)
    right = DominoEngine.from_restart_state(state)

    # Mutating the source after capture cannot mutate the immutable snapshot.
    engine.hands[0].append((99, 99))
    assert (99, 99) not in state.hands[0]

    while not left.game_over:
        left_action = _canonical_action(left)
        right_action = _canonical_action(right)
        assert left_action == right_action
        left.step(left_action)
        right.step(right_action)
        assert left.to_dict() == right.to_dict()


def test_capture_records_only_pre_action_genuine_opponent_tile_choices():
    network = _network()
    random.seed(0)
    np.random.seed(0)
    result = _collect_steps_vs_random(
        network,
        dict(DEFAULT_REWARD_SCHEMA),
        1.0,
        ruleset_name=RULESET,
        capture_opponent_decision_restarts=True,
    )
    (
        _samples, _events, _winner, learner_position, _terminal, captures
    ) = result
    assert captures
    assert [item.snapshot_ordinal for item in captures] == list(
        range(len(captures))
    )
    for capture in captures:
        restored = DominoEngine.from_restart_state(capture.engine_state)
        assert restored.turn == capture.source_turn
        assert restored.current_player == 1 - learner_position
        tile_actions = _tile_play_actions(restored.valid_actions())
        assert len(tile_actions) == capture.source_legal_tile_action_count
        assert len(tile_actions) >= 2


def test_restart_learner_immediately_acts_from_the_original_opponent_seat():
    network = _network()
    random.seed(0)
    np.random.seed(0)
    captured = _collect_steps_vs_random(
        network,
        dict(DEFAULT_REWARD_SCHEMA),
        1.0,
        ruleset_name=RULESET,
        capture_opponent_decision_restarts=True,
    )[5][0]
    restart = _restart_from_capture(captured)
    random.seed(123)
    np.random.seed(123)
    (
        samples, _events, _winner, learner_position, _terminal
    ) = collect_steps_from_restart(
        network,
        "random",
        None,
        restart,
        dict(DEFAULT_REWARD_SCHEMA),
        1.0,
        ruleset_name=RULESET,
    )
    assert learner_position == restart.restart_learner_position
    assert learner_position != restart.original_learner_position
    assert samples  # The first captured state is itself a trainable choice.


def test_parallel_restart_results_are_invariant_to_worker_count():
    safety = ParallelSafetyConfig(
        memory_reserve_mb=0,
        estimated_worker_mb=1,
        max_worker_rss_mb=1024,
    )

    def collect(workers):
        network = _network()
        schema = dict(DEFAULT_REWARD_SCHEMA)
        schema["reward_distance_mode"] = "decision-turn"
        runner = RLRolloutRunner(
            network,
            opponent_buckets=("random",),
            schema=schema,
            gamma_f=0.95,
            ruleset_name=RULESET,
            safety=safety,
        )
        try:
            runner.set_workers(workers)
            runner.sync_current(network)
            plan = build_match_plan(
                opponent_pool=runner.opponent_pool,
                performance_tracker=runner.performance_tracker,
                selected_buckets=("random",),
                uniform_rotation=UniformRotationState(("random",)),
                difficulty_weight=0.0,
                iteration=1,
                first_absolute_game=0,
                game_count=20,
                base_seed=99,
            )
            normal, _normal_info = runner.collect_games(
                plan,
                99,
                capture_opponent_decision_restarts=True,
            )
            restarts, _restart_info = runner.collect_restarts(normal, 99, 1)
            capture_identity = [
                (
                    result["game_index"],
                    tuple(
                        (
                            item.snapshot_ordinal,
                            item.source_turn,
                            replace(item.engine_state, game_id=0),
                        )
                        for item in result["captured_restarts"]
                    ),
                )
                for result in normal
            ]
            return (
                _result_fingerprint(normal, "game_index"),
                capture_identity,
                _result_fingerprint(restarts, "restart_index"),
            )
        finally:
            runner.close()

    assert collect(1) == collect(2)


def test_neural_restarts_reuse_the_source_opponent_identity_and_bank_slot():
    network = _network()
    runner = RLRolloutRunner(
        network,
        opponent_buckets=("recent",),
        schema=dict(DEFAULT_REWARD_SCHEMA),
        gamma_f=1.0,
        ruleset_name=RULESET,
        safety=ParallelSafetyConfig(
            memory_reserve_mb=0,
            estimated_worker_mb=1,
            max_worker_rss_mb=1024,
        ),
    )
    try:
        runner.set_workers(1)
        runner.sync_current(network)
        allocated_before = runner.bank.allocated_opponent_count
        plan = build_match_plan(
            opponent_pool=runner.opponent_pool,
            performance_tracker=runner.performance_tracker,
            selected_buckets=("recent",),
            uniform_rotation=UniformRotationState(("recent",)),
            difficulty_weight=0.0,
            iteration=1,
            first_absolute_game=0,
            game_count=20,
            base_seed=321,
        )
        normal, _info = runner.collect_games(
            plan,
            321,
            capture_opponent_decision_restarts=True,
        )
        source_by_game = {result["game_index"]: result for result in normal}
        restarts, _restart_info = runner.collect_restarts(normal, 321, 1)
        assert restarts
        for result in restarts:
            source = source_by_game[result["source_game_index"]]
            assert result["opponent_kind"] == "policy_snapshot"
            assert result["opponent_id"] == source["opponent_id"]
            assert result["bank_slot"] == source["bank_slot"]
        assert runner.bank.allocated_opponent_count == allocated_before
    finally:
        runner.close()


def _save_supervised(path):
    network = _network()
    np.savez(
        path,
        **{
            name: np.asarray(getattr(network, name))
            for name in network.weight_names
        },
    )


def _train_restart_run(
    root,
    *,
    total_games=40,
    stop_after=None,
    resume_weights=None,
    resume_state=None,
    callback=None,
    gpi=20,
    ppo_max_epochs=1,
    create_supervised=True,
):
    sl_path = Path(root) / "supervised.npz"
    if create_supervised and not sl_path.exists():
        _save_supervised(sl_path)
    return train(
        RLTrainingOptions(
            ruleset_name=RULESET,
            total_training_games=total_games,
            gpi=gpi,
            opponent_buckets=("random",),
            # Pinned: the default `lookup-table` baseline has no packaged
            # artifact for this ruleset, and this test is not about it.
            baseline=("batch-mean",),
            difficulty_weight=0.0,
            opponent_decision_restarts=True,
            seed=1234,
            ppo_max_epochs=ppo_max_epochs,
        ),
        RLResourceOptions(
            sl_weights_path=sl_path,
            rl_weights_path=Path(root) / "training.npz",
            device="cpu",
            workers=1,
            safety_config=ParallelSafetyConfig(
                memory_reserve_mb=0,
                estimated_worker_mb=1,
                max_worker_rss_mb=1024,
            ),
        ),
        RLExecutionOptions(
            quiet=True,
            checkpoint_interval=1,
            numbered_checkpoints=True,
            fresh_from_sl=resume_weights is None,
            stop_after_training_games=stop_after,
            resume_weights_path=resume_weights,
            resume_state_file=resume_state,
            metrics_callback=callback,
        ),
    )


def test_iteration_keeps_gpi_counters_separate_from_restart_augmentation(tmp_path):
    rows = []
    summary = _train_restart_run(
        tmp_path,
        total_games=20,
        callback=rows.append,
    )
    assert summary["completed_training_games"] == 20
    assert summary["total_restart_episodes"] > 0
    assert summary["total_restart_decision_samples"] > 0
    assert summary["total_decision_samples"] == (
        summary["total_normal_decision_samples"]
        + summary["total_restart_decision_samples"]
    )
    assert len(rows) == 1
    assert rows[0]["games"] == 20
    assert rows[0]["restart_continuation_episodes"] > 0
    assert rows[0]["decisions"] == (
        rows[0]["normal_decisions"] + rows[0]["restart_decisions"]
    )


def test_training_reports_random_initialization_when_supervised_is_missing(
    tmp_path,
):
    summary = _train_restart_run(
        tmp_path,
        total_games=20,
        create_supervised=False,
    )
    assert summary["initialization_source"] == "random"
    assert Path(summary["rl_weights_path"]).is_file()


def test_combined_normal_and_restart_buffer_receives_one_ppo_update(tmp_path):
    rows = []
    summary = _train_restart_run(
        tmp_path,
        total_games=100,
        gpi=100,
        ppo_max_epochs=2,
        callback=rows.append,
    )
    assert summary["policy_updates_completed"] == 1
    assert len(rows) == 1
    row = rows[0]
    assert row["normal_decisions"] > 0
    assert row["restart_decisions"] > 0
    assert row["decisions"] >= 256
    assert row["target_decisions_per_minibatch"] == 512
    assert row["min_decisions_per_minibatch"] == 256
    assert row["epochs_completed"] in (1, 2)
    assert row["optimizer_steps"] == (
        row["effective_minibatches"] * row["epochs_completed"]
    )


def test_restart_configuration_and_counters_resume_exactly(tmp_path):
    full_root = tmp_path / "full"
    split_root = tmp_path / "split"
    full_root.mkdir()
    split_root.mkdir()
    full = _train_restart_run(full_root)
    partial = _train_restart_run(split_root, stop_after=20)
    resumed = _train_restart_run(
        split_root,
        resume_weights=partial["rl_weights_path"],
        resume_state=resume_state_path(partial["rl_weights_path"]),
    )
    assert resumed["opponent_decision_restarts"] is True
    assert resumed["total_restart_episodes"] == full["total_restart_episodes"]
    assert resumed["total_restart_decision_samples"] == (
        full["total_restart_decision_samples"]
    )
    with np.load(full["rl_weights_path"], allow_pickle=False) as left:
        with np.load(resumed["rl_weights_path"], allow_pickle=False) as right:
            assert left.files == right.files
            for name in left.files:
                np.testing.assert_array_equal(left[name], right[name])
