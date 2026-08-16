"""Opponent-pool, difficulty, and exact matchmaking contracts."""

from __future__ import annotations

import numpy as np
import pytest

from agents.rl_nn import PolicyNetwork
from training.rl.config import (
    RLExecutionOptions,
    RLResourceOptions,
    RLTrainingOptions,
    resolve_training_options,
)
from training.rl.matchmaking import (
    OpponentPerformanceTracker,
    aggregate_match_results,
    build_match_plan,
    difficulty_from_win_rate,
)
from training.rl.pool import (
    HEURISTIC_OPPONENT_ID,
    K_RECENT,
    MEDIUM_TERM_CAPACITY,
    MEDIUM_TERM_INTERVAL_ITERATIONS,
    OpponentPool,
    RANDOM_OPPONENT_ID,
    SharedPolicyBank,
    canonicalize_bucket_names,
    unique_neural_capacity,
)


def _network(seed=1):
    return PolicyNetwork(
        input_size=8,
        hidden1_size=6,
        hidden2_size=4,
        output_size=5,
        random_seed=seed,
        device="cpu",
    )


def _pool(bucket_names=("heuristic", "recent")):
    network = _network()
    bank = SharedPolicyBank(network, unique_neural_capacity(bucket_names))
    return network, bank, OpponentPool(
        bank,
        selected_buckets=bucket_names,
        initial_network=network,
    )


def _tracker(pool):
    tracker = OpponentPerformanceTracker()
    tracker.ensure(record.opponent_id for record in pool.active_opponents())
    return tracker


def test_bucket_configuration_is_named_canonical_and_strict():
    assert canonicalize_bucket_names(
        "medium_term,recent,random,heuristic"
    ) == (
        "heuristic",
        "random",
        "recent",
        "medium_term",
    )
    assert canonicalize_bucket_names("recent,heuristic") == (
        "heuristic",
        "recent",
    )
    assert canonicalize_bucket_names(("recent",)) == ("recent",)
    with pytest.raises(ValueError, match="duplicate"):
        canonicalize_bucket_names("recent,recent")
    with pytest.raises(ValueError, match="Unknown"):
        canonicalize_bucket_names("recent,archive")
    with pytest.raises(ValueError, match="at least one"):
        canonicalize_bucket_names("")

    resolved = resolve_training_options(
        RLTrainingOptions(
            opponent_buckets=("recent", "heuristic"),
            difficulty_weight=1,
        ),
        RLResourceOptions(workers=1),
        RLExecutionOptions(),
    )
    assert resolved.training.opponent_buckets == ("heuristic", "recent")
    assert resolved.training.difficulty_weight == 1.0
    with pytest.raises(ValueError, match="between 0 and 1"):
        resolve_training_options(
            RLTrainingOptions(difficulty_weight=1.01),
            RLResourceOptions(workers=1),
            RLExecutionOptions(),
        )


@pytest.mark.parametrize(
    ("win_rate", "expected"),
    ((0.2, 1.0), (0.375, 1.0), (0.5, 0.5), (0.6, 0.1), (0.625, 0.0), (0.9, 0.0)),
)
def test_difficulty_calibration(win_rate, expected):
    assert difficulty_from_win_rate(win_rate) == pytest.approx(expected)


def test_performance_prior_smoothing_and_decay_return_toward_half():
    tracker = OpponentPerformanceTracker()
    tracker.ensure(("opponent",))
    assert tracker.estimated_win_rate("opponent") == 0.5
    tracker.update(("opponent",), {
        "opponent": {"wins": 1, "losses": 0},
    })
    after_one_game = tracker.estimated_win_rate("opponent")
    assert 0.5 < after_one_game < 0.55
    for _iteration in range(20):
        tracker.update(("opponent",), {})
    assert tracker.estimated_win_rate("opponent") < after_one_game
    assert tracker.estimated_win_rate("opponent") > 0.5


def test_performance_tracker_accepts_only_consistent_win_loss_outcomes():
    tracker = OpponentPerformanceTracker()
    tracker.ensure(("opponent",))
    tracker.update(("opponent",), {
        "opponent": {"games": 3, "wins": 2, "losses": 1},
    })
    performance = tracker.performance("opponent")
    assert performance.lifetime_wins == 2
    assert performance.lifetime_losses == 1
    assert set(tracker.export_state()["opponents"]["opponent"]) == {
        "decayed_wins",
        "decayed_losses",
        "lifetime_wins",
        "lifetime_losses",
    }

    with pytest.raises(ValueError, match="unsupported outcome fields"):
        tracker.update(("opponent",), {
            "opponent": {"wins": 1, "losses": 0, "draws": 0},
        })
    with pytest.raises(ValueError, match="wins plus losses"):
        tracker.update(("opponent",), {
            "opponent": {"games": 2, "wins": 1, "losses": 0},
        })


@pytest.mark.parametrize(
    ("alpha", "uniform_budget", "difficulty_budget"),
    ((0.0, 2000, 0), (0.5, 1000, 1000), (1.0, 0, 2000)),
)
def test_match_plan_has_exact_component_and_game_budgets(
    alpha,
    uniform_budget,
    difficulty_budget,
):
    _network_value, bank, pool = _pool()
    try:
        plan = build_match_plan(
            opponent_pool=pool,
            performance_tracker=_tracker(pool),
            selected_buckets=("heuristic", "recent"),
            difficulty_weight=alpha,
            iteration=1,
            first_absolute_game=4000,
            game_count=2000,
            base_seed=42,
        )
    finally:
        bank.close()
    assert plan.uniform_budget == uniform_budget
    assert plan.difficulty_budget == difficulty_budget
    assert len(plan.assignments) == 2000
    assert sum(value.game_count for value in plan.allocations) == 2000
    assert {value.game_index for value in plan.assignments} == set(range(4000, 6000))
    if alpha == 0.5:
        uniform_by_bucket = {
            bucket: sum(
                value.uniform_games
                for value in plan.allocations
                if value.bucket_name == bucket
            )
            for bucket in ("heuristic", "recent")
        }
        assert uniform_by_bucket == {"heuristic": 500, "recent": 500}


def test_match_plan_hash_is_repeatable_and_partial_budget_is_exact():
    _network_value, bank, pool = _pool()
    try:
        tracker = _tracker(pool)
        arguments = {
            "opponent_pool": pool,
            "performance_tracker": tracker,
            "selected_buckets": ("heuristic", "recent"),
            "difficulty_weight": 0.5,
            "iteration": 7,
            "first_absolute_game": 12000,
            "game_count": 37,
            "base_seed": 91,
        }
        first = build_match_plan(**arguments)
        second = build_match_plan(**arguments)
    finally:
        bank.close()
    assert first == second
    assert first.plan_sha256 == second.plan_sha256
    assert sum(value.game_count for value in first.allocations) == 37
    assert first.uniform_budget == 18
    assert first.difficulty_budget == 19


def test_match_results_have_only_wins_and_losses_and_reject_invalid_winner():
    _network_value, bank, pool = _pool(("random",))
    try:
        plan = build_match_plan(
            opponent_pool=pool,
            performance_tracker=_tracker(pool),
            selected_buckets=("random",),
            difficulty_weight=0.5,
            iteration=1,
            first_absolute_game=0,
            game_count=2,
            base_seed=42,
        )
        results = []
        for index, assignment in enumerate(plan.assignments):
            learner_position = index
            results.append({
                "game_index": assignment.game_index,
                "bucket_name": assignment.bucket_name,
                "opponent_id": assignment.opponent_id,
                "opponent_kind": assignment.opponent_kind,
                "bank_slot": assignment.bank_slot,
                "winner": 0,
                "learner_position": learner_position,
            })
        by_opponent, by_bucket = aggregate_match_results(plan, results)
        assert by_opponent[RANDOM_OPPONENT_ID] == {
            "games": 2,
            "wins": 1,
            "losses": 1,
        }
        assert by_bucket["random"] == {
            "games": 2,
            "wins": 1,
            "losses": 1,
        }

        results[0]["winner"] = None
        with pytest.raises(ValueError, match="game-result draws do not exist"):
            aggregate_match_results(plan, results)
    finally:
        bank.close()


def test_recent_fifo_is_logical_and_heuristic_consumes_no_slot():
    network, bank, pool = _pool()
    try:
        assert pool.bank_slot(HEURISTIC_OPPONENT_ID) is None
        initial_id = pool.bucket_members("recent")[0]
        for iteration in range(1, K_RECENT + 5):
            network.W1 += np.float32(0.0001)
            pool.consider_updated_policy(
                network,
                iteration=iteration,
                completed_games=iteration * 10,
                has_samples=True,
            )
        recent = pool.bucket_members("recent")
        assert len(recent) == K_RECENT
        assert initial_id not in recent
        assert pool.unique_neural_opponent_count == K_RECENT
        assert bank.allocated_opponent_count == K_RECENT
        assert pool.size == K_RECENT + 1
    finally:
        bank.close()


def test_medium_term_admission_uses_absolute_successful_iteration_cadence():
    network, bank, pool = _pool(("medium_term",))
    try:
        initial_id = pool.bucket_members("medium_term")[0]
        assert pool.opponent(initial_id).introduced_iteration == 0
        for iteration in range(1, MEDIUM_TERM_INTERVAL_ITERATIONS):
            assert pool.consider_updated_policy(
                network,
                iteration=iteration,
                completed_games=iteration * 2000,
                has_samples=True,
            ) is None
        assert pool.bucket_members("medium_term") == (initial_id,)

        assert pool.consider_updated_policy(
            network,
            iteration=10,
            completed_games=20_000,
            has_samples=False,
        ) is None
        assert pool.bucket_members("medium_term") == (initial_id,)
        admitted = pool.consider_updated_policy(
            network,
            iteration=20,
            completed_games=40_000,
            has_samples=True,
        )
        assert pool.bucket_members("medium_term") == (
            initial_id,
            admitted.opponent_id,
        )
        assert pool.last_completed_rl_iteration == 20
    finally:
        bank.close()


def test_medium_term_capacity_fifo_and_four_million_game_horizon():
    network, bank, pool = _pool(("medium_term",))
    try:
        initial_id = pool.bucket_members("medium_term")[0]
        for iteration in range(
            MEDIUM_TERM_INTERVAL_ITERATIONS,
            (MEDIUM_TERM_CAPACITY + 1) * MEDIUM_TERM_INTERVAL_ITERATIONS,
            MEDIUM_TERM_INTERVAL_ITERATIONS,
        ):
            pool.consider_updated_policy(
                network,
                iteration=iteration,
                completed_games=iteration * 2000,
                has_samples=True,
            )
        members = pool.bucket_members("medium_term")
        status = pool.observability(games_per_iteration=2000)["buckets"][
            "medium_term"
        ]
        assert len(members) == MEDIUM_TERM_CAPACITY
        assert initial_id not in members
        assert pool.opponent(members[0]).introduced_iteration == 10
        assert pool.opponent(members[-1]).introduced_iteration == 2000
        assert status["nominal_historical_span_games"] == 4_000_000
        assert status["exact_historical_span_games"] == 3_980_000
        assert status["fifo_evictions"] == 1
    finally:
        bank.close()


def test_recent_and_medium_term_share_identity_slot_and_tracker():
    network, bank, pool = _pool(("recent", "medium_term"))
    try:
        initial_id = pool.bucket_members("recent")[0]
        assert pool.bucket_members("medium_term") == (initial_id,)
        assert pool.unique_neural_opponent_count == 1
        admitted = pool.consider_updated_policy(
            network,
            iteration=10,
            completed_games=20_000,
            has_samples=True,
        )
        assert pool.bucket_members("recent")[-1] == admitted.opponent_id
        assert pool.bucket_members("medium_term")[-1] == admitted.opponent_id
        assert pool.bank_slot(admitted.opponent_id) is not None
        assert pool.unique_neural_opponent_count == 2
        assert pool.observability()["recent_medium_term_overlap_count"] == 2

        tracker = _tracker(pool)
        tracker.update(
            (record.opponent_id for record in pool.active_opponents()),
            {admitted.opponent_id: {"games": 2, "wins": 1, "losses": 1}},
        )
        assert tracker.performance(admitted.opponent_id).lifetime_wins == 1
        assert tracker.performance(admitted.opponent_id).lifetime_losses == 1
    finally:
        bank.close()


def test_recent_eviction_does_not_release_a_medium_term_member():
    network, bank, pool = _pool(("recent", "medium_term"))
    try:
        initial_id = pool.bucket_members("recent")[0]
        initial_slot = pool.bank_slot(initial_id)
        for iteration in range(1, K_RECENT + 1):
            pool.consider_updated_policy(
                network,
                iteration=iteration,
                completed_games=iteration * 2000,
                has_samples=True,
            )
        assert initial_id not in pool.bucket_members("recent")
        assert initial_id in pool.bucket_members("medium_term")
        assert pool.bank_slot(initial_id) == initial_slot
        assert bank.allocated_opponent_count == 201
    finally:
        bank.close()


def test_medium_term_match_plan_is_exact_and_preserves_source_buckets():
    _network_value, bank, pool = _pool(
        ("heuristic", "recent", "medium_term")
    )
    try:
        plan = build_match_plan(
            opponent_pool=pool,
            performance_tracker=_tracker(pool),
            selected_buckets=("heuristic", "recent", "medium_term"),
            difficulty_weight=0.5,
            iteration=1,
            first_absolute_game=0,
            game_count=2000,
            base_seed=42,
        )
        repeated = build_match_plan(
            opponent_pool=pool,
            performance_tracker=_tracker(pool),
            selected_buckets=("heuristic", "recent", "medium_term"),
            difficulty_weight=0.5,
            iteration=1,
            first_absolute_game=0,
            game_count=2000,
            base_seed=42,
        )
        assert repeated == plan
        assert len(plan.assignments) == 2000
        assert sum(item.game_count for item in plan.allocations) == 2000
        assert {item.bucket_name for item in plan.allocations} == {
            "heuristic",
            "recent",
            "medium_term",
        }
        neural_id = pool.bucket_members("recent")[0]
        sources = {
            item.bucket_name
            for item in plan.allocations
            if item.opponent_id == neural_id and item.game_count
        }
        assert sources == {"recent", "medium_term"}
    finally:
        bank.close()


def test_random_bucket_is_fixed_and_consumes_no_policy_bank_slot():
    _network_value, bank, pool = _pool(("random",))
    try:
        assert pool.bucket_members("random") == (RANDOM_OPPONENT_ID,)
        assert pool.bank_slot(RANDOM_OPPONENT_ID) is None
        assert pool.unique_neural_opponent_count == 0
        assert bank.allocated_opponent_count == 0
        assert pool.size == 1

        plan = build_match_plan(
            opponent_pool=pool,
            performance_tracker=_tracker(pool),
            selected_buckets=("random",),
            difficulty_weight=0.5,
            iteration=1,
            first_absolute_game=0,
            game_count=7,
            base_seed=42,
        )
        assert len(plan.assignments) == 7
        assert {item.opponent_id for item in plan.assignments} == {
            RANDOM_OPPONENT_ID
        }
        assert {item.opponent_kind for item in plan.assignments} == {"random"}
        assert {item.bank_slot for item in plan.assignments} == {None}
    finally:
        bank.close()


def test_pool_export_restore_preserves_ids_order_weights_and_performance():
    network, bank, pool = _pool()
    restored_bank = None
    try:
        pool.consider_updated_policy(
            network,
            iteration=1,
            completed_games=20,
            has_samples=True,
        )
        tracker = _tracker(pool)
        tracker.update(
            (record.opponent_id for record in pool.active_opponents()),
            {HEURISTIC_OPPONENT_ID: {"wins": 3, "losses": 2}},
        )
        state = pool.export_state()
        weights = pool.export_weights()
        tracker_state = tracker.export_state()

        restored_network = _network(seed=3)
        restored_bank = SharedPolicyBank(
            restored_network,
            unique_neural_capacity(("heuristic", "recent")),
        )
        restored = OpponentPool(
            restored_bank,
            selected_buckets=("heuristic", "recent"),
            initial_network=restored_network,
        )
        restored.restore_state(state, weights)
        restored_tracker = OpponentPerformanceTracker()
        restored_tracker.restore_state(
            tracker_state,
            (record.opponent_id for record in restored.active_opponents()),
        )
        assert restored.export_state() == state
        assert restored_tracker.export_state() == tracker_state
        for opponent_id, original in weights.items():
            copied = restored.export_weights()[opponent_id]
            for name in original:
                np.testing.assert_array_equal(copied[name], original[name])
    finally:
        bank.close()
        if restored_bank is not None:
            restored_bank.close()


def test_medium_term_resume_restores_order_overlap_and_next_absolute_cadence():
    network, bank, pool = _pool(("recent", "medium_term"))
    restored_bank = None
    try:
        for iteration in range(1, 10):
            pool.consider_updated_policy(
                network,
                iteration=iteration,
                completed_games=iteration * 2000,
                has_samples=True,
            )
        state = pool.export_state()
        weights = pool.export_weights()
        # Each full bank owns 401 shared-memory descriptors. Close the source
        # before constructing the independent restored bank so the test stays
        # below ordinary per-process file-descriptor limits.
        bank.close()
        bank = None
        restored_network = _network(seed=4)
        restored_bank = SharedPolicyBank(
            restored_network,
            unique_neural_capacity(("recent", "medium_term")),
        )
        restored = OpponentPool(
            restored_bank,
            selected_buckets=("recent", "medium_term"),
            initial_network=restored_network,
        )
        restored.restore_state(state, weights)
        before = restored.bucket_members("medium_term")
        milestone = restored.consider_updated_policy(
            restored_network,
            iteration=10,
            completed_games=20_000,
            has_samples=True,
        )
        assert restored.bucket_members("medium_term") == (
            *before,
            milestone.opponent_id,
        )
        assert restored.bucket_members("recent")[-1] == milestone.opponent_id
        assert restored.observability()[
            "recent_medium_term_overlap_count"
        ] == 2
    finally:
        if bank is not None:
            bank.close()
        if restored_bank is not None:
            restored_bank.close()
