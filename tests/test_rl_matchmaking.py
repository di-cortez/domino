"""Opponent-pool, difficulty, and exact matchmaking contracts."""

from __future__ import annotations

from dataclasses import dataclass

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
from training.rl.checkpoint_archive import ArchiveRecord
from training.rl.pool import (
    HEURISTIC_OPPONENT_ID,
    HISTORICAL_MINIMUM_AGE_ITERATIONS,
    HISTORICAL_UNIFORM_CAPACITY,
    K_RECENT,
    MEDIUM_TERM_BAND_WIDTH_ITERATIONS,
    MEDIUM_TERM_CAPACITY,
    MEDIUM_TERM_INTERVAL_ITERATIONS,
    OpponentPool,
    POOL_POLICY_VERSION,
    POOL_SCHEMA_VERSION,
    RANDOM_OPPONENT_ID,
    RECENT_BAND_WIDTH_ITERATIONS,
    SharedPolicyBank,
    canonicalize_bucket_names,
    historical_uniform_cutoff_iteration,
    historical_uniform_selection_diagnostics,
    medium_term_cutoff_iteration,
    pool_policy_manifest,
    select_historical_uniform_records,
    select_medium_term_records,
    sorted_archive_records,
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
        "historical_uniform,medium_term,recent,random,heuristic"
    ) == (
        "heuristic",
        "random",
        "recent",
        "medium_term",
        "historical_uniform",
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


def test_neural_bucket_bands_are_disjoint_by_construction():
    assert RECENT_BAND_WIDTH_ITERATIONS == K_RECENT
    assert MEDIUM_TERM_BAND_WIDTH_ITERATIONS == (
        MEDIUM_TERM_CAPACITY * MEDIUM_TERM_INTERVAL_ITERATIONS
    )
    assert HISTORICAL_MINIMUM_AGE_ITERATIONS == (
        RECENT_BAND_WIDTH_ITERATIONS + MEDIUM_TERM_BAND_WIDTH_ITERATIONS
    )
    assert HISTORICAL_UNIFORM_CAPACITY == 200
    assert unique_neural_capacity(
        ("heuristic", "recent", "medium_term", "historical_uniform")
    ) == 600
    assert unique_neural_capacity(("heuristic", "random")) == 0


def test_pool_policy_manifest_publishes_versioned_band_boundaries():
    manifest = pool_policy_manifest(("recent", "medium_term", "historical_uniform"))
    assert manifest["schema_version"] == POOL_SCHEMA_VERSION
    assert manifest["policy_version"] == POOL_POLICY_VERSION
    assert manifest["bucket_registry_order"] == [
        "heuristic",
        "random",
        "recent",
        "medium_term",
        "historical_uniform",
    ]
    assert manifest["band_policy"] == {
        "recent_band_width_iterations": 200,
        "medium_term_band_width_iterations": 2000,
        "historical_minimum_age_iterations": 2200,
        "band_eligibility_coordinate": "completed_iteration",
        "historical_spacing_coordinate": "completed_rl_games",
        "historical_selection_tie_breaking": (
            "absolute_target_error_then_older_iteration_then_checkpoint_id"
        ),
    }
    historical = manifest["bucket_definitions"]["historical_uniform"]
    assert historical["capacity"] == HISTORICAL_UNIFORM_CAPACITY
    assert historical["neural"] is True
    assert historical["admission_interval_iterations"] == (
        MEDIUM_TERM_INTERVAL_ITERATIONS
    )
    assert manifest["bucket_definitions"]["medium_term"]["retention_rule"] == (
        "delayed_fifo_archive_window"
    )


@dataclass(frozen=True)
class _ArchiveMetadata:
    """Minimal band-selection input, matching the archive record contract."""

    checkpoint_id: str
    opponent_id: str
    completed_iteration: int
    completed_rl_games: int


def _archive_history(milestone_count, *, games_per_iteration=2000):
    """Build a dense archive at the fixed ten-iteration cadence."""
    return tuple(
        _ArchiveMetadata(
            checkpoint_id=f"checkpoint:{index:010d}",
            opponent_id=f"snapshot:{index:010d}",
            completed_iteration=index * MEDIUM_TERM_INTERVAL_ITERATIONS,
            completed_rl_games=(
                index * MEDIUM_TERM_INTERVAL_ITERATIONS * games_per_iteration
            ),
        )
        for index in range(milestone_count)
    )


def _iterations(records):
    return tuple(record.completed_iteration for record in records)


class _StubArchive:
    """Read side of the checkpoint archive, without files or hashing."""

    def __init__(self, network):
        self._network = network
        self.records = []

    def write(self, record, *, iteration, completed_games):
        if record is None or iteration % MEDIUM_TERM_INTERVAL_ITERATIONS:
            return
        self.records.append(_ArchiveMetadata(
            checkpoint_id=record.checkpoint_id,
            opponent_id=record.opponent_id,
            completed_iteration=int(iteration),
            completed_rl_games=int(completed_games),
        ))

    def load_weights(self, checkpoint_id):
        assert any(
            record.checkpoint_id == checkpoint_id for record in self.records
        )
        return {
            name: np.asarray(getattr(self._network, name)).copy()
            for name in self._network.weight_names
        }


def _run_iterations(pool, network, last_iteration, *, games_per_iteration=2000):
    """Advance a pool exactly as one archive-cadence training loop would."""
    archive = _StubArchive(network)
    archive.write(pool.initial_snapshot_record, iteration=0, completed_games=0)
    for iteration in range(1, last_iteration + 1):
        completed_games = iteration * games_per_iteration
        archive.write(
            pool.consider_updated_policy(
                network,
                iteration=iteration,
                completed_games=completed_games,
                has_samples=True,
            ),
            iteration=iteration,
            completed_games=completed_games,
        )
        if iteration % MEDIUM_TERM_INTERVAL_ITERATIONS == 0:
            pool.reconcile_archive_backed_buckets(
                archive.records,
                completed_iteration=iteration,
                load_weights=archive.load_weights,
            )
    return archive


def _member_iterations(pool, bucket_name):
    return tuple(
        pool.opponent(opponent_id).introduced_iteration
        for opponent_id in pool.bucket_members(bucket_name)
    )


def test_band_cutoffs_delay_admission_by_whole_band_widths():
    assert medium_term_cutoff_iteration(2200) == 2000
    assert historical_uniform_cutoff_iteration(2200) == 0
    assert medium_term_cutoff_iteration(199) == -1
    assert historical_uniform_cutoff_iteration(2190) == -10


def test_sorted_archive_records_rejects_duplicate_checkpoint_identities():
    history = _archive_history(3)
    with pytest.raises(ValueError, match="duplicate checkpoint"):
        sorted_archive_records((*history, history[0]))


def test_medium_term_band_begins_only_after_the_recent_region():
    history = _archive_history(300)
    assert select_medium_term_records(history, completed_iteration=199) == ()
    assert _iterations(
        select_medium_term_records(history, completed_iteration=200)
    ) == (0,)
    assert _iterations(
        select_medium_term_records(history, completed_iteration=209)
    ) == (0,)
    assert _iterations(
        select_medium_term_records(history, completed_iteration=210)
    ) == (0, 10)


def test_medium_term_band_is_full_and_delayed_at_the_expected_boundary():
    history = _archive_history(300)
    full = select_medium_term_records(history, completed_iteration=2190)
    assert len(full) == MEDIUM_TERM_CAPACITY
    assert _iterations(full)[:2] == (0, 10)
    assert _iterations(full)[-1] == 1990
    shifted = select_medium_term_records(history, completed_iteration=2200)
    assert len(shifted) == MEDIUM_TERM_CAPACITY
    assert _iterations(shifted)[0] == 10
    assert _iterations(shifted)[-1] == 2000


def test_medium_term_band_searches_backward_past_recent_identities():
    history = _archive_history(300)
    excluded = {history[200].opponent_id}
    selected = select_medium_term_records(
        history,
        completed_iteration=2200,
        excluded_opponent_ids=excluded,
    )
    assert len(selected) == MEDIUM_TERM_CAPACITY
    assert history[200].opponent_id not in {
        record.opponent_id for record in selected
    }
    assert _iterations(selected)[0] == 0
    assert _iterations(selected)[-1] == 1990


def test_historical_uniform_band_begins_only_after_the_full_delay():
    history = _archive_history(300)
    assert select_historical_uniform_records(
        history,
        completed_iteration=2190,
    ) == ()
    first = select_historical_uniform_records(history, completed_iteration=2200)
    assert _iterations(first) == (0,)


def test_historical_uniform_includes_every_eligible_record_until_capacity():
    history = _archive_history(400)
    eligible_iteration = (
        HISTORICAL_MINIMUM_AGE_ITERATIONS
        + HISTORICAL_UNIFORM_CAPACITY * MEDIUM_TERM_INTERVAL_ITERATIONS
        - MEDIUM_TERM_INTERVAL_ITERATIONS
    )
    selected = select_historical_uniform_records(
        history,
        completed_iteration=eligible_iteration,
    )
    assert len(selected) == HISTORICAL_UNIFORM_CAPACITY
    assert _iterations(selected)[0] == 0
    assert _iterations(selected)[-1] == 1990


def test_historical_uniform_selects_a_dense_regular_grid():
    history = _archive_history(2 * HISTORICAL_UNIFORM_CAPACITY - 1)
    selected = select_historical_uniform_records(
        history,
        completed_iteration=(
            history[-1].completed_iteration + HISTORICAL_MINIMUM_AGE_ITERATIONS
        ),
    )
    assert len(selected) == HISTORICAL_UNIFORM_CAPACITY
    assert selected[0] is history[0]
    assert selected[-1] is history[-1]
    assert _iterations(selected) == tuple(
        index * 2 * MEDIUM_TERM_INTERVAL_ITERATIONS
        for index in range(HISTORICAL_UNIFORM_CAPACITY)
    )


def test_historical_uniform_spaces_targets_by_games_not_record_rank():
    dense = tuple(
        _ArchiveMetadata(
            checkpoint_id=f"checkpoint:{index:010d}",
            opponent_id=f"snapshot:{index:010d}",
            completed_iteration=index * MEDIUM_TERM_INTERVAL_ITERATIONS,
            completed_rl_games=index,
        )
        for index in range(HISTORICAL_UNIFORM_CAPACITY)
    )
    sparse = tuple(
        _ArchiveMetadata(
            checkpoint_id=f"checkpoint:{index:010d}",
            opponent_id=f"snapshot:{index:010d}",
            completed_iteration=index * MEDIUM_TERM_INTERVAL_ITERATIONS,
            completed_rl_games=(
                HISTORICAL_UNIFORM_CAPACITY
                + (index - HISTORICAL_UNIFORM_CAPACITY) * 1000
            ),
        )
        for index in range(
            HISTORICAL_UNIFORM_CAPACITY,
            2 * HISTORICAL_UNIFORM_CAPACITY,
        )
    )
    history = (*dense, *sparse)
    selected = select_historical_uniform_records(
        history,
        completed_iteration=(
            history[-1].completed_iteration + HISTORICAL_MINIMUM_AGE_ITERATIONS
        ),
    )
    assert len(selected) == HISTORICAL_UNIFORM_CAPACITY
    # A rank-based grid would have taken every second record. Uniform game
    # coordinates instead jump two hundred ranks past the compressed era,
    # keeping only its oldest member.
    assert selected[0] is dense[0]
    assert selected[1] is sparse[1]
    assert sum(record in dense for record in selected) == 1


def test_historical_uniform_resolves_half_distance_ties_toward_the_older_record():
    step = 1000
    games = [0, step, 3 * step]
    games.extend(
        2 * step * (index - 1)
        for index in range(3, HISTORICAL_UNIFORM_CAPACITY + 1)
    )
    history = tuple(
        _ArchiveMetadata(
            checkpoint_id=f"checkpoint:{index:010d}",
            opponent_id=f"snapshot:{index:010d}",
            completed_iteration=index * MEDIUM_TERM_INTERVAL_ITERATIONS,
            completed_rl_games=value,
        )
        for index, value in enumerate(games)
    )
    selected = select_historical_uniform_records(
        history,
        completed_iteration=(
            history[-1].completed_iteration + HISTORICAL_MINIMUM_AGE_ITERATIONS
        ),
    )
    # The second target falls exactly between the records at 1000 and 3000.
    assert selected[1].completed_rl_games == step


def test_historical_uniform_excludes_recent_and_medium_identities():
    history = _archive_history(2 * HISTORICAL_UNIFORM_CAPACITY - 1)
    completed_iteration = (
        history[-1].completed_iteration + HISTORICAL_MINIMUM_AGE_ITERATIONS
    )
    excluded = {history[2].opponent_id, history[4].opponent_id}
    selected = select_historical_uniform_records(
        history,
        completed_iteration=completed_iteration,
        excluded_opponent_ids=excluded,
    )
    assert len(selected) == HISTORICAL_UNIFORM_CAPACITY
    assert not excluded & {record.opponent_id for record in selected}
    iterations = _iterations(selected)
    assert len(set(iterations)) == len(iterations)
    assert list(iterations) == sorted(iterations)


def test_band_selection_accepts_real_archive_records():
    records = tuple(
        ArchiveRecord(
            checkpoint_id=f"checkpoint:{index:010d}",
            opponent_id=f"snapshot:{index:010d}",
            completed_iteration=index * MEDIUM_TERM_INTERVAL_ITERATIONS,
            completed_rl_games=index * MEDIUM_TERM_INTERVAL_ITERATIONS * 2000,
            filename=f"checkpoint_iter{index:06d}.npz",
            file_size=1024,
            sha256="0" * 64,
            created_at="2026-08-17T00:00:00+00:00",
            weight_names=("w1",),
            weight_shapes=((2, 2),),
        )
        for index in range(3)
    )
    assert _iterations(
        select_medium_term_records(records, completed_iteration=210)
    ) == (0, 10)


def test_historical_uniform_diagnostics_expose_spacing_quality():
    history = _archive_history(2 * HISTORICAL_UNIFORM_CAPACITY - 1)
    selected = select_historical_uniform_records(
        history,
        completed_iteration=(
            history[-1].completed_iteration + HISTORICAL_MINIMUM_AGE_ITERATIONS
        ),
    )
    diagnostics = historical_uniform_selection_diagnostics(
        selected,
        eligible_records=history,
    )
    ideal_gap = 2 * MEDIUM_TERM_INTERVAL_ITERATIONS * 2000
    assert diagnostics["selected_count"] == HISTORICAL_UNIFORM_CAPACITY
    assert diagnostics["eligible_record_count"] == len(history)
    assert diagnostics["ideal_gap_games"] == pytest.approx(ideal_gap)
    assert diagnostics["minimum_gap_games"] == ideal_gap
    assert diagnostics["maximum_gap_games"] == ideal_gap
    assert diagnostics["maximum_absolute_target_error_games"] == pytest.approx(0.0)
    assert diagnostics["archive_thinned_in_region"] is False
    assert historical_uniform_selection_diagnostics(
        (),
        eligible_records=(),
    )["ideal_gap_games"] is None


def test_historical_diagnostics_report_a_thinned_archive_region():
    dense = _archive_history(2 * HISTORICAL_UNIFORM_CAPACITY - 1)
    # Drop one interior milestone, exactly as archive thinning would.
    thinned = (*dense[:50], *dense[51:])
    selected = select_historical_uniform_records(
        thinned,
        completed_iteration=(
            thinned[-1].completed_iteration + HISTORICAL_MINIMUM_AGE_ITERATIONS
        ),
    )
    diagnostics = historical_uniform_selection_diagnostics(
        selected,
        eligible_records=thinned,
    )
    assert diagnostics["archive_thinned_in_region"] is True
    assert diagnostics["eligible_record_count"] == len(thinned)
    assert diagnostics["maximum_absolute_target_error_games"] > 0


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


def test_empty_configured_bucket_receives_zero_games_and_keeps_its_row():
    buckets = ("heuristic", "recent", "historical_uniform")
    _network_value, bank, pool = _pool(buckets)
    try:
        assert pool.bucket_members("historical_uniform") == ()
        assert pool.available_bucket_names() == ("heuristic", "recent")
        plan = build_match_plan(
            opponent_pool=pool,
            performance_tracker=_tracker(pool),
            selected_buckets=buckets,
            difficulty_weight=0.5,
            iteration=1,
            first_absolute_game=0,
            game_count=101,
            base_seed=42,
        )
        assert plan.configured_buckets == buckets
        assert plan.available_buckets == ("heuristic", "recent")
        assert sum(value.game_count for value in plan.allocations) == 101
        assert "historical_uniform" not in {
            value.bucket_name for value in plan.allocations
        }
        results = [
            {
                "game_index": assignment.game_index,
                "bucket_name": assignment.bucket_name,
                "opponent_id": assignment.opponent_id,
                "opponent_kind": assignment.opponent_kind,
                "bank_slot": assignment.bank_slot,
                "winner": 0,
                "learner_position": 0,
            }
            for assignment in plan.assignments
        ]
        _by_opponent, by_bucket = aggregate_match_results(plan, results)
        assert list(by_bucket) == list(buckets)
        assert by_bucket["historical_uniform"] == {
            "games": 0,
            "wins": 0,
            "losses": 0,
        }
    finally:
        bank.close()


@pytest.mark.parametrize("difficulty_weight", (0.0, 0.5, 1.0))
def test_warm_up_redistributes_the_complete_budget_over_available_buckets(
    difficulty_weight,
):
    buckets = ("heuristic", "recent", "medium_term", "historical_uniform")
    _network_value, bank, pool = _pool(buckets)
    try:
        # ``medium_term`` still receives the initial snapshot today; the
        # delayed band replaces that admission in a later change.
        pool.buckets_by_name["medium_term"].member_ids.clear()
        assert pool.available_bucket_names() == ("heuristic", "recent")
        plan = build_match_plan(
            opponent_pool=pool,
            performance_tracker=_tracker(pool),
            selected_buckets=buckets,
            difficulty_weight=difficulty_weight,
            iteration=3,
            first_absolute_game=0,
            game_count=2000,
            base_seed=7,
        )
        assert plan.uniform_budget + plan.difficulty_budget == 2000
        assert plan.difficulty_budget == round(difficulty_weight * 2000)
        assert sum(value.game_count for value in plan.allocations) == 2000
        assert sum(value.uniform_games for value in plan.allocations) == (
            plan.uniform_budget
        )
        assert sum(value.difficulty_games for value in plan.allocations) == (
            plan.difficulty_budget
        )
        assert len(plan.assignments) == 2000
    finally:
        bank.close()


def test_plan_hash_distinguishes_identical_allocations_under_new_availability():
    arguments = {
        "difficulty_weight": 0.5,
        "iteration": 1,
        "first_absolute_game": 0,
        "game_count": 64,
        "base_seed": 11,
    }
    _network_value, narrow_bank, narrow = _pool(("heuristic", "recent"))
    try:
        narrow_plan = build_match_plan(
            opponent_pool=narrow,
            performance_tracker=_tracker(narrow),
            selected_buckets=("heuristic", "recent"),
            **arguments,
        )
    finally:
        narrow_bank.close()
    buckets = ("heuristic", "recent", "historical_uniform")
    _network_value, wide_bank, wide = _pool(buckets)
    try:
        wide_plan = build_match_plan(
            opponent_pool=wide,
            performance_tracker=_tracker(wide),
            selected_buckets=buckets,
            **arguments,
        )
    finally:
        wide_bank.close()
    assert wide_plan.available_buckets == narrow_plan.available_buckets
    assert [
        (value.bucket_name, value.opponent_id, value.game_count)
        for value in wide_plan.allocations
    ] == [
        (value.bucket_name, value.opponent_id, value.game_count)
        for value in narrow_plan.allocations
    ]
    assert wide_plan.plan_sha256 != narrow_plan.plan_sha256


def test_matchmaking_rejects_a_configuration_with_no_available_bucket():
    _network_value, bank, pool = _pool(("historical_uniform",))
    try:
        with pytest.raises(ValueError, match="Every configured opponent bucket"):
            build_match_plan(
                opponent_pool=pool,
                performance_tracker=_tracker(pool),
                selected_buckets=("historical_uniform",),
                difficulty_weight=0.5,
                iteration=1,
                first_absolute_game=0,
                game_count=8,
                base_seed=42,
            )
    finally:
        bank.close()


def test_configuration_requires_a_bucket_available_from_the_first_iteration():
    with pytest.raises(ValueError, match="at least one bucket that is"):
        resolve_training_options(
            RLTrainingOptions(
                opponent_buckets=("medium_term", "historical_uniform"),
            ),
            RLResourceOptions(workers=1),
            RLExecutionOptions(),
        )
    resolved = resolve_training_options(
        RLTrainingOptions(
            opponent_buckets=("recent", "medium_term", "historical_uniform"),
        ),
        RLResourceOptions(workers=1),
        RLExecutionOptions(),
    )
    assert resolved.training.opponent_buckets == (
        "recent",
        "medium_term",
        "historical_uniform",
    )


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


def test_a_fresh_milestone_never_enters_a_delayed_band_directly():
    network, bank, pool = _pool(("recent", "medium_term"))
    try:
        assert pool.bucket_members("medium_term") == ()
        _run_iterations(pool, network, MEDIUM_TERM_INTERVAL_ITERATIONS * 3)
        assert pool.bucket_members("medium_term") == ()
        assert pool.last_completed_rl_iteration == 30
        assert len(pool.bucket_members("recent")) == 31
    finally:
        bank.close()


def test_medium_term_admits_the_baseline_exactly_when_it_leaves_recent():
    network, bank, pool = _pool(("recent", "medium_term"))
    try:
        archive = _run_iterations(pool, network, K_RECENT - 1)
        assert pool.bucket_members("medium_term") == ()
        assert _member_iterations(pool, "recent")[0] == 0

        baseline = pool.initial_snapshot_record
        for iteration in (K_RECENT, K_RECENT + MEDIUM_TERM_INTERVAL_ITERATIONS):
            for step in range(pool.last_completed_rl_iteration + 1, iteration + 1):
                archive.write(
                    pool.consider_updated_policy(
                        network,
                        iteration=step,
                        completed_games=step * 2000,
                        has_samples=True,
                    ),
                    iteration=step,
                    completed_games=step * 2000,
                )
            pool.reconcile_archive_backed_buckets(
                archive.records,
                completed_iteration=iteration,
                load_weights=archive.load_weights,
            )
        assert _member_iterations(pool, "recent") == tuple(
            range(11, K_RECENT + MEDIUM_TERM_INTERVAL_ITERATIONS + 1)
        )
        assert _member_iterations(pool, "medium_term") == (0, 10)
        # The rehydrated baseline keeps its original archived identity.
        rehydrated = pool.opponent(pool.bucket_members("medium_term")[0])
        assert rehydrated.opponent_id == baseline.opponent_id
        assert rehydrated.checkpoint_id == baseline.checkpoint_id
        assert rehydrated.introduced_iteration == 0
        assert rehydrated.introduced_at_rl_games == 0
        assert pool.bank_slot(rehydrated.opponent_id) is not None
        assert pool.observability()["total_bucket_overlap_count"] == 0
    finally:
        bank.close()


def test_medium_term_is_a_full_delayed_band_at_the_expected_boundary():
    network, bank, pool = _pool(("recent", "medium_term"))
    try:
        _run_iterations(
            pool,
            network,
            MEDIUM_TERM_BAND_WIDTH_ITERATIONS + K_RECENT - 10,
        )
        members = pool.bucket_members("medium_term")
        status = pool.observability(games_per_iteration=2000)["buckets"][
            "medium_term"
        ]
        assert len(members) == MEDIUM_TERM_CAPACITY
        assert _member_iterations(pool, "medium_term")[0] == 0
        assert _member_iterations(pool, "medium_term")[-1] == 1990
        assert _member_iterations(pool, "recent")[-1] == 2190
        assert status["nominal_historical_span_games"] == 4_000_000
        assert status["exact_historical_span_games"] == 3_980_000
        assert status["band_rebalances"] > 0
        assert pool.observability()["total_bucket_overlap_count"] == 0
        assert pool.unique_neural_opponent_count == K_RECENT + MEDIUM_TERM_CAPACITY
        assert bank.allocated_opponent_count == K_RECENT + MEDIUM_TERM_CAPACITY
    finally:
        bank.close()


def test_a_direct_band_move_preserves_identity_slot_and_performance():
    buckets = ("recent", "medium_term", "historical_uniform")
    network, bank, pool = _pool(buckets)
    try:
        archive = _run_iterations(
            pool,
            network,
            HISTORICAL_MINIMUM_AGE_ITERATIONS - MEDIUM_TERM_INTERVAL_ITERATIONS,
        )
        assert pool.bucket_members("historical_uniform") == ()
        moving_id = pool.bucket_members("medium_term")[0]
        assert pool.opponent(moving_id).introduced_iteration == 0
        slot_before = pool.bank_slot(moving_id)
        record_before = pool.opponent(moving_id)
        tracker = _tracker(pool)
        tracker.update(
            (record.opponent_id for record in pool.active_opponents()),
            {moving_id: {"games": 4, "wins": 3, "losses": 1}},
        )

        for step in range(
            pool.last_completed_rl_iteration + 1,
            HISTORICAL_MINIMUM_AGE_ITERATIONS + 1,
        ):
            archive.write(
                pool.consider_updated_policy(
                    network,
                    iteration=step,
                    completed_games=step * 2000,
                    has_samples=True,
                ),
                iteration=step,
                completed_games=step * 2000,
            )
        pool.reconcile_archive_backed_buckets(
            archive.records,
            completed_iteration=HISTORICAL_MINIMUM_AGE_ITERATIONS,
            load_weights=archive.load_weights,
        )
        # The oldest medium member becomes the first historical representative
        # inside one transaction, so nothing about it is rebuilt.
        assert pool.bucket_members("historical_uniform") == (moving_id,)
        assert moving_id not in pool.bucket_members("medium_term")
        assert pool.bank_slot(moving_id) == slot_before
        assert pool.opponent(moving_id) == record_before
        assert tracker.performance(moving_id).lifetime_wins == 3
        assert tracker.performance(moving_id).lifetime_losses == 1
        assert pool.observability()["total_bucket_overlap_count"] == 0
    finally:
        bank.close()


def test_iterations_without_samples_cannot_create_a_band_overlap():
    buckets = ("recent", "medium_term")
    network, bank, pool = _pool(buckets)
    try:
        archive = _StubArchive(network)
        archive.write(
            pool.initial_snapshot_record,
            iteration=0,
            completed_games=0,
        )
        # Only every third iteration trains, so ``recent`` reaches far further
        # back in absolute iterations than a dense run would.
        for iteration in range(1, K_RECENT * 3 + 1):
            completed_games = iteration * 2000
            archive.write(
                pool.consider_updated_policy(
                    network,
                    iteration=iteration,
                    completed_games=completed_games,
                    has_samples=iteration % 3 == 0,
                ),
                iteration=iteration,
                completed_games=completed_games,
            )
            if iteration % MEDIUM_TERM_INTERVAL_ITERATIONS == 0:
                pool.reconcile_archive_backed_buckets(
                    archive.records,
                    completed_iteration=iteration,
                    load_weights=archive.load_weights,
                )
        assert pool.observability()["total_bucket_overlap_count"] == 0
        recent_iterations = set(_member_iterations(pool, "recent"))
        medium_iterations = set(_member_iterations(pool, "medium_term"))
        assert not recent_iterations & medium_iterations
        assert max(medium_iterations) < min(recent_iterations)
    finally:
        bank.close()


def test_a_failed_archive_load_leaves_the_pool_and_slots_unchanged():
    buckets = ("recent", "medium_term")
    network, bank, pool = _pool(buckets)
    try:
        archive = _run_iterations(pool, network, K_RECENT - 1)
        before_members = {
            name: pool.bucket_members(name) for name in buckets
        }
        before_slots = dict(pool.bank_slot_by_opponent_id)
        allocated = bank.allocated_opponent_count

        def _failing_load(checkpoint_id):
            raise ValueError(f"corrupted archive entry: {checkpoint_id}")

        for step in range(K_RECENT, K_RECENT + 1):
            archive.write(
                pool.consider_updated_policy(
                    network,
                    iteration=step,
                    completed_games=step * 2000,
                    has_samples=True,
                ),
                iteration=step,
                completed_games=step * 2000,
            )
        with pytest.raises(ValueError, match="corrupted archive entry"):
            pool.reconcile_archive_backed_buckets(
                archive.records,
                completed_iteration=K_RECENT,
                load_weights=_failing_load,
            )
        assert pool.bucket_members("medium_term") == before_members["medium_term"]
        assert bank.allocated_opponent_count == allocated
        # The recent admission at this iteration is the only intended change.
        assert set(before_slots) - set(pool.bank_slot_by_opponent_id) == {
            pool.initial_snapshot_record.opponent_id
        }
    finally:
        bank.close()


def test_delayed_band_match_plan_is_exact_and_sources_are_disjoint():
    buckets = ("heuristic", "recent", "medium_term")
    network, bank, pool = _pool(buckets)
    try:
        _run_iterations(pool, network, K_RECENT + MEDIUM_TERM_INTERVAL_ITERATIONS)
        arguments = {
            "opponent_pool": pool,
            "selected_buckets": buckets,
            "difficulty_weight": 0.5,
            "iteration": 1,
            "first_absolute_game": 0,
            "game_count": 2000,
            "base_seed": 42,
        }
        plan = build_match_plan(performance_tracker=_tracker(pool), **arguments)
        repeated = build_match_plan(
            performance_tracker=_tracker(pool),
            **arguments,
        )
        assert repeated == plan
        assert len(plan.assignments) == 2000
        assert sum(item.game_count for item in plan.allocations) == 2000
        assert plan.available_buckets == buckets
        sources = {}
        for item in plan.allocations:
            sources.setdefault(item.opponent_id, set()).add(item.bucket_name)
        assert all(len(value) == 1 for value in sources.values())
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


def test_delayed_band_resume_restores_order_and_the_next_cadence_refresh():
    buckets = ("recent", "medium_term")
    network, bank, pool = _pool(buckets)
    restored_bank = None
    try:
        archive = _run_iterations(
            pool,
            network,
            K_RECENT + MEDIUM_TERM_INTERVAL_ITERATIONS,
        )
        saved_medium = pool.bucket_members("medium_term")
        assert _member_iterations(pool, "medium_term") == (0, 10)
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
            unique_neural_capacity(buckets),
        )
        restored = OpponentPool(
            restored_bank,
            selected_buckets=buckets,
            initial_network=restored_network,
        )
        restored.restore_state(state, weights)
        assert restored.bucket_members("medium_term") == saved_medium
        assert restored.last_completed_rl_iteration == 210

        for step in range(211, 221):
            archive.write(
                restored.consider_updated_policy(
                    restored_network,
                    iteration=step,
                    completed_games=step * 2000,
                    has_samples=True,
                ),
                iteration=step,
                completed_games=step * 2000,
            )
        # The refresh is neither skipped nor duplicated across the resume.
        restored.reconcile_archive_backed_buckets(
            archive.records,
            completed_iteration=220,
            load_weights=archive.load_weights,
        )
        assert _member_iterations(restored, "medium_term") == (0, 10, 20)
        assert restored.observability()["total_bucket_overlap_count"] == 0
    finally:
        if bank is not None:
            bank.close()
        if restored_bank is not None:
            restored_bank.close()
