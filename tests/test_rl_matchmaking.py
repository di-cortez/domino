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
    UniformRotationState,
    aggregate_match_results,
    build_match_plan,
    cyclic_remainder_allocation,
    difficulty_from_win_rate,
    matchmaking_policy_manifest,
)
from training.rl.checkpoint_archive import ArchiveRecord
from training.rl.pool import (
    BUCKET_SPECIFICATIONS,
    CHAMPION_BUCKET_NAMES,
    CHAMPION_VS_HEURISTIC_CAPACITY,
    CHAMPION_VS_LEARNER_CAPACITY,
    HEURISTIC_OPPONENT_ID,
    HISTORICAL_BAND_BUCKET_NAMES,
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
    ARCHIVE_BACKED_BUCKET_NAMES,
    BOOTSTRAP_CAPABLE_BUCKETS,
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


def test_champion_bucket_is_registered_after_the_historical_bands():
    assert canonicalize_bucket_names(
        "champion_vs_heuristic,recent,heuristic"
    ) == ("heuristic", "recent", "champion_vs_heuristic")
    specification = BUCKET_SPECIFICATIONS["champion_vs_heuristic"]
    assert specification.capacity == CHAMPION_VS_HEURISTIC_CAPACITY == 200
    assert specification.neural
    assert specification.admission_rule == "top_5_of_50_racing_vs_heuristic"
    assert specification.retention_rule == (
        "guaranteed_new_champions_evict_lowest_heuristic_win_rate"
    )
    # Its cadence counts successful snapshots, not absolute iterations.
    assert specification.admission_interval_iterations is None
    # It starts empty, so it can neither bootstrap a run nor be archive-derived.
    assert "champion_vs_heuristic" not in BOOTSTRAP_CAPABLE_BUCKETS
    assert "champion_vs_heuristic" not in ARCHIVE_BACKED_BUCKET_NAMES
    assert unique_neural_capacity((
        "heuristic",
        "recent",
        "medium_term",
        "historical_uniform",
        "champion_vs_heuristic",
    )) == 800
    # Every neural bucket at once, which is the largest bank a supported
    # selection can reserve. The conservative policy sums capacities without
    # compressing the overlap the champion buckets are allowed to have.
    assert unique_neural_capacity((
        "heuristic",
        "recent",
        "medium_term",
        "historical_uniform",
        "champion_vs_heuristic",
        "champion_vs_learner",
    )) == 1000


def test_learner_champion_bucket_is_registered_after_the_heuristic_one():
    """Registry order fixes the deterministic order champion events run in."""
    assert canonicalize_bucket_names(
        "champion_vs_learner,champion_vs_heuristic,recent,heuristic"
    ) == (
        "heuristic",
        "recent",
        "champion_vs_heuristic",
        "champion_vs_learner",
    )
    assert CHAMPION_BUCKET_NAMES == (
        "champion_vs_heuristic",
        "champion_vs_learner",
    )
    specification = BUCKET_SPECIFICATIONS["champion_vs_learner"]
    assert specification.capacity == CHAMPION_VS_LEARNER_CAPACITY == 200
    assert specification.neural
    assert specification.admission_rule == (
        "top_5_of_50_racing_vs_current_learner"
    )
    # Not the heuristic rule: a score against a target that moves between
    # events cannot decide which incumbent leaves a full bucket.
    assert specification.retention_rule == (
        "guaranteed_new_champions_evict_lowest_current_difficulty"
    )
    assert specification.admission_interval_iterations is None
    assert "champion_vs_learner" not in BOOTSTRAP_CAPABLE_BUCKETS
    assert "champion_vs_learner" not in ARCHIVE_BACKED_BUCKET_NAMES
    # Overlapping the chronological bands is allowed, so it is not one of the
    # buckets whose memberships must stay pairwise disjoint.
    assert "champion_vs_learner" not in HISTORICAL_BAND_BUCKET_NAMES


def test_the_champion_policy_manifest_separates_the_two_targets():
    manifest = pool_policy_manifest(("heuristic", "recent"))["champion_policy"]
    # What the buckets share is stated once; what makes them different is
    # stated per target, so no reader can take one bucket's retention rule as
    # the policy of champion buckets in general.
    assert manifest["common"]["candidate_batch_size"] == 50
    assert manifest["common"]["final_survivors"] == 5
    assert manifest["common"]["overlaps_other_champion_bucket"] is True
    heuristic = manifest["targets"]["champion_vs_heuristic"]
    learner = manifest["targets"]["champion_vs_learner"]
    assert heuristic["target"] == "fixed_heuristic"
    assert heuristic["durable_admission_score"] == (
        "final_stage_win_rate_versus_fixed_heuristic"
    )
    assert learner["target"] == "frozen_post_update_current_learner"
    assert learner["durable_admission_score"] is None
    assert learner["retention_signal"] == "opponent_performance_difficulty"
    assert learner["retention_signal_is_decayed"] is True
    assert learner["eviction"] == "lowest_current_difficulty_then_opponent_id"


def test_champion_bucket_requires_recent_in_this_version():
    with pytest.raises(ValueError, match="requires recent"):
        resolve_training_options(
            RLTrainingOptions(
                opponent_buckets=("heuristic", "champion_vs_heuristic"),
            ),
            RLResourceOptions(workers=1),
            RLExecutionOptions(),
        )
    resolved = resolve_training_options(
        RLTrainingOptions(
            opponent_buckets=("champion_vs_heuristic", "recent", "heuristic"),
        ),
        RLResourceOptions(workers=1),
        RLExecutionOptions(),
    )
    assert resolved.training.opponent_buckets == (
        "heuristic",
        "recent",
        "champion_vs_heuristic",
    )


def test_both_champion_buckets_require_recent_and_name_themselves():
    """One rule, one error style, whichever champion target asked for it."""
    with pytest.raises(ValueError, match="champion_vs_learner currently requires recent"):
        resolve_training_options(
            RLTrainingOptions(
                opponent_buckets=("heuristic", "champion_vs_learner"),
            ),
            RLResourceOptions(workers=1),
            RLExecutionOptions(),
        )
    # Both selected: the message names both rather than one hard-coded bucket.
    with pytest.raises(
        ValueError,
        match="champion_vs_heuristic, champion_vs_learner currently require ",
    ):
        resolve_training_options(
            RLTrainingOptions(opponent_buckets=(
                "heuristic",
                "champion_vs_heuristic",
                "champion_vs_learner",
            )),
            RLResourceOptions(workers=1),
            RLExecutionOptions(),
        )
    resolved = resolve_training_options(
        RLTrainingOptions(opponent_buckets=(
            "champion_vs_learner",
            "champion_vs_heuristic",
            "recent",
            "heuristic",
        )),
        RLResourceOptions(workers=1),
        RLExecutionOptions(),
    )
    assert resolved.training.opponent_buckets == (
        "heuristic",
        "recent",
        "champion_vs_heuristic",
        "champion_vs_learner",
    )


def test_champion_only_selection_is_rejected_before_the_recent_rule():
    # The bootstrap rule fires first because champion alone can never play its
    # own first iteration either.
    with pytest.raises(ValueError, match="available from the first iteration"):
        resolve_training_options(
            RLTrainingOptions(opponent_buckets=("champion_vs_heuristic",)),
            RLResourceOptions(workers=1),
            RLExecutionOptions(),
        )


def test_overlap_reporting_separates_forbidden_pairs_from_champion_pairs():
    buckets = (
        "heuristic",
        "recent",
        "medium_term",
        "historical_uniform",
        "champion_vs_heuristic",
        "champion_vs_learner",
    )
    network, bank, pool = _pool(buckets)
    try:
        _run_iterations(pool, network, 20)
        state = pool.observability()
        assert set(state["forbidden_historical_overlap_counts"]) == {
            "recent|medium_term",
            "recent|historical_uniform",
            "medium_term|historical_uniform",
        }
        assert state["total_forbidden_historical_overlap_count"] == 0
        # Every champion pair is reported, including champion-to-champion.
        assert set(state["champion_overlap_counts"]) == {
            "champion_vs_heuristic|recent",
            "champion_vs_heuristic|medium_term",
            "champion_vs_heuristic|historical_uniform",
            "champion_vs_learner|recent",
            "champion_vs_learner|medium_term",
            "champion_vs_learner|historical_uniform",
            "champion_vs_heuristic|champion_vs_learner",
        }
        assert set(state["champion_overlap_counts"].values()) == {0}

        # Champion memberships over live recent identities are legal and are
        # counted as intentional overlap, never as an invariant failure. Both
        # buckets race the same batch, so all three pairs become non-zero.
        champions = pool.bucket_members("recent")[:5]
        pool.apply_champion_vs_heuristic_result(
            champions,
            {opponent_id: 0.6 for opponent_id in champions},
        )
        pool.apply_champion_vs_learner_result(champions, {})
        state = pool.observability()
        assert state["total_forbidden_historical_overlap_count"] == 0
        counts = state["champion_overlap_counts"]
        assert counts["champion_vs_heuristic|recent"] == 5
        assert counts["champion_vs_learner|recent"] == 5
        assert counts["champion_vs_heuristic|champion_vs_learner"] == 5
        # One identity, one slot, one record: overlap is not duplication.
        assert pool.unique_neural_opponent_count == len(
            pool.bucket_members("recent")
        )

        # Non-zero champion-to-band overlap needs a champion that has aged out
        # of recent, which only the archive reconcile produces. That path is
        # exercised in tests/test_rl_champion_pool.py, where the band
        # membership is real rather than manufactured.
    finally:
        bank.close()


def test_an_empty_champion_bucket_is_configured_but_unavailable():
    buckets = ("heuristic", "recent", "champion_vs_heuristic")
    network, bank, pool = _pool(buckets)
    try:
        _run_iterations(pool, network, 5)
        assert pool.bucket_members("champion_vs_heuristic") == ()
        assert pool.available_bucket_names() == ("heuristic", "recent")
        plan = _plan(pool, UniformRotationState(buckets), iteration=1)
        assert plan.configured_buckets == buckets
        assert plan.available_buckets == ("heuristic", "recent")
        assert all(
            value.bucket_name != "champion_vs_heuristic"
            for value in plan.allocations
        )
        # The compact metrics header keeps a fixed shape, so the empty bucket
        # still reports a zero row.
        _by_opponent, by_bucket = aggregate_match_results(plan, [
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
        ])
        assert by_bucket["champion_vs_heuristic"] == {
            "games": 0,
            "wins": 0,
            "losses": 0,
        }
    finally:
        bank.close()


def test_pool_policy_manifest_publishes_versioned_band_boundaries():
    manifest = pool_policy_manifest(("recent", "medium_term", "historical_uniform"))
    assert manifest["schema_version"] == POOL_SCHEMA_VERSION
    assert manifest["policy_version"] == POOL_POLICY_VERSION
    # Registry order is also the deterministic order champion events run in,
    # so the two champion buckets are pinned to the end and to each other.
    assert manifest["bucket_registry_order"] == [
        "heuristic",
        "random",
        "recent",
        "medium_term",
        "historical_uniform",
        "champion_vs_heuristic",
        "champion_vs_learner",
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


def _members(count, *, prefix="snapshot", start=0):
    return tuple(f"{prefix}:{index:010d}" for index in range(start, start + count))


def _rotate_once(state, bucket_name, members, budget):
    counts, next_anchors = state.plan_uniform_allocation(
        {bucket_name: budget},
        {bucket_name: members},
    )
    state.commit(next_anchors)
    return {
        opponent_id: value
        for (_bucket, opponent_id), value in counts.items()
    }


def _extra_receivers(counts, base):
    return tuple(
        opponent_id
        for opponent_id, value in sorted(counts.items())
        if value > base
    )


def test_cyclic_allocation_is_exact_and_never_differs_by_more_than_one():
    members = _members(7)
    for total in range(0, 30):
        counts, _anchor = cyclic_remainder_allocation(
            total,
            members,
            anchor=None,
        )
        assert set(counts) == set(members)
        assert sum(counts.values()) == total
        assert max(counts.values()) - min(counts.values()) <= 1


def test_zero_budget_allocates_nothing_and_leaves_the_anchor_untouched():
    members = _members(4)
    state = UniformRotationState(("heuristic", "recent"))
    counts = _rotate_once(state, "recent", members, 0)
    assert set(counts.values()) == {0}
    assert state.anchor("recent") is None
    # A later non-zero budget must still start from the first canonical member.
    counts = _rotate_once(state, "recent", members, 1)
    assert _extra_receivers(counts, 0) == (members[0],)


def test_a_single_member_bucket_absorbs_the_complete_bucket_budget():
    members = _members(1)
    state = UniformRotationState(("heuristic", "recent"))
    counts = _rotate_once(state, "recent", members, 13)
    assert counts == {members[0]: 13}
    # A one-member bucket never has a remainder, so there is no bias to rotate.
    assert state.anchor("recent") is None


def test_a_budget_below_the_member_count_rotates_through_the_whole_bucket():
    members = _members(200)
    state = UniformRotationState(("recent",))
    windows = []
    for _iteration in range(4):
        counts = _rotate_once(state, "recent", members, 66)
        assert sum(counts.values()) == 66
        windows.append(_extra_receivers(counts, 0))
    assert windows[0] == members[0:66]
    assert windows[1] == members[66:132]
    assert windows[2] == members[132:198]
    # The fourth window wraps past the end and resumes at the first identity.
    assert set(windows[3]) == set(members[198:200]) | set(members[0:64])
    assert state.anchor("recent") == members[63]


def test_a_budget_equal_to_the_member_count_has_no_remainder_to_rotate():
    members = _members(200)
    state = UniformRotationState(("recent",))
    counts = _rotate_once(state, "recent", members, 200)
    assert set(counts.values()) == {1}
    assert state.anchor("recent") is None


def test_a_budget_above_the_member_count_rotates_only_the_remainder():
    members = _members(200)
    state = UniformRotationState(("recent",))
    counts = _rotate_once(state, "recent", members, 266)
    assert sum(counts.values()) == 266
    assert set(counts.values()) == {1, 2}
    assert _extra_receivers(counts, 1) == members[0:66]
    counts = _rotate_once(state, "recent", members, 266)
    assert _extra_receivers(counts, 1) == members[66:132]


def test_fixed_membership_coverage_is_balanced_over_a_complete_cycle():
    members = _members(10)
    state = UniformRotationState(("recent",))
    totals = {opponent_id: 0 for opponent_id in members}
    for _iteration in range(10):
        counts = _rotate_once(state, "recent", members, 3)
        for opponent_id, value in counts.items():
            totals[opponent_id] += value
    assert sum(totals.values()) == 30
    assert max(totals.values()) - min(totals.values()) <= 1


def test_a_departed_anchor_continues_at_its_insertion_point():
    members = _members(6)
    state = UniformRotationState(("recent",))
    _rotate_once(state, "recent", members, 3)
    assert state.anchor("recent") == members[2]
    # The anchored identity leaves the bucket before the next allocation.
    survivors = tuple(
        opponent_id for opponent_id in members if opponent_id != members[2]
    )
    counts = _rotate_once(state, "recent", survivors, 2)
    assert _extra_receivers(counts, 0) == (members[3], members[4])


def test_repeated_fifo_membership_shifts_never_reset_the_rotation():
    state = UniformRotationState(("recent",))
    window = 8
    receivers = []
    for step in range(6):
        members = _members(window, start=step)
        counts = _rotate_once(state, "recent", members, 2)
        receivers.extend(_extra_receivers(counts, 0))
    # Every iteration keeps moving forward: no identity is served twice and the
    # first canonical member is never re-selected after the opening iteration.
    assert len(set(receivers)) == len(receivers)
    assert receivers[2:] and _members(1)[0] not in receivers[2:]


def test_a_full_membership_replacement_still_allocates_deterministically():
    state = UniformRotationState(("historical_uniform",))
    first = _members(5, prefix="snapshot", start=0)
    counts = _rotate_once(state, "historical_uniform", first, 2)
    assert _extra_receivers(counts, 0) == first[0:2]
    # A rebalance can replace every identity at once; the anchor is gone but
    # its insertion point is still well defined against the new membership.
    second = _members(5, prefix="snapshot", start=100)
    counts = _rotate_once(state, "historical_uniform", second, 2)
    assert _extra_receivers(counts, 0) == second[0:2]
    counts = _rotate_once(state, "historical_uniform", second, 2)
    assert _extra_receivers(counts, 0) == second[2:4]


def test_each_bucket_rotates_independently_over_the_same_identity():
    shared = _members(3)
    state = UniformRotationState(("recent", "medium_term"))
    counts, next_anchors = state.plan_uniform_allocation(
        {"recent": 1, "medium_term": 2},
        {"recent": shared, "medium_term": shared},
    )
    state.commit(next_anchors)
    assert counts[("recent", shared[0])] == 1
    assert counts[("medium_term", shared[0])] == 1
    assert counts[("medium_term", shared[1])] == 1
    assert state.anchor("recent") == shared[0]
    assert state.anchor("medium_term") == shared[1]


def test_planning_alone_never_advances_a_durable_anchor():
    members = _members(4)
    state = UniformRotationState(("recent",))
    counts, next_anchors = state.plan_uniform_allocation(
        {"recent": 2},
        {"recent": members},
    )
    assert next_anchors == {"recent": members[1]}
    assert sum(counts.values()) == 2
    # A caller that abandons the plan must leave the rotation exactly as it was.
    assert state.anchor("recent") is None
    counts = _rotate_once(state, "recent", members, 2)
    assert _extra_receivers(counts, 0) == members[0:2]


def test_an_unavailable_bucket_is_registered_but_never_advanced():
    state = UniformRotationState(("heuristic", "recent", "historical_uniform"))
    assert state.configured_buckets == (
        "heuristic",
        "recent",
        "historical_uniform",
    )
    assert state.anchors() == {
        "heuristic": None,
        "recent": None,
        "historical_uniform": None,
    }
    members = _members(3)
    _rotate_once(state, "recent", members, 1)
    assert state.anchor("historical_uniform") is None
    with pytest.raises(KeyError):
        state.anchor("champion")


def test_rotation_state_rejects_unsorted_members_and_unknown_buckets():
    state = UniformRotationState(("recent",))
    with pytest.raises(ValueError):
        state.plan_uniform_allocation(
            {"recent": 1},
            {"recent": ("snapshot:b", "snapshot:a")},
        )
    with pytest.raises(KeyError):
        state.plan_uniform_allocation({"medium_term": 1}, {"medium_term": ()})
    with pytest.raises(KeyError):
        state.commit({"medium_term": "snapshot:a"})
    with pytest.raises(ValueError):
        cyclic_remainder_allocation(3, (), anchor=None)


def test_rotation_state_export_restore_is_exact_and_version_checked():
    members = _members(5)
    state = UniformRotationState(("heuristic", "recent"))
    _rotate_once(state, "recent", members, 3)
    exported = state.export_state()
    assert exported["anchors"] == {"heuristic": None, "recent": members[2]}

    restored = UniformRotationState(("heuristic", "recent"))
    restored.restore_state(exported, ("heuristic", "recent"))
    assert restored.anchors() == state.anchors()
    assert _rotate_once(restored, "recent", members, 2) == _rotate_once(
        state,
        "recent",
        members,
        2,
    )

    with pytest.raises(ValueError):
        restored.restore_state(exported, ("heuristic", "recent", "medium_term"))
    stale = {
        "policy_manifest": {
            **matchmaking_policy_manifest(),
            "policy_version": matchmaking_policy_manifest()["policy_version"] + 1,
        },
        "anchors": exported["anchors"],
    }
    with pytest.raises(ValueError):
        restored.restore_state(stale, ("heuristic", "recent"))


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
            uniform_rotation=UniformRotationState(("heuristic", "recent")),
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
        first = build_match_plan(
            uniform_rotation=UniformRotationState(("heuristic", "recent")),
            **arguments,
        )
        second = build_match_plan(
            uniform_rotation=UniformRotationState(("heuristic", "recent")),
            **arguments,
        )
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
            uniform_rotation=UniformRotationState(buckets),
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
            uniform_rotation=UniformRotationState(buckets),
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
            uniform_rotation=UniformRotationState(("heuristic", "recent")),
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
            uniform_rotation=UniformRotationState(buckets),
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
                uniform_rotation=UniformRotationState(("historical_uniform",)),
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


def _uniform_by_member(plan, bucket_name):
    return {
        value.opponent_id: value.uniform_games
        for value in plan.allocations
        if value.bucket_name == bucket_name
    }


def _difficulty_by_member(plan, bucket_name):
    return {
        value.opponent_id: value.difficulty_games
        for value in plan.allocations
        if value.bucket_name == bucket_name
    }


def _uniform_by_bucket(plan):
    totals = {name: 0 for name in plan.configured_buckets}
    for value in plan.allocations:
        totals[value.bucket_name] += value.uniform_games
    return totals


def _plan(pool, rotation, *, iteration, game_count=64, difficulty_weight=0.5):
    return build_match_plan(
        opponent_pool=pool,
        performance_tracker=_tracker(pool),
        uniform_rotation=rotation,
        selected_buckets=tuple(pool.selected_buckets),
        difficulty_weight=difficulty_weight,
        iteration=iteration,
        first_absolute_game=(iteration - 1) * game_count,
        game_count=game_count,
        base_seed=7,
    )


def test_uniform_member_remainder_rotates_across_successive_plans():
    buckets = ("heuristic", "recent")
    network, bank, pool = _pool(buckets)
    try:
        _run_iterations(pool, network, 20)
        members = tuple(sorted(pool.bucket_members("recent")))
        assert len(members) == 21
        rotation = UniformRotationState(buckets)
        served = []
        anchors = []
        for iteration in range(1, 4):
            plan = _plan(pool, rotation, iteration=iteration)
            uniform = _uniform_by_member(plan, "recent")
            # 64 games -> 32 uniform -> 16 per bucket over 21 recent members.
            assert sum(uniform.values()) == 16
            assert set(uniform.values()) == {0, 1}
            served.append({
                opponent_id
                for opponent_id, value in uniform.items()
                if value
            })
            anchors.append(dict(plan.uniform_rotation_after)["recent"])
        # Consecutive windows of 16 walk forward and wrap around the 21 members.
        assert served[0] == set(members[0:16])
        assert served[1] == set(members[16:21]) | set(members[0:11])
        assert served[2] == set(members[11:21]) | set(members[0:6])
        # The anchor is the last member of each window in rotation order, which
        # is not the lexicographic maximum once a window has wrapped.
        assert anchors == [members[15], members[10], members[5]]
    finally:
        bank.close()


def test_every_member_receives_uniform_games_within_one_full_cycle():
    buckets = ("heuristic", "recent")
    network, bank, pool = _pool(buckets)
    try:
        _run_iterations(pool, network, 20)
        members = set(pool.bucket_members("recent"))
        rotation = UniformRotationState(buckets)
        served = set()
        for iteration in range(1, 3):
            plan = _plan(pool, rotation, iteration=iteration)
            served |= {
                opponent_id
                for opponent_id, value in _uniform_by_member(plan, "recent").items()
                if value
            }
        # 21 members at 16 uniform games per plan need ceil(21/16) == 2 plans.
        assert served == members
    finally:
        bank.close()


def test_bucket_level_uniform_allocation_is_unaffected_by_the_rotation():
    buckets = ("heuristic", "recent")
    network, bank, pool = _pool(buckets)
    try:
        _run_iterations(pool, network, 20)
        rotation = UniformRotationState(buckets)
        totals = [
            _uniform_by_bucket(_plan(pool, rotation, iteration=iteration))
            for iteration in range(1, 5)
        ]
        assert totals == [{"heuristic": 16, "recent": 16}] * 4
    finally:
        bank.close()


def test_the_rotation_anchor_changes_only_uniform_member_counts():
    buckets = ("heuristic", "recent")
    network, bank, pool = _pool(buckets)
    try:
        _run_iterations(pool, network, 20)
        fresh = UniformRotationState(buckets)
        advanced = UniformRotationState(buckets)
        # Consume three rotation steps so the two states disagree.
        for iteration in range(1, 4):
            _plan(pool, advanced, iteration=iteration)
        first = _plan(pool, fresh, iteration=9)
        second = _plan(pool, advanced, iteration=9)
        assert _uniform_by_member(first, "recent") != _uniform_by_member(
            second,
            "recent",
        )
        assert _difficulty_by_member(first, "recent") == _difficulty_by_member(
            second,
            "recent",
        )
        assert first.uniform_budget == second.uniform_budget
        assert first.difficulty_budget == second.difficulty_budget
        assert _uniform_by_bucket(first) == _uniform_by_bucket(second)
    finally:
        bank.close()


def test_component_member_counts_sum_exactly_to_their_bucket_budgets():
    buckets = ("heuristic", "recent")
    network, bank, pool = _pool(buckets)
    try:
        _run_iterations(pool, network, 20)
        rotation = UniformRotationState(buckets)
        for iteration in range(1, 6):
            plan = _plan(pool, rotation, iteration=iteration, game_count=2000)
            uniform = sum(value.uniform_games for value in plan.allocations)
            difficulty = sum(value.difficulty_games for value in plan.allocations)
            assert uniform == plan.uniform_budget
            assert difficulty == plan.difficulty_budget
            assert uniform + difficulty == 2000
            assert len(plan.assignments) == 2000
    finally:
        bank.close()


def test_plan_identity_records_the_rotation_transition():
    buckets = ("heuristic", "recent")
    network, bank, pool = _pool(buckets)
    try:
        _run_iterations(pool, network, 20)
        fresh = UniformRotationState(buckets)
        advanced = UniformRotationState(buckets)
        _plan(pool, advanced, iteration=1)
        first = _plan(pool, fresh, iteration=5)
        second = _plan(pool, advanced, iteration=5)
        assert first.uniform_rotation_before != second.uniform_rotation_before
        assert first.plan_sha256 != second.plan_sha256
        assert [name for name, _anchor in first.uniform_rotation_before] == list(
            buckets
        )
        # A bucket with a single member never has a remainder to rotate.
        assert dict(first.uniform_rotation_after)["heuristic"] is None
    finally:
        bank.close()


def test_a_rejected_plan_never_consumes_a_rotation_step(monkeypatch):
    buckets = ("heuristic", "recent")
    network, bank, pool = _pool(buckets)
    try:
        _run_iterations(pool, network, 20)
        members = tuple(sorted(pool.bucket_members("recent")))
        rotation = UniformRotationState(buckets)

        def _reject(_plan_value, _pool_value):
            raise ValueError("injected plan rejection")

        monkeypatch.setattr(
            "training.rl.matchmaking.validate_match_plan",
            _reject,
        )
        with pytest.raises(ValueError, match="injected plan rejection"):
            _plan(pool, rotation, iteration=1)
        assert rotation.anchors() == {"heuristic": None, "recent": None}
        monkeypatch.undo()
        # The abandoned plan consumed nothing, so the first window is still the
        # first canonical members.
        plan = _plan(pool, rotation, iteration=1)
        assert tuple(
            opponent_id
            for opponent_id, value in sorted(
                _uniform_by_member(plan, "recent").items()
            )
            if value
        ) == members[0:16]
    finally:
        bank.close()


def test_an_unavailable_bucket_keeps_its_anchor_until_it_becomes_available():
    buckets = ("heuristic", "recent", "medium_term")
    network, bank, pool = _pool(buckets)
    try:
        _run_iterations(pool, network, 20)
        rotation = UniformRotationState(buckets)
        plan = _plan(pool, rotation, iteration=1)
        assert plan.available_buckets == ("heuristic", "recent")
        assert dict(plan.uniform_rotation_after)["medium_term"] is None
        assert rotation.anchor("medium_term") is None
    finally:
        bank.close()


def test_split_and_uninterrupted_runs_produce_the_same_uniform_allocations():
    buckets = ("heuristic", "recent")
    network, bank, pool = _pool(buckets)
    try:
        _run_iterations(pool, network, 20)
        uninterrupted = UniformRotationState(buckets)
        expected = []
        for iteration in range(1, 7):
            plan = _plan(pool, uninterrupted, iteration=iteration)
            expected.append((plan.plan_sha256, _uniform_by_member(plan, "recent")))

        # Replay the first three plans, checkpoint, then continue from the
        # restored state exactly as a resumed process would.
        split = UniformRotationState(buckets)
        actual = []
        for iteration in range(1, 4):
            plan = _plan(pool, split, iteration=iteration)
            actual.append((plan.plan_sha256, _uniform_by_member(plan, "recent")))
        exported = split.export_state()
        resumed = UniformRotationState(buckets)
        resumed.restore_state(exported, buckets)
        for iteration in range(4, 7):
            plan = _plan(pool, resumed, iteration=iteration)
            actual.append((plan.plan_sha256, _uniform_by_member(plan, "recent")))

        assert actual == expected
        assert resumed.anchors() == uninterrupted.anchors()
    finally:
        bank.close()


def test_a_resumed_rotation_does_not_restart_from_the_first_member():
    buckets = ("heuristic", "recent")
    network, bank, pool = _pool(buckets)
    try:
        _run_iterations(pool, network, 20)
        members = tuple(sorted(pool.bucket_members("recent")))
        rotation = UniformRotationState(buckets)
        _plan(pool, rotation, iteration=1)
        resumed = UniformRotationState(buckets)
        resumed.restore_state(rotation.export_state(), buckets)
        plan = _plan(pool, resumed, iteration=2)
        served = {
            opponent_id
            for opponent_id, value in _uniform_by_member(plan, "recent").items()
            if value
        }
        # A reconstructed-from-membership anchor would replay members[0:16].
        assert served != set(members[0:16])
        assert served == set(members[16:21]) | set(members[0:11])
    finally:
        bank.close()


def test_match_results_have_only_wins_and_losses_and_reject_invalid_winner():
    _network_value, bank, pool = _pool(("random",))
    try:
        plan = build_match_plan(
            opponent_pool=pool,
            performance_tracker=_tracker(pool),
            selected_buckets=("random",),
            uniform_rotation=UniformRotationState(("random",)),
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
        assert pool.observability()["total_forbidden_historical_overlap_count"] == 0
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
        assert pool.observability()["total_forbidden_historical_overlap_count"] == 0
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
        assert pool.observability()["total_forbidden_historical_overlap_count"] == 0
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
        assert pool.observability()["total_forbidden_historical_overlap_count"] == 0
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
        plan = build_match_plan(
            performance_tracker=_tracker(pool),
            uniform_rotation=UniformRotationState(buckets),
            **arguments,
        )
        repeated = build_match_plan(
            performance_tracker=_tracker(pool),
            uniform_rotation=UniformRotationState(buckets),
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
            uniform_rotation=UniformRotationState(("random",)),
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
        assert restored.observability()["total_forbidden_historical_overlap_count"] == 0
    finally:
        if bank is not None:
            bank.close()
        if restored_bank is not None:
            restored_bank.close()
