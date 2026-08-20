"""Champion bucket membership, retention, and overlap contracts.

The racing algorithm lives in ``tests/test_rl_champion_evaluation.py``. This
module owns the pool side: which snapshots become candidates, what a committed
event does to membership and scores, how a full bucket evicts, and how a
champion identity behaves when it also belongs to a chronological band.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pytest

from agents.rl_nn import PolicyNetwork
from training.rl.matchmaking import (
    OpponentPerformanceTracker,
    UniformRotationState,
    aggregate_match_results,
    build_match_plan,
)
from training.rl.pool import (
    CHAMPION_CANDIDATE_BATCH_SIZE,
    CHAMPION_FINAL_SURVIVORS,
    CHAMPION_VS_HEURISTIC_BUCKET,
    CHAMPION_VS_HEURISTIC_CAPACITY,
    K_RECENT,
    MEDIUM_TERM_INTERVAL_ITERATIONS,
    OpponentPool,
    SharedPolicyBank,
    unique_neural_capacity,
)
# The pool now keys champion state by bucket. Every existing champion test
# targets the fixed-heuristic bucket, so it gets one short local name rather
# than the constant repeated inside dozens of assertions.
HEURISTIC_CHAMPION = CHAMPION_VS_HEURISTIC_BUCKET


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


@dataclass(frozen=True)
class _ArchiveMetadata:
    """Minimal band-selection input, matching the archive record contract."""

    checkpoint_id: str
    opponent_id: str
    completed_iteration: int
    completed_rl_games: int


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


def _uniform_by_member(plan, bucket_name):
    return {
        value.opponent_id: value.uniform_games
        for value in plan.allocations
        if value.bucket_name == bucket_name
    }


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


def _champion_pool(bucket_names=("heuristic", "recent", "champion_vs_heuristic")):
    return _pool(bucket_names)


def _race(pool, ids, win_rates):
    return pool.apply_champion_vs_heuristic_result(
        ids,
        dict(zip(ids, win_rates)),
    )


def _champion_members(pool):
    return pool.bucket_members("champion_vs_heuristic")


def test_only_successful_updates_become_champion_candidates():
    network, bank, pool = _champion_pool()
    try:
        # The warm-start policy is not a candidate.
        assert pool.champion_pending_candidate_ids(HEURISTIC_CHAMPION) == ()
        assert pool.champion_completed_event_count(HEURISTIC_CHAMPION) == 0

        pool.consider_updated_policy(
            network,
            iteration=1,
            completed_games=2000,
            has_samples=True,
        )
        assert len(pool.champion_pending_candidate_ids(HEURISTIC_CHAMPION)) == 1
        # An iteration with no trainable decisions produces no snapshot.
        pool.consider_updated_policy(
            network,
            iteration=2,
            completed_games=4000,
            has_samples=False,
        )
        assert len(pool.champion_pending_candidate_ids(HEURISTIC_CHAMPION)) == 1

        for iteration in range(3, CHAMPION_CANDIDATE_BATCH_SIZE + 2):
            pool.consider_updated_policy(
                network,
                iteration=iteration,
                completed_games=iteration * 2000,
                has_samples=True,
            )
        pending = pool.champion_pending_candidate_ids(HEURISTIC_CHAMPION)
        assert len(pending) == CHAMPION_CANDIDATE_BATCH_SIZE == 50
        assert pool.champion_candidate_batch_is_ready(HEURISTIC_CHAMPION)
        assert len(set(pending)) == len(pending)
        assert set(pending) <= set(pool.bucket_members("recent"))
        # A 51st snapshot before the event would make the batch overlap.
        with pytest.raises(RuntimeError, match="already complete"):
            pool.consider_updated_policy(
                network,
                iteration=CHAMPION_CANDIDATE_BATCH_SIZE + 2,
                completed_games=200_000,
                has_samples=True,
            )
    finally:
        bank.close()


def test_candidate_batches_never_overlap_across_events():
    network, bank, pool = _champion_pool()
    try:
        for iteration in range(1, CHAMPION_CANDIDATE_BATCH_SIZE + 1):
            pool.consider_updated_policy(
                network,
                iteration=iteration,
                completed_games=iteration * 2000,
                has_samples=True,
            )
        first_batch = pool.champion_pending_candidate_ids(HEURISTIC_CHAMPION)
        _race(pool, first_batch[:5], [0.9, 0.8, 0.7, 0.6, 0.5])
        assert pool.champion_pending_candidate_ids(HEURISTIC_CHAMPION) == ()
        assert pool.champion_completed_event_count(HEURISTIC_CHAMPION) == 1

        for iteration in range(
            CHAMPION_CANDIDATE_BATCH_SIZE + 1,
            2 * CHAMPION_CANDIDATE_BATCH_SIZE + 1,
        ):
            pool.consider_updated_policy(
                network,
                iteration=iteration,
                completed_games=iteration * 2000,
                has_samples=True,
            )
        second_batch = pool.champion_pending_candidate_ids(HEURISTIC_CHAMPION)
        assert len(second_batch) == CHAMPION_CANDIDATE_BATCH_SIZE
        assert not set(second_batch) & set(first_batch)
    finally:
        bank.close()


def _fill_champion_bucket(pool, network, *, events):
    """Run ``events`` synthetic races admitting five champions each."""
    iteration = pool.last_completed_rl_iteration
    admitted = []
    for event in range(events):
        batch = []
        for _step in range(CHAMPION_FINAL_SURVIVORS):
            iteration += 1
            record = pool.consider_updated_policy(
                network,
                iteration=iteration,
                completed_games=iteration * 2000,
                has_samples=True,
            )
            batch.append(record.opponent_id)
        # Descending scores per event so later events are strictly weaker.
        _race(pool, batch, [1.0 - event * 0.001 - index * 0.0001
                            for index in range(CHAMPION_FINAL_SURVIVORS)])
        admitted.extend(batch)
    return admitted


def test_champion_bucket_fills_five_at_a_time_without_eviction():
    network, bank, pool = _champion_pool()
    try:
        for event in range(1, 6):
            _fill_champion_bucket(pool, network, events=1)
            assert len(_champion_members(pool)) == 5 * event
            assert pool.champion_completed_event_count(HEURISTIC_CHAMPION) == event
        assert set(pool.heuristic_champion_win_rates()) == set(_champion_members(pool))
        assert pool.observability()["buckets"]["champion_vs_heuristic"][
            "score_evictions"
        ] == 0
    finally:
        bank.close()


def test_a_full_champion_bucket_evicts_its_weakest_stored_scores():
    network, bank, pool = _champion_pool()
    try:
        events = CHAMPION_VS_HEURISTIC_CAPACITY // CHAMPION_FINAL_SURVIVORS
        _fill_champion_bucket(pool, network, events=events)
        assert len(_champion_members(pool)) == CHAMPION_VS_HEURISTIC_CAPACITY

        before = pool.heuristic_champion_win_rates()
        weakest = sorted(before.items(), key=lambda item: (item[1], item[0]))
        expected_evicted = {opponent_id for opponent_id, _score in weakest[:5]}

        newcomers = []
        for step in range(CHAMPION_FINAL_SURVIVORS):
            record = pool.consider_updated_policy(
                network,
                iteration=10_000 + step,
                completed_games=(10_000 + step) * 2000,
                has_samples=True,
            )
            newcomers.append(record.opponent_id)
        summary = _race(pool, newcomers, [0.5] * CHAMPION_FINAL_SURVIVORS)

        members = set(_champion_members(pool))
        assert len(members) == CHAMPION_VS_HEURISTIC_CAPACITY
        assert set(summary["evicted"]) == expected_evicted
        assert not expected_evicted & members
        assert set(newcomers) <= members
        assert set(pool.heuristic_champion_win_rates()) == members
        assert pool.observability()["buckets"]["champion_vs_heuristic"][
            "score_evictions"
        ] == 5
    finally:
        bank.close()


def test_new_champions_are_admitted_even_when_weaker_than_every_incumbent():
    network, bank, pool = _champion_pool()
    try:
        events = CHAMPION_VS_HEURISTIC_CAPACITY // CHAMPION_FINAL_SURVIVORS
        _fill_champion_bucket(pool, network, events=events)
        incumbents = set(_champion_members(pool))
        assert min(pool.heuristic_champion_win_rates().values()) > 0.1

        newcomers = []
        for step in range(CHAMPION_FINAL_SURVIVORS):
            record = pool.consider_updated_policy(
                network,
                iteration=20_000 + step,
                completed_games=(20_000 + step) * 2000,
                has_samples=True,
            )
            newcomers.append(record.opponent_id)
        # Strictly worse than every incumbent, and still guaranteed admission.
        _race(pool, newcomers, [0.0] * CHAMPION_FINAL_SURVIVORS)
        members = set(_champion_members(pool))
        assert set(newcomers) <= members
        assert len(incumbents - members) == CHAMPION_FINAL_SURVIVORS
        assert all(
            pool.heuristic_champion_win_rates()[opponent_id] == 0.0
            for opponent_id in newcomers
        )
    finally:
        bank.close()


def test_equal_champion_scores_evict_the_smaller_opponent_id_first():
    network, bank, pool = _champion_pool()
    try:
        events = CHAMPION_VS_HEURISTIC_CAPACITY // CHAMPION_FINAL_SURVIVORS
        _fill_champion_bucket(pool, network, events=events)
        # Flatten every stored score so only the documented ID rule can decide.
        members = sorted(_champion_members(pool))
        pool._heuristic_win_rate_by_opponent_id = {
            opponent_id: 0.5 for opponent_id in members
        }
        newcomers = []
        for step in range(CHAMPION_FINAL_SURVIVORS):
            record = pool.consider_updated_policy(
                network,
                iteration=30_000 + step,
                completed_games=(30_000 + step) * 2000,
                has_samples=True,
            )
            newcomers.append(record.opponent_id)
        summary = _race(pool, newcomers, [0.5] * CHAMPION_FINAL_SURVIVORS)
        assert list(summary["evicted"]) == members[:CHAMPION_FINAL_SURVIVORS]
    finally:
        bank.close()


def test_a_champion_result_is_rejected_before_it_mutates_anything():
    network, bank, pool = _champion_pool()
    try:
        _fill_champion_bucket(pool, network, events=2)
        members = _champion_members(pool)
        scores = pool.heuristic_champion_win_rates()

        fresh = [
            pool.consider_updated_policy(
                network,
                iteration=500 + step,
                completed_games=(500 + step) * 2000,
                has_samples=True,
            ).opponent_id
            for step in range(CHAMPION_FINAL_SURVIVORS)
        ]
        with pytest.raises(ValueError, match="exactly 5 champions"):
            _race(pool, fresh[:4], [0.5] * 4)
        with pytest.raises(ValueError, match="already an incumbent"):
            _race(pool, members[:5], [0.5] * 5)
        with pytest.raises(KeyError):
            _race(pool, ["snapshot:9999999999", *fresh[:4]], [0.5] * 5)
        with pytest.raises(ValueError, match="invalid win rate"):
            _race(pool, fresh, [0.5, 0.5, 0.5, 0.5, 1.5])
        with pytest.raises(ValueError, match="repeats an opponent"):
            _race(pool, [fresh[0]] * 5, [0.5] * 5)
        with pytest.raises(ValueError, match="cover exactly"):
            pool.apply_champion_vs_heuristic_result(
                fresh,
                {fresh[0]: 0.5},
            )

        assert _champion_members(pool) == members
        assert pool.heuristic_champion_win_rates() == scores
        assert pool.champion_completed_event_count(HEURISTIC_CHAMPION) == 2
    finally:
        bank.close()


def _drive_champion_events(pool, network, *, iterations, ascending=True):
    """Advance training and race every complete batch, as iteration.py will.

    Event scores are strictly ordered so a test can predict which incumbents
    are the weakest. With ``ascending`` the oldest event is the weakest, which
    is also the event whose champions have long since left ``recent``.
    """
    first = pool.last_completed_rl_iteration + 1
    events = []
    for iteration in range(first, first + iterations):
        pool.consider_updated_policy(
            network,
            iteration=iteration,
            completed_games=iteration * 2000,
            has_samples=True,
        )
        if not pool.champion_candidate_batch_is_ready(HEURISTIC_CHAMPION):
            continue
        batch = pool.champion_pending_candidate_ids(HEURISTIC_CHAMPION)[:CHAMPION_FINAL_SURVIVORS]
        index = len(events)
        base = 0.1 + index * 0.002 if ascending else 0.9 - index * 0.002
        _race(pool, batch, [base + step * 0.0001 for step in range(len(batch))])
        events.append(batch)
    return events


def test_a_champion_keeps_its_identity_after_leaving_every_other_bucket():
    network, bank, pool = _champion_pool()
    try:
        events = _drive_champion_events(
            pool,
            network,
            iterations=CHAMPION_CANDIDATE_BATCH_SIZE,
        )
        champions = events[0]
        slots = {
            opponent_id: pool.bank_slot(opponent_id)
            for opponent_id in champions
        }
        # Overlap reuses one record, one slot, and one checkpoint identity.
        assert set(champions) <= set(pool.bucket_members("recent"))
        assert set(champions) <= set(_champion_members(pool))

        # Push every champion out of recent by ordinary FIFO retention.
        _drive_champion_events(pool, network, iterations=K_RECENT + 10)

        assert not set(champions) & set(pool.bucket_members("recent"))
        # The champion membership alone keeps the identity and its slot alive.
        assert set(champions) <= set(_champion_members(pool))
        for opponent_id in champions:
            assert pool.opponent(opponent_id).kind == "policy_snapshot"
            assert pool.bank_slot(opponent_id) == slots[opponent_id]
        assert set(pool.heuristic_champion_win_rates()) == set(_champion_members(pool))
    finally:
        bank.close()


def test_an_evicted_champion_with_no_other_membership_releases_its_slot():
    network, bank, pool = _champion_pool()
    try:
        events = CHAMPION_VS_HEURISTIC_CAPACITY // CHAMPION_FINAL_SURVIVORS
        _drive_champion_events(
            pool,
            network,
            iterations=events * CHAMPION_CANDIDATE_BATCH_SIZE,
        )
        assert len(_champion_members(pool)) == CHAMPION_VS_HEURISTIC_CAPACITY

        weakest = sorted(
            pool.heuristic_champion_win_rates().items(),
            key=lambda item: (item[1], item[0]),
        )[:CHAMPION_FINAL_SURVIVORS]
        doomed = [opponent_id for opponent_id, _score in weakest]
        # The weakest event is also the oldest, so these left recent long ago
        # and the champion membership is their only remaining reference.
        assert not set(doomed) & set(pool.bucket_members("recent"))

        newcomers = [
            pool.consider_updated_policy(
                network,
                iteration=pool.last_completed_rl_iteration + 1,
                completed_games=(pool.last_completed_rl_iteration + 1) * 2000,
                has_samples=True,
            ).opponent_id
            for _step in range(CHAMPION_FINAL_SURVIVORS)
        ]
        # Measured after the newcomers exist so only the event's own releases
        # are counted.
        before = pool.unique_neural_opponent_count
        _race(pool, newcomers, [0.99] * CHAMPION_FINAL_SURVIVORS)

        for opponent_id in doomed:
            assert pool.bank_slot(opponent_id) is None
            assert opponent_id not in pool.heuristic_champion_win_rates()
        assert pool.unique_neural_opponent_count == (
            before - CHAMPION_FINAL_SURVIVORS
        )
        assert len(_champion_members(pool)) == CHAMPION_VS_HEURISTIC_CAPACITY
    finally:
        bank.close()


def test_a_full_champion_bucket_refuses_generic_fifo_eviction():
    network, bank, pool = _champion_pool()
    try:
        events = CHAMPION_VS_HEURISTIC_CAPACITY // CHAMPION_FINAL_SURVIVORS
        _fill_champion_bucket(pool, network, events=events)
        extra = pool.consider_updated_policy(
            network,
            iteration=60_000,
            completed_games=120_000_000,
            has_samples=True,
        )
        with pytest.raises(RuntimeError, match="retains members by strength"):
            pool.add_membership("champion_vs_heuristic", extra.opponent_id)
        assert len(_champion_members(pool)) == CHAMPION_VS_HEURISTIC_CAPACITY
    finally:
        bank.close()


def test_champion_state_survives_export_and_restore():
    buckets = ("heuristic", "recent", "champion_vs_heuristic")
    network, bank, pool = _champion_pool(buckets)
    try:
        _fill_champion_bucket(pool, network, events=3)
        for iteration in range(900, 907):
            pool.consider_updated_policy(
                network,
                iteration=iteration,
                completed_games=iteration * 2000,
                has_samples=True,
            )
        state = pool.export_state()
        weights = pool.export_weights()
        expected_members = _champion_members(pool)
        expected_scores = pool.heuristic_champion_win_rates()
        expected_pending = pool.champion_pending_candidate_ids(HEURISTIC_CHAMPION)
    finally:
        bank.close()

    restored_bank = SharedPolicyBank(network, unique_neural_capacity(buckets))
    try:
        restored = OpponentPool(restored_bank, selected_buckets=buckets)
        restored.restore_state(state, weights)
        assert restored.bucket_members("champion_vs_heuristic") == expected_members
        assert restored.heuristic_champion_win_rates() == expected_scores
        assert restored.champion_pending_candidate_ids(HEURISTIC_CHAMPION) == expected_pending
        assert restored.champion_completed_event_count(HEURISTIC_CHAMPION) == 3
        # Exact float equality: a champion score is a stored measurement.
        assert all(
            restored.heuristic_champion_win_rates()[opponent_id] == value
            for opponent_id, value in expected_scores.items()
        )
    finally:
        restored_bank.close()


def test_restoring_inconsistent_champion_state_is_rejected():
    buckets = ("heuristic", "recent", "champion_vs_heuristic")
    network, bank, pool = _champion_pool(buckets)
    try:
        _fill_champion_bucket(pool, network, events=2)
        state = pool.export_state()
        weights = pool.export_weights()
    finally:
        bank.close()

    def _restore(mutate):
        broken = json.loads(json.dumps(state))
        mutate(broken)
        restored_bank = SharedPolicyBank(network, unique_neural_capacity(buckets))
        try:
            restored = OpponentPool(restored_bank, selected_buckets=buckets)
            restored.restore_state(broken, weights)
        finally:
            restored_bank.close()

    def _heuristic_state(value):
        return value["champion_state_by_bucket"][HEURISTIC_CHAMPION]

    with pytest.raises(ValueError, match="do not describe the current membership"):
        _restore(lambda value: _heuristic_state(value)[
            "heuristic_win_rate_by_opponent_id"
        ].popitem())
    with pytest.raises(ValueError, match="missing champion state"):
        _restore(lambda value: value.pop("champion_state_by_bucket"))
    # A state whose champion block covers the wrong set of buckets cannot be
    # read as this run's state, whichever bucket the saved block described.
    with pytest.raises(ValueError, match="selected champion buckets"):
        _restore(lambda value: value["champion_state_by_bucket"].pop(
            HEURISTIC_CHAMPION
        ))
    with pytest.raises(ValueError, match="complete pending candidate batch"):
        _restore(lambda value: _heuristic_state(value).__setitem__(
            "pending_candidate_ids",
            list(value["buckets"]["recent"]["member_ids"][:1])
            * 1 + [f"snapshot:{index:010d}" for index in range(900, 949)],
        ))


# ---------------------------------------------------------------------------
# Stage 8 additions: champion overlap in the match plan, and the two resume
# boundaries of specification 28 that need a real pool to be observable.
# ---------------------------------------------------------------------------


def _synthetic_results(plan, *, learner_wins_every_nth=3):
    """Return one deterministic rollout result per planned assignment."""
    results = []
    for assignment in plan.assignments:
        learner_position = assignment.game_index % 2
        learner_won = assignment.game_index % learner_wins_every_nth == 0
        results.append({
            "game_index": assignment.game_index,
            "bucket_name": assignment.bucket_name,
            "opponent_id": assignment.opponent_id,
            "opponent_kind": assignment.opponent_kind,
            "bank_slot": assignment.bank_slot,
            "learner_position": learner_position,
            "winner": (
                learner_position if learner_won else 1 - learner_position
            ),
        })
    return results


def _fill_champion_bucket_with_scores(pool, network, score_for_event, *, events):
    """Fill the champion bucket while dictating each event's stored scores."""
    iteration = pool.last_completed_rl_iteration
    for event in range(events):
        batch = []
        for _step in range(CHAMPION_FINAL_SURVIVORS):
            iteration += 1
            batch.append(pool.consider_updated_policy(
                network,
                iteration=iteration,
                completed_games=iteration * 2000,
                has_samples=True,
            ).opponent_id)
        base = score_for_event(event)
        _race(pool, batch, [
            base + step * 0.000001 for step in range(CHAMPION_FINAL_SURVIVORS)
        ])
    return iteration


def test_an_evicted_rotation_anchor_continues_at_its_insertion_point():
    """Specification 28G, on the bucket where it is actually distinguishable.

    ``recent`` evicts strictly oldest-first and identities sort by age, so a
    departed ``recent`` anchor always has insertion point zero: correct, but
    indistinguishable from a reset. ``champion_vs_heuristic`` evicts by stored
    score, which removes members from the middle of the sorted order, so it is
    the bucket that can prove the continuation rule.
    """
    buckets = ("heuristic", "recent", "champion_vs_heuristic")
    network, bank, pool = _champion_pool(buckets)
    try:
        events = CHAMPION_VS_HEURISTIC_CAPACITY // CHAMPION_FINAL_SURVIVORS
        middle = events // 2
        # A V-shaped score profile makes the weakest incumbents the middle-aged
        # ones, so the next event evicts from the middle of the sorted order.
        last = _fill_champion_bucket_with_scores(
            pool,
            network,
            lambda event: 0.5 + abs(event - middle) * 0.001,
            events=events,
        )
        members = tuple(sorted(_champion_members(pool)))
        assert len(members) == CHAMPION_VS_HEURISTIC_CAPACITY

        scores = pool.heuristic_champion_win_rates()
        doomed = tuple(sorted(
            sorted(scores, key=lambda item: (scores[item], item))[
                :CHAMPION_FINAL_SURVIVORS
            ]
        ))
        anchor = doomed[-1]
        anchor_index = members.index(anchor)
        # The eviction must not be at either end, or the insertion point would
        # collapse to zero and the test would prove nothing.
        assert 0 < anchor_index < len(members) - 1

        rotation = UniformRotationState(buckets)
        rotation.commit({"champion_vs_heuristic": anchor})

        newcomers = []
        for step in range(CHAMPION_FINAL_SURVIVORS):
            newcomers.append(pool.consider_updated_policy(
                network,
                iteration=last + 1 + step,
                completed_games=(last + 1 + step) * 2000,
                has_samples=True,
            ).opponent_id)
        summary = _race(pool, newcomers, [0.9] * CHAMPION_FINAL_SURVIVORS)
        assert tuple(sorted(summary["evicted"])) == doomed

        survivors = tuple(sorted(_champion_members(pool)))
        # 60 games -> 30 uniform -> 10 per bucket over 200 champion members.
        plan = _plan(pool, rotation, iteration=1, game_count=60)
        served = tuple(sorted(
            opponent_id
            for opponent_id, value in _uniform_by_member(
                plan,
                "champion_vs_heuristic",
            ).items()
            if value
        ))
        start = next(
            index for index, opponent_id in enumerate(survivors)
            if opponent_id > anchor
        )
        assert start > 0
        assert served == tuple(sorted(survivors[start:start + 10]))
        # A reset would have served the first ten members instead.
        assert served != tuple(survivors[:10])
    finally:
        bank.close()


def _drive_champion_and_archive(pool, network, last_iteration):
    """Advance training, race every batch, and refresh the delayed bands.

    Champions are chosen from the archive-milestone iterations inside each
    batch, so a champion is also an identity the bands will later claim. That
    is the only way to reach specification 28H.
    """
    archive = _StubArchive(network)
    archive.write(pool.initial_snapshot_record, iteration=0, completed_games=0)
    events = []
    for iteration in range(1, last_iteration + 1):
        completed_games = iteration * 2000
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
        if pool.champion_candidate_batch_is_ready(HEURISTIC_CHAMPION):
            milestones = [
                opponent_id
                for opponent_id in pool.champion_pending_candidate_ids(HEURISTIC_CHAMPION)
                if pool.opponent(opponent_id).introduced_iteration
                % MEDIUM_TERM_INTERVAL_ITERATIONS == 0
            ]
            batch = milestones[:CHAMPION_FINAL_SURVIVORS]
            assert len(batch) == CHAMPION_FINAL_SURVIVORS
            _race(pool, batch, [
                0.7 + step * 0.0001 for step in range(len(batch))
            ])
            events.append(tuple(batch))
        if iteration % MEDIUM_TERM_INTERVAL_ITERATIONS == 0:
            pool.reconcile_archive_backed_buckets(
                archive.records,
                completed_iteration=iteration,
                load_weights=archive.load_weights,
            )
    return events


def test_a_band_rebalance_keeps_an_identity_that_is_also_a_champion():
    """Specification 28H: one record, one slot, one score across a band move."""
    buckets = ("heuristic", "recent", "medium_term", "champion_vs_heuristic")
    network, bank, pool = _champion_pool(buckets)
    try:
        # 250 iterations puts the medium cutoff at 50, which is exactly the
        # first batch's milestone range, while recent has already dropped it.
        events = _drive_champion_and_archive(pool, network, 250)
        champions = events[0]
        assert champions == tuple(
            f"snapshot:{index:010d}" for index in (10, 20, 30, 40, 50)
        )
        slots = {
            opponent_id: pool.bank_slot(opponent_id)
            for opponent_id in champions
        }
        scores = {
            opponent_id: pool.heuristic_champion_win_rates()[opponent_id]
            for opponent_id in champions
        }
        tracker = _tracker(pool)
        tracker.update(
            (record.opponent_id for record in pool.active_opponents()),
            {champions[0]: {"games": 6, "wins": 5, "losses": 1}},
        )
        difficulty_before = tracker.estimated_win_rate(champions[0])

        recent = set(pool.bucket_members("recent"))
        medium = set(pool.bucket_members("medium_term"))
        # The rebalance moved them out of recent and into the delayed band.
        assert not set(champions) & recent
        assert set(champions) <= medium
        assert set(champions) <= set(_champion_members(pool))

        for opponent_id in champions:
            assert pool.bank_slot(opponent_id) == slots[opponent_id]
            assert pool.opponent(opponent_id).kind == "policy_snapshot"
            # Exact float equality: a champion score is a stored measurement.
            assert pool.heuristic_champion_win_rates()[opponent_id] == scores[opponent_id]
        assert tracker.estimated_win_rate(champions[0]) == difficulty_before

        state = pool.observability()
        assert state["total_forbidden_historical_overlap_count"] == 0
        assert state["champion_overlap_counts"][
            "champion_vs_heuristic|medium_term"
        ] == len(champions)
        # One weight copy per identity, so overlap costs no extra bank slot.
        assert len(set(pool.bank_slot(opponent_id)
                       for opponent_id in _champion_members(pool))) == len(
            _champion_members(pool)
        )
        assert pool.unique_neural_opponent_count == len(
            recent | medium | set(_champion_members(pool))
        )
    finally:
        bank.close()


def test_one_identity_in_recent_and_champion_receives_games_through_both():
    """Specification 27: overlap is two memberships over one durable identity."""
    buckets = ("heuristic", "recent", "champion_vs_heuristic")
    network, bank, pool = _champion_pool(buckets)
    try:
        _run_iterations(pool, network, 40)
        champions = pool.bucket_members("recent")[:CHAMPION_FINAL_SURVIVORS]
        _race(pool, champions, [0.6] * CHAMPION_FINAL_SURVIVORS)
        overlapping = champions[0]
        assert overlapping in pool.bucket_members("recent")
        assert overlapping in _champion_members(pool)

        game_count = 600
        plan = _plan(pool, UniformRotationState(buckets), iteration=1,
                     game_count=game_count)

        # Two allocation memberships for one identity.
        memberships = [
            value for value in plan.allocations
            if value.opponent_id == overlapping
        ]
        assert len(memberships) == 2
        assert {value.bucket_name for value in memberships} == {
            "recent",
            "champion_vs_heuristic",
        }
        # One weight copy: both memberships name the same physical bank slot.
        assert len({value.bank_slot for value in memberships}) == 1
        assert len({
            assignment.bank_slot
            for assignment in plan.assignments
            if assignment.opponent_id == overlapping
        }) == 1

        # Both components allocate through both memberships.
        assert all(value.uniform_games > 0 for value in memberships)
        assert all(value.difficulty_games > 0 for value in memberships)

        opponent_results, bucket_results = aggregate_match_results(
            plan,
            _synthetic_results(plan),
        )

        # Buckets are attributed separately; the identity is combined.
        by_bucket_games = {
            value.bucket_name: sum(
                1 for assignment in plan.assignments
                if assignment.opponent_id == overlapping
                and assignment.bucket_name == value.bucket_name
            )
            for value in memberships
        }
        assert len(by_bucket_games) == 2
        assert all(count > 0 for count in by_bucket_games.values())
        assert opponent_results[overlapping]["games"] == sum(
            by_bucket_games.values()
        )
        assert bucket_results["recent"]["games"] > by_bucket_games["recent"]

        # Exactly one performance row exists for the overlapping identity, so
        # the tracker cannot be updated twice for the same games.
        assert len(opponent_results) == len({
            value.opponent_id for value in plan.allocations
        })
        tracker = OpponentPerformanceTracker()
        active = tuple(
            record.opponent_id for record in pool.active_opponents()
        )
        tracker.update(active, opponent_results)
        combined = OpponentPerformanceTracker()
        combined.update(active, {
            overlapping: dict(opponent_results[overlapping]),
        })
        assert tracker.estimated_win_rate(overlapping) == (
            combined.estimated_win_rate(overlapping)
        )

        # Exact GPI and unique game IDs are unaffected by the overlap.
        assert len(plan.assignments) == game_count
        assert len({
            assignment.game_index for assignment in plan.assignments
        }) == game_count
        assert sum(
            value["games"] for value in bucket_results.values()
        ) == game_count
    finally:
        bank.close()
