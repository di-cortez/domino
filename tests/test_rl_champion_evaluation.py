"""Deterministic champion-vs-heuristic racing contracts.

Every test injects a fake stage evaluator. A real event plays 100,000 games and
has no place in the unit suite; what must be verified here is the algorithm,
the seed panels, the seat balance, and the ranking.
"""

from __future__ import annotations

import pytest

from training.rl.champion_evaluation import (
    _evaluate_champion_race,
    CHAMPION_EVALUATION_GAMES,
    CHAMPION_EVALUATION_MODE,
    CHAMPION_TARGET_CURRENT_LEARNER,
    CHAMPION_TARGET_HEURISTIC,
    CHAMPION_FINAL_GAMES,
    CHAMPION_RACING_STAGES,
    CHAMPION_STAGE_1_GAMES,
    CHAMPION_STAGE_1_SURVIVORS,
    CHAMPION_STAGE_2_GAMES,
    CHAMPION_STAGE_2_SURVIVORS,
    CHAMPION_STAGE_3_GAMES,
    CHAMPION_STAGE_3_SURVIVORS,
    build_stage_specs,
    champion_evaluation_game_count,
    champion_evaluation_policy_manifest,
    champion_seat_position,
    champion_stage_candidate_counts,
    champion_stage_seed,
    evaluate_champion_vs_heuristic,
    evaluate_champion_vs_learner,
    rank_stage_candidates,
    tally_stage_results,
)
from training.rl.pool import (
    CHAMPION_CANDIDATE_BATCH_SIZE,
    CHAMPION_FINAL_SURVIVORS,
    CHAMPION_VS_HEURISTIC_BUCKET,
    CHAMPION_VS_LEARNER_BUCKET,
)
# The pool now keys champion state by bucket. Every existing champion test
# targets the fixed-heuristic bucket, so it gets one short local name rather
# than the constant repeated inside dozens of assertions.
HEURISTIC_CHAMPION = CHAMPION_VS_HEURISTIC_BUCKET


def _candidates(count=CHAMPION_CANDIDATE_BATCH_SIZE):
    return tuple(f"snapshot:{index:010d}" for index in range(count))


def _slots(candidate_ids):
    return {
        candidate_id: index
        for index, candidate_id in enumerate(candidate_ids)
    }


class _ScriptedEvaluator:
    """Decide each game from a fixed per-candidate strength, not from play.

    ``strength`` maps a candidate ID to how many of the first games it wins, so
    a test can dictate an exact ranking while still exercising the real spec,
    tally, and ranking code.
    """

    def __init__(self, strength):
        self.strength = dict(strength)
        self.stages = []

    def __call__(self, specs):
        self.stages.append(specs)
        results = []
        for spec in specs:
            wins = self.strength[spec.candidate_id]
            candidate_won = spec.game_index < wins
            results.append({
                "sequence": spec.sequence,
                "candidate_id": spec.candidate_id,
                "game_index": spec.game_index,
                "candidate_position": spec.candidate_position,
                "winner": (
                    spec.candidate_position
                    if candidate_won
                    else 1 - spec.candidate_position
                ),
            })
        return results


def _descending_strength(candidate_ids):
    """Give earlier candidates strictly more wins in every stage."""
    return {
        candidate_id: CHAMPION_FINAL_GAMES - index
        for index, candidate_id in enumerate(candidate_ids)
    }


def test_the_fixed_racing_policy_costs_exactly_one_hundred_thousand_games():
    assert champion_stage_candidate_counts() == (
        CHAMPION_CANDIDATE_BATCH_SIZE,
        CHAMPION_STAGE_1_SURVIVORS,
        CHAMPION_STAGE_2_SURVIVORS,
        CHAMPION_STAGE_3_SURVIVORS,
    ) == (50, 40, 30, 20)
    assert [stage.games_per_candidate for stage in CHAMPION_RACING_STAGES] == [
        CHAMPION_STAGE_1_GAMES,
        CHAMPION_STAGE_2_GAMES,
        CHAMPION_STAGE_3_GAMES,
        CHAMPION_FINAL_GAMES,
    ] == [500, 500, 500, 2000]
    assert [stage.survivors for stage in CHAMPION_RACING_STAGES] == [
        40,
        30,
        20,
        CHAMPION_FINAL_SURVIVORS,
    ] == [40, 30, 20, 5]
    assert (
        50 * 500 + 40 * 500 + 30 * 500 + 20 * 2000
        == 25_000 + 20_000 + 15_000 + 40_000
        == champion_evaluation_game_count()
        == CHAMPION_EVALUATION_GAMES
        == 100_000
    )
    manifest = champion_evaluation_policy_manifest()
    assert manifest["total_games"] == 100_000
    assert manifest["counts_toward_gpi"] is False
    assert manifest["seed_excludes_candidate_identity"] is True
    assert manifest["candidate_action_mode"] == CHAMPION_EVALUATION_MODE == (
        "evaluation"
    )


def test_every_candidate_in_a_stage_faces_the_identical_seed_panel():
    candidates = _candidates()
    specs = build_stage_specs(
        candidates,
        _slots(candidates),
        base_seed=42,
        seed_namespace=HEURISTIC_CHAMPION,
        target_kind=CHAMPION_TARGET_HEURISTIC,
        event_index=0,
        stage_index=0,
    )
    by_candidate = {}
    for spec in specs:
        by_candidate.setdefault(spec.candidate_id, []).append(
            (spec.game_index, spec.seed, spec.candidate_position)
        )
    panels = {tuple(sorted(value)) for value in by_candidate.values()}
    assert len(panels) == 1
    assert len(by_candidate) == CHAMPION_CANDIDATE_BATCH_SIZE
    assert len(specs) == CHAMPION_CANDIDATE_BATCH_SIZE * CHAMPION_STAGE_1_GAMES
    assert len({spec.sequence for spec in specs}) == len(specs)


def _heuristic_seed(base_seed=42, *, event_index=0, stage_index=0, game_index=7):
    return champion_stage_seed(
        base_seed,
        seed_namespace=HEURISTIC_CHAMPION,
        event_index=event_index,
        stage_index=stage_index,
        game_index=game_index,
    )


def test_the_stage_seed_ignores_candidate_identity_and_changes_per_stage():
    first = _heuristic_seed()
    # Same inputs, no candidate anywhere in the derivation.
    assert first == _heuristic_seed()
    panels = {
        _heuristic_seed(stage_index=stage)
        for stage in range(len(CHAMPION_RACING_STAGES))
    }
    assert len(panels) == len(CHAMPION_RACING_STAGES)
    events = {_heuristic_seed(event_index=event) for event in range(4)}
    assert len(events) == 4
    assert first not in (
        _heuristic_seed(43),
        _heuristic_seed(game_index=8),
    )


def test_the_two_champion_buckets_never_share_a_seed_panel():
    """Same coordinates, different target: the deals must not be reused.

    Both buckets normally run event N off the same candidate stream, so every
    integer coordinate of their panels can coincide. Only the namespace keeps
    the two evaluations statistically independent.
    """
    for event_index in range(3):
        for stage_index in range(len(CHAMPION_RACING_STAGES)):
            for game_index in (0, 7, 499):
                shared = {
                    champion_stage_seed(
                        42,
                        seed_namespace=namespace,
                        event_index=event_index,
                        stage_index=stage_index,
                        game_index=game_index,
                    )
                    for namespace in (HEURISTIC_CHAMPION,
                                      CHAMPION_VS_LEARNER_BUCKET)
                }
                assert len(shared) == 2


def test_the_seed_namespace_cannot_be_omitted_or_invented():
    """No default: a caller that forgets it fails loudly rather than aliasing."""
    with pytest.raises(TypeError):
        champion_stage_seed(42, event_index=0, stage_index=0, game_index=7)
    with pytest.raises(ValueError, match="Unknown champion seed namespace"):
        champion_stage_seed(
            42,
            seed_namespace="recent",
            event_index=0,
            stage_index=0,
            game_index=7,
        )


@pytest.mark.parametrize("stage_index", range(len(CHAMPION_RACING_STAGES)))
def test_every_stage_balances_candidate_seats_exactly(stage_index):
    stage = CHAMPION_RACING_STAGES[stage_index]
    candidates = _candidates(champion_stage_candidate_counts()[stage_index])
    specs = build_stage_specs(
        candidates,
        _slots(candidates),
        base_seed=11,
        seed_namespace=HEURISTIC_CHAMPION,
        target_kind=CHAMPION_TARGET_HEURISTIC,
        event_index=2,
        stage_index=stage_index,
    )
    for candidate_id in candidates:
        seats = [
            spec.candidate_position
            for spec in specs
            if spec.candidate_id == candidate_id
        ]
        assert len(seats) == stage.games_per_candidate
        assert seats.count(0) == seats.count(1) == stage.games_per_candidate // 2
    assert champion_seat_position(0) == 0
    assert champion_seat_position(1) == 1
    assert champion_seat_position(2) == 0


def test_the_learner_race_uses_the_same_funnel_and_its_own_namespace():
    """One algorithm, two targets: only the panel and the target differ."""
    candidates = _candidates()
    heuristic = _ScriptedEvaluator(_descending_strength(candidates))
    learner = _ScriptedEvaluator(_descending_strength(candidates))
    heuristic_result = evaluate_champion_vs_heuristic(
        candidate_ids=candidates,
        bank_slots=_slots(candidates),
        play_games=heuristic,
        base_seed=7,
        event_index=3,
    )
    learner_result = evaluate_champion_vs_learner(
        candidate_ids=candidates,
        bank_slots=_slots(candidates),
        play_games=learner,
        base_seed=7,
        event_index=3,
    )

    # Identical mechanics: same cost, same funnel, same winners for the same
    # scripted strengths.
    assert learner_result.total_games == CHAMPION_EVALUATION_GAMES == 100_000
    assert [len(stage) for stage in learner.stages] == [
        len(stage) for stage in heuristic.stages
    ] == [50 * 500, 40 * 500, 30 * 500, 20 * 2000]
    assert learner_result.champion_ids == heuristic_result.champion_ids
    assert learner_result.final_win_rates == heuristic_result.final_win_rates
    assert learner_result.bucket_name == CHAMPION_VS_LEARNER_BUCKET
    assert heuristic_result.bucket_name == HEURISTIC_CHAMPION

    # Same event index and the same run seed, yet not one shared deal.
    for learner_stage, heuristic_stage in zip(learner.stages, heuristic.stages):
        learner_seeds = {spec.seed for spec in learner_stage}
        heuristic_seeds = {spec.seed for spec in heuristic_stage}
        assert learner_seeds & heuristic_seeds == set()

    # The target travels on the spec, and the bank slot keeps addressing the
    # candidate: the learner is not an opponent slot.
    for stage in learner.stages:
        assert {spec.target_kind for spec in stage} == {
            CHAMPION_TARGET_CURRENT_LEARNER
        }
    for stage in heuristic.stages:
        assert {spec.target_kind for spec in stage} == {
            CHAMPION_TARGET_HEURISTIC
        }
    slots = _slots(candidates)
    for spec in learner.stages[0]:
        assert spec.bank_slot == slots[spec.candidate_id]


def test_the_learner_race_balances_seats_and_shares_one_panel_per_stage():
    candidates = _candidates()
    evaluator = _ScriptedEvaluator(_descending_strength(candidates))
    evaluate_champion_vs_learner(
        candidate_ids=candidates,
        bank_slots=_slots(candidates),
        play_games=evaluator,
        base_seed=99,
        event_index=0,
    )
    for stage_index, stage_specs in enumerate(evaluator.stages):
        stage = CHAMPION_RACING_STAGES[stage_index]
        by_candidate = {}
        for spec in stage_specs:
            by_candidate.setdefault(spec.candidate_id, []).append(
                (spec.game_index, spec.seed, spec.candidate_position)
            )
        # One panel for the whole stage, and an exact 50/50 seat split.
        assert len({tuple(sorted(v)) for v in by_candidate.values()}) == 1
        for seats in by_candidate.values():
            positions = [item[2] for item in seats]
            half = stage.games_per_candidate // 2
            assert positions.count(0) == positions.count(1) == half
    # Fresh panel between stages: no screening luck is replayed.
    panels = [{spec.seed for spec in stage} for stage in evaluator.stages]
    for index, panel in enumerate(panels):
        for other in panels[index + 1:]:
            assert panel & other == set()


def test_a_race_against_an_unknown_bucket_is_rejected():
    candidates = _candidates()
    with pytest.raises(ValueError, match="is not a champion bucket"):
        _evaluate_champion_race(
            bucket_name="recent",
            candidate_ids=candidates,
            bank_slots=_slots(candidates),
            play_games=_ScriptedEvaluator(_descending_strength(candidates)),
            base_seed=1,
            event_index=0,
        )


def test_the_evaluation_manifest_separates_common_racing_from_the_targets():
    manifest = champion_evaluation_policy_manifest()
    assert manifest["policy_version"] == 2
    assert manifest["total_games"] == 100_000
    assert manifest["candidate_action_mode"] == CHAMPION_EVALUATION_MODE
    assert manifest["stage_scoring"] == "current_stage_games_only"
    assert manifest["counts_toward_gpi"] is False
    # No top-level seed namespace any more: it is not a property the two
    # targets share.
    assert "seed_namespace" not in manifest
    # The fixed policy of both targets is always published; which ones a run
    # actually races is recorded separately, in registry order regardless of
    # the order the caller passed them.
    assert manifest["selected_targets"] == []
    assert champion_evaluation_policy_manifest(
        (CHAMPION_VS_LEARNER_BUCKET, HEURISTIC_CHAMPION)
    )["selected_targets"] == [HEURISTIC_CHAMPION, CHAMPION_VS_LEARNER_BUCKET]
    assert champion_evaluation_policy_manifest(
        (CHAMPION_VS_LEARNER_BUCKET,)
    )["selected_targets"] == [CHAMPION_VS_LEARNER_BUCKET]
    heuristic = manifest["targets"][HEURISTIC_CHAMPION]
    learner = manifest["targets"][CHAMPION_VS_LEARNER_BUCKET]
    assert heuristic["seed_namespace"] != learner["seed_namespace"]
    assert heuristic["opponent_kind"] == "strategic_agent"
    assert heuristic["win_rates_are_durable"] is True
    assert learner["opponent_kind"] == "frozen_post_update_current_learner"
    assert learner["target_action_mode"] == CHAMPION_EVALUATION_MODE
    assert learner["win_rates_are_durable"] is False


# The funnel, the ranking, the tally, and the result shape belong to the one
# shared racing core, so each contract is checked through both wrappers.
RACE_EVALUATORS = [evaluate_champion_vs_heuristic, evaluate_champion_vs_learner]


@pytest.mark.parametrize("race", RACE_EVALUATORS)
def test_racing_narrows_fifty_candidates_to_five_and_scores_only_the_final(race):
    candidates = _candidates()
    evaluator = _ScriptedEvaluator(_descending_strength(candidates))
    result = race(
        candidate_ids=candidates,
        bank_slots=_slots(candidates),
        play_games=evaluator,
        base_seed=7,
        event_index=0,
    )
    assert [len(stage) for stage in evaluator.stages] == [
        50 * 500,
        40 * 500,
        30 * 500,
        20 * 2000,
    ]
    assert result.total_games == CHAMPION_EVALUATION_GAMES == 100_000
    assert result.champion_ids == candidates[:CHAMPION_FINAL_SURVIVORS]
    assert set(result.final_win_rates) == set(result.champion_ids)
    # The stored score is the final stage's rate, never a cumulative 3,500-game
    # rate and never an earlier stage's.
    for index, candidate_id in enumerate(result.champion_ids):
        assert result.final_win_rates[candidate_id] == (
            CHAMPION_FINAL_GAMES - index
        ) / CHAMPION_FINAL_GAMES
    assert len(result.stage_summaries) == 4
    assert result.stage_summaries[-1].survivors == result.champion_ids
    # Exactly five identities and exactly five final win rates, and every one
    # of them measured over the final stage alone.
    assert len(result.champion_ids) == len(result.final_win_rates) == 5
    # Higher candidate win rate is better: the strongest scripted candidate
    # leads, so the direction of the ranked quantity is the candidate's.
    rates = [result.final_win_rates[item] for item in result.champion_ids]
    assert rates == sorted(rates, reverse=True)


@pytest.mark.parametrize("race", RACE_EVALUATORS)
def test_each_stage_ranks_only_its_own_games(race):
    candidates = _candidates()
    # Screen through on stage one, then collapse: an implementation that
    # accumulated earlier scores would still carry these candidates.
    strength = {
        candidate_id: CHAMPION_STAGE_1_GAMES
        for candidate_id in candidates[:CHAMPION_STAGE_1_SURVIVORS]
    }
    strength.update({
        candidate_id: 0
        for candidate_id in candidates[CHAMPION_STAGE_1_SURVIVORS:]
    })

    class _CollapsingEvaluator(_ScriptedEvaluator):
        def __call__(self, specs):
            if specs[0].stage_index >= 1:
                # Reverse the ordering from stage two onward.
                self.strength = {
                    candidate_id: index
                    for index, candidate_id in enumerate(candidates)
                }
            return super().__call__(specs)

    result = race(
        candidate_ids=candidates,
        bank_slots=_slots(candidates),
        play_games=_CollapsingEvaluator(strength),
        base_seed=3,
        event_index=1,
    )
    # Stage one kept the first 40; from stage two the strongest of those are
    # the highest-indexed, so the final five are the tail of that group.
    assert result.champion_ids == tuple(reversed(
        candidates[
            CHAMPION_STAGE_1_SURVIVORS - CHAMPION_FINAL_SURVIVORS:
            CHAMPION_STAGE_1_SURVIVORS
        ]
    ))


def test_exact_ranking_ties_resolve_by_smaller_opponent_id():
    wins = {"snapshot:b": 10, "snapshot:a": 10, "snapshot:c": 11}
    assert rank_stage_candidates(wins) == (
        "snapshot:c",
        "snapshot:a",
        "snapshot:b",
    )
    # Every candidate tied: the order is the canonical ID order, not insertion.
    tied = {"snapshot:z": 5, "snapshot:m": 5, "snapshot:a": 5}
    assert rank_stage_candidates(tied) == (
        "snapshot:a",
        "snapshot:m",
        "snapshot:z",
    )


# The tally and its rejections are target-independent by construction; the
# parametrization records that rather than leaving it implied.
RACE_BUCKETS = [HEURISTIC_CHAMPION, CHAMPION_VS_LEARNER_BUCKET]


def _final_stage_specs(bucket_name, *, base_seed=5):
    from training.rl.champion_evaluation import CHAMPION_TARGET_KIND_BY_BUCKET

    candidates = _candidates(CHAMPION_STAGE_3_SURVIVORS)
    return build_stage_specs(
        candidates,
        _slots(candidates),
        base_seed=base_seed,
        seed_namespace=bucket_name,
        target_kind=CHAMPION_TARGET_KIND_BY_BUCKET[bucket_name],
        event_index=0,
        stage_index=3,
    )


@pytest.mark.parametrize("bucket_name", RACE_BUCKETS)
def test_stage_results_are_independent_of_arrival_order(bucket_name):
    candidates = _candidates(CHAMPION_STAGE_3_SURVIVORS)
    specs = _final_stage_specs(bucket_name)
    evaluator = _ScriptedEvaluator(_descending_strength(candidates))
    results = evaluator(specs)
    ordered = tally_stage_results(specs, results)
    # A different worker count changes only completion order.
    shuffled = list(reversed(results))
    assert tally_stage_results(specs, shuffled) == ordered
    assert sum(ordered.values()) > 0


@pytest.mark.parametrize("bucket_name", RACE_BUCKETS)
def test_a_stage_rejects_missing_duplicate_and_non_binary_results(bucket_name):
    candidates = _candidates(CHAMPION_STAGE_3_SURVIVORS)
    specs = _final_stage_specs(bucket_name)
    evaluator = _ScriptedEvaluator(_descending_strength(candidates))
    results = evaluator(specs)

    with pytest.raises(ValueError, match="did not complete every stage game"):
        tally_stage_results(specs, results[:-1])
    with pytest.raises(ValueError, match="duplicate or unexpected"):
        tally_stage_results(specs, results + [results[0]])

    invalid = [dict(item) for item in results]
    invalid[0]["winner"] = None
    with pytest.raises(ValueError, match="invalid winner"):
        tally_stage_results(specs, invalid)

    moved = [dict(item) for item in results]
    moved[0]["candidate_position"] = 1 - moved[0]["candidate_position"]
    with pytest.raises(ValueError, match="changed its assigned seat"):
        tally_stage_results(specs, moved)


def test_an_event_requires_a_complete_non_overlapping_candidate_batch():
    candidates = _candidates(49)
    with pytest.raises(ValueError, match="exactly 50 candidates"):
        evaluate_champion_vs_heuristic(
            candidate_ids=candidates,
            bank_slots=_slots(candidates),
            play_games=_ScriptedEvaluator(_descending_strength(candidates)),
            base_seed=1,
            event_index=0,
        )
    repeated = _candidates(49) + ("snapshot:0000000000",)
    with pytest.raises(ValueError, match="repeats an identity"):
        evaluate_champion_vs_heuristic(
            candidate_ids=repeated,
            bank_slots=_slots(_candidates()),
            play_games=_ScriptedEvaluator(_descending_strength(_candidates())),
            base_seed=1,
            event_index=0,
        )


def test_a_candidate_without_active_weights_is_rejected_before_any_game():
    candidates = _candidates()
    slots = _slots(candidates)
    slots[candidates[7]] = None
    with pytest.raises(ValueError, match="no active weights"):
        build_stage_specs(
            candidates,
            slots,
            base_seed=1,
        seed_namespace=HEURISTIC_CHAMPION,
        target_kind=CHAMPION_TARGET_HEURISTIC,
            event_index=0,
            stage_index=0,
        )
    del slots[candidates[7]]
    with pytest.raises(KeyError, match="no policy bank slot"):
        build_stage_specs(
            candidates,
            slots,
            base_seed=1,
        seed_namespace=HEURISTIC_CHAMPION,
        target_kind=CHAMPION_TARGET_HEURISTIC,
            event_index=0,
            stage_index=0,
        )


class _ChampionPoolFixture:
    """A real pool holding a full pending candidate batch."""

    def __init__(self, buckets, candidate_count):
        from agents.rl_nn import PolicyNetwork
        from training.rl.pool import (
            OpponentPool,
            SharedPolicyBank,
            unique_neural_capacity,
        )

        self.network = PolicyNetwork(
            input_size=8,
            hidden1_size=6,
            hidden2_size=4,
            output_size=5,
            random_seed=1,
            device="cpu",
        )
        self.bank = SharedPolicyBank(
            self.network,
            unique_neural_capacity(buckets),
        )
        self.pool = OpponentPool(
            self.bank,
            selected_buckets=buckets,
            initial_network=self.network,
        )
        for iteration in range(1, candidate_count + 1):
            self.pool.consider_updated_policy(
                self.network,
                iteration=iteration,
                completed_games=iteration * 2000,
                has_samples=True,
            )

    def close(self):
        self.bank.close()


@pytest.fixture(name="champion_pool")
def _champion_pool_fixture():
    fixture = _ChampionPoolFixture(
        ("heuristic", "recent", "champion_vs_heuristic"),
        CHAMPION_CANDIDATE_BATCH_SIZE,
    )
    try:
        yield fixture.pool
    finally:
        fixture.close()


def test_the_racing_result_feeds_pool_admission_directly(champion_pool):
    pool = champion_pool
    candidates = pool.champion_pending_candidate_ids(HEURISTIC_CHAMPION)
    assert pool.champion_candidate_batch_is_ready(HEURISTIC_CHAMPION)
    result = evaluate_champion_vs_heuristic(
        candidate_ids=candidates,
        bank_slots={
            candidate_id: pool.bank_slot(candidate_id)
            for candidate_id in candidates
        },
        play_games=_ScriptedEvaluator(_descending_strength(candidates)),
        base_seed=99,
        event_index=pool.champion_completed_event_count(HEURISTIC_CHAMPION),
    )
    summary = pool.apply_champion_vs_heuristic_result(
        result.champion_ids,
        result.final_win_rates,
    )
    assert summary["admitted"] == result.champion_ids
    assert pool.bucket_members("champion_vs_heuristic") == result.champion_ids
    assert pool.heuristic_champion_win_rates() == dict(result.final_win_rates)
    assert pool.champion_pending_candidate_ids(HEURISTIC_CHAMPION) == ()
    assert pool.champion_completed_event_count(HEURISTIC_CHAMPION) == 1


# ---------------------------------------------------------------------------
# Stage 7: the training-loop trigger.
#
# ``_run_champion_evaluation`` is the only bridge between the racing algorithm
# and the training loop. These tests drive it against a real pool and a stub
# runner, so the wiring is verified without playing 100,000 games.
# ---------------------------------------------------------------------------


class _StubReporter:
    """Record what the iteration would have printed."""

    def __init__(self):
        self.fallbacks = []
        self.events = []

    def rollout_fallback(self, iteration, run_info):
        self.fallbacks.append((iteration, run_info))

    def champion_event(self, summary):
        self.events.append(summary)


class _ExplodingTracker:
    """Fail loudly if a racing event ever reads or writes difficulty evidence.

    Champion scores measure strength against a fixed heuristic; matchmaking
    difficulty measures how hard an opponent is for the current learner. They
    are different quantities and must never feed each other.
    """

    def __getattr__(self, name):
        raise AssertionError(
            f"champion evaluation touched performance_tracker.{name}"
        )


class _StubRunner:
    """Serve champion stages from a scripted evaluator, per stage run info."""

    def __init__(self, pool, evaluator, *, fallbacks=()):
        from diagnostics.parallel_runner import ParallelRunInfo

        self.opponent_pool = pool
        self.evaluator = evaluator
        self.performance_tracker = _ExplodingTracker()
        self._fallbacks = list(fallbacks)
        self._run_info_type = ParallelRunInfo
        self.stage_sizes = []
        self.stage_buckets = []
        self.last_runtime_profile = {"sentinel": 1}

    def evaluate_champion_games(self, specs, *, bucket_name=None):
        self.stage_sizes.append(len(specs))
        self.stage_buckets.append(bucket_name)
        fallback_count = (
            self._fallbacks.pop(0) if self._fallbacks else 0
        )
        run_info = self._run_info_type(
            requested_workers=4,
            initial_workers=4,
            final_workers=4 if not fallback_count else 2,
            fallback_count=fallback_count,
            fallback_history=[{"reason": "test"}] * fallback_count,
            attempted_worker_counts=[4],
        )
        return list(self.evaluator(specs)), run_info


def _stub_context(pool, evaluator, *, effective_seed=4242, fallbacks=()):
    from types import SimpleNamespace

    from training.rl.reporting import _new_parallel_summary

    return SimpleNamespace(
        runner=_StubRunner(pool, evaluator, fallbacks=fallbacks),
        reporter=_StubReporter(),
        parallel_summary=_new_parallel_summary(4),
        effective_seed=effective_seed,
        runtime_profile=_RecordingProfile(),
        network=None,
    )


def test_no_event_runs_until_the_candidate_batch_is_complete():
    from training.rl.iteration import _run_champion_evaluation

    fixture = _ChampionPoolFixture(
        ("heuristic", "recent", "champion_vs_heuristic"),
        CHAMPION_CANDIDATE_BATCH_SIZE - 1,
    )
    try:
        context = _stub_context(fixture.pool, _ScriptedEvaluator({}))
        assert not fixture.pool.champion_candidate_batch_is_ready(HEURISTIC_CHAMPION)
        assert _run_champion_evaluation(context, 49, HEURISTIC_CHAMPION) is None
        # The worker pool must not be touched at all on a non-event iteration.
        assert context.runner.stage_sizes == []
        assert context.parallel_summary["attempted_worker_counts"] == []
        assert fixture.pool.champion_completed_event_count(HEURISTIC_CHAMPION) == 0
    finally:
        fixture.close()


def test_a_ready_batch_races_commits_and_reports_one_event(champion_pool):
    from training.rl.iteration import _run_champion_evaluation
    from training.rl.pool import CHAMPION_VS_HEURISTIC_CAPACITY

    candidates = champion_pool.champion_pending_candidate_ids(HEURISTIC_CHAMPION)
    expected_slots = {
        candidate_id: champion_pool.bank_slot(candidate_id)
        for candidate_id in candidates
    }
    context = _stub_context(
        champion_pool,
        _ScriptedEvaluator(_descending_strength(candidates)),
    )
    summary = _run_champion_evaluation(context, 50, HEURISTIC_CHAMPION)

    assert summary is not None
    # One worker-pool run per racing stage, sized by the fixed stage table.
    assert context.runner.stage_sizes == [
        candidates_in_stage * stage.games_per_candidate
        for candidates_in_stage, stage in zip(
            champion_stage_candidate_counts(),
            CHAMPION_RACING_STAGES,
        )
    ]
    assert summary["racing_games"] == CHAMPION_EVALUATION_GAMES
    assert summary["candidates"] == CHAMPION_CANDIDATE_BATCH_SIZE
    assert summary["stage_candidates"] == champion_stage_candidate_counts()
    assert summary["iteration"] == 50
    # The committed count is one-based; the seed panel used the zero-based
    # index that preceded it.
    assert summary["event_index"] == 1
    assert summary["racing_event_index"] == 0
    assert summary["survivors"] == summary["admitted"]
    assert len(summary["admitted"]) == CHAMPION_FINAL_SURVIVORS
    assert summary["evicted"] == ()
    assert summary["membership_count"] == CHAMPION_FINAL_SURVIVORS
    assert summary["capacity"] == CHAMPION_VS_HEURISTIC_CAPACITY

    # The commit is durable and the batch is consumed.
    assert champion_pool.bucket_members("champion_vs_heuristic") == (
        summary["admitted"]
    )
    assert champion_pool.heuristic_champion_win_rates() == summary["win_rates"]
    assert champion_pool.champion_pending_candidate_ids(HEURISTIC_CHAMPION) == ()
    assert champion_pool.champion_completed_event_count(HEURISTIC_CHAMPION) == 1

    # Every champion is active when the event returns, so the performance
    # retention that runs next in the iteration cannot drop one of them.
    active = {record.opponent_id for record in champion_pool.active_opponents()}
    assert set(summary["admitted"]) <= active

    # Every candidate raced from the slot the pool actually holds.
    raced_slots = {
        spec.candidate_id: spec.bank_slot
        for spec in context.runner.evaluator.stages[0]
    }
    assert raced_slots == expected_slots


def test_the_event_seeds_from_the_run_seed_and_the_committed_event_count(
    champion_pool,
):
    """A replayed event must reuse the panels the crashed attempt used."""
    from training.rl.iteration import _run_champion_evaluation

    candidates = champion_pool.champion_pending_candidate_ids(HEURISTIC_CHAMPION)
    evaluator = _ScriptedEvaluator(_descending_strength(candidates))
    context = _stub_context(champion_pool, evaluator, effective_seed=777)
    _run_champion_evaluation(context, 50, HEURISTIC_CHAMPION)
    first_panel = tuple(
        spec.seed for spec in evaluator.stages[0][:CHAMPION_STAGE_1_GAMES]
    )

    assert first_panel == tuple(
        champion_stage_seed(777, seed_namespace=HEURISTIC_CHAMPION, event_index=0, stage_index=0, game_index=index)
        for index in range(CHAMPION_STAGE_1_GAMES)
    )
    # A second event on the same run must not replay the first one's panel.
    assert first_panel != tuple(
        champion_stage_seed(777, seed_namespace=HEURISTIC_CHAMPION, event_index=1, stage_index=0, game_index=index)
        for index in range(CHAMPION_STAGE_1_GAMES)
    )


class _BankReadingRunner(_StubRunner):
    """A stub runner that reports what the workers would actually see.

    Champion games are still scripted, but every stage first copies the bank's
    current-policy region. A worker views that region live, so this is the same
    thing the real ``_WORKER_CURRENT_POLICY`` resolves to.
    """

    def __init__(self, pool, evaluator, bank, **kwargs):
        super().__init__(pool, evaluator, **kwargs)
        self.bank = bank
        self.published_current = []
        self.sync_calls = 0

    def sync_current(self, network):
        self.sync_calls += 1
        self.bank.write_current(network)

    def evaluate_champion_games(self, specs, *, bucket_name=None):
        self.published_current.append(self.bank.read_current())
        return super().evaluate_champion_games(specs, bucket_name=bucket_name)


class _TrackingTracker:
    """A real tracker that records whether anything mutated it."""

    def __init__(self, opponent_ids):
        from training.rl.matchmaking import OpponentPerformanceTracker

        self.tracker = OpponentPerformanceTracker()
        self.tracker.ensure(opponent_ids)

    def snapshot(self):
        import json

        return json.dumps(self.tracker.export_state(), sort_keys=True)

    def difficulty_snapshot(self, opponent_ids):
        return self.tracker.difficulty_snapshot(opponent_ids)


def _learner_context(fixture, evaluator, *, effective_seed=4242):
    from types import SimpleNamespace

    from training.rl.reporting import _new_parallel_summary

    runner = _BankReadingRunner(fixture.pool, evaluator, fixture.bank)
    runner.performance_tracker = _TrackingTracker(
        record.opponent_id for record in fixture.pool.active_opponents()
    )
    return SimpleNamespace(
        runner=runner,
        reporter=_StubReporter(),
        parallel_summary=_new_parallel_summary(4),
        effective_seed=effective_seed,
        runtime_profile=_RecordingProfile(),
        network=None,
    )


class _RecordingProfile:
    def __init__(self):
        self.sections = {}

    def add(self, section, seconds):
        self.sections[section] = self.sections.get(section, 0.0) + seconds


def _distinct_network(seed):
    from agents.rl_nn import PolicyNetwork

    return PolicyNetwork(
        input_size=8,
        hidden1_size=6,
        hidden2_size=4,
        output_size=5,
        random_seed=seed,
        device="cpu",
    )


def test_the_learner_race_targets_the_post_update_learner():
    """The event must face the policy the update produced, not the one it replaced.

    ``run_iteration`` publishes the current learner once, at the top, before the
    rollouts. The champion event runs after PPO has already changed the parent
    network, so without an explicit re-publish the bank still holds the
    pre-update weights and 100,000 games would rank candidates against a policy
    that no longer exists. Every funnel, seed, seat and ranking test would still
    pass, which is why this one asserts the actual shared-bank weights.
    """
    import numpy as np

    from training.rl.iteration import _run_champion_evaluation
    from training.rl.pool import CHAMPION_VS_LEARNER_BUCKET

    fixture = _ChampionPoolFixture(
        ("heuristic", "recent", "champion_vs_learner"),
        CHAMPION_CANDIDATE_BATCH_SIZE,
    )
    try:
        pre_update = _distinct_network(11)
        post_update = _distinct_network(22)
        assert not np.array_equal(
            np.asarray(pre_update.W1), np.asarray(post_update.W1)
        )

        # What run_iteration published at the top of the iteration, before the
        # rollouts and before the update.
        fixture.bank.write_current(pre_update)

        candidates = fixture.pool.champion_pending_candidate_ids(
            CHAMPION_VS_LEARNER_BUCKET
        )
        context = _learner_context(
            fixture,
            _ScriptedEvaluator(_descending_strength(candidates)),
        )
        # The update has happened: the parent network is the post-update one.
        context.network = post_update

        summary = _run_champion_evaluation(
            context,
            50,
            CHAMPION_VS_LEARNER_BUCKET,
        )
        assert summary is not None

        # Every stage of the event faced the post-update weights, and the same
        # ones throughout: the target may not move between stages either.
        assert len(context.runner.published_current) == len(
            CHAMPION_RACING_STAGES
        )
        for seen in context.runner.published_current:
            np.testing.assert_array_equal(
                seen["W1"], np.asarray(post_update.W1)
            )
        assert not np.array_equal(
            context.runner.published_current[0]["W1"],
            np.asarray(pre_update.W1),
        )
        # And the region still holds them when the event ends, so nothing
        # rewrote the target mid-event.
        np.testing.assert_array_equal(
            fixture.bank.read_current()["W1"], np.asarray(post_update.W1)
        )
    finally:
        fixture.close()


def test_a_learner_event_reads_difficulty_without_disturbing_the_tracker():
    """Retention reads the tracker; the race must leave no trace in it."""
    from training.rl.iteration import _run_champion_evaluation
    from training.rl.pool import CHAMPION_VS_LEARNER_BUCKET

    fixture = _ChampionPoolFixture(
        ("heuristic", "recent", "champion_vs_learner"),
        CHAMPION_CANDIDATE_BATCH_SIZE,
    )
    try:
        fixture.bank.write_current(fixture.network)
        candidates = fixture.pool.champion_pending_candidate_ids(
            CHAMPION_VS_LEARNER_BUCKET
        )
        context = _learner_context(
            fixture,
            _ScriptedEvaluator(_descending_strength(candidates)),
        )
        context.network = fixture.network
        before = context.runner.performance_tracker.snapshot()

        _run_champion_evaluation(context, 50, CHAMPION_VS_LEARNER_BUCKET)

        # Byte-identical: neither the 100,000 racing games nor the retention
        # read may add evidence or create a row.
        assert context.runner.performance_tracker.snapshot() == before
    finally:
        fixture.close()


def test_racing_stages_are_accounted_for_in_the_parallel_summary(champion_pool):
    """Racing worker runs are visible, and never counted as rollout batches."""
    from training.rl.iteration import _run_champion_evaluation

    candidates = champion_pool.champion_pending_candidate_ids(HEURISTIC_CHAMPION)
    context = _stub_context(
        champion_pool,
        _ScriptedEvaluator(_descending_strength(candidates)),
        fallbacks=(0, 1, 0, 0),
    )
    _run_champion_evaluation(context, 50, HEURISTIC_CHAMPION)

    summary = context.parallel_summary
    # The phase is named for the target, so an iteration that runs both events
    # reports two separate 100,000-game costs instead of one opaque 200,000.
    assert summary["champion_vs_heuristic_evaluation_batches"] == len(
        CHAMPION_RACING_STAGES
    )
    assert "champion_evaluation_batches" not in summary
    assert summary["rollout_batches"] == 0
    assert summary["fallback_count"] == 1
    assert [item["rl_phase"] for item in summary["fallback_history"]] == [
        "champion_vs_heuristic_evaluation"
    ]
    assert [item["iteration"] for item in summary["fallback_history"]] == [50]
    assert len(context.reporter.fallbacks) == 1
