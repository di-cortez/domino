"""Deterministic champion racing for both champion opponent buckets.

This module owns racing mechanics only: the stage table, the common seed panels,
seat balancing, outcome validation, and deterministic ranking. It never touches
bucket membership, GPI counters, PPO buffers, or learner-difficulty evidence.
Admission of the surviving five belongs to the pool's per-bucket commit methods.

One funnel serves both targets. ``champion_vs_heuristic`` races candidates
against the fixed ``StrategicAgent``; ``champion_vs_learner`` races them against
the frozen post-update learner. The mechanics are identical and deliberately
share an implementation; what differs is the target, the seed namespace, and
what the caller may do with the returned win rates. A heuristic score stays
comparable across events because its target never moves, so the pool stores it.
A learner score does not, so it is an event result and nothing more.

Execution is injected. The evaluators receive a ``play_games`` callable so the
racing algorithm can be tested without a worker pool, and so the CPU-only pool
plumbing stays in ``training/rl/parallel.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

from training.rl.pool import (
    CHAMPION_BUCKET_NAMES,
    CHAMPION_CANDIDATE_BATCH_SIZE,
    CHAMPION_FINAL_SURVIVORS,
    CHAMPION_VS_HEURISTIC_BUCKET,
    CHAMPION_VS_LEARNER_BUCKET,
)
from training.utils.seeding import stable_seed
from middleware.rulesets import DEFAULT_RULESET_NAME


CHAMPION_EVALUATION_POLICY_VERSION = 2
CHAMPION_STAGE_1_GAMES = 500
CHAMPION_STAGE_1_SURVIVORS = 40
CHAMPION_STAGE_2_GAMES = 500
CHAMPION_STAGE_2_SURVIVORS = 30
CHAMPION_STAGE_3_GAMES = 500
CHAMPION_STAGE_3_SURVIVORS = 20
CHAMPION_FINAL_GAMES = 2000
# Champion selection is an evaluation, not an on-policy rollout, so the frozen
# candidate plays its highest-probability legal action rather than sampling.
# Both seats use this mode when the target is itself a policy: the race is a
# ranking diagnostic and must not add action-sampling noise to it.
CHAMPION_EVALUATION_MODE = "evaluation"
# What a candidate plays against. The execution layer reads this off the spec;
# it is never inferred from the bank slot, which addresses the candidate alone.
CHAMPION_TARGET_HEURISTIC = "heuristic"
CHAMPION_TARGET_CURRENT_LEARNER = "current_learner"
CHAMPION_TARGET_KINDS = (
    CHAMPION_TARGET_HEURISTIC,
    CHAMPION_TARGET_CURRENT_LEARNER,
)
# The seed namespace of each bucket is its own name, so two events with the same
# integer coordinates never share a panel. Kept as an explicit mapping rather
# than a fallback default: a namespace that silently defaults to the heuristic
# one would couple two different evaluation targets to the same deals.
CHAMPION_SEED_NAMESPACE_BY_BUCKET = {
    CHAMPION_VS_HEURISTIC_BUCKET: CHAMPION_VS_HEURISTIC_BUCKET,
    CHAMPION_VS_LEARNER_BUCKET: CHAMPION_VS_LEARNER_BUCKET,
}
CHAMPION_TARGET_KIND_BY_BUCKET = {
    CHAMPION_VS_HEURISTIC_BUCKET: CHAMPION_TARGET_HEURISTIC,
    CHAMPION_VS_LEARNER_BUCKET: CHAMPION_TARGET_CURRENT_LEARNER,
}


@dataclass(frozen=True)
class ChampionStage:
    """One screening round of the racing event."""

    games_per_candidate: int
    survivors: int


CHAMPION_RACING_STAGES = (
    ChampionStage(CHAMPION_STAGE_1_GAMES, CHAMPION_STAGE_1_SURVIVORS),
    ChampionStage(CHAMPION_STAGE_2_GAMES, CHAMPION_STAGE_2_SURVIVORS),
    ChampionStage(CHAMPION_STAGE_3_GAMES, CHAMPION_STAGE_3_SURVIVORS),
    ChampionStage(CHAMPION_FINAL_GAMES, CHAMPION_FINAL_SURVIVORS),
)


def champion_stage_candidate_counts():
    """Return how many candidates enter each stage, newest survivors first."""
    counts = [CHAMPION_CANDIDATE_BATCH_SIZE]
    for stage in CHAMPION_RACING_STAGES[:-1]:
        counts.append(stage.survivors)
    return tuple(counts)


def champion_evaluation_game_count():
    """Return the exact number of games one complete racing event plays."""
    return sum(
        candidates * stage.games_per_candidate
        for candidates, stage in zip(
            champion_stage_candidate_counts(),
            CHAMPION_RACING_STAGES,
        )
    )


def _validate_stage_table():
    """Fail at import if the fixed racing policy is internally inconsistent."""
    counts = champion_stage_candidate_counts()
    previous = CHAMPION_CANDIDATE_BATCH_SIZE
    for candidates, stage in zip(counts, CHAMPION_RACING_STAGES):
        if candidates != previous:
            raise ValueError("Champion stage candidate counts are inconsistent")
        if not 0 < stage.survivors < candidates:
            raise ValueError(
                "Each champion stage must eliminate at least one candidate"
            )
        # Exact seat balance is only possible with an even game count.
        if stage.games_per_candidate % 2:
            raise ValueError(
                "Champion stage games must be even for exact seat balance"
            )
        previous = stage.survivors
    if previous != CHAMPION_FINAL_SURVIVORS:
        raise ValueError("The final champion stage must produce the survivors")


_validate_stage_table()
CHAMPION_EVALUATION_GAMES = champion_evaluation_game_count()


def champion_evaluation_policy_manifest(selected_buckets=()):
    """Return the fixed internal racing policy for provenance and reporting.

    Both target blocks are always published, because the policy is fixed
    whether or not a given run uses it and a stored manifest should stay
    comparable across runs. ``selected_targets`` records which of them this run
    actually raced, so a reader of one metrics header alone can tell the two
    apart without cross-referencing the pool manifest.
    """
    selected = set(selected_buckets)
    return {
        "policy_version": CHAMPION_EVALUATION_POLICY_VERSION,
        "selected_targets": [
            name for name in CHAMPION_BUCKET_NAMES if name in selected
        ],
        "candidate_batch_size": CHAMPION_CANDIDATE_BATCH_SIZE,
        "stages": [
            {
                "games_per_candidate": stage.games_per_candidate,
                "survivors": stage.survivors,
                "candidates": candidates,
            }
            for candidates, stage in zip(
                champion_stage_candidate_counts(),
                CHAMPION_RACING_STAGES,
            )
        ],
        "total_games": CHAMPION_EVALUATION_GAMES,
        "seed_excludes_candidate_identity": True,
        "seat_assignment": "game_index_modulo_two",
        "stage_scoring": "current_stage_games_only",
        "ranking_tie_breaking": "more_wins_then_smaller_opponent_id",
        "candidate_action_mode": CHAMPION_EVALUATION_MODE,
        "counts_toward_gpi": False,
        # Everything above is shared by both buckets. Everything below is what
        # makes an event against one target incomparable with the other.
        "targets": {
            CHAMPION_VS_HEURISTIC_BUCKET: {
                "opponent_kind": "strategic_agent",
                "seed_namespace": CHAMPION_SEED_NAMESPACE_BY_BUCKET[
                    CHAMPION_VS_HEURISTIC_BUCKET
                ],
                "target_kind": CHAMPION_TARGET_HEURISTIC,
                "target_action_mode": None,
                "win_rates_are_durable": True,
            },
            CHAMPION_VS_LEARNER_BUCKET: {
                "opponent_kind": "frozen_post_update_current_learner",
                "seed_namespace": CHAMPION_SEED_NAMESPACE_BY_BUCKET[
                    CHAMPION_VS_LEARNER_BUCKET
                ],
                "target_kind": CHAMPION_TARGET_CURRENT_LEARNER,
                "target_action_mode": CHAMPION_EVALUATION_MODE,
                # The target of an old event is a historical policy, so the
                # rate it produced describes nothing about the present.
                "win_rates_are_durable": False,
            },
        },
    }


@dataclass(frozen=True)
class ChampionGameSpec:
    """One immutable evaluation game, executed by a CPU-only worker."""

    stage_index: int
    candidate_id: str
    # The candidate's opponent bank slot, always. The current learner is not
    # addressed by slot: it lives in the bank's separate current-policy region,
    # and opponent slot 0 is the first opponent, not that region.
    bank_slot: int
    target_kind: str
    game_index: int
    seed: int
    candidate_position: int
    # Unique inside one stage. ``game_index`` repeats across candidates because
    # they share a panel, so the execution layer needs its own key to retain
    # completed work across a worker fallback.
    sequence: int


@dataclass(frozen=True)
class ChampionStageSummary:
    """Transient per-stage result kept only for reporting."""

    stage_index: int
    candidates: int
    games_per_candidate: int
    survivors: tuple[str, ...]
    best_win_rate: float
    worst_win_rate: float


@dataclass(frozen=True)
class ChampionEvaluationResult:
    """Final five champions plus the evidence needed to report the event.

    ``final_win_rates`` is deliberately not named for either target. It is the
    candidate's win rate over the final stage against whatever this bucket
    races, and whether it may become durable state is the caller's decision,
    not a property of the number.
    """

    bucket_name: str
    event_index: int
    champion_ids: tuple[str, ...]
    final_win_rates: dict[str, float]
    total_games: int
    elapsed_seconds: float
    stage_summaries: tuple[ChampionStageSummary, ...]


def champion_stage_seed(
    base_seed,
    *,
    seed_namespace,
    event_index,
    stage_index,
    game_index,
):
    """Return the common panel seed shared by every candidate in one stage.

    The candidate identity is deliberately absent. All candidates in a stage
    face the identical sequence of deals, so the race compares play rather than
    luck. Every stage and every event gets a fresh panel, so screening luck is
    never replayed by the stage that follows it.

    ``seed_namespace`` has no default on purpose. The two champion buckets run
    their events on the same integer coordinates, so a namespace that could be
    omitted would eventually let one bucket's event silently inherit the other's
    deals -- coupling two different evaluation targets to one random panel and
    quietly destroying the independence the common-panel design is for.
    """
    if seed_namespace not in CHAMPION_SEED_NAMESPACE_BY_BUCKET.values():
        raise ValueError(
            f"Unknown champion seed namespace: {seed_namespace!r}"
        )
    return stable_seed(
        base_seed,
        str(seed_namespace),
        int(event_index),
        int(stage_index),
        int(game_index),
    )


def champion_seat_position(game_index):
    """Return which seat the candidate takes for one panel game.

    Deterministic and independent of the RNG stream, matching the diagnostics
    contract. The RL rollout path instead draws its seat from the seeded stream,
    which would couple the seat to the deal and break the exact 50/50 split.
    """
    return int(game_index) % 2


def build_stage_specs(
    candidate_ids,
    bank_slots,
    *,
    base_seed,
    seed_namespace,
    target_kind,
    event_index,
    stage_index,
):
    """Return every game of one stage in deterministic candidate/game order."""
    if target_kind not in CHAMPION_TARGET_KINDS:
        raise ValueError(f"Unknown champion target kind: {target_kind!r}")
    stage = CHAMPION_RACING_STAGES[stage_index]
    specs = []
    for candidate_id in candidate_ids:
        try:
            bank_slot = bank_slots[candidate_id]
        except KeyError as exc:
            raise KeyError(
                f"Champion candidate {candidate_id!r} has no policy bank slot"
            ) from exc
        if bank_slot is None:
            raise ValueError(
                f"Champion candidate {candidate_id!r} has no active weights"
            )
        for game_index in range(stage.games_per_candidate):
            specs.append(ChampionGameSpec(
                stage_index=int(stage_index),
                candidate_id=candidate_id,
                bank_slot=int(bank_slot),
                target_kind=str(target_kind),
                game_index=game_index,
                seed=champion_stage_seed(
                    base_seed,
                    seed_namespace=seed_namespace,
                    event_index=event_index,
                    stage_index=stage_index,
                    game_index=game_index,
                ),
                candidate_position=champion_seat_position(game_index),
                sequence=len(specs),
            ))
    return tuple(specs)


def tally_stage_results(specs, results):
    """Validate one stage's returned games and return wins per candidate.

    Results are matched by identity and game index, never by arrival order, so
    the tally is independent of worker count and scheduling.
    """
    expected = {}
    for spec in specs:
        key = (spec.candidate_id, spec.game_index)
        if key in expected:
            raise ValueError("Champion stage requested a duplicate game")
        expected[key] = spec
    wins = {spec.candidate_id: 0 for spec in specs}
    losses = {spec.candidate_id: 0 for spec in specs}
    seats = {spec.candidate_id: [0, 0] for spec in specs}
    seen = set()
    for result in results:
        key = (result["candidate_id"], int(result["game_index"]))
        if key in seen or key not in expected:
            raise ValueError(
                "Champion evaluation returned a duplicate or unexpected game"
            )
        seen.add(key)
        spec = expected[key]
        winner = result["winner"]
        if winner not in (0, 1):
            raise ValueError(
                f"Champion evaluation game {key} returned invalid winner "
                f"{winner!r}; game-result draws do not exist in this ruleset"
            )
        if int(result["candidate_position"]) != spec.candidate_position:
            raise ValueError(
                f"Champion evaluation game {key} changed its assigned seat"
            )
        seats[spec.candidate_id][spec.candidate_position] += 1
        if winner == spec.candidate_position:
            wins[spec.candidate_id] += 1
        else:
            losses[spec.candidate_id] += 1
    if seen != set(expected):
        raise ValueError("Champion evaluation did not complete every stage game")
    stage = CHAMPION_RACING_STAGES[specs[0].stage_index]
    half = stage.games_per_candidate // 2
    for candidate_id, counts in seats.items():
        if counts != [half, half]:
            raise ValueError(
                f"Champion candidate {candidate_id!r} played {counts} seat "
                f"games; exactly {half} in each seat are required"
            )
        if wins[candidate_id] + losses[candidate_id] != (
            stage.games_per_candidate
        ):
            raise ValueError(
                f"Champion candidate {candidate_id!r} has inconsistent "
                "win/loss totals for this stage"
            )
    return wins


def rank_stage_candidates(wins):
    """Return candidates strongest first, breaking exact ties by opponent ID.

    The tie-break has no statistical meaning. It exists so a fixed-seed run is
    reproducible and auditable. Ranking by wins is equivalent to ranking by win
    rate because every candidate in a stage shares the denominator.
    """
    return tuple(sorted(wins, key=lambda item: (-wins[item], item)))


def _evaluate_champion_race(
    *,
    bucket_name,
    candidate_ids,
    bank_slots,
    play_games,
    base_seed,
    event_index,
):
    """Race one complete candidate batch and return the surviving champions.

    The single funnel behind both champion buckets. Each stage ranks using only
    the games it just played: earlier stage scores are discarded, so a candidate
    that screened through on a lucky panel has to prove itself again on the next
    one. Only the final stage's win rate reaches the result.

    The bucket name selects the seed namespace and the target, which is why the
    two buckets cannot accidentally share a panel or a target: neither has a
    default a caller could fall through.
    """
    try:
        seed_namespace = CHAMPION_SEED_NAMESPACE_BY_BUCKET[bucket_name]
        target_kind = CHAMPION_TARGET_KIND_BY_BUCKET[bucket_name]
    except KeyError as exc:
        raise ValueError(
            f"{bucket_name!r} is not a champion bucket"
        ) from exc

    candidates = tuple(candidate_ids)
    if len(candidates) != CHAMPION_CANDIDATE_BATCH_SIZE:
        raise ValueError(
            "A racing event needs exactly "
            f"{CHAMPION_CANDIDATE_BATCH_SIZE} candidates, got {len(candidates)}"
        )
    if len(set(candidates)) != len(candidates):
        raise ValueError("Champion candidate batch repeats an identity")

    started = time.perf_counter()
    survivors = candidates
    total_games = 0
    summaries = []
    wins = {}
    for stage_index, stage in enumerate(CHAMPION_RACING_STAGES):
        specs = build_stage_specs(
            survivors,
            bank_slots,
            base_seed=base_seed,
            seed_namespace=seed_namespace,
            target_kind=target_kind,
            event_index=event_index,
            stage_index=stage_index,
        )
        wins = tally_stage_results(specs, play_games(specs))
        total_games += len(specs)
        ranked = rank_stage_candidates(wins)
        survivors = ranked[:stage.survivors]
        summaries.append(ChampionStageSummary(
            stage_index=stage_index,
            candidates=len(ranked),
            games_per_candidate=stage.games_per_candidate,
            survivors=survivors,
            best_win_rate=wins[ranked[0]] / stage.games_per_candidate,
            worst_win_rate=wins[ranked[-1]] / stage.games_per_candidate,
        ))
    if total_games != CHAMPION_EVALUATION_GAMES:
        raise RuntimeError(
            f"Champion evaluation played {total_games} games, expected "
            f"{CHAMPION_EVALUATION_GAMES}"
        )
    return ChampionEvaluationResult(
        bucket_name=str(bucket_name),
        event_index=int(event_index),
        champion_ids=survivors,
        # Only the final stage's rate survives; every earlier score is gone.
        final_win_rates={
            candidate_id: wins[candidate_id] / CHAMPION_FINAL_GAMES
            for candidate_id in survivors
        },
        total_games=total_games,
        elapsed_seconds=time.perf_counter() - started,
        stage_summaries=tuple(summaries),
    )


def evaluate_champion_vs_heuristic(
    *,
    candidate_ids,
    bank_slots,
    play_games,
    base_seed,
    event_index,
):
    """Race one batch against the fixed heuristic.

    The returned win rates are comparable across events because the target
    never changes, which is why the pool stores them as champion scores.
    """
    return _evaluate_champion_race(
        bucket_name=CHAMPION_VS_HEURISTIC_BUCKET,
        candidate_ids=candidate_ids,
        bank_slots=bank_slots,
        play_games=play_games,
        base_seed=base_seed,
        event_index=event_index,
    )


def evaluate_champion_vs_learner(
    *,
    candidate_ids,
    bank_slots,
    play_games,
    base_seed,
    event_index,
):
    """Race one batch against the frozen post-update current learner.

    The caller must publish the post-update learner into the shared bank's
    current-policy region before the first stage, and must not write to it again
    until the event is over: every stage of one event has to face exactly the
    same target weights.

    The returned win rates are an event result only. They are measured against a
    policy that will have moved on by the next event, so they must never become
    the score that decides which incumbent leaves a full bucket.
    """
    return _evaluate_champion_race(
        bucket_name=CHAMPION_VS_LEARNER_BUCKET,
        candidate_ids=candidate_ids,
        bank_slots=bank_slots,
        play_games=play_games,
        base_seed=base_seed,
        event_index=event_index,
    )


def play_champion_game(
    policy,
    candidate_position,
    *,
    target_kind=CHAMPION_TARGET_HEURISTIC,
    current_learner_policy=None,
    ruleset_name=DEFAULT_RULESET_NAME,
    use_opponent_suit_features=True,
):
    """Play one frozen candidate against this bucket's target, return a winner.

    The game loop is ``diagnostics.gameplay.play_game``, the same deterministic
    evaluation path the pairwise diagnostics use. Only the worker plumbing is
    new: that module's parallel runner loads agents from weight files, which
    cannot serve 140 in-memory candidates without one temporary file or one pool
    restart per candidate.

    Against the current learner both seats are neural and both play in
    evaluation mode. The two agents are always separate objects, even when the
    candidate happens to hold the same weights as the learner: an ``RLAgent`` is
    mutable per-game state, and sharing one between the seats would let each
    player observe the other's bookkeeping.
    """
    from agents.heuristic_agent import StrategicAgent
    from agents.rl_agent import RLAgent
    from diagnostics.gameplay import play_game

    candidate = RLAgent(
        policy,
        mode=CHAMPION_EVALUATION_MODE,
        ruleset=ruleset_name,
        use_opponent_suit_features=use_opponent_suit_features,
    )
    if target_kind == CHAMPION_TARGET_HEURISTIC:
        target = StrategicAgent(ruleset=ruleset_name)
    elif target_kind == CHAMPION_TARGET_CURRENT_LEARNER:
        if current_learner_policy is None:
            raise ValueError(
                "A champion game against the current learner needs the "
                "current-policy view; the candidate's bank slot addresses the "
                "candidate alone and never the learner"
            )
        target = RLAgent(
            current_learner_policy,
            mode=CHAMPION_EVALUATION_MODE,
            ruleset=ruleset_name,
            use_opponent_suit_features=use_opponent_suit_features,
        )
    else:
        raise ValueError(f"Unknown champion target kind: {target_kind!r}")
    record = play_game(
        candidate,
        target,
        agent_position=int(candidate_position),
        suppress_agent_output=True,
        ruleset=ruleset_name,
    )
    outcome = record["result"]
    if outcome not in ("win", "loss"):
        raise ValueError(
            f"Champion evaluation game returned {outcome!r}; game-result draws "
            "do not exist in this ruleset"
        )
    position = int(candidate_position)
    return position if outcome == "win" else 1 - position
