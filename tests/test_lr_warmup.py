"""Deterministic tests for the KL-gated learning-rate warmup schedule."""

import json

import pytest

from training.rl.lr_warmup import (
    DEFAULT_WARMUP_KL_THRESHOLD,
    WARMUP_TRACE_COLUMNS,
    WarmupSchedule,
    normalize_warmup,
    prepare_warmup_trace,
    replay_metrics,
    warmup_trace_row,
    write_warmup_trace_row,
)
from training.rl.ppo import PPO_STOP_KL

LOW_KL = 0.001
HIGH_KL = 0.02


def _climb(schedule, kl, limit=10_000):
    """Feed one constant KL until the ladder finishes; return promotion iterations."""
    promotions = []
    for iteration in range(1, limit + 1):
        if schedule.observe(kl).promoted:
            promotions.append(iteration)
        if not schedule.active:
            break
    return promotions


def test_default_threshold_is_half_the_enforced_stop_kl():
    # PPO_TARGET_KL is declarative; the gate must follow the limit that binds.
    assert DEFAULT_WARMUP_KL_THRESHOLD == PPO_STOP_KL / 2.0 == 0.0075


def test_ladder_rungs_match_the_specification():
    schedule = WarmupSchedule(0.01)
    rungs = []
    for exponent in range(6, -1, -1):
        schedule.exponent = exponent
        rungs.append(round(schedule.learning_rate / 0.01, 5))
    assert rungs == [0.08779, 0.13169, 0.19753, 0.2963, 0.44444, 0.66667, 1.0]


def test_low_kl_finishes_at_the_theoretical_minimum():
    # 100 hold before the first climb, then five cycles of cooldown + hold.
    schedule = WarmupSchedule(0.01)
    assert _climb(schedule, LOW_KL) == [100, 300, 500, 700, 900, 1100]


def test_last_rung_is_the_nominal_rate_exactly():
    schedule = WarmupSchedule(0.01)
    _climb(schedule, LOW_KL)
    assert schedule.exponent == 0
    assert schedule.learning_rate == 0.01
    assert not schedule.active


def test_high_kl_never_climbs():
    schedule = WarmupSchedule(0.01)
    for _ in range(5_000):
        schedule.observe(HIGH_KL)
    assert schedule.exponent == 6
    assert schedule.active


def test_consecutive_promotions_are_separated_by_hold_plus_cooldown():
    schedule = WarmupSchedule(0.01, hold_iterations=5, cooldown_iterations=10)
    promotions = _climb(schedule, LOW_KL)
    gaps = {later - earlier for earlier, later in zip(promotions, promotions[1:])}
    assert gaps == {15}


def test_one_excursion_above_the_threshold_resets_the_streak():
    # alpha 0 removes smoothing so a single sample moves the filter fully.
    schedule = WarmupSchedule(0.01, hold_iterations=10, ema_alpha=0.0)
    for _ in range(9):
        decision = schedule.observe(LOW_KL)
    assert decision.hold_streak == 9
    assert schedule.observe(HIGH_KL).hold_streak == 0


def test_cooldown_blocks_promotion_but_keeps_measuring():
    schedule = WarmupSchedule(0.01, hold_iterations=3, cooldown_iterations=5)
    for _ in range(3):
        schedule.observe(LOW_KL)
    ema_after_climb = schedule.state_dict()["ema"]
    decision = schedule.observe(HIGH_KL)
    assert decision.cooldown_remaining == 4
    assert not decision.promoted
    # The filter moved on a cooldown sample; freezing it would be the bug.
    assert decision.ema_max_kl > ema_after_climb


def test_ema_is_seeded_with_the_first_observation_not_zero():
    schedule = WarmupSchedule(0.01)
    decision = schedule.observe(0.22)
    assert decision.ema_max_kl == pytest.approx(0.22)


def test_missing_kl_leaves_the_filter_alone_and_breaks_the_streak():
    schedule = WarmupSchedule(0.01, hold_iterations=10)
    for _ in range(5):
        schedule.observe(LOW_KL)
    before = schedule.state_dict()["ema"]
    decision = schedule.observe(None)
    assert decision.ema_max_kl == before
    assert decision.hold_streak == 0


def test_applied_and_next_rates_differ_only_on_a_promotion():
    schedule = WarmupSchedule(0.01, hold_iterations=2, cooldown_iterations=0)
    held = schedule.observe(LOW_KL)
    assert held.applied_learning_rate == held.next_learning_rate
    climbed = schedule.observe(LOW_KL)
    assert climbed.promoted
    assert climbed.next_learning_rate == pytest.approx(
        climbed.applied_learning_rate * 1.5
    )


def test_state_dict_reproduces_every_following_promotion():
    original = WarmupSchedule(0.01)
    for _ in range(250):
        original.observe(LOW_KL)
    restored = WarmupSchedule(0.01)
    restored.load_state_dict(original.state_dict())
    assert restored.learning_rate == original.learning_rate
    assert _climb(original, LOW_KL) == _climb(restored, LOW_KL)


@pytest.mark.parametrize(
    "overrides",
    [
        {"ema_alpha": 1.0},
        {"ema_alpha": -0.1},
        {"factor": 1.0},
        {"hold_iterations": 0},
        {"cooldown_iterations": -1},
        {"kl_threshold": 0.0},
        {"exponent": -1},
    ],
)
def test_invalid_parameters_are_rejected(overrides):
    with pytest.raises(ValueError):
        WarmupSchedule(0.01, **overrides)


def test_non_positive_nominal_rate_is_rejected():
    with pytest.raises(ValueError):
        WarmupSchedule(0.0)


@pytest.mark.parametrize("off", [None, False])
def test_off_spellings_normalize_to_none(off):
    assert normalize_warmup(off) is None


def test_true_normalizes_to_every_default():
    assert normalize_warmup(True) == {
        "exponent": 6,
        "factor": 1.5,
        "ema_alpha": 0.9,
        "kl_threshold": 0.0075,
        "hold_iterations": 100,
        "cooldown_iterations": 100,
    }


def test_partial_mapping_fills_defaults_and_is_idempotent():
    normalized = normalize_warmup({"hold_iterations": 50})
    assert normalized["hold_iterations"] == 50
    assert normalized["exponent"] == 6
    assert normalize_warmup(normalized) == normalized


def test_unknown_warmup_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown warmup parameter"):
        normalize_warmup({"patience": 3})


def test_schedule_builds_from_a_normalized_mapping():
    schedule = WarmupSchedule.from_warmup(0.001, {"exponent": 2})
    assert schedule.learning_rate == pytest.approx(0.001 / 1.5 ** 2)
    with pytest.raises(ValueError):
        WarmupSchedule.from_warmup(0.001, None)


def test_trace_truncates_rows_past_the_resumed_iteration(tmp_path):
    path = tmp_path / "warmup_schedule.jsonl"
    warmup = normalize_warmup(True)
    prepare_warmup_trace(
        path, 0, nominal_learning_rate=0.01, warmup=warmup
    )
    schedule = WarmupSchedule.from_warmup(0.01, warmup)
    with open(path, "a", encoding="utf-8") as stream:
        for iteration in range(1, 8):
            decision = schedule.observe(LOW_KL)
            write_warmup_trace_row(
                stream, warmup_trace_row(iteration, decision, LOW_KL)
            )
    # Resume at iteration 4: rows 5..7 belong to an unfinished future.
    prepare_warmup_trace(
        path, 4, nominal_learning_rate=0.01, warmup=warmup
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header["columns"] == list(WARMUP_TRACE_COLUMNS)
    assert [json.loads(line)[0] for line in lines[1:]] == [1, 2, 3, 4]


def test_replay_reads_max_kl_from_a_metrics_file(tmp_path):
    path = tmp_path / "training_metrics.jsonl"
    columns = ["iteration", "max_approx_kl"]
    header = {
        "columns": columns,
        "metadata": {
            "run_configuration": {"rl_config": {"learning_rate": 0.01}}
        },
    }
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(json.dumps(header) + "\n")
        for iteration in range(1, 3000):
            stream.write(json.dumps([iteration, LOW_KL]) + "\n")
    nominal, events, schedule = replay_metrics(path)
    assert nominal == 0.01
    assert [iteration for iteration, _ in events] == [
        100, 300, 500, 700, 900, 1100
    ]
    assert not schedule.active
