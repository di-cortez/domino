"""Correctness tests for the selectable policy-gradient baseline.

The contract under test is that the baseline is the *only* term subtracted
from a return, and that advantage normalization only rescales. Together those
keep ``zero`` and ``constant`` observable instead of being silently re-centered
back into ``batch-mean``.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from training.pipeline import _rl_config, parse_args
from training.rl.cli import parse_args as rl_parse_args
from training.rl.cli import training_options_from_args
from training.rl.config import resolve_training_options
from training.rl import baseline as baselines
from training.rl.baseline import BaselineSpec
from training.rl.iteration import _reinforce_policy_update
from training.rl.ppo import PPOBuffer
from training.rl.resume import (
    RLTrainingConfiguration,
    _validate_resume_configuration,
)


REWARDS = (1.0, -1.0, 3.0, 0.0)
CRITIC_VALUE = 0.5


def _sample(index, reward, *, state_size=3, action_size=4):
    legal_mask = np.zeros((action_size, 1), dtype=np.bool_)
    legal_mask[0, 0] = True
    legal_mask[1, 0] = True
    return SimpleNamespace(
        x=np.full((state_size, 1), index / 100.0, dtype=np.float32),
        action_index=index % 2,
        legal_mask=legal_mask,
        old_log_prob=0.0,
        policy_reward=float(reward),
        local_reward=0.0,
        terminal_reward=float(reward),
        agent_hand_size=2,
        opponent_hand_size=3,
    )


def _samples(rewards=REWARDS):
    return [_sample(index, reward) for index, reward in enumerate(rewards)]


def _critic_values(rewards=REWARDS):
    return np.full(len(rewards), CRITIC_VALUE, dtype=np.float32)


class _SignalCapturingNetwork:
    """Minimal REINFORCE learner that records the policy signal it receives."""

    def __init__(self, *, use_value_head=False):
        self.xp = np
        self.use_value_head = use_value_head
        self.policy_signal = None
        self.value_returns = None
        self.forward_calls = 0
        self.value_calls = 0

    def forward(self, x_batch, training=False):
        self.forward_calls += 1
        return x_batch

    def predict_values(self, x_batch, training=False):
        self.value_calls += 1
        return np.full((1, x_batch.shape[1]), CRITIC_VALUE, dtype=np.float32)

    def backward_policy_gradient(
        self,
        actions,
        policy_signal,
        *,
        legal_masks,
        entropy_coef,
        value_returns,
        value_coef,
        clip_grad_norm,
    ):
        self.policy_signal = np.asarray(policy_signal, dtype=np.float32).copy()
        self.value_returns = value_returns
        return {"grad_clipped": False}


def _reinforce_signal(
    spec,
    *,
    normalize,
    use_value_head=False,
    lookup_values=None,
):
    network = _SignalCapturingNetwork(use_value_head=use_value_head)
    _reinforce_policy_update(
        network,
        _samples(),
        entropy_coef=0.0,
        normalize_advantages=normalize,
        baseline=spec,
        use_value_head=use_value_head,
        value_coef=0.5,
        lookup_values=lookup_values,
    )
    return network, network.policy_signal.reshape(-1)


# ---------------------------------------------------------------------------
# Baseline definitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, expected",
    [
        (BaselineSpec(kind="zero"), [1.0, -1.0, 3.0, 0.0]),
        (BaselineSpec(kind="constant", constant=2.0), [-1.0, -3.0, 1.0, -2.0]),
        (BaselineSpec(kind="batch-mean"), [0.25, -1.75, 2.25, -0.75]),
        (BaselineSpec(kind="value-head"), [0.5, -1.5, 2.5, -0.5]),
    ],
)
def test_each_baseline_subtracts_its_own_term(spec, expected):
    buffer = PPOBuffer.from_samples(
        _samples(),
        baseline=spec,
        normalize=False,
        old_values=_critic_values(),
    )
    assert np.allclose(buffer.advantages, expected)


def test_the_four_baselines_stay_distinct_under_normalization():
    """Normalization must not collapse zero and constant into batch-mean."""
    signals = {}
    for spec in (
        BaselineSpec(kind="zero"),
        BaselineSpec(kind="constant", constant=2.0),
        BaselineSpec(kind="batch-mean"),
        BaselineSpec(kind="value-head"),
    ):
        buffer = PPOBuffer.from_samples(
            _samples(),
            baseline=spec,
            normalize=True,
            old_values=_critic_values(),
        )
        signals[spec.label] = np.asarray(buffer.advantages)

    labels = sorted(signals)
    for index, left in enumerate(labels):
        for right in labels[index + 1:]:
            assert not np.allclose(signals[left], signals[right]), (
                f"{left} and {right} produced the same advantages"
            )
    # Only batch-mean is centered; a re-centering normalizer would zero them all.
    assert abs(float(signals["batch-mean"].mean())) < 1e-6
    assert float(signals["zero"].mean()) > 0.0
    assert float(signals["constant(2)"].mean()) < 0.0


def test_normalization_only_rescales_and_leaves_the_center_alone():
    buffer = PPOBuffer.from_samples(
        _samples(),
        baseline=BaselineSpec(kind="zero"),
        normalize=True,
    )
    rewards = np.asarray(REWARDS, dtype=np.float32)
    assert np.allclose(buffer.advantages, rewards / rewards.std(), atol=1e-6)


def test_the_default_baseline_reproduces_the_original_normalization():
    """The pre-flag default must stay numerically identical."""
    rewards = np.asarray(REWARDS, dtype=np.float32)
    original = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    buffer = PPOBuffer.from_samples(_samples(), normalize=True)
    assert buffer.baseline.kind == baselines.BATCH_MEAN
    assert np.allclose(buffer.advantages, original, atol=1e-6)


def test_a_zero_variance_iteration_keeps_its_constant_baseline_offset():
    """Zero variance must not be an excuse to re-center to zero."""
    buffer = PPOBuffer.from_samples(
        _samples([2.0] * 4),
        baseline=BaselineSpec(kind="constant", constant=0.5),
        normalize=True,
    )
    assert buffer.advantage_std_zero
    assert np.allclose(buffer.advantages, 1.5)


def test_the_value_head_baseline_needs_critic_predictions():
    with pytest.raises(ValueError, match="one critic prediction per decision"):
        PPOBuffer.from_samples(
            _samples(),
            baseline=BaselineSpec(kind="value-head"),
            old_values=None,
        )


def test_the_buffer_records_the_baseline_it_applied():
    buffer = PPOBuffer.from_samples(
        _samples(),
        baseline=BaselineSpec(kind="constant", constant=2.0),
        normalize=False,
    )
    assert buffer.baseline.label == "constant(2)"
    assert buffer.baseline_mean == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# The single-update REINFORCE path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, expected, use_value_head",
    [
        (BaselineSpec(kind="zero"), [1.0, -1.0, 3.0, 0.0], False),
        (
            BaselineSpec(kind="constant", constant=2.0),
            [-1.0, -3.0, 1.0, -2.0],
            False,
        ),
        (BaselineSpec(kind="batch-mean"), [0.25, -1.75, 2.25, -0.75], False),
        (BaselineSpec(kind="value-head"), [0.5, -1.5, 2.5, -0.5], True),
    ],
)
def test_reinforce_applies_the_same_baselines(spec, expected, use_value_head):
    _network, signal = _reinforce_signal(
        spec,
        normalize=False,
        use_value_head=use_value_head,
    )
    assert np.allclose(signal, expected)


def test_reinforce_normalization_only_rescales():
    _network, signal = _reinforce_signal(
        BaselineSpec(kind="zero"),
        normalize=True,
    )
    rewards = np.asarray(REWARDS, dtype=np.float32)
    assert np.allclose(signal, rewards / rewards.std(), atol=1e-5)


def test_reinforce_subtracts_one_lookup_value_per_decision():
    _network, signal = _reinforce_signal(
        BaselineSpec(kind="lookup-table"),
        normalize=False,
        lookup_values=np.asarray([0.5, -0.5, 1.0, 0.25], dtype=np.float32),
    )

    assert np.allclose(signal, [0.5, -0.5, 2.0, -0.25])


def test_a_non_critic_baseline_still_trains_the_critic():
    """--value-head with another baseline isolates training V from using it."""
    network, signal = _reinforce_signal(
        BaselineSpec(kind="batch-mean"),
        normalize=False,
        use_value_head=True,
    )
    assert np.allclose(signal, [0.25, -1.75, 2.25, -0.75])
    assert network.value_calls == 1
    assert np.allclose(np.asarray(network.value_returns).reshape(-1), REWARDS)


# ---------------------------------------------------------------------------
# Parsing, resolution, and persistence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tokens, kind, constant",
    [
        (["zero"], "zero", 0.0),
        (["constant", "2"], "constant", 2.0),
        (["constant", "-1.5"], "constant", -1.5),
        (["batch_mean"], "batch-mean", 0.0),
        (["batch-mean"], "batch-mean", 0.0),
        (["lookup_table"], "lookup-table", 0.0),
        (["value_head"], "value-head", 0.0),
    ],
)
def test_command_line_tokens_parse_into_one_baseline(tokens, kind, constant):
    spec = baselines.from_tokens(tokens)
    assert spec.kind == kind
    assert spec.constant == pytest.approx(constant)


@pytest.mark.parametrize(
    "tokens, message",
    [
        (["bogus"], "Unknown baseline"),
        (["constant"], "exactly one value"),
        (["constant", "abc"], "needs a number"),
        (["constant", "nan"], "finite number"),
        (["zero", "2"], "takes no value"),
        ([], "needs a kind"),
    ],
)
def test_malformed_baseline_tokens_are_rejected(tokens, message):
    with pytest.raises(ValueError, match=message):
        baselines.from_tokens(tokens)


@pytest.mark.parametrize(
    "use_value_head, normalize, expected",
    [
        (True, True, "value-head"),
        (True, False, "value-head"),
        (False, True, "batch-mean"),
        (False, False, "zero"),
    ],
)
def test_an_unset_baseline_resolves_to_the_previous_behavior(
    use_value_head,
    normalize,
    expected,
):
    spec = baselines.resolve(
        None,
        use_value_head=use_value_head,
        normalize_advantages=normalize,
    )
    assert spec.kind == expected


def test_the_value_head_baseline_requires_the_critic_head():
    with pytest.raises(ValueError, match="needs the critic"):
        baselines.resolve(
            BaselineSpec(kind="value-head"),
            use_value_head=False,
            normalize_advantages=True,
        )


def test_a_baseline_round_trips_through_its_persisted_mapping():
    spec = BaselineSpec(kind="constant", constant=2.5)
    assert BaselineSpec.from_mapping(spec.as_mapping()) == spec
    assert baselines.from_tokens(spec.as_tokens()) == spec


def test_a_run_predating_the_flag_resolves_to_the_baseline_it_used():
    assert baselines.from_run_config({"locked_arguments": {}}) is None
    assert baselines.from_run_config({}) is None
    assert baselines.from_run_config(
        {"locked_arguments": {"baseline": ["constant", "2"]}}
    ) == BaselineSpec(kind="constant", constant=2.0)


def test_a_non_constant_baseline_rejects_a_stray_value():
    with pytest.raises(ValueError, match="takes no value"):
        BaselineSpec(kind="zero", constant=1.0)


# ---------------------------------------------------------------------------
# The flag on both entry points
# ---------------------------------------------------------------------------
#
# ``--baseline`` is declared once, by ``baselines.add_argument``, and reaches
# the canonical pipeline and the standalone RL command through the shared
# ``add_optional_rl_arguments``. Both are exercised here so the two can never
# drift apart.


def _pipeline(argv):
    return parse_args(["small", *argv])


def _standalone(argv):
    return rl_parse_args(list(argv))


ENTRY_POINTS = pytest.mark.parametrize(
    "parse",
    [_pipeline, _standalone],
    ids=["pipeline", "rl-cli"],
)


@ENTRY_POINTS
@pytest.mark.parametrize(
    "argv, expected",
    [
        ([], None),
        (["--baseline", "zero"], ["zero"]),
        (["--baseline", "constant", "2"], ["constant", "2"]),
        (["--baseline", "batch-mean"], ["batch-mean"]),
        (["--baseline", "lookup-table"], ["lookup-table"]),
        (["--baseline", "value-head", "--value-head"], ["value-head"]),
    ],
)
def test_the_flag_stores_json_safe_tokens(parse, argv, expected):
    """Tokens, not the typed object: locked_arguments persists this verbatim."""
    assert parse(argv).baseline == expected


@ENTRY_POINTS
@pytest.mark.parametrize(
    "argv",
    [
        ["--baseline", "value-head"],
        ["--baseline", "bogus"],
        ["--baseline", "constant"],
        ["--baseline", "constant", "abc"],
        ["--baseline", "zero", "2"],
    ],
)
def test_the_flag_rejects_bad_input_at_parse_time(parse, argv):
    with pytest.raises(SystemExit):
        parse(argv)


@ENTRY_POINTS
@pytest.mark.parametrize(
    "argv, expected",
    [
        ([], "batch-mean"),
        (["--baseline", "zero"], "zero"),
        (["--baseline", "constant", "2"], "constant(2)"),
        (["--baseline", "lookup-table"], "lookup-table"),
        (["--baseline", "value-head", "--value-head"], "value-head"),
        (["--value-head"], "value-head"),
        (["--value-head", "--baseline", "zero"], "zero"),
    ],
)
def test_the_flag_reaches_the_resolved_training_options(parse, argv, expected):
    args = parse(argv)
    # The pipeline fills this in from its level; the standalone command has it.
    args.ppo_max_epochs = getattr(args, "ppo_max_epochs", None) or 4
    training, resources, execution = training_options_from_args(args)
    resolved = resolve_training_options(
        training,
        replace(
            resources,
            sl_weights_path="sl.npz",
            rl_weights_path="rl.npz",
        ),
        execution,
    )
    assert resolved.training.baseline.label == expected


# ---------------------------------------------------------------------------
# Resume identity
# ---------------------------------------------------------------------------


_RL_CONFIG = {
    "games_per_iteration": 100,
    "opponent_buckets": ["heuristic", "recent"],
    "difficulty_weight": 0.5,
    "opponent_decision_restarts": False,
    "learning_rate": 0.001,
    "entropy_coef": 0.01,
    "weight_decay": 0.0,
    "dropout_rate": 0.0,
    "log_interval": 1,
    "checkpoint_interval": 1,
    "use_value_head": False,
    "value_coef": 0.5,
    "gamma_f": 1.0,
    "reward_eta": 0.5,
    "gamma_i": 0.90,
    "clip_grad_norm": 1.0,
    "normalize_advantages": True,
    "moving_average_window": 10,
    "worker_memory_reserve_mb": 512,
    "worker_estimated_mb": 256,
    "worker_max_rss_mb": 1024,
}


def _run_config(locked_arguments):
    return {
        "config_hash_version": 4,
        "pipeline_level": "forever",
        "run_name": None,
        "seed": 42,
        "target_rl_games": None,
        "ruleset_version": 1,
        "ruleset_name": "double-six",
        "encoder_size": 161,
        "action_count": 56,
        "network_architecture": [161, 256, 128, 56],
        "algorithm": "ppo_v2_decision_minibatches",
        "supervised_weights_sha256": "0" * 64,
        "configuration_sha256": "1" * 64,
        "ppo_config": {"max_epochs": 4},
        "rl_config": dict(_RL_CONFIG),
        "diagnostic_config": {},
        "locked_arguments": dict(locked_arguments),
    }


def _saved_configuration(locked_arguments):
    return RLTrainingConfiguration.from_run_config(
        _run_config(locked_arguments),
        total_training_games=1000,
        selected_workers=1,
        device="cpu",
    )


def test_the_baseline_is_absent_from_the_immutable_rl_config():
    """A new rl_config member would make every existing run unresumable."""
    assert "baseline" not in _rl_config(parse_args(["small"]))


def test_a_saved_run_restores_the_baseline_it_was_created_with():
    saved = _saved_configuration({"baseline": ["constant", "2"]})

    assert saved.baseline == {"kind": "constant", "constant": 2.0}


def test_a_lookup_run_persists_the_ruleset_artifact_identity():
    saved = _saved_configuration({"baseline": ["lookup-table"]})

    assert saved.baseline == {"kind": "lookup-table", "constant": 0.0}
    assert len(saved.baseline_artifact_sha256) == 64


def test_a_run_created_before_the_flag_still_resumes():
    saved = _saved_configuration({})

    assert saved.baseline is None
    # A checkpoint written before either feature carries neither key.
    legacy = dict(saved.to_dict())
    legacy.pop("baseline")
    legacy.pop("baseline_artifact_sha256")
    restored = RLTrainingConfiguration.from_mapping(legacy)

    assert restored == saved
    _validate_resume_configuration({"configuration": legacy}, saved)


def test_resuming_with_a_different_lookup_artifact_is_rejected():
    requested = _saved_configuration({"baseline": ["lookup-table"]})
    saved = requested.to_dict()
    saved["baseline_artifact_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="baseline_artifact_sha256"):
        _validate_resume_configuration({"configuration": saved}, requested)


def test_resuming_with_a_different_baseline_is_rejected():
    saved = _saved_configuration({"baseline": ["constant", "2"]})
    requested = _saved_configuration({"baseline": ["zero"]})

    with pytest.raises(ValueError, match="baseline"):
        _validate_resume_configuration({"configuration": saved.to_dict()}, requested)
