"""Contracts for the numeric ``--baseline`` grammar and the three critic kinds.

Two independent pieces of work meet in one enum. The grammar lets a constant be
spelled as a bare number, and the enum gains two further critics that subtract
exactly what ``value-head`` subtracts but are wired to the policy differently.
Every test here protects one property the experiment comparing them depends on.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from agents.rl_nn import CRITIC_GRADIENT_PREFIX, PolicyNetwork
from training.pipeline import _rl_config, parse_args
from training.rl import baseline as baselines
from training.rl.baseline import (
    BATCH_MEAN,
    CONSTANT,
    CRITIC_KINDS,
    VALUE_HEAD,
    VALUE_HEAD_NO_UP,
    VALUE_HEAD_OWN_NN,
    ZERO,
    BaselineSpec,
    from_tokens,
)
from training.rl.cli import parse_args as rl_parse_args
from training.rl.cli import training_options_from_args


def _pipeline(argv):
    return parse_args(["small", *argv])


def _standalone(argv):
    return rl_parse_args(list(argv))


ENTRY_POINTS = pytest.mark.parametrize(
    "parse",
    [_pipeline, _standalone],
    ids=["pipeline", "rl-cli"],
)


# --------------------------------------------------------------------------
# The numeric grammar
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "value"),
    [("2", 2.0), ("0", 0.0), ("-1.5", -1.5), ("1e3", 1000.0), ("2.5", 2.5)],
)
def test_a_bare_number_is_the_constant_baseline(token, value):
    spec = from_tokens([token])

    assert spec.kind == CONSTANT
    assert spec.constant == value


def test_a_number_is_the_value_and_never_an_index_into_the_kinds():
    """The biggest trap in this grammar, so it gets its own test.

    The kinds are conventionally numbered 0-5 when they are discussed, but that
    numbering never reaches the command line: ``--baseline 2`` asks for the
    constant 2, not for the third kind.
    """
    assert from_tokens(["2"]) == BaselineSpec(kind=CONSTANT, constant=2.0)
    assert from_tokens(["2"]).kind != BATCH_MEAN
    assert from_tokens(["3"]).kind != VALUE_HEAD
    assert from_tokens(["0"]).kind != ZERO


def test_constant_zero_is_kept_distinct_from_the_zero_baseline():
    """They cancel to the same gradient but record different intent.

    Collapsing them would erase what the run asked for from
    ``locked_arguments`` and give the two invocations one configuration hash.
    """
    constant_zero = from_tokens(["0"])
    zero = from_tokens(["zero"])

    assert constant_zero != zero
    assert constant_zero.label != zero.label
    assert constant_zero.as_mapping() != zero.as_mapping()
    # Numerically identical all the same.
    returns = np.array([[1.0, -1.0, 3.0, 0.0]], dtype=np.float32)
    assert np.array_equal(
        baselines.subtract(constant_zero, returns),
        baselines.subtract(zero, returns),
    )


def test_the_legacy_constant_spelling_still_round_trips():
    """Every run already created with a constant must stay resumable.

    ``as_tokens`` has always written ``["constant", "2.0"]`` into
    ``locked_arguments`` and checkpoints, so that spelling has to keep parsing.
    """
    spec = BaselineSpec(kind=CONSTANT, constant=2.0)

    assert from_tokens(["constant", "2"]) == spec
    assert from_tokens(spec.as_tokens()) == spec
    assert BaselineSpec.from_mapping(spec.as_mapping()) == spec


def test_a_number_takes_nothing_after_it():
    """Otherwise ``--baseline 2 small`` silently eats the pipeline level."""
    with pytest.raises(ValueError, match="constant baseline and takes"):
        from_tokens(["2", "small"])


@pytest.mark.parametrize("token", ["nan", "inf", "-inf"])
def test_a_non_finite_constant_is_rejected(token):
    with pytest.raises(ValueError, match="finite number"):
        from_tokens([token])


def test_an_unknown_name_names_the_kinds_and_the_numeric_form():
    with pytest.raises(ValueError, match="bare number"):
        from_tokens(["kind-of-baseline"])


@ENTRY_POINTS
def test_the_numeric_form_reaches_both_entry_points(parse):
    assert parse(["--baseline", "2"]).baseline == ["2"]
    assert parse(["--baseline", "-0.5"]).baseline == ["-0.5"]


def test_a_negative_constant_survives_argparse():
    """Argparse only treats ``-x`` as an option when the parser has one.

    Neither CLI declares a numeric-looking option, so negative constants parse.
    It is a property of the whole parser rather than of this flag, which is why
    it is pinned here.
    """
    for parse in (_pipeline, _standalone):
        spec = BaselineSpec.from_mapping(parse(["--baseline", "-2.5"]).baseline)
        assert spec == BaselineSpec(kind=CONSTANT, constant=-2.5)


@ENTRY_POINTS
def test_a_number_after_the_flag_is_rejected_not_swallowed(parse, capsys):
    with pytest.raises(SystemExit):
        parse(["--baseline", "2", "batch-mean"])
    assert "constant baseline and takes" in capsys.readouterr().err


def test_the_numeric_form_is_locked_into_the_run_identity():
    args = parse_args(["small", "--baseline", "2"])

    assert "baseline" not in _rl_config(args)
    assert args.baseline == ["2"]


# --------------------------------------------------------------------------
# The enum
# --------------------------------------------------------------------------


def test_every_critic_kind_requires_the_value_head():
    assert set(CRITIC_KINDS) == {
        VALUE_HEAD,
        VALUE_HEAD_NO_UP,
        VALUE_HEAD_OWN_NN,
    }
    for kind in CRITIC_KINDS:
        assert BaselineSpec(kind=kind).requires_value_head
    for kind in (ZERO, CONSTANT, BATCH_MEAN):
        assert not BaselineSpec(kind=kind).requires_value_head


@pytest.mark.parametrize(
    ("kind", "updates_trunk", "owns_network"),
    [
        (VALUE_HEAD, True, False),
        (VALUE_HEAD_NO_UP, False, False),
        (VALUE_HEAD_OWN_NN, False, True),
    ],
)
def test_each_critic_kind_declares_its_own_wiring(
    kind, updates_trunk, owns_network
):
    spec = BaselineSpec(kind=kind)

    assert spec.critic_wiring == kind
    assert spec.critic_updates_trunk is updates_trunk
    assert spec.critic_owns_network is owns_network


def test_a_non_critic_baseline_declares_no_wiring():
    for kind in (ZERO, CONSTANT, BATCH_MEAN):
        spec = BaselineSpec(kind=kind)
        assert spec.critic_wiring is None
        assert spec.critic_updates_trunk is False
        assert spec.critic_owns_network is False


@pytest.mark.parametrize(
    ("spelling", "kind"),
    [
        ("value-head-no-up", VALUE_HEAD_NO_UP),
        ("value_head_no_up", VALUE_HEAD_NO_UP),
        ("value-head-own-nn", VALUE_HEAD_OWN_NN),
        ("value_head_own_nn", VALUE_HEAD_OWN_NN),
    ],
)
def test_both_spellings_of_the_new_kinds_parse(spelling, kind):
    assert from_tokens([spelling]).kind == kind


@ENTRY_POINTS
@pytest.mark.parametrize("kind", [VALUE_HEAD_NO_UP, VALUE_HEAD_OWN_NN])
def test_a_new_critic_without_the_value_head_is_rejected(parse, kind, capsys):
    with pytest.raises(SystemExit):
        parse(["--baseline", kind])
    assert "needs the critic" in capsys.readouterr().err


@pytest.mark.parametrize("kind", [VALUE_HEAD_NO_UP, VALUE_HEAD_OWN_NN])
def test_a_new_critic_reaches_the_training_options(kind):
    options = training_options_from_args(
        _standalone(["--value-head", "--baseline", kind])
    )

    assert BaselineSpec.from_mapping(options[0].baseline).kind == kind


@pytest.mark.parametrize("kind", CRITIC_KINDS)
def test_every_critic_kind_subtracts_one_prediction_per_decision(kind):
    """The three differ in wiring, never in what they subtract."""
    returns = np.array([[1.0, -1.0, 3.0, 0.0]], dtype=np.float32)
    predictions = np.array([[0.1, 0.9, -0.4, 0.6]], dtype=np.float32)
    spec = BaselineSpec(kind=kind)

    assert np.array_equal(
        baselines.subtract(spec, returns, value_predictions=predictions),
        returns - predictions,
    )
    with pytest.raises(ValueError, match="one critic prediction"):
        baselines.subtract(spec, returns)


# --------------------------------------------------------------------------
# The wiring, on the network
# --------------------------------------------------------------------------


def _network(**kwargs):
    return PolicyNetwork(
        input_size=6,
        output_size=4,
        hidden_sizes=(5, 3),
        learning_rate=0.1,
        random_seed=17,
        device="cpu",
        **kwargs,
    )


def _batch(seed=0, columns=8):
    rng = np.random.default_rng(seed)
    return {
        "x": np.asarray(rng.normal(size=(6, columns)), dtype=np.float32),
        "actions": [index % 4 for index in range(columns)],
        "masks": np.ones((4, columns), dtype=bool),
        "advantages": np.asarray(rng.normal(size=columns), dtype=np.float32),
        "returns": np.asarray(rng.normal(size=columns), dtype=np.float32),
        "old_values": np.zeros(columns, dtype=np.float32),
        "old_log_probs": np.zeros(columns, dtype=np.float32),
    }


def _ppo_step(network, batch, *, value_coef=0.5):
    return network.backward_ppo(
        batch["x"],
        batch["actions"],
        batch["masks"],
        batch["old_log_probs"],
        batch["advantages"],
        returns=batch["returns"],
        old_values=batch["old_values"],
        value_coef=value_coef,
        entropy_coef=0.0,
        clip_grad_norm=None,
    )


def _arrays(network, names, owner=None):
    owner = network if owner is None else owner
    return {name: np.array(getattr(owner, name)) for name in names}


def test_no_up_trains_the_critic_identically_and_the_trunk_differently():
    """The defining property of ``value-head-no-up``.

    ``Wv``/``bv`` use the last hidden activation as data, not as a node to
    differentiate through, so suppressing the trunk contribution cannot change
    them. The hidden stack is exactly what does change.
    """
    batch = _batch()
    updated = {}
    for label, flag in (("value-head", True), ("no-up", False)):
        network = _network(use_value_head=True, critic_updates_trunk=flag)
        network.Wv = np.full((1, 3), 0.3, dtype=np.float32)
        network.bv = np.zeros((1, 1), dtype=np.float32)
        _ppo_step(network, batch)
        updated[label] = _arrays(
            network,
            (*network.weight_names, "Wv", "bv"),
        )

    shared, no_up = updated["value-head"], updated["no-up"]
    for name in ("Wv", "bv"):
        assert np.array_equal(shared[name], no_up[name]), name
    for name in ("W1", "b1", "W2", "b2"):
        assert not np.array_equal(shared[name], no_up[name]), name
    # The output layer sits after the critic's entry point in the backward
    # pass, so it never sees the critic either way.
    for name in ("W3", "b3"):
        assert np.array_equal(shared[name], no_up[name]), name


def test_no_up_keeps_reporting_a_value_loss():
    """The critic is still trained; only its reach is cut."""
    network = _network(use_value_head=True, critic_updates_trunk=False)
    metrics = _ppo_step(network, _batch())

    assert metrics["value_loss"] is not None
    assert np.isfinite(metrics["value_loss"])


def test_own_nn_shares_no_array_with_the_policy():
    network = _network(use_value_head=True, critic_owns_network=True)

    assert network.critic_network is not None
    assert not hasattr(network, "Wv")
    policy_arrays = [getattr(network, name) for name in network.weight_names]
    for name in network.critic_network.weight_names:
        critic_array = getattr(network.critic_network, name)
        assert all(critic_array is not array for array in policy_arrays)


def test_own_nn_mirrors_the_policy_stack_with_one_linear_output():
    network = _network(use_value_head=True, critic_owns_network=True)
    critic = network.critic_network

    assert critic.hidden_sizes == network.hidden_sizes
    assert critic.W1.shape[1] == network.W1.shape[1]
    assert getattr(critic, f"W{critic.layer_count}").shape[0] == 1
    # One linear output, not a one-class softmax that would be constant 1.0.
    values = critic.forward(_batch()["x"])
    assert values.shape == (1, 8)
    assert not np.allclose(values, 1.0)


def test_own_nn_critic_loss_never_reaches_the_policy():
    """Turning the critic's loss on and off leaves the policy bit-identical."""
    batch = _batch()
    without = _network(use_value_head=True, critic_owns_network=True)
    _ppo_step(without, batch, value_coef=0.0)
    with_critic = _network(use_value_head=True, critic_owns_network=True)
    _ppo_step(with_critic, batch, value_coef=0.5)

    for name in without.weight_names:
        assert np.array_equal(
            np.asarray(getattr(without, name)),
            np.asarray(getattr(with_critic, name)),
        ), name
    moved = [
        name for name in without.critic_network.weight_names
        if not np.array_equal(
            np.asarray(getattr(without.critic_network, name)),
            np.asarray(getattr(with_critic.critic_network, name)),
        )
    ]
    assert moved


def test_own_nn_parameter_names_are_prefixed_and_resolvable():
    network = _network(use_value_head=True, critic_owns_network=True)

    names = network.critic_parameter_names
    assert names == tuple(
        f"{CRITIC_GRADIENT_PREFIX}{name}"
        for name in network.critic_network.weight_names
    )
    for name in names:
        target = network.parameter_array(name)
        attribute = name[len(CRITIC_GRADIENT_PREFIX):]
        assert target is getattr(network.critic_network, attribute)
    for name in network.weight_names:
        assert network.parameter_array(name) is getattr(network, name)


def test_a_shared_head_reports_the_two_arrays_it_actually_has():
    assert _network(use_value_head=True).critic_parameter_names == ("Wv", "bv")
    assert _network().critic_parameter_names == ()


def test_weight_decay_reaches_the_separate_critic():
    network = _network(
        use_value_head=True,
        critic_owns_network=True,
        weight_decay=0.01,
    )
    decayed = network.decayed_weight_names

    for name in network.critic_network.weight_names:
        prefixed = f"{CRITIC_GRADIENT_PREFIX}{name}"
        assert (prefixed in decayed) is name.startswith("W")


def test_the_reinforce_path_supports_every_wiring():
    batch = _batch()
    losses = {}
    for label, kwargs in (
        ("value-head", {}),
        ("no-up", {"critic_updates_trunk": False}),
        ("own-nn", {"critic_owns_network": True}),
    ):
        network = _network(use_value_head=True, **kwargs)
        # ``predict_values`` owns the forward pass the update differentiates,
        # exactly as the REINFORCE iteration path calls it.
        network.predict_values(batch["x"], training=True)
        metrics = network.backward_policy_gradient(
            action_indices=batch["actions"],
            policy_rewards=batch["advantages"].reshape(1, -1),
            legal_masks=batch["masks"],
            entropy_coef=0.0,
            clip_grad_norm=None,
            value_returns=batch["returns"].reshape(1, -1),
            value_coef=0.5,
        )
        losses[label] = metrics["value_loss"]
    assert all(value is not None for value in losses.values())
    # The two shared-trunk wirings start from the same critic and the same
    # cache, so their first value loss is identical; the separate critic is a
    # different function and need not agree.
    assert losses["value-head"] == losses["no-up"]


def test_a_separate_critic_refuses_to_reuse_a_forward_that_never_ran():
    network = _network(use_value_head=True, critic_owns_network=True)
    network.forward(_batch()["x"], training=True)

    with pytest.raises(RuntimeError, match="predict_values"):
        network.backward_policy_gradient(
            action_indices=_batch()["actions"],
            policy_rewards=_batch()["advantages"].reshape(1, -1),
            legal_masks=_batch()["masks"],
            entropy_coef=0.0,
            clip_grad_norm=None,
            value_returns=_batch()["returns"].reshape(1, -1),
            value_coef=0.5,
        )


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_a_separate_critic_round_trips_through_a_checkpoint(tmp_path):
    network = _network(use_value_head=True, critic_owns_network=True)
    _ppo_step(network, _batch())
    path = tmp_path / "policy.npz"
    network.save(str(path))

    restored = PolicyNetwork.load(
        str(path),
        use_value_head=True,
        critic_owns_network=True,
        device="cpu",
    )
    for name in network.critic_network.weight_names:
        assert np.array_equal(
            np.asarray(getattr(network.critic_network, name)),
            np.asarray(getattr(restored.critic_network, name)),
        ), name


def test_a_checkpoint_without_critic_arrays_still_loads(tmp_path):
    """A run started under another baseline leaves a fresh critic, not a crash."""
    path = tmp_path / "policy.npz"
    _network(use_value_head=True).save(str(path))

    restored = PolicyNetwork.load(
        str(path),
        use_value_head=True,
        critic_owns_network=True,
        device="cpu",
    )
    assert restored.critic_network is not None


def test_a_shared_head_checkpoint_is_unchanged_by_this_work(tmp_path):
    path = tmp_path / "policy.npz"
    _network(use_value_head=True).save(str(path))

    with np.load(path, allow_pickle=False) as data:
        assert "Wv" in data.files and "bv" in data.files
        assert not [
            name for name in data.files
            if name.startswith(CRITIC_GRADIENT_PREFIX)
        ]


def test_a_frozen_opponent_never_carries_a_separate_critic():
    """A pool opponent only acts, so duplicating a policy-sized critic is waste."""
    network = _network(use_value_head=True, critic_owns_network=True)
    clone = network.clone()

    assert clone.critic_network is None
    assert clone.use_value_head is False
    assert clone.weight_names == network.weight_names
