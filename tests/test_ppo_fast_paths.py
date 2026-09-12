"""Bit-identity of the PPO optimizer fast paths against the original update.

``ppo_update`` asks ``backward_ppo`` for less than the method can report: its
epoch statistics come from a whole-buffer evaluation, so the minibatch's own
statistics were computed and thrown away millions of times per run. Every fast
path here removes work whose result was discarded, never work the gradient
reads, so the contract under test is exact equality of the parameters -- not
closeness -- against ``_reference_ppo_step``, a verbatim copy of the update as
it stood before any fast path existed.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.nn import GPU_ENABLED
from agents.rl_nn import PolicyNetwork, _shared_gradient_hook
from training.rl.ppo import ppo_update

from tests.test_ppo import _FakePPONetwork, _buffer

WIRINGS = {
    "no_critic": {},
    "shared_critic": {"use_value_head": True},
    "no_up_critic": {"use_value_head": True, "critic_updates_trunk": False},
    "own_critic": {"use_value_head": True, "critic_owns_network": True},
}
DEVICES = [
    "cpu",
    pytest.param(
        "gpu",
        marks=pytest.mark.skipif(not GPU_ENABLED, reason="CuPy GPU unavailable"),
    ),
]
OPTIMIZER_KEYS = {
    "grad_norm",
    "grad_clipped",
    "applied_grad_norm",
    "optimizer_steps",
    "grad_rejected",
}


def _network(wiring, *, dropout_rate=0.0, learning_rate=0.05, device="cpu"):
    # The separate critic draws its initialization from the global generator.
    np.random.seed(123)
    if device == "gpu":
        import cupy  # pylint: disable=import-outside-toplevel

        cupy.random.seed(123)
    return PolicyNetwork(
        input_size=7,
        output_size=5,
        hidden_sizes=(6, 4),
        learning_rate=learning_rate,
        random_seed=19,
        device=device,
        dropout_rate=dropout_rate,
        **WIRINGS[wiring],
    )


def _batch(seed, columns=24):
    rng = np.random.default_rng(seed)
    masks = rng.random((5, columns)) < 0.6
    masks[:2, :] = True
    actions = np.asarray([
        rng.choice(np.flatnonzero(masks[:, column]))
        for column in range(columns)
    ])
    return {
        "x": np.asarray(rng.normal(size=(7, columns)), dtype=np.float32),
        "actions": actions,
        "masks": masks,
        # Spread around the current policy so both surrogate branches and the
        # ratio clip are exercised.
        "old_log_probs": np.asarray(
            np.log(rng.uniform(0.05, 0.9, size=columns)),
            dtype=np.float32,
        ),
        "advantages": np.asarray(rng.normal(size=columns), dtype=np.float32),
        "returns": np.asarray(rng.normal(size=columns), dtype=np.float32),
        "old_values": np.asarray(rng.normal(size=columns) * 0.1, dtype=np.float32),
    }


def _value_arguments(network, batch):
    if not network.use_value_head:
        return {}
    return {"returns": batch["returns"], "old_values": batch["old_values"]}


def _reference_ppo_step(
    network,
    x,
    action_indices,
    legal_masks,
    old_log_probs,
    advantages,
    *,
    returns=None,
    old_values=None,
    value_coef=0.5,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    clip_grad_norm=5.0,
    log_ratio_limit=20.0,
):
    """The masked PPO step exactly as it was before any fast path.

    Kept here, not imported, so the comparison survives later edits to the
    production method. Only the arithmetic that reaches the parameters is
    copied; the reported metrics are compared separately.
    """
    xp = network.xp
    network.forward(x, training=True)
    logits = network.cache[network.logits_key]
    sample_count = logits.shape[1]
    action_indices = xp.asarray(action_indices, dtype=xp.int64).reshape(-1)
    legal_masks = xp.asarray(legal_masks, dtype=xp.bool_)
    columns = xp.arange(sample_count)
    masked_logits = xp.where(legal_masks, logits, -xp.inf)
    shifted = masked_logits - xp.max(masked_logits, axis=0, keepdims=True)
    exponentials = xp.exp(shifted)
    masked_policy = exponentials / xp.sum(exponentials, axis=0, keepdims=True)
    probability_floor = xp.asarray(
        np.finfo(np.float32).tiny,
        dtype=masked_policy.dtype,
    )
    log_policy = xp.log(xp.maximum(masked_policy, probability_floor))
    new_log_probs = log_policy[action_indices, columns]
    entropy = -xp.sum(masked_policy * log_policy, axis=0)

    dtype = logits.dtype
    old_log_probs = xp.asarray(old_log_probs, dtype=dtype).reshape(-1)
    advantages = xp.asarray(advantages, dtype=dtype).reshape(-1)
    log_ratio = xp.clip(
        new_log_probs - old_log_probs,
        -float(log_ratio_limit),
        float(log_ratio_limit),
    )
    ratio = xp.exp(log_ratio)
    clipped_ratio = xp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    unclipped = ratio * advantages
    clipped = clipped_ratio * advantages
    active_weights = xp.where(unclipped <= clipped, ratio * advantages, 0.0)
    sampled = xp.zeros_like(masked_policy)
    sampled[action_indices, xp.arange(sample_count)] = 1.0
    entropy_row = entropy.reshape(1, -1)
    log_policy = xp.log(xp.maximum(masked_policy, probability_floor))
    dz3_policy = (masked_policy - sampled) * active_weights.reshape(1, -1)
    dz3_entropy = masked_policy * (log_policy + entropy_row)
    dz3 = dz3_policy + float(entropy_coef) * dz3_entropy

    last_hidden = network.cache[network.last_hidden_activation_key]
    inverse_count = xp.asarray(1.0 / sample_count, dtype=dz3.dtype)
    value_terms = network._ppo_value_update_terms(  # pylint: disable=protected-access
        x,
        last_hidden,
        returns,
        old_values,
        sample_count=sample_count,
        dtype=dz3.dtype,
        value_coef=value_coef,
        clip_epsilon=clip_epsilon,
    )
    gradients = network.backpropagate_layers(
        dz3,
        inverse_count,
        hidden_gradient_hook=_shared_gradient_hook(
            None if value_terms is None
            else value_terms["shared_hidden_gradient"]
        ),
    )
    if value_terms is not None:
        gradients.update(value_terms["gradients"])
    grad_norm = network._as_float(  # pylint: disable=protected-access
        xp.sqrt(sum(xp.sum(gradient ** 2) for gradient in gradients.values()))
    )
    return network._apply_gradient_step(  # pylint: disable=protected-access
        gradients,
        grad_norm,
        clip_grad_norm,
        dz3.dtype,
    )


def _host(array):
    return np.array(array.get() if hasattr(array, "get") else array)


def _parameters(network):
    return {
        name: _host(network.parameter_array(name))
        for name in (*network.weight_names, *network.critic_parameter_names)
    }


def _assert_same_parameters(left, right):
    assert left.keys() == right.keys()
    for name in left:
        assert left[name].dtype == right[name].dtype, name
        assert left[name].tobytes() == right[name].tobytes(), name


# One unclipped step, one that must clip, and one ordinary bound, so both sides
# of the norm clip reach the parameters whatever the wiring's gradient scale.
CLIP_NORMS = (None, 1e-3, 5.0)


def _run_steps(network, step, *, entropy_coef, **step_kwargs):
    # The policy draws dropout masks from the host generator; a separate
    # critic inherits the supervised draw from its backend's generator.
    np.random.seed(2024)
    if network.device == "gpu":
        network.xp.random.seed(2024)
    results = []
    for seed, clip_grad_norm in enumerate(CLIP_NORMS):
        batch = _batch(seed)
        results.append(step(
            network,
            batch["x"],
            batch["actions"],
            batch["masks"],
            batch["old_log_probs"],
            batch["advantages"],
            **_value_arguments(network, batch),
            entropy_coef=entropy_coef,
            clip_grad_norm=clip_grad_norm,
            **step_kwargs,
        ))
    return results


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("wiring", sorted(WIRINGS))
@pytest.mark.parametrize("dropout_rate", [0.0, 0.25])
# A coefficient far below any real one must still take the regularized path:
# only an exact zero may skip the entropy gradient.
@pytest.mark.parametrize("entropy_coef", [0.0, 1e-12, 0.03])
@pytest.mark.parametrize("collect_metrics", [True, False])
def test_every_step_mode_matches_the_original_update_bit_for_bit(
    device, wiring, dropout_rate, entropy_coef, collect_metrics
):
    reference = _network(wiring, dropout_rate=dropout_rate, device=device)
    candidate = _network(wiring, dropout_rate=dropout_rate, device=device)
    _assert_same_parameters(_parameters(reference), _parameters(candidate))

    expected = _run_steps(
        reference,
        _reference_ppo_step,
        entropy_coef=entropy_coef,
    )
    actual = _run_steps(
        candidate,
        PolicyNetwork.backward_ppo,
        entropy_coef=entropy_coef,
        collect_metrics=collect_metrics,
    )

    _assert_same_parameters(_parameters(reference), _parameters(candidate))
    assert candidate.optimizer_step_count == reference.optimizer_step_count == 3
    for before, after in zip(expected, actual):
        for key in OPTIMIZER_KEYS:
            assert after[key] == before[key], key
    assert [result["grad_clipped"] for result in expected] == [False, True, False]


def test_minimal_metrics_carry_only_the_optimizer_result_and_the_profile():
    network = _network("shared_critic")
    batch = _batch(0)

    result = network.backward_ppo(
        batch["x"],
        batch["actions"],
        batch["masks"],
        batch["old_log_probs"],
        batch["advantages"],
        **_value_arguments(network, batch),
        collect_metrics=False,
    )

    assert set(result) == OPTIMIZER_KEYS | {"runtime_profile_detail"}


def test_full_metrics_remain_the_default_contract():
    network = _network("shared_critic")
    batch = _batch(0)

    result = network.backward_ppo(
        batch["x"],
        batch["actions"],
        batch["masks"],
        batch["old_log_probs"],
        batch["advantages"],
        **_value_arguments(network, batch),
    )

    for key in (
        "policy_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "ratio_mean",
        "ratio_min",
        "ratio_max",
        "value_loss",
        "value_clip_fraction",
    ):
        assert np.isfinite(result[key]), key


@pytest.mark.parametrize("collect_metrics", [True, False])
def test_minimal_metrics_still_reject_a_non_finite_gradient(collect_metrics):
    network = _network("no_critic")
    before = _parameters(network)
    batch = _batch(1)
    advantages = batch["advantages"].copy()
    advantages[3] = np.inf

    result = network.backward_ppo(
        batch["x"],
        batch["actions"],
        batch["masks"],
        batch["old_log_probs"],
        advantages,
        collect_metrics=collect_metrics,
    )

    assert result["grad_rejected"] is True
    assert network.optimizer_step_count == 0
    _assert_same_parameters(before, _parameters(network))


@pytest.mark.parametrize("collect_metrics", [True, False])
def test_minimal_metrics_still_refuse_a_non_finite_value_loss(collect_metrics):
    network = _network("own_critic")
    before = _parameters(network)
    batch = _batch(2)
    returns = batch["returns"].copy()
    returns[0] = np.inf

    with pytest.raises(FloatingPointError, match="value loss"):
        network.backward_ppo(
            batch["x"],
            batch["actions"],
            batch["masks"],
            batch["old_log_probs"],
            batch["advantages"],
            returns=returns,
            old_values=batch["old_values"],
            collect_metrics=collect_metrics,
        )
    _assert_same_parameters(before, _parameters(network))


def test_ppo_update_asks_the_optimizer_for_minimal_metrics():
    network = _FakePPONetwork()
    requested = []
    original = network.backward_ppo

    def recording(*args, **kwargs):
        requested.append(kwargs.get("collect_metrics", True))
        return original(*args, **kwargs)

    network.backward_ppo = recording
    ppo_update(
        network,
        _buffer(512),
        base_seed=3,
        iteration=1,
        entropy_coef=0.0,
        max_epochs=2,
    )

    assert requested and not any(requested)


@pytest.mark.parametrize("device", DEVICES)
def test_observed_log_probabilities_do_not_depend_on_building_the_entropy(device):
    network = _network("no_critic", device=device)
    batch = _batch(4, columns=64)
    # Saturate a few legal logits so some probabilities reach the float floor.
    network.parameter_array("b3")[0] = 300.0

    full = network._evaluate_masked_actions(  # pylint: disable=protected-access
        batch["x"], batch["masks"], batch["actions"],
        training=False, need_entropy=True,
    )
    lean = network._evaluate_masked_actions(  # pylint: disable=protected-access
        batch["x"], batch["masks"], batch["actions"],
        training=False, need_entropy=False,
    )

    assert _host(full[0]).tobytes() == _host(lean[0]).tobytes()
    assert _host(full[2]).tobytes() == _host(lean[2]).tobytes()
    assert lean[1] is None and lean[3] is None
    assert np.any(_host(full[0]) == np.float32(np.log(np.finfo(np.float32).tiny)))
    public = network.evaluate_actions(batch["x"], batch["masks"], batch["actions"])
    assert _host(public[1]).tobytes() == _host(full[1]).tobytes()


@pytest.mark.parametrize(
    ("entropy_coef", "collect_metrics", "builds_entropy"),
    [
        (0.0, False, False),
        (0.0, True, True),
        (1e-12, False, True),
        (0.03, False, True),
    ],
)
def test_only_an_exact_zero_coefficient_without_metrics_skips_the_entropy(
    entropy_coef, collect_metrics, builds_entropy
):
    network = _network("no_critic")
    batch = _batch(5)
    requested = []
    original = network._evaluate_masked_actions  # pylint: disable=protected-access

    def recording(*args, **kwargs):
        requested.append(kwargs["need_entropy"])
        return original(*args, **kwargs)

    network._evaluate_masked_actions = recording  # pylint: disable=protected-access
    network.backward_ppo(
        batch["x"],
        batch["actions"],
        batch["masks"],
        batch["old_log_probs"],
        batch["advantages"],
        entropy_coef=entropy_coef,
        collect_metrics=collect_metrics,
    )

    assert requested == [builds_entropy]
