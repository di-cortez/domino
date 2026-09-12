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

from types import SimpleNamespace

import numpy as np
import pytest

from agents.nn import GPU_ENABLED
from agents.rl_nn import PolicyNetwork, _shared_gradient_hook
from training.rl import ppo
from training.rl.ppo import (
    PPOBuffer,
    PPOBufferStorage,
    evaluate_full_buffer,
    full_buffer_indices,
    ppo_update,
)

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

    with np.errstate(invalid="ignore", over="ignore"):
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


def _reference_forward(network, x, training):
    """``SupervisedNeuralNetwork.forward`` exactly as it was before the split."""
    x = network._to_backend(x)  # pylint: disable=protected-access
    last = network.layer_count
    cache = {"X": x}
    activation = x
    for index in range(1, last):
        pre_activation = network.xp.dot(
            getattr(network, f"W{index}"),
            activation,
        ) + getattr(network, f"b{index}")
        activation, mask = network._hidden_dropout(  # pylint: disable=protected-access
            network.relu(pre_activation),
            training,
        )
        cache[f"Z{index}"] = pre_activation
        cache[f"A{index}"] = activation
        if mask is not None:
            cache[f"D{index}"] = mask
    logits = network.xp.dot(
        getattr(network, f"W{last}"),
        activation,
    ) + getattr(network, f"b{last}")
    probabilities = network.softmax(logits)
    cache[f"Z{last}"] = logits
    cache[f"A{last}"] = probabilities
    return probabilities, cache


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("hidden_sizes", [(6,), (6, 4), (8, 6, 4)])
@pytest.mark.parametrize("dropout_rate", [0.0, 0.3])
@pytest.mark.parametrize("training", [False, True])
def test_forward_still_matches_the_original_layer_loop(
    device, hidden_sizes, dropout_rate, training
):
    network = PolicyNetwork(
        input_size=7,
        output_size=5,
        hidden_sizes=hidden_sizes,
        random_seed=3,
        device=device,
        dropout_rate=dropout_rate,
    )
    x = np.random.default_rng(8).normal(size=(7, 16)).astype(np.float32)

    np.random.seed(77)
    expected, expected_cache = _reference_forward(network, x, training)
    expected_draws = np.random.random(4)
    np.random.seed(77)
    actual = network.forward(x, training=training)
    actual_draws = np.random.random(4)

    assert _host(actual).tobytes() == _host(expected).tobytes()
    assert list(network.cache) == list(expected_cache)
    for key, value in expected_cache.items():
        assert _host(network.cache[key]).tobytes() == _host(value).tobytes(), key
    # The same number of dropout draws, in the same order.
    assert actual_draws.tobytes() == expected_draws.tobytes()


def test_logits_forward_caches_everything_but_the_output_softmax():
    network = _network("no_critic", dropout_rate=0.2)
    x = _batch(6)["x"]

    np.random.seed(5)
    network.forward(x, training=True)
    full_cache = dict(network.cache)
    np.random.seed(5)
    logits = network._forward_logits(x, training=True)  # pylint: disable=protected-access

    last = f"A{network.layer_count}"
    assert set(full_cache) - set(network.cache) == {last}
    assert logits is network.cache[network.logits_key]
    for key, value in network.cache.items():
        assert value.tobytes() == full_cache[key].tobytes(), key


@pytest.mark.parametrize("wiring", sorted(WIRINGS))
def test_masked_ppo_paths_never_build_the_full_support_softmax(wiring, monkeypatch):
    network = _network(wiring, dropout_rate=0.2)
    batch = _batch(7)

    def forbidden(_logits):
        raise AssertionError("the unmasked softmax was computed")

    monkeypatch.setattr(network, "softmax", forbidden)
    if network.critic_network is not None:
        monkeypatch.setattr(network.critic_network, "softmax", forbidden)
    network.evaluate_actions(batch["x"], batch["masks"], batch["actions"])
    for collect_metrics in (True, False):
        network.backward_ppo(
            batch["x"],
            batch["actions"],
            batch["masks"],
            batch["old_log_probs"],
            batch["advantages"],
            **_value_arguments(network, batch),
            entropy_coef=0.01,
            collect_metrics=collect_metrics,
        )
    if network.use_value_head:
        network.critic_values(batch["x"])


def test_a_separate_critic_still_caches_its_value_as_the_output_activation():
    network = _network("own_critic")
    critic = network.critic_network
    x = _batch(8)["x"]

    values = critic.forward(x)
    _expected, expected_cache = _reference_forward(critic, x, False)

    last = critic.layer_count
    assert values.tobytes() == expected_cache[f"Z{last}"].tobytes()
    assert critic.cache[f"A{last}"] is values


@pytest.mark.parametrize("batch_size", [512, 2048, 4096, 8192])
@pytest.mark.parametrize("decisions", [1, 511, 512, 513, 4095, 4096, 4097, 8327])
def test_evaluation_partitions_cover_every_decision_once_in_order(
    batch_size, decisions
):
    partitions = full_buffer_indices(decisions, batch_size)

    assert np.array_equal(np.concatenate(partitions), np.arange(decisions))
    assert all(part.size == batch_size for part in partitions[:-1])
    assert 1 <= partitions[-1].size <= batch_size


def test_evaluation_partitions_default_to_the_evaluation_constant(monkeypatch):
    monkeypatch.setattr(ppo, "PPO_FULL_BUFFER_EVAL_BATCH_SIZE", 300)
    assert [part.size for part in full_buffer_indices(700)] == [300, 300, 100]
    assert ppo.PPO_TARGET_DECISIONS_PER_MINIBATCH == 512


def _real_buffer(network, decisions, *, seed, with_values):
    rng = np.random.default_rng(seed)
    samples = []
    for index in range(decisions):
        mask = rng.random((5, 1)) < 0.5
        mask[:2] = True
        legal = np.flatnonzero(mask[:, 0])
        samples.append(SimpleNamespace(
            x=rng.normal(size=(7, 1)).astype(np.float32),
            legal_mask=mask,
            action_index=int(rng.choice(legal)),
            old_log_prob=float(np.log(rng.uniform(0.1, 0.9))),
            policy_reward=float(rng.normal()),
            local_reward=0.0,
            terminal_reward=float(index % 3),
        ))
    old_values = None
    if with_values:
        old_values = rng.normal(size=decisions).astype(np.float32) * 0.1
    return PPOBuffer.from_samples(samples, old_values=old_values)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("wiring", ["no_critic", "shared_critic", "own_critic"])
def test_evaluation_statistics_do_not_depend_on_the_partition_size(device, wiring):
    network = _network(wiring, device=device)
    buffer = _real_buffer(
        network, 1337, seed=9, with_values=network.use_value_head
    )
    storage = PPOBufferStorage(network, buffer)
    try:
        results = {
            size: evaluate_full_buffer(
                network,
                storage,
                full_buffer_indices(buffer.size, size),
                ppo.PPO_CLIP_EPSILON,
            )
            for size in (97, 512, 1337, 4096)
        }
    finally:
        storage.close()

    reference = results[512]
    for size, result in results.items():
        assert result.keys() == reference.keys()
        for key, expected in reference.items():
            if expected is None:
                assert result[key] is None, (size, key)
            else:
                # Partition sums are float32; the global sum is float64.
                assert result[key] == pytest.approx(
                    expected, rel=1e-5, abs=1e-6
                ), (size, key)


class _EvaluationOOMNetwork(_FakePPONetwork):
    """Fails any evaluation forward pass wider than the optimizer's batch."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.memory_failures = 0
        self.released = 0

    def evaluate_actions(self, states, legal_masks, actions):
        if np.asarray(actions).size > ppo.PPO_TARGET_DECISIONS_PER_MINIBATCH:
            self.memory_failures += 1
            raise MemoryError("simulated evaluation OOM")
        return super().evaluate_actions(states, legal_masks, actions)

    def release_disposable_cache(self):
        self.released += 1
        super().release_disposable_cache()


def test_an_evaluation_that_runs_out_of_memory_retries_at_optimizer_size():
    buffer = _buffer(1300)
    healthy = _FakePPONetwork(ratio_after_update=1.001)
    starved = _EvaluationOOMNetwork(ratio_after_update=1.001)

    expected = ppo_update(
        healthy, buffer, base_seed=2, iteration=5, entropy_coef=0.0, max_epochs=3
    )
    actual = ppo_update(
        starved, buffer, base_seed=2, iteration=5, entropy_coef=0.0, max_epochs=3
    )

    # Only the first epoch's evaluation fails; later epochs keep the size that
    # fit instead of failing again.
    assert starved.memory_failures == 1
    assert actual["runtime_profile_detail"]["full_buffer_evaluation"][
        "batch_size_memory_fallbacks"
    ] == 1
    assert starved.eval_batch_sizes[1:] == [512, 512, 276] * 3
    assert actual["optimizer_steps"] == expected["optimizer_steps"]
    assert starved.optimizer_step_count == healthy.optimizer_step_count
    for key in ("epochs_completed", "final_approx_kl", "stopped_by_kl"):
        assert actual[key] == expected[key], key


def test_an_evaluation_failure_other_than_memory_is_not_retried(monkeypatch):
    network = _FakePPONetwork()
    storage = PPOBufferStorage(network, _buffer(600))
    calls = []

    def explode(*_args, **_kwargs):
        calls.append(1)
        raise FloatingPointError("PPO full-buffer metrics produced NaN/Inf.")

    monkeypatch.setattr(ppo, "evaluate_full_buffer", explode)
    try:
        with pytest.raises(FloatingPointError):
            ppo._evaluate_full_buffer_within_memory(  # pylint: disable=protected-access
                network, storage, full_buffer_indices(600), {}
            )
    finally:
        storage.close()
    # A diverged epoch goes straight to the rollback, never to a retry.
    assert calls == [1]


def _reference_evaluate_full_buffer(network, storage, partitions, clip_epsilon):
    """``evaluate_full_buffer`` as it was before its reductions left the host.

    Every partition reads each reduction back at once and accumulates Python
    floats; only the metrics are kept, not the profile.
    """
    xp = network.xp
    total = 0
    surrogate_sum = entropy_sum = kl_sum = ratio_sum = 0.0
    clipped_count = 0
    ratio_min = float("inf")
    ratio_max = float("-inf")
    deficit_max = float("-inf")
    value_loss_sum = value_sum = value_square_sum = 0.0
    return_sum = return_square_sum = 0.0
    value_error_sum = value_error_square_sum = 0.0
    value_clipped_count = 0
    lower = 1.0 - float(clip_epsilon)
    upper = 1.0 + float(clip_epsilon)
    use_value_head = getattr(network, "use_value_head", False)
    for indices in partitions:
        batch = storage.batch(indices)
        new_log_probs, entropy, _policy = network.evaluate_actions(
            batch["states"], batch["legal_masks"], batch["actions"]
        )
        batch_deficit = ppo._legal_logit_deficit_max(  # pylint: disable=protected-access
            network, batch["legal_masks"]
        )
        if batch_deficit is not None:
            deficit_max = max(deficit_max, batch_deficit)
        values = value_losses = value_delta = None
        if use_value_head:
            values = network.critic_values(batch["states"])
            value_losses, _gradient, value_delta = network.clipped_value_loss_terms(
                values,
                batch["returns"].reshape(1, -1),
                batch["old_values"].reshape(1, -1),
                clip_epsilon,
            )
        log_ratio = new_log_probs - batch["old_log_probs"]
        ratio = xp.exp(log_ratio)
        finite = xp.all(xp.isfinite(ratio)) & xp.all(xp.isfinite(entropy))
        if values is not None:
            finite = finite & xp.all(xp.isfinite(values)) & xp.all(
                xp.isfinite(value_losses)
            )
        if not bool(network._as_float(finite)):  # pylint: disable=protected-access
            raise FloatingPointError("PPO full-buffer metrics produced NaN/Inf.")
        as_float = network._as_float  # pylint: disable=protected-access
        clipped_ratio = xp.clip(ratio, lower, upper)
        surrogate = xp.minimum(
            ratio * batch["advantages"], clipped_ratio * batch["advantages"]
        )
        total += int(len(indices))
        surrogate_sum += as_float(xp.sum(surrogate))
        entropy_sum += as_float(xp.sum(entropy))
        kl_sum += as_float(xp.sum((ratio - 1.0) - log_ratio))
        clipped_count += int(as_float(xp.sum((ratio < lower) | (ratio > upper))))
        ratio_sum += as_float(xp.sum(ratio))
        ratio_min = min(ratio_min, as_float(xp.min(ratio)))
        ratio_max = max(ratio_max, as_float(xp.max(ratio)))
        if values is not None:
            flat_values = values.reshape(-1)
            returns = batch["returns"].reshape(-1)
            errors = returns - flat_values
            value_loss_sum += as_float(xp.sum(value_losses))
            value_sum += as_float(xp.sum(flat_values))
            value_square_sum += as_float(xp.sum(flat_values ** 2))
            return_sum += as_float(xp.sum(returns))
            return_square_sum += as_float(xp.sum(returns ** 2))
            value_error_sum += as_float(xp.sum(errors))
            value_error_square_sum += as_float(xp.sum(errors ** 2))
            value_clipped_count += int(as_float(
                xp.sum(xp.abs(value_delta) > float(clip_epsilon))
            ))
    result = {
        "policy_loss": float(-surrogate_sum / total),
        "entropy": float(entropy_sum / total),
        "approx_kl": max(0.0, float(kl_sum / total)),
        "clip_fraction": float(clipped_count / total),
        "ratio_mean": float(ratio_sum / total),
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
        "legal_logit_deficit_max": (
            None if deficit_max == float("-inf") else deficit_max
        ),
        "value_loss": None,
        "value_clip_fraction": None,
        "value_mean": None,
        "value_std": None,
        "explained_variance": None,
    }
    if use_value_head:
        value_mean = value_sum / total
        value_variance = max(0.0, value_square_sum / total - value_mean ** 2)
        return_mean = return_sum / total
        return_variance = max(0.0, return_square_sum / total - return_mean ** 2)
        error_mean = value_error_sum / total
        error_variance = max(0.0, value_error_square_sum / total - error_mean ** 2)
        result.update({
            "value_loss": float(value_loss_sum / total),
            "value_clip_fraction": float(value_clipped_count / total),
            "value_mean": float(value_mean),
            "value_std": float(np.sqrt(value_variance)),
            "explained_variance": (
                None
                if return_variance <= ppo.ADVANTAGE_EPSILON
                else float(1.0 - error_variance / return_variance)
            ),
        })
    return result


def _assert_identical_metrics(left, right):
    assert left.keys() == right.keys()
    for key, value in left.items():
        other = right[key]
        assert type(value) is type(other), key
        if isinstance(value, float):
            assert np.float64(value).tobytes() == np.float64(other).tobytes(), key
        else:
            assert value == other, key


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("wiring", ["no_critic", "shared_critic", "own_critic"])
@pytest.mark.parametrize("partition_size", [97, 512, 4096])
def test_device_accumulated_metrics_equal_host_accumulated_metrics_bit_for_bit(
    device, wiring, partition_size
):
    network = _network(wiring, device=device)
    buffer = _real_buffer(
        network, 1337, seed=11, with_values=network.use_value_head
    )
    # Extreme ratios on both sides: floored behavior probabilities send the
    # ratio towards exp(87), near-certain ones towards zero.
    old_log_probs = np.array(buffer.old_log_probs)
    old_log_probs[::101] = np.log(np.finfo(np.float32).tiny)
    old_log_probs[50::101] = 0.0
    buffer = PPOBuffer(**{
        **{field: getattr(buffer, field) for field in buffer.__dataclass_fields__},
        "old_log_probs": old_log_probs,
    })
    storage = PPOBufferStorage(network, buffer)
    partitions = full_buffer_indices(buffer.size, partition_size)
    try:
        expected = _reference_evaluate_full_buffer(
            network, storage, partitions, ppo.PPO_CLIP_EPSILON
        )
        actual = evaluate_full_buffer(
            network, storage, partitions, ppo.PPO_CLIP_EPSILON
        )
    finally:
        storage.close()

    assert expected["ratio_max"] > 1e30
    _assert_identical_metrics(expected, actual)


def test_every_statistic_reaches_the_host_in_one_transfer(monkeypatch):
    network = _network("shared_critic")
    buffer = _real_buffer(network, 1337, seed=12, with_values=True)
    storage = PPOBufferStorage(network, buffer)
    transfers = []
    original = ppo._to_numpy  # pylint: disable=protected-access

    def counting(value, **kwargs):
        transfers.append(1)
        return original(value, **kwargs)

    scalar_reads = []
    as_float = network._as_float  # pylint: disable=protected-access

    def counting_scalar(value):
        scalar_reads.append(1)
        return as_float(value)

    monkeypatch.setattr(ppo, "_to_numpy", counting)
    monkeypatch.setattr(network, "_as_float", counting_scalar)
    partitions = full_buffer_indices(buffer.size, 97)
    try:
        evaluate_full_buffer(network, storage, partitions, ppo.PPO_CLIP_EPSILON)
    finally:
        storage.close()

    # The floating statistics and the integer counts: two, for 14 partitions.
    assert len(transfers) == 2
    # Only the public action evaluation's two mask checks still read back.
    assert len(scalar_reads) == 2 * len(partitions)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("poisoned_partition", [0, 2])
def test_a_non_finite_partition_still_fails_the_whole_evaluation(
    device, poisoned_partition
):
    network = _network("shared_critic", device=device)
    buffer = _real_buffer(network, 1300, seed=13, with_values=True)
    storage = PPOBufferStorage(network, buffer)
    partitions = full_buffer_indices(buffer.size, 512)
    states = np.array(buffer.states)
    states[0, partitions[poisoned_partition][7]] = np.nan
    buffer = PPOBuffer(**{
        **{field: getattr(buffer, field) for field in buffer.__dataclass_fields__},
        "states": states,
    })
    storage.close()
    storage = PPOBufferStorage(network, buffer)
    try:
        with pytest.raises(FloatingPointError, match="full-buffer metrics"):
            evaluate_full_buffer(network, storage, partitions, ppo.PPO_CLIP_EPSILON)
    finally:
        storage.close()


def test_a_poisoned_epoch_is_rolled_back_through_the_real_evaluation():
    network = _network("no_critic", learning_rate=0.01)
    buffer = _real_buffer(network, 1100, seed=14, with_values=False)
    before = _parameters(network)
    original_step = network.backward_ppo
    calls = []

    def poisoning_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        calls.append(1)
        if len(calls) == 2:
            network.parameter_array("W3")[0, 0] = np.nan
        return result

    network.backward_ppo = poisoning_step
    metrics = ppo_update(
        network, buffer, base_seed=1, iteration=1, entropy_coef=0.0, max_epochs=4
    )

    assert metrics["diverged_epoch"] == 1
    assert metrics["epochs_completed"] == 0
    assert metrics["stopped_by_kl"] is False
    assert network.optimizer_step_count == 0
    _assert_same_parameters(before, _parameters(network))
