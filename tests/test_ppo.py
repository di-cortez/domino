"""Focused correctness tests for masked PPO and its decision-buffer storage."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np

from agents.network_architecture import policy_layer_names
from agents.rl_nn import NonFinitePolicyError, PolicyNetwork
from training.rl.ppo import (
    PPO_LOG_RATIO_LIMIT,
    PPOBuffer,
    _legal_logit_deficit_max,
    _max_or_none,
    PPOBufferStorage,
    clipped_surrogate,
    effective_minibatches,
    log_ratio_statistics,
    minibatch_indices,
    ppo_update,
    scale_advantages,
    requested_minibatches,
)


def _sample(index, *, state_size=3, action_size=4, reward=None, old_log_prob=0.0):
    state = np.full((state_size, 1), index / 100.0, dtype=np.float32)
    legal_mask = np.zeros((action_size, 1), dtype=np.bool_)
    legal_mask[0, 0] = True
    legal_mask[1, 0] = True
    value = float((index % 7) - 3 if reward is None else reward)
    return SimpleNamespace(
        x=state,
        action_index=index % 2,
        legal_mask=legal_mask,
        old_log_prob=float(old_log_prob),
        policy_reward=value,
        local_reward=value / 10.0,
        terminal_reward=value - value / 10.0,
    )


def _buffer(size=256, *, rewards=None):
    samples = [
        _sample(
            index,
            reward=None if rewards is None else rewards[index],
        )
        for index in range(size)
    ]
    return PPOBuffer.from_samples(samples)


class _FakePPONetwork:
    """Small NumPy learner exposing the interface used by ``ppo_update``."""

    def __init__(self, ratio_after_update=1.0, *, device="cpu", fail_first_eval=False):
        self.xp = np
        self.device = device
        self.ratio_after_update = ratio_after_update
        self.fail_first_eval = bool(fail_first_eval)
        self.eval_calls = 0
        self.eval_batch_sizes = []
        self.optimizer_step_count = 0
        self.cache = {}
        self.hidden_sizes = (2, 2)
        self.W1 = np.zeros((2, 3), dtype=np.float32)
        self.b1 = np.zeros((2, 1), dtype=np.float32)
        self.W2 = np.zeros((2, 2), dtype=np.float32)
        self.b2 = np.zeros((2, 1), dtype=np.float32)
        self.W3 = np.zeros((4, 2), dtype=np.float32)
        self.b3 = np.zeros((4, 1), dtype=np.float32)

    @property
    def layer_count(self):
        return len(self.hidden_sizes) + 1

    @property
    def weight_names(self):
        return policy_layer_names(len(self.hidden_sizes))

    @property
    def last_hidden_activation_key(self):
        return f"A{len(self.hidden_sizes)}"

    @property
    def logits_key(self):
        return f"Z{self.layer_count}"

    @property
    def critic_parameter_names(self):
        """No critic: this double never enables a value head."""
        return ()

    def parameter_array(self, name):
        """Resolve one parameter name the way ``PolicyNetwork`` does."""
        return getattr(self, name)

    def snapshot_parameters(self):
        """Mirror ``PolicyNetwork``: one detached copy per trainable array."""
        return {
            name: self.parameter_array(name).copy()
            for name in (*self.weight_names, *self.critic_parameter_names)
        }

    def restore_parameters(self, arrays):
        for name, value in arrays.items():
            setattr(self, name, value)

    def evaluate_actions(self, states, legal_masks, actions):
        self.eval_calls += 1
        if self.fail_first_eval and self.eval_calls == 1:
            raise MemoryError("simulated CUDA workspace OOM")
        count = int(np.asarray(actions).size)
        self.eval_batch_sizes.append(count)
        action_size = int(np.asarray(legal_masks).shape[0])
        self.cache = {
            "Z3": np.zeros((action_size, count), dtype=np.float32),
            "A2": np.zeros((2, count), dtype=np.float32),
            "Z2": np.zeros((2, count), dtype=np.float32),
            "A1": np.zeros((2, count), dtype=np.float32),
            "Z1": np.zeros((2, count), dtype=np.float32),
        }
        if self.optimizer_step_count == 0:
            ratio = 1.0
        elif callable(self.ratio_after_update):
            ratio = float(self.ratio_after_update(self.optimizer_step_count))
        else:
            ratio = float(self.ratio_after_update)
        log_probs = np.full(count, np.log(ratio), dtype=np.float32)
        entropy = np.full(count, 0.5, dtype=np.float32)
        policy = np.zeros((action_size, count), dtype=np.float32)
        policy[:2] = 0.5
        return log_probs, entropy, policy

    def backward_ppo(self, *args, **kwargs):
        self.optimizer_step_count += 1
        return {
            "grad_norm": 2.0,
            "applied_grad_norm": 2.0,
            "grad_clipped": False,
        }

    @staticmethod
    def _as_float(value):
        return float(value)

    @staticmethod
    def _is_backend_memory_error(exc):
        return isinstance(exc, MemoryError)

    def synchronize(self):
        return None

    def release_disposable_cache(self):
        self.cache = {}


def test_initial_policy_has_unit_ratio_zero_kl_and_zero_clip_fraction():
    old = np.asarray([-0.2, -1.3, -3.0], dtype=np.float32)
    stats = log_ratio_statistics(old.copy(), old)

    assert stats["ratio_mean"] == 1.0
    assert stats["ratio_min"] == 1.0
    assert stats["ratio_max"] == 1.0
    assert stats["approx_kl"] == 0.0
    assert stats["clip_fraction"] == 0.0


def test_clipped_surrogate_handles_positive_and_negative_advantages():
    ratios = np.asarray([1.30, 0.70, 1.30, 0.70])
    advantages = np.asarray([1.0, -1.0, -1.0, 1.0])

    actual = clipped_surrogate(ratios, advantages)

    # Positive/high and negative/low ratios clip; the other two stay active.
    np.testing.assert_allclose(actual, [1.2, -0.8, -1.3, 0.7])


def test_masked_evaluation_assigns_zero_probability_to_illegal_actions():
    network = PolicyNetwork(
        input_size=3,
        hidden1_size=4,
        hidden2_size=3,
        output_size=4,
        learning_rate=0.1,
        random_seed=3,
        device="cpu",
    )
    for name in network.weight_names:
        getattr(network, name)[:] = 0.0
    states = np.ones((3, 1), dtype=np.float32)
    mask = np.asarray([[True], [False], [True], [False]])

    log_probs, _entropy, policy = network.evaluate_actions(states, mask, [2])

    assert policy[1, 0] == 0.0
    assert policy[3, 0] == 0.0
    assert np.isclose(policy[:, 0].sum(), 1.0)
    assert np.isclose(log_probs[0], np.log(0.5), atol=1e-7)


def test_ppo_step_uses_saved_mask_and_increases_positive_action_probability():
    network = PolicyNetwork(
        input_size=3,
        hidden1_size=4,
        hidden2_size=3,
        output_size=4,
        learning_rate=0.2,
        random_seed=4,
        device="cpu",
    )
    for name in network.weight_names:
        getattr(network, name)[:] = 0.0
    states = np.ones((3, 1), dtype=np.float32)
    mask = np.asarray([[True], [True], [False], [False]])
    old_log_prob, _entropy, before = network.evaluate_actions(states, mask, [0])

    network.backward_ppo(
        states,
        [0],
        mask,
        old_log_prob.copy(),
        [1.0],
        entropy_coef=0.0,
        clip_grad_norm=None,
    )
    _new_log_prob, _entropy, after = network.evaluate_actions(states, mask, [0])

    assert after[0, 0] > before[0, 0]
    assert after[2, 0] == 0.0
    assert network.optimizer_step_count == 1


def test_clipped_value_loss_uses_the_larger_loss_and_clips_its_gradient():
    network = PolicyNetwork(
        input_size=3,
        hidden1_size=4,
        hidden2_size=3,
        output_size=4,
        use_value_head=True,
        device="cpu",
    )
    losses, gradients, deltas = network.clipped_value_loss_terms(
        np.asarray([[0.4, 0.4]], dtype=np.float32),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        0.2,
    )

    np.testing.assert_allclose(losses, [[0.32, 0.08]], atol=1e-7)
    np.testing.assert_allclose(gradients, [[0.0, 0.4]], atol=1e-7)
    np.testing.assert_allclose(deltas, [[0.4, 0.4]], atol=1e-7)


def test_ppo_value_head_updates_critic_and_reports_value_metrics():
    network = PolicyNetwork(
        input_size=3,
        hidden1_size=4,
        hidden2_size=3,
        output_size=4,
        learning_rate=0.05,
        random_seed=17,
        use_value_head=True,
        device="cpu",
    )
    samples = [
        _sample(index, reward=(-1.0 if index % 2 else 1.0))
        for index in range(300)
    ]
    states = np.hstack([sample.x for sample in samples])
    masks = np.hstack([sample.legal_mask for sample in samples])
    actions = [sample.action_index for sample in samples]
    old_log_probs, _entropy, _policy = network.evaluate_actions(
        states,
        masks,
        actions,
    )
    for sample, old_log_prob in zip(samples, old_log_probs):
        sample.old_log_prob = float(old_log_prob)
    old_values = np.asarray(network.predict_values(states)).reshape(-1)
    buffer = PPOBuffer.from_samples(samples, old_values=old_values)
    critic_before = network.Wv.copy()

    metrics = ppo_update(
        network,
        buffer,
        base_seed=42,
        iteration=1,
        entropy_coef=0.0,
        value_coef=0.5,
        max_epochs=2,
    )

    assert not np.array_equal(network.Wv, critic_before)
    assert metrics["value_loss"] is not None
    assert metrics["final_value_loss"] >= 0.0
    assert 0.0 <= metrics["final_value_clip_fraction"] <= 1.0
    assert metrics["final_value_mean"] is not None
    assert metrics["final_value_std"] is not None
    assert all(row["value_loss"] is not None for row in metrics["epoch_metrics"])


def test_ppo_value_head_requires_pre_update_values_in_the_buffer():
    network = PolicyNetwork(
        input_size=3,
        hidden1_size=4,
        hidden2_size=3,
        output_size=4,
        use_value_head=True,
        device="cpu",
    )
    with np.testing.assert_raises_regex(ValueError, "old_values are required"):
        ppo_update(
            network,
            _buffer(8),
            base_seed=42,
            iteration=1,
            entropy_coef=0.0,
        )


def test_requested_minibatches_are_derived_only_from_decision_count():
    expected = {1: 1, 255: 1, 256: 1, 512: 1, 513: 2, 1024: 2}
    assert {
        decisions: requested_minibatches(decisions)
        for decisions in expected
    } == expected


def test_decision_minibatches_follow_target_minimum_and_omitted_tail_policy():
    expected = {
        255: ([], 255),
        256: ([256], 0),
        511: ([511], 0),
        512: ([512], 0),
        700: ([512], 188),
        768: ([512, 256], 0),
        1025: ([512, 512], 1),
    }
    for decision_count, (sizes, omitted_size) in expected.items():
        parts, omitted = minibatch_indices(decision_count, seed=987)
        assert [len(part) for part in parts] == sizes
        assert len(omitted) == omitted_size
        assert effective_minibatches(decision_count) == len(parts)
        combined = np.concatenate((*parts, omitted))
        assert len(combined) == decision_count
        assert len(np.unique(combined)) == decision_count
        np.testing.assert_array_equal(np.sort(combined), np.arange(decision_count))


def test_omitted_tail_changes_with_the_deterministic_epoch_seed():
    _parts_a, omitted_a = minibatch_indices(700, seed=11)
    _parts_b, omitted_b = minibatch_indices(700, seed=12)
    assert omitted_a.size == omitted_b.size == 188
    assert not np.array_equal(np.sort(omitted_a), np.sort(omitted_b))


def test_full_buffer_kl_still_evaluates_the_optimizer_omitted_tail():
    network = _FakePPONetwork(ratio_after_update=1.001)
    metrics = ppo_update(
        network,
        _buffer(700),
        base_seed=42,
        iteration=7,
        entropy_coef=0.0,
        max_epochs=2,
    )
    assert metrics["minibatch_sizes"] == [512]
    assert metrics["decisions_used_per_epoch"] == 512
    assert metrics["decisions_omitted_per_epoch"] == 188
    assert metrics["optimizer_steps"] == 2
    # One workspace probe is followed by a complete 512+188 evaluation after
    # each epoch. The omitted optimizer tail still participates in KL control.
    assert network.eval_batch_sizes[-4:] == [512, 188, 512, 188]


def test_fewer_than_minimum_decisions_produces_an_explicit_noop():
    network = _FakePPONetwork(ratio_after_update=1.001)
    metrics = ppo_update(
        network,
        _buffer(255),
        base_seed=42,
        iteration=8,
        entropy_coef=0.0,
        max_epochs=16,
    )
    assert metrics["insufficient_decisions"]
    assert metrics["optimizer_steps"] == 0
    assert metrics["epochs_completed"] == 0
    assert metrics["decisions_omitted_per_epoch"] == 255
    assert network.optimizer_step_count == 0


def test_advantages_are_scaled_once_globally_and_zero_std_is_safe():
    # Scaling is deliberately center-free: the baseline owns the center, so a
    # non-zero mean must survive to keep zero and constant baselines visible.
    scaled, std_zero, raw_mean, raw_std = scale_advantages([1, 2, 3, 4])
    assert not std_zero
    assert raw_mean == 2.5
    assert raw_std > 0
    assert float(scaled.mean()) > 0.0
    assert np.isclose(float(scaled.std()), 1.0, atol=1e-6)

    centered, std_zero, raw_mean, raw_std = scale_advantages(
        np.asarray([1, 2, 3, 4], dtype=np.float32) - 2.5
    )
    assert not std_zero
    assert abs(float(centered.mean())) < 1e-7
    assert np.isclose(float(centered.std()), 1.0, atol=1e-6)

    constant, std_zero, _mean, raw_std = scale_advantages([7, 7, 7])
    assert std_zero
    assert raw_std == 0.0
    assert np.all(constant == 7.0)
    assert np.all(np.isfinite(constant))


def test_kl_early_stop_occurs_only_after_the_completed_epoch():
    network = _FakePPONetwork(ratio_after_update=1.30)
    metrics = ppo_update(
        network,
        _buffer(),
        base_seed=42,
        iteration=1,
        entropy_coef=0.0,
        max_epochs=16,
    )

    assert metrics["stopped_by_kl"]
    assert metrics["epochs_completed"] == 1
    assert metrics["final_approx_kl"] > 0.015
    assert metrics["target_kl"] == 0.01
    assert metrics["stop_kl"] == 0.015
    assert [row["epoch"] for row in metrics["epoch_metrics"]] == [1]
    assert metrics["optimizer_steps"] == metrics["effective_minibatches"]
    assert network.optimizer_step_count == metrics["optimizer_steps"]


def test_kl_early_stop_can_end_a_sixteen_epoch_budget_after_several_epochs():
    network = _FakePPONetwork(
        ratio_after_update=lambda optimizer_steps: (
            1.001 if optimizer_steps < 5 else 1.30
        )
    )
    metrics = ppo_update(
        network,
        _buffer(),
        base_seed=42,
        iteration=2,
        entropy_coef=0.0,
        max_epochs=16,
    )

    assert metrics["stopped_by_kl"]
    assert metrics["epochs_completed"] == 5
    assert metrics["final_approx_kl"] > metrics["stop_kl"]
    assert metrics["optimizer_steps"] == 5 * metrics["effective_minibatches"]
    assert [row["epoch"] for row in metrics["epoch_metrics"]] == list(range(1, 6))


def test_small_kl_runs_all_sixteen_epochs_and_counts_every_optimizer_step():
    network = _FakePPONetwork(ratio_after_update=1.001)
    metrics = ppo_update(
        network,
        _buffer(),
        base_seed=42,
        iteration=3,
        entropy_coef=0.0,
        max_epochs=16,
    )

    assert not metrics["stopped_by_kl"]
    assert metrics["epochs_completed"] == 16
    assert metrics["optimizer_steps"] == 16 * metrics["effective_minibatches"]
    assert [row["epoch"] for row in metrics["epoch_metrics"]] == list(range(1, 17))
    assert network.optimizer_step_count == metrics["optimizer_steps"]


def test_fixed_kl_policy_is_reported_and_only_stop_kl_controls_early_stopping():
    network = _FakePPONetwork(ratio_after_update=1.001)
    metrics = ppo_update(
        network,
        _buffer(),
        base_seed=42,
        iteration=4,
        entropy_coef=0.0,
        max_epochs=2,
    )

    assert not metrics["stopped_by_kl"]
    assert metrics["epochs_completed"] == 2
    assert metrics["target_kl"] == 0.01
    assert metrics["stop_kl"] == 0.015


def test_ppo_rejects_more_than_sixteen_epochs():
    with np.testing.assert_raises_regex(ValueError, "between 1 and 16"):
        ppo_update(
            _FakePPONetwork(),
            _buffer(),
            base_seed=42,
            iteration=4,
            entropy_coef=0.0,
            max_epochs=17,
        )


def test_complete_gpu_copy_and_ram_batches_are_equivalent():
    buffer = _buffer(16)
    gpu_network = _FakePPONetwork(device="gpu")
    cpu_network = _FakePPONetwork(device="cpu")
    indices = np.asarray([1, 5, 9, 15], dtype=np.int64)

    with mock.patch("training.rl.ppo.effective_gpu_available_bytes", return_value=10**9):
        gpu = PPOBufferStorage(gpu_network, buffer)
    ram = PPOBufferStorage(cpu_network, buffer)
    try:
        assert gpu.location == "gpu"
        assert ram.location == "ram"
        for key in ("states", "actions", "legal_masks", "old_log_probs", "advantages", "returns"):
            np.testing.assert_array_equal(gpu.batch(indices)[key], ram.batch(indices)[key])
    finally:
        gpu.close()
        ram.close()


def test_simulated_gpu_workspace_oom_falls_back_before_any_optimizer_step():
    network = _FakePPONetwork(
        ratio_after_update=1.001,
        device="gpu",
        fail_first_eval=True,
    )
    with mock.patch("training.rl.ppo.effective_gpu_available_bytes", return_value=10**9):
        metrics = ppo_update(
            network,
            _buffer(),
            base_seed=9,
            iteration=3,
            entropy_coef=0.0,
            max_epochs=2,
        )

    assert metrics["buffer_location"] == "ram_streamed"
    assert "workspace probe" in metrics["buffer_preflight"]["fallback_reason"]
    assert metrics["optimizer_steps"] == 2 * metrics["effective_minibatches"]
    assert network.optimizer_step_count == metrics["optimizer_steps"]


def test_buffer_rejects_forced_single_option_and_illegal_observed_actions():
    single = _sample(0)
    single.legal_mask[:] = False
    single.legal_mask[0, 0] = True
    with np.testing.assert_raises_regex(ValueError, "forced or single-option"):
        PPOBuffer.from_samples([single])

    illegal = _sample(0)
    illegal.action_index = 3
    with np.testing.assert_raises_regex(ValueError, "outside its legal mask"):
        PPOBuffer.from_samples([illegal])


def _stable_network(learning_rate=0.2, seed=11):
    """Return a small zeroed policy, the shape the numerical guards act on."""
    network = PolicyNetwork(
        input_size=3,
        hidden1_size=4,
        hidden2_size=3,
        output_size=4,
        learning_rate=learning_rate,
        random_seed=seed,
        device="cpu",
    )
    for name in network.weight_names:
        getattr(network, name)[:] = 0.0
    return network


def test_non_finite_gradient_norm_is_rejected_without_touching_the_weights():
    """A NaN norm must not reach the weights: ``nan > clip`` is False."""
    network = _stable_network()
    network.W1[:] = 0.25
    before = network.snapshot_parameters()
    gradients = {name: np.ones_like(getattr(network, name)) for name in network.weight_names}

    metrics = network._apply_gradient_step(
        gradients,
        float("nan"),
        5.0,
        np.float32,
    )

    assert metrics["grad_rejected"] is True
    assert metrics["optimizer_steps"] == 0
    assert network.optimizer_step_count == 0
    for name, value in before.items():
        np.testing.assert_array_equal(network.parameter_array(name), value)


def test_infinite_gradient_norm_is_rejected_instead_of_scaled_into_nan():
    """Clipping an infinite norm would compute ``inf * 0.0`` and write NaN."""
    network = _stable_network()
    before = network.snapshot_parameters()
    gradients = {
        name: np.full_like(getattr(network, name), np.inf)
        for name in network.weight_names
    }

    metrics = network._apply_gradient_step(
        gradients,
        float("inf"),
        5.0,
        np.float32,
    )

    assert metrics["grad_rejected"] is True
    assert metrics["grad_clipped"] is False
    for name, value in before.items():
        np.testing.assert_array_equal(network.parameter_array(name), value)
        assert np.all(np.isfinite(network.parameter_array(name)))


def test_finite_gradient_step_reports_no_rejection_and_still_clips():
    """The guard is inert on a healthy step, including a clipped one."""
    network = _stable_network()
    gradients = {
        name: np.full_like(getattr(network, name), 10.0)
        for name in network.weight_names
    }
    norm = float(np.sqrt(sum(float(np.sum(g ** 2)) for g in gradients.values())))

    metrics = network._apply_gradient_step(gradients, norm, 5.0, np.float32)

    assert metrics["grad_rejected"] is False
    assert metrics["grad_clipped"] is True
    assert metrics["applied_grad_norm"] == 5.0
    assert network.optimizer_step_count == 1


def test_floored_behavior_probability_cannot_overflow_the_ratio_into_nan():
    """The rollout floors ``old_log_prob`` at ``log(tiny)``; -87 must be safe.

    With a negative advantage the PPO minimum selects the *unclipped* branch,
    so the raw ratio reaches the gradient. Unbounded, the ratio here is
    ``exp(86.64) = 4.25e37``; the advantage is chosen well past the float32
    limit of ``3.40e38 / 4.25e37 = 8.0`` so the overflow is decisive rather
    than marginal. Every illegal action would then compute ``0 * inf``.
    """
    network = _stable_network(learning_rate=0.05)
    states = np.ones((3, 1), dtype=np.float32)
    mask = np.asarray([[True], [True], [False], [False]])
    floored = np.asarray(
        [np.log(np.finfo(np.float32).tiny)],
        dtype=np.float32,
    )

    metrics = network.backward_ppo(
        states,
        [0],
        mask,
        floored,
        [-32.0],
        entropy_coef=0.01,
        clip_grad_norm=5.0,
    )

    assert np.isfinite(metrics["ratio_max"])
    assert metrics["grad_rejected"] is False
    for name in network.weight_names:
        assert np.all(np.isfinite(network.parameter_array(name))), name


def test_log_ratio_limit_bounds_the_gradient_ratio_but_not_the_reported_one():
    """The bound belongs to the gradient path; reporting stays honest."""
    network = _stable_network(learning_rate=0.0)
    states = np.ones((3, 1), dtype=np.float32)
    mask = np.asarray([[True], [True], [False], [False]])
    floored = np.asarray(
        [np.log(np.finfo(np.float32).tiny)],
        dtype=np.float32,
    )

    metrics = network.backward_ppo(
        states,
        [0],
        mask,
        floored,
        [1.0],
        entropy_coef=0.0,
        clip_grad_norm=None,
    )

    assert metrics["ratio_max"] <= np.exp(PPO_LOG_RATIO_LIMIT) * (1.0 + 1e-6)
    # log_ratio_statistics is a reporting path and must not be clamped.
    honest = log_ratio_statistics(np.asarray([-0.7]), floored)
    assert honest["ratio_max"] > np.exp(PPO_LOG_RATIO_LIMIT)


def test_diverged_epoch_is_rolled_back_and_is_not_reported_as_a_kl_stop():
    """A policy that cannot be evaluated must never reach the next rollout."""
    network = _FakePPONetwork()
    network.W1[:] = 0.5
    before = network.snapshot_parameters()
    buffer = _buffer(512)

    def explode(*_args, **_kwargs):
        raise NonFinitePolicyError("PPO full-buffer metrics produced NaN/Inf.")

    with mock.patch("training.rl.ppo.evaluate_full_buffer", side_effect=explode):
        metrics = ppo_update(
            network,
            buffer,
            base_seed=5,
            iteration=1,
            entropy_coef=0.01,
            max_epochs=16,
        )

    assert metrics["diverged_epoch"] == 1
    assert "NaN/Inf" in metrics["divergence_reason"]
    assert metrics["stopped_by_kl"] is False
    assert metrics["epochs_completed"] == 0
    assert metrics["optimizer_steps"] == 0
    assert metrics["insufficient_decisions"] is False
    # The network must be indistinguishable from its pre-epoch self, including
    # the checkpointed step counter.
    assert network.optimizer_step_count == 0
    for name, value in before.items():
        np.testing.assert_array_equal(network.parameter_array(name), value)


def test_a_healthy_update_reports_no_divergence_and_records_weight_magnitude():
    """The divergence fields stay inert, and the drift observable is present."""
    network = _FakePPONetwork()
    network.W1[:] = 0.75
    buffer = _buffer(512)

    metrics = ppo_update(
        network,
        buffer,
        base_seed=5,
        iteration=1,
        entropy_coef=0.01,
        max_epochs=2,
    )

    assert metrics["diverged_epoch"] is None
    assert metrics["divergence_reason"] is None
    assert metrics["rejected_minibatches"] == 0
    assert metrics["policy_weight_max_abs"] == 0.75
    assert metrics["log_ratio_limit"] == PPO_LOG_RATIO_LIMIT


def test_legal_logit_deficit_measures_the_gap_the_rollout_sampler_dies_on():
    """The metric must be the crash condition itself, not a proxy for it.

    A legal action underflows out of a full-support softmax exactly when its
    logit sits more than ``-log(tiny)`` below the *global* maximum, which the
    illegal actions are free to hold. Measuring the legal spread alone, or the
    global spread alone, would both miss it.
    """
    network = _FakePPONetwork()
    logits = np.array(
        [[200.0, 1.0], [10.0, 2.0], [8.0, 3.0], [0.0, 4.0]],
        dtype=np.float32,
    )
    legal_masks = np.array(
        [[False, True], [True, True], [True, False], [False, False]],
    )
    network.cache = {network.logits_key: logits}

    # Column 0: global max 200 on an illegal action, best legal 10 -> 190.
    # Column 1: the global max is itself legal -> 0.
    assert _legal_logit_deficit_max(network, legal_masks) == 190.0

    # A network that publishes no logits yields no measurement rather than an
    # exception, so no caller has to know which network class it holds.
    network.cache = {}
    assert _legal_logit_deficit_max(network, legal_masks) is None


def test_legal_logit_deficit_reduces_with_max_across_epochs():
    """One bad decision anywhere in the iteration is what matters."""
    assert _max_or_none([1.0, None, 7.0, 3.0]) == 7.0
    assert _max_or_none([None, None]) is None
    assert _max_or_none([]) is None
