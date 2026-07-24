"""Focused correctness tests for masked PPO and its decision-buffer storage."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np

from agents.rl_nn import PolicyNetwork
from training.ppo import (
    PPOBuffer,
    PPOBufferStorage,
    clipped_surrogate,
    effective_minibatches,
    log_ratio_statistics,
    minibatch_indices,
    normalize_advantages,
    ppo_update,
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
        self.optimizer_step_count = 0
        self.cache = {}
        self.W1 = np.zeros((2, 3), dtype=np.float32)
        self.b1 = np.zeros((2, 1), dtype=np.float32)
        self.W2 = np.zeros((2, 2), dtype=np.float32)
        self.b2 = np.zeros((2, 1), dtype=np.float32)
        self.W3 = np.zeros((4, 2), dtype=np.float32)
        self.b3 = np.zeros((4, 1), dtype=np.float32)

    def evaluate_actions(self, states, legal_masks, actions):
        self.eval_calls += 1
        if self.fail_first_eval and self.eval_calls == 1:
            raise MemoryError("simulated CUDA workspace OOM")
        count = int(np.asarray(actions).size)
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

    actual = clipped_surrogate(ratios, advantages, clip_epsilon=0.2)

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


def test_requested_minibatches_match_the_required_gpi_table():
    expected = {100: 4, 200: 4, 400: 4, 600: 5, 800: 7, 1000: 8, 2000: 16}
    assert {gpi: requested_minibatches(gpi) for gpi in expected} == expected


def test_minibatch_partitions_never_drop_duplicate_or_empty_indices():
    for decision_count, requested in ((1, 1), (127, 4), (128, 4), (513, 4), (2051, 16)):
        effective = effective_minibatches(decision_count, requested)
        parts = minibatch_indices(decision_count, effective, seed=987)
        combined = np.concatenate(parts)
        assert all(len(part) > 0 for part in parts)
        assert len(combined) == decision_count
        assert len(np.unique(combined)) == decision_count
        np.testing.assert_array_equal(np.sort(combined), np.arange(decision_count))
        assert max(map(len, parts)) - min(map(len, parts)) <= 1


def test_advantages_are_normalized_once_globally_and_zero_std_is_safe():
    normalized, std_zero, raw_mean, raw_std = normalize_advantages([1, 2, 3, 4])
    assert not std_zero
    assert raw_mean == 2.5
    assert raw_std > 0
    assert abs(float(normalized.mean())) < 1e-7
    assert np.isclose(float(normalized.std()), 1.0, atol=1e-6)

    constant, std_zero, _mean, raw_std = normalize_advantages([7, 7, 7])
    assert std_zero
    assert raw_std == 0.0
    assert np.all(constant == 0.0)
    assert np.all(np.isfinite(constant))


def test_kl_early_stop_occurs_only_after_the_completed_epoch():
    network = _FakePPONetwork(ratio_after_update=1.30)
    metrics = ppo_update(
        network,
        _buffer(),
        actual_games=100,
        base_seed=42,
        iteration=1,
        entropy_coef=0.0,
        clip_grad_norm=5.0,
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
            1.001 if optimizer_steps < 10 else 1.30
        )
    )
    metrics = ppo_update(
        network,
        _buffer(),
        actual_games=100,
        base_seed=42,
        iteration=2,
        entropy_coef=0.0,
        clip_grad_norm=5.0,
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
        actual_games=100,
        base_seed=42,
        iteration=3,
        entropy_coef=0.0,
        clip_grad_norm=5.0,
        max_epochs=16,
    )

    assert not metrics["stopped_by_kl"]
    assert metrics["epochs_completed"] == 16
    assert metrics["optimizer_steps"] == 16 * metrics["effective_minibatches"]
    assert [row["epoch"] for row in metrics["epoch_metrics"]] == list(range(1, 17))
    assert network.optimizer_step_count == metrics["optimizer_steps"]


def test_ppo_rejects_more_than_sixteen_epochs():
    with np.testing.assert_raises_regex(ValueError, "between one and 16"):
        ppo_update(
            _FakePPONetwork(),
            _buffer(),
            actual_games=100,
            base_seed=42,
            iteration=4,
            entropy_coef=0.0,
            clip_grad_norm=5.0,
            max_epochs=17,
        )


def test_complete_gpu_copy_and_ram_batches_are_equivalent():
    buffer = _buffer(16)
    gpu_network = _FakePPONetwork(device="gpu")
    cpu_network = _FakePPONetwork(device="cpu")
    indices = np.asarray([1, 5, 9, 15], dtype=np.int64)

    with mock.patch("training.ppo.effective_gpu_available_bytes", return_value=10**9):
        gpu = PPOBufferStorage(gpu_network, buffer, prefer_gpu=True)
    ram = PPOBufferStorage(cpu_network, buffer, prefer_gpu=True)
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
    with mock.patch("training.ppo.effective_gpu_available_bytes", return_value=10**9):
        metrics = ppo_update(
            network,
            _buffer(),
            actual_games=100,
            base_seed=9,
            iteration=3,
            entropy_coef=0.0,
            clip_grad_norm=5.0,
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
