"""User-facing reporting, metrics, and cumulative RL runtime profiling.

This module intentionally has no dependency on :mod:`training.rl.training_loop` so
logging, metrics persistence, and profile aggregation remain reusable without
reintroducing the rollout/orchestrator import cycle.
"""

import json
import os
from pathlib import Path
import secrets
import time

import numpy as np

from training.rl.rollout import REWARD_ZERO_EPSILON
from training.rl.constants import (
    RL_WORKER_AUTOTUNE_FRACTION,
    RL_WORKER_AUTOTUNE_MINIMUM_GAIN,
)
from training.rl.ppo import (
    POLICY_GRADIENT_CLIP_NORM,
    fixed_ppo_policy,
    ppo_is_enabled,
)
from training.rl.resume import resume_state_path
from training.rl.matchmaking import matchmaking_policy_manifest
from utils.resource_limits import MIB
from utils.runtime_status import format_duration, print_memory_report


TRAINING_METRICS_FORMAT = "domino_rl_training_metrics"
TRAINING_METRICS_VERSION = 3
TRAINING_METRIC_COLUMNS = (
    "iteration",
    "total_iterations",
    "games",
    "cumulative_games",
    "decisions",
    "wins_in_batch",
    "batch_win_rate",
    "moving_average_win_rate",
    "reward_mean",
    "reward_std",
    "reward_min",
    "reward_max",
    "good_pct",
    "neutral_pct",
    "bad_pct",
    "entropy",
    "gradient_norm_max",
    "applied_gradient_norm",
    "gradient_clipped",
    "value_loss",
    "moving_average_value_loss",
    "requested_minibatches",
    "effective_minibatches",
    "minibatch_sizes",
    "epochs_completed",
    "stopped_by_kl",
    "optimizer_steps",
    "final_approx_kl",
    "max_approx_kl",
    "final_clip_fraction",
    "final_policy_loss",
    "gradient_norm_mean",
    "buffer_location",
    "buffer_bytes",
    "selected_workers",
    "opponent_count",
    "unique_neural_opponent_count",
    "matchmaking",
    "rollout_seconds",
    "update_seconds",
    "iteration_seconds",
    "checkpoint_written",
    "checkpoint_path",
    "elapsed_training_seconds",
)


def build_training_metrics_header(
    context,
    *,
    supervised_weights_sha256,
    requested_device,
):
    """Build the immutable metadata header for an RL metrics stream."""
    training = context.training
    resources = context.resources
    execution = context.execution
    return {
        "run_configuration_sha256": (
            context.resume_configuration.run_configuration_sha256
        ),
        "run_configuration": dict(execution.run_configuration or {}),
        "training": {
            "effective_seed": int(context.effective_seed),
            "algorithm": context.algorithm,
            "total_training_games": int(training.total_training_games),
            "games_per_iteration": int(context.selected_gpi),
            "opponent_buckets": list(training.opponent_buckets),
            "difficulty_weight": float(training.difficulty_weight),
            "learning_rate": float(training.learning_rate),
            "entropy_coef": float(training.entropy_coef),
            "use_value_head": bool(training.use_value_head),
            "value_coef": float(training.value_coef),
            "gamma": float(training.gamma),
            "reward_schema": training.reward_schema,
            "reward_constants": dict(context.schema),
            "clip_grad_norm": POLICY_GRADIENT_CLIP_NORM,
            "normalize_advantages": bool(training.normalize_advantages),
            "weight_decay": float(training.weight_decay),
            "dropout_rate": float(training.dropout_rate),
            "moving_average_window": int(execution.moving_average_window),
            "requested_device": requested_device,
            "resolved_device": context.network.device,
            "requested_workers": resources.workers,
            "selected_workers": int(context.selected_workers),
            "supervised_weights_sha256": supervised_weights_sha256,
            "ppo_configuration": fixed_ppo_policy(training.ppo_max_epochs),
            "opponent_pool": context.runner.opponent_pool.manifest(),
            "matchmaking_policy": matchmaking_policy_manifest(),
            "checkpoint_archive": context.checkpoint_archive.manifest(),
        },
    }


def build_iteration_metrics_row(
    context,
    state,
    *,
    iteration,
    games,
    batch_size,
    wins,
    moving_win_rate,
    reward_summary,
    gradient_metrics,
    ppo_metrics,
    rollout_elapsed,
    update_elapsed,
    checkpoint_written,
    checkpoint_path,
    iteration_started,
    matchmaking,
):
    """Build the complete in-memory metrics row for one RL iteration."""
    moving_value_loss = (
        float(sum(state.value_loss_window) / len(state.value_loss_window))
        if state.value_loss_window else None
    )
    reward = reward_summary or {}
    return {
        "iteration": int(iteration),
        "total_iterations": int(context.final_iteration),
        "games": int(games),
        "cumulative_games": int(state.completed_training_games),
        "cumulative_training_games": int(state.completed_training_games),
        "total_training_games": int(context.training.total_training_games),
        "games_per_iteration": int(context.selected_gpi),
        "decision_sample_count": int(batch_size),
        "decisions": int(batch_size),
        "wins_in_batch": int(wins),
        "batch_win_rate": float(wins / games),
        "moving_average_win_rate": float(moving_win_rate),
        "reward_mean": reward.get("reward_mean"),
        "reward_std": reward.get("reward_std"),
        "reward_min": reward.get("reward_min"),
        "reward_max": reward.get("reward_max"),
        "good_pct": reward.get("good_pct"),
        "neutral_pct": reward.get("neutral_pct"),
        "bad_pct": reward.get("bad_pct"),
        "entropy": None if gradient_metrics is None else float(gradient_metrics["entropy"]),
        "grad_norm": None if gradient_metrics is None else float(gradient_metrics["grad_norm"]),
        "applied_grad_norm": None if gradient_metrics is None else float(gradient_metrics["applied_grad_norm"]),
        "grad_clipped": False if gradient_metrics is None else bool(gradient_metrics["grad_clipped"]),
        "value_loss": None if gradient_metrics is None else gradient_metrics["value_loss"],
        "moving_average_value_loss": moving_value_loss,
        "requested_minibatches": None if ppo_metrics is None else int(ppo_metrics["requested_minibatches"]),
        "effective_minibatches": None if ppo_metrics is None else int(ppo_metrics["effective_minibatches"]),
        "minibatch_sizes": None if ppo_metrics is None else ppo_metrics["minibatch_sizes"],
        "epochs_completed": None if ppo_metrics is None else int(ppo_metrics["epochs_completed"]),
        "stopped_by_kl": False if ppo_metrics is None else bool(ppo_metrics["stopped_by_kl"]),
        "optimizer_steps": 0 if gradient_metrics is None else int(
            ppo_metrics["optimizer_steps"] if ppo_metrics is not None else 1
        ),
        "final_approx_kl": None if ppo_metrics is None else float(ppo_metrics["final_approx_kl"]),
        "max_approx_kl": None if ppo_metrics is None else float(ppo_metrics["max_approx_kl"]),
        "final_clip_fraction": None if ppo_metrics is None else float(ppo_metrics["final_clip_fraction"]),
        "final_entropy": None if ppo_metrics is None else float(ppo_metrics["final_entropy"]),
        "final_policy_loss": None if ppo_metrics is None else float(ppo_metrics["final_policy_loss"]),
        "gradient_norm_mean": None if ppo_metrics is None else float(ppo_metrics["gradient_norm_mean"]),
        "gradient_norm_max": None if ppo_metrics is None else float(ppo_metrics["gradient_norm_max"]),
        "buffer_location": None if ppo_metrics is None else ppo_metrics["buffer_location"],
        "buffer_bytes": 0 if ppo_metrics is None else int(ppo_metrics["buffer_bytes"]),
        "selected_workers": int(context.runner.worker_count),
        "opponent_count": int(context.runner.opponent_pool.size),
        "unique_neural_opponent_count": int(
            context.runner.opponent_pool.unique_neural_opponent_count
        ),
        "matchmaking": matchmaking,
        "rollout_seconds": float(rollout_elapsed),
        "ppo_seconds": float(update_elapsed if ppo_metrics else 0.0),
        "rollout_duration_s": float(rollout_elapsed),
        "update_duration_s": float(update_elapsed),
        "total_iteration_seconds": float(time.perf_counter() - iteration_started),
        "iteration_duration_s": float(time.perf_counter() - iteration_started),
        "checkpoint_written": bool(checkpoint_written),
        "checkpoint_path": checkpoint_path,
        "elapsed_training_s": float(
            time.perf_counter() - context.training_perf_started
        ),
        "rl_training_algorithm": context.algorithm,
    }


def _autotune_summary(context):
    """Return the stable compatibility view of worker autotuning."""
    worker_results = context.adaptive_tuning.get("worker_results", [])
    return {
        "optimal_workers": int(context.selected_workers),
        "candidate_workers": [
            int(row["requested_workers"]) for row in worker_results
        ],
        "benchmark_fraction": RL_WORKER_AUTOTUNE_FRACTION,
        "minimum_gain": RL_WORKER_AUTOTUNE_MINIMUM_GAIN,
        "iterations_per_test": None,
        "games_per_test": int(
            context.adaptive_tuning.get("worker_test_games", 0)
        ),
        "reused_iteration_count": 0,
        "reused_game_count": 0,
        "discarded_game_count": sum(
            int(row.get("actual_games", 0)) for row in worker_results
        ),
        "attempts": worker_results,
    }


def build_training_summary(
    session,
    *,
    actual_final_iteration,
    stopped_by_shutdown,
    final_runtime_workers,
    opponent_count,
    unique_neural_opponent_count,
    bucket_sizes,
    runtime_profile_delta,
    elapsed_time,
):
    """Build the public result of a complete or interrupted RL invocation."""
    context = session.context
    state = session.state
    training = context.training
    resources = context.resources
    execution = context.execution
    return {
        "iterations": int(actual_final_iteration),
        "rl_iterations_completed": int(actual_final_iteration),
        "completed_iterations_this_run": int(
            state.completed_iterations_this_invocation
        ),
        "start_iteration": int(session.start_iteration),
        "start_training_games": int(
            state.completed_training_games - state.completed_this_invocation
        ),
        "games_per_iteration": int(context.selected_gpi),
        "requested_games_per_iteration": int(training.gpi),
        "total_training_games": int(training.total_training_games),
        "completed_training_games": int(state.completed_training_games),
        "invocation_target_training_games": int(context.invocation_target_games),
        "shutdown_requested": bool(stopped_by_shutdown),
        "opponent_buckets": list(training.opponent_buckets),
        "difficulty_weight": float(training.difficulty_weight),
        "learning_rate": training.learning_rate,
        "entropy_coef": training.entropy_coef,
        "use_value_head": training.use_value_head,
        "value_coef": training.value_coef if training.use_value_head else None,
        "gamma": training.gamma,
        "reward_schema": training.reward_schema,
        "clip_grad_norm": POLICY_GRADIENT_CLIP_NORM,
        "normalize_advantages": training.normalize_advantages,
        "weight_decay": training.weight_decay,
        "dropout_rate": training.dropout_rate,
        "moving_average_window": execution.moving_average_window,
        "seed": training.seed,
        "effective_seed": context.effective_seed,
        "device": context.network.device,
        "requested_device": session.requested_device,
        "device_fallback_reason": session.device_fallback_reason,
        "requested_workers": resources.workers,
        "selected_workers": int(context.selected_workers),
        "final_workers": int(final_runtime_workers),
        "autotune": _autotune_summary(context),
        "adaptive_tuning": context.adaptive_tuning,
        "adaptive_tuning_path": str(resources.adaptive_tuning_path),
        "parallel": context.parallel_summary,
        "rl_weights_path": str(state.final_weights_path),
        "metrics_output_path": str(session.metrics_path),
        "initialization_source": session.initialization_source,
        "fresh_from_sl": bool(execution.fresh_from_sl),
        "numbered_checkpoints": bool(execution.numbered_checkpoints),
        "total_decision_samples": int(state.total_decision_samples),
        "trainable_decisions_seen": int(state.total_decision_samples),
        "policy_updates_completed": int(state.policy_updates_completed),
        "decisions_per_game": float(
            state.total_decision_samples / max(1, state.completed_training_games)
        ),
        "clipped_iteration_count": int(state.clipped_iteration_count),
        "clipped_iteration_rate": float(
            state.clipped_iteration_count / max(1, actual_final_iteration)
        ),
        "opponent_count": int(opponent_count),
        "unique_neural_opponent_count": int(unique_neural_opponent_count),
        "opponent_bucket_sizes": dict(bucket_sizes),
        "total_rollout_duration_s": float(state.total_rollout_duration_s),
        "total_update_duration_s": float(state.total_update_duration_s),
        "optimizer_step_count": int(context.network.optimizer_step_count),
        "elapsed_rl_seconds": float(
            context.restored_elapsed_rl_seconds
            + time.perf_counter() - context.training_perf_started
        ),
        "resume_state_path": (
            str(resume_state_path(state.final_weights_path))
            if execution.numbered_checkpoints else None
        ),
        "rl_training_algorithm": context.algorithm,
        "run_configuration_sha256": (
            context.resume_configuration.run_configuration_sha256
        ),
        "ppo_configuration": fixed_ppo_policy(training.ppo_max_epochs),
        "opponent_pool": context.runner.opponent_pool.manifest(),
        "checkpoint_archive": context.checkpoint_archive.manifest(),
        "runtime_profile_delta": runtime_profile_delta,
        "duration_s": elapsed_time,
    }


class RLTrainingReporter:
    """Keep presentation concerns out of the RL orchestration loop."""

    def __init__(self, *, quiet=False, status_callback=None):
        self.quiet = bool(quiet)
        if status_callback is not None:
            self._status_callback = status_callback
        elif self.quiet:
            self._status_callback = lambda _message: None
        else:
            self._status_callback = lambda message: print(message, flush=True)

    def status(self, message):
        """Emit a compact/pipeline-safe status message."""
        self._status_callback(message)

    def device_fallback(self, reason):
        if reason:
            self.status(
                "RL memory safety: automatic GPU selection fell back to CPU "
                f"because {reason}."
            )

    def tuning_header(self):
        self.status("-" * 70)
        self.status("RL rollout-worker tuning")
        self.status("-" * 70)

    def update_configuration(
        self,
        *,
        algorithm,
        use_value_head,
        value_coef,
        dropout_rate,
        weight_decay,
        ppo_max_epochs,
        gpi,
        opponent_buckets,
        difficulty_weight,
    ):
        """Describe the selected fixed algorithm policy once per invocation."""
        policy = fixed_ppo_policy(ppo_max_epochs)
        fixed = policy["fixed_policy"]
        self.status(f"Fixed GPI: {int(gpi)}.")
        self.status(
            "Opponent buckets: " + ", ".join(opponent_buckets)
            + f" | difficulty weight: {float(difficulty_weight):g}."
        )
        self.status("RL update configuration:")
        self.status(
            f"  value head: {'on' if use_value_head else 'off'}"
            + (f" | value coefficient: {value_coef:g}" if use_value_head else "")
        )
        self.status(
            "  regularization: "
            + (f"dropout {dropout_rate:g}" if dropout_rate > 0 else "dropout off")
            + " | "
            + (
                f"decoupled weight decay {weight_decay:g}"
                if weight_decay > 0
                else "weight decay off"
            )
        )
        if ppo_is_enabled(ppo_max_epochs):
            self.status(
                f"  algorithm: {algorithm} | clip epsilon: "
                f"{fixed['clip_epsilon']:.2f} | target KL: "
                f"{fixed['target_kl']:.3f} | stop KL: {fixed['stop_kl']:.3f}"
            )
            self.status(
                f"  max epochs: {ppo_max_epochs} | minibatches: adaptive, "
                f"{fixed['min_minibatches']} to {fixed['max_minibatches']} | "
                "preferred buffer: GPU | fallback: RAM"
            )
        else:
            self.status(
                f"  algorithm: {algorithm} | one full-buffer policy-gradient "
                "update per iteration"
            )
            self.status(
                "  PPO minibatches, ratios, clipping, KL control, and "
                "post-update full-buffer evaluation: disabled"
            )
        self.status("-" * 70)

    def resource_preflight(
        self,
        *,
        requested_device,
        selected_device,
        estimated_host_bytes,
    ):
        if self.quiet:
            return
        print_memory_report("RL training startup memory")
        print(
            "RL resource preflight: "
            f"requested device={requested_device!r}, "
            f"selected device={selected_device!r}, estimated peak host "
            f"allocation {estimated_host_bytes / MIB:.1f} MiB."
        )

    def worker_cap(self, requested, selected, reason):
        self.status(
            f"Selected workers reduced from {requested} to {selected} by "
            f"current resource preflight: {reason}."
        )

    def resumed(self, iteration, games, pool_size):
        self.status(
            f"Resuming RL after iteration {iteration} and {games} real games; "
            f"restored {pool_size} opponent-pool snapshot(s)."
        )

    def rollout_fallback(self, iteration, rollout_info):
        self.status(
            f"RL iteration {iteration} retained completed games and reduced "
            f"workers to {rollout_info.final_workers}: "
            f"{rollout_info.fallback_history[-1]['reason']}."
        )

    def checkpoint(self, path, completed_games, total_games, elapsed):
        if not self.quiet:
            print(
                f"  [checkpoint] saved {path} | "
                f"{completed_games}/{total_games} games | time since previous "
                f"checkpoint: {format_duration(elapsed)}"
            )

    def iteration(
        self,
        *,
        iteration,
        log_interval,
        games,
        completed_games,
        total_games,
        reward_summary,
        wins,
        moving_win_rate,
        win_window_size,
        opponent_buckets,
        difficulty_weight,
        opponent_count,
        unique_neural_opponent_count,
        bucket_results,
        gradient_metrics,
        use_value_head,
        value_loss_window,
        ppo_window,
        ppo_max_epochs,
    ):
        if self.quiet or iteration % log_interval:
            return
        if reward_summary is None:
            print(f"Iteration {iteration} | {games} games | no real policy decisions")
            return
        bucket_text = ", ".join(
            f"{name} {value['games']} games/{value['wins'] / value['games']:.1%} wins"
            for name, value in bucket_results.items()
        )
        print(
            f"Iteration {iteration} | games {games} | cumulative "
            f"{completed_games}/{total_games} | reward mean/std/min/max: "
            f"{reward_summary['reward_mean']:+.2f}/"
            f"{reward_summary['reward_std']:.2f}/"
            f"{reward_summary['reward_min']:+.2f}/"
            f"{reward_summary['reward_max']:+.2f} | good/neutral/bad: "
            f"{reward_summary['good_pct']:.0f}%/"
            f"{reward_summary['neutral_pct']:.0f}%/"
            f"{reward_summary['bad_pct']:.0f}% | aggregate mixture wins: "
            f"{wins}/{games} (avg/{win_window_size}: {moving_win_rate:.1%})"
            f" | opponents: {opponent_count} ({unique_neural_opponent_count} "
            f"neural) | grad: {_gradient_log_text(gradient_metrics)}"
        )
        print(
            "  Matchmaking: buckets " + ",".join(opponent_buckets)
            + f" | difficulty weight {difficulty_weight:g} | {bucket_text}"
        )
        value_predictions = gradient_metrics.get("value_predictions_before_update")
        if use_value_head and value_predictions is not None:
            print(
                "  Value head: pre-update V(s) mean/std/min/max "
                f"{value_predictions['mean']:+.3f}/"
                f"{value_predictions['std']:.3f}/"
                f"{value_predictions['min']:+.3f}/"
                f"{value_predictions['max']:+.3f} over "
                f"{value_predictions['sample_count']} decisions | value loss "
                f"{gradient_metrics['value_loss']:.3f} "
                f"(avg/{len(value_loss_window)}: "
                f"{sum(value_loss_window) / len(value_loss_window):.3f})"
            )
        if ppo_is_enabled(ppo_max_epochs):
            _print_ppo_window(ppo_window)

    def complete(self, elapsed, weights_path):
        if not self.quiet:
            print(f"\nTraining complete. Total elapsed time: {format_duration(elapsed)}.")
            print(f"Final weights: {weights_path}")


class RLRuntimeProfile:
    """Accumulate one RL invocation's hierarchical runtime profile."""

    def __init__(self):
        self.started = time.perf_counter()
        self.sections = {}
        self.ppo_sections = {}
        self.rollout_worker = {}
        self.ppo_optimizer_step = {}
        self.ppo_full_buffer_evaluation = {}

    def add(self, section, seconds):
        """Add elapsed seconds to one top-level runtime section."""
        self.sections[section] = self.sections.get(section, 0.0) + float(seconds)

    @staticmethod
    def merge_numeric_tree(target, source):
        """Recursively add numeric counters while preserving nested schemas."""
        for key, value in source.items():
            if isinstance(value, dict):
                RLRuntimeProfile.merge_numeric_tree(
                    target.setdefault(key, {}),
                    value,
                )
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                target[key] = target.get(key, 0) + value

    def merge_rollout_worker(self, source):
        """Merge sampled worker-side rollout timings and counters."""
        self.merge_numeric_tree(self.rollout_worker, source)

    def merge_ppo_metrics(self, metrics):
        """Merge PPO phase totals and its two detailed nested profiles."""
        for name, seconds in metrics["runtime_timing_seconds"].items():
            if name != "total":
                self.ppo_sections[name] = (
                    self.ppo_sections.get(name, 0.0) + float(seconds)
                )
        detail = metrics.get("runtime_profile_detail", {})
        self.merge_numeric_tree(
            self.ppo_optimizer_step,
            detail.get("optimizer_step", {}),
        )
        self.merge_numeric_tree(
            self.ppo_full_buffer_evaluation,
            detail.get("full_buffer_evaluation", {}),
        )

    def accounted_seconds(self):
        """Return the sum already assigned to top-level runtime sections."""
        return sum(self.sections.values())

    def finish(self, *, games, iterations, decisions, optimizer_steps):
        """Close accounting and return the stable public profile schema."""
        total_seconds = time.perf_counter() - self.started
        self.sections["unaccounted"] = max(
            0.0,
            total_seconds - self.accounted_seconds(),
        )
        return {
            "execution_count": 1,
            "games": int(games),
            "iterations": int(iterations),
            "decisions": int(decisions),
            "optimizer_steps": int(optimizer_steps),
            "execution_seconds": float(total_seconds),
            "sections_seconds": {
                name: float(seconds)
                for name, seconds in self.sections.items()
            },
            "ppo_sections_seconds": {
                name: float(seconds)
                for name, seconds in self.ppo_sections.items()
            },
            "rollout_worker": self.rollout_worker,
            "ppo_optimizer_step": self.ppo_optimizer_step,
            "ppo_full_buffer_evaluation": self.ppo_full_buffer_evaluation,
        }


def _reward_signal_summary(samples, xp=None):
    """Return compact diagnostics for finalized decision rewards.

    ``reward_std`` disambiguates a falling value loss from a merely
    low-variance batch: since a value head that has not learned anything
    predicts close to the batch mean, its loss is approximately
    ``0.5 * reward_std ** 2`` -- logging the standard deviation next to the
    loss makes that identity checkable instead of hidden behind a noisy
    scalar.

    ``xp`` should be the training run's resolved array backend (``train()``
    passes ``network.xp``); it defaults to NumPy for direct callers, which is
    fine here since this is small-scale summary math, not the training path.
    """
    if xp is None:
        xp = np
    rewards = xp.asarray([sample.policy_reward for sample in samples], dtype=float)
    local_rewards = xp.asarray([sample.local_reward for sample in samples], dtype=float)
    total = rewards.size

    good = xp.sum(rewards > REWARD_ZERO_EPSILON)
    neutral = xp.sum(xp.abs(rewards) <= REWARD_ZERO_EPSILON)
    bad = xp.sum(rewards < -REWARD_ZERO_EPSILON)

    return {
        "reward_mean": float(xp.mean(rewards)),
        "reward_std": float(xp.std(rewards)),
        "reward_min": float(xp.min(rewards)),
        "reward_max": float(xp.max(rewards)),
        "local_mean": float(xp.mean(local_rewards)),
        "good_pct": float(100.0 * good / total),
        "neutral_pct": float(100.0 * neutral / total),
        "bad_pct": float(100.0 * bad / total),
    }


def _gradient_log_text(metrics):
    """Return a compact gradient-norm string for the iteration log."""
    suffix = " clipped" if metrics.get("grad_clipped") else ""
    return f"{metrics['grad_norm']:.2f}{suffix}"


def _new_parallel_summary(requested_workers):
    """Return mutable aggregate metadata for all RL worker-pool phases."""
    return {
        "requested_workers": requested_workers,
        "initial_workers": None,
        "final_workers": None,
        "peak_worker_rss_mb": 0.0,
        "peak_total_children_rss_mb": 0.0,
        "min_available_memory_mb": None,
        "fallback_count": 0,
        "fallback_history": [],
        "attempted_worker_counts": [],
        "safety_capped": False,
        "memory_monitoring_available": True,
        "workers_cpu_only": True,
        "rollout_batches": 0,
    }


def _merge_parallel_summary(summary, run_info, *, phase, iteration):
    """Accumulate one rollout/evaluation pool run into the public summary."""
    if summary["initial_workers"] is None:
        summary["initial_workers"] = run_info.initial_workers
    summary["final_workers"] = run_info.final_workers
    summary["peak_worker_rss_mb"] = max(
        summary["peak_worker_rss_mb"],
        run_info.peak_worker_rss_mb,
    )
    summary["peak_total_children_rss_mb"] = max(
        summary["peak_total_children_rss_mb"],
        run_info.peak_total_children_rss_mb,
    )
    available = run_info.min_available_memory_mb
    if available is not None:
        current = summary["min_available_memory_mb"]
        summary["min_available_memory_mb"] = (
            available if current is None else min(current, available)
        )
    summary["fallback_count"] += run_info.fallback_count
    for item in run_info.fallback_history:
        tagged = dict(item)
        tagged["rl_phase"] = phase
        tagged["iteration"] = int(iteration)
        summary["fallback_history"].append(tagged)
    summary["attempted_worker_counts"].extend(run_info.attempted_worker_counts)
    summary["safety_capped"] = summary["safety_capped"] or run_info.safety_capped
    summary["memory_monitoring_available"] = (
        summary["memory_monitoring_available"]
        and run_info.memory_monitoring_available
    )
    summary[f"{phase}_batches"] += 1


def _print_ppo_window(rows):
    """Print the requested ten-iteration PPO aggregate without minibatch chatter."""
    rows = list(rows)
    if not rows:
        return
    count = len(rows)
    effective = [row["effective_minibatches"] for row in rows]
    epochs = [row["epochs_completed"] for row in rows]
    buffer_bytes = [row["buffer_bytes"] for row in rows]
    print(
        f"  PPO/{count}: GPI {rows[-1]['games']} | decisions "
        f"{sum(row['decisions'] for row in rows)} total/"
        f"{np.mean([row['decisions'] for row in rows]):.1f} avg | "
        f"minibatches requested {np.mean([row['requested_minibatches'] for row in rows]):.1f} avg, "
        f"effective {np.mean(effective):.1f}/{min(effective)}/{max(effective)} avg/min/max"
    )
    print(
        f"  PPO/{count}: optimizer steps {sum(row['optimizer_steps'] for row in rows)} total/"
        f"{np.mean([row['optimizer_steps'] for row in rows]):.1f} avg | epochs "
        f"{np.mean(epochs):.1f}/{min(epochs)}/{max(epochs)} avg/min/max | "
        f"KL stops {sum(row['stopped_by_kl'] for row in rows)}/{count} | final KL "
        f"{np.mean([row['final_approx_kl'] for row in rows]):.5f} avg/"
        f"{max(row['final_approx_kl'] for row in rows):.5f} max"
    )
    print(
        f"  PPO/{count}: clip fraction {np.mean([row['final_clip_fraction'] for row in rows]):.3f} | "
        f"policy loss {np.mean([row['final_policy_loss'] for row in rows]):+.4f} | "
        f"entropy {np.mean([row['final_entropy'] for row in rows]):.4f} | grad norm "
        f"{np.mean([row['gradient_norm_mean'] for row in rows]):.3f} avg/"
        f"{max(row['gradient_norm_max'] for row in rows):.3f} max"
    )
    value_rows = [row for row in rows if row.get("final_value_loss") is not None]
    if value_rows:
        explained = [
            row["final_explained_variance"]
            for row in value_rows
            if row.get("final_explained_variance") is not None
        ]
        explained_text = (
            f"{np.mean(explained):+.3f} avg"
            if explained else "undefined"
        )
        print(
            f"  PPO/{count}: value loss "
            f"{np.mean([row['final_value_loss'] for row in value_rows]):.4f} | "
            f"value clip fraction "
            f"{np.mean([row['final_value_clip_fraction'] for row in value_rows]):.3f} | "
            f"explained variance {explained_text}"
        )
    gpu_count = sum(row["buffer_location"] == "gpu" for row in rows)
    print(
        f"  PPO/{count}: buffer GPU {gpu_count}, RAM {count - gpu_count} | bytes "
        f"{np.mean(buffer_bytes):.0f} avg/{max(buffer_bytes)} max | PPO update "
        f"{sum(row['ppo_seconds'] for row in rows):.2f}s total/"
        f"{np.mean([row['ppo_seconds'] for row in rows]):.3f}s avg | rollout "
        f"{sum(row['rollout_seconds'] for row in rows):.2f}s total/"
        f"{np.mean([row['rollout_seconds'] for row in rows]):.3f}s avg"
    )


def _metrics_header(metadata):
    return {
        "format": TRAINING_METRICS_FORMAT,
        "version": TRAINING_METRICS_VERSION,
        "columns": list(TRAINING_METRIC_COLUMNS),
        "metadata": dict(metadata or {}),
    }


def _metric_values(row):
    """Project one verbose in-memory row onto the compact stable schema."""
    values = {
        "iteration": row["iteration"],
        "total_iterations": row["total_iterations"],
        "games": row["games"],
        "cumulative_games": row["cumulative_games"],
        "decisions": row["decisions"],
        "wins_in_batch": row["wins_in_batch"],
        "batch_win_rate": row["batch_win_rate"],
        "moving_average_win_rate": row["moving_average_win_rate"],
        "reward_mean": row["reward_mean"],
        "reward_std": row["reward_std"],
        "reward_min": row["reward_min"],
        "reward_max": row["reward_max"],
        "good_pct": row["good_pct"],
        "neutral_pct": row["neutral_pct"],
        "bad_pct": row["bad_pct"],
        "entropy": (
            row["final_entropy"]
            if row.get("final_entropy") is not None
            else row.get("entropy")
        ),
        "gradient_norm_max": (
            row["gradient_norm_max"]
            if row.get("gradient_norm_max") is not None
            else row.get("grad_norm")
        ),
        "applied_gradient_norm": row.get("applied_grad_norm"),
        "gradient_clipped": row["grad_clipped"],
        "value_loss": row["value_loss"],
        "moving_average_value_loss": row["moving_average_value_loss"],
        "requested_minibatches": row["requested_minibatches"],
        "effective_minibatches": row["effective_minibatches"],
        "minibatch_sizes": row["minibatch_sizes"],
        "epochs_completed": row["epochs_completed"],
        "stopped_by_kl": row["stopped_by_kl"],
        "optimizer_steps": row["optimizer_steps"],
        "final_approx_kl": row["final_approx_kl"],
        "max_approx_kl": row["max_approx_kl"],
        "final_clip_fraction": row["final_clip_fraction"],
        "final_policy_loss": row["final_policy_loss"],
        "gradient_norm_mean": row["gradient_norm_mean"],
        "buffer_location": row["buffer_location"],
        "buffer_bytes": row["buffer_bytes"],
        "selected_workers": row["selected_workers"],
        "opponent_count": row["opponent_count"],
        "unique_neural_opponent_count": row[
            "unique_neural_opponent_count"
        ],
        "matchmaking": row["matchmaking"],
        "rollout_seconds": row["rollout_seconds"],
        "update_seconds": row["update_duration_s"],
        "iteration_seconds": row["iteration_duration_s"],
        "checkpoint_written": row["checkpoint_written"],
        "checkpoint_path": row["checkpoint_path"],
        "elapsed_training_seconds": row["elapsed_training_s"],
    }
    return [values[column] for column in TRAINING_METRIC_COLUMNS]


def read_training_metrics(path):
    """Read the current metrics header and decode per-iteration dictionaries."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Training metrics file is empty: {path}.")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Training metrics header is invalid: {path}.") from exc
    if (
        not isinstance(header, dict)
        or header.get("format") != TRAINING_METRICS_FORMAT
        or header.get("version") != TRAINING_METRICS_VERSION
        or header.get("columns") != list(TRAINING_METRIC_COLUMNS)
    ):
        raise ValueError(
            f"Unsupported training metrics format in {path}; "
            f"expected v{TRAINING_METRICS_VERSION}."
        )
    rows = []
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        try:
            values = json.loads(line)
        except json.JSONDecodeError:
            if line_number == len(lines):
                break
            raise ValueError(
                f"Training metrics row {line_number} is invalid."
            )
        if not isinstance(values, list) or len(values) != len(
            TRAINING_METRIC_COLUMNS
        ):
            raise ValueError(
                f"Training metrics row {line_number} has the wrong column count."
            )
        rows.append(dict(zip(TRAINING_METRIC_COLUMNS, values)))
    return header, rows


def _prepare_metrics_file(path, start_iteration, metadata=None):
    """Create a current trace or truncate it to the exact resumed iteration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _metrics_header(metadata)
    retained = []
    if start_iteration and path.is_file():
        existing_header, existing_rows = read_training_metrics(path)
        existing_hash = existing_header.get("metadata", {}).get(
            "run_configuration_sha256"
        )
        requested_hash = header.get("metadata", {}).get(
            "run_configuration_sha256"
        )
        if existing_hash != requested_hash:
            raise ValueError(
                "Training metrics configuration hash does not match the "
                "resumed run."
            )
        header = existing_header
        for row in existing_rows:
            if int(row["iteration"]) <= int(start_iteration):
                retained.append(row)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
    )
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(header, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
            for row in retained:
                values = [row[column] for column in TRAINING_METRIC_COLUMNS]
                stream.write(
                    json.dumps(values, separators=(",", ":"), allow_nan=False)
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_metrics_row(stream, row):
    """Append and durably flush one RL metrics row."""
    stream.write(
        json.dumps(
            _metric_values(row),
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    stream.flush()
    os.fsync(stream.fileno())
