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

from training.rl.pool import CHAMPION_VS_HEURISTIC_BUCKET
from training.rl.champion_evaluation import (
    champion_evaluation_policy_manifest,
)
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
from training.rl.matchmaking import (
    matchmaking_component_budgets,
    matchmaking_policy_manifest,
)
from utils.resource_limits import MIB
from utils.runtime_status import format_duration, print_memory_report


TRAINING_METRICS_FORMAT = "domino_rl_training_metrics"
TRAINING_METRICS_VERSION = 6
BUCKET_RESULT_COLUMNS = ("games", "wins", "losses")
EVENT_RESULT_COLUMNS = (
    "opponent_draws",
    "opponent_passes",
    "learner_draws",
    "learner_passes",
)
TRAINING_METRIC_COLUMNS = (
    "iteration",
    "total_iterations",
    "games",
    "cumulative_games",
    "decisions",
    "normal_decisions",
    "restart_decisions",
    "restart_captured_states",
    "restart_continuation_episodes",
    "cumulative_normal_decisions",
    "cumulative_restart_decisions",
    "cumulative_restart_episodes",
    "restart_wins",
    "restart_losses",
    "normal_event_stats",
    "restart_event_stats",
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
    "target_decisions_per_minibatch",
    "min_decisions_per_minibatch",
    "decisions_omitted_per_epoch",
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
    "bucket_results",
    "rollout_seconds",
    "restart_seconds",
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
            "ruleset_name": training.ruleset_name,
            "algorithm": context.algorithm,
            "total_training_games": int(training.total_training_games),
            "games_per_iteration": int(context.selected_gpi),
            "opponent_buckets": list(training.opponent_buckets),
            "difficulty_weight": float(training.difficulty_weight),
            "opponent_decision_restarts": bool(
                training.opponent_decision_restarts
            ),
            "learning_rate": float(training.learning_rate),
            "entropy_coef": float(training.entropy_coef),
            "use_value_head": bool(training.use_value_head),
            "value_coef": float(training.value_coef),
            "gamma": float(training.gamma),
            "alpha": float(training.alpha),
            "event_reward_decay": float(training.event_reward_decay),
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
            # The fixed racing policy is run provenance, like the two manifests
            # above it, and is absent when no champion bucket is selected.
            "champion_evaluation": (
                champion_evaluation_policy_manifest(
                    context.runner.opponent_pool
                    .selected_champion_bucket_names()
                )
                if context.runner.opponent_pool.selected_champion_bucket_names()
                else None
            ),
            "checkpoint_archive": context.checkpoint_archive.manifest(),
        },
    }


def _optional_float(mapping, key):
    """Return one optional numeric metric without coercing ``None``."""
    if mapping is None or mapping.get(key) is None:
        return None
    return float(mapping[key])


def build_iteration_metrics_row(
    context,
    state,
    *,
    iteration,
    games,
    batch_size,
    normal_batch_size,
    restart_batch_size,
    restart_episode_count,
    restart_elapsed,
    restart_summary,
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
    bucket_results,
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
        "normal_decisions": int(normal_batch_size),
        "restart_decisions": int(restart_batch_size),
        "restart_captured_states": int(restart_episode_count),
        "restart_continuation_episodes": int(restart_episode_count),
        "cumulative_normal_decisions": int(
            state.total_normal_decision_samples
        ),
        "cumulative_restart_decisions": int(
            state.total_restart_decision_samples
        ),
        "cumulative_restart_episodes": int(state.total_restart_episodes),
        "restart_wins": int(restart_summary["wins"]),
        "restart_losses": int(restart_summary["losses"]),
        "normal_event_stats": [
            int(restart_summary["normal_events"][name])
            for name in EVENT_RESULT_COLUMNS
        ],
        "restart_event_stats": [
            int(restart_summary["restart_events"][name])
            for name in EVENT_RESULT_COLUMNS
        ],
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
        "target_decisions_per_minibatch": None if ppo_metrics is None else int(
            ppo_metrics["target_decisions_per_minibatch"]
        ),
        "min_decisions_per_minibatch": None if ppo_metrics is None else int(
            ppo_metrics["min_decisions_per_minibatch"]
        ),
        "decisions_omitted_per_epoch": None if ppo_metrics is None else int(
            ppo_metrics["decisions_omitted_per_epoch"]
        ),
        "minibatch_sizes": None if ppo_metrics is None else ppo_metrics["minibatch_sizes"],
        "epochs_completed": None if ppo_metrics is None else int(ppo_metrics["epochs_completed"]),
        "stopped_by_kl": False if ppo_metrics is None else bool(ppo_metrics["stopped_by_kl"]),
        "optimizer_steps": 0 if gradient_metrics is None else int(
            ppo_metrics["optimizer_steps"] if ppo_metrics is not None else 1
        ),
        "final_approx_kl": _optional_float(ppo_metrics, "final_approx_kl"),
        "max_approx_kl": _optional_float(ppo_metrics, "max_approx_kl"),
        "final_clip_fraction": _optional_float(ppo_metrics, "final_clip_fraction"),
        "final_entropy": _optional_float(ppo_metrics, "final_entropy"),
        "final_policy_loss": _optional_float(ppo_metrics, "final_policy_loss"),
        "gradient_norm_mean": _optional_float(ppo_metrics, "gradient_norm_mean"),
        "gradient_norm_max": _optional_float(ppo_metrics, "gradient_norm_max"),
        "buffer_location": None if ppo_metrics is None else ppo_metrics["buffer_location"],
        "buffer_bytes": 0 if ppo_metrics is None else int(ppo_metrics["buffer_bytes"]),
        "selected_workers": int(context.runner.worker_count),
        "opponent_count": int(context.runner.opponent_pool.size),
        "unique_neural_opponent_count": int(
            context.runner.opponent_pool.unique_neural_opponent_count
        ),
        "bucket_results": [
            [
                int(bucket_results[name][column])
                for column in BUCKET_RESULT_COLUMNS
            ]
            for name in context.training.opponent_buckets
        ],
        "rollout_seconds": float(rollout_elapsed),
        "restart_seconds": float(restart_elapsed),
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
        "ruleset_name": training.ruleset_name,
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
        "opponent_decision_restarts": bool(
            training.opponent_decision_restarts
        ),
        "learning_rate": training.learning_rate,
        "entropy_coef": training.entropy_coef,
        "use_value_head": training.use_value_head,
        "value_coef": training.value_coef if training.use_value_head else None,
        "gamma": training.gamma,
        "alpha": training.alpha,
        "event_reward_decay": training.event_reward_decay,
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
        "total_normal_decision_samples": int(
            state.total_normal_decision_samples
        ),
        "total_restart_decision_samples": int(
            state.total_restart_decision_samples
        ),
        "total_restart_episodes": int(state.total_restart_episodes),
        "total_restart_duration_s": float(state.total_restart_duration_s),
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
        "opponent_pool_final_state": (
            context.runner.opponent_pool.observability(
                games_per_iteration=context.selected_gpi
            )
        ),
        "checkpoint_archive": context.checkpoint_archive.manifest(),
        "runtime_profile_delta": runtime_profile_delta,
        "duration_s": elapsed_time,
    }


def _overlap_text(counts):
    """Render one pairwise overlap mapping for a single console line."""
    if not counts:
        return "n/a"
    return ", ".join(f"{pair} {count}" for pair, count in counts.items())


def _short_opponent_id(opponent_id):
    """Return a compact rotation anchor for one console line."""
    if opponent_id is None:
        return "start"
    _prefix, _separator, suffix = str(opponent_id).partition(":")
    return suffix.lstrip("0") or "0" if suffix else str(opponent_id)


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
        opponent_decision_restarts,
    ):
        """Describe the selected fixed algorithm policy once per invocation."""
        policy = fixed_ppo_policy(ppo_max_epochs)
        fixed = policy["fixed_policy"]
        self.status(f"Fixed GPI: {int(gpi)}.")
        self.status(
            "Opponent buckets: " + ", ".join(opponent_buckets)
            + f" | difficulty weight: {float(difficulty_weight):g}."
        )
        self.status(
            "Opponent-decision restarts: "
            + ("on" if opponent_decision_restarts else "off")
            + "."
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
                f"  max epochs: {ppo_max_epochs} | minibatches: target "
                f"{fixed['target_decisions_per_minibatch']} decisions, minimum "
                f"{fixed['min_decisions_per_minibatch']}, maximum "
                f"{fixed['max_minibatches']} | "
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

    def champion_event(self, summary):
        """Report one completed racing event outside the log-interval gate.

        An event costs 100,000 evaluation games and happens roughly once every
        fifty successful updates, so it is always printed when the reporter is
        not quiet. Hiding it behind ``log_interval`` would make the most
        expensive operation in the run the least visible one.
        """
        if self.quiet:
            return
        funnel = " -> ".join(
            str(count) for count in summary["stage_candidates"]
        )
        print(
            f"  {summary['bucket_name']} event {summary['event_index']} @ "
            f"iteration {summary['iteration']} | candidates "
            f"{summary['candidates']} ({funnel} -> "
            f"{len(summary['survivors'])}) | racing games "
            f"{summary['racing_games']} in "
            f"{format_duration(summary['seconds'])} | survivors "
            f"{len(summary['survivors'])} | admitted "
            f"{len(summary['admitted'])} | evicted "
            f"{len(summary['evicted'])} | bucket size "
            f"{summary['membership_count']}/{summary['capacity']}"
        )
        champions = ", ".join(
            f"{_short_opponent_id(opponent_id)} "
            f"{summary['win_rates'][opponent_id]:.1%}"
            for opponent_id in summary["admitted"]
        )
        # Spelled out in full because the opposite direction is also reported
        # elsewhere: OpponentPerformanceTracker.estimated_win_rate is the
        # *learner's* win rate against the opponent, and confusing the two
        # would invert the meaning of every number on this line.
        target = (
            "heuristic"
            if summary["bucket_name"] == CHAMPION_VS_HEURISTIC_BUCKET
            else "current learner"
        )
        print(
            f"    New champions, candidate win rate vs {target}: {champions}"
        )
        if summary["evicted_difficulties"]:
            evicted = ", ".join(
                f"{_short_opponent_id(opponent_id)} "
                f"{value:.2f}"
                for opponent_id, value in sorted(
                    summary["evicted_difficulties"].items()
                )
            )
            print(f"    Evicted, current difficulty: {evicted}")

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
        opponent_pool_state,
        uniform_rotation_anchors,
        bucket_results,
        gradient_metrics,
        use_value_head,
        value_loss_window,
        ppo_window,
        ppo_max_epochs,
        restart_summary,
    ):
        if self.quiet or iteration % log_interval:
            return
        if reward_summary is None:
            print(f"Iteration {iteration} | {games} games | no real policy decisions")
            return
        bucket_text = ", ".join(
            f"{name} {value['games']} games/"
            + (
                f"{value['wins'] / value['games']:.1%} wins"
                if value["games"]
                else "n/a"
            )
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
        available = opponent_pool_state["available_buckets"]
        # One short identity per bucket is enough to audit the rotation; a
        # per-member column set would add hundreds of fields per iteration.
        rotation_text = ", ".join(
            f"{name} {_short_opponent_id(anchor)}"
            for name, anchor in uniform_rotation_anchors.items()
        ) or "n/a"
        print(
            "  Matchmaking: buckets " + ",".join(opponent_buckets)
            + " | available " + (",".join(available) if available else "none")
            + f" | difficulty weight {difficulty_weight:g} | {bucket_text}"
            + f" | uniform rotation after {rotation_text}"
        )
        if restart_summary["enabled"]:
            print(
                "  Opponent-decision restarts: "
                f"{restart_summary['captured_states']} states/continuations | "
                f"{restart_summary['restart_decisions']} restart decisions + "
                f"{restart_summary['normal_decisions']} normal | "
                f"{restart_summary['seconds']:.2f}s"
            )
        membership_text = ", ".join(
            f"{name} {value['membership_count']}/{value['capacity']}"
            for name, value in opponent_pool_state["buckets"].items()
        )
        forbidden_counts = opponent_pool_state[
            "forbidden_historical_overlap_counts"
        ]
        forbidden_text = _overlap_text(forbidden_counts)
        champion_counts = opponent_pool_state["champion_overlap_counts"]
        print(
            f"  Opponent pool: memberships {membership_text} | "
            f"{opponent_pool_state['unique_neural_opponent_count']} unique "
            f"neural policies | forbidden overlaps {forbidden_text}"
            # Champion overlap is a weighting mechanism, so it is reported
            # separately from the pairs that must stay empty.
            + (
                f" | champion overlaps {_overlap_text(champion_counts)}"
                if champion_counts
                else ""
            )
        )
        historical = opponent_pool_state["buckets"].get("historical_uniform")
        diagnostics = (
            None if historical is None else historical["selection_diagnostics"]
        )
        if diagnostics and diagnostics["ideal_gap_games"] is not None:
            print(
                "  Historical band: "
                f"{diagnostics['selected_count']} of "
                f"{diagnostics['eligible_record_count']} archive candidates | "
                f"ideal gap {diagnostics['ideal_gap_games']:.0f} games | "
                f"actual gap {diagnostics['minimum_gap_games']}-"
                f"{diagnostics['maximum_gap_games']} | target error "
                f"{diagnostics['mean_absolute_target_error_games']:.0f} mean/"
                f"{diagnostics['maximum_absolute_target_error_games']:.0f} max"
                + (
                    " | archive thinned in region"
                    if diagnostics["archive_thinned_in_region"]
                    else ""
                )
            )
        value_predictions = (
            None if gradient_metrics is None else
            gradient_metrics.get("value_predictions_before_update")
        )
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
    if metrics is None:
        return "not updated"
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
    key = f"{phase}_batches"
    summary[key] = summary.get(key, 0) + 1


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
        f"minibatches requested "
        f"{np.mean([row['requested_minibatches'] for row in rows]):.1f} avg, "
        f"effective {np.mean(effective):.1f}/{min(effective)}/{max(effective)} "
        "avg/min/max | omitted decisions/epoch "
        f"{np.mean([row['decisions_omitted_per_epoch'] for row in rows]):.1f} avg"
    )
    restart_episodes = sum(row.get("restart_episodes", 0) for row in rows)
    if restart_episodes:
        print(
            f"  PPO/{count}: opponent-decision restarts {restart_episodes} | "
            f"restart decisions {sum(row.get('restart_decisions', 0) for row in rows)} | "
            f"restart wall time {sum(row.get('restart_seconds', 0.0) for row in rows):.2f}s"
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
    metadata = dict(metadata or {})
    training = metadata.get("training", {})
    bucket_order = list(training.get("opponent_buckets", ()))
    games_per_iteration = int(training.get("games_per_iteration", 0))
    difficulty_weight = float(training.get("difficulty_weight", 0.0))
    uniform_budget, difficulty_budget = matchmaking_component_budgets(
        games_per_iteration,
        difficulty_weight,
    )
    return {
        "format": TRAINING_METRICS_FORMAT,
        "version": TRAINING_METRICS_VERSION,
        "columns": list(TRAINING_METRIC_COLUMNS),
        "bucket_results": {
            "bucket_order": bucket_order,
            "columns": list(BUCKET_RESULT_COLUMNS),
            "nominal_uniform_budget": uniform_budget,
            "nominal_difficulty_budget": difficulty_budget,
        },
        "event_results": {
            "columns": list(EVENT_RESULT_COLUMNS),
            "domains": ["normal", "opponent_decision_restarts"],
        },
        "metadata": metadata,
    }


def _metric_values(row):
    """Project one verbose in-memory row onto the compact stable schema."""
    values = {
        "iteration": row["iteration"],
        "total_iterations": row["total_iterations"],
        "games": row["games"],
        "cumulative_games": row["cumulative_games"],
        "decisions": row["decisions"],
        "normal_decisions": row.get("normal_decisions", row["decisions"]),
        "restart_decisions": row.get("restart_decisions", 0),
        "restart_captured_states": row.get("restart_captured_states", 0),
        "restart_continuation_episodes": row[
            "restart_continuation_episodes"
        ] if "restart_continuation_episodes" in row else 0,
        "cumulative_normal_decisions": row.get(
            "cumulative_normal_decisions", row["decisions"]
        ),
        "cumulative_restart_decisions": row.get(
            "cumulative_restart_decisions", 0
        ),
        "cumulative_restart_episodes": row.get(
            "cumulative_restart_episodes", 0
        ),
        "restart_wins": row.get("restart_wins", 0),
        "restart_losses": row.get("restart_losses", 0),
        "normal_event_stats": row.get("normal_event_stats", [0, 0, 0, 0]),
        "restart_event_stats": row.get("restart_event_stats", [0, 0, 0, 0]),
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
        "target_decisions_per_minibatch": row.get(
            "target_decisions_per_minibatch", 512
        ),
        "min_decisions_per_minibatch": row.get(
            "min_decisions_per_minibatch", 256
        ),
        "decisions_omitted_per_epoch": row.get(
            "decisions_omitted_per_epoch", 0
        ),
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
        "bucket_results": row["bucket_results"],
        "rollout_seconds": row["rollout_seconds"],
        "restart_seconds": row.get("restart_seconds", 0.0),
        "update_seconds": row["update_duration_s"],
        "iteration_seconds": row["iteration_duration_s"],
        "checkpoint_written": row["checkpoint_written"],
        "checkpoint_path": row["checkpoint_path"],
        "elapsed_training_seconds": row["elapsed_training_s"],
    }
    return [values[column] for column in TRAINING_METRIC_COLUMNS]


def _read_training_metrics_header(stream, path):
    """Read and validate one metrics header from an already-open stream."""
    line = stream.readline()
    if not line:
        raise ValueError(f"Training metrics file is empty: {path}.")
    try:
        header = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Training metrics header is invalid: {path}.") from exc
    bucket_schema = header.get("bucket_results", {}) if isinstance(header, dict) else {}
    event_schema = header.get("event_results", {}) if isinstance(header, dict) else {}
    if (
        not isinstance(header, dict)
        or header.get("format") != TRAINING_METRICS_FORMAT
        or header.get("version") != TRAINING_METRICS_VERSION
        or header.get("columns") != list(TRAINING_METRIC_COLUMNS)
        or bucket_schema.get("columns") != list(BUCKET_RESULT_COLUMNS)
        or event_schema.get("columns") != list(EVENT_RESULT_COLUMNS)
        or not isinstance(bucket_schema.get("bucket_order"), list)
        or not bucket_schema["bucket_order"]
        or len(set(bucket_schema["bucket_order"]))
        != len(bucket_schema["bucket_order"])
    ):
        raise ValueError(
            f"Unsupported training metrics format in {path}; "
            f"expected v{TRAINING_METRICS_VERSION}."
        )
    return header


def _iter_training_metric_values(stream, path, bucket_count):
    """Yield validated rows while holding at most one JSON line in memory."""
    for line_number, line in enumerate(stream, start=2):
        if not line.strip():
            continue
        try:
            values = json.loads(line)
        except json.JSONDecodeError as exc:
            if not line.endswith("\n"):
                break
            raise ValueError(
                f"Training metrics row {line_number} is invalid."
            ) from exc
        if not isinstance(values, list) or len(values) != len(
            TRAINING_METRIC_COLUMNS
        ):
            raise ValueError(
                f"Training metrics row {line_number} has the wrong column count."
            )
        bucket_results = values[TRAINING_METRIC_COLUMNS.index("bucket_results")]
        if (
            not isinstance(bucket_results, list)
            or len(bucket_results) != bucket_count
            or any(
                not isinstance(result, list)
                or len(result) != len(BUCKET_RESULT_COLUMNS)
                or any(type(value) is not int or value < 0 for value in result)
                or result[0] != result[1] + result[2]
                for result in bucket_results
            )
            or sum(result[0] for result in bucket_results)
            != values[TRAINING_METRIC_COLUMNS.index("games")]
        ):
            raise ValueError(
                f"Training metrics row {line_number} has invalid bucket results."
            )
        for column in ("normal_event_stats", "restart_event_stats"):
            event_values = values[TRAINING_METRIC_COLUMNS.index(column)]
            if (
                not isinstance(event_values, list)
                or len(event_values) != len(EVENT_RESULT_COLUMNS)
                or any(
                    type(value) is not int or value < 0
                    for value in event_values
                )
            ):
                raise ValueError(
                    f"Training metrics row {line_number} has invalid "
                    f"{column}."
                )
        yield values


def read_training_metrics(path):
    """Decode metrics incrementally, without first copying the whole text."""
    path = Path(path)
    rows = []
    with open(path, encoding="utf-8") as stream:
        header = _read_training_metrics_header(stream, path)
        bucket_count = len(header["bucket_results"]["bucket_order"])
        for values in _iter_training_metric_values(stream, path, bucket_count):
            rows.append(dict(zip(TRAINING_METRIC_COLUMNS, values)))
    return header, rows


def _prepare_metrics_file(path, start_iteration, metadata=None):
    """Create or stream-truncate a trace to the exact resumed iteration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _metrics_header(metadata)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
    )
    source = None
    try:
        with open(temporary, "w", encoding="utf-8") as output:
            if start_iteration and path.is_file():
                source = open(path, encoding="utf-8")
                existing_header = _read_training_metrics_header(source, path)
                existing_hash = existing_header.get("metadata", {}).get(
                    "run_configuration_sha256"
                )
                requested_hash = header.get("metadata", {}).get(
                    "run_configuration_sha256"
                )
                if existing_hash != requested_hash:
                    raise ValueError(
                        "Training metrics configuration hash does not match "
                        "the resumed run."
                    )
                header = existing_header
            output.write(
                json.dumps(header, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
            if source is not None:
                iteration_index = TRAINING_METRIC_COLUMNS.index("iteration")
                bucket_count = len(header["bucket_results"]["bucket_order"])
                for values in _iter_training_metric_values(
                    source,
                    path,
                    bucket_count,
                ):
                    if int(values[iteration_index]) > int(start_iteration):
                        break
                    output.write(
                        json.dumps(
                            values,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if source is not None:
            source.close()
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
