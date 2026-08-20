"""One complete rollout, policy update, checkpoint, and reporting iteration."""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from training.rl.config import (
    RLExecutionOptions,
    RLResourceOptions,
    RLTrainingOptions,
)
from training.rl.ppo import (
    POLICY_GRADIENT_CLIP_NORM,
    ppo_is_enabled,
    update_from_samples,
)
from training.rl.champion_evaluation import (
    evaluate_champion_vs_heuristic,
    evaluate_champion_vs_learner,
)
from training.rl.checkpoint_archive import ARCHIVE_INTERVAL_ITERATIONS
from training.rl.matchmaking import (
    aggregate_match_results,
    build_match_plan,
)
from training.rl.pool import (
    CHAMPION_VS_HEURISTIC_BUCKET,
    CHAMPION_VS_LEARNER_BUCKET,
    select_medium_term_staging_records,
)
from training.rl.reporting import (
    build_iteration_metrics_row,
    _merge_parallel_summary,
    _reward_signal_summary,
    _write_metrics_row,
)
from training.rl.resume import (
    _atomic_network_save,
    _save_numbered_resume_checkpoint,
    _training_state_payload,
)
from training.rl.rollout import REWARD_ZERO_EPSILON
from utils.resource_limits import ensure_ram_available


@dataclass(frozen=True)
class IterationContext:
    """Stable collaborators and configuration shared by all iterations."""

    training: RLTrainingOptions
    resources: RLResourceOptions
    execution: RLExecutionOptions
    network: Any
    runner: Any
    checkpoint_archive: Any
    reporter: Any
    runtime_profile: Any
    parallel_summary: dict
    resume_configuration: Any
    adaptive_tuning: dict
    metrics_stream: Any
    algorithm: str
    schema: dict
    effective_seed: int
    selected_gpi: int
    selected_workers: int
    invocation_target_games: int
    final_iteration: int
    restored_elapsed_rl_seconds: float
    training_perf_started: float


@dataclass
class IterationState:
    """Mutable counters and rolling windows advanced by ``run_iteration``."""

    completed_training_games: int
    win_rate_window: Any
    value_loss_window: Any
    ppo_window: Any
    total_decision_samples: int
    total_normal_decision_samples: int
    total_restart_decision_samples: int
    total_restart_episodes: int
    total_restart_duration_s: float
    policy_updates_completed: int
    clipped_iteration_count: int
    total_rollout_duration_s: float
    total_update_duration_s: float
    last_checkpoint_time: float
    last_saved_iteration: int
    final_weights_path: Path
    completed_this_invocation: int = 0
    completed_iterations_this_invocation: int = 0


def long_horizon_pin_checkpoint_ids(
    opponent_pool,
    checkpoint_archive,
    *,
    completed_iteration,
):
    """Return every archive file the delayed bands need now or imminently.

    Staging milestones are included because nothing else keeps them alive
    between their archive write and their delayed admission. The union is
    published only after both band memberships are final, so a checkpoint
    leaving ``medium_term`` for ``historical_uniform`` is never briefly
    unpinned.
    """
    bucket_names = opponent_pool.archive_backed_bucket_names()
    if not bucket_names:
        return ()
    requested = set()
    for name in bucket_names:
        requested.update(opponent_pool.checkpoint_ids_for_bucket(name))
    retained = checkpoint_archive.retained_records()
    if "medium_term" in bucket_names:
        requested.update(
            record.checkpoint_id
            for record in select_medium_term_staging_records(
                retained,
                completed_iteration=completed_iteration,
                pending_opponent_ids=opponent_pool.bucket_members("recent")
                if "recent" in opponent_pool.buckets_by_name
                else (),
            )
        )
    # Exact resume owns active weights on its own, so a restored member whose
    # archive file is gone must not block the run from pinning the rest.
    return tuple(sorted(
        requested & {record.checkpoint_id for record in retained}
    ))


def refresh_archive_backed_buckets(
    opponent_pool,
    checkpoint_archive,
    *,
    completed_iteration,
):
    """Rebuild both delayed bands and publish the final archive pin union."""
    summary = opponent_pool.reconcile_archive_backed_buckets(
        checkpoint_archive.retained_records(),
        completed_iteration=completed_iteration,
        load_weights=checkpoint_archive.load_weights,
    )
    checkpoint_archive.set_pinned_checkpoint_ids(
        long_horizon_pin_checkpoint_ids(
            opponent_pool,
            checkpoint_archive,
            completed_iteration=completed_iteration,
        )
    )
    return summary


def champion_evaluation_phase(bucket_name):
    """Return the runtime/parallel accounting phase name for one champion target.

    Kept distinct per bucket so an iteration that runs both events reports two
    separate 100,000-game costs rather than one indistinguishable 200,000.
    """
    return f"{bucket_name}_evaluation"


def _publish_champion_target(context, bucket_name):
    """Make the worker-visible current policy the one this event must race.

    ``run_iteration`` publishes the learner once, at the top, before the
    rollouts. By the time a champion event runs, PPO has already replaced the
    parent network, so the bank's current-policy region still holds the policy
    the update *replaced*. Racing against it would rank 50 candidates against a
    learner that no longer exists, and every funnel, seed, seat, and ranking
    check would still pass.

    Published once, immediately before the first stage. Nothing may write to
    that region again until the event finishes: workers view it live, so a
    second write mid-event would change the target between candidates and
    destroy the comparison the common seed panels exist to protect.
    """
    if bucket_name != CHAMPION_VS_LEARNER_BUCKET:
        return
    context.runner.sync_current(context.network)


def _commit_champion_event(context, bucket_name, result):
    """Commit one racing result through the bucket's own retention rule.

    The two buckets share the race and nothing else here. The heuristic bucket
    stores the win rates it just measured, because its target is fixed and the
    numbers stay comparable. The learner bucket discards them and retains on
    current decayed difficulty instead, which the pool cannot read for itself:
    the tracker belongs to matchmaking, and ``pool.py`` importing it would
    invert ownership.
    """
    opponent_pool = context.runner.opponent_pool
    if bucket_name == CHAMPION_VS_HEURISTIC_BUCKET:
        return opponent_pool.apply_champion_vs_heuristic_result(
            result.champion_ids,
            result.final_win_rates,
        )
    incumbent_difficulties = (
        context.runner.performance_tracker.difficulty_snapshot(
            opponent_pool.bucket_members(bucket_name)
        )
    )
    return opponent_pool.apply_champion_vs_learner_result(
        result.champion_ids,
        incumbent_difficulties,
    )


def _run_champion_evaluation(context, iteration, bucket_name):
    """Race one complete candidate batch and commit its five champions.

    Returns ``None`` unless the named bucket's pending list holds a full
    non-overlapping batch. The event is deliberately placed after the archive
    refresh and before performance retention: it changes champion membership,
    and specification 9.8 requires the tracker to be reconciled only once every
    final membership for the iteration is known.

    Nothing here touches GPI counters, ``bucket_results``, difficulty
    evidence, or PPO buffers. The only durable effect is the champion
    membership transaction committed by the pool.
    """
    opponent_pool = context.runner.opponent_pool
    if not opponent_pool.champion_selected(bucket_name):
        return None
    if not opponent_pool.champion_candidate_batch_is_ready(bucket_name):
        return None
    candidate_ids = opponent_pool.champion_pending_candidate_ids(bucket_name)
    bank_slots = {
        opponent_id: opponent_pool.bank_slot(opponent_id)
        for opponent_id in candidate_ids
    }
    # Only reached on an iteration that actually races, so a run that never
    # completes a batch pays nothing for this.
    _publish_champion_target(context, bucket_name)

    def play_stage(specs):
        """Execute one racing stage and account for its worker-pool run."""
        results, run_info = context.runner.evaluate_champion_games(
            specs,
            bucket_name=bucket_name,
        )
        _merge_parallel_summary(
            context.parallel_summary,
            run_info,
            phase=champion_evaluation_phase(bucket_name),
            iteration=iteration,
        )
        if run_info.fallback_count:
            context.reporter.rollout_fallback(iteration, run_info)
        return results

    evaluate = (
        evaluate_champion_vs_heuristic
        if bucket_name == CHAMPION_VS_HEURISTIC_BUCKET
        else evaluate_champion_vs_learner
    )
    result = evaluate(
        candidate_ids=candidate_ids,
        bank_slots=bank_slots,
        play_games=play_stage,
        base_seed=context.effective_seed,
        # The seed panel is keyed by the number of events already committed,
        # so a deterministic replay after a crash reuses the same deals.
        event_index=opponent_pool.champion_completed_event_count(bucket_name),
    )
    commit = _commit_champion_event(context, bucket_name, result)
    return {
        "bucket_name": bucket_name,
        "iteration": int(iteration),
        # ``event_index`` counts committed events, so it is one-based here and
        # one higher than the zero-based index that seeded the panels.
        "event_index": int(commit["event_index"]),
        "racing_event_index": int(result.event_index),
        "candidates": len(candidate_ids),
        "racing_games": int(result.total_games),
        "seconds": float(result.elapsed_seconds),
        "stage_candidates": tuple(
            item.candidates for item in result.stage_summaries
        ),
        "survivors": tuple(result.champion_ids),
        "admitted": tuple(commit["admitted"]),
        "evicted": tuple(commit["evicted"]),
        "membership_count": int(commit["membership_count"]),
        "capacity": int(opponent_pool.buckets_by_name[bucket_name].capacity),
        # The candidate's win rate against whatever this bucket raced. Durable
        # only for the heuristic bucket; reported either way.
        "win_rates": dict(result.final_win_rates),
        "evicted_difficulties": dict(commit.get("evicted_difficulties", {})),
    }


def _run_champion_evaluations(context, iteration):
    """Run every ready champion event in deterministic registry order.

    Both buckets are fed by the same successful updates, so their batches
    normally complete on the same iteration and both events run here, one after
    the other. Registry order fixes which goes first; nothing may derive the
    order from dictionary insertion.
    """
    events = []
    opponent_pool = context.runner.opponent_pool
    for bucket_name in opponent_pool.selected_champion_bucket_names():
        section_started = time.perf_counter()
        event = _run_champion_evaluation(context, iteration, bucket_name)
        # The section exists only on the iterations that actually raced, so a
        # profile that reports it is proof the 100,000 games were timed apart
        # from rollout work rather than folded into it.
        if event is None:
            continue
        context.runtime_profile.add(
            champion_evaluation_phase(bucket_name),
            time.perf_counter() - section_started,
        )
        events.append(event)
    return tuple(events)


def _value_prediction_summary(values):
    """Summarize one batch of pre-update value-head predictions on the host."""
    host_values = values.get() if hasattr(values, "get") else values
    flattened = np.asarray(host_values, dtype=np.float64).reshape(-1)
    return {
        "sample_count": int(flattened.size),
        "mean": float(np.mean(flattened)),
        "std": float(np.std(flattened)),
        "min": float(np.min(flattened)),
        "max": float(np.max(flattened)),
    }


def _event_totals(results):
    """Sum the four rollout event counters without mixing result domains."""
    names = (
        "opponent_draws",
        "opponent_passes",
        "learner_draws",
        "learner_passes",
    )
    return {
        name: sum(int(result["event_stats"][name]) for result in results)
        for name in names
    }


def _reinforce_policy_update(
    network,
    batch,
    *,
    entropy_coef,
    normalize_advantages,
    use_value_head,
    value_coef,
    collect_value_predictions=False,
):
    """Apply the one-full-buffer update selected by ``ppo_max_epochs=1``."""
    xp = network.xp
    x_batch = xp.hstack([xp.asarray(sample.x) for sample in batch])
    actions = [sample.action_index for sample in batch]
    legal_masks = xp.hstack([
        xp.asarray(sample.legal_mask, dtype=xp.bool_)
        for sample in batch
    ])
    rewards = xp.asarray(
        [sample.policy_reward for sample in batch],
        dtype=xp.float32,
    ).reshape(1, -1)
    value_returns = None
    value_predictions = None
    policy_signal = rewards
    # This forward pass owns the cache that ``backward_policy_gradient``
    # differentiates, so it is the update pass and must apply dropout.
    if use_value_head:
        values = network.predict_values(x_batch, training=True)
        if collect_value_predictions:
            value_predictions = _value_prediction_summary(values)
        policy_signal = rewards - values
        value_returns = rewards
    else:
        network.forward(x_batch, training=True)
    if normalize_advantages:
        mean = xp.mean(policy_signal)
        std = float(xp.std(policy_signal))
        if std > REWARD_ZERO_EPSILON:
            policy_signal = (policy_signal - mean) / (std + REWARD_ZERO_EPSILON)
        else:
            policy_signal = policy_signal - mean
    metrics = network.backward_policy_gradient(
        actions,
        policy_signal,
        legal_masks=legal_masks,
        entropy_coef=entropy_coef,
        value_returns=value_returns,
        value_coef=value_coef,
        clip_grad_norm=POLICY_GRADIENT_CLIP_NORM,
    )
    if value_predictions is not None:
        metrics["value_predictions_before_update"] = value_predictions
    return metrics


def _update_policy(context, state, batch, iteration):
    """Update the policy once and return gradient/PPO metrics and duration."""
    if not batch:
        return None, None, 0.0
    training = context.training
    profile = context.runtime_profile
    section_started = time.perf_counter()
    sample_bytes = sum(
        int(getattr(sample.x, "nbytes", 0))
        + int(getattr(sample.legal_mask, "nbytes", 0))
        for sample in batch
    )
    ensure_ram_available(
        max(1, sample_bytes * 4),
        context.resources.safety_config.memory_reserve_mb,
        f"RL iteration {iteration} decision-buffer assembly",
    )
    profile.add(
        "decision_buffer_memory_preflight",
        time.perf_counter() - section_started,
    )
    collect_values = (
        training.use_value_head
        and not context.execution.quiet
        and iteration % context.execution.log_interval == 0
    )
    update_started = time.perf_counter()
    ppo_metrics = None
    if ppo_is_enabled(training.ppo_max_epochs):
        ppo_result = update_from_samples(
            context.network,
            batch,
            base_seed=context.effective_seed,
            iteration=iteration,
            entropy_coef=training.entropy_coef,
            value_coef=training.value_coef,
            normalize_advantages=training.normalize_advantages,
            max_epochs=training.ppo_max_epochs,
            collect_value_predictions=collect_values,
        )
        profile.add(
            "ppo_buffer_assembly_and_advantage_normalization",
            ppo_result.buffer_seconds,
        )
        profile.add("ppo_update", ppo_result.update_seconds)
        ppo_metrics = ppo_result.metrics
        profile.merge_ppo_metrics(ppo_metrics)
        gradient_metrics = (
            None if ppo_metrics.get("insufficient_decisions") else ppo_metrics
        )
    else:
        reinforce_started = time.perf_counter()
        gradient_metrics = _reinforce_policy_update(
            context.network,
            batch,
            entropy_coef=training.entropy_coef,
            normalize_advantages=training.normalize_advantages,
            use_value_head=training.use_value_head,
            value_coef=training.value_coef,
            collect_value_predictions=collect_values,
        )
        context.network.synchronize()
        profile.add(
            "reinforce_policy_update",
            time.perf_counter() - reinforce_started,
        )
    update_elapsed = time.perf_counter() - update_started
    state.total_update_duration_s += update_elapsed
    state.total_decision_samples += len(batch)
    if gradient_metrics is not None and gradient_metrics["grad_clipped"]:
        state.clipped_iteration_count += 1
    if training.use_value_head and gradient_metrics is not None:
        state.value_loss_window.append(gradient_metrics["value_loss"])
    if gradient_metrics is not None:
        state.policy_updates_completed += 1
    return gradient_metrics, ppo_metrics, update_elapsed


def _checkpoint(context, state, iteration):
    """Write a scheduled checkpoint and return paths exposed to callbacks."""
    execution = context.execution
    if iteration % execution.checkpoint_interval != 0:
        return False, None, None
    training_state = _training_state_payload(
        win_rate_window=state.win_rate_window,
        value_loss_window=state.value_loss_window,
        ppo_window=state.ppo_window,
        total_decision_samples=state.total_decision_samples,
        total_normal_decision_samples=state.total_normal_decision_samples,
        total_restart_decision_samples=state.total_restart_decision_samples,
        total_restart_episodes=state.total_restart_episodes,
        total_restart_duration_s=state.total_restart_duration_s,
        policy_updates_completed=state.policy_updates_completed,
        clipped_iteration_count=state.clipped_iteration_count,
        total_rollout_duration_s=state.total_rollout_duration_s,
        total_update_duration_s=state.total_update_duration_s,
        elapsed_rl_seconds=(
            context.restored_elapsed_rl_seconds
            + time.perf_counter() - context.training_perf_started
        ),
    )
    if execution.numbered_checkpoints:
        checkpoint_path, checkpoint_state_path = _save_numbered_resume_checkpoint(
            context.network,
            context.runner,
            context.resources.rl_weights_path,
            iteration,
            context.resume_configuration,
            context.runner.worker_count,
            state.completed_training_games,
            context.adaptive_tuning,
            training_state,
        )
        state.final_weights_path = checkpoint_path
        state.last_saved_iteration = iteration
        checkpoint_resume_path = str(checkpoint_state_path)
    else:
        _atomic_network_save(
            context.network,
            context.resources.rl_weights_path,
        )
        checkpoint_path = Path(context.resources.rl_weights_path)
        checkpoint_resume_path = None
    now = time.time()
    checkpoint_elapsed = now - state.last_checkpoint_time
    state.last_checkpoint_time = now
    context.reporter.checkpoint(
        checkpoint_path,
        state.completed_training_games,
        context.training.total_training_games,
        checkpoint_elapsed,
    )
    return True, str(checkpoint_path), checkpoint_resume_path


def _emit_iteration_outputs(
    context,
    state,
    *,
    iteration,
    row,
    checkpoint_written,
    checkpoint_path,
    checkpoint_resume_path,
):
    """Persist metrics and notify all optional pipeline callbacks."""
    section_started = time.perf_counter()
    serialized_row = {
        key: round(float(value), 5)
        if isinstance(value, (float, np.floating))
        else value
        for key, value in row.items()
    }
    _write_metrics_row(context.metrics_stream, serialized_row)
    context.runtime_profile.add(
        "metrics_jsonl_write_and_fsync",
        time.perf_counter() - section_started,
    )
    section_started = time.perf_counter()
    execution = context.execution
    if (
        execution.checkpoint_callback is not None
        and checkpoint_written
        and execution.numbered_checkpoints
    ):
        execution.checkpoint_callback({
            "rl_weights_path": checkpoint_path,
            "resume_state_path": checkpoint_resume_path,
            "completed_training_games": int(state.completed_training_games),
            "rl_iterations_completed": int(iteration),
            "games_per_iteration": int(context.selected_gpi),
            "final_workers": int(context.runner.worker_count),
        })
    if execution.metrics_callback is not None:
        execution.metrics_callback(dict(row))
    if execution.progress_callback is not None:
        execution.progress_callback(
            state.completed_training_games,
            context.training.total_training_games,
        )
    context.runtime_profile.add(
        "pipeline_and_progress_callbacks",
        time.perf_counter() - section_started,
    )


def run_iteration(context, state, iteration):
    """Run one complete RL iteration and return whether shutdown was requested."""
    execution = context.execution
    if execution.shutdown_requested is not None and execution.shutdown_requested():
        return True
    iteration_started = time.perf_counter()
    games = min(
        context.selected_gpi,
        context.invocation_target_games - state.completed_training_games,
    )
    previous_games = state.completed_training_games
    accounted_before = context.runtime_profile.accounted_seconds()
    section_started = time.perf_counter()
    context.runner.sync_current(context.network)
    context.runtime_profile.add(
        "policy_snapshot_synchronization",
        time.perf_counter() - section_started,
    )
    section_started = time.perf_counter()
    match_plan = build_match_plan(
        opponent_pool=context.runner.opponent_pool,
        performance_tracker=context.runner.performance_tracker,
        uniform_rotation=context.runner.uniform_rotation,
        selected_buckets=context.training.opponent_buckets,
        difficulty_weight=context.training.difficulty_weight,
        iteration=iteration,
        first_absolute_game=previous_games,
        game_count=games,
        base_seed=context.effective_seed,
    )
    context.runtime_profile.add(
        "matchmaking_plan_construction",
        time.perf_counter() - section_started,
    )
    rollout_started = time.perf_counter()
    rollout_results, rollout_info = context.runner.collect_games(
        match_plan,
        context.effective_seed,
        capture_opponent_decision_restarts=(
            context.training.opponent_decision_restarts
        ),
    )
    rollout_elapsed = time.perf_counter() - rollout_started
    context.runtime_profile.add("rollout_game_execution", rollout_elapsed)
    context.runtime_profile.merge_rollout_worker(
        context.runner.last_runtime_profile
    )
    normal_rollout_elapsed = rollout_elapsed

    section_started = time.perf_counter()
    _merge_parallel_summary(
        context.parallel_summary,
        rollout_info,
        phase="rollout",
        iteration=iteration,
    )
    if rollout_info.fallback_count:
        context.reporter.rollout_fallback(iteration, rollout_info)
    normal_batch = []
    for result in rollout_results:
        normal_batch.extend(result["samples"])
    opponent_results, bucket_results = aggregate_match_results(
        match_plan,
        rollout_results,
    )
    wins = sum(value["wins"] for value in bucket_results.values())
    state.win_rate_window.append(wins / games)
    moving_win_rate = sum(state.win_rate_window) / len(state.win_rate_window)
    context.runtime_profile.add(
        "matchmaking_result_aggregation",
        time.perf_counter() - section_started,
    )
    section_started = time.perf_counter()
    context.runner.performance_tracker.update(
        (
            record.opponent_id
            for record in context.runner.opponent_pool.active_opponents()
        ),
        opponent_results,
    )
    context.runtime_profile.add(
        "opponent_performance_update",
        time.perf_counter() - section_started,
    )

    restart_results = []
    restart_elapsed = 0.0
    restart_info = None
    if context.training.opponent_decision_restarts:
        restart_started = time.perf_counter()
        restart_results, restart_info = context.runner.collect_restarts(
            rollout_results,
            context.effective_seed,
            iteration,
        )
        restart_elapsed = time.perf_counter() - restart_started
        context.runtime_profile.add(
            "opponent_decision_restart_execution",
            restart_elapsed,
        )
        context.runtime_profile.merge_rollout_worker(
            context.runner.last_runtime_profile
        )
        _merge_parallel_summary(
            context.parallel_summary,
            restart_info,
            phase="opponent_decision_restarts",
            iteration=iteration,
        )
        if restart_info.fallback_count:
            context.reporter.rollout_fallback(iteration, restart_info)

    restart_batch = []
    for result in restart_results:
        restart_batch.extend(result["samples"])
    batch = [*normal_batch, *restart_batch]
    rollout_elapsed = normal_rollout_elapsed + restart_elapsed
    state.total_rollout_duration_s += rollout_elapsed
    state.total_normal_decision_samples += len(normal_batch)
    state.total_restart_decision_samples += len(restart_batch)
    state.total_restart_episodes += len(restart_results)
    state.total_restart_duration_s += restart_elapsed
    restart_wins = sum(
        result["winner"] == result["learner_position"]
        for result in restart_results
    )
    restart_summary = {
        "enabled": bool(context.training.opponent_decision_restarts),
        "captured_states": int(len(restart_results)),
        "continuation_episodes": int(len(restart_results)),
        "wins": int(restart_wins),
        "losses": int(len(restart_results) - restart_wins),
        "normal_decisions": int(len(normal_batch)),
        "restart_decisions": int(len(restart_batch)),
        "normal_events": _event_totals(rollout_results),
        "restart_events": _event_totals(restart_results),
        "seconds": float(restart_elapsed),
    }
    section_started = time.perf_counter()
    reward_summary = (
        _reward_signal_summary(batch, context.network.xp) if batch else None
    )
    context.runtime_profile.add(
        "reward_statistics",
        time.perf_counter() - section_started,
    )
    gradient_metrics, ppo_metrics, update_elapsed = _update_policy(
        context,
        state,
        batch,
        iteration,
    )

    state.completed_training_games += games
    state.completed_this_invocation += games
    state.completed_iterations_this_invocation += 1
    section_started = time.perf_counter()
    opponent_pool = context.runner.opponent_pool
    admitted_record = opponent_pool.consider_updated_policy(
        context.network,
        iteration=iteration,
        completed_games=state.completed_training_games,
        has_samples=gradient_metrics is not None,
    )
    context.runtime_profile.add(
        "opponent_pool_admission_and_retention",
        time.perf_counter() - section_started,
    )
    section_started = time.perf_counter()
    if admitted_record is not None:
        # A fresh milestone is always inside the staging window, so it is
        # pinned on arrival and the refresh below only widens the union.
        context.checkpoint_archive.consider_snapshot(
            context.network,
            admitted_record,
            iteration=iteration,
            completed_games=state.completed_training_games,
            pinned=bool(opponent_pool.archive_backed_bucket_names()),
        )
    # The age boundary advances even when an iteration produced no trainable
    # decisions, so the bands refresh on cadence rather than on admission.
    if iteration % ARCHIVE_INTERVAL_ITERATIONS == 0:
        refresh_archive_backed_buckets(
            opponent_pool,
            context.checkpoint_archive,
            completed_iteration=iteration,
        )
    context.runtime_profile.add(
        "checkpoint_archive_update",
        time.perf_counter() - section_started,
    )
    # Racing events run here, between the archive refresh and performance
    # retention, so their evaluation games stay out of rollout timing and their
    # admissions are visible to the tracker reconciliation below. Each event
    # times itself, so a double-event iteration reports two costs.
    champion_events = _run_champion_evaluations(context, iteration)
    section_started = time.perf_counter()
    # Trackers are reconciled only after the membership transaction, so an
    # identity moving between buckets keeps its accumulated evidence.
    active_opponent_ids = tuple(
        record.opponent_id for record in opponent_pool.active_opponents()
    )
    context.runner.performance_tracker.ensure(active_opponent_ids)
    context.runner.performance_tracker.retain_only(active_opponent_ids)
    context.runtime_profile.add(
        "opponent_performance_retention",
        time.perf_counter() - section_started,
    )
    if ppo_metrics is not None and not ppo_metrics.get("insufficient_decisions"):
        state.ppo_window.append({
            **{
                name: value
                for name, value in ppo_metrics.items()
                if name not in {
                    "runtime_timing_seconds",
                    "runtime_profile_detail",
                }
            },
            "games": int(games),
            "decisions": int(len(batch)),
            "normal_decisions": int(len(normal_batch)),
            "restart_decisions": int(len(restart_batch)),
            "restart_episodes": int(len(restart_results)),
            "ppo_seconds": float(update_elapsed),
            "rollout_seconds": float(normal_rollout_elapsed),
            "restart_seconds": float(restart_elapsed),
        })

    section_started = time.perf_counter()
    checkpoint_written, checkpoint_path, checkpoint_resume_path = _checkpoint(
        context,
        state,
        iteration,
    )
    context.runtime_profile.add(
        "checkpoint_serialization",
        time.perf_counter() - section_started,
    )
    section_started = time.perf_counter()
    context.reporter.iteration(
        iteration=iteration,
        log_interval=execution.log_interval,
        games=games,
        completed_games=state.completed_training_games,
        total_games=context.training.total_training_games,
        reward_summary=reward_summary,
        wins=wins,
        moving_win_rate=moving_win_rate,
        win_window_size=len(state.win_rate_window),
        opponent_buckets=context.training.opponent_buckets,
        difficulty_weight=context.training.difficulty_weight,
        opponent_count=context.runner.opponent_pool.size,
        unique_neural_opponent_count=(
            context.runner.opponent_pool.unique_neural_opponent_count
        ),
        opponent_pool_state=context.runner.opponent_pool.observability(
            games_per_iteration=context.selected_gpi
        ),
        uniform_rotation_anchors=context.runner.uniform_rotation.anchors(),
        bucket_results=bucket_results,
        gradient_metrics=gradient_metrics,
        use_value_head=context.training.use_value_head,
        value_loss_window=state.value_loss_window,
        ppo_window=state.ppo_window,
        ppo_max_epochs=context.training.ppo_max_epochs,
        restart_summary=restart_summary,
    )
    # Reported after the iteration summary so each event reads as a consequence
    # of the update that completed its candidate batch.
    for champion_event in champion_events:
        context.reporter.champion_event(champion_event)
    context.runtime_profile.add(
        "console_logging",
        time.perf_counter() - section_started,
    )
    section_started = time.perf_counter()
    row = build_iteration_metrics_row(
        context,
        state,
        iteration=iteration,
        games=games,
        batch_size=len(batch),
        normal_batch_size=len(normal_batch),
        restart_batch_size=len(restart_batch),
        restart_episode_count=len(restart_results),
        restart_elapsed=restart_elapsed,
        restart_summary=restart_summary,
        wins=wins,
        moving_win_rate=moving_win_rate,
        reward_summary=reward_summary,
        gradient_metrics=gradient_metrics,
        ppo_metrics=ppo_metrics,
        rollout_elapsed=rollout_elapsed,
        update_elapsed=update_elapsed,
        checkpoint_written=checkpoint_written,
        checkpoint_path=checkpoint_path,
        iteration_started=iteration_started,
        bucket_results=bucket_results,
    )
    context.runtime_profile.add(
        "metrics_payload_construction",
        time.perf_counter() - section_started,
    )
    _emit_iteration_outputs(
        context,
        state,
        iteration=iteration,
        row=row,
        checkpoint_written=checkpoint_written,
        checkpoint_path=checkpoint_path,
        checkpoint_resume_path=checkpoint_resume_path,
    )
    accounted = context.runtime_profile.accounted_seconds() - accounted_before
    context.runtime_profile.add(
        "iteration_control_overhead",
        max(0.0, time.perf_counter() - iteration_started - accounted),
    )
    return bool(
        execution.shutdown_requested is not None
        and execution.shutdown_requested()
    )
