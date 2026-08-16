"""Prepare a complete RL invocation, including exact resume restoration."""

from dataclasses import dataclass, replace
import math
from pathlib import Path
import random
import secrets
import time

import numpy as np

from diagnostics.parallel_runner import ParallelSafetyConfig
from training.rl.adaptive_tuning import (
    atomic_write_json as atomic_write_tuning_json,
    hardware_metadata,
    hardware_warning,
    run_worker_tuning,
)
from training.rl.checkpoint_archive import CheckpointArchive, archive_policy_manifest
from training.rl.config import (
    RLExecutionOptions,
    RLResourceOptions,
    RLTrainingOptions,
    resolve_training_options,
)
from training.rl.iteration import IterationContext, IterationState
from training.rl.parallel import RLRolloutRunner
from training.rl.matchmaking import matchmaking_policy_manifest
from training.rl.pool import pool_policy_manifest, unique_neural_capacity
from training.rl.reporting import (
    RLRuntimeProfile,
    RLTrainingReporter,
    _new_parallel_summary,
    _prepare_metrics_file,
    build_training_metrics_header,
)
from training.rl.resume import (
    RLTrainingConfiguration,
    _load_initial_network,
    _restore_rng_state,
    _restore_training_windows,
    _sl_checkpoint_sha256,
    _validate_resume_configuration,
    load_resume_state,
)
from utils.repository import current_git_commit
from utils.resource_limits import choose_safe_rl_device, ensure_ram_available


@dataclass(frozen=True)
class RLTrainingSession:
    """Everything needed by the small top-level RL orchestration loop."""

    context: IterationContext
    state: IterationState
    reporter: RLTrainingReporter
    runtime_profile: RLRuntimeProfile
    metrics_path: Path
    start_time: float
    start_iteration: int
    iterations_to_run: int
    requested_device: str
    device_fallback_reason: str | None
    initialization_source: str
    decision_samples_at_invocation_start: int
    optimizer_steps_at_invocation_start: int


@dataclass(frozen=True)
class _ResumeInputs:
    """Options and durable state after optional checkpoint hydration."""

    training: RLTrainingOptions
    resources: RLResourceOptions
    execution: RLExecutionOptions
    metadata: dict | None
    pool_state: dict | None
    pool_weights: dict
    performance_state: dict | None
    saved_configuration: RLTrainingConfiguration | None
    completed_training_games: int
    start_iteration: int


def _resume_inputs(training, resources, execution, reporter):
    """Validate a resume pair and make its saved options authoritative."""
    has_weights = execution.resume_weights_path is not None
    has_state = execution.resume_state_file is not None
    if has_weights != has_state:
        raise ValueError(
            "resume_weights_path and resume_state_file must be provided together"
        )
    if execution.fresh_from_sl and has_weights:
        raise ValueError("fresh_from_sl cannot be combined with a resume pair")
    if not has_weights:
        return _ResumeInputs(
            training=training,
            resources=resources,
            execution=execution,
            metadata=None,
            pool_state=None,
            pool_weights={},
            performance_state=None,
            saved_configuration=None,
            completed_training_games=0,
            start_iteration=0,
        )

    metadata, pool_weights = load_resume_state(
        execution.resume_weights_path,
        execution.resume_state_file,
    )
    saved = RLTrainingConfiguration.from_mapping(metadata["configuration"])
    reporter.status(
        "Resume configuration loaded from disk; command-line and caller "
        "training options are ignored."
    )
    return _ResumeInputs(
        training=replace(
            training,
            iterations=None,
            ruleset_name=saved.ruleset_name,
            total_training_games=saved.total_training_games,
            gpi=saved.selected_gpi,
            opponent_buckets=saved.opponent_buckets,
            difficulty_weight=saved.difficulty_weight,
            opponent_decision_restarts=saved.opponent_decision_restarts,
            learning_rate=saved.learning_rate,
            entropy_coef=saved.entropy_coef,
            weight_decay=saved.weight_decay,
            dropout_rate=saved.dropout_rate,
            use_value_head=saved.use_value_head,
            value_coef=saved.value_coef,
            gamma=saved.gamma,
            alpha=saved.alpha,
            event_reward_decay=saved.event_reward_decay,
            normalize_advantages=saved.normalize_advantages,
            seed=saved.effective_seed,
            ppo_max_epochs=saved.ppo_max_epochs,
        ),
        resources=replace(
            resources,
            device=saved.device,
            workers=saved.selected_workers,
            safety_config=ParallelSafetyConfig(
                memory_reserve_mb=saved.worker_memory_reserve_mb,
                estimated_worker_mb=saved.worker_estimated_mb,
                max_worker_rss_mb=saved.worker_max_rss_mb,
            ),
            retune_workers=False,
            adaptive_tuning_training_games=saved.total_training_games,
        ),
        execution=replace(
            execution,
            log_interval=saved.log_interval,
            checkpoint_interval=saved.checkpoint_interval,
            moving_average_window=saved.moving_average_window,
            fresh_from_sl=False,
            numbered_checkpoints=True,
        ),
        metadata=metadata,
        pool_state=metadata["opponent_pool_state"],
        pool_weights=pool_weights,
        performance_state=metadata["opponent_performance_state"],
        saved_configuration=saved,
        completed_training_games=int(metadata["completed_training_games"]),
        start_iteration=int(metadata["completed_iteration"]),
    )


def _initialization_source(metadata, execution, initial_rl_weights_path):
    if metadata is not None:
        return "numbered_rl_resume"
    if execution.fresh_from_sl or initial_rl_weights_path is None:
        return "supervised"
    if Path(initial_rl_weights_path).exists():
        return "existing_rl"
    return "supervised"


def _resume_configuration(
    inputs,
    *,
    network,
    selected_workers,
    effective_seed,
    supervised_weights_sha256,
    algorithm,
    reporter,
):
    """Build or validate the immutable configuration stored in checkpoints."""
    training = inputs.training
    resources = inputs.resources
    execution = inputs.execution
    saved = inputs.saved_configuration
    if saved is not None:
        if execution.run_configuration is not None:
            expected = RLTrainingConfiguration.from_run_config(
                execution.run_configuration,
                total_training_games=saved.total_training_games,
                selected_workers=saved.selected_workers,
                device=saved.device,
            )
            _validate_resume_configuration(
                inputs.metadata,
                expected,
                emit_status=reporter.status,
            )
        saved.warn_if_commit_changed(reporter.status)
        return saved
    if execution.run_configuration is not None:
        return RLTrainingConfiguration.from_run_config(
            execution.run_configuration,
            total_training_games=training.total_training_games,
            selected_workers=selected_workers,
            device=network.device,
        )
    return RLTrainingConfiguration.from_mapping({
        "total_training_games": int(training.total_training_games),
        "ruleset_name": training.ruleset_name,
        "selected_gpi": int(training.gpi),
        "selected_workers": int(selected_workers),
        "rl_training_algorithm": algorithm,
        "log_interval": int(execution.log_interval),
        "checkpoint_interval": int(execution.checkpoint_interval),
        "moving_average_window": int(execution.moving_average_window),
        "opponent_buckets": tuple(training.opponent_buckets),
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
        "normalize_advantages": bool(training.normalize_advantages),
        "weight_decay": float(training.weight_decay),
        "dropout_rate": float(training.dropout_rate),
        "effective_seed": int(effective_seed),
        "device": network.device,
        "sl_weights_sha256": supervised_weights_sha256,
        "ppo_max_epochs": int(training.ppo_max_epochs),
        "worker_memory_reserve_mb": int(
            resources.safety_config.memory_reserve_mb
        ),
        "worker_estimated_mb": int(
            resources.safety_config.estimated_worker_mb
        ),
        "worker_max_rss_mb": int(
            resources.safety_config.max_worker_rss_mb
        ),
        "opponent_system_policy": {
            "pool": pool_policy_manifest(training.opponent_buckets),
            "matchmaking": matchmaking_policy_manifest(),
            "checkpoint_archive": archive_policy_manifest(),
        },
        "run_configuration_sha256": None,
        "git_commit": current_git_commit(),
    })


def _invocation_target(training, execution, completed_games):
    """Validate and return this invocation's exact stopping point."""
    if completed_games >= training.total_training_games:
        raise ValueError(
            "The resume checkpoint has already completed the requested game budget."
        )
    target = (
        int(training.total_training_games)
        if execution.stop_after_training_games is None
        else int(execution.stop_after_training_games)
    )
    if not completed_games < target <= training.total_training_games:
        raise ValueError(
            "stop_after_training_games must be above completed games and no "
            "greater than total_training_games"
        )
    return target


def prepare_training_session(training=None, resources=None, execution=None):
    """Resolve options, restore state, tune workers, and open an RL session."""
    training = training or RLTrainingOptions()
    resources = resources or RLResourceOptions()
    execution = execution or RLExecutionOptions()
    runtime_profile = RLRuntimeProfile()
    reporter = RLTrainingReporter(
        quiet=execution.quiet,
        status_callback=execution.status_callback,
    )
    inputs = _resume_inputs(training, resources, execution, reporter)
    resolved = resolve_training_options(
        inputs.training,
        inputs.resources,
        inputs.execution,
    )
    inputs = replace(
        inputs,
        training=resolved.training,
        resources=resolved.resources,
        execution=resolved.execution,
    )
    training = inputs.training
    resources = inputs.resources
    execution = inputs.execution
    effective_seed = (
        int(training.seed) if training.seed is not None else secrets.randbits(63)
    )
    invocation_target = _invocation_target(
        training,
        execution,
        inputs.completed_training_games,
    )
    random.seed(effective_seed)
    np.random.seed(effective_seed & 0xFFFFFFFF)
    requested_device = resources.device
    device, device_fallback_reason = choose_safe_rl_device(resources.device)
    initial_weights = (
        execution.resume_weights_path
        if inputs.metadata is not None
        else (
            None if execution.numbered_checkpoints else resources.rl_weights_path
        )
    )
    initialization_source = _initialization_source(
        inputs.metadata,
        execution,
        initial_weights,
    )
    network = _load_initial_network(
        training.learning_rate,
        resources.sl_weights_path,
        initial_weights,
        quiet=execution.quiet,
        use_value_head=training.use_value_head,
        device=device,
        fresh_from_sl=execution.fresh_from_sl,
        expected_training_algorithm=resolved.algorithm,
        weight_decay=training.weight_decay,
        dropout_rate=training.dropout_rate,
        ruleset=training.ruleset_name,
    )
    if inputs.metadata is not None:
        network.load_optimizer_state_dict(inputs.metadata["optimizer_state"])
    supervised_hash = (
        inputs.saved_configuration.sl_weights_sha256
        if inputs.saved_configuration is not None
        else _sl_checkpoint_sha256(resources.sl_weights_path)
    )
    reporter.device_fallback(device_fallback_reason)

    saved_tuning = (
        None
        if inputs.metadata is None
        else inputs.metadata.get("adaptive_tuning")
    )
    if saved_tuning and not resources.retune_workers:
        warning = hardware_warning(saved_tuning, hardware_metadata(network.device))
        if warning:
            reporter.status(f"Warning: {warning}")
    reporter.tuning_header()
    runtime_profile.add(
        "validation_model_load_and_resume",
        time.perf_counter() - runtime_profile.started,
    )
    tuning_started = time.perf_counter()
    adaptive_tuning = run_worker_tuning(
        network,
        gpi=training.gpi,
        total_training_games=resolved.tuning_training_games,
        workers=resources.workers,
        retune_workers=resources.retune_workers,
        saved_tuning=saved_tuning,
        base_seed=effective_seed,
        opponent_buckets=training.opponent_buckets,
        difficulty_weight=training.difficulty_weight,
        schema=resolved.schema,
        gamma=training.gamma,
        ruleset_name=training.ruleset_name,
        safety=resources.safety_config,
        pool_state=inputs.pool_state,
        pool_weights=inputs.pool_weights,
        performance_state=inputs.performance_state,
        output_path=resources.adaptive_tuning_path,
        status_callback=reporter.status,
    )
    runtime_profile.add("adaptive_tuning", time.perf_counter() - tuning_started)
    runner_setup_started = time.perf_counter()
    selected_workers = int(adaptive_tuning["selected_workers"])
    reporter.update_configuration(
        algorithm=resolved.algorithm,
        use_value_head=training.use_value_head,
        value_coef=training.value_coef,
        dropout_rate=training.dropout_rate,
        weight_decay=training.weight_decay,
        ppo_max_epochs=training.ppo_max_epochs,
        gpi=training.gpi,
        opponent_buckets=training.opponent_buckets,
        difficulty_weight=training.difficulty_weight,
        opponent_decision_restarts=training.opponent_decision_restarts,
    )
    resume_configuration = _resume_configuration(
        inputs,
        network=network,
        selected_workers=selected_workers,
        effective_seed=effective_seed,
        supervised_weights_sha256=supervised_hash,
        algorithm=resolved.algorithm,
        reporter=reporter,
    )

    remaining_games = invocation_target - inputs.completed_training_games
    iterations_to_run = math.ceil(remaining_games / training.gpi)
    final_iteration = inputs.start_iteration + iterations_to_run
    estimated_batch_bytes = min(training.gpi, remaining_games) * 52 * 4096
    policy_bytes = sum(
        int(getattr(network, name).nbytes) for name in network.weight_names
    )
    shared_pool_size = unique_neural_capacity(training.opponent_buckets)
    estimated_shared_bytes = (1 + shared_pool_size) * policy_bytes
    ensure_ram_available(
        estimated_shared_bytes + estimated_batch_bytes,
        resources.safety_config.memory_reserve_mb,
        "RL decision buffer and shared-policy preflight",
    )
    reporter.resource_preflight(
        requested_device=requested_device,
        selected_device=network.device,
        estimated_host_bytes=estimated_shared_bytes + estimated_batch_bytes,
    )
    runner = RLRolloutRunner(
        network,
        opponent_buckets=training.opponent_buckets,
        schema=resolved.schema,
        gamma=training.gamma,
        ruleset_name=training.ruleset_name,
        safety=resources.safety_config,
    )
    if inputs.pool_state is not None:
        runner.restore_opponent_pool(
            inputs.pool_state,
            inputs.pool_weights,
            inputs.performance_state,
        )
        if (
            runner.opponent_pool.last_completed_rl_iteration
            != inputs.start_iteration
        ):
            raise ValueError(
                "Resume opponent-pool iteration counter does not match the "
                "selected exact checkpoint"
            )
    checkpoint_archive = CheckpointArchive(resources.artifact_directory)
    checkpoint_archive.reconcile(inputs.start_iteration)
    medium_term_checkpoint_ids = (
        runner.opponent_pool.checkpoint_ids_for_bucket("medium_term")
    )
    if inputs.pool_state is None:
        checkpoint_archive.consider_snapshot(
            network,
            runner.opponent_pool.initial_snapshot_record,
            iteration=0,
            completed_games=0,
            pinned=bool(medium_term_checkpoint_ids),
        )
    checkpoint_archive.set_pinned_checkpoint_ids(medium_term_checkpoint_ids)
    actual_workers, was_capped, cap_reason = runner.set_workers(selected_workers)
    if was_capped:
        reporter.worker_cap(selected_workers, actual_workers, cap_reason)
        selected_workers = actual_workers
        adaptive_tuning["selected_workers"] = int(actual_workers)
        adaptive_tuning["worker_source"] += "_safety_capped"
        atomic_write_tuning_json(resources.adaptive_tuning_path, adaptive_tuning)
        if inputs.saved_configuration is None:
            resume_configuration = replace(
                resume_configuration,
                selected_workers=int(actual_workers),
            )
    if inputs.metadata is not None:
        _restore_rng_state(inputs.metadata["rng_state"])
        reporter.resumed(
            inputs.start_iteration,
            inputs.completed_training_games,
            runner.opponent_pool.size,
        )

    windows = _restore_training_windows(
        inputs.metadata,
        execution.moving_average_window,
    )
    win_rate_window, value_loss_window, ppo_window, restored_state = windows
    parallel_summary = _new_parallel_summary(resources.workers)
    parallel_summary["initial_workers"] = int(actual_workers)
    preliminary_context = IterationContext(
        training=training,
        resources=resources,
        execution=execution,
        network=network,
        runner=runner,
        checkpoint_archive=checkpoint_archive,
        reporter=reporter,
        runtime_profile=runtime_profile,
        parallel_summary=parallel_summary,
        resume_configuration=resume_configuration,
        adaptive_tuning=adaptive_tuning,
        metrics_stream=None,
        algorithm=resolved.algorithm,
        schema=resolved.schema,
        effective_seed=effective_seed,
        selected_gpi=int(training.gpi),
        selected_workers=int(selected_workers),
        invocation_target_games=invocation_target,
        final_iteration=final_iteration,
        restored_elapsed_rl_seconds=float(
            restored_state.get("elapsed_rl_seconds", 0.0)
        ),
        training_perf_started=0.0,
    )
    metrics_header = build_training_metrics_header(
        preliminary_context,
        supervised_weights_sha256=supervised_hash,
        requested_device=requested_device,
    )
    metrics_path = _prepare_metrics_file(
        resources.metrics_output_path,
        inputs.start_iteration,
        metadata=metrics_header,
    )
    metrics_stream = open(metrics_path, "a", encoding="utf-8")
    start_time = time.time()
    training_perf_started = time.perf_counter()
    runtime_profile.add(
        "resource_preflight_runner_setup_and_metrics_open",
        training_perf_started - runner_setup_started,
    )
    context = replace(
        preliminary_context,
        metrics_stream=metrics_stream,
        training_perf_started=training_perf_started,
    )
    state = IterationState(
        completed_training_games=inputs.completed_training_games,
        win_rate_window=win_rate_window,
        value_loss_window=value_loss_window,
        ppo_window=ppo_window,
        total_decision_samples=int(
            restored_state.get("total_decision_samples", 0)
        ),
        total_normal_decision_samples=int(
            restored_state.get("total_normal_decision_samples", 0)
        ),
        total_restart_decision_samples=int(
            restored_state.get("total_restart_decision_samples", 0)
        ),
        total_restart_episodes=int(
            restored_state.get("total_restart_episodes", 0)
        ),
        total_restart_duration_s=float(
            restored_state.get("total_restart_duration_s", 0.0)
        ),
        policy_updates_completed=int(
            restored_state.get("policy_updates_completed", 0)
        ),
        clipped_iteration_count=int(
            restored_state.get("clipped_iteration_count", 0)
        ),
        total_rollout_duration_s=float(
            restored_state.get("total_rollout_duration_s", 0.0)
        ),
        total_update_duration_s=float(
            restored_state.get("total_update_duration_s", 0.0)
        ),
        last_checkpoint_time=start_time,
        last_saved_iteration=inputs.start_iteration,
        final_weights_path=(
            Path(execution.resume_weights_path)
            if execution.resume_weights_path is not None
            else Path(resources.rl_weights_path)
        ),
    )
    return RLTrainingSession(
        context=context,
        state=state,
        reporter=reporter,
        runtime_profile=runtime_profile,
        metrics_path=metrics_path,
        start_time=start_time,
        start_iteration=inputs.start_iteration,
        iterations_to_run=iterations_to_run,
        requested_device=requested_device,
        device_fallback_reason=device_fallback_reason,
        initialization_source=initialization_source,
        decision_samples_at_invocation_start=state.total_decision_samples,
        optimizer_steps_at_invocation_start=int(network.optimizer_step_count),
    )
