"""On-policy self-play with masked PPO minibatches by default.

The policy controls only real tile-play decisions. Draw, pass, and single-option
tile-play turns are forced by the rules engine and do not enter the learner
trajectory. Local draw/pass events are distributed to all earlier real
decisions with temporal decay, then combined with a uniform terminal reward and
used as PPO advantages after one global normalization per iteration. The
historical one-update REINFORCE/value-head path remains available explicitly
through ``--no-ppo`` for regression comparisons.

Independent games within each iteration run in deterministic CPU-only workers.
Workers read frozen policies from shared memory and return trajectories; only
the parent process assembles batches, updates weights, writes checkpoints, or
uses the GPU.
"""

import math
from pathlib import Path
import random
import secrets
import time

import numpy as np

from training.rl_cli import (
    _training_kwargs_from_args,
    add_optional_rl_arguments,
    parse_args,
    parse_rl_worker_count,
)
from training.rl_config import (
    DEFAULT_CLIP_GRAD_NORM,
    DEFAULT_DEVICE,
    DEFAULT_GPI,
    DEFAULT_ITERATIONS,
    DEFAULT_MOVING_AVERAGE_WINDOW,
    DEFAULT_NORMALIZE_ADVANTAGES,
    DEFAULT_POOL_REFRESH_GAMES,
    DEFAULT_PPO_ENABLED,
    DEFAULT_TOTAL_TRAINING_GAMES,
    RL_WEIGHTS,
    SL_WEIGHTS,
    TRAINING_OPPONENT,
    VALUE_COEF,
    resolve_training_options,
)
from training.rl_rollout import (
    DEFAULT_GAMMA,
    DEFAULT_REWARD_SCHEMA,
    EVENT_REWARD_DECAY,
    FINAL_PIP_PENALTY,
    LEARNER_DRAW_PENALTY,
    LEARNER_PASS_PENALTY,
    OPPONENT_DRAW_REWARD,
    OPPONENT_PASS_REWARD,
    REWARD_SCHEMAS,
    REWARD_ZERO_EPSILON,
    TERMINAL_LOSS_REWARD,
    TERMINAL_TIE_REWARD,
    TERMINAL_WIN_REWARD,
    EventStats,
    TrainingSample,
    _collect_self_play_steps,
    _collect_steps_vs_heuristic,
    _event_reward_for_action,
    _finish_episode_with_rewards,
    _play_training_game,
    _play_training_game_unprofiled,
    _profile_worker_section,
    _profile_worker_start,
    _remaining_pips,
    _terminal_reward,
    _tile_play_actions,
)
from training.rl_resume import (
    LEGACY_TRAINING_ALGORITHM,
    PPO_TRAINING_ALGORITHM,
    RESUME_POLICY_WEIGHT_NAMES,
    RESUME_STATE_VERSION,
    SUPPORTED_RESUME_STATE_VERSIONS,
    _atomic_network_save,
    _atomic_resume_state_save,
    _checkpoint_matches_encoder,
    _file_sha256,
    _load_initial_network,
    _nested_tuple,
    _restore_rng_state,
    _restore_training_windows,
    _resume_configuration,
    _rng_state_metadata,
    _save_numbered_resume_checkpoint,
    _sl_checkpoint_sha256,
    _training_state_payload,
    _validate_resume_configuration,
    load_resume_state,
    numbered_checkpoint_path,
    resume_state_path,
)
from training.rl_reporting import (
    RLRuntimeProfile,
    _gradient_log_text,
    _merge_parallel_summary,
    _new_parallel_summary,
    _prepare_metrics_file,
    _print_ppo_window,
    _reward_signal_summary,
    _write_metrics_row,
)
from training.rl_parallel import (
    DEFAULT_RL_MINIMUM_GAIN,
    DEFAULT_RL_WORKER_CANDIDATES,
    DEFAULT_RL_WORKERS,
    RLRolloutRunner,
)
from training.adaptive_tuning import (
    DEFAULT_WORKER_BENCHMARK_FRACTION,
    atomic_write_json as atomic_write_tuning_json,
    hardware_metadata,
    hardware_warning,
    run_worker_tuning,
)
from training.ppo import (
    DEFAULT_CLIP_EPSILON,
    DEFAULT_GAMES_PER_MINIBATCH_SCALE,
    DEFAULT_GPU_BUFFER_SAFETY_FRACTION,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MAX_MINIBATCHES,
    DEFAULT_MIN_DECISIONS_PER_MINIBATCH,
    DEFAULT_MIN_MINIBATCHES,
    DEFAULT_STOP_KL,
    DEFAULT_TARGET_KL,
    PPOBuffer,
    ppo_update,
)
from utils.resource_limits import (
    MIB,
    choose_safe_rl_device,
    ensure_ram_available,
)
from utils.runtime_status import format_duration, print_memory_report


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


def _legacy_policy_update(
    network,
    batch,
    *,
    entropy_coef,
    clip_grad_norm,
    normalize_advantages,
    use_value_head,
    value_coef,
    collect_value_predictions=False,
):
    """Apply the historical one-full-buffer update for ``--no-ppo``."""
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
    if use_value_head:
        values = network.predict_values(x_batch)
        if collect_value_predictions:
            value_predictions = _value_prediction_summary(values)
        policy_signal = rewards - values
        value_returns = rewards
    else:
        network.forward(x_batch)
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
        clip_grad_norm=clip_grad_norm,
    )
    if value_predictions is not None:
        metrics["value_predictions_before_update"] = value_predictions
    return metrics


def train(
    iterations=None,
    total_training_games=None,
    gpi=DEFAULT_GPI,
    retune_workers=False,
    training_opponent=TRAINING_OPPONENT,
    learning_rate=0.001,
    entropy_coef=0.01,
    log_interval=10,
    checkpoint_interval=50,
    pool_refresh_games=DEFAULT_POOL_REFRESH_GAMES,
    max_pool_size=50,
    sl_weights_path=SL_WEIGHTS,
    rl_weights_path=RL_WEIGHTS,
    quiet=False,
    progress_callback=None,
    use_value_head=False,
    value_coef=VALUE_COEF,
    gamma=DEFAULT_GAMMA,
    reward_schema=DEFAULT_REWARD_SCHEMA,
    clip_grad_norm=DEFAULT_CLIP_GRAD_NORM,
    normalize_advantages=DEFAULT_NORMALIZE_ADVANTAGES,
    moving_average_window=DEFAULT_MOVING_AVERAGE_WINDOW,
    seed=None,
    device=DEFAULT_DEVICE,
    sl_weights_data=None,
    workers=DEFAULT_RL_WORKERS,
    safety_config=None,
    autotune_fraction=DEFAULT_WORKER_BENCHMARK_FRACTION,
    autotune_minimum_gain=DEFAULT_RL_MINIMUM_GAIN,
    worker_candidates=DEFAULT_RL_WORKER_CANDIDATES,
    status_callback=None,
    metrics_callback=None,
    metrics_output_path=None,
    adaptive_tuning_path=None,
    start_iteration=0,
    resume_weights_path=None,
    resume_state_file=None,
    numbered_checkpoints=False,
    fresh_from_sl=False,
    ppo_enabled=DEFAULT_PPO_ENABLED,
    ppo_clip_epsilon=DEFAULT_CLIP_EPSILON,
    ppo_target_kl=DEFAULT_TARGET_KL,
    ppo_stop_kl=DEFAULT_STOP_KL,
    ppo_max_epochs=DEFAULT_MAX_EPOCHS,
    ppo_min_minibatches=DEFAULT_MIN_MINIBATCHES,
    ppo_max_minibatches=DEFAULT_MAX_MINIBATCHES,
    ppo_games_per_minibatch_scale=DEFAULT_GAMES_PER_MINIBATCH_SCALE,
    ppo_min_decisions_per_minibatch=DEFAULT_MIN_DECISIONS_PER_MINIBATCH,
    prefer_gpu_buffer=True,
    gpu_buffer_safety_fraction=DEFAULT_GPU_BUFFER_SAFETY_FRACTION,
    stop_after_training_games=None,
    shutdown_requested=None,
    allow_total_training_games_extension=False,
    adaptive_tuning_training_games=None,
    force_resume_incompatible=False,
    checkpoint_callback=None,
):
    """Train an exact game budget with the selected on-policy update rule.

    Passing ``iterations`` keeps the old programmatic workload contract and
    implies ``iterations * gpi`` real games. Normal CLI and
    pipeline runs use ``total_training_games`` directly, allowing the final
    iteration to be partial. Worker-benchmark games are always discarded.
    """
    runtime_profile = RLRuntimeProfile()

    resolved_options = resolve_training_options(
        iterations=iterations,
        total_training_games=total_training_games,
        gpi=gpi,
        adaptive_tuning_training_games=adaptive_tuning_training_games,
        retune_workers=retune_workers,
        checkpoint_interval=checkpoint_interval,
        log_interval=log_interval,
        pool_refresh_games=pool_refresh_games,
        max_pool_size=max_pool_size,
        moving_average_window=moving_average_window,
        autotune_fraction=autotune_fraction,
        autotune_minimum_gain=autotune_minimum_gain,
        training_opponent=training_opponent,
        reward_schema=reward_schema,
        ppo_enabled=ppo_enabled,
        use_value_head=use_value_head,
        ppo_clip_epsilon=ppo_clip_epsilon,
        ppo_target_kl=ppo_target_kl,
        ppo_stop_kl=ppo_stop_kl,
        ppo_max_epochs=ppo_max_epochs,
        ppo_min_minibatches=ppo_min_minibatches,
        ppo_max_minibatches=ppo_max_minibatches,
        ppo_games_per_minibatch_scale=ppo_games_per_minibatch_scale,
        ppo_min_decisions_per_minibatch=ppo_min_decisions_per_minibatch,
        gpu_buffer_safety_fraction=gpu_buffer_safety_fraction,
        normalize_advantages=normalize_advantages,
        workers=workers,
        safety_config=safety_config,
    )
    retune_workers = resolved_options.retune_workers
    gpi = resolved_options.gpi
    total_training_games = resolved_options.total_training_games
    tuning_training_games = resolved_options.tuning_training_games
    algorithm = resolved_options.algorithm
    normalize_advantages = resolved_options.normalize_advantages
    workers = resolved_options.workers
    safety_config = resolved_options.safety_config
    schema = resolved_options.schema
    has_resume_weights = resume_weights_path is not None
    has_resume_state = resume_state_file is not None
    if has_resume_weights != has_resume_state:
        raise ValueError(
            "resume_weights_path and resume_state_file must be provided together"
        )
    if start_iteration < 0:
        raise ValueError("start_iteration must be non-negative")
    if start_iteration > 0 and not has_resume_weights:
        raise ValueError("A positive start_iteration requires a complete resume pair")
    if fresh_from_sl and has_resume_weights:
        raise ValueError("fresh_from_sl cannot be combined with a resume pair")

    resume_metadata = None
    resume_pool_snapshots = ()
    completed_training_games = 0
    if has_resume_weights:
        resume_metadata, resume_pool_snapshots = load_resume_state(
            resume_weights_path,
            resume_state_file,
        )
        completed_iteration = int(resume_metadata["completed_iteration"])
        if completed_iteration != int(start_iteration):
            raise ValueError(
                f"Resume state completed iteration {completed_iteration}, but "
                f"start_iteration is {start_iteration}."
            )
        completed_training_games = int(
            resume_metadata["completed_training_games"]
        )
        saved_seed = int(resume_metadata["configuration"]["effective_seed"])
        if seed is not None and int(seed) != saved_seed:
            raise ValueError(
                f"Resume state uses seed {saved_seed}, but seed {int(seed)} was requested."
            )
        effective_seed = saved_seed
    else:
        effective_seed = int(seed) if seed is not None else secrets.randbits(63)
    if completed_training_games >= total_training_games:
        raise ValueError(
            "The resume checkpoint has already completed the requested game budget."
        )
    invocation_target_games = (
        int(total_training_games)
        if stop_after_training_games is None
        else int(stop_after_training_games)
    )
    if not completed_training_games < invocation_target_games <= total_training_games:
        raise ValueError(
            "stop_after_training_games must be above completed games and no "
            "greater than total_training_games"
        )

    random.seed(effective_seed)
    np.random.seed(effective_seed & 0xFFFFFFFF)
    requested_device = device
    device, device_fallback_reason = choose_safe_rl_device(device)
    initial_rl_weights_path = (
        resume_weights_path
        if resume_metadata is not None
        else (None if numbered_checkpoints else rl_weights_path)
    )
    if resume_metadata is not None:
        initialization_source = "numbered_rl_resume"
    elif fresh_from_sl or initial_rl_weights_path is None:
        initialization_source = "supervised"
    elif Path(initial_rl_weights_path).exists():
        initialization_source = "existing_rl"
    else:
        initialization_source = "supervised"
    network = _load_initial_network(
        learning_rate,
        sl_weights_path,
        initial_rl_weights_path,
        quiet=quiet,
        use_value_head=use_value_head,
        device=device,
        sl_weights_data=sl_weights_data,
        fresh_from_sl=fresh_from_sl,
        expected_training_algorithm=algorithm,
    )
    if resume_metadata is not None:
        network.load_optimizer_state_dict(resume_metadata["optimizer_state"])
    sl_weights_sha256 = _sl_checkpoint_sha256(sl_weights_path, sl_weights_data)

    if status_callback is not None:
        emit_status = status_callback
    elif quiet:
        emit_status = lambda _message: None
    else:
        emit_status = lambda message: print(message, flush=True)
    if device_fallback_reason:
        emit_status(
            "RL memory safety: automatic GPU selection fell back to CPU because "
            f"{device_fallback_reason}."
        )

    tuning_path = Path(adaptive_tuning_path) if adaptive_tuning_path else (
        Path(rl_weights_path).parent / "adaptive_tuning.json"
    )
    saved_tuning = (
        None if resume_metadata is None else resume_metadata.get("adaptive_tuning")
    )
    if saved_tuning and not retune_workers:
        warning = hardware_warning(saved_tuning, hardware_metadata(network.device))
        if warning:
            emit_status(f"Warning: {warning}")
    emit_status("-" * 70)
    emit_status("RL rollout-worker tuning")
    emit_status("-" * 70)
    runtime_profile.add(
        "validation_model_load_and_resume",
        time.perf_counter() - runtime_profile.started,
    )
    adaptive_tuning_started = time.perf_counter()
    adaptive_tuning = run_worker_tuning(
        network,
        gpi=gpi,
        total_training_games=tuning_training_games,
        workers=workers,
        retune_workers=retune_workers,
        saved_tuning=saved_tuning,
        worker_benchmark_fraction=autotune_fraction,
        worker_minimum_gain=autotune_minimum_gain,
        worker_candidates=worker_candidates,
        base_seed=effective_seed,
        training_opponent=training_opponent,
        schema=schema,
        gamma=gamma,
        max_pool_size=max_pool_size,
        safety=safety_config,
        pool_snapshots=resume_pool_snapshots,
        output_path=tuning_path,
        status_callback=emit_status,
    )
    runtime_profile.add(
        "adaptive_tuning",
        time.perf_counter() - adaptive_tuning_started,
    )
    runner_setup_started = time.perf_counter()
    selected_gpi = int(gpi)
    selected_workers = int(adaptive_tuning["selected_workers"])
    emit_status(f"Fixed GPI: {selected_gpi}.")
    emit_status("RL update configuration:")
    if ppo_enabled:
        emit_status(
            f"  algorithm: {algorithm} | clip epsilon: {ppo_clip_epsilon:.2f} | "
            f"target KL: {ppo_target_kl:.3f} | stop KL: {ppo_stop_kl:.3f}"
        )
        emit_status(
            f"  max epochs: {ppo_max_epochs} | minibatches: adaptive, "
            f"{ppo_min_minibatches} to {ppo_max_minibatches} | preferred "
            "buffer: GPU | fallback: RAM"
        )
    else:
        emit_status(
            f"  algorithm: {algorithm} | one full-buffer policy-gradient "
            "update per iteration"
        )
        emit_status(
            "  PPO minibatches, ratios, clipping, KL control, and post-update "
            "full-buffer evaluation: disabled"
        )
    emit_status("-" * 70)

    resume_configuration = _resume_configuration(
        total_training_games=total_training_games,
        selected_gpi=selected_gpi,
        selected_workers=selected_workers,
        rl_training_algorithm=algorithm,
        training_opponent=training_opponent,
        learning_rate=learning_rate,
        entropy_coef=entropy_coef,
        pool_refresh_games=pool_refresh_games,
        max_pool_size=max_pool_size,
        use_value_head=use_value_head,
        value_coef=value_coef,
        gamma=gamma,
        reward_schema=reward_schema,
        clip_grad_norm=clip_grad_norm,
        normalize_advantages=normalize_advantages,
        effective_seed=effective_seed,
        device=network.device,
        sl_weights_sha256=sl_weights_sha256,
        ppo_clip_epsilon=ppo_clip_epsilon,
        ppo_target_kl=ppo_target_kl,
        ppo_stop_kl=ppo_stop_kl,
        ppo_max_epochs=ppo_max_epochs,
        ppo_min_minibatches=ppo_min_minibatches,
        ppo_max_minibatches=ppo_max_minibatches,
        ppo_games_per_minibatch_scale=ppo_games_per_minibatch_scale,
        ppo_min_decisions_per_minibatch=ppo_min_decisions_per_minibatch,
        prefer_gpu_buffer=prefer_gpu_buffer,
        gpu_buffer_safety_fraction=gpu_buffer_safety_fraction,
    )
    if resume_metadata is not None:
        ignored = []
        # Backward-compatible programmatic resume: historical callers used a
        # short ``iterations`` run to create an interruption checkpoint, then
        # resumed with a larger final iteration target. The game offset stored
        # in v2 still makes that extension exact.
        if iterations is not None:
            ignored.append("total_training_games")
        if allow_total_training_games_extension:
            ignored.append("total_training_games")
        if retune_workers:
            ignored.append("selected_workers")
        if force_resume_incompatible:
            emit_status(
                "SEVERE WARNING: exact resume compatibility validation was "
                "explicitly overridden. Reproducibility is not guaranteed."
            )
        else:
            _validate_resume_configuration(
                resume_metadata,
                resume_configuration,
                ignored_keys=ignored,
            )

    remaining_training_games = invocation_target_games - completed_training_games
    iterations_to_run = math.ceil(remaining_training_games / selected_gpi)
    final_iteration = int(start_iteration) + iterations_to_run
    estimated_batch_bytes = min(selected_gpi, remaining_training_games) * 52 * 4096
    policy_bytes = sum(
        int(getattr(network, name).nbytes)
        for name in RESUME_POLICY_WEIGHT_NAMES
    )
    shared_pool_size = max_pool_size if training_opponent == "self_play" else 0
    estimated_shared_bytes = (1 + shared_pool_size) * policy_bytes
    ensure_ram_available(
        estimated_shared_bytes + estimated_batch_bytes,
        safety_config.memory_reserve_mb,
        "RL self-play decision buffer and shared-policy preflight",
    )
    if not quiet:
        print_memory_report("RL self-play startup memory")
        print(
            "RL resource preflight: "
            f"requested device={requested_device!r}, selected device={network.device!r}, "
            f"estimated peak host allocation "
            f"{(estimated_shared_bytes + estimated_batch_bytes) / MIB:.1f} MiB."
        )

    runner = RLRolloutRunner(
        network,
        training_opponent=training_opponent,
        schema=schema,
        gamma=gamma,
        max_pool_size=shared_pool_size,
        safety=safety_config,
    )
    if resume_pool_snapshots:
        runner.restore_pool_snapshots(
            resume_pool_snapshots,
            metadata=(resume_metadata or {}).get("opponent_pool_metadata"),
        )
    actual_workers, was_capped, cap_reason = runner.set_workers(selected_workers)
    if was_capped:
        emit_status(
            f"Selected workers reduced from {selected_workers} to {actual_workers} "
            f"by current resource preflight: {cap_reason}."
        )
        selected_workers = actual_workers
        adaptive_tuning["selected_workers"] = int(actual_workers)
        adaptive_tuning["worker_source"] += "_safety_capped"
        atomic_write_tuning_json(tuning_path, adaptive_tuning)
        resume_configuration["selected_workers"] = int(actual_workers)

    if resume_metadata is not None:
        _restore_rng_state(resume_metadata["rng_state"])
        emit_status(
            f"Resuming RL after iteration {start_iteration} and "
            f"{completed_training_games} real games; restored "
            f"{len(resume_pool_snapshots)} opponent-pool snapshot(s)."
        )
    win_rate_window, value_loss_window, ppo_window, restored_state = (
        _restore_training_windows(resume_metadata, moving_average_window)
    )
    total_decision_samples = int(restored_state.get("total_decision_samples", 0))
    ppo_updates_completed = int(restored_state.get("ppo_updates_completed", 0))
    clipped_iteration_count = int(restored_state.get("clipped_iteration_count", 0))
    total_rollout_duration_s = float(
        restored_state.get("total_rollout_duration_s", 0.0)
    )
    total_update_duration_s = float(
        restored_state.get("total_update_duration_s", 0.0)
    )
    restored_elapsed_rl_seconds = float(
        restored_state.get("elapsed_rl_seconds", 0.0)
    )
    parallel_summary = _new_parallel_summary(workers)
    parallel_summary["initial_workers"] = int(actual_workers)

    if metrics_output_path is None:
        weights = Path(rl_weights_path)
        metrics_output_path = weights.with_name(f"{weights.stem}_training_metrics.jsonl")
    metrics_path = _prepare_metrics_file(metrics_output_path, start_iteration)
    metrics_stream = open(metrics_path, "a", encoding="utf-8")
    start_time = time.time()
    training_perf_started = time.perf_counter()
    runtime_profile.add(
        "resource_preflight_runner_setup_and_metrics_open",
        training_perf_started - runner_setup_started,
    )
    last_checkpoint_time = start_time
    last_saved_iteration = int(start_iteration)
    final_weights_path = (
        Path(resume_weights_path)
        if resume_weights_path is not None
        else Path(rl_weights_path)
    )
    completed_this_invocation = 0
    completed_iterations_this_invocation = 0
    stopped_by_shutdown = False
    decision_samples_at_invocation_start = int(total_decision_samples)
    optimizer_steps_at_invocation_start = int(network.optimizer_step_count)
    try:
        for local_iteration in range(1, iterations_to_run + 1):
            if shutdown_requested is not None and shutdown_requested():
                stopped_by_shutdown = True
                break
            iteration = int(start_iteration) + local_iteration
            iteration_started = time.perf_counter()
            games_this_iteration = min(
                selected_gpi,
                invocation_target_games - completed_training_games,
            )
            previous_training_games = completed_training_games
            iteration_accounted_before = runtime_profile.accounted_seconds()
            section_started = time.perf_counter()
            runner.sync_current(network)
            runtime_profile.add(
                "policy_snapshot_synchronization",
                time.perf_counter() - section_started,
            )
            rollout_started = time.perf_counter()
            rollout_results, rollout_info = runner.collect_games(
                previous_training_games,
                games_this_iteration,
                effective_seed,
            )
            rollout_elapsed = time.perf_counter() - rollout_started
            runtime_profile.add("rollout_game_execution", rollout_elapsed)
            runtime_profile.merge_rollout_worker(runner.last_runtime_profile)
            total_rollout_duration_s += rollout_elapsed
            section_started = time.perf_counter()
            _merge_parallel_summary(
                parallel_summary,
                rollout_info,
                phase="rollout",
                iteration=iteration,
            )
            if rollout_info.fallback_count:
                emit_status(
                    f"RL iteration {iteration} retained completed games and "
                    f"reduced workers to {rollout_info.final_workers}: "
                    f"{rollout_info.fallback_history[-1]['reason']}."
                )

            batch = []
            wins = 0
            for result in rollout_results:
                batch.extend(result["samples"])
                wins += int(result["winner"] == result["learner_position"])
            win_rate_window.append(wins / games_this_iteration)
            moving_win_rate = sum(win_rate_window) / len(win_rate_window)
            runtime_profile.add(
                "rollout_parent_aggregation",
                time.perf_counter() - section_started,
            )
            section_started = time.perf_counter()
            reward_summary = _reward_signal_summary(batch, network.xp) if batch else None
            runtime_profile.add(
                "reward_statistics",
                time.perf_counter() - section_started,
            )
            gradient_metrics = None
            ppo_metrics = None
            update_elapsed = 0.0
            if batch:
                section_started = time.perf_counter()
                sample_bytes = sum(
                    int(getattr(sample.x, "nbytes", 0))
                    + int(getattr(sample.legal_mask, "nbytes", 0))
                    for sample in batch
                )
                ensure_ram_available(
                    max(1, sample_bytes * 4),
                    safety_config.memory_reserve_mb,
                    f"RL iteration {iteration} decision-buffer assembly",
                )
                runtime_profile.add(
                    "decision_buffer_memory_preflight",
                    time.perf_counter() - section_started,
                )
                update_started = time.perf_counter()
                if ppo_enabled:
                    buffer_started = time.perf_counter()
                    decision_buffer = PPOBuffer.from_samples(
                        batch,
                        normalize=normalize_advantages,
                    )
                    runtime_profile.add(
                        "ppo_buffer_assembly_and_advantage_normalization",
                        time.perf_counter() - buffer_started,
                    )
                    ppo_started = time.perf_counter()
                    ppo_metrics = ppo_update(
                        network,
                        decision_buffer,
                        actual_games=games_this_iteration,
                        base_seed=effective_seed,
                        iteration=iteration,
                        entropy_coef=entropy_coef,
                        clip_grad_norm=clip_grad_norm,
                        clip_epsilon=ppo_clip_epsilon,
                        target_kl=ppo_target_kl,
                        stop_kl=ppo_stop_kl,
                        max_epochs=ppo_max_epochs,
                        min_minibatches=ppo_min_minibatches,
                        max_minibatches=ppo_max_minibatches,
                        games_per_minibatch_scale=ppo_games_per_minibatch_scale,
                        min_decisions_per_minibatch=ppo_min_decisions_per_minibatch,
                        prefer_gpu_buffer=prefer_gpu_buffer,
                        gpu_buffer_safety_fraction=gpu_buffer_safety_fraction,
                    )
                    network.synchronize()
                    runtime_profile.add(
                        "ppo_update",
                        time.perf_counter() - ppo_started,
                    )
                    runtime_profile.merge_ppo_metrics(ppo_metrics)
                    gradient_metrics = ppo_metrics
                else:
                    legacy_started = time.perf_counter()
                    gradient_metrics = _legacy_policy_update(
                        network,
                        batch,
                        entropy_coef=entropy_coef,
                        clip_grad_norm=clip_grad_norm,
                        normalize_advantages=normalize_advantages,
                        use_value_head=use_value_head,
                        value_coef=value_coef,
                        collect_value_predictions=(
                            use_value_head
                            and not quiet
                            and iteration % log_interval == 0
                        ),
                    )
                    network.synchronize()
                    runtime_profile.add(
                        "legacy_policy_update",
                        time.perf_counter() - legacy_started,
                    )
                update_elapsed = time.perf_counter() - update_started
                total_update_duration_s += update_elapsed
                total_decision_samples += len(batch)
                if gradient_metrics["grad_clipped"]:
                    clipped_iteration_count += 1
                if use_value_head:
                    value_loss_window.append(gradient_metrics["value_loss"])
                ppo_updates_completed += 1

            completed_training_games += games_this_iteration
            completed_this_invocation += games_this_iteration
            completed_iterations_this_invocation += 1
            crossed_pool_refresh = (
                previous_training_games // pool_refresh_games
                < completed_training_games // pool_refresh_games
            )
            section_started = time.perf_counter()
            if batch and training_opponent == "self_play" and crossed_pool_refresh:
                runner.append_pool_snapshot(network, metadata={
                    "origin": "training_update",
                    "introduced_at_rl_games": int(completed_training_games),
                })
            runtime_profile.add(
                "opponent_pool_refresh",
                time.perf_counter() - section_started,
            )

            ppo_log_row = None
            if ppo_metrics is not None:
                ppo_log_row = {
                    **{
                        name: value
                        for name, value in ppo_metrics.items()
                        if name not in {
                            "runtime_timing_seconds",
                            "runtime_profile_detail",
                        }
                    },
                    "games": int(games_this_iteration),
                    "decisions": int(len(batch)),
                    "ppo_seconds": float(update_elapsed),
                    "rollout_seconds": float(rollout_elapsed),
                }
                ppo_window.append(ppo_log_row)

            checkpoint_written = False
            checkpoint_path = None
            checkpoint_resume_path = None
            section_started = time.perf_counter()
            if iteration % checkpoint_interval == 0:
                training_state = _training_state_payload(
                    win_rate_window=win_rate_window,
                    value_loss_window=value_loss_window,
                    ppo_window=ppo_window,
                    total_decision_samples=total_decision_samples,
                    ppo_updates_completed=ppo_updates_completed,
                    clipped_iteration_count=clipped_iteration_count,
                    total_rollout_duration_s=total_rollout_duration_s,
                    total_update_duration_s=total_update_duration_s,
                    elapsed_rl_seconds=(
                        restored_elapsed_rl_seconds
                        + time.perf_counter() - training_perf_started
                    ),
                )
                if numbered_checkpoints:
                    checkpoint_weights_path, checkpoint_state_path = (
                        _save_numbered_resume_checkpoint(
                            network,
                            runner,
                            rl_weights_path,
                            iteration,
                            resume_configuration,
                            runner.worker_count,
                            completed_training_games,
                            adaptive_tuning,
                            training_state,
                        )
                    )
                    final_weights_path = checkpoint_weights_path
                    last_saved_iteration = iteration
                    checkpoint_resume_path = str(checkpoint_state_path)
                else:
                    _atomic_network_save(network, rl_weights_path)
                    checkpoint_weights_path = Path(rl_weights_path)
                checkpoint_written = True
                checkpoint_path = str(checkpoint_weights_path)
                now = time.time()
                checkpoint_elapsed = now - last_checkpoint_time
                last_checkpoint_time = now
                if not quiet:
                    print(
                        f"  [checkpoint] saved {checkpoint_weights_path} | "
                        f"{completed_training_games}/{total_training_games} games | "
                        f"time since previous checkpoint: "
                        f"{format_duration(checkpoint_elapsed)}"
                    )
            runtime_profile.add(
                "checkpoint_serialization",
                time.perf_counter() - section_started,
            )

            section_started = time.perf_counter()
            if iteration % log_interval == 0 and not quiet:
                if reward_summary is None:
                    print(
                        f"Iteration {iteration} | {games_this_iteration} games | "
                        "no real policy decisions"
                    )
                else:
                    win_label = "vs pool" if training_opponent == "self_play" else "vs heuristic"
                    pool_suffix = (
                        f" | pool: {len(runner.bank.pool_slots)}"
                        if training_opponent == "self_play" else ""
                    )
                    print(
                        f"Iteration {iteration} | games {games_this_iteration} | cumulative "
                        f"{completed_training_games}/{total_training_games} | reward "
                        f"mean/std/min/max: {reward_summary['reward_mean']:+.2f}/"
                        f"{reward_summary['reward_std']:.2f}/"
                        f"{reward_summary['reward_min']:+.2f}/"
                        f"{reward_summary['reward_max']:+.2f} | good/neutral/bad: "
                        f"{reward_summary['good_pct']:.0f}%/"
                        f"{reward_summary['neutral_pct']:.0f}%/"
                        f"{reward_summary['bad_pct']:.0f}% | wins {win_label}: "
                        f"{wins}/{games_this_iteration} "
                        f"(avg/{len(win_rate_window)}: {moving_win_rate:.1%})"
                        f"{pool_suffix} | grad: {_gradient_log_text(gradient_metrics)}"
                    )
                    value_predictions = gradient_metrics.get(
                        "value_predictions_before_update"
                    )
                    if use_value_head and value_predictions is not None:
                        print(
                            "  Value head: pre-update V(s) mean/std/min/max "
                            f"{value_predictions['mean']:+.3f}/"
                            f"{value_predictions['std']:.3f}/"
                            f"{value_predictions['min']:+.3f}/"
                            f"{value_predictions['max']:+.3f} over "
                            f"{value_predictions['sample_count']} decisions | "
                            f"value loss {gradient_metrics['value_loss']:.3f} "
                            f"(avg/{len(value_loss_window)}: "
                            f"{sum(value_loss_window) / len(value_loss_window):.3f})"
                        )
                    if ppo_enabled:
                        _print_ppo_window(ppo_window)
            runtime_profile.add(
                "console_logging",
                time.perf_counter() - section_started,
            )

            section_started = time.perf_counter()
            moving_value_loss = (
                float(sum(value_loss_window) / len(value_loss_window))
                if value_loss_window else None
            )
            row = {
                "iteration": int(iteration),
                "total_iterations": int(final_iteration),
                "games": int(games_this_iteration),
                "cumulative_games": int(completed_training_games),
                "cumulative_training_games": int(completed_training_games),
                "total_training_games": int(total_training_games),
                "games_per_iteration": int(selected_gpi),
                "decision_sample_count": int(len(batch)),
                "decisions": int(len(batch)),
                "wins_in_batch": int(wins),
                "batch_win_rate": float(wins / games_this_iteration),
                "moving_average_win_rate": float(moving_win_rate),
                "reward_mean": None if reward_summary is None else float(reward_summary["reward_mean"]),
                "reward_std": None if reward_summary is None else float(reward_summary["reward_std"]),
                "reward_min": None if reward_summary is None else float(reward_summary["reward_min"]),
                "reward_max": None if reward_summary is None else float(reward_summary["reward_max"]),
                "good_pct": None if reward_summary is None else float(reward_summary["good_pct"]),
                "neutral_pct": None if reward_summary is None else float(reward_summary["neutral_pct"]),
                "bad_pct": None if reward_summary is None else float(reward_summary["bad_pct"]),
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
                "selected_workers": int(runner.worker_count),
                "pool_size": int(len(runner.bank.pool_slots)),
                "rollout_seconds": float(rollout_elapsed),
                "ppo_seconds": float(update_elapsed if ppo_enabled else 0.0),
                "rollout_duration_s": float(rollout_elapsed),
                "update_duration_s": float(update_elapsed),
                "total_iteration_seconds": float(time.perf_counter() - iteration_started),
                "iteration_duration_s": float(time.perf_counter() - iteration_started),
                "checkpoint_written": bool(checkpoint_written),
                "checkpoint_path": checkpoint_path,
                "elapsed_training_s": float(time.perf_counter() - training_perf_started),
                "rl_training_algorithm": algorithm,
            }
            runtime_profile.add(
                "metrics_payload_construction",
                time.perf_counter() - section_started,
            )
            section_started = time.perf_counter()
            _write_metrics_row(metrics_stream, row)
            runtime_profile.add(
                "metrics_jsonl_write_and_fsync",
                time.perf_counter() - section_started,
            )
            section_started = time.perf_counter()
            if (
                checkpoint_callback is not None
                and checkpoint_written
                and numbered_checkpoints
            ):
                checkpoint_callback({
                    "rl_weights_path": checkpoint_path,
                    "resume_state_path": checkpoint_resume_path,
                    "completed_training_games": int(completed_training_games),
                    "rl_iterations_completed": int(iteration),
                    "games_per_iteration": int(selected_gpi),
                    "final_workers": int(runner.worker_count),
                })
            if metrics_callback is not None:
                metrics_callback(dict(row))
            if progress_callback is not None:
                progress_callback(completed_training_games, total_training_games)
            runtime_profile.add(
                "pipeline_and_progress_callbacks",
                time.perf_counter() - section_started,
            )
            accounted_this_iteration = (
                runtime_profile.accounted_seconds() - iteration_accounted_before
            )
            runtime_profile.add(
                "iteration_control_overhead",
                max(
                    0.0,
                    time.perf_counter() - iteration_started - accounted_this_iteration,
                ),
            )

            if shutdown_requested is not None and shutdown_requested():
                stopped_by_shutdown = True
                break

        finalization_started = time.perf_counter()
        if completed_training_games != invocation_target_games and not stopped_by_shutdown:
            raise AssertionError(
                f"RL completed {completed_training_games} games, expected "
                f"{invocation_target_games}."
            )
        actual_final_iteration = (
            int(start_iteration) + completed_iterations_this_invocation
        )
        final_resume_path = resume_state_path(final_weights_path)
        if numbered_checkpoints and (
            last_saved_iteration != actual_final_iteration
            or not final_weights_path.is_file()
            or not final_resume_path.is_file()
        ):
            training_state = _training_state_payload(
                win_rate_window=win_rate_window,
                value_loss_window=value_loss_window,
                ppo_window=ppo_window,
                total_decision_samples=total_decision_samples,
                ppo_updates_completed=ppo_updates_completed,
                clipped_iteration_count=clipped_iteration_count,
                total_rollout_duration_s=total_rollout_duration_s,
                total_update_duration_s=total_update_duration_s,
                elapsed_rl_seconds=(
                    restored_elapsed_rl_seconds
                    + time.perf_counter() - training_perf_started
                ),
            )
            final_weights_path, _final_state_path = _save_numbered_resume_checkpoint(
                network,
                runner,
                rl_weights_path,
                actual_final_iteration,
                resume_configuration,
                runner.worker_count,
                completed_training_games,
                adaptive_tuning,
                training_state,
            )
            last_saved_iteration = actual_final_iteration
    finally:
        metrics_stream.close()
        final_runtime_workers = runner.worker_count
        pool_snapshot_count = len(runner.bank.pool_slots)
        runner.close()

    parallel_summary["final_workers"] = final_runtime_workers
    if not numbered_checkpoints:
        _atomic_network_save(network, rl_weights_path)
        final_weights_path = Path(rl_weights_path)
    elapsed_time = time.time() - start_time
    if not quiet:
        print(f"\nTraining complete. Total elapsed time: {format_duration(elapsed_time)}.")
        print(f"Final weights: {final_weights_path}")

    worker_results = adaptive_tuning.get("worker_results", [])
    autotune_summary = {
        "optimal_workers": int(selected_workers),
        "candidate_workers": [
            int(row["requested_workers"]) for row in worker_results
        ],
        "benchmark_fraction": float(autotune_fraction),
        "minimum_gain": float(autotune_minimum_gain),
        "iterations_per_test": None,
        "games_per_test": int(adaptive_tuning.get("worker_test_games", 0)),
        "reused_iteration_count": 0,
        "reused_game_count": 0,
        "discarded_game_count": sum(
            int(row.get("actual_games", 0))
            for row in worker_results
        ),
        "attempts": worker_results,
    }
    completed_iterations = completed_iterations_this_invocation
    runtime_profile.add(
        "final_checkpoint_shutdown_and_summary",
        time.perf_counter() - finalization_started,
    )
    runtime_profile_delta = runtime_profile.finish(
        games=completed_this_invocation,
        iterations=completed_iterations_this_invocation,
        decisions=total_decision_samples - decision_samples_at_invocation_start,
        optimizer_steps=(
            network.optimizer_step_count - optimizer_steps_at_invocation_start
        ),
    )
    return {
        "iterations": int(actual_final_iteration),
        "rl_iterations_completed": int(actual_final_iteration),
        "completed_iterations_this_run": int(completed_iterations),
        "start_iteration": int(start_iteration),
        "start_training_games": int(
            completed_training_games - completed_this_invocation
        ),
        "games_per_iteration": int(selected_gpi),
        "requested_games_per_iteration": int(gpi),
        "total_training_games": int(total_training_games),
        "completed_training_games": int(completed_training_games),
        "invocation_target_training_games": int(invocation_target_games),
        "shutdown_requested": bool(stopped_by_shutdown),
        "training_opponent": training_opponent,
        "learning_rate": learning_rate,
        "entropy_coef": entropy_coef,
        "use_value_head": use_value_head,
        "value_coef": value_coef if use_value_head else None,
        "gamma": gamma,
        "reward_schema": reward_schema,
        "clip_grad_norm": clip_grad_norm,
        "normalize_advantages": normalize_advantages,
        "moving_average_window": moving_average_window,
        "seed": seed,
        "effective_seed": effective_seed,
        "device": network.device,
        "requested_device": requested_device,
        "device_fallback_reason": device_fallback_reason,
        "requested_workers": workers,
        "selected_workers": int(selected_workers),
        "final_workers": int(final_runtime_workers),
        "autotune": autotune_summary,
        "adaptive_tuning": adaptive_tuning,
        "adaptive_tuning_path": str(tuning_path),
        "parallel": parallel_summary,
        "rl_weights_path": str(final_weights_path),
        "metrics_output_path": str(metrics_path),
        "initialization_source": initialization_source,
        "fresh_from_sl": bool(fresh_from_sl),
        "numbered_checkpoints": bool(numbered_checkpoints),
        "pool_refresh_games": int(pool_refresh_games),
        "total_decision_samples": int(total_decision_samples),
        "trainable_decisions_seen": int(total_decision_samples),
        "ppo_updates_completed": int(ppo_updates_completed),
        "decisions_per_game": float(
            total_decision_samples / max(1, completed_training_games)
        ),
        "clipped_iteration_count": int(clipped_iteration_count),
        "clipped_iteration_rate": float(
            clipped_iteration_count / max(1, actual_final_iteration)
        ),
        "pool_snapshot_count": int(pool_snapshot_count),
        "total_rollout_duration_s": float(total_rollout_duration_s),
        "total_update_duration_s": float(total_update_duration_s),
        "optimizer_step_count": int(network.optimizer_step_count),
        "elapsed_rl_seconds": float(
            restored_elapsed_rl_seconds + time.perf_counter() - training_perf_started
        ),
        "resume_state_path": (
            str(resume_state_path(final_weights_path))
            if numbered_checkpoints else None
        ),
        "rl_training_algorithm": algorithm,
        "ppo_enabled": bool(ppo_enabled),
        "ppo_configuration": {
            "clip_epsilon": float(ppo_clip_epsilon),
            "target_kl": float(ppo_target_kl),
            "stop_kl": float(ppo_stop_kl),
            "max_epochs": int(ppo_max_epochs),
            "min_minibatches": int(ppo_min_minibatches),
            "max_minibatches": int(ppo_max_minibatches),
            "games_per_minibatch_scale": int(ppo_games_per_minibatch_scale),
            "min_decisions_per_minibatch": int(ppo_min_decisions_per_minibatch),
            "prefer_gpu_buffer": bool(prefer_gpu_buffer),
            "gpu_buffer_safety_fraction": float(gpu_buffer_safety_fraction),
        },
        "runtime_profile_delta": runtime_profile_delta,
        "duration_s": elapsed_time,
    }


def _run_compact_cli(args, training_kwargs):
    """Run the standalone CLI with the pipeline's compact presentation."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    print("\nRL self-play")
    started = time.time()
    fixed_gpi = training_kwargs["gpi"]
    planned_games = (
        args.iterations * fixed_gpi
        if args.iterations is not None
        else (
            DEFAULT_TOTAL_TRAINING_GAMES
            if args.total_training_games is None
            else args.total_training_games
        )
    )
    initial_games = 0
    if args.resume_weights_path and args.resume_state_file:
        resume_metadata, _pool = load_resume_state(
            args.resume_weights_path,
            args.resume_state_file,
        )
        initial_games = int(resume_metadata["completed_training_games"])

    if tqdm is None:
        progress_interval = max(1, planned_games // 10)
        last_reported = initial_games

        def progress(done, total):
            nonlocal last_reported
            if done == total or done - last_reported >= progress_interval:
                print(f"RL self-play progress: {done}/{total} games", flush=True)
                last_reported = done

        def status(message):
            print(message, flush=True)

        summary = train(
            **training_kwargs,
            quiet=True,
            progress_callback=progress,
            status_callback=status,
        )
    else:
        with tqdm(
            total=planned_games,
            initial=initial_games,
            desc="RL self-play",
            unit="game",
            leave=True,
        ) as progress_bar:

            def progress(done, total):
                if progress_bar.total != total:
                    progress_bar.total = total
                    progress_bar.refresh()
                if done > progress_bar.n:
                    progress_bar.update(done - progress_bar.n)

            summary = train(
                **training_kwargs,
                quiet=True,
                progress_callback=progress,
                status_callback=tqdm.write,
            )

    elapsed = time.time() - started
    print(
        f"RL self-play complete in {format_duration(elapsed)} | "
        f"{summary['total_training_games']} exact training games in "
        f"{summary['completed_iterations_this_run']} iteration(s), GPI "
        f"{summary['games_per_iteration']}, "
        f"{summary['selected_workers']} rollout worker(s), "
        f"algorithm {summary['rl_training_algorithm']}, "
        f"weights {summary['rl_weights_path']}"
    )
    return summary


def main(argv=None):
    args = parse_args(argv)
    training_kwargs = _training_kwargs_from_args(args)
    if args.compact:
        return _run_compact_cli(args, training_kwargs)
    return train(**training_kwargs)


if __name__ == "__main__":
    main()
