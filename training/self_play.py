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

import argparse
import json
import math
import os
from pathlib import Path
import random
import secrets
import time

import numpy as np

from agents.rl_nn import DEVICES
from diagnostics.parallel_runner import (
    MAX_PARALLEL_WORKERS,
    ParallelSafetyConfig,
    cap_parallel_workers,
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
from training.rl_parallel import (
    DEFAULT_RL_AUTOTUNE_FRACTION,
    DEFAULT_RL_MINIMUM_GAIN,
    DEFAULT_RL_WORKER_CANDIDATES,
    DEFAULT_RL_WORKERS,
    RLRolloutRunner,
    worker_count as parse_rl_worker_count,
)
from training.adaptive_tuning import (
    DEFAULT_GPI_BENCHMARK_GAMES_TARGET,
    DEFAULT_GPI_BENCHMARK_WORKERS,
    DEFAULT_GPI_CANDIDATES,
    DEFAULT_WORKER_BENCHMARK_FRACTION,
    atomic_write_json as atomic_write_tuning_json,
    hardware_metadata,
    hardware_warning,
    run_adaptive_tuning,
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
    MemorySafetyError,
    choose_safe_rl_device,
    effective_gpu_available_bytes,
    ensure_ram_available,
)
from utils.runtime_status import format_duration, print_memory_report

# The array backend for a given run is resolved once, inside train(), from
# the network's resolved `device` parameter -- it always
# matches whatever PolicyNetwork itself is using, rather than being fixed at
# import time.
DEFAULT_DEVICE = "auto"
DEFAULT_ITERATIONS = 1000
DEFAULT_GAMES_PER_ITERATION = 100
DEFAULT_TOTAL_TRAINING_GAMES = 100_000
DEFAULT_POOL_REFRESH_GAMES = 400
DEFAULT_ADAPTIVE_GPI = True
DEFAULT_PPO_ENABLED = True

SL_WEIGHTS = "models/domino_sl_weights.npz"
RL_WEIGHTS = "models/domino_rl_weights.npz"
TRAINING_OPPONENT = "self_play"
VALUE_COEF = 0.5
DEFAULT_CLIP_GRAD_NORM = 5.0
DEFAULT_MOVING_AVERAGE_WINDOW = 10
# ``None`` resolves to on for PPO and off for the legacy one-update regression
# path. Explicit CLI flags always win.
DEFAULT_NORMALIZE_ADVANTAGES = None

def _reward_signal_summary(samples, xp=None):
    """Return compact diagnostics for finalized decision rewards.

    ``reward_std`` disambiguates a falling value loss from a merely
    low-variance batch: since a value head that has not learned anything
    predicts close to the batch mean, its loss is approximately
    ``0.5 * reward_std ** 2`` — logging the standard deviation next to the
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


def _legacy_policy_update(
    network,
    batch,
    *,
    entropy_coef,
    clip_grad_norm,
    normalize_advantages,
    use_value_head,
    value_coef,
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
    policy_signal = rewards
    if use_value_head:
        values = network.predict_values(x_batch)
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
    return network.backward_policy_gradient(
        actions,
        policy_signal,
        legal_masks=legal_masks,
        entropy_coef=entropy_coef,
        value_returns=value_returns,
        value_coef=value_coef,
        clip_grad_norm=clip_grad_norm,
    )


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
    gpu_count = sum(row["buffer_location"] == "gpu" for row in rows)
    print(
        f"  PPO/{count}: buffer GPU {gpu_count}, RAM {count - gpu_count} | bytes "
        f"{np.mean(buffer_bytes):.0f} avg/{max(buffer_bytes)} max | PPO update "
        f"{sum(row['ppo_seconds'] for row in rows):.2f}s total/"
        f"{np.mean([row['ppo_seconds'] for row in rows]):.3f}s avg | rollout "
        f"{sum(row['rollout_seconds'] for row in rows):.2f}s total/"
        f"{np.mean([row['rollout_seconds'] for row in rows]):.3f}s avg"
    )


def _prepare_metrics_file(path, start_iteration):
    """Create or truncate the built-in JSONL trace to the resumed checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    retained = []
    if start_iteration and path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("iteration", 0)) <= int(start_iteration):
                retained.append(row)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
    )
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            for row in retained:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def train(
    iterations=None,
    total_training_games=None,
    games_per_iteration=None,
    adaptive_gpi=None,
    gpi_candidates=DEFAULT_GPI_CANDIDATES,
    gpi_benchmark_games_target=DEFAULT_GPI_BENCHMARK_GAMES_TARGET,
    retune_gpi=False,
    retune_workers=False,
    retune_all=False,
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
    """Train an exact game budget with isolated adaptive tuning and PPO v1.

    Passing ``iterations`` keeps the old programmatic workload contract and
    implies ``iterations * games_per_iteration`` real games. Normal CLI and
    pipeline runs use ``total_training_games`` directly, allowing the final
    iteration to be partial. Benchmark games are always discarded.
    """
    runtime_profile_started = time.perf_counter()
    runtime_sections = {}
    runtime_ppo_sections = {}
    runtime_rollout_worker = {}
    runtime_ppo_optimizer_detail = {}
    runtime_ppo_full_buffer_detail = {}

    def add_runtime(section, seconds):
        runtime_sections[section] = runtime_sections.get(section, 0.0) + float(seconds)

    def merge_numeric_tree(target, source):
        for key, value in source.items():
            if isinstance(value, dict):
                merge_numeric_tree(target.setdefault(key, {}), value)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                target[key] = target.get(key, 0) + value

    retune_gpi = bool(retune_gpi or retune_all)
    retune_workers = bool(retune_workers or retune_all)
    manual_gpi_explicit = games_per_iteration is not None
    games_per_iteration = (
        DEFAULT_GAMES_PER_ITERATION
        if games_per_iteration is None
        else int(games_per_iteration)
    )
    if games_per_iteration < 1:
        raise ValueError("games_per_iteration must be positive")
    if iterations is not None:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        implied_total = int(iterations) * int(games_per_iteration)
        if total_training_games is not None and int(total_training_games) != implied_total:
            raise ValueError(
                "iterations * games_per_iteration conflicts with total_training_games"
            )
        total_training_games = implied_total
        if adaptive_gpi is None:
            adaptive_gpi = False
    else:
        total_training_games = (
            DEFAULT_TOTAL_TRAINING_GAMES
            if total_training_games is None
            else int(total_training_games)
        )
        if adaptive_gpi is None:
            adaptive_gpi = DEFAULT_ADAPTIVE_GPI and not manual_gpi_explicit
    if total_training_games < 1:
        raise ValueError("total_training_games must be positive")
    tuning_training_games = (
        int(total_training_games)
        if adaptive_tuning_training_games is None
        else int(adaptive_tuning_training_games)
    )
    if tuning_training_games < 1:
        raise ValueError("adaptive_tuning_training_games must be positive")
    if checkpoint_interval < 1 or log_interval < 1:
        raise ValueError("checkpoint_interval and log_interval must be positive")
    if pool_refresh_games < 1:
        raise ValueError("pool_refresh_games must be positive")
    if max_pool_size < 0:
        raise ValueError("max_pool_size must be non-negative")
    if moving_average_window < 1:
        raise ValueError("moving_average_window must be positive")
    if not 0 < float(autotune_fraction) <= 1:
        raise ValueError("autotune_fraction must be in (0, 1]")
    if float(autotune_minimum_gain) < 0:
        raise ValueError("autotune_minimum_gain must be non-negative")
    if training_opponent not in ("self_play", "heuristic"):
        raise ValueError("training_opponent must be 'self_play' or 'heuristic'.")
    if reward_schema not in REWARD_SCHEMAS:
        raise ValueError(f"Unknown reward_schema {reward_schema!r}.")
    if ppo_enabled and use_value_head:
        raise ValueError(
            "PPO v1 keeps the critic disabled; use --no-ppo for value-head regression."
        )
    if ppo_enabled:
        if not 0 < float(ppo_clip_epsilon) < 1:
            raise ValueError("ppo_clip_epsilon must be in (0, 1)")
        if not 0 < float(ppo_target_kl) <= float(ppo_stop_kl):
            raise ValueError("PPO KL thresholds require 0 < target_kl <= stop_kl")
        if not 1 <= int(ppo_max_epochs) <= 4:
            raise ValueError("ppo_max_epochs must be between 1 and 4")
        if int(ppo_min_minibatches) < 1 or int(ppo_max_minibatches) < int(
            ppo_min_minibatches
        ):
            raise ValueError("Invalid PPO minibatch bounds")
        if int(ppo_games_per_minibatch_scale) < 1:
            raise ValueError("ppo_games_per_minibatch_scale must be positive")
        if int(ppo_min_decisions_per_minibatch) < 1:
            raise ValueError("ppo_min_decisions_per_minibatch must be positive")
        if not 0 < float(gpu_buffer_safety_fraction) <= 1:
            raise ValueError("gpu_buffer_safety_fraction must be in (0, 1]")
    algorithm = PPO_TRAINING_ALGORITHM if ppo_enabled else LEGACY_TRAINING_ALGORITHM
    normalize_advantages = (
        bool(ppo_enabled)
        if normalize_advantages is None
        else bool(normalize_advantages)
    )
    if workers != "auto":
        workers = int(workers)
        if not 1 <= workers <= MAX_PARALLEL_WORKERS:
            raise ValueError(
                f"workers must be 'auto' or between 1 and {MAX_PARALLEL_WORKERS}"
            )
    safety_config = safety_config or ParallelSafetyConfig()
    schema = REWARD_SCHEMAS[reward_schema]

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
    if saved_tuning and not (retune_gpi or retune_workers):
        warning = hardware_warning(saved_tuning, hardware_metadata(network.device))
        if warning:
            emit_status(f"Warning: {warning}")
    emit_status("-" * 70)
    emit_status("Adaptive RL tuning")
    emit_status("-" * 70)
    add_runtime(
        "validation_model_load_and_resume",
        time.perf_counter() - runtime_profile_started,
    )
    adaptive_tuning_started = time.perf_counter()
    adaptive_tuning = run_adaptive_tuning(
        network,
        total_training_games=tuning_training_games,
        manual_gpi=games_per_iteration,
        adaptive_gpi=adaptive_gpi,
        workers=workers,
        retune_gpi=retune_gpi,
        retune_workers=retune_workers,
        saved_tuning=saved_tuning,
        gpi_candidates=gpi_candidates,
        gpi_benchmark_games_target=gpi_benchmark_games_target,
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
    add_runtime(
        "adaptive_tuning",
        time.perf_counter() - adaptive_tuning_started,
    )
    runner_setup_started = time.perf_counter()
    selected_gpi = int(adaptive_tuning["selected_gpi"])
    selected_workers = int(adaptive_tuning["selected_workers"])
    emit_status("PPO configuration:")
    emit_status(
        f"  enabled: {bool(ppo_enabled)} | clip epsilon: {ppo_clip_epsilon:.2f} | "
        f"target KL: {ppo_target_kl:.3f} | stop KL: {ppo_stop_kl:.3f}"
    )
    emit_status(
        f"  max epochs: {ppo_max_epochs} | minibatches: adaptive, "
        f"{ppo_min_minibatches} to {ppo_max_minibatches} | preferred buffer: GPU | "
        "fallback: RAM"
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
        if retune_gpi:
            ignored.append("selected_gpi")
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
        "RL self-play, PPO buffer, and shared-policy preflight",
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
    add_runtime(
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
            iteration_accounted_before = sum(runtime_sections.values())
            section_started = time.perf_counter()
            runner.sync_current(network)
            add_runtime("policy_snapshot_synchronization", time.perf_counter() - section_started)
            rollout_started = time.perf_counter()
            rollout_results, rollout_info = runner.collect_games(
                previous_training_games,
                games_this_iteration,
                effective_seed,
            )
            rollout_elapsed = time.perf_counter() - rollout_started
            add_runtime("rollout_game_execution", rollout_elapsed)
            merge_numeric_tree(
                runtime_rollout_worker,
                runner.last_runtime_profile,
            )
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
            add_runtime(
                "rollout_parent_aggregation",
                time.perf_counter() - section_started,
            )
            section_started = time.perf_counter()
            reward_summary = _reward_signal_summary(batch, network.xp) if batch else None
            add_runtime("reward_statistics", time.perf_counter() - section_started)
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
                add_runtime(
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
                    add_runtime(
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
                    add_runtime("ppo_update", time.perf_counter() - ppo_started)
                    for name, seconds in ppo_metrics[
                        "runtime_timing_seconds"
                    ].items():
                        if name != "total":
                            runtime_ppo_sections[name] = (
                                runtime_ppo_sections.get(name, 0.0) + float(seconds)
                            )
                    runtime_detail = ppo_metrics.get("runtime_profile_detail", {})
                    merge_numeric_tree(
                        runtime_ppo_optimizer_detail,
                        runtime_detail.get("optimizer_step", {}),
                    )
                    merge_numeric_tree(
                        runtime_ppo_full_buffer_detail,
                        runtime_detail.get("full_buffer_evaluation", {}),
                    )
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
                    )
                    network.synchronize()
                    add_runtime(
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
            add_runtime("opponent_pool_refresh", time.perf_counter() - section_started)

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
            add_runtime("checkpoint_serialization", time.perf_counter() - section_started)

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
                    value_suffix = ""
                    if use_value_head and value_loss_window:
                        value_suffix = (
                            f" | value loss: {gradient_metrics['value_loss']:.3f} "
                            f"(avg/{len(value_loss_window)}: "
                            f"{sum(value_loss_window) / len(value_loss_window):.3f})"
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
                        f"{value_suffix}"
                    )
                    if ppo_enabled:
                        _print_ppo_window(ppo_window)
            add_runtime("console_logging", time.perf_counter() - section_started)

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
            add_runtime("metrics_payload_construction", time.perf_counter() - section_started)
            section_started = time.perf_counter()
            metrics_stream.write(json.dumps(row, sort_keys=True) + "\n")
            metrics_stream.flush()
            os.fsync(metrics_stream.fileno())
            add_runtime("metrics_jsonl_write_and_fsync", time.perf_counter() - section_started)
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
            add_runtime("pipeline_and_progress_callbacks", time.perf_counter() - section_started)
            accounted_this_iteration = (
                sum(runtime_sections.values()) - iteration_accounted_before
            )
            add_runtime(
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
    add_runtime("final_checkpoint_shutdown_and_summary", time.perf_counter() - finalization_started)
    runtime_total_seconds = time.perf_counter() - runtime_profile_started
    runtime_accounted_seconds = sum(runtime_sections.values())
    runtime_sections["unaccounted"] = max(
        0.0,
        runtime_total_seconds - runtime_accounted_seconds,
    )
    runtime_profile_delta = {
        "execution_count": 1,
        "games": int(completed_this_invocation),
        "iterations": int(completed_iterations_this_invocation),
        "decisions": int(total_decision_samples - decision_samples_at_invocation_start),
        "optimizer_steps": int(
            network.optimizer_step_count - optimizer_steps_at_invocation_start
        ),
        "execution_seconds": float(runtime_total_seconds),
        "sections_seconds": {
            name: float(seconds) for name, seconds in runtime_sections.items()
        },
        "ppo_sections_seconds": {
            name: float(seconds) for name, seconds in runtime_ppo_sections.items()
        },
        "rollout_worker": runtime_rollout_worker,
        "ppo_optimizer_step": runtime_ppo_optimizer_detail,
        "ppo_full_buffer_evaluation": runtime_ppo_full_buffer_detail,
    }
    return {
        "iterations": int(actual_final_iteration),
        "rl_iterations_completed": int(actual_final_iteration),
        "completed_iterations_this_run": int(completed_iterations),
        "start_iteration": int(start_iteration),
        "start_training_games": int(
            completed_training_games - completed_this_invocation
        ),
        "games_per_iteration": int(selected_gpi),
        "requested_games_per_iteration": int(games_per_iteration),
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


def add_optional_rl_arguments(parser, *, fresh_from_sl_default=False):
    """Add self-play hyperparameter and rollout-resource flags to ``parser``."""
    group = parser.add_argument_group("optional reinforcement-learning controls")
    group.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=(
            "Legacy/manual iteration budget. When supplied, total games are "
            "iterations x games-per-iteration and adaptive GPI is off unless "
            "explicitly re-enabled."
        ),
    )
    group.add_argument(
        "--total-training-games",
        type=int,
        default=None,
        help=(
            f"Exact number of real training games; benchmark games are excluded "
            f"(normal default: {DEFAULT_TOTAL_TRAINING_GAMES})."
        ),
    )
    group.add_argument(
        "--games-per-iteration",
        type=int,
        default=None,
        help=(
            f"Manual GPI (default fallback: {DEFAULT_GAMES_PER_ITERATION}); "
            "specifying it disables GPI autotuning unless --adaptive-gpi is also set."
        ),
    )
    adaptive = group.add_mutually_exclusive_group()
    adaptive.add_argument(
        "--adaptive-gpi",
        dest="adaptive_gpi",
        action="store_true",
        default=None,
        help="Benchmark the fixed GPI candidate list before real training (default).",
    )
    adaptive.add_argument(
        "--no-adaptive-gpi",
        dest="adaptive_gpi",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Use the manual/default games-per-iteration directly.",
    )
    group.add_argument(
        "--gpi-candidates",
        nargs="+",
        type=int,
        default=list(DEFAULT_GPI_CANDIDATES),
        metavar="N",
        help=(
            "GPI values tested with "
            f"{DEFAULT_GPI_BENCHMARK_WORKERS} frozen-policy workers."
        ),
    )
    group.add_argument(
        "--gpi-benchmark-games-target",
        type=int,
        default=DEFAULT_GPI_BENCHMARK_GAMES_TARGET,
        help="Candidate budget used as floor(target / GPI) complete batches.",
    )
    group.add_argument("--retune-gpi", action="store_true")
    group.add_argument("--retune-workers", action="store_true")
    group.add_argument("--retune-all", action="store_true")
    group.add_argument(
        "--training-opponent",
        choices=("self_play", "heuristic"),
        default=TRAINING_OPPONENT,
        help="Play against a pool of frozen snapshots or the fixed heuristic agent.",
    )
    group.add_argument("--learning-rate", type=float, default=0.001)
    group.add_argument("--entropy-coef", type=float, default=0.01)
    group.add_argument("--log-interval", type=int, default=10)
    group.add_argument("--checkpoint-interval", type=int, default=50)
    group.add_argument(
        "--pool-refresh-games",
        type=int,
        default=DEFAULT_POOL_REFRESH_GAMES,
        help=(
            "Cumulative training-game interval between opponent-pool snapshots; "
            "a threshold crossed inside a batch refreshes once after that batch."
        ),
    )
    group.add_argument("--max-pool-size", type=int, default=50)
    group.add_argument("--sl-weights-path", default=SL_WEIGHTS)
    group.add_argument("--rl-weights-path", default=RL_WEIGHTS)
    group.add_argument(
        "--adaptive-tuning-path",
        default=None,
        help="Adaptive-tuning JSON path (default: next to RL weights).",
    )
    group.add_argument(
        "--metrics-output-path",
        default=None,
        help="Per-iteration JSONL path (default: next to RL weights).",
    )
    initialization = group.add_mutually_exclusive_group()
    initialization.add_argument(
        "--fresh-from-sl",
        dest="fresh_from_sl",
        action="store_true",
        default=fresh_from_sl_default,
        help=(
            "Initialize the policy from --sl-weights-path even when the RL "
            "output already exists; replace that output only after success."
        ),
    )
    initialization.add_argument(
        "--continue-existing-rl",
        dest="fresh_from_sl",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Continue from --rl-weights-path when it exists.",
    )
    group.add_argument(
        "--numbered-checkpoints",
        action="store_true",
        help=(
            "Write iteration-suffixed weights and an atomic opponent-pool state "
            "for safe interruption recovery."
        ),
    )
    group.add_argument(
        "--start-iteration",
        type=int,
        default=0,
        help="Absolute completed iteration when continuing a numbered checkpoint.",
    )
    group.add_argument(
        "--resume-weights-path",
        default=None,
        help="Iteration-suffixed weights file from a complete resume pair.",
    )
    group.add_argument(
        "--resume-state-file",
        default=None,
        help="Auxiliary .resume.npz file paired with --resume-weights-path.",
    )
    group.add_argument(
        "--value-head",
        action="store_true",
        help=(
            "Train a linear V(s) baseline (the critic) and use reward-minus-value "
            "policy advantages in the legacy path; combine with --no-ppo."
        ),
    )
    group.add_argument("--value-coef", type=float, default=VALUE_COEF)
    group.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help="Terminal-reward discount per remaining real decision (1.0 = no discount).",
    )
    group.add_argument(
        "--reward-schema",
        choices=tuple(REWARD_SCHEMAS),
        default=DEFAULT_REWARD_SCHEMA,
        help="Named preset for the terminal/event reward constants.",
    )
    group.add_argument(
        "--clip-grad-norm",
        type=float,
        default=DEFAULT_CLIP_GRAD_NORM,
        help="Gradient-norm clipping threshold for the policy-gradient update.",
    )
    group.add_argument(
        "--normalize-advantages",
        dest="normalize_advantages",
        action="store_true",
        default=DEFAULT_NORMALIZE_ADVANTAGES,
        help="Normalize advantages once over the complete iteration buffer.",
    )
    group.add_argument(
        "--no-normalize-advantages",
        dest="normalize_advantages",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Disable whole-buffer advantage normalization.",
    )
    group.add_argument(
        "--moving-average-window",
        type=int,
        default=DEFAULT_MOVING_AVERAGE_WINDOW,
        help="Trailing-iteration window for the value-loss/win-rate moving averages "
        "in the log (point values are noisy; use this for judging a plateau).",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fix random/numpy state for reproducible comparisons between configurations.",
    )
    group.add_argument(
        "--device",
        choices=DEVICES,
        default=DEFAULT_DEVICE,
        help="Array backend: 'auto' matches GPU_ENABLED (CuPy when installed, "
        "else NumPy) -- unchanged from prior behavior. 'cpu'/'gpu' force one "
        "backend regardless of what's installed/enabled globally.",
    )
    group.add_argument(
        "--rl-workers",
        type=parse_rl_worker_count,
        default=DEFAULT_RL_WORKERS,
        help=(
            f"CPU-only rollout workers or 'auto' for isolated discarded tuning "
            f"(maximum {MAX_PARALLEL_WORKERS})."
        ),
    )
    group.add_argument(
        "--rl-autotune-fraction",
        type=float,
        default=DEFAULT_RL_AUTOTUNE_FRACTION,
        help="Fraction of the real game budget discarded by each worker candidate.",
    )
    group.add_argument(
        "--rl-autotune-min-gain",
        type=float,
        default=DEFAULT_RL_MINIMUM_GAIN,
        help=(
            "Stop worker tuning when marginal rollout-throughput gain over the "
            "previous accepted candidate is below this value."
        ),
    )
    group.add_argument("--rl-memory-reserve-mb", type=int, default=512)
    group.add_argument("--rl-estimated-worker-mb", type=int, default=256)
    group.add_argument("--rl-max-worker-rss-mb", type=int, default=1024)
    ppo = parser.add_argument_group("PPO v1 controls")
    ppo_toggle = ppo.add_mutually_exclusive_group()
    ppo_toggle.add_argument(
        "--ppo",
        dest="ppo_enabled",
        action="store_true",
        default=DEFAULT_PPO_ENABLED,
        help="Use masked PPO with minibatches (default).",
    )
    ppo_toggle.add_argument(
        "--no-ppo",
        dest="ppo_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Use the historical one-update REINFORCE path for regression.",
    )
    ppo.add_argument("--ppo-clip-epsilon", type=float, default=DEFAULT_CLIP_EPSILON)
    ppo.add_argument("--ppo-target-kl", type=float, default=DEFAULT_TARGET_KL)
    ppo.add_argument("--ppo-stop-kl", type=float, default=DEFAULT_STOP_KL)
    ppo.add_argument("--ppo-max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    ppo.add_argument(
        "--ppo-min-minibatches",
        type=int,
        default=DEFAULT_MIN_MINIBATCHES,
    )
    ppo.add_argument(
        "--ppo-max-minibatches",
        type=int,
        default=DEFAULT_MAX_MINIBATCHES,
    )
    ppo.add_argument(
        "--ppo-games-per-minibatch-scale",
        type=int,
        default=DEFAULT_GAMES_PER_MINIBATCH_SCALE,
    )
    ppo.add_argument(
        "--ppo-min-decisions-per-minibatch",
        type=int,
        default=DEFAULT_MIN_DECISIONS_PER_MINIBATCH,
    )
    buffer_group = ppo.add_mutually_exclusive_group()
    buffer_group.add_argument(
        "--prefer-gpu-buffer",
        dest="prefer_gpu_buffer",
        action="store_true",
        default=True,
    )
    buffer_group.add_argument(
        "--no-prefer-gpu-buffer",
        dest="prefer_gpu_buffer",
        action="store_false",
        default=argparse.SUPPRESS,
    )
    ppo.add_argument(
        "--gpu-buffer-safety-fraction",
        type=float,
        default=DEFAULT_GPU_BUFFER_SAFETY_FRACTION,
    )
    return parser


def parse_args(argv=None):
    """Parse optional self-play training controls."""
    parser = argparse.ArgumentParser(
        description="Train the domino policy with reinforcement learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_optional_rl_arguments(parser)
    parser.add_argument(
        "--compact",
        action="store_true",
        help=(
            "Show isolated adaptive tuning, one game progress bar, and one "
            "final summary instead of per-iteration/checkpoint logs."
        ),
    )
    return parser.parse_args(argv)


def _training_kwargs_from_args(args):
    """Translate CLI arguments into the public ``train`` keyword interface."""
    manual_gpi_supplied = args.games_per_iteration is not None
    games_per_iteration = (
        DEFAULT_GAMES_PER_ITERATION
        if args.games_per_iteration is None
        else args.games_per_iteration
    )
    adaptive_gpi = (
        (not manual_gpi_supplied and args.iterations is None)
        if args.adaptive_gpi is None
        else bool(args.adaptive_gpi)
    )
    return {
        "iterations": args.iterations,
        "total_training_games": (
            args.total_training_games
            if args.iterations is not None
            else (
                DEFAULT_TOTAL_TRAINING_GAMES
                if args.total_training_games is None
                else args.total_training_games
            )
        ),
        "games_per_iteration": games_per_iteration,
        "adaptive_gpi": adaptive_gpi,
        "gpi_candidates": tuple(args.gpi_candidates),
        "gpi_benchmark_games_target": args.gpi_benchmark_games_target,
        "retune_gpi": args.retune_gpi,
        "retune_workers": args.retune_workers,
        "retune_all": args.retune_all,
        "training_opponent": args.training_opponent,
        "learning_rate": args.learning_rate,
        "entropy_coef": args.entropy_coef,
        "log_interval": args.log_interval,
        "checkpoint_interval": args.checkpoint_interval,
        "pool_refresh_games": args.pool_refresh_games,
        "max_pool_size": args.max_pool_size,
        "sl_weights_path": args.sl_weights_path,
        "rl_weights_path": args.rl_weights_path,
        "adaptive_tuning_path": args.adaptive_tuning_path,
        "metrics_output_path": args.metrics_output_path,
        "fresh_from_sl": args.fresh_from_sl,
        "use_value_head": args.value_head,
        "value_coef": args.value_coef,
        "gamma": args.gamma,
        "reward_schema": args.reward_schema,
        "clip_grad_norm": args.clip_grad_norm,
        "normalize_advantages": args.normalize_advantages,
        "moving_average_window": args.moving_average_window,
        "seed": args.seed,
        "device": args.device,
        "workers": args.rl_workers,
        "safety_config": ParallelSafetyConfig(
            memory_reserve_mb=args.rl_memory_reserve_mb,
            estimated_worker_mb=args.rl_estimated_worker_mb,
            max_worker_rss_mb=args.rl_max_worker_rss_mb,
        ),
        "autotune_fraction": args.rl_autotune_fraction,
        "autotune_minimum_gain": args.rl_autotune_min_gain,
        "start_iteration": args.start_iteration,
        "resume_weights_path": args.resume_weights_path,
        "resume_state_file": args.resume_state_file,
        "numbered_checkpoints": args.numbered_checkpoints,
        "ppo_enabled": args.ppo_enabled,
        "ppo_clip_epsilon": args.ppo_clip_epsilon,
        "ppo_target_kl": args.ppo_target_kl,
        "ppo_stop_kl": args.ppo_stop_kl,
        "ppo_max_epochs": args.ppo_max_epochs,
        "ppo_min_minibatches": args.ppo_min_minibatches,
        "ppo_max_minibatches": args.ppo_max_minibatches,
        "ppo_games_per_minibatch_scale": args.ppo_games_per_minibatch_scale,
        "ppo_min_decisions_per_minibatch": args.ppo_min_decisions_per_minibatch,
        "prefer_gpu_buffer": args.prefer_gpu_buffer,
        "gpu_buffer_safety_fraction": args.gpu_buffer_safety_fraction,
    }


def _run_compact_cli(args, training_kwargs):
    """Run the standalone CLI with the pipeline's compact presentation."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    print("\nRL self-play")
    started = time.time()
    manual_gpi = training_kwargs["games_per_iteration"]
    planned_games = (
        args.iterations * manual_gpi
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
        f"{summary['completed_iterations_this_run']} iteration(s), selected GPI "
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
