"""Defaults and side-effect-free option resolution for RL self-play."""

from dataclasses import dataclass

from diagnostics.parallel_runner import MAX_PARALLEL_WORKERS, ParallelSafetyConfig
from training.ppo import MAX_PPO_EPOCHS
from training.rl_resume import LEGACY_TRAINING_ALGORITHM, PPO_TRAINING_ALGORITHM
from training.rl_rollout import REWARD_SCHEMAS


# The array backend is resolved once inside train() and always matches the
# PolicyNetwork backend selected for that run.
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


@dataclass(frozen=True)
class ResolvedTrainingOptions:
    """Validated values needed before model loading or resume side effects."""

    retune_gpi: bool
    retune_workers: bool
    games_per_iteration: int
    total_training_games: int
    tuning_training_games: int
    adaptive_gpi: bool
    algorithm: str
    normalize_advantages: bool
    workers: int | str
    safety_config: ParallelSafetyConfig
    schema: dict


def resolve_training_options(
    *,
    iterations,
    total_training_games,
    games_per_iteration,
    adaptive_gpi,
    adaptive_tuning_training_games,
    retune_gpi,
    retune_workers,
    retune_all,
    checkpoint_interval,
    log_interval,
    pool_refresh_games,
    max_pool_size,
    moving_average_window,
    autotune_fraction,
    autotune_minimum_gain,
    training_opponent,
    reward_schema,
    ppo_enabled,
    use_value_head,
    ppo_clip_epsilon,
    ppo_target_kl,
    ppo_stop_kl,
    ppo_max_epochs,
    ppo_min_minibatches,
    ppo_max_minibatches,
    ppo_games_per_minibatch_scale,
    ppo_min_decisions_per_minibatch,
    gpu_buffer_safety_fraction,
    normalize_advantages,
    workers,
    safety_config,
):
    """Normalize and validate options that do not require checkpoint I/O."""
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
        if (
            total_training_games is not None
            and int(total_training_games) != implied_total
        ):
            raise ValueError(
                "iterations * games_per_iteration conflicts with "
                "total_training_games"
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
            "PPO v1 keeps the critic disabled; use --no-ppo for "
            "value-head regression."
        )
    if ppo_enabled:
        if not 0 < float(ppo_clip_epsilon) < 1:
            raise ValueError("ppo_clip_epsilon must be in (0, 1)")
        if not 0 < float(ppo_target_kl) <= float(ppo_stop_kl):
            raise ValueError(
                "PPO KL thresholds require 0 < target_kl <= stop_kl"
            )
        if not 1 <= int(ppo_max_epochs) <= MAX_PPO_EPOCHS:
            raise ValueError(
                f"ppo_max_epochs must be between 1 and {MAX_PPO_EPOCHS}"
            )
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
    algorithm = (
        PPO_TRAINING_ALGORITHM if ppo_enabled else LEGACY_TRAINING_ALGORITHM
    )
    normalize_advantages = (
        bool(ppo_enabled)
        if normalize_advantages is None
        else bool(normalize_advantages)
    )
    if workers != "auto":
        workers = int(workers)
        if not 1 <= workers <= MAX_PARALLEL_WORKERS:
            raise ValueError(
                f"workers must be 'auto' or between 1 and "
                f"{MAX_PARALLEL_WORKERS}"
            )
    safety_config = safety_config or ParallelSafetyConfig()
    return ResolvedTrainingOptions(
        retune_gpi=retune_gpi,
        retune_workers=retune_workers,
        games_per_iteration=games_per_iteration,
        total_training_games=total_training_games,
        tuning_training_games=tuning_training_games,
        adaptive_gpi=adaptive_gpi,
        algorithm=algorithm,
        normalize_advantages=normalize_advantages,
        workers=workers,
        safety_config=safety_config,
        schema=REWARD_SCHEMAS[reward_schema],
    )
