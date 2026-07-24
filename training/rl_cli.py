"""Argument parsing and CLI-to-training option translation for RL."""

import argparse

from agents.rl_nn import DEVICES
from diagnostics.parallel_runner import MAX_PARALLEL_WORKERS, ParallelSafetyConfig
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
    MAX_PPO_EPOCHS,
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
    COMMON_GPI_VALUES,
    RL_WEIGHTS,
    SL_WEIGHTS,
    TRAINING_OPPONENT,
    VALUE_COEF,
)
from training.rl_parallel import (
    DEFAULT_RL_AUTOTUNE_FRACTION,
    DEFAULT_RL_MINIMUM_GAIN,
    DEFAULT_RL_WORKERS,
    worker_count as parse_rl_worker_count,
)
from training.rl_rollout import DEFAULT_GAMMA, DEFAULT_REWARD_SCHEMA, REWARD_SCHEMAS


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def add_optional_rl_arguments(
    parser,
    *,
    fresh_from_sl_default=False,
    ppo_max_epochs_default=DEFAULT_MAX_EPOCHS,
    expose_gpi=True,
):
    """Add self-play hyperparameter and rollout-resource flags to ``parser``."""
    group = parser.add_argument_group("optional reinforcement-learning controls")
    group.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=(
            "Legacy/manual iteration budget. When supplied, total games are "
            "iterations x GPI."
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
    if expose_gpi:
        group.add_argument(
            "--gpi",
            type=_positive_int,
            default=DEFAULT_GPI,
            help=(
                "Fixed games per RL iteration. "
                f"Common values: "
                f"{', '.join(str(value) for value in COMMON_GPI_VALUES)} "
                f"(default: {DEFAULT_GPI})."
            ),
        )
    else:
        parser.set_defaults(gpi=DEFAULT_GPI)
    group.add_argument("--retune-workers", action="store_true")
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
    ppo = parser.add_argument_group("RL update algorithm and PPO v1 controls")
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
        help=(
            "Use one full-buffer REINFORCE update per iteration, without PPO "
            "minibatches, ratios, clipping, KL control, or post-update "
            "full-buffer evaluation."
        ),
    )
    ppo.add_argument("--ppo-clip-epsilon", type=float, default=DEFAULT_CLIP_EPSILON)
    ppo.add_argument("--ppo-target-kl", type=float, default=DEFAULT_TARGET_KL)
    ppo.add_argument("--ppo-stop-kl", type=float, default=DEFAULT_STOP_KL)
    ppo.add_argument(
        "--ppo-max-epochs",
        type=int,
        default=ppo_max_epochs_default,
        help=(
            "Maximum PPO epochs over one on-policy buffer; whole-buffer KL is "
            "checked after every epoch. Standalone and finite canonical "
            f"profiles default to {DEFAULT_MAX_EPOCHS}; canonical forever "
            f"defaults to {MAX_PPO_EPOCHS}."
        ),
    )
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
        "gpi": args.gpi,
        "retune_workers": args.retune_workers,
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
