"""Argument parsing and CLI-to-training option translation for RL."""

import argparse
from dataclasses import replace
import time

from agents.rl_nn import DEVICES
from diagnostics.parallel_runner import MAX_PARALLEL_WORKERS, ParallelSafetyConfig
from training.rl.ppo import (
    DEFAULT_PPO_MAX_EPOCHS,
    MAX_PPO_EPOCHS,
    PPO_DISABLED_EPOCHS,
)
from training.rl.config import (
    DEFAULT_DEVICE,
    DEFAULT_DIFFICULTY_WEIGHT,
    DEFAULT_GPI,
    DEFAULT_MOVING_AVERAGE_WINDOW,
    DEFAULT_NORMALIZE_ADVANTAGES,
    DEFAULT_TOTAL_TRAINING_GAMES,
    COMMON_GPI_VALUES,
    RL_WEIGHTS,
    RLExecutionOptions,
    RLResourceOptions,
    RLTrainingOptions,
    SL_WEIGHTS,
    VALUE_COEF,
)
from training.rl.pool import DEFAULT_OPPONENT_BUCKETS, canonicalize_bucket_names
from training.rl.parallel import (
    DEFAULT_RL_WORKERS,
    worker_count as parse_rl_worker_count,
)
from training.rl.rollout import ALPHA, DEFAULT_GAMMA, EVENT_REWARD_DECAY
from training.rl.resume import load_resume_state
from training.utils.cli_args import add_regularization_arguments, positive_int
from utils.runtime_status import format_duration
from middleware.rulesets import DEFAULT_RULESET_NAME, RULESET_NAMES
from utils.ruleset_paths import (
    default_rl_weights_path,
    default_sl_weights_path,
)


def parse_opponent_buckets(value):
    """Parse one comma-separated bucket selection into canonical registry order."""
    try:
        return canonicalize_bucket_names(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def unit_interval_parser(name):
    """Build a parser for one closed [0, 1] hyperparameter named ``name``."""

    def parse(value):
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                f"{name} must be a number between 0 and 1"
            ) from exc
        if not 0.0 <= parsed <= 1.0:
            raise argparse.ArgumentTypeError(f"{name} must be between 0 and 1")
        return parsed

    return parse


parse_difficulty_weight = unit_interval_parser("difficulty weight")
parse_gamma = unit_interval_parser("gamma")
parse_alpha = unit_interval_parser("alpha")
parse_event_reward_decay = unit_interval_parser("event reward decay")


def add_optional_rl_arguments(
    parser,
    *,
    fresh_from_sl_default=False,
    ppo_max_epochs_default=DEFAULT_PPO_MAX_EPOCHS,
    expose_gpi=True,
    include_regularization=True,
    expose_resume_pair=True,
):
    """Add RL hyperparameter and rollout-resource flags to ``parser``.

    ``include_regularization=False`` is for combined parsers that already
    installed the shared ``--weight-decay``/``--dropout`` controls through
    :func:`training.supervised.cli.add_optional_training_arguments`.
    """
    group = parser.add_argument_group("optional reinforcement-learning controls")
    group.add_argument(
        "--ruleset",
        choices=RULESET_NAMES,
        default=DEFAULT_RULESET_NAME,
        help="Named compact domino ruleset.",
    )
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
            type=positive_int,
            choices=COMMON_GPI_VALUES,
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
        "--opponent-buckets",
        type=parse_opponent_buckets,
        default=DEFAULT_OPPONENT_BUCKETS,
        metavar="NAMES",
        help=(
            "Comma-separated opponent buckets: heuristic, random, recent, and "
            "medium_term. Input order is canonicalized."
        ),
    )
    group.add_argument(
        "--difficulty-weight",
        type=parse_difficulty_weight,
        default=DEFAULT_DIFFICULTY_WEIGHT,
        help=(
            "Convex matchmaking mixture: 0 is uniform across buckets/members, "
            "0.5 is half uniform and half difficulty-based, and 1 is entirely "
            "difficulty-based."
        ),
    )
    group.add_argument(
        "--opponent-decision-restarts",
        action="store_true",
        help=(
            "Augment each RL iteration with same-iteration continuations from "
            "every genuine opponent decision state encountered during normal "
            "GPI games. Restart samples join normal samples before one update."
        ),
    )
    group.add_argument("--learning-rate", type=float, default=0.001)
    group.add_argument("--entropy-coef", type=float, default=0.01)
    if include_regularization:
        add_regularization_arguments(group)
    group.add_argument("--log-interval", type=int, default=10)
    group.add_argument("--checkpoint-interval", type=int, default=50)
    group.add_argument("--sl-weights-path", default=None)
    group.add_argument("--rl-weights-path", default=None)
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
    if expose_resume_pair:
        group.add_argument(
            "--numbered-checkpoints",
            action="store_true",
            help=(
                "Write iteration-suffixed weights and an atomic opponent-pool "
                "state for safe interruption recovery."
            ),
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
    else:
        parser.set_defaults(
            numbered_checkpoints=False,
            resume_weights_path=None,
            resume_state_file=None,
        )
    group.add_argument(
        "--value-head",
        action="store_true",
        help=(
            "Train a linear V(s) critic and use reward-minus-value policy "
            "advantages with PPO or the single-update REINFORCE path."
        ),
    )
    group.add_argument("--value-coef", type=float, default=VALUE_COEF)
    group.add_argument(
        "--gamma",
        type=parse_gamma,
        default=DEFAULT_GAMMA,
        help="Terminal-reward discount per remaining real decision (1.0 = no discount).",
    )
    group.add_argument(
        "--alpha",
        type=parse_alpha,
        default=ALPHA,
        help=(
            "Convex mix of the two reward components per decision: "
            "R = (1 - alpha) * gamma**k * terminal + alpha * local. "
            "0 trains on the terminal outcome alone, 1 on local event "
            "shaping alone."
        ),
    )
    group.add_argument(
        "--event-reward-decay",
        type=parse_event_reward_decay,
        default=EVENT_REWARD_DECAY,
        help=(
            "Per-turn decay applied to a draw/pass event reward as it is "
            "credited backwards to the real decisions preceding the event "
            "(0 credits only the immediately preceding decision)."
        ),
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
    group.add_argument("--rl-memory-reserve-mb", type=int, default=512)
    group.add_argument("--rl-estimated-worker-mb", type=int, default=256)
    group.add_argument("--rl-max-worker-rss-mb", type=int, default=1024)
    ppo = parser.add_argument_group("RL update algorithm")
    ppo.add_argument(
        "--ppo-max-epochs",
        type=int,
        choices=range(PPO_DISABLED_EPOCHS, MAX_PPO_EPOCHS + 1),
        default=ppo_max_epochs_default,
        help=(
            "Maximum epochs over one on-policy buffer. Value 1 selects the "
            "single-update REINFORCE path with no PPO control overhead; values "
            "2-16 select PPO with fixed clipping, minibatch, KL, and buffer "
            f"policies. Standalone defaults to {DEFAULT_PPO_MAX_EPOCHS}; "
            f"canonical forever defaults to {MAX_PPO_EPOCHS}."
        ),
    )
    return parser


def parse_args(argv=None):
    """Parse optional reinforcement-learning controls."""
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


def training_options_from_args(args):
    """Translate CLI arguments into the three public RL option groups."""
    training = RLTrainingOptions(
        iterations=args.iterations,
        ruleset_name=args.ruleset,
        total_training_games=(
            args.total_training_games
            if args.iterations is not None
            else (
                DEFAULT_TOTAL_TRAINING_GAMES
                if args.total_training_games is None
                else args.total_training_games
            )
        ),
        gpi=args.gpi,
        opponent_buckets=args.opponent_buckets,
        difficulty_weight=args.difficulty_weight,
        opponent_decision_restarts=args.opponent_decision_restarts,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        use_value_head=args.value_head,
        value_coef=args.value_coef,
        gamma=args.gamma,
        alpha=args.alpha,
        event_reward_decay=args.event_reward_decay,
        normalize_advantages=args.normalize_advantages,
        seed=args.seed,
        ppo_max_epochs=args.ppo_max_epochs,
    )
    resources = RLResourceOptions(
        sl_weights_path=(
            args.sl_weights_path or default_sl_weights_path(args.ruleset)
        ),
        rl_weights_path=(
            args.rl_weights_path or default_rl_weights_path(args.ruleset)
        ),
        device=args.device,
        workers=args.rl_workers,
        safety_config=ParallelSafetyConfig(
            memory_reserve_mb=args.rl_memory_reserve_mb,
            estimated_worker_mb=args.rl_estimated_worker_mb,
            max_worker_rss_mb=args.rl_max_worker_rss_mb,
        ),
        retune_workers=args.retune_workers,
    )
    execution = RLExecutionOptions(
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        moving_average_window=args.moving_average_window,
        resume_weights_path=args.resume_weights_path,
        resume_state_file=args.resume_state_file,
        numbered_checkpoints=args.numbered_checkpoints,
        fresh_from_sl=args.fresh_from_sl,
    )
    return training, resources, execution


def _run_compact_cli(args, options):
    """Run the standalone CLI with the pipeline's compact presentation."""
    from training.rl.training_loop import train

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    print("\nRL training")
    started = time.time()
    training, resources, execution = options
    fixed_gpi = training.gpi
    planned_games = (
        args.iterations * fixed_gpi
        if args.iterations is not None
        else training.total_training_games
    )
    initial_games = 0
    if execution.resume_weights_path and execution.resume_state_file:
        resume_metadata, _pool = load_resume_state(
            execution.resume_weights_path,
            execution.resume_state_file,
        )
        planned_games = int(
            resume_metadata["configuration"]["total_training_games"]
        )
        initial_games = int(resume_metadata["completed_training_games"])

    if tqdm is None:
        progress_interval = max(1, planned_games // 10)
        last_reported = initial_games

        def progress(done, total):
            nonlocal last_reported
            if done == total or done - last_reported >= progress_interval:
                print(f"RL progress: {done}/{total} games", flush=True)
                last_reported = done

        summary = train(
            training,
            resources,
            replace(
                execution,
                quiet=True,
                progress_callback=progress,
                status_callback=lambda message: print(message, flush=True),
            ),
        )
    else:
        with tqdm(
            total=planned_games,
            initial=initial_games,
            desc="RL training",
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
                training,
                resources,
                replace(
                    execution,
                    quiet=True,
                    progress_callback=progress,
                    status_callback=tqdm.write,
                ),
            )

    elapsed = time.time() - started
    print(
        f"RL training complete in {format_duration(elapsed)} | "
        f"{summary['completed_training_games']} exact training games, "
        f"GPI {summary['games_per_iteration']}, "
        f"{summary['selected_workers']} rollout worker(s), "
        f"algorithm {summary['rl_training_algorithm']}, "
        f"weights {summary['rl_weights_path']}"
    )
    return summary


def main(argv=None):
    """Run the standalone RL command."""
    from training.rl.training_loop import train

    args = parse_args(argv)
    options = training_options_from_args(args)
    if args.compact:
        return _run_compact_cli(args, options)
    return train(*options)


if __name__ == "__main__":
    main()
