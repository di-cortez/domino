#!/usr/bin/env python3
"""Run the full domino data, training, self-play, and diagnostics pipeline."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.runtime_status import format_duration, pipeline_compute_report

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

BASE_DATASET_GAMES = 100000
BASE_SUPERVISED_EPOCHS = 5000
BASE_RL_ITERATIONS = 1000
BASE_RL_GAMES_PER_ITERATION = 100
BASE_DIAGNOSTIC_GAMES = 10000

SCALE_FACTORS = {
    "default": 1.0,
    "small": 0.2,
    "big": 5.0,
    "huge": 20.0,
}

@dataclass(frozen=True)
class PipelineConfig:
    """Concrete workload sizes for one pipeline run."""

    scale_name: str
    scale_factor: float
    dataset_games: int
    supervised_epochs: int
    rl_iterations: int
    rl_games_per_iteration: int
    diagnostic_games: int


def _scaled_count(base_count, scale_factor):
    """Scale an integer workload while keeping every stage runnable."""
    return max(1, int(round(base_count * scale_factor)))


def _build_config(scale_name):
    """Return workload sizes for the requested scale."""
    scale_factor = SCALE_FACTORS[scale_name]
    return PipelineConfig(
        scale_name=scale_name,
        scale_factor=scale_factor,
        dataset_games=_scaled_count(BASE_DATASET_GAMES, scale_factor),
        supervised_epochs=_scaled_count(BASE_SUPERVISED_EPOCHS, scale_factor),
        rl_iterations=_scaled_count(BASE_RL_ITERATIONS, scale_factor),
        rl_games_per_iteration=BASE_RL_GAMES_PER_ITERATION,
        diagnostic_games=_scaled_count(BASE_DIAGNOSTIC_GAMES, scale_factor),
    )


def _silent_import(module_name):
    """Import a module while hiding import-time status prints from compact runs."""
    with contextlib.redirect_stdout(io.StringIO()):
        return importlib.import_module(module_name)


@contextlib.contextmanager
def _progress(label, total, unit):
    """Yield an absolute-progress callback compatible with the training modules."""
    if tqdm is None:
        print(f"{label}: {total} {unit}")

        def callback(_done, _total):
            return None

        yield callback
        return

    with tqdm(total=total, desc=label, unit=unit, leave=True) as bar:

        def callback(done, reported_total):
            if reported_total != bar.total:
                bar.total = reported_total
                bar.refresh()
            if done > bar.n:
                bar.update(done - bar.n)

        yield callback


def _run_stage(label, total, unit, runner, summary_text):
    """Run one pipeline stage with a progress bar and one compact summary line."""
    print(f"\n{label}")
    start_time = time.time()
    with _progress(label, total, unit) as progress_callback:
        summary = runner(progress_callback)
    elapsed_time = time.time() - start_time
    print(
        f"{label} complete in {format_duration(elapsed_time)}"
        f" | {summary_text(summary)}"
    )
    return summary


def _diagnostic_summary_text(summary):
    """Format independently selected matchup worker counts for the pipeline."""
    worker_text = ", ".join(
        f"{matchup}={worker_count}"
        for matchup, worker_count in summary[
            "selected_workers_by_matchup"
        ].items()
    )
    return (
        f"{summary['evaluated_matchups']} matchups x "
        f"{summary['game_count_per_matchup']} games = "
        f"{summary['evaluated_matchups'] * summary['game_count_per_matchup']} "
        f"total games | workers: {worker_text}"
    )


def _run_dataset(config, args):
    """Generate the supervised JSONL dataset."""
    dataset_generator = importlib.import_module("training.dataset_generator")

    def dataset_status(message):
        if tqdm is not None:
            tqdm.write(message)
        else:
            print(message, flush=True)

    return _run_stage(
        "Dataset generation",
        config.dataset_games,
        "game",
        lambda progress: dataset_generator.generate_dataset(
            game_count=config.dataset_games,
            output_file="dataset/supervised_dataset.jsonl",
            quiet=True,
            progress_callback=progress,
            workers=args.dataset_workers,
            safety_config=dataset_generator.ParallelSafetyConfig(
                memory_reserve_mb=args.dataset_memory_reserve_mb,
                estimated_worker_mb=args.dataset_estimated_worker_mb,
                max_worker_rss_mb=args.dataset_max_worker_rss_mb,
            ),
            autotune_fraction=args.dataset_autotune_fraction,
            autotune_minimum_gain=args.dataset_autotune_min_gain,
            seed=args.dataset_seed,
            status_callback=dataset_status,
        ),
        lambda summary: (
            f"{summary['saved_turn_count']} real decisions, "
            f"{summary['skipped_turn_count']} forced turns skipped, "
            f"{summary['selected_workers']} worker(s)"
        ),
    )


def _run_supervised_training(config, args):
    """Train the supervised policy with compact epoch progress."""
    training_loop = _silent_import("training.training_loop")

    def supervised_status(message):
        if tqdm is not None:
            tqdm.write(message)
        else:
            print(message, flush=True)

    return _run_stage(
        "Supervised training",
        config.supervised_epochs,
        "epoch",
        lambda progress: training_loop.train_supervised(
            epochs=config.supervised_epochs,
            batch_size=args.sl_batch_size,
            quiet=True,
            progress_callback=progress,
            status_callback=supervised_status,
            weight_decay=args.weight_decay,
            early_stopping_patience=args.early_stopping,
            lr_decay_factor=args.lr_decay,
            lr_decay_patience=args.lr_decay_patience,
            training_plateau_enabled=not args.disable_training_plateau,
            training_plateau_window=args.sl_training_plateau_window,
            training_plateau_patience=args.sl_training_plateau_patience,
            training_plateau_min_epochs=args.sl_training_plateau_min_epochs,
            training_plateau_min_relative_improvement=(
                args.sl_training_plateau_min_relative_improvement
            ),
            device=args.sl_device,
            autotune_batch_size=not args.sl_no_batch_autotune,
            memory_reserve_mb=args.sl_memory_reserve_mb,
            gpu_memory_reserve_mb=args.sl_gpu_memory_reserve_mb,
            seed=args.sl_seed,
        ),
        lambda summary: (
            f"{summary['epochs']}/{summary['requested_epochs']} epochs, "
            f"best validation loss {summary['best_validation_loss']:.4f}, "
            f"{summary['total_examples']} examples, "
            f"stop={summary.get('stopping_reason', 'epoch_limit').replace('_', ' ')}"
        ),
    )


def _run_rl_training(config, args):
    """Run reinforcement-learning self-play with compact iteration progress."""
    self_play = importlib.import_module("training.self_play")

    def rl_status(message):
        if tqdm is not None:
            tqdm.write(message)
        else:
            print(message, flush=True)

    manual_gpi = (
        config.rl_games_per_iteration
        if args.games_per_iteration is None
        else args.games_per_iteration
    )
    explicit_iterations = args.iterations
    total_training_games = (
        explicit_iterations * manual_gpi
        if explicit_iterations is not None
        else (
            args.total_training_games
            if args.total_training_games is not None
            else config.rl_iterations * config.rl_games_per_iteration
        )
    )
    adaptive_gpi = (
        (args.games_per_iteration is None and explicit_iterations is None)
        if args.adaptive_gpi is None
        else bool(args.adaptive_gpi)
    )

    return _run_stage(
        "RL self-play",
        config.rl_iterations,
        "iter",
        lambda progress: self_play.train(
            iterations=explicit_iterations,
            total_training_games=(
                args.total_training_games
                if explicit_iterations is not None
                else total_training_games
            ),
            games_per_iteration=manual_gpi,
            adaptive_gpi=adaptive_gpi,
            gpi_candidates=tuple(args.gpi_candidates),
            gpi_benchmark_games_target=args.gpi_benchmark_games_target,
            retune_gpi=args.retune_gpi,
            retune_workers=args.retune_workers,
            retune_all=args.retune_all,
            training_opponent=args.training_opponent,
            learning_rate=args.learning_rate,
            entropy_coef=args.entropy_coef,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
            pool_refresh_games=args.pool_refresh_games,
            max_pool_size=args.max_pool_size,
            sl_weights_path=args.sl_weights_path,
            rl_weights_path=args.rl_weights_path,
            adaptive_tuning_path=args.adaptive_tuning_path,
            metrics_output_path=args.metrics_output_path,
            fresh_from_sl=args.fresh_from_sl,
            quiet=True,
            progress_callback=progress,
            use_value_head=args.value_head,
            value_coef=args.value_coef,
            gamma=args.gamma,
            reward_schema=args.reward_schema,
            clip_grad_norm=args.clip_grad_norm,
            normalize_advantages=args.normalize_advantages,
            moving_average_window=args.moving_average_window,
            seed=args.seed,
            device=args.device,
            workers=args.rl_workers,
            safety_config=self_play.ParallelSafetyConfig(
                memory_reserve_mb=args.rl_memory_reserve_mb,
                estimated_worker_mb=args.rl_estimated_worker_mb,
                max_worker_rss_mb=args.rl_max_worker_rss_mb,
            ),
            autotune_fraction=args.rl_autotune_fraction,
            autotune_minimum_gain=args.rl_autotune_min_gain,
            ppo_enabled=args.ppo_enabled,
            ppo_clip_epsilon=args.ppo_clip_epsilon,
            ppo_target_kl=args.ppo_target_kl,
            ppo_stop_kl=args.ppo_stop_kl,
            ppo_max_epochs=args.ppo_max_epochs,
            ppo_min_minibatches=args.ppo_min_minibatches,
            ppo_max_minibatches=args.ppo_max_minibatches,
            ppo_games_per_minibatch_scale=args.ppo_games_per_minibatch_scale,
            ppo_min_decisions_per_minibatch=(
                args.ppo_min_decisions_per_minibatch
            ),
            prefer_gpu_buffer=args.prefer_gpu_buffer,
            gpu_buffer_safety_fraction=args.gpu_buffer_safety_fraction,
            status_callback=rl_status,
        ),
        lambda summary: (
            f"{summary['total_training_games']} exact games in "
            f"{summary['completed_iterations_this_run']} iteration(s), "
            f"selected GPI {summary['games_per_iteration']}, "
            f"{summary['selected_workers']} rollout worker(s), "
            f"algorithm {summary['rl_training_algorithm']}, "
            f"weights {summary['rl_weights_path']}"
        ),
    )


def _diagnostic_workload(config):
    """Return the diagnostics module, matchup count, and aggregate game count."""
    evaluate = importlib.import_module("diagnostics.evaluate")
    _agents, matchups = evaluate.diagnostic_plan()
    matchup_count = len(matchups)
    return evaluate, matchup_count, matchup_count * config.diagnostic_games


def _run_diagnostics(config, args, rl_weights, neural_weights):
    """Run the four agent-vs-random diagnostics with one progress bar."""
    evaluate, matchup_count, total_games = _diagnostic_workload(config)

    def diagnostic_status(message):
        if tqdm is not None:
            tqdm.write(message)
        else:
            print(message, flush=True)

    return _run_stage(
        f"Diagnostics ({matchup_count} matchups)",
        total_games,
        "game",
        lambda progress: evaluate.run_all_pairs(
            game_count=config.diagnostic_games,
            output_dir=evaluate.DEFAULT_OUTPUT_DIR,
            quiet=True,
            progress_callback=progress,
            workers=args.diagnostic_workers,
            safety_config=evaluate.ParallelSafetyConfig(
                memory_reserve_mb=args.diagnostic_memory_reserve_mb,
                estimated_worker_mb=args.diagnostic_estimated_worker_mb,
                max_worker_rss_mb=args.diagnostic_max_worker_rss_mb,
            ),
            autotune_fraction=args.diagnostic_autotune_fraction,
            autotune_minimum_gain=args.diagnostic_autotune_min_gain,
            seed=args.diagnostic_seed,
            rl_weights=rl_weights,
            neural_weights=neural_weights,
            status_callback=diagnostic_status,
        ),
        _diagnostic_summary_text,
    )


def parse_args(argv=None):
    """Parse the optional workload scale."""
    parser = argparse.ArgumentParser(
        description=(
            "Run dataset generation, supervised training, RL self-play, and "
            "agent-vs-random diagnostics with compact progress output."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "scale",
        nargs="?",
        default="default",
        choices=tuple(SCALE_FACTORS),
        help=(
            "Workload scale. 'small' is 5x smaller, 'big' is 5x larger, "
            "and 'huge' is 20x larger than the defaults."
        ),
    )
    dataset_generator = importlib.import_module("training.dataset_generator")
    dataset = parser.add_argument_group("dataset multiprocessing controls")
    dataset.add_argument(
        "--dataset-workers",
        type=dataset_generator._worker_count,
        default=dataset_generator.DEFAULT_DATASET_WORKERS,
        help="CPU-only dataset workers or 'auto' for retained online tuning.",
    )
    dataset.add_argument(
        "--dataset-autotune-fraction",
        type=float,
        default=dataset_generator.DEFAULT_DATASET_AUTOTUNE_FRACTION,
        help="Fraction of dataset games retained by each worker-count test.",
    )
    dataset.add_argument(
        "--dataset-autotune-min-gain",
        type=float,
        default=dataset_generator.DEFAULT_DATASET_MINIMUM_GAIN,
        help="Minimum marginal throughput gain required for a larger pool.",
    )
    dataset.add_argument("--dataset-memory-reserve-mb", type=int, default=512)
    dataset.add_argument("--dataset-estimated-worker-mb", type=int, default=256)
    dataset.add_argument("--dataset-max-worker-rss-mb", type=int, default=1024)
    dataset.add_argument("--dataset-seed", type=int, default=None)

    training_loop = _silent_import("training.training_loop")
    training_loop.add_optional_training_arguments(parser)
    self_play = importlib.import_module("training.self_play")
    self_play.add_optional_rl_arguments(parser, fresh_from_sl_default=True)
    evaluate = _silent_import("diagnostics.evaluate")
    diagnostics = parser.add_argument_group("diagnostic multiprocessing controls")
    diagnostics.add_argument(
        "--diagnostic-workers",
        type=evaluate._worker_count,
        default=evaluate.DEFAULT_DIAGNOSTIC_WORKERS,
        help="CPU-only diagnostic workers or 'auto' for retained online tuning.",
    )
    diagnostics.add_argument(
        "--diagnostic-autotune-fraction",
        type=float,
        default=evaluate.DEFAULT_AUTOTUNE_FRACTION,
        help="Fraction of each matchup retained by each worker-count test.",
    )
    diagnostics.add_argument(
        "--diagnostic-autotune-min-gain",
        type=float,
        default=evaluate.DEFAULT_MINIMUM_GAIN,
        help="Minimum marginal throughput gain required to test a larger pool.",
    )
    diagnostics.add_argument("--diagnostic-memory-reserve-mb", type=int, default=512)
    diagnostics.add_argument("--diagnostic-estimated-worker-mb", type=int, default=256)
    diagnostics.add_argument("--diagnostic-max-worker-rss-mb", type=int, default=1024)
    diagnostics.add_argument("--diagnostic-seed", type=int, default=None)
    return parser.parse_args(argv)


def main():
    """Run every pipeline stage in sequence."""
    args = parse_args()
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    config = _build_config(args.scale)
    requested_rl_games = (
        args.iterations
        * (args.games_per_iteration or config.rl_games_per_iteration)
        if args.iterations is not None
        else (
            args.total_training_games
            if args.total_training_games is not None
            else config.rl_iterations * config.rl_games_per_iteration
        )
    )
    _evaluate, diagnostic_matchups, diagnostic_total_games = (
        _diagnostic_workload(config)
    )
    print(
        "Pipeline scale "
        f"{config.scale_name} ({config.scale_factor:g}x): "
        f"{config.dataset_games} dataset games, "
        f"up to {config.supervised_epochs} supervised epochs, "
        f"{requested_rl_games} exact RL games with "
        f"{'adaptive' if args.adaptive_gpi is not False and args.games_per_iteration is None and args.iterations is None else 'manual'} GPI, "
        f"diagnostics with {config.diagnostic_games} games per matchup "
        f"({diagnostic_matchups} matchups, {diagnostic_total_games} total games)."
    )
    print(pipeline_compute_report(args.device, args.sl_device))
    if args.fresh_from_sl:
        print(
            "RL initialization: fresh from the supervised checkpoint produced "
            "by this pipeline run; an existing RL output is ignored until the "
            "new model atomically replaces it."
        )
    else:
        print("RL initialization: continue from the existing RL checkpoint when present.")
    if args.sl_batch_size is not None:
        print(f"Supervised batch size: fixed at {args.sl_batch_size:,}.")
    elif args.sl_no_batch_autotune:
        print(
            "Supervised batch autotuning: disabled; using the selected "
            "device default (CPU 1,024 or GPU 2,048)."
        )
    else:
        print(
            "Supervised batches: automatic retained benchmark "
            "(10 complete epochs per candidate; starts at CPU 1,024 or GPU "
            "2,048, then doubles up to 1,048,576; stops below 10% marginal "
            "gain)."
        )
        print(
            "Every supervised batch test updates the live model and counts "
            "toward the requested epoch total."
        )
    if not args.disable_training_plateau:
        print(
            "Supervised training-loss plateau stop: enabled after batch "
            f"tuning and at least {args.sl_training_plateau_min_epochs} "
            f"epochs; {args.sl_training_plateau_patience} consecutive "
            f"{args.sl_training_plateau_window}-epoch median blocks below "
            f"{args.sl_training_plateau_min_relative_improvement:.3%} "
            "relative improvement."
        )
    else:
        print("Supervised training-loss plateau stop: disabled.")
    if args.dataset_workers == "auto":
        print(
            "Dataset workers: automatic retained benchmark "
            "(1, 2, 4, 6, ... up to 20; stops below 10% marginal gain)."
        )
    else:
        print(f"Dataset workers: fixed at {args.dataset_workers}.")
    if args.rl_workers == "auto":
        print(
            "RL rollout workers: isolated discarded throughput benchmark "
            "(1, 2, 4, 6, ... up to 20; 1% of the real budget per candidate)."
        )
        print(
            "Worker-tuning games use separate seeds and do not change weights, "
            "the pool, RNG state, or the real training-game counter."
        )
    else:
        print(f"RL rollout workers: fixed at {args.rl_workers}.")
    if args.diagnostic_workers == "auto":
        print(
            "Diagnostic workers: independent retained benchmark per matchup "
            "(1, 2, 4, 6, ... up to 20; stops below 10% marginal gain)."
        )
        print(
            "Diagnostic worker testing starts after RL so every benchmark game "
            "uses the newly trained checkpoint and remains in the final report."
        )
    else:
        print(f"Diagnostic workers: fixed at {args.diagnostic_workers}.")

    start_time = time.time()
    dataset_summary = _run_dataset(config, args)
    supervised_summary = _run_supervised_training(config, args)
    print(
        "RL startup note: rollout workers are CPU-only; policy aggregation and "
        "gradient updates remain in the main process."
    )
    rl_summary = _run_rl_training(config, args)
    diagnostics_summary = _run_diagnostics(
        config,
        args,
        rl_weights=rl_summary["rl_weights_path"],
        neural_weights=supervised_summary["weights_file"],
    )
    elapsed_time = time.time() - start_time

    print("\nPipeline complete")
    print(f"Total elapsed time: {format_duration(elapsed_time)}")
    print(f"Dataset: {dataset_summary['output_file']}")
    print(f"Supervised weights: {supervised_summary['weights_file']}")
    if supervised_summary.get("loss_plot_file"):
        print(f"Supervised loss graph: {supervised_summary['loss_plot_file']}")
    print(f"RL weights: {rl_summary['rl_weights_path']}")
    worker_text = ", ".join(
        f"{matchup}={worker_count}"
        for matchup, worker_count in diagnostics_summary[
            "selected_workers_by_matchup"
        ].items()
    )
    print(
        "Diagnostics: "
        f"{diagnostics_summary['evaluated_matchups']} matchups x "
        f"{diagnostics_summary['game_count_per_matchup']} games in "
        f"diagnostics/results/all_pairs/ | workers: {worker_text}"
    )


# Keep the historical stage helpers importable for focused tests and tooling,
# while making both public entry points execute the canonical game-budgeted
# pipeline. ``python -m train_script.run_pipeline`` and
# ``python -m training.pipeline`` are
# therefore equivalent.
from training import pipeline as _canonical_pipeline

PipelineConfig = _canonical_pipeline.PipelineConfig
SCALE_FACTORS = _canonical_pipeline.SCALE_FACTORS
_build_config = _canonical_pipeline._build_config
parse_args = _canonical_pipeline.parse_args
main = _canonical_pipeline.main


if __name__ == "__main__":
    main()
