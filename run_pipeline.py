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

from utils.runtime_status import format_duration

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

ROOT = Path(__file__).resolve().parent
BASE_DATASET_GAMES = 30000
BASE_SUPERVISED_EPOCHS = 1000
BASE_RL_ITERATIONS = 1000
BASE_RL_GAMES_PER_ITERATION = 40
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

        def callback(done, _total):
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


def _run_dataset(config):
    """Generate the supervised JSONL dataset."""
    dataset_generator = importlib.import_module("training.dataset_generator")

    return _run_stage(
        "Dataset generation",
        config.dataset_games,
        "game",
        lambda progress: dataset_generator.generate_dataset(
            game_count=config.dataset_games,
            output_file="dataset/supervised_dataset.jsonl",
            quiet=True,
            progress_callback=progress,
        ),
        lambda summary: (
            f"{summary['saved_turn_count']} real decisions, "
            f"{summary['skipped_turn_count']} forced turns skipped"
        ),
    )


def _run_supervised_training(config):
    """Train the supervised policy with compact epoch progress."""
    training_loop = _silent_import("training.training_loop")

    return _run_stage(
        "Supervised training",
        config.supervised_epochs,
        "epoch",
        lambda progress: training_loop.train_supervised(
            epochs=config.supervised_epochs,
            batch_size=training_loop.BATCH_SIZE,
            quiet=True,
            progress_callback=progress,
        ),
        lambda summary: (
            f"best validation loss {summary['best_validation_loss']:.4f}, "
            f"{summary['total_examples']} examples"
        ),
    )


def _run_rl_training(config):
    """Run reinforcement-learning self-play with compact iteration progress."""
    self_play = importlib.import_module("training.self_play")

    return _run_stage(
        "RL self-play",
        config.rl_iterations,
        "iter",
        lambda progress: self_play.train(
            iterations=config.rl_iterations,
            games_per_iteration=config.rl_games_per_iteration,
            quiet=True,
            progress_callback=progress,
        ),
        lambda summary: (
            f"{summary['iterations']} iterations x "
            f"{summary['games_per_iteration']} games, "
            f"weights {summary['rl_weights_path']}"
        ),
    )


def _run_diagnostics(config):
    """Run the all-pairs diagnostics matrix with one aggregate progress bar."""
    evaluate = importlib.import_module("diagnostics.evaluate")
    pair_count = len(evaluate.CANONICAL_AGENTS) * (len(evaluate.CANONICAL_AGENTS) + 1) // 2
    total_games = pair_count * config.diagnostic_games

    return _run_stage(
        "Diagnostics",
        total_games,
        "game",
        lambda progress: evaluate.run_all_pairs(
            game_count=config.diagnostic_games,
            output_dir=evaluate.DEFAULT_OUTPUT_DIR,
            quiet=True,
            progress_callback=progress,
        ),
        lambda summary: (
            f"{summary['evaluated_matchups']} matchups, "
            f"{summary['game_count_per_matchup']} games per matchup"
        ),
    )


def parse_args():
    """Parse the optional workload scale."""
    parser = argparse.ArgumentParser(
        description=(
            "Run dataset generation, supervised training, RL self-play, and "
            "all-pairs diagnostics with compact progress output."
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
    return parser.parse_args()


def main():
    """Run every pipeline stage in sequence."""
    args = parse_args()
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    config = _build_config(args.scale)
    print(
        "Pipeline scale "
        f"{config.scale_name} ({config.scale_factor:g}x): "
        f"{config.dataset_games} dataset games, "
        f"{config.supervised_epochs} supervised epochs, "
        f"{config.rl_iterations} RL iterations, "
        f"{config.diagnostic_games} diagnostics games per matchup."
    )

    start_time = time.time()
    dataset_summary = _run_dataset(config)
    supervised_summary = _run_supervised_training(config)
    rl_summary = _run_rl_training(config)
    diagnostics_summary = _run_diagnostics(config)
    elapsed_time = time.time() - start_time

    print("\nPipeline complete")
    print(f"Total elapsed time: {format_duration(elapsed_time)}")
    print(f"Dataset: {dataset_summary['output_file']}")
    print(f"Supervised weights: {supervised_summary['weights_file']}")
    print(f"RL weights: {rl_summary['rl_weights_path']}")
    print(f"Diagnostics: {diagnostics_summary['evaluated_matchups']} matchups in diagnostics/results/all_pairs/")


if __name__ == "__main__":
    main()
