"""Canonical seed-addressed supervised and game-budgeted RL pipeline."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from diagnostics import evaluate
from diagnostics.rl_progress import (
    final_diagnostic_seed,
    periodic_diagnostic_seed,
    read_periodic_history,
    rebuild_progress_reports,
    run_periodic_diagnostic,
)
from diagnostics.parallel_runner import MAX_DIAGNOSTIC_WORKERS, ParallelSafetyConfig
from training import dataset_generator, self_play, training_loop
from training.canonical_assets import (
    ArtifactCompatibilityError,
    canonical_asset_paths,
    canonical_generation_config,
    canonical_training_config,
    inspect_canonical_dataset,
    inspect_canonical_weights,
    write_dataset_metadata,
    write_weights_metadata,
)
from training.canonical_run import (
    canonical_run_dir,
    create_run_config,
    load_resume_point,
    publish_checkpoint,
    update_diagnostic_markers,
)
from utils.artifacts import atomic_copy, atomic_write_json, file_sha256
from utils.runtime_status import format_duration, pipeline_compute_report

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 42
CANONICAL_DATASET_GAMES = 100_000
CANONICAL_SUPERVISED_MAX_EPOCHS = 5_000
PERIODIC_DIAGNOSTIC_EVERY_GAMES = 100_000
PERIODIC_DIAGNOSTIC_GAMES = 100_000
FOREVER_TUNING_REFERENCE_GAMES = 500_000
FOREVER_INTERNAL_TARGET = 1 << 62
PERIODIC_DIAGNOSTIC_TUNING_FORMAT_VERSION = 1
PERIODIC_DIAGNOSTIC_TUNING_FILE = "periodic_diagnostic_tuning.json"


@dataclass(frozen=True)
class PipelineConfig:
    """Canonical workload for one named pipeline level."""

    scale_name: str
    total_rl_games: int | None
    diagnostic_games: int
    periodic_diagnostics: bool
    resume_supported: bool
    final_all_pairs: bool
    dataset_games: int = CANONICAL_DATASET_GAMES
    supervised_epochs: int = CANONICAL_SUPERVISED_MAX_EPOCHS
    rl_games_per_iteration: int = self_play.DEFAULT_GAMES_PER_ITERATION

    @property
    def unbounded(self):
        return self.total_rl_games is None

    @property
    def rl_iterations(self):
        if self.total_rl_games is None:
            return 0
        return (
            self.total_rl_games + self.rl_games_per_iteration - 1
        ) // self.rl_games_per_iteration

    @property
    def scale_factor(self):
        baseline = 500_000
        return 0.0 if self.total_rl_games is None else self.total_rl_games / baseline


PIPELINE_LEVELS = {
    "small": PipelineConfig("small", 100_000, 10_000, False, False, True),
    "default": PipelineConfig("default", 500_000, 10_000, False, False, True),
    "big": PipelineConfig("big", 2_000_000, 1_000_000, True, True, True),
    "huge": PipelineConfig("huge", 10_000_000, 1_000_000, True, True, True),
    "forever": PipelineConfig("forever", None, 0, True, True, False),
}
SCALE_FACTORS = {
    name: config.scale_factor for name, config in PIPELINE_LEVELS.items()
}


def _build_config(scale_name):
    return PIPELINE_LEVELS[scale_name]


def _diagnostic_summary_text(summary):
    worker_text = ", ".join(
        f"{matchup}={worker_count}"
        for matchup, worker_count in summary["selected_workers_by_matchup"].items()
    )
    return (
        f"{summary['evaluated_matchups']} matchups x "
        f"{summary['game_count_per_matchup']} games = "
        f"{summary['evaluated_matchups'] * summary['game_count_per_matchup']} "
        f"total games | workers: {worker_text}"
    )


def _status(message):
    if tqdm is None:
        print(message, flush=True)
    else:
        tqdm.write(message)


def _dataset_safety(args):
    return {
        "memory_reserve_mb": args.dataset_memory_reserve_mb,
        "estimated_worker_mb": args.dataset_estimated_worker_mb,
        "max_worker_rss_mb": args.dataset_max_worker_rss_mb,
    }


def _dataset_generation_identity(args, dataset_games):
    return canonical_generation_config(
        dataset_games=dataset_games,
        workers=args.dataset_workers,
        tuning={
            "fraction": args.dataset_autotune_fraction,
            "minimum_gain": args.dataset_autotune_min_gain,
        },
        safety=_dataset_safety(args),
    )


def _supervised_training_identity(args, max_epochs):
    return canonical_training_config(
        max_epochs=int(max_epochs),
        batch_size=args.sl_batch_size,
        weight_decay=float(args.weight_decay),
        early_stopping_patience=args.early_stopping,
        lr_decay_factor=args.lr_decay,
        lr_decay_patience=int(args.lr_decay_patience),
        training_plateau_enabled=not args.disable_training_plateau,
        training_plateau_window=int(args.sl_training_plateau_window),
        training_plateau_patience=int(args.sl_training_plateau_patience),
        training_plateau_min_epochs=int(args.sl_training_plateau_min_epochs),
        training_plateau_min_relative_improvement=float(
            args.sl_training_plateau_min_relative_improvement
        ),
        device=args.sl_device,
        autotune_batch_size=not args.sl_no_batch_autotune,
        memory_reserve_mb=int(args.sl_memory_reserve_mb),
        gpu_memory_reserve_mb=int(args.sl_gpu_memory_reserve_mb),
        validation_split=0.15,
        initial_learning_rate=training_loop.INITIAL_SUPERVISED_LEARNING_RATE,
    )


def _progress_callback(label, total, unit):
    if tqdm is None:
        last = {"done": 0}

        def callback(done, reported_total):
            step = max(1, (reported_total or total or 1) // 10)
            if done == reported_total or done - last["done"] >= step:
                suffix = f"/{reported_total}" if reported_total else ""
                print(f"{label}: {done}{suffix} {unit}", flush=True)
                last["done"] = done

        return contextlib.nullcontext(callback)

    @contextlib.contextmanager
    def manager():
        with tqdm(total=total, desc=label, unit=unit, leave=True) as bar:
            def callback(done, reported_total):
                if reported_total is not None and bar.total != reported_total:
                    bar.total = reported_total
                if done > bar.n:
                    bar.update(done - bar.n)
            yield callback

    return manager()


def ensure_canonical_supervised_assets(root, config, args):
    """Reuse or explicitly rebuild the canonical dataset and SL checkpoint."""
    seed = int(args.seed)
    dataset_games = int(args.dataset_games or config.dataset_games)
    max_epochs = int(args.supervised_max_epochs or config.supervised_epochs)
    paths = canonical_asset_paths(root, seed)
    generation_config = _dataset_generation_identity(args, dataset_games)
    rebuild_dataset = bool(
        args.rebuild_dataset or args.rebuild_supervised_assets
    )
    retrain_weights = bool(
        args.retrain_supervised or args.rebuild_supervised_assets
    )

    dataset_check = inspect_canonical_dataset(
        paths,
        seed=seed,
        dataset_games=dataset_games,
        generation_config=generation_config,
    )
    dataset_check.require_compatible_or_missing(
        rebuild=rebuild_dataset,
        label="supervised dataset",
    )
    dataset_summary = None
    if dataset_check.compatible and not rebuild_dataset:
        dataset_metadata = dataset_check.metadata
        dataset_status = "reused"
    else:
        print("\nCanonical dataset generation")
        with _progress_callback("Canonical dataset", dataset_games, "game") as progress:
            dataset_summary = dataset_generator.generate_dataset(
                game_count=dataset_games,
                output_file=paths.dataset,
                quiet=True,
                progress_callback=progress,
                workers=args.dataset_workers,
                safety_config=ParallelSafetyConfig(**_dataset_safety(args)),
                autotune_fraction=args.dataset_autotune_fraction,
                autotune_minimum_gain=args.dataset_autotune_min_gain,
                seed=seed,
                status_callback=_status,
            )
        dataset_metadata = write_dataset_metadata(
            paths,
            root=root,
            seed=seed,
            dataset_games=dataset_games,
            dataset_summary=dataset_summary,
            generation_config=generation_config,
        )
        dataset_status = "generated"
        retrain_weights = True

    training_config = _supervised_training_identity(args, max_epochs)
    weights_check = inspect_canonical_weights(
        paths,
        seed=seed,
        dataset_sha256=dataset_metadata["dataset_sha256"],
        training_config=training_config,
    )
    weights_check.require_compatible_or_missing(
        rebuild=retrain_weights,
        label="supervised weights",
    )
    supervised_summary = None
    if weights_check.compatible and not retrain_weights:
        weights_metadata = weights_check.metadata
        weights_status = "reused"
    else:
        print("\nCanonical supervised training")
        with _progress_callback("Supervised training", max_epochs, "epoch") as progress:
            supervised_summary = training_loop.train_supervised(
                epochs=max_epochs,
                batch_size=args.sl_batch_size,
                dataset_file=paths.dataset,
                weights_file=paths.weights,
                cache_file=paths.encoded_cache,
                quiet=True,
                progress_callback=progress,
                status_callback=_status,
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
                seed=seed,
                resume_existing_weights=False,
            )
        weights_metadata = write_weights_metadata(
            paths,
            root=root,
            seed=seed,
            dataset_sha256=dataset_metadata["dataset_sha256"],
            training_config=training_config,
            training_summary=supervised_summary,
        )
        weights_status = "trained"

    print("\n" + "-" * 70)
    print("Canonical supervised assets")
    print("-" * 70)
    print(f"Seed: {seed}")
    print(f"Dataset: {paths.dataset}")
    print(f"Dataset status: {dataset_status}")
    print(f"Dataset SHA-256: {dataset_metadata['dataset_sha256']}")
    print(f"Weights: {paths.weights}")
    print(f"Weights status: {weights_status}")
    print(f"Weights SHA-256: {weights_metadata['weights_sha256']}")
    print(
        "Supervised epochs: "
        f"{weights_metadata.get('epochs_completed')}/"
        f"{weights_metadata.get('max_epochs')} "
        f"(best: {weights_metadata.get('best_epoch')})"
    )
    print(f"Supervised stop: {weights_metadata.get('stopping_reason')}")
    print("-" * 70)
    return {
        "paths": paths,
        "dataset_status": dataset_status,
        "weights_status": weights_status,
        "dataset_metadata": dataset_metadata,
        "weights_metadata": weights_metadata,
        "dataset_summary": dataset_summary,
        "supervised_summary": supervised_summary,
    }


def _ppo_config(args):
    return {
        "clip_epsilon": float(args.ppo_clip_epsilon),
        "target_kl": float(args.ppo_target_kl),
        "stop_kl": float(args.ppo_stop_kl),
        "max_epochs": int(args.ppo_max_epochs),
        "min_minibatches": int(args.ppo_min_minibatches),
        "max_minibatches": int(args.ppo_max_minibatches),
        "games_per_minibatch_scale": int(args.ppo_games_per_minibatch_scale),
        "min_decisions_per_minibatch": int(
            args.ppo_min_decisions_per_minibatch
        ),
        "prefer_gpu_buffer": bool(args.prefer_gpu_buffer),
        "gpu_buffer_safety_fraction": float(args.gpu_buffer_safety_fraction),
    }


def _rl_config(args):
    normalize_advantages = (
        bool(args.ppo_enabled)
        if args.normalize_advantages is None
        else bool(args.normalize_advantages)
    )
    return {
        "training_opponent": args.training_opponent,
        "learning_rate": float(args.learning_rate),
        "entropy_coef": float(args.entropy_coef),
        "pool_refresh_games": int(args.pool_refresh_games),
        "max_pool_size": int(args.max_pool_size),
        "reward_schema": args.reward_schema,
        "gamma": float(args.gamma),
        "clip_grad_norm": float(args.clip_grad_norm),
        "normalize_advantages": normalize_advantages,
    }


def _archive_run_dir(run_dir):
    if not Path(run_dir).exists():
        return None
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    archive = Path(run_dir).with_name(f"{Path(run_dir).name}.archive-{timestamp}")
    counter = 1
    while archive.exists():
        archive = Path(run_dir).with_name(
            f"{Path(run_dir).name}.archive-{timestamp}-{counter}"
        )
        counter += 1
    os.replace(run_dir, archive)
    return archive


def _copy_lineage_reports(source_dir, destination_dir):
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "periodic_diagnostics.jsonl",
        "rl_vs_random_progress.csv",
        "rl_vs_random_progress.png",
        "rl_vs_random_progress_logx.png",
        "best_checkpoint.json",
        "training_metrics.jsonl",
        "adaptive_tuning.json",
    ):
        source = source_dir / name
        destination = destination_dir / name
        if source.is_file() and not destination.exists():
            atomic_copy(source, destination)


class ShutdownFlag:
    """Signal handler that requests a boundary checkpoint on first signal."""

    def __init__(self):
        self.requested = False
        self.signal_name = None
        self._previous = {}

    def __call__(self):
        return self.requested

    def _handler(self, signum, _frame):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        self.signal_name = signal.Signals(signum).name
        _status(
            f"Shutdown requested by {self.signal_name}; the current RL "
            "iteration will finish before a safe checkpoint."
        )

    def __enter__(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handler)
        return self

    def __exit__(self, *_exc):
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)


def next_training_stop(completed_games, target_games, milestone_every, periodic):
    """Return an exact segment boundary without ever exceeding the target."""
    completed_games = int(completed_games)
    if periodic:
        milestone = (
            completed_games // int(milestone_every) + 1
        ) * int(milestone_every)
    else:
        milestone = target_games
    if target_games is None:
        return milestone
    return min(int(target_games), int(milestone))


def _cumulative_rl_games_per_second(completed_games, prior_seconds, invocation_seconds):
    """Return throughput across the complete persisted RL training lineage."""
    elapsed = float(prior_seconds) + float(invocation_seconds)
    return float(completed_games) / elapsed if elapsed > 0.0 else 0.0


def _rl_progress(total, initial):
    if tqdm is None:
        return contextlib.nullcontext(None)

    @contextlib.contextmanager
    def manager():
        if total is None:
            bar_format = "{desc}: {n_fmt} games [{elapsed}{postfix}]"
        else:
            bar_format = (
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}{postfix}]"
            )
        with tqdm(
            total=total,
            initial=initial,
            desc="RL self-play",
            unit="game",
            leave=True,
            bar_format=bar_format,
        ) as bar:
            yield bar

    return manager()


def _periodic_worker_tuning_identity(args):
    return {
        "format_version": PERIODIC_DIAGNOSTIC_TUNING_FORMAT_VERSION,
        "pipeline_level": "forever",
        "seed": int(args.seed),
        "diagnostic_seed": int(periodic_diagnostic_seed(args.seed)),
        "opponent": "random",
        "diagnostic_games": int(args.periodic_diagnostic_games),
        "autotune_fraction": float(args.diagnostic_autotune_fraction),
        "autotune_minimum_gain": float(args.diagnostic_autotune_min_gain),
        "safety_config": {
            "memory_reserve_mb": int(args.diagnostic_memory_reserve_mb),
            "estimated_worker_mb": int(args.diagnostic_estimated_worker_mb),
            "max_worker_rss_mb": int(args.diagnostic_max_worker_rss_mb),
        },
    }


def _write_periodic_worker_tuning(
    run_dir,
    args,
    selected_workers,
    *,
    source,
    selected_at_rl_games,
):
    value = {
        **_periodic_worker_tuning_identity(args),
        "selected_workers": int(selected_workers),
        "source": source,
        "selected_at_rl_games": int(selected_at_rl_games),
    }
    atomic_write_json(Path(run_dir) / PERIODIC_DIAGNOSTIC_TUNING_FILE, value)
    return value


def _resolve_periodic_diagnostic_workers(run_dir, level, args):
    """Reuse one persisted forever diagnostic-worker selection when possible."""
    requested = args.diagnostic_workers
    if level != "forever" or requested != "auto":
        source = "manual" if requested != "auto" else "per-diagnostic autotune"
        return requested, source

    expected = _periodic_worker_tuning_identity(args)
    tuning_path = Path(run_dir) / PERIODIC_DIAGNOSTIC_TUNING_FILE
    had_saved_tuning = tuning_path.is_file()
    if had_saved_tuning:
        try:
            saved = json.loads(tuning_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            saved = None
        if isinstance(saved, dict) and all(
            saved.get(name) == value for name, value in expected.items()
        ):
            try:
                selected = int(saved["selected_workers"])
            except (KeyError, TypeError, ValueError):
                selected = 0
            if 1 <= selected <= MAX_DIAGNOSTIC_WORKERS:
                return selected, "saved forever selection"
        return "auto", "one-time forever autotune after configuration change"

    history = read_periodic_history(Path(run_dir) / "periodic_diagnostics.jsonl")
    for row in reversed(history):
        try:
            matches_identity = (
                row.get("pipeline_level") == "forever"
                and int(row.get("seed", -1)) == expected["seed"]
                and int(row.get("diagnostic_seed", -1))
                == expected["diagnostic_seed"]
                and row.get("opponent") == expected["opponent"]
                and int(row.get("diagnostic_games", -1))
                == expected["diagnostic_games"]
            )
        except (TypeError, ValueError):
            continue
        if matches_identity:
            try:
                selected = int(row["selected_workers"])
            except (KeyError, TypeError, ValueError):
                continue
            if 1 <= selected <= MAX_DIAGNOSTIC_WORKERS:
                _write_periodic_worker_tuning(
                    run_dir,
                    args,
                    selected,
                    source="recovered_from_periodic_history",
                    selected_at_rl_games=int(row["rl_games"]),
                )
                return selected, "recovered forever selection"
    return "auto", "one-time forever autotune"


def _run_periodic_point(
    *,
    args,
    run_dir,
    level,
    checkpoint,
    games,
    iterations,
    optimizer_steps,
    elapsed_rl_seconds,
    pipeline_started,
):
    diagnostic_workers, worker_source = _resolve_periodic_diagnostic_workers(
        run_dir,
        level,
        args,
    )
    row, appended = run_periodic_diagnostic(
        run_dir=run_dir,
        pipeline_level=level,
        seed=args.seed,
        rl_games=games,
        rl_iterations=iterations,
        optimizer_steps=optimizer_steps,
        checkpoint_path=checkpoint,
        diagnostic_games=args.periodic_diagnostic_games,
        rl_elapsed_seconds=elapsed_rl_seconds,
        wall_clock_seconds=time.time() - pipeline_started,
        workers=diagnostic_workers,
        safety_config=ParallelSafetyConfig(
            memory_reserve_mb=args.diagnostic_memory_reserve_mb,
            estimated_worker_mb=args.diagnostic_estimated_worker_mb,
            max_worker_rss_mb=args.diagnostic_max_worker_rss_mb,
        ),
        autotune_fraction=args.diagnostic_autotune_fraction,
        autotune_minimum_gain=args.diagnostic_autotune_min_gain,
        status_callback=_status,
    )
    if (
        level == "forever"
        and args.diagnostic_workers == "auto"
        and diagnostic_workers == "auto"
    ):
        _write_periodic_worker_tuning(
            run_dir,
            args,
            row["selected_workers"],
            source="one_time_autotune",
            selected_at_rl_games=int(row["rl_games"]),
        )
        worker_source = "autotuned once and saved"
    action = "recorded" if appended else "reused"
    print("\n" + "-" * 70)
    print(f"Periodic RL diagnostic ({action})")
    print("-" * 70)
    print(f"Checkpoint: {games:,} RL games")
    print(
        f"Weights: {Path(row['checkpoint_path']).name} | "
        f"SHA-256: {row['checkpoint_sha256'][:12]}..."
    )
    print(f"Opponent: random | games: {row['diagnostic_games']:,}")
    print(f"Workers: {row['selected_workers']} ({worker_source})")
    print(
        f"Wins/draws/losses: {row['wins']:,}/{row['draws']:,}/"
        f"{row['losses']:,}"
    )
    print(
        f"Win rate: {row['win_rate']:.2%} | score: {row['score']:.2%} | "
        f"95% CI: [{row['ci95_win_rate_low']:.2%}, "
        f"{row['ci95_win_rate_high']:.2%}]"
    )
    print(f"Time: {format_duration(row['diagnostic_seconds'])}")
    print(f"History: {Path(run_dir) / 'periodic_diagnostics.jsonl'}")
    print(f"Graph: {Path(run_dir) / 'rl_vs_random_progress.png'}")
    print("-" * 70)
    return row


def run_rl_pipeline(root, config, args, assets, *, pipeline_started):
    """Run finite milestone segments or an unbounded sequence with exact resume."""
    seed = int(args.seed)
    target = (
        int(args.total_training_games)
        if args.total_training_games is not None and not config.unbounded
        else config.total_rl_games
    )
    run_dir = canonical_run_dir(root, config.scale_name, seed)
    if args.restart_rl:
        archive = _archive_run_dir(run_dir)
        if archive is not None:
            print(f"Archived previous RL run at {archive}.")
    ppo_config = _ppo_config(args)
    supervised_path = assets["paths"].weights
    supervised_hash = assets["weights_metadata"]["weights_sha256"]

    source_run_dir = Path(args.resume_from) if args.resume_from else run_dir
    resuming = bool(args.resume or args.resume_from)
    if args.resume and not config.resume_supported:
        raise ValueError(f"--resume is not supported for pipeline level {config.scale_name}.")
    if run_dir.exists() and (run_dir / "training_state.json").exists() and not resuming:
        raise FileExistsError(
            f"RL run {run_dir} already exists. Use --resume or --restart-rl explicitly."
        )
    resume_point = None
    lineage = []
    if resuming:
        resume_point = load_resume_point(
            source_run_dir,
            seed=seed,
            supervised_weights_sha256=supervised_hash,
            ppo_config=ppo_config,
            force_incompatible=args.force_resume_incompatible,
        )
        if args.resume_from and source_run_dir.resolve() != run_dir.resolve():
            if (run_dir / "training_state.json").exists():
                raise FileExistsError(
                    f"Destination run {run_dir} already contains training state."
                )
            source_lineage = []
            source_config_path = source_run_dir / "run_config.json"
            if source_config_path.is_file():
                try:
                    source_lineage = list(json.loads(
                        source_config_path.read_text(encoding="utf-8")
                    ).get("lineage", ()))
                except (OSError, json.JSONDecodeError, TypeError) as exc:
                    raise ValueError(
                        f"Source lineage config cannot be read: "
                        f"{source_config_path}."
                    ) from exc
            lineage = source_lineage + [{
                "source_run_dir": str(source_run_dir),
                "source_rl_games": resume_point.completed_games,
                "source_latest_weights_sha256": resume_point.training_state[
                    "latest_weights_sha256"
                ],
            }]
            _copy_lineage_reports(source_run_dir, run_dir)
    create_run_config(
        run_dir,
        root=root,
        pipeline_level=config.scale_name,
        seed=seed,
        target_rl_games=target,
        supervised_weights_path=supervised_path,
        supervised_weights_sha256=supervised_hash,
        ppo_config=ppo_config,
        rl_config=_rl_config(args),
        diagnostic_config={
            "periodic_seed": int(periodic_diagnostic_seed(seed)),
            "periodic_seed_namespace": "periodic_rl_vs_random",
            "periodic_games": int(args.periodic_diagnostic_games),
            "periodic_every_rl_games": int(
                args.periodic_diagnostic_every_games
            ),
            "final_seed": int(final_diagnostic_seed(seed)),
            "final_seed_namespace": "final_all_pairs_holdout",
            "final_games_per_matchup": int(
                args.final_diagnostic_games or config.diagnostic_games
            ) if config.final_all_pairs else None,
        },
        lineage=lineage,
        allow_target_extension=bool(args.resume_from),
    )

    completed = 0 if resume_point is None else resume_point.completed_games
    iterations = 0 if resume_point is None else resume_point.completed_iterations
    latest_weights = supervised_path if resume_point is None else resume_point.weights_path
    optimizer_steps = 0 if resume_point is None else int(
        resume_point.training_state["optimizer_steps_completed"]
    )
    elapsed_rl = 0.0 if resume_point is None else float(
        resume_point.training_state["elapsed_rl_seconds"]
    )
    history = read_periodic_history(run_dir / "periodic_diagnostics.jsonl")
    last_periodic = max((int(row["rl_games"]) for row in history), default=0)
    next_boundary = next_training_stop(
        completed,
        target,
        args.periodic_diagnostic_every_games,
        config.periodic_diagnostics,
    ) if target is None or completed < target else None
    print("\n" + "-" * 70)
    print("Canonical RL run")
    print("-" * 70)
    print(f"Pipeline level: {config.scale_name}")
    print(f"Target RL games: {'forever' if target is None else f'{target:,}'}")
    print(f"Seed: {seed}")
    print(f"Supervised checkpoint: {supervised_path}")
    print(f"Resume: {'yes' if resuming else 'no'}")
    print(f"Games already completed: {completed:,}")
    if target is not None:
        print(f"Games remaining: {max(0, target - completed):,}")
    if config.periodic_diagnostics:
        if args.skip_periodic_diagnostics:
            periodic_text = "skipped"
        elif next_boundary is None:
            periodic_text = "none"
        else:
            periodic_text = f"{next_boundary:,}"
        print(f"Next periodic diagnostic: {periodic_text}")
    print("-" * 70)

    if config.periodic_diagnostics and not args.skip_periodic_diagnostics:
        _run_periodic_point(
            args=args,
            run_dir=run_dir,
            level=config.scale_name,
            checkpoint=supervised_path,
            games=0,
            iterations=0,
            optimizer_steps=0,
            elapsed_rl_seconds=0.0,
            pipeline_started=pipeline_started,
        )

    if (
        config.periodic_diagnostics
        and not args.skip_periodic_diagnostics
        and completed > 0
        and completed % int(args.periodic_diagnostic_every_games) == 0
        and last_periodic < completed
    ):
        state = resume_point.training_state
        checkpoint_value = (
            state.get("latest_milestone_checkpoint")
            or state["latest_weights_path"]
        )
        checkpoint = Path(checkpoint_value)
        if not checkpoint.is_absolute():
            checkpoint = resume_point.run_dir / checkpoint
        _run_periodic_point(
            args=args,
            run_dir=run_dir,
            level=config.scale_name,
            checkpoint=checkpoint,
            games=completed,
            iterations=iterations,
            optimizer_steps=optimizer_steps,
            elapsed_rl_seconds=elapsed_rl,
            pipeline_started=pipeline_started,
        )
        last_periodic = completed
    if (
        resume_point is not None
        and resume_point.run_dir.resolve() == run_dir.resolve()
        and (
        int(resume_point.training_state.get("last_periodic_diagnostic_game", 0))
        != last_periodic
        )
    ):
        next_periodic = (
            (last_periodic // int(args.periodic_diagnostic_every_games) + 1)
            * int(args.periodic_diagnostic_every_games)
            if config.periodic_diagnostics else None
        )
        if target is not None and next_periodic is not None and next_periodic > target:
            next_periodic = None
        update_diagnostic_markers(
            run_dir,
            last_periodic_diagnostic_game=last_periodic,
            next_periodic_diagnostic_game=next_periodic,
        )
        resume_point = load_resume_point(
            run_dir,
            seed=seed,
            supervised_weights_sha256=supervised_hash,
            ppo_config=ppo_config,
            force_incompatible=args.force_resume_incompatible,
        )

    internal_target = FOREVER_INTERNAL_TARGET if target is None else int(target)
    periodic_boundaries = config.periodic_diagnostics
    checkpoint_base = run_dir / "checkpoint_states" / "training.npz"
    metrics_path = run_dir / "training_metrics.jsonl"
    tuning_path = run_dir / "adaptive_tuning.json"
    last_summary = None

    with ShutdownFlag() as shutdown, _rl_progress(target, completed) as progress_bar:
        while target is None or completed < target:
            if shutdown() and resume_point is not None:
                break
            stop_at = next_training_stop(
                completed,
                target,
                args.periodic_diagnostic_every_games,
                periodic_boundaries,
            )
            base_kwargs = self_play._training_kwargs_from_args(args)
            base_kwargs.update({
                "iterations": None,
                "total_training_games": internal_target,
                "stop_after_training_games": stop_at,
                "adaptive_tuning_training_games": (
                    FOREVER_TUNING_REFERENCE_GAMES if target is None else int(target)
                ),
                "sl_weights_path": str(supervised_path),
                "rl_weights_path": str(checkpoint_base),
                "adaptive_tuning_path": str(tuning_path),
                "metrics_output_path": str(metrics_path),
                "numbered_checkpoints": True,
                "start_iteration": iterations,
                "resume_weights_path": (
                    None if resume_point is None else str(resume_point.weights_path)
                ),
                "resume_state_file": (
                    None if resume_point is None else str(resume_point.resume_state_path)
                ),
                "fresh_from_sl": resume_point is None,
                "allow_total_training_games_extension": bool(args.resume_from),
                "force_resume_incompatible": bool(args.force_resume_incompatible),
                "shutdown_requested": shutdown,
                "quiet": True,
                "status_callback": _status,
            })

            def progress(done, _reported_total):
                if progress_bar is not None and done > progress_bar.n:
                    progress_bar.update(done - progress_bar.n)

            def metrics(row):
                if progress_bar is not None:
                    rate = _cumulative_rl_games_per_second(
                        int(row["cumulative_games"]),
                        elapsed_rl,
                        float(row["elapsed_training_s"]),
                    )
                    progress_bar.set_postfix(
                        gpi=row["games_per_iteration"],
                        iteration=row["iteration"],
                        avg_games_s=f"{rate:.1f}",
                        next_diagnostic=(
                            f"{stop_at:,}"
                            if (
                                config.periodic_diagnostics
                                and not args.skip_periodic_diagnostics
                            ) else "off"
                        ),
                    )

            def publish_scheduled_checkpoint(event):
                checkpoint_games = int(event["completed_training_games"])
                checkpoint_milestone = (
                    config.periodic_diagnostics
                    and checkpoint_games
                    % int(args.periodic_diagnostic_every_games)
                    == 0
                )
                checkpoint_next_periodic = (
                    (
                        checkpoint_games
                        // int(args.periodic_diagnostic_every_games)
                        + 1
                    )
                    * int(args.periodic_diagnostic_every_games)
                    if config.periodic_diagnostics else None
                )
                if (
                    target is not None
                    and checkpoint_next_periodic is not None
                    and checkpoint_next_periodic > target
                ):
                    checkpoint_next_periodic = None
                publish_checkpoint(
                    run_dir,
                    root=root,
                    pipeline_level=config.scale_name,
                    seed=seed,
                    target_rl_games=target,
                    supervised_weights_path=supervised_path,
                    supervised_weights_sha256=supervised_hash,
                    summary={
                        **event,
                        "ppo_configuration": ppo_config,
                    },
                    last_periodic_diagnostic_game=last_periodic,
                    next_periodic_diagnostic_game=checkpoint_next_periodic,
                    milestone=checkpoint_milestone,
                    reason=(
                        "periodic_milestone_pending_diagnostic"
                        if checkpoint_milestone
                        else "scheduled_checkpoint"
                    ),
                )

            base_kwargs["progress_callback"] = progress
            base_kwargs["metrics_callback"] = metrics
            base_kwargs["checkpoint_callback"] = publish_scheduled_checkpoint
            last_summary = self_play.train(**base_kwargs)
            completed = int(last_summary["completed_training_games"])
            iterations = int(last_summary["rl_iterations_completed"])
            optimizer_steps = int(last_summary["optimizer_step_count"])
            elapsed_rl = float(last_summary["elapsed_rl_seconds"])
            milestone = (
                completed % int(args.periodic_diagnostic_every_games) == 0
            )
            next_periodic = (
                (completed // int(args.periodic_diagnostic_every_games) + 1)
                * int(args.periodic_diagnostic_every_games)
                if config.periodic_diagnostics else None
            )
            if target is not None and next_periodic is not None and next_periodic > target:
                next_periodic = None
            state = publish_checkpoint(
                run_dir,
                root=root,
                pipeline_level=config.scale_name,
                seed=seed,
                target_rl_games=target,
                supervised_weights_path=supervised_path,
                supervised_weights_sha256=supervised_hash,
                summary=last_summary,
                last_periodic_diagnostic_game=last_periodic,
                next_periodic_diagnostic_game=next_periodic,
                milestone=milestone and config.periodic_diagnostics,
                reason="shutdown" if (
                    last_summary["shutdown_requested"] or shutdown()
                ) else (
                    "periodic_milestone" if milestone else "target_complete"
                ),
            )
            resume_point = load_resume_point(
                run_dir,
                seed=seed,
                supervised_weights_sha256=supervised_hash,
                ppo_config=ppo_config,
                force_incompatible=args.force_resume_incompatible,
            )
            latest_weights = resume_point.weights_path
            if last_summary["shutdown_requested"] or shutdown():
                break
            if (
                milestone
                and config.periodic_diagnostics
                and not args.skip_periodic_diagnostics
            ):
                checkpoint = run_dir / state["latest_milestone_checkpoint"]
                _run_periodic_point(
                    args=args,
                    run_dir=run_dir,
                    level=config.scale_name,
                    checkpoint=checkpoint,
                    games=completed,
                    iterations=iterations,
                    optimizer_steps=optimizer_steps,
                    elapsed_rl_seconds=elapsed_rl,
                    pipeline_started=pipeline_started,
                )
                last_periodic = completed
                update_diagnostic_markers(
                    run_dir,
                    last_periodic_diagnostic_game=last_periodic,
                    next_periodic_diagnostic_game=next_periodic,
                )
            if target is not None and completed >= target:
                break
            if shutdown():
                break

        shutdown_seen = shutdown()

    return {
        "run_dir": str(run_dir),
        "rl_weights_path": str(latest_weights),
        "completed_training_games": completed,
        "rl_iterations_completed": iterations,
        "optimizer_steps_completed": optimizer_steps,
        "elapsed_rl_seconds": elapsed_rl,
        "shutdown_requested": bool(
            shutdown_seen
            or (last_summary and last_summary["shutdown_requested"])
        ),
        "target_rl_games": target,
        "summary": last_summary,
    }


def run_final_diagnostics(root, config, args, assets, rl_result):
    if config.unbounded or not config.final_all_pairs or args.skip_final_diagnostic:
        return None
    game_count = int(args.final_diagnostic_games or config.diagnostic_games)
    run_dir = Path(rl_result["run_dir"])
    output = run_dir / "final_diagnostics"
    print(
        f"\nFinal holdout diagnostics: 5 matchups x {game_count:,} games "
        f"({5 * game_count:,} total)."
    )
    summary = evaluate.run_all_pairs(
        game_count=game_count,
        output_dir=output,
        seed=final_diagnostic_seed(args.seed),
        rl_weights=rl_result["rl_weights_path"],
        neural_weights=assets["paths"].weights,
        quiet=True,
        workers=args.diagnostic_workers,
        safety_config=ParallelSafetyConfig(
            memory_reserve_mb=args.diagnostic_memory_reserve_mb,
            estimated_worker_mb=args.diagnostic_estimated_worker_mb,
            max_worker_rss_mb=args.diagnostic_max_worker_rss_mb,
        ),
        autotune_fraction=args.diagnostic_autotune_fraction,
        autotune_minimum_gain=args.diagnostic_autotune_min_gain,
        status_callback=_status,
    )
    print(_diagnostic_summary_text(summary))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run canonical seed-addressed supervised assets, exact-game RL, "
            "resume, and level-specific diagnostics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "scale",
        nargs="?",
        default="default",
        choices=tuple(PIPELINE_LEVELS),
        help="Canonical pipeline level.",
    )
    dataset = parser.add_argument_group("canonical dataset controls")
    dataset.add_argument(
        "--dataset-workers",
        type=dataset_generator._worker_count,
        default=dataset_generator.DEFAULT_DATASET_WORKERS,
    )
    dataset.add_argument(
        "--dataset-autotune-fraction",
        type=float,
        default=dataset_generator.DEFAULT_DATASET_AUTOTUNE_FRACTION,
    )
    dataset.add_argument(
        "--dataset-autotune-min-gain",
        type=float,
        default=dataset_generator.DEFAULT_DATASET_MINIMUM_GAIN,
    )
    dataset.add_argument("--dataset-memory-reserve-mb", type=int, default=512)
    dataset.add_argument("--dataset-estimated-worker-mb", type=int, default=256)
    dataset.add_argument("--dataset-max-worker-rss-mb", type=int, default=1024)
    dataset.add_argument("--dataset-games", type=int, default=None, help=argparse.SUPPRESS)
    training_loop.add_optional_training_arguments(parser)
    self_play.add_optional_rl_arguments(parser, fresh_from_sl_default=True)
    diagnostics = parser.add_argument_group("diagnostic controls")
    diagnostics.add_argument(
        "--diagnostic-workers",
        type=evaluate._worker_count,
        default=evaluate.DEFAULT_DIAGNOSTIC_WORKERS,
    )
    diagnostics.add_argument(
        "--diagnostic-autotune-fraction",
        type=float,
        default=evaluate.DEFAULT_AUTOTUNE_FRACTION,
    )
    diagnostics.add_argument(
        "--diagnostic-autotune-min-gain",
        type=float,
        default=evaluate.DEFAULT_MINIMUM_GAIN,
    )
    diagnostics.add_argument("--diagnostic-memory-reserve-mb", type=int, default=512)
    diagnostics.add_argument("--diagnostic-estimated-worker-mb", type=int, default=256)
    diagnostics.add_argument("--diagnostic-max-worker-rss-mb", type=int, default=1024)
    canonical = parser.add_argument_group("canonical pipeline controls")
    canonical.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        type=Path,
        metavar="RUN_DIR",
        help=(
            "Resume the canonical run. With no value, use the default run "
            "directory; with RUN_DIR, behave like --resume-from RUN_DIR."
        ),
    )
    canonical.add_argument("--resume-from", type=Path)
    canonical.add_argument("--restart-rl", action="store_true")
    canonical.add_argument("--force-resume-incompatible", action="store_true")
    canonical.add_argument("--rebuild-dataset", action="store_true")
    canonical.add_argument("--retrain-supervised", action="store_true")
    canonical.add_argument("--rebuild-supervised-assets", action="store_true")
    canonical.add_argument("--skip-final-diagnostic", action="store_true")
    canonical.add_argument("--skip-periodic-diagnostics", action="store_true")
    canonical.add_argument(
        "--periodic-diagnostic-games",
        type=int,
        default=PERIODIC_DIAGNOSTIC_GAMES,
    )
    canonical.add_argument(
        "--periodic-diagnostic-every-games",
        type=int,
        default=PERIODIC_DIAGNOSTIC_EVERY_GAMES,
    )
    canonical.add_argument("--final-diagnostic-games", type=int)
    canonical.add_argument("--supervised-max-epochs", type=int, help=argparse.SUPPRESS)
    canonical.add_argument("--artifact-root", type=Path, default=ROOT)
    parser.set_defaults(seed=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if isinstance(args.resume, Path):
        if args.resume_from is not None:
            parser.error("--resume RUN_DIR cannot be combined with --resume-from")
        args.resume_from = args.resume
        args.resume = False
    return args


def validate_args(args, config):
    if args.resume and args.resume_from:
        raise ValueError("Use either --resume or --resume-from, not both.")
    if args.restart_rl and (args.resume or args.resume_from):
        raise ValueError("--restart-rl cannot be combined with resume options.")
    if (args.resume or args.resume_from) and not config.resume_supported:
        raise ValueError(
            "RL resume is not supported for pipeline level "
            f"{config.scale_name}."
        )
    if args.iterations is not None:
        raise ValueError(
            "Canonical pipelines use RL game budgets; --iterations is available "
            "only in training.self_play."
        )
    if config.unbounded and args.total_training_games is not None:
        raise ValueError("The forever level cannot have --total-training-games.")
    if args.total_training_games is not None and args.total_training_games < 1:
        raise ValueError("total_training_games must be positive.")
    if args.dataset_games is not None and args.dataset_games < 1:
        raise ValueError("dataset_games must be positive.")
    if args.supervised_max_epochs is not None and args.supervised_max_epochs < 1:
        raise ValueError("supervised_max_epochs must be positive.")
    if args.final_diagnostic_games is not None and args.final_diagnostic_games < 1:
        raise ValueError("final_diagnostic_games must be positive.")
    if args.sl_seed is not None and int(args.sl_seed) != int(args.seed):
        raise ValueError(
            "Canonical assets use --seed for every stage; remove the conflicting "
            "legacy --sl-seed value."
        )
    if not args.fresh_from_sl:
        raise ValueError(
            "Canonical new runs always start from supervised weights; use "
            "--resume or --resume-from to continue RL."
        )
    for name in (
        "periodic_diagnostic_games",
        "periodic_diagnostic_every_games",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive.")
    if not args.ppo_enabled:
        raise ValueError("Canonical RL runs require the current PPO algorithm.")
    if args.value_head:
        raise ValueError("Canonical PPO runs do not use the legacy value head.")


def main(argv=None):
    args = parse_args(argv)
    config = _build_config(args.scale)
    validate_args(args, config)
    root = Path(args.artifact_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    started = time.time()
    effective_dataset_games = int(args.dataset_games or config.dataset_games)
    effective_supervised_epochs = int(
        args.supervised_max_epochs or config.supervised_epochs
    )
    effective_rl_target = (
        None
        if config.unbounded
        else int(
            config.total_rl_games
            if args.total_training_games is None
            else args.total_training_games
        )
    )
    target_text = (
        "unbounded" if effective_rl_target is None else f"{effective_rl_target:,}"
    )
    print(
        f"Canonical pipeline: level={config.scale_name}, seed={args.seed}, "
        f"dataset={effective_dataset_games:,} games, supervised max="
        f"{effective_supervised_epochs:,} epochs, RL target={target_text} games."
    )
    print(pipeline_compute_report(args.device, args.sl_device))
    assets = ensure_canonical_supervised_assets(root, config, args)
    rl_result = run_rl_pipeline(
        root,
        config,
        args,
        assets,
        pipeline_started=started,
    )
    diagnostics = None
    if not rl_result["shutdown_requested"]:
        diagnostics = run_final_diagnostics(root, config, args, assets, rl_result)
    print("\nCanonical pipeline finished")
    print(f"Elapsed: {format_duration(time.time() - started)}")
    print(f"Supervised weights: {assets['paths'].weights}")
    print(f"RL run directory: {rl_result['run_dir']}")
    print(f"RL games completed: {rl_result['completed_training_games']:,}")
    if config.unbounded:
        print("Forever ended without an automatic final all-pairs diagnostic.")
    return {
        "assets": assets,
        "rl": rl_result,
        "diagnostics": diagnostics,
    }


if __name__ == "__main__":
    main()
