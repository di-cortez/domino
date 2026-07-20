"""Generate deterministic supervised examples with CPU multiprocessing.

Only states with at least two legal tile-play actions are written. Forced draw,
pass, opening-double, and single-tile-play turns are excluded. Worker autotuning
games are retained in the final JSONL file.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sqlite3
import tempfile
import time
from pathlib import Path

from diagnostics.parallel_runner import (
    MAX_PARALLEL_WORKERS,
    ParallelSafetyConfig,
    cap_parallel_workers,
    game_seed,
)
from training.dataset_parallel import (
    DEFAULT_DATASET_AUTOTUNE_FRACTION,
    DEFAULT_DATASET_MINIMUM_GAIN,
    _is_real_decision_state,
    _legal_tile_actions_from_state,
    _normalize_action,
    autotune_dataset_workers,
    evaluate_dataset_game_specs,
)
from utils.runtime_status import format_duration, print_memory_report

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DEFAULT_GAME_COUNT = 30000
DEFAULT_OUTPUT_FILE = "dataset/supervised_dataset.jsonl"
DEFAULT_DATASET_WORKERS = "auto"
DEFAULT_DATASET_SEED = None


def _worker_count(value):
    """Parse ``auto`` or a worker count bounded by the hard limit."""
    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "workers must be 'auto' or an integer"
        ) from exc
    if not 1 <= parsed <= MAX_PARALLEL_WORKERS:
        raise argparse.ArgumentTypeError(
            f"workers must be between 1 and {MAX_PARALLEL_WORKERS}"
        )
    return parsed


def _public_autotune_summary(tuning: dict) -> dict:
    """Remove the in-memory completed-id set from persisted run metadata."""
    return {
        "optimal_workers": tuning["optimal_workers"],
        "candidate_workers": tuning["candidate_workers"],
        "benchmark_fraction": tuning["benchmark_fraction"],
        "minimum_gain": tuning["minimum_gain"],
        "games_per_test": tuning["games_per_test"],
        "reused_game_count": tuning["reused_game_count"],
        "attempts": tuning["attempts"],
    }


def generate_dataset(
    game_count,
    output_file,
    quiet=False,
    progress_callback=None,
    *,
    workers=DEFAULT_DATASET_WORKERS,
    safety_config=None,
    autotune_fraction=DEFAULT_DATASET_AUTOTUNE_FRACTION,
    autotune_minimum_gain=DEFAULT_DATASET_MINIMUM_GAIN,
    seed=DEFAULT_DATASET_SEED,
    status_callback=None,
):
    """Generate an ordered JSONL dataset using retained CPU worker autotuning.

    Workers return one compact payload per game. The parent inserts payloads
    into a disposable SQLite database keyed by absolute game id, which keeps
    RAM bounded and permits a deterministic final ``ORDER BY game_index``.
    The destination JSONL is atomically replaced only after every game succeeds.
    """
    if game_count < 1:
        raise ValueError("game_count must be positive")
    if workers != "auto":
        workers = int(workers)
        if not 1 <= workers <= MAX_PARALLEL_WORKERS:
            raise ValueError(
                f"workers must be 'auto' or between 1 and {MAX_PARALLEL_WORKERS}"
            )
    safety_config = safety_config or ParallelSafetyConfig()
    effective_seed = int(seed) if seed is not None else secrets.randbits(63)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    if not quiet:
        print(f"Generating {game_count} games...")
        print_memory_report("Dataset generation startup memory")

    progress_bar = None
    if progress_callback is None and not quiet and tqdm is not None:
        progress_bar = tqdm(
            total=game_count,
            desc="Generating dataset",
            unit="game",
            leave=True,
        )

        def effective_progress(done, _total):
            if done > progress_bar.n:
                progress_bar.update(done - progress_bar.n)
    else:
        effective_progress = progress_callback

    if status_callback is not None:
        emit_status = status_callback
    elif quiet:
        emit_status = lambda _message: None
    elif tqdm is not None:
        emit_status = tqdm.write
    else:
        emit_status = lambda message: print(message, flush=True)

    database_fd, database_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.games-",
        suffix=".sqlite3",
        dir=output_path.parent,
    )
    os.close(database_fd)
    staging_fd = None
    staging_name = None
    connection = None

    completed_game_count = 0
    try:
        connection = sqlite3.connect(database_name)
        # The database is temporary and rebuilt on failure, so durability
        # synchronization and a rollback journal add cost without safety value.
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            """
            CREATE TABLE games (
                game_index INTEGER PRIMARY KEY,
                game_seed TEXT NOT NULL,
                jsonl_payload TEXT NOT NULL,
                saved_turn_count INTEGER NOT NULL,
                skipped_turn_count INTEGER NOT NULL
            )
            """
        )

        def store_result(result):
            nonlocal completed_game_count
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO games (
                    game_index,
                    game_seed,
                    jsonl_payload,
                    saved_turn_count,
                    skipped_turn_count
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    int(result["game_index"]),
                    str(result["game_seed"]),
                    result["jsonl_payload"],
                    int(result["saved_turn_count"]),
                    int(result["skipped_turn_count"]),
                ),
            )
            if cursor.rowcount:
                completed_game_count += 1
                if effective_progress is not None:
                    effective_progress(completed_game_count, game_count)

        if workers == "auto":
            tuning = autotune_dataset_workers(
                game_count=game_count,
                base_seed=effective_seed,
                safety=safety_config,
                result_callback=store_result,
                benchmark_fraction=autotune_fraction,
                minimum_gain=autotune_minimum_gain,
                status_callback=emit_status,
            )
            selected_workers = tuning["optimal_workers"]
            completed_ids = tuning["completed_game_ids"]
            autotune_summary = _public_autotune_summary(tuning)
        else:
            selected_workers, was_capped, cap_reason = cap_parallel_workers(
                workers,
                safety_config,
            )
            if was_capped:
                emit_status(
                    f"Fixed dataset workers reduced from {workers} to "
                    f"{selected_workers} by resource preflight: {cap_reason}."
                )
            completed_ids = set()
            autotune_summary = {
                "optimal_workers": selected_workers,
                "candidate_workers": [workers],
                "benchmark_fraction": 0.0,
                "minimum_gain": autotune_minimum_gain,
                "games_per_test": 0,
                "reused_game_count": 0,
                "attempts": [],
            }

        missing_specs = (
            (game_index, game_seed(effective_seed, game_index))
            for game_index in range(game_count)
            if game_index not in completed_ids
        )
        _results, execution_info = evaluate_dataset_game_specs(
            game_specs=missing_specs,
            requested_workers=selected_workers,
            result_callback=store_result,
            safety=safety_config,
            retain_results=False,
        )

        row = connection.execute(
            """
            SELECT COUNT(*),
                   COALESCE(SUM(saved_turn_count), 0),
                   COALESCE(SUM(skipped_turn_count), 0)
            FROM games
            """
        ).fetchone()
        produced_games, saved_turn_count, skipped_turn_count = map(int, row)
        if produced_games != game_count:
            raise RuntimeError(
                f"dataset generation produced {produced_games}/{game_count} games"
            )

        staging_fd, staging_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.tmp-",
            dir=output_path.parent,
        )
        with os.fdopen(staging_fd, "w", encoding="utf-8") as stream:
            staging_fd = None
            for (payload,) in connection.execute(
                "SELECT jsonl_payload FROM games ORDER BY game_index"
            ):
                stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging_name, output_path)
        staging_name = None
    finally:
        if connection is not None:
            connection.close()
        if staging_fd is not None:
            os.close(staging_fd)
        if staging_name is not None:
            Path(staging_name).unlink(missing_ok=True)
        Path(database_name).unlink(missing_ok=True)
        if progress_bar is not None:
            progress_bar.close()

    elapsed_time = time.time() - start_time
    if not quiet:
        print("-" * 40)
        print("GENERATION COMPLETE")
        print(f"Real decision pairs: {saved_turn_count}")
        print(f"Forced turns skipped: {skipped_turn_count}")
        print(f"Selected workers: {selected_workers}")
        print(f"Output file: {output_path}")
        print(f"Elapsed time: {format_duration(elapsed_time)}")

    return {
        "game_count": game_count,
        "saved_turn_count": saved_turn_count,
        "skipped_turn_count": skipped_turn_count,
        "output_file": str(output_path),
        "requested_workers": workers,
        "selected_workers": selected_workers,
        "effective_seed": effective_seed,
        "autotune": autotune_summary,
        "parallel": execution_info.to_dict(),
        "duration_s": elapsed_time,
    }


def main(argv=None):
    """Parse CLI options and generate the supervised dataset."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic supervised domino examples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-n", "--games", type=int, default=DEFAULT_GAME_COUNT)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT_FILE))
    parser.add_argument(
        "-j",
        "--workers",
        type=_worker_count,
        default=DEFAULT_DATASET_WORKERS,
        help=f"CPU-only workers or 'auto' (maximum {MAX_PARALLEL_WORKERS}).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_DATASET_SEED)
    parser.add_argument(
        "--autotune-fraction",
        type=float,
        default=DEFAULT_DATASET_AUTOTUNE_FRACTION,
    )
    parser.add_argument(
        "--autotune-min-gain",
        type=float,
        default=DEFAULT_DATASET_MINIMUM_GAIN,
    )
    parser.add_argument("--memory-reserve-mb", type=int, default=512)
    parser.add_argument("--estimated-worker-mb", type=int, default=256)
    parser.add_argument("--max-worker-rss-mb", type=int, default=1024)
    args = parser.parse_args(argv)

    generate_dataset(
        game_count=args.games,
        output_file=args.output,
        workers=args.workers,
        seed=args.seed,
        autotune_fraction=args.autotune_fraction,
        autotune_minimum_gain=args.autotune_min_gain,
        safety_config=ParallelSafetyConfig(
            memory_reserve_mb=args.memory_reserve_mb,
            estimated_worker_mb=args.estimated_worker_mb,
            max_worker_rss_mb=args.max_worker_rss_mb,
        ),
    )


if __name__ == "__main__":
    main()
