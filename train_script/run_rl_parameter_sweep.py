"""
In-process counterpart to ``train_script/run_rl_parameter_sweep.sh``.

The shell script launches a fresh ``python -m training.self_play`` subprocess
per sweep point. Each subprocess pays Python interpreter startup, re-imports
numpy/cupy/agents/middleware, and re-reads the SL checkpoint from disk --
and, on a GPU machine, re-initializes a CUDA context. For a large sweep that
fixed cost per point can dwarf the actual training time, especially with a
small ``--rl-iterations``.

This script does the same grid search (learning_rate x gamma x
games_per_iteration, 3x3x3 = 27 combinations, plus a separate value_coef
axis of 10 values) and the same critic-off-then-on structure, but as a
single persistent process: the SL checkpoint is read from disk exactly once
into memory (``_load_sl_weights_once``) and reused for every one of the 72
default sweep points via ``training.self_play.train(..., sl_weights_data=...)``,
and every point calls ``training.self_play.train()`` and
``diagnostics.pairwise.run_pairwise()`` directly instead of spawning a
subprocess. The final comparative-table stage
(``diagnostics.rl_sweep_table.build_report``) also runs in-process for the
same reason.

Output is identical to the shell script's: the same
``models/rl_test/domino_rl[_critic]_<tag>.npz`` /
``diagnostics/results/domino_rl[_critic]_<tag>/`` naming, the same
``sweep_run.json`` schema, so ``diagnostics/rl_sweep_table.py`` and anything
else that reads that output works unchanged regardless of which driver
produced it.

The shell script is untouched and still works exactly as before; this is an
additional, faster way to run the same sweep, not a replacement.

Usage:
    python train_script/run_rl_parameter_sweep.py
    python train_script/run_rl_parameter_sweep.py --rl-iterations 2000 --resume
    python train_script/run_rl_parameter_sweep.py --device cpu --quiet-training
"""

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from diagnostics.pairwise import run_pairwise
from diagnostics.rl_sweep_table import build_report
from training import self_play
from utils.runtime_status import format_duration

DEFAULT_RL_ITERATIONS = 10
DEFAULT_SL_WEIGHTS_PATH = ROOT / "models" / "domino_sl_weights.npz"
DEFAULT_DIAGNOSTIC_GAMES = 100
DEFAULT_SEED = 42
DEFAULT_MODEL_DIR = ROOT / "models" / "rl_test"
DEFAULT_RESULTS_DIR = ROOT / "diagnostics" / "results"
DEFAULT_REPORT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "rl_sweep_table"
DEFAULT_DEVICE = "cpu"
DEFAULT_RL_WORKERS = "auto"

# Baselines and sweep values mirror train_script/run_rl_parameter_sweep.sh
# exactly -- see that script's header comment for where each range comes
# from (diagnostics/hyperparameter_sweep.py and the historical sweep table
# in references/explicacoes/relatorios/teste_1/plano_correcao.tex).
BASELINE_LEARNING_RATE = 0.001
BASELINE_GAMMA = 1.0
BASELINE_GAMES_PER_ITERATION = 40
BASELINE_VALUE_COEF = 0.5

LR_VALUES = (0.0005, 0.001, 0.005)
GAMMA_VALUES = (1.0, 0.97, 0.9)
GAMES_PER_ITERATION_VALUES = (40, 80, 160)
VALUE_COEF_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def load_sl_weights_once(sl_weights_path):
    """Read the SL checkpoint from disk exactly once, into plain host arrays.

    The returned dict is passed as ``sl_weights_data`` to every
    ``training.self_play.train()`` call below, so
    ``PolicyNetwork.load_from_sl`` never re-reads ``sl_weights_path`` from
    disk. Materializing into a plain dict (rather than keeping the
    ``np.load`` zip file open) keeps this safe to reuse across an arbitrary
    number of training calls.
    """
    with np.load(sl_weights_path, allow_pickle=False) as npz:
        return {name: npz[name] for name in npz.files}


def _grid_tag(lr, gamma, gpi):
    """Return this grid point's run tag: "default" at the exact baseline."""
    if lr == BASELINE_LEARNING_RATE and gamma == BASELINE_GAMMA and gpi == BASELINE_GAMES_PER_ITERATION:
        return "default"
    return f"lr{lr}_gamma{gamma}_gpi{gpi}"


def iter_sweep_points():
    """Yield (tag, learning_rate, gamma, games_per_iteration, value_coef).

    27 grid points (learning_rate x gamma x games_per_iteration) followed by
    9 value_coef points (value_coef varied alone, baseline excluded since
    it's already covered by the grid's "default" point) -- 36 points total,
    each run once per critic setting by ``run_sweep``.
    """
    for lr, gamma, gpi in product(LR_VALUES, GAMMA_VALUES, GAMES_PER_ITERATION_VALUES):
        yield _grid_tag(lr, gamma, gpi), lr, gamma, gpi, BASELINE_VALUE_COEF

    for vc in VALUE_COEF_VALUES:
        if vc == BASELINE_VALUE_COEF:
            continue
        yield f"value_coef_{vc}", BASELINE_LEARNING_RATE, BASELINE_GAMMA, BASELINE_GAMES_PER_ITERATION, vc


def _write_sweep_run_json(diag_dir, **fields):
    """Write sweep_run.json with the same schema as the shell script's version."""
    with open(diag_dir / "sweep_run.json", "w", encoding="utf-8") as f:
        json.dump(fields, f, indent=2, ensure_ascii=False)


def run_sweep_point(
    *,
    tag,
    critic_enabled,
    learning_rate,
    gamma,
    games_per_iteration,
    value_coef,
    rl_iterations,
    diagnostic_games,
    seed,
    model_dir,
    results_dir,
    sl_weights_path,
    sl_weights_data,
    device,
    rl_workers,
    resume,
    diag_plots,
    quiet_training,
):
    """Train (or reuse) and diagnose one sweep point. Returns its run name."""
    name = "domino_rl"
    if critic_enabled:
        name += "_critic"
    name += f"_{tag}"

    model_path = Path(model_dir) / f"{name}.npz"
    diag_dir = Path(results_dir) / name
    diag_dir.mkdir(parents=True, exist_ok=True)
    critic_label = "on" if critic_enabled else "off"

    if resume and model_path.exists():
        print(f"[{name}] training skipped (--resume, {model_path} already exists)")
    else:
        print(
            f"[{name}] RL training: {rl_iterations} iterations, "
            f"lr={learning_rate} gamma={gamma} games/iter={games_per_iteration} "
            f"value_coef={value_coef} critic={critic_label} -> {model_path}"
        )
        start_time = time.time()
        self_play.train(
            iterations=rl_iterations,
            games_per_iteration=games_per_iteration,
            learning_rate=learning_rate,
            gamma=gamma,
            value_coef=value_coef,
            sl_weights_path=str(sl_weights_path),
            sl_weights_data=sl_weights_data,
            rl_weights_path=str(model_path),
            seed=seed,
            device=device,
            workers=rl_workers,
            use_value_head=critic_enabled,
            quiet=quiet_training,
        )
        print(f"[{name}] training complete in {format_duration(time.time() - start_time)}")

    print(f"[{name}] diagnostics: rl vs random ({diagnostic_games} games) -> {diag_dir}/")
    run_pairwise(
        "rl",
        "random",
        game_count=diagnostic_games,
        weights=model_path,
        seed=seed,
        output_dir=diag_dir,
        generate_plots=diag_plots,
        print_console_summary=not quiet_training,
        print_memory_summary=False,
    )

    _write_sweep_run_json(
        diag_dir,
        run_name=name,
        varied_parameter=tag,
        critic_enabled=critic_enabled,
        learning_rate=learning_rate,
        gamma=gamma,
        games_per_iteration=games_per_iteration,
        value_coef=value_coef,
        rl_iterations=rl_iterations,
        rl_workers=rl_workers,
        seed=seed,
        diagnostic_games=diagnostic_games,
        sl_weights_path=str(sl_weights_path),
        model_path=str(model_path),
    )
    return name


def run_sweep(
    *,
    rl_iterations=DEFAULT_RL_ITERATIONS,
    sl_weights_path=DEFAULT_SL_WEIGHTS_PATH,
    diagnostic_games=DEFAULT_DIAGNOSTIC_GAMES,
    seed=DEFAULT_SEED,
    model_dir=DEFAULT_MODEL_DIR,
    results_dir=DEFAULT_RESULTS_DIR,
    resume=False,
    diag_plots=True,
    device=DEFAULT_DEVICE,
    rl_workers=DEFAULT_RL_WORKERS,
    report_output_dir=DEFAULT_REPORT_OUTPUT_DIR,
    skip_report=False,
    quiet_training=False,
):
    """Run the full grid + value_coef sweep for both critic settings.

    Returns the list of run names produced.
    """
    model_dir = Path(model_dir)
    results_dir = Path(results_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    sweep_points = list(iter_sweep_points())
    total_points = len(sweep_points) * 2
    print(
        f"RL parameter sweep: {total_points} training+diagnostics runs "
        f"({rl_iterations} iterations each: {len(sweep_points)}-point grid+value_coef "
        "x 2 critic settings)"
    )

    print(f"Loading SL weights once: {sl_weights_path}")
    sl_weights_data = load_sl_weights_once(sl_weights_path)

    run_names = []
    for critic_enabled in (False, True):
        for tag, lr, gamma, gpi, vc in sweep_points:
            run_names.append(
                run_sweep_point(
                    tag=tag,
                    critic_enabled=critic_enabled,
                    learning_rate=lr,
                    gamma=gamma,
                    games_per_iteration=gpi,
                    value_coef=vc,
                    rl_iterations=rl_iterations,
                    diagnostic_games=diagnostic_games,
                    seed=seed,
                    model_dir=model_dir,
                    results_dir=results_dir,
                    sl_weights_path=sl_weights_path,
                    sl_weights_data=sl_weights_data,
                    device=device,
                    rl_workers=rl_workers,
                    resume=resume,
                    diag_plots=diag_plots,
                    quiet_training=quiet_training,
                )
            )

    if skip_report:
        print("\nComparative table skipped (--skip-report).")
    else:
        print(f"\nComparative table: parsing sweep_run.json + summary.json for every run -> {report_output_dir}/")
        build_report(results_dir=results_dir, output_dir=report_output_dir)

    return run_names


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "In-process RL hyperparameter sweep: loads the SL checkpoint once and "
            "reuses it for every sweep point, instead of spawning a subprocess per "
            "point like train_script/run_rl_parameter_sweep.sh does."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rl-iterations", type=int, default=DEFAULT_RL_ITERATIONS)
    parser.add_argument("--sl-weights-path", default=str(DEFAULT_SL_WEIGHTS_PATH))
    parser.add_argument("--diagnostic-games", type=int, default=DEFAULT_DIAGNOSTIC_GAMES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip training a checkpoint that already exists on disk; still (re)run its diagnostics.",
    )
    parser.add_argument(
        "--diag-no-plots",
        action="store_true",
        help="Skip the per-run diagnostic PNG plots (CSV/JSON are always written).",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument(
        "--rl-workers",
        type=self_play.parse_rl_worker_count,
        default=DEFAULT_RL_WORKERS,
        help="CPU-only rollout workers for each sequential sweep point.",
    )
    parser.add_argument("--report-output-dir", default=str(DEFAULT_REPORT_OUTPUT_DIR))
    parser.add_argument(
        "--skip-report", action="store_true", help="Skip the final comparative-table stage."
    )
    parser.add_argument(
        "--quiet-training",
        action="store_true",
        help="Suppress training.self_play's per-iteration logs and diagnostics.pairwise's "
        "per-matchup console summary (off by default, matching the shell script).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    start_time = time.time()

    run_sweep(
        rl_iterations=args.rl_iterations,
        sl_weights_path=Path(args.sl_weights_path),
        diagnostic_games=args.diagnostic_games,
        seed=args.seed,
        model_dir=Path(args.model_dir),
        results_dir=Path(args.results_dir),
        resume=args.resume,
        diag_plots=not args.diag_no_plots,
        device=args.device,
        rl_workers=args.rl_workers,
        report_output_dir=Path(args.report_output_dir),
        skip_report=args.skip_report,
        quiet_training=args.quiet_training,
    )

    elapsed = time.time() - start_time
    print(f"\nRL parameter sweep complete. Total elapsed execution time: {format_duration(elapsed)}")


if __name__ == "__main__":
    main()
