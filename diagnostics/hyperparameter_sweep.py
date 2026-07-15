"""Diagnostic benchmark that sweeps RL hyperparameters and logs win rates.

For each sweep point this module trains a fresh RL checkpoint with
``training.self_play`` (starting from the existing supervised checkpoint),
then evaluates ``heuristic``, ``neural``, and the freshly trained ``rl``
policy against a uniform random agent and in self-play (the same agent
mirrored against itself), using ``diagnostics.pairwise.evaluate_pair``.

Every record — the exact RL hyperparameters used plus every matchup's
win/draw/loss rates — is appended to a single JSON array on disk
(``--output``, default ``diagnostics/results/hyperparameter_sweep.json``), so
repeated invocations accumulate a growing benchmark log instead of
overwriting it.

The sweep isolates one hyperparameter axis at a time (learning rate, reward
schema, gamma discount), holding the other two at their baseline value, and
runs the full three-axis sweep twice: once with the actor-critic value head
on, once with it off (direct REINFORCE) — see ``training/self_play.py``.

Usage:
    python -m diagnostics.hyperparameter_sweep
    python -m diagnostics.hyperparameter_sweep --rl-iterations 300 --diagnostic-games 1000
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.pairwise import DEFAULT_GAME_COUNT, evaluate_pair
from training import self_play
from utils.runtime_status import format_duration

DEFAULT_OUTPUT = ROOT / "diagnostics" / "results" / "hyperparameter_sweep.json"
DEFAULT_CHECKPOINT_DIR = ROOT / "models" / "hyperparameter_sweep"
DEFAULT_SL_WEIGHTS = ROOT / "models" / "domino_sl_weights.npz"

BASELINE_LEARNING_RATE = 0.001
BASELINE_GAMMA = 1.0
BASELINE_REWARD_SCHEMA = "default"
BASELINE_VALUE_COEF = 0.5

DEFAULT_LR_VALUES = (0.0005, 0.001, 0.005)
DEFAULT_GAMMA_VALUES = (1.0, 0.97, 0.9)
DEFAULT_REWARD_SCHEMAS = tuple(self_play.REWARD_SCHEMAS)

DEFAULT_RL_ITERATIONS = 150
DEFAULT_RL_GAMES_PER_ITERATION = 40
DEFAULT_DIAGNOSTIC_GAMES = 500


def _match_rates(agent_name, opponent_name, game_count, weights=None, opponent_weights=None, seed=None):
    """Play one matchup and return compact win/draw/loss rates."""
    games = evaluate_pair(
        agent_name,
        opponent_name,
        game_count=game_count,
        weights=weights,
        opponent_weights=opponent_weights,
        seed=seed,
        suppress_agent_output=True,
    )
    counts = Counter(game["result"] for game in games)
    total = len(games)
    return {
        "games": total,
        "win_rate": counts.get("win", 0) / total,
        "draw_rate": counts.get("draw", 0) / total,
        "loss_rate": counts.get("loss", 0) / total,
    }


def _static_benchmarks(game_count, sl_weights_path, seed):
    """Evaluate heuristic and neural agents vs random and in self-play once.

    These do not depend on RL hyperparameters, so they are computed a single
    time and reused across every sweep record.
    """
    return {
        "heuristic_vs_random": _match_rates("heuristic", "random", game_count, seed=seed),
        "heuristic_self_play": _match_rates("heuristic", "heuristic", game_count, seed=seed),
        "neural_vs_random": _match_rates(
            "neural", "random", game_count, weights=sl_weights_path, seed=seed
        ),
        "neural_self_play": _match_rates(
            "neural",
            "neural",
            game_count,
            weights=sl_weights_path,
            opponent_weights=sl_weights_path,
            seed=seed,
        ),
    }


def _train_rl_checkpoint(
    sl_weights_path,
    rl_weights_path,
    *,
    iterations,
    games_per_iteration,
    learning_rate,
    gamma,
    reward_schema,
    use_value_head,
    value_coef,
    quiet=True,
):
    """Train one sweep-point RL checkpoint from the shared SL weights."""
    return self_play.train(
        iterations=iterations,
        games_per_iteration=games_per_iteration,
        learning_rate=learning_rate,
        sl_weights_path=str(sl_weights_path),
        rl_weights_path=str(rl_weights_path),
        use_value_head=use_value_head,
        value_coef=value_coef,
        gamma=gamma,
        reward_schema=reward_schema,
        quiet=quiet,
    )


def _append_json_record(path, record):
    """Append one record to a JSON array on disk, creating it if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path} does not contain a JSON array; refusing to append.")
    else:
        data = []
    data.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _sweep_axes(lr_values, gamma_values, reward_schemas):
    """Return (axis_name, values, other_axes_held_at_baseline) for one-at-a-time sweeps."""
    return (
        (
            "learning_rate",
            lr_values,
            {"gamma": BASELINE_GAMMA, "reward_schema": BASELINE_REWARD_SCHEMA},
        ),
        (
            "reward_schema",
            reward_schemas,
            {"learning_rate": BASELINE_LEARNING_RATE, "gamma": BASELINE_GAMMA},
        ),
        (
            "gamma",
            gamma_values,
            {"learning_rate": BASELINE_LEARNING_RATE, "reward_schema": BASELINE_REWARD_SCHEMA},
        ),
    )


def run_sweep(
    *,
    sl_weights_path=DEFAULT_SL_WEIGHTS,
    checkpoint_dir=DEFAULT_CHECKPOINT_DIR,
    output_json=DEFAULT_OUTPUT,
    rl_iterations=DEFAULT_RL_ITERATIONS,
    rl_games_per_iteration=DEFAULT_RL_GAMES_PER_ITERATION,
    diagnostic_games=DEFAULT_DIAGNOSTIC_GAMES,
    lr_values=DEFAULT_LR_VALUES,
    gamma_values=DEFAULT_GAMMA_VALUES,
    reward_schemas=DEFAULT_REWARD_SCHEMAS,
    value_coef=BASELINE_VALUE_COEF,
    seed=None,
    quiet_training=True,
):
    """Run the full one-axis-at-a-time sweep, critic on then critic off.

    Appends one JSON record per sweep point to ``output_json`` as it goes, so
    a run that is interrupted partway still leaves completed points on disk.
    Returns the list of records produced by this call.
    """
    sl_weights_path = Path(sl_weights_path)
    checkpoint_dir = Path(checkpoint_dir)
    output_json = Path(output_json)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Static baselines: heuristic/neural vs random and self-play "
        f"({diagnostic_games} games per matchup)..."
    )
    static_benchmarks = _static_benchmarks(diagnostic_games, sl_weights_path, seed)

    axes = _sweep_axes(lr_values, gamma_values, reward_schemas)
    records = []

    for critic_enabled in (True, False):
        critic_label = "critic_on" if critic_enabled else "critic_off"
        for axis_name, axis_values, held_constants in axes:
            for value in axis_values:
                hyperparameters = {
                    "learning_rate": BASELINE_LEARNING_RATE,
                    "gamma": BASELINE_GAMMA,
                    "reward_schema": BASELINE_REWARD_SCHEMA,
                }
                hyperparameters.update(held_constants)
                hyperparameters[axis_name] = value

                tag = f"{critic_label}_{axis_name}_{value}"
                checkpoint_path = checkpoint_dir / f"rl_{tag}.npz"

                print(
                    f"\n[{tag}] training {rl_iterations} RL iterations "
                    f"(lr={hyperparameters['learning_rate']}, "
                    f"gamma={hyperparameters['gamma']}, "
                    f"reward_schema={hyperparameters['reward_schema']}, "
                    f"critic={'on' if critic_enabled else 'off'})"
                )
                start_time = time.time()
                _train_rl_checkpoint(
                    sl_weights_path,
                    checkpoint_path,
                    iterations=rl_iterations,
                    games_per_iteration=rl_games_per_iteration,
                    learning_rate=hyperparameters["learning_rate"],
                    gamma=hyperparameters["gamma"],
                    reward_schema=hyperparameters["reward_schema"],
                    use_value_head=critic_enabled,
                    value_coef=value_coef,
                    quiet=quiet_training,
                )
                train_duration_s = time.time() - start_time

                rl_vs_random = _match_rates(
                    "rl", "random", diagnostic_games, weights=checkpoint_path, seed=seed
                )
                rl_self_play = _match_rates(
                    "rl",
                    "rl",
                    diagnostic_games,
                    weights=checkpoint_path,
                    opponent_weights=checkpoint_path,
                    seed=seed,
                )

                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "swept_axis": axis_name,
                    "critic_enabled": critic_enabled,
                    "hyperparameters": {
                        "learning_rate": hyperparameters["learning_rate"],
                        "gamma": hyperparameters["gamma"],
                        "reward_schema": hyperparameters["reward_schema"],
                        "value_coef": value_coef if critic_enabled else None,
                        "rl_iterations": rl_iterations,
                        "rl_games_per_iteration": rl_games_per_iteration,
                    },
                    "rl_checkpoint": str(checkpoint_path),
                    "rl_training_duration_s": train_duration_s,
                    "diagnostic_games_per_matchup": diagnostic_games,
                    "results": {
                        **static_benchmarks,
                        "rl_vs_random": rl_vs_random,
                        "rl_self_play": rl_self_play,
                    },
                }
                records.append(record)
                _append_json_record(output_json, record)

                print(
                    f"[{tag}] done in {format_duration(train_duration_s)} | "
                    f"rl_vs_random win_rate={rl_vs_random['win_rate']:.1%} | "
                    f"rl_self_play win_rate={rl_self_play['win_rate']:.1%}"
                )

    print(f"\nSweep complete: {len(records)} records appended to {output_json}")
    return records


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Sweep RL hyperparameters (learning rate, reward schema, gamma) one "
            "axis at a time, with the critic on and then off, benchmarking "
            "heuristic/neural/rl vs random and in self-play at each point."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sl-weights-path", default=str(DEFAULT_SL_WEIGHTS))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--rl-iterations", type=int, default=DEFAULT_RL_ITERATIONS)
    parser.add_argument(
        "--rl-games-per-iteration", type=int, default=DEFAULT_RL_GAMES_PER_ITERATION
    )
    parser.add_argument(
        "--diagnostic-games",
        type=int,
        default=DEFAULT_DIAGNOSTIC_GAMES,
        help=f"Games per matchup (evaluate.py default is {DEFAULT_GAME_COUNT}; kept smaller here for sweep speed).",
    )
    parser.add_argument("--value-coef", type=float, default=BASELINE_VALUE_COEF)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr-values", type=float, nargs="+", default=list(DEFAULT_LR_VALUES))
    parser.add_argument("--gamma-values", type=float, nargs="+", default=list(DEFAULT_GAMMA_VALUES))
    parser.add_argument(
        "--reward-schemas",
        nargs="+",
        choices=tuple(self_play.REWARD_SCHEMAS),
        default=list(DEFAULT_REWARD_SCHEMAS),
    )
    parser.add_argument(
        "--verbose-training",
        action="store_true",
        help="Print full training.self_play logs for every sweep point (default: quiet).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_sweep(
        sl_weights_path=Path(args.sl_weights_path),
        checkpoint_dir=Path(args.checkpoint_dir),
        output_json=Path(args.output),
        rl_iterations=args.rl_iterations,
        rl_games_per_iteration=args.rl_games_per_iteration,
        diagnostic_games=args.diagnostic_games,
        lr_values=args.lr_values,
        gamma_values=args.gamma_values,
        reward_schemas=args.reward_schemas,
        value_coef=args.value_coef,
        seed=args.seed,
        quiet_training=not args.verbose_training,
    )


if __name__ == "__main__":
    main()
