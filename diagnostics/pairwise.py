"""
Evaluate one domino agent against another and write compact diagnostics.

This module is the pairwise diagnostics helper. The high-level entry point
``diagnostics.evaluate`` calls this module repeatedly to build a complete
all-pairs matrix, but this file can still be executed directly when only one
matchup is needed:

    python -m diagnostics.pairwise --agent heuristic --opponent random

The evaluated agent alternates between player 0 and player 1 so the result is
not dominated by first-player advantage. Outputs are written to a dedicated
folder containing JSON, CSV, and optional PNG plots.
"""

import argparse
import contextlib
import csv
import io
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from diagnostics.plots import LABEL, generate_plots as write_diagnostic_plots, summarize
from middleware.domino_engine import DominoEngine
from utils.runtime_status import format_duration, print_memory_report

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

DEFAULT_AGENT = "rl"
DEFAULT_OPPONENT = "heuristic"
DEFAULT_GAME_COUNT = 10000
DEFAULT_SEED = None
DEFAULT_AGENT_WEIGHTS = None
DEFAULT_OPPONENT_WEIGHTS = None
DEFAULT_OUTPUT_DIR = None
DEFAULT_GENERATE_PLOTS = True

CANONICAL_AGENTS = ("rl", "neural", "heuristic", "random")
LEGACY_AGENT_ALIASES = {"sl": "neural"}
AVAILABLE_AGENTS = CANONICAL_AGENTS + tuple(LEGACY_AGENT_ALIASES)

DEFAULT_WEIGHTS = {
    "rl": ROOT / "models" / "domino_rl_weights.npz",
    "neural": ROOT / "models" / "domino_sl_weights.npz",
}


def normalize_agent_name(agent_name):
    """Return the canonical diagnostics name for an agent."""
    normalized = agent_name.strip().lower()
    normalized = LEGACY_AGENT_ALIASES.get(normalized, normalized)
    if normalized not in CANONICAL_AGENTS:
        raise ValueError(f"Unknown agent {agent_name!r}. Options: {CANONICAL_AGENTS}")
    return normalized


def resolve_weights_path(agent_name, weights_path=None):
    """Return an existing checkpoint path for a checkpoint-backed agent."""
    agent_name = normalize_agent_name(agent_name)
    if agent_name not in DEFAULT_WEIGHTS:
        return None

    path = Path(weights_path or DEFAULT_WEIGHTS[agent_name])
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {agent_name} checkpoint: {path}. "
            "Generate the model first or pass an explicit weights path."
        )
    return path


def create_agent(agent_name, weights_path=None):
    """Create an agent by name, importing checkpoint-backed classes only when used."""
    agent_name = normalize_agent_name(agent_name)

    if agent_name == "rl":
        from agents.rl_agent import RLAgent

        return RLAgent.load(str(resolve_weights_path("rl", weights_path)), mode="evaluation")
    if agent_name == "neural":
        from agents.neural_agent import NeuralAgent

        return NeuralAgent.load(str(resolve_weights_path("neural", weights_path)))
    if agent_name == "heuristic":
        from agents.heuristic_agent import StrategicAgent

        return StrategicAgent()
    if agent_name == "random":
        from agents.agent import RandomAgent

        return RandomAgent()

    raise ValueError(f"Unknown agent {agent_name!r}. Options: {CANONICAL_AGENTS}")


def is_forced_draw(legal_actions):
    """Return True when the only legal action is drawing from the stock."""
    return len(legal_actions) == 1 and legal_actions[0] == ("DRAW", None)


def is_forced_pass(legal_actions):
    """Return True when the only legal action is passing."""
    return len(legal_actions) == 1 and legal_actions[0] is None


def count_tile_play_options(legal_actions):
    """Count voluntary tile-play options, excluding forced draw/pass."""
    return sum(
        1
        for action in legal_actions
        if action is not None and action != ("DRAW", None)
    )


def empty_choice_stats():
    """Return counters for how often the evaluated agent really had a choice."""
    return {
        "agent_real_decision_turns": 0,
        "agent_forced_tile_turns": 0,
        "agent_forced_draws": 0,
        "agent_forced_passes": 0,
        "agent_choice_histogram": {},
    }


def update_choice_stats(stats, legal_actions):
    """Update choice counters from the evaluated agent's legal actions."""
    if is_forced_draw(legal_actions):
        stats["agent_forced_draws"] += 1
        return

    if is_forced_pass(legal_actions):
        stats["agent_forced_passes"] += 1
        return

    option_count = count_tile_play_options(legal_actions)
    stats["agent_choice_histogram"][str(option_count)] = (
        stats["agent_choice_histogram"].get(str(option_count), 0) + 1
    )

    if option_count >= 2:
        stats["agent_real_decision_turns"] += 1
    else:
        stats["agent_forced_tile_turns"] += 1


def play_game(agent, opponent, agent_position, suppress_agent_output=True):
    """Play one game and return the outcome from the evaluated agent's view."""
    agents = [None, None]
    agents[agent_position] = agent
    agents[1 - agent_position] = opponent

    engine = DominoEngine(player_count=2)
    choice_stats = empty_choice_stats()

    while not engine.game_over:
        state = engine._get_state()
        current_player = state["current_player"]
        legal_actions = engine.valid_actions(current_player)

        if current_player == agent_position:
            update_choice_stats(choice_stats, legal_actions)

        if suppress_agent_output:
            with contextlib.redirect_stdout(io.StringIO()):
                action = agents[current_player].choose_move(state, legal_actions)
        else:
            action = agents[current_player].choose_move(state, legal_actions)

        engine.step(action)

    final_state = engine.to_dict()
    winner = final_state["winner"]

    if winner == -1:
        result = "draw"
    elif winner == agent_position:
        result = "win"
    else:
        result = "loss"

    pips = [sum(tile[0] + tile[1] for tile in hand) for hand in final_state["hands"]]

    return {
        "game": None,
        "agent_position": agent_position,
        "result": result,
        "turns": final_state["turn"],
        "agent_remaining_pips": pips[agent_position],
        "opponent_remaining_pips": pips[1 - agent_position],
        **choice_stats,
    }


def evaluate_pair(
    agent_name,
    opponent_name,
    game_count=DEFAULT_GAME_COUNT,
    weights=None,
    opponent_weights=None,
    seed=None,
    progress_callback=None,
    suppress_agent_output=True,
):
    """Run a batch of games while alternating the evaluated agent's position."""
    agent_name = normalize_agent_name(agent_name)
    opponent_name = normalize_agent_name(opponent_name)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    agent = create_agent(agent_name, weights)
    opponent = create_agent(opponent_name, opponent_weights)

    games = []
    for i in range(game_count):
        record = play_game(
            agent,
            opponent,
            agent_position=i % 2,
            suppress_agent_output=suppress_agent_output,
        )
        record["game"] = i + 1
        games.append(record)
        if progress_callback:
            progress_callback(i + 1, game_count)
    return games


# Backward-compatible name for older imports.
evaluate = evaluate_pair


def save_csv(games, path):
    """Write compact per-game records to CSV."""
    fields = [
        "game",
        "agent_position",
        "result",
        "turns",
        "agent_remaining_pips",
        "opponent_remaining_pips",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: game[field] for field in fields} for game in games)


def add_choice_summary(summary, games):
    """Attach aggregate choice-opportunity statistics to a pairwise summary."""
    histogram = {}
    for game in games:
        for option_count, count in game["agent_choice_histogram"].items():
            histogram[option_count] = histogram.get(option_count, 0) + count

    evaluated_turns = sum(
        game["agent_real_decision_turns"]
        + game["agent_forced_tile_turns"]
        + game["agent_forced_draws"]
        + game["agent_forced_passes"]
        for game in games
    )

    real_decisions = sum(game["agent_real_decision_turns"] for game in games)

    summary["choice_opportunities"] = {
        "evaluated_agent_turns": evaluated_turns,
        "real_decision_turns": real_decisions,
        "real_decision_rate": real_decisions / evaluated_turns if evaluated_turns else 0.0,
        "forced_tile_turns": sum(game["agent_forced_tile_turns"] for game in games),
        "forced_draws": sum(game["agent_forced_draws"] for game in games),
        "forced_passes": sum(game["agent_forced_passes"] for game in games),
        "choice_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
    }
    return summary


def print_summary(summary, duration_s):
    """Print the main pairwise metrics in a compact console format."""
    game_count = summary["game_count"]
    games_per_second = game_count / duration_s if duration_s else float("inf")
    print(f"\n===== Diagnostics: {summary['agent']} vs {summary['opponent']} =====")
    print(
        f"Games: {game_count} | time: {format_duration(duration_s)} "
        f"({games_per_second:.1f} games/s)"
    )
    for key in ("win", "draw", "loss"):
        rate = summary["rates"][key]
        print(f"  {LABEL[key]:<8} {summary['counts'][key]:>5}  ({rate:6.1%})")
    lo, hi = summary["win_ci95"]
    print(f"  95% CI (win rate): [{lo:.1%}, {hi:.1%}]")
    for position in ("0", "1"):
        info = summary["by_position"][position]
        print(
            f"  As player {position}: {info['win_rate']:.1%} wins "
            f"across {info['games']} games"
        )
    print(f"  Turns per game: {summary['mean_turns']:.1f} +/- {summary['std_turns']:.1f}")
    print(
        "  Remaining pips (mean): "
        f"agent {summary['mean_agent_remaining_pips']:.1f} | "
        f"opponent {summary['mean_opponent_remaining_pips']:.1f}"
    )
    choice_info = summary.get("choice_opportunities")
    if choice_info:
        print(
            "  Choice opportunities: "
            f"{choice_info['real_decision_turns']}/"
            f"{choice_info['evaluated_agent_turns']} turns "
            f"({choice_info['real_decision_rate']:.1%})"
        )
        print(
            "  Forced turns: "
            f"tile {choice_info['forced_tile_turns']}, "
            f"draw {choice_info['forced_draws']}, "
            f"pass {choice_info['forced_passes']}"
        )
        print(f"  Choice histogram: {choice_info['choice_histogram']}")


def run_pairwise(
    agent_name,
    opponent_name,
    game_count=DEFAULT_GAME_COUNT,
    weights=None,
    opponent_weights=None,
    seed=None,
    output_dir=None,
    generate_plots=DEFAULT_GENERATE_PLOTS,
    print_console_summary=True,
    suppress_agent_output=True,
    print_memory_summary=True,
):
    """Run one matchup and write the standard pairwise artifacts."""
    agent_name = normalize_agent_name(agent_name)
    opponent_name = normalize_agent_name(opponent_name)
    output_dir = output_dir or ROOT / "diagnostics" / "results" / "pairwise" / f"{agent_name}_vs_{opponent_name}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Evaluating {agent_name} vs {opponent_name} over {game_count} games "
        "(starting position alternates every game)"
    )
    if print_console_summary and print_memory_summary:
        print_memory_report("Diagnostics startup memory")

    progress_bar = None
    if print_console_summary and tqdm is not None:
        progress_bar = tqdm(
            total=game_count,
            desc=f"{agent_name} vs {opponent_name}",
            unit="game",
            leave=True,
        )

    def progress(_done, _total):
        if progress_bar is not None:
            progress_bar.update(1)

    start_time = time.time()
    try:
        games = evaluate_pair(
            agent_name,
            opponent_name,
            game_count,
            weights=weights,
            opponent_weights=opponent_weights,
            seed=seed,
            progress_callback=progress,
            suppress_agent_output=suppress_agent_output,
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    duration = time.time() - start_time

    summary = summarize(games, agent_name, opponent_name, seed)
    summary = add_choice_summary(summary, games)
    summary["duration_s"] = duration
    if print_console_summary:
        print_summary(summary, duration)

    save_csv(games, output_dir / "games.csv")
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if generate_plots:
        write_diagnostic_plots(games, summary, output_dir)

    if print_console_summary:
        print(f"\nResults saved in {output_dir}/")
        if generate_plots:
            print(
                "  cumulative_rates.png, result_distribution.png, wins_by_position.png, "
                "game_lengths.png, choice_opportunities.png"
            )
        print("  games.csv, summary.json")

    return {
        "summary": summary,
        "games": games,
        "output_dir": str(output_dir),
        "duration_s": duration,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one domino agent against another over N games.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--agent", choices=AVAILABLE_AGENTS, default=DEFAULT_AGENT)
    parser.add_argument("--opponent", choices=AVAILABLE_AGENTS, default=DEFAULT_OPPONENT)
    parser.add_argument(
        "-n",
        "--games",
        type=int,
        default=DEFAULT_GAME_COUNT,
        help="Number of games to play in this matchup.",
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_AGENT_WEIGHTS)
    parser.add_argument("--opponent-weights", type=Path, default=DEFAULT_OPPONENT_WEIGHTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder. Defaults to diagnostics/results/pairwise/<agent>_vs_<opponent>.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=not DEFAULT_GENERATE_PLOTS,
        help="Skip PNG generation and write only CSV/JSON plus console output.",
    )
    parser.add_argument(
        "--show-agent-output",
        action="store_true",
        help="Do not suppress print calls made by the agents during evaluation.",
    )
    args = parser.parse_args()

    run_pairwise(
        args.agent,
        args.opponent,
        game_count=args.games,
        weights=args.weights,
        opponent_weights=args.opponent_weights,
        seed=args.seed,
        output_dir=args.output,
        generate_plots=not args.no_plots,
        suppress_agent_output=not args.show_agent_output,
    )


if __name__ == "__main__":
    main()
