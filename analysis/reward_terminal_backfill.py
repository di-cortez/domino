#!/usr/bin/env python3
"""Reconstruct terminal-outcome diagnostics for a run that predates them.

``training_metrics.jsonl`` only carries the empty-hand/blocked split from
schema v8 onward. A run recorded before that still leaves the evidence behind:
its checkpoint archive retains the policy every few iterations, so replaying
each retained checkpoint through the production rollout path recovers how the
outcome split, the pip margin, and the terminal/immediate mixture moved across
training.

What this is not: a record of the games that were actually played. Each
checkpoint is replayed against itself rather than against the opponent pool it
faced at that point in the run, and with fresh seeds. It therefore measures how
the policy *behaves* at each stage -- which is what the reward diagnostics are
about -- and not the historical matchmaking. Numbers here will not match a v8
metrics file row for row, and are not a substitute for recording the columns
during the run.

Usage:
    python -m analysis.reward_terminal_backfill models/rl/<run> --games 600
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agents.rl_agent import RLAgent
from agents.rl_nn import PolicyNetwork
from middleware.domino_engine import DominoEngine
from training.rl.reward_distance import resolve_reward_distance_mode
from training.rl.reward_model import DEFAULT_GAMMA_F
from training.rl.rollout import (
    DEFAULT_REWARD_SCHEMA,
    EventStats,
    TerminalStats,
    _event_reward_for_action,
    _finish_episode_with_rewards,
    _terminal_outcome,
    _tile_play_actions,
)


def replay_checkpoint(weights_path, *, ruleset_name, schema, gamma_f, games, seed):
    """Play ``games`` self-play games with one checkpoint and aggregate them."""
    network = PolicyNetwork.load(str(weights_path), device="cpu")
    local_metric, _terminal_metric = resolve_reward_distance_mode(
        schema["reward_distance_mode"]
    )
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)

    totals = TerminalStats()
    terminal_halves = []
    immediate_halves = []
    for _ in range(games):
        engine = DominoEngine(player_count=2, ruleset=ruleset_name)
        learner_position = random.randint(0, 1)
        learner = RLAgent(network, mode="training", ruleset=ruleset_name)
        agents = [None, None]
        agents[learner_position] = learner
        agents[1 - learner_position] = RLAgent(
            network, mode="stochastic_evaluation", ruleset=ruleset_name
        )
        event_stats = EventStats()
        while not engine.game_over:
            state = engine._get_state()
            current = state["current_player"]
            legal = engine.valid_actions(current)
            tiles = _tile_play_actions(legal)
            if current == learner_position and len(tiles) == 1:
                action = tiles[0]
            else:
                action = agents[current].choose_move(state, legal)
            event_reward = _event_reward_for_action(
                current, learner_position, action, event_stats, schema
            )
            if event_reward is not None:
                learner.add_decayed_event_reward(
                    event_turn=state["turn"],
                    base_reward=event_reward,
                    decay_lambda=schema["gamma_i"],
                    distance_metric=local_metric,
                )
            engine.step(action, return_state=False, legal_actions=legal)

        utility, stats = _terminal_outcome(engine, learner_position, schema)
        totals.add(stats)
        for sample in _finish_episode_with_rewards(
            learner,
            utility,
            gamma_f,
            schema["reward_eta"],
            terminal_turn=engine.turn,
            reward_distance_mode=schema["reward_distance_mode"],
        ):
            terminal_halves.append(abs(sample.terminal_reward))
            immediate_halves.append(abs(sample.local_reward))

    empty_games = totals.empty_hand_wins + totals.empty_hand_losses
    blocked_games = totals.blocked_wins + totals.blocked_losses
    decided = empty_games + blocked_games
    terminal_abs = float(np.mean(terminal_halves)) if terminal_halves else 0.0
    immediate_abs = float(np.mean(immediate_halves)) if immediate_halves else 0.0
    magnitude = terminal_abs + immediate_abs
    return {
        "empty_hand_games": empty_games,
        "blocked_games": blocked_games,
        "blocked_share": blocked_games / decided if decided else 0.0,
        "empty_hand_win_rate": (
            totals.empty_hand_wins / empty_games if empty_games else 0.0
        ),
        "blocked_win_rate": (
            totals.blocked_wins / blocked_games if blocked_games else 0.0
        ),
        "mean_pip_margin": (
            totals.blocked_margin_sum / blocked_games if blocked_games else 0.0
        ),
        "mean_blocked_magnitude": (
            totals.blocked_magnitude_sum / blocked_games if blocked_games else 0.0
        ),
        "terminal_abs_mean": terminal_abs,
        "local_abs_mean": immediate_abs,
        "immediate_share": immediate_abs / magnitude if magnitude else 0.0,
    }


def retained_checkpoints(run_dir, stride):
    """Return every ``stride``-th archived checkpoint, oldest first."""
    manifest_path = Path(run_dir) / "checkpoint_archive" / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"No checkpoint archive at {manifest_path}. Without retained "
            "checkpoints a finished run cannot be reconstructed; only its "
            "final weights remain."
        )
    with open(manifest_path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    records = sorted(
        manifest["checkpoints"], key=lambda item: item["completed_iteration"]
    )
    return records[::stride]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="RL run directory holding checkpoint_archive/")
    parser.add_argument("--ruleset", default="double-three")
    parser.add_argument(
        "--games", type=int, default=600,
        help="self-play games replayed per checkpoint (default: 600)",
    )
    parser.add_argument(
        "--stride", type=int, default=100,
        help="replay every Nth retained checkpoint (default: 100)",
    )
    parser.add_argument("--gamma-f", type=float, default=DEFAULT_GAMMA_F)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--json-out", help="also write the rows to this JSON file")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    schema = dict(DEFAULT_REWARD_SCHEMA)
    checkpoints = retained_checkpoints(run_dir, args.stride)
    print(
        f"Replaying {len(checkpoints)} checkpoints x {args.games} self-play "
        f"games ({args.ruleset}).\n"
    )
    header = (
        f"{'games':>12} {'blocked%':>9} {'emptyWR':>8} {'blockWR':>8} "
        f"{'margin':>7} {'mean_m':>7} {'|G_T|':>7} {'|G_I|':>7} {'imm%':>6}"
    )
    print(header)

    rows = []
    for record in checkpoints:
        summary = replay_checkpoint(
            run_dir / "checkpoint_archive" / record["filename"],
            ruleset_name=args.ruleset,
            schema=schema,
            gamma_f=args.gamma_f,
            games=args.games,
            seed=args.seed,
        )
        summary["completed_rl_games"] = int(record["completed_rl_games"])
        summary["completed_iteration"] = int(record["completed_iteration"])
        rows.append(summary)
        print(
            f"{summary['completed_rl_games']:>12,} "
            f"{summary['blocked_share']:>8.1%} "
            f"{summary['empty_hand_win_rate']:>7.1%} "
            f"{summary['blocked_win_rate']:>7.1%} "
            f"{summary['mean_pip_margin']:>7.2f} "
            f"{summary['mean_blocked_magnitude']:>7.3f} "
            f"{summary['terminal_abs_mean']:>7.4f} "
            f"{summary['local_abs_mean']:>7.4f} "
            f"{summary['immediate_share']:>5.1%}"
        )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as stream:
            json.dump(rows, stream, indent=2)
        print(f"\nWrote {len(rows)} rows to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
