"""Reinforcement-learning refinement for the domino policy."""

import random
import time
from collections import deque

import numpy as np

from agents.heuristic_agent import StrategicAgent
from agents.nn import GPU_ENABLED
from agents.rl_agent import RLAgent
from agents.rl_nn import PolicyNetwork
from middleware.domino_engine import DominoEngine
from middleware.middleware import GameManager

if GPU_ENABLED:
    import cupy as xp
else:
    xp = np

SL_WEIGHTS = "models/domino_sl_weights.npz"
RL_WEIGHTS = "models/domino_rl_weights.npz"
TRAINING_OPPONENT = "self_play"


def _play_game(agents):
    engine = DominoEngine(player_count=len(agents))
    manager = GameManager(engine, agents)
    info, _ = manager.play_full_game()
    return info["winner"]


def _winner_rewards(winner, player_count=2):
    if winner == -1:
        return [0.0] * player_count
    return [1.0 if player == winner else -1.0 for player in range(player_count)]


def _collect_self_play_steps(network, pool):
    """Play one game against a frozen policy snapshot and collect learner steps."""
    learner_position = random.randint(0, 1)
    opponent_network = random.choice(pool) if pool else network

    learner = RLAgent(network, mode="training")
    opponent = RLAgent(opponent_network, mode="training")
    agents = [None, None]
    agents[learner_position] = learner
    agents[1 - learner_position] = opponent

    winner = _play_game(agents)
    reward = _winner_rewards(winner)[learner_position]
    return learner.finish_episode(reward), winner, learner_position


def _collect_steps_vs_heuristic(network):
    """Play one training game against the fixed heuristic agent."""
    learner_position = random.randint(0, 1)
    learner = RLAgent(network, mode="training")
    agents = [None, None]
    agents[learner_position] = learner
    agents[1 - learner_position] = StrategicAgent()

    winner = _play_game(agents)
    reward = _winner_rewards(winner)[learner_position]
    return learner.finish_episode(reward), winner, learner_position


def evaluate_against_heuristic(network, game_count=200):
    """Measure greedy RL play against the fixed heuristic reference."""
    wins = 0
    draws = 0
    for i in range(game_count):
        rl_position = i % 2
        agents = [None, None]
        agents[rl_position] = RLAgent(network, mode="evaluation")
        agents[1 - rl_position] = StrategicAgent()

        winner = _play_game(agents)
        if winner == rl_position:
            wins += 1
        elif winner == -1:
            draws += 1

    return wins / game_count, draws / game_count


def train(
    iterations=1000,
    games_per_iteration=40,
    training_opponent=TRAINING_OPPONENT,
    learning_rate=0.001,
    entropy_coef=0.01,
    log_interval=10,
    checkpoint_interval=50,
    pool_interval=10,
    max_pool_size=50,
    evaluation_games=200,
    sl_weights_path=SL_WEIGHTS,
    rl_weights_path=RL_WEIGHTS,
):
    """
    Train the policy with REINFORCE plus a learned value baseline.

    ``training_opponent`` can be ``"self_play"`` for a pool of frozen previous
    policy snapshots, or ``"heuristic"`` for direct play against StrategicAgent.
    The heuristic evaluation is always external to training.
    """
    if training_opponent not in ("self_play", "heuristic"):
        raise ValueError("training_opponent must be 'self_play' or 'heuristic'.")

    try:
        network = PolicyNetwork.load(rl_weights_path, learning_rate=learning_rate)
        print(f"Resuming RL training from {rl_weights_path}")
    except FileNotFoundError:
        network = PolicyNetwork.load_from_sl(sl_weights_path, learning_rate=learning_rate)
        print(f"Initializing RL policy from supervised weights: {sl_weights_path}")

    pool = None
    if training_opponent == "self_play":
        pool = deque(maxlen=max_pool_size)
        pool.append(network.clone())

    start_time = time.time()
    for iteration in range(1, iterations + 1):
        batch = []
        wins = 0

        for _ in range(games_per_iteration):
            if training_opponent == "self_play":
                steps, winner, learner_position = _collect_self_play_steps(network, pool)
            else:
                steps, winner, learner_position = _collect_steps_vs_heuristic(network)
            batch.extend(steps)
            if winner == learner_position:
                wins += 1

        if not batch:
            continue

        x_batch = xp.hstack([x for x, _, _ in batch])
        action_indices = [action_index for _, action_index, _ in batch]
        returns = xp.array([reward for _, _, reward in batch], dtype=float).reshape(1, -1)

        values = network.predict_values(x_batch)
        advantages = returns - values

        metrics = network.backward_policy_gradient(
            action_indices,
            advantages,
            returns=returns,
            entropy_coef=entropy_coef,
        )

        if training_opponent == "self_play" and iteration % pool_interval == 0:
            pool.append(network.clone())

        if iteration % log_interval == 0:
            win_label = "vs pool" if training_opponent == "self_play" else "vs heuristic"
            pool_suffix = f" | pool: {len(pool)}" if training_opponent == "self_play" else ""
            print(
                f"Iteration {iteration} | steps: {len(batch)} | "
                f"mean advantage: {float(xp.mean(advantages)):.3f} | "
                f"entropy: {metrics['entropy']:.3f} | "
                f"value loss: {metrics['value_loss']:.3f} | "
                f"wins {win_label}: {wins}/{games_per_iteration}{pool_suffix}"
            )

        if iteration % checkpoint_interval == 0:
            network.save(rl_weights_path)
            win_rate, draw_rate = evaluate_against_heuristic(network, game_count=evaluation_games)
            print(
                f"  [checkpoint] saved {rl_weights_path} | "
                f"greedy vs heuristic: {win_rate:.1%} wins, "
                f"{draw_rate:.1%} draws ({evaluation_games} games)"
            )

    network.save(rl_weights_path)
    elapsed_time = time.time() - start_time
    print(f"\nTraining complete in {elapsed_time:.1f}s. Final weights: {rl_weights_path}")


if __name__ == "__main__":
    train()
