"""Reinforcement-learning self-play for the domino policy network.

The policy controls only real tile-play decisions. Draw, pass, and single-option
tile-play turns are forced by the rules engine and do not enter the learner
trajectory. Rewards are multiplied when the learner had many legal tile-play
options, so rare high-choice decisions have a stronger learning signal.
"""

from collections import deque
import random
import time

from agents.heuristic_agent import StrategicAgent
from agents.rl_agent import RLAgent
from agents.rl_nn import PolicyNetwork
from agents.encoder import DominoEncoder
from agents.nn import GPU_ENABLED
from middleware.domino_engine import DominoEngine
from middleware.middleware import GameManager
from utils.runtime_status import format_duration, print_memory_report

if GPU_ENABLED:
    import cupy as xp
else:
    import numpy as xp

SL_WEIGHTS = "models/domino_sl_weights.npz"
RL_WEIGHTS = "models/domino_rl_weights.npz"
TRAINING_OPPONENT = "self_play"

OPPONENT_DRAW_REWARD = 0.05
OPPONENT_PASS_REWARD = 0.05
FINAL_PIP_PENALTY = 0.001

CHOICE_MULTIPLIER_3_OPTIONS = 2.0
CHOICE_MULTIPLIER_4_OPTIONS = 5.0
CHOICE_MULTIPLIER_5_PLUS_OPTIONS = 10.0


def _tile_play_actions(legal_actions):
    """Return legal tile-play actions, excluding forced draw/pass."""
    return [
        action
        for action in legal_actions
        if action is not None and action != ("DRAW", None)
    ]


def _choice_multiplier(option_count):
    """Return the reward multiplier for a real decision with many options."""
    if option_count >= 5:
        return CHOICE_MULTIPLIER_5_PLUS_OPTIONS
    if option_count == 4:
        return CHOICE_MULTIPLIER_4_OPTIONS
    if option_count == 3:
        return CHOICE_MULTIPLIER_3_OPTIONS
    return 1.0


def _reset_decision_multipliers(learner_agent):
    learner_agent._real_decision_multipliers = []


def _append_decision_multiplier(learner_agent, option_count):
    learner_agent._real_decision_multipliers.append(_choice_multiplier(option_count))


def _finish_episode_with_multipliers(learner_agent, final_reward):
    """Finish an episode and multiply each saved decision return by its option count."""
    steps = learner_agent.finish_episode(final_reward)
    multipliers = getattr(learner_agent, "_real_decision_multipliers", [])
    learner_agent._real_decision_multipliers = []

    if len(steps) != len(multipliers):
        raise RuntimeError(
            "RL trajectory/multiplier mismatch: "
            f"{len(steps)} saved steps but {len(multipliers)} multipliers."
        )

    return [
        (x, action_index, legal_mask, reward * multiplier)
        for (x, action_index, legal_mask, reward), multiplier in zip(steps, multipliers)
    ]


def _remaining_pips(hand):
    return sum(tile[0] + tile[1] for tile in hand)


def _terminal_reward(engine, learner_position):
    """Return win/loss/draw reward plus final remaining-pip penalty."""
    winner = engine.winner
    if winner == -1:
        outcome_reward = 0.0
    elif winner == learner_position:
        outcome_reward = 1.0
    else:
        outcome_reward = -1.0

    pip_penalty = FINAL_PIP_PENALTY * _remaining_pips(engine.hands[learner_position])
    return outcome_reward - pip_penalty


def _play_game(agents):
    """Play one evaluation game and return only the winner id."""
    engine = DominoEngine(player_count=len(agents))
    manager = GameManager(engine, agents)
    info, _ = manager.play_full_game()
    return info["winner"]


def _play_training_game(agents, learner_position, learner_agent):
    """Play one RL training game and attach intermediate shaping rewards.

    The learner receives +0.05 when the opponent is forced to draw and +0.05
    when the opponent is forced to pass. If the opponent draws and then passes,
    both rewards are applied to the learner's most recent real decision.
    """
    engine = DominoEngine(player_count=len(agents))
    _reset_decision_multipliers(learner_agent)

    while not engine.game_over:
        state = engine._get_state()
        current_player = state["current_player"]
        legal_actions = engine.valid_actions(current_player)
        tile_actions = _tile_play_actions(legal_actions)

        if current_player == learner_position and len(tile_actions) == 1:
            action = tile_actions[0]
        else:
            saved_step_count = len(learner_agent.trajectory)
            action = agents[current_player].choose_move(state, legal_actions)

            if (
                current_player == learner_position
                and len(tile_actions) >= 2
                and len(learner_agent.trajectory) > saved_step_count
            ):
                _append_decision_multiplier(learner_agent, len(tile_actions))

        if current_player != learner_position:
            if action == ("DRAW", None):
                learner_agent.add_reward_to_last_decision(OPPONENT_DRAW_REWARD)
            elif action is None:
                learner_agent.add_reward_to_last_decision(OPPONENT_PASS_REWARD)

        engine.step(action)

    return engine


def _collect_self_play_steps(network, pool):
    """Play one game against a frozen policy snapshot and collect learner steps."""
    learner_position = random.randint(0, 1)
    opponent_network = random.choice(pool) if pool else network

    learner = RLAgent(network, mode="training")
    opponent = RLAgent(opponent_network, mode="training")
    agents = [None, None]
    agents[learner_position] = learner
    agents[1 - learner_position] = opponent

    engine = _play_training_game(agents, learner_position, learner)
    reward = _terminal_reward(engine, learner_position)
    return _finish_episode_with_multipliers(learner, reward), engine.winner, learner_position


def _collect_steps_vs_heuristic(network):
    """Play one training game against the fixed heuristic agent."""
    learner_position = random.randint(0, 1)
    learner = RLAgent(network, mode="training")
    agents = [None, None]
    agents[learner_position] = learner
    agents[1 - learner_position] = StrategicAgent()

    engine = _play_training_game(agents, learner_position, learner)
    reward = _terminal_reward(engine, learner_position)
    return _finish_episode_with_multipliers(learner, reward), engine.winner, learner_position


def evaluate_against_heuristic(network, game_count=200):
    """Evaluate deterministic RL play against the heuristic reference."""
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


def _checkpoint_matches_encoder(network):
    """Return True when a loaded checkpoint matches the current encoder shape."""
    encoder = DominoEncoder()
    return (
        network.W1.shape[1] == encoder.VECTOR_SIZE
        and network.W3.shape[0] == len(encoder.all_actions)
    )


def _load_initial_network(learning_rate, sl_weights_path, rl_weights_path):
    """Load a compatible RL checkpoint or initialize from compatible SL weights."""
    try:
        network = PolicyNetwork.load(rl_weights_path, learning_rate=learning_rate)
        if not _checkpoint_matches_encoder(network):
            raise ValueError(
                f"RL checkpoint {rl_weights_path} has shape "
                f"input={network.W1.shape[1]}, output={network.W3.shape[0]}, "
                "but the current encoder expects input=168, output=56."
            )
        print(f"Resuming RL training from {rl_weights_path}")
        return network
    except FileNotFoundError:
        pass

    network = PolicyNetwork.load_from_sl(sl_weights_path, learning_rate=learning_rate)
    if not _checkpoint_matches_encoder(network):
        raise ValueError(
            f"SL checkpoint {sl_weights_path} has shape "
            f"input={network.W1.shape[1]}, output={network.W3.shape[0]}, "
            "but the current encoder expects input=168, output=56. "
            "Regenerate the supervised dataset and retrain SL first."
        )
    print(f"Initializing RL policy from supervised weights: {sl_weights_path}")
    return network


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
    Intermediate reward shaping is intentionally weak and transparent: the
    learner gets a small bonus when the opponent is forced to draw/pass and a
    small terminal penalty for final remaining pips. Each saved decision return
    is multiplied by the number-of-options schedule defined at the top of this
    file.
    """
    if training_opponent not in ("self_play", "heuristic"):
        raise ValueError("training_opponent must be 'self_play' or 'heuristic'.")

    print_memory_report("RL self-play startup memory")
    network = _load_initial_network(learning_rate, sl_weights_path, rl_weights_path)

    pool = None
    if training_opponent == "self_play":
        pool = deque(maxlen=max_pool_size)
        pool.append(network.clone())

    start_time = time.time()
    last_checkpoint_time = start_time
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

        x_batch = xp.hstack([x for x, _, _, _ in batch])
        action_indices = [action_index for _, action_index, _, _ in batch]
        legal_masks = xp.hstack([
            xp.asarray(legal_mask)
            for _, _, legal_mask, _ in batch
        ])
        returns = xp.array([reward for _, _, _, reward in batch], dtype=float).reshape(1, -1)

        values = network.predict_values(x_batch)
        advantages = returns - values

        metrics = network.backward_policy_gradient(
            action_indices,
            advantages,
            legal_masks=legal_masks,
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
                f"mean return: {float(xp.mean(returns)):.2f} | "
                f"max return: {float(xp.max(returns)):.1f} | "
                f"min return: {float(xp.min(returns)):.1f} | "
                f"mean adv: {float(xp.mean(advantages)):.2f} | "
                f"value loss: {metrics['value_loss']:.2f} | "
                f"wins {win_label}: {wins}/{games_per_iteration}{pool_suffix}"
            )

        if iteration % checkpoint_interval == 0:
            network.save(rl_weights_path)
            win_rate, draw_rate = evaluate_against_heuristic(network, game_count=evaluation_games)
            now = time.time()
            checkpoint_elapsed = now - last_checkpoint_time
            last_checkpoint_time = now
            print(
                f"  [checkpoint] saved {rl_weights_path} | "
                f"time since previous checkpoint: {format_duration(checkpoint_elapsed)} | "
                f"deterministic RL vs heuristic: {win_rate:.1%} wins, "
                f"{draw_rate:.1%} draws ({evaluation_games} games)"
            )

    network.save(rl_weights_path)
    elapsed_time = time.time() - start_time
    print(f"\nTraining complete. Total elapsed time: {format_duration(elapsed_time)}.")
    print(f"Final weights: {rl_weights_path}")


if __name__ == "__main__":
    train()
