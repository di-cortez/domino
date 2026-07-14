"""Direct REINFORCE self-play for the domino policy network.

The policy controls only real tile-play decisions. Draw, pass, and single-option
tile-play turns are forced by the rules engine and do not enter the learner
trajectory. Local draw/pass events are distributed to all earlier real
decisions with temporal decay, then combined with a uniform terminal reward and
an option-count multiplier.
"""

from collections import deque
from dataclasses import dataclass
import random
import time

from agents.encoder import DominoEncoder
from agents.heuristic_agent import StrategicAgent
from agents.nn import GPU_ENABLED
from agents.rl_agent import RLAgent
from agents.rl_nn import PolicyNetwork
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

TERMINAL_WIN_REWARD = 0.50
TERMINAL_TIE_REWARD = 0.00
TERMINAL_LOSS_REWARD = -0.50
FINAL_PIP_PENALTY = 0.001

OPPONENT_DRAW_REWARD = 0.02
LEARNER_DRAW_PENALTY = -0.02
OPPONENT_PASS_REWARD = 0.10
LEARNER_PASS_PENALTY = -0.10
EVENT_REWARD_DECAY = 0.90

MIN_REAL_DECISION_OPTIONS = 2
THREE_OPTION_DECISION = 3
FOUR_OPTION_DECISION = 4
FIVE_PLUS_OPTION_DECISION = 5
CHOICE_MULTIPLIER_2_OPTIONS = 1.0
CHOICE_MULTIPLIER_3_OPTIONS = 2.0
CHOICE_MULTIPLIER_4_OPTIONS = 5.0
CHOICE_MULTIPLIER_5_PLUS_OPTIONS = 10.0

REWARD_ZERO_EPSILON = 1e-8


@dataclass
class EventStats:
    """Raw draw/pass event counts collected during one training game or batch."""

    opponent_draws: int = 0
    opponent_passes: int = 0
    learner_draws: int = 0
    learner_passes: int = 0

    def add(self, other):
        self.opponent_draws += other.opponent_draws
        self.opponent_passes += other.opponent_passes
        self.learner_draws += other.learner_draws
        self.learner_passes += other.learner_passes


@dataclass(frozen=True)
class TrainingSample:
    """One finalized real decision used by the REINFORCE update."""

    x: object
    action_index: int
    legal_mask: object
    policy_reward: float
    raw_reward: float
    local_reward: float
    terminal_reward: float
    multiplier: float
    option_count: int


def _tile_play_actions(legal_actions):
    """Return legal tile-play actions, excluding forced draw/pass."""
    return [
        action
        for action in legal_actions
        if action is not None and action != ("DRAW", None)
    ]


def _choice_multiplier(option_count):
    """Return the reward multiplier for a real decision with many options."""
    if option_count < MIN_REAL_DECISION_OPTIONS:
        raise ValueError("Real RL decisions must have at least two legal tile actions.")
    if option_count >= FIVE_PLUS_OPTION_DECISION:
        return CHOICE_MULTIPLIER_5_PLUS_OPTIONS
    if option_count == FOUR_OPTION_DECISION:
        return CHOICE_MULTIPLIER_4_OPTIONS
    if option_count == THREE_OPTION_DECISION:
        return CHOICE_MULTIPLIER_3_OPTIONS
    return CHOICE_MULTIPLIER_2_OPTIONS


def _event_reward_for_action(current_player, learner_position, action, event_stats):
    """Return the local event reward for draw/pass actions and update counts."""
    if current_player != learner_position:
        if action == ("DRAW", None):
            event_stats.opponent_draws += 1
            return OPPONENT_DRAW_REWARD
        if action is None:
            event_stats.opponent_passes += 1
            return OPPONENT_PASS_REWARD
    else:
        if action == ("DRAW", None):
            event_stats.learner_draws += 1
            return LEARNER_DRAW_PENALTY
        if action is None:
            event_stats.learner_passes += 1
            return LEARNER_PASS_PENALTY
    return None


def _remaining_pips(hand):
    return sum(tile[0] + tile[1] for tile in hand)


def _terminal_reward(engine, learner_position):
    """Return terminal outcome reward plus final remaining-pip penalty."""
    winner = engine.winner
    if winner == -1:
        outcome_reward = TERMINAL_TIE_REWARD
    elif winner == learner_position:
        outcome_reward = TERMINAL_WIN_REWARD
    else:
        outcome_reward = TERMINAL_LOSS_REWARD

    pip_penalty = FINAL_PIP_PENALTY * _remaining_pips(engine.hands[learner_position])
    return outcome_reward - pip_penalty


def _finish_episode_with_rewards(learner_agent, terminal_reward):
    """Finalize one learner trajectory into policy-gradient training samples."""
    finished_steps = learner_agent.finish_episode(terminal_reward)
    samples = []
    for step in finished_steps:
        multiplier = _choice_multiplier(step.option_count)
        policy_reward = step.raw_reward * multiplier
        samples.append(
            TrainingSample(
                x=step.x,
                action_index=step.action_index,
                legal_mask=step.legal_mask,
                policy_reward=policy_reward,
                raw_reward=step.raw_reward,
                local_reward=step.local_reward,
                terminal_reward=step.terminal_reward,
                multiplier=multiplier,
                option_count=step.option_count,
            )
        )
    return samples


def _reward_signal_summary(samples):
    """Return compact diagnostics for finalized decision rewards."""
    rewards = xp.asarray([sample.policy_reward for sample in samples], dtype=float)
    local_rewards = xp.asarray([sample.local_reward for sample in samples], dtype=float)
    total = rewards.size

    good = xp.sum(rewards > REWARD_ZERO_EPSILON)
    neutral = xp.sum(xp.abs(rewards) <= REWARD_ZERO_EPSILON)
    bad = xp.sum(rewards < -REWARD_ZERO_EPSILON)

    return {
        "reward_mean": float(xp.mean(rewards)),
        "reward_min": float(xp.min(rewards)),
        "reward_max": float(xp.max(rewards)),
        "local_mean": float(xp.mean(local_rewards)),
        "good_pct": float(100.0 * good / total),
        "neutral_pct": float(100.0 * neutral / total),
        "bad_pct": float(100.0 * bad / total),
    }


def _gradient_log_text(metrics):
    """Return a compact gradient-norm string for the iteration log."""
    suffix = " clipped" if metrics.get("grad_clipped") else ""
    return f"{metrics['grad_norm']:.2f}{suffix}"


def _play_game(agents):
    """Play one evaluation game and return only the winner id."""
    engine = DominoEngine(player_count=len(agents))
    manager = GameManager(engine, agents)
    info, _ = manager.play_full_game()
    return info["winner"]


def _play_training_game(agents, learner_position, learner_agent):
    """Play one RL training game and attach decayed local event rewards."""
    engine = DominoEngine(player_count=len(agents))
    event_stats = EventStats()

    while not engine.game_over:
        state = engine._get_state()
        current_player = state["current_player"]
        legal_actions = engine.valid_actions(current_player)
        tile_actions = _tile_play_actions(legal_actions)

        if current_player == learner_position and len(tile_actions) == 1:
            action = tile_actions[0]
        else:
            action = agents[current_player].choose_move(state, legal_actions)

        event_reward = _event_reward_for_action(
            current_player,
            learner_position,
            action,
            event_stats,
        )
        if event_reward is not None:
            learner_agent.add_decayed_event_reward(
                event_turn=state["turn"],
                base_reward=event_reward,
                decay_lambda=EVENT_REWARD_DECAY,
            )

        engine.step(action)

    return engine, event_stats


def _collect_self_play_steps(network, pool):
    """Play one game against a frozen policy snapshot and collect learner samples."""
    learner_position = random.randint(0, 1)
    opponent_network = random.choice(pool) if pool else network

    learner = RLAgent(network, mode="training")
    opponent = RLAgent(opponent_network, mode="evaluation")
    agents = [None, None]
    agents[learner_position] = learner
    agents[1 - learner_position] = opponent

    engine, event_stats = _play_training_game(agents, learner_position, learner)
    reward = _terminal_reward(engine, learner_position)
    samples = _finish_episode_with_rewards(learner, reward)
    return samples, event_stats, engine.winner, learner_position


def _collect_steps_vs_heuristic(network):
    """Play one training game against the fixed heuristic agent."""
    learner_position = random.randint(0, 1)
    learner = RLAgent(network, mode="training")
    agents = [None, None]
    agents[learner_position] = learner
    agents[1 - learner_position] = StrategicAgent()

    engine, event_stats = _play_training_game(agents, learner_position, learner)
    reward = _terminal_reward(engine, learner_position)
    samples = _finish_episode_with_rewards(learner, reward)
    return samples, event_stats, engine.winner, learner_position


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


def _load_initial_network(learning_rate, sl_weights_path, rl_weights_path, quiet=False):
    """Load a compatible RL checkpoint or initialize from compatible SL weights."""
    try:
        network = PolicyNetwork.load(rl_weights_path, learning_rate=learning_rate)
        if not _checkpoint_matches_encoder(network):
            raise ValueError(
                f"RL checkpoint {rl_weights_path} has shape "
                f"input={network.W1.shape[1]}, output={network.W3.shape[0]}, "
                "but the current encoder expects input=168, output=56."
            )
        if not quiet:
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
    if not quiet:
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
    quiet=False,
    progress_callback=None,
):
    """Train the policy with direct REINFORCE and decayed local rewards.

    ``training_opponent`` can be ``"self_play"`` for a pool of frozen previous
    policy snapshots, or ``"heuristic"`` for direct play against StrategicAgent.
    The update uses the reward assigned to each real decision directly, with no
    extra prediction target beyond the masked policy.
    """
    if training_opponent not in ("self_play", "heuristic"):
        raise ValueError("training_opponent must be 'self_play' or 'heuristic'.")

    if not quiet:
        print_memory_report("RL self-play startup memory")
    network = _load_initial_network(
        learning_rate,
        sl_weights_path,
        rl_weights_path,
        quiet=quiet,
    )

    pool = None
    if training_opponent == "self_play":
        pool = deque(maxlen=max_pool_size)
        pool.append(network.clone())

    start_time = time.time()
    last_checkpoint_time = start_time
    for iteration in range(1, iterations + 1):
        batch = []
        event_totals = EventStats()
        wins = 0

        for _ in range(games_per_iteration):
            if training_opponent == "self_play":
                samples, event_stats, winner, learner_position = _collect_self_play_steps(
                    network,
                    pool,
                )
            else:
                samples, event_stats, winner, learner_position = _collect_steps_vs_heuristic(
                    network
                )
            batch.extend(samples)
            event_totals.add(event_stats)
            if winner == learner_position:
                wins += 1

        if not batch:
            if progress_callback is not None:
                progress_callback(iteration, iterations)
            continue

        x_batch = xp.hstack([sample.x for sample in batch])
        action_indices = [sample.action_index for sample in batch]
        legal_masks = xp.hstack([
            xp.asarray(sample.legal_mask)
            for sample in batch
        ])
        policy_rewards = xp.array(
            [sample.policy_reward for sample in batch],
            dtype=float,
        ).reshape(1, -1)

        network.forward(x_batch)
        metrics = network.backward_policy_gradient(
            action_indices,
            policy_rewards,
            legal_masks=legal_masks,
            entropy_coef=entropy_coef,
        )

        if training_opponent == "self_play" and iteration % pool_interval == 0:
            pool.append(network.clone())

        if iteration % log_interval == 0 and not quiet:
            reward_summary = _reward_signal_summary(batch)
            win_label = "vs pool" if training_opponent == "self_play" else "vs heuristic"
            pool_suffix = f" | pool: {len(pool)}" if training_opponent == "self_play" else ""
            print(
                f"Iteration {iteration} | decisions: {len(batch)} | "
                "reward mean/min/max: "
                f"{reward_summary['reward_mean']:+.2f}/"
                f"{reward_summary['reward_min']:+.2f}/"
                f"{reward_summary['reward_max']:+.2f} | "
                "good/neutral/bad: "
                f"{reward_summary['good_pct']:.0f}%/"
                f"{reward_summary['neutral_pct']:.0f}%/"
                f"{reward_summary['bad_pct']:.0f}% | "
                f"local mean: {reward_summary['local_mean']:+.3f} | "
                "opp D/P: "
                f"{event_totals.opponent_draws}/{event_totals.opponent_passes}, "
                f"self D/P: {event_totals.learner_draws}/{event_totals.learner_passes} | "
                f"wins {win_label}: {wins}/{games_per_iteration}{pool_suffix} | "
                f"grad: {_gradient_log_text(metrics)}"
            )

        if iteration % checkpoint_interval == 0:
            network.save(rl_weights_path)
            win_rate, draw_rate = evaluate_against_heuristic(network, game_count=evaluation_games)
            now = time.time()
            checkpoint_elapsed = now - last_checkpoint_time
            last_checkpoint_time = now
            if not quiet:
                print(
                    f"  [checkpoint] saved {rl_weights_path} | "
                    f"time since previous checkpoint: {format_duration(checkpoint_elapsed)} | "
                    f"deterministic RL vs heuristic: {win_rate:.1%} wins, "
                    f"{draw_rate:.1%} draws ({evaluation_games} games)"
                )

        if progress_callback is not None:
            progress_callback(iteration, iterations)

    network.save(rl_weights_path)
    elapsed_time = time.time() - start_time
    if not quiet:
        print(f"\nTraining complete. Total elapsed time: {format_duration(elapsed_time)}.")
        print(f"Final weights: {rl_weights_path}")

    return {
        "iterations": iterations,
        "games_per_iteration": games_per_iteration,
        "training_opponent": training_opponent,
        "rl_weights_path": rl_weights_path,
        "duration_s": elapsed_time,
    }


if __name__ == "__main__":
    train()
