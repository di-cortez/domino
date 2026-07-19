"""REINFORCE self-play with an optional learned value baseline.

The policy controls only real tile-play decisions. Draw, pass, and single-option
tile-play turns are forced by the rules engine and do not enter the learner
trajectory. Local draw/pass events are distributed to all earlier real
decisions with temporal decay, then combined with a uniform terminal reward and
an option-count multiplier. Direct reward updates are the default; a value head
can optionally convert those rewards into reward-minus-value advantages.

Independent games within each iteration run in deterministic CPU-only workers.
Workers read frozen policies from shared memory and return trajectories; only
the parent process assembles batches, updates weights, writes checkpoints, or
uses the GPU.
"""

import argparse
from collections import deque
from dataclasses import dataclass
import random
import secrets
import time

import numpy as np

from agents.encoder import DominoEncoder
from agents.heuristic_agent import StrategicAgent
from agents.rl_agent import RLAgent
from agents.rl_nn import DEVICES, PolicyNetwork
from middleware.domino_engine import DominoEngine
from middleware.middleware import GameManager
from diagnostics.parallel_runner import (
    MAX_PARALLEL_WORKERS,
    ParallelSafetyConfig,
    cap_parallel_workers,
    game_seed,
)
from training.rl_parallel import (
    DEFAULT_RL_AUTOTUNE_FRACTION,
    DEFAULT_RL_MINIMUM_GAIN,
    DEFAULT_RL_WORKER_CANDIDATES,
    DEFAULT_RL_WORKERS,
    RLRolloutRunner,
    RetainedRLWorkerAutotuner,
    worker_count as parse_rl_worker_count,
)
from utils.resource_limits import (
    MIB,
    MemorySafetyError,
    choose_safe_rl_device,
    effective_gpu_available_bytes,
    ensure_ram_available,
)
from utils.runtime_status import format_duration, print_memory_report

# The array backend for a given run is resolved once, inside train(), from
# the `device` parameter (see agents/rl_nn.py::_resolve_device) -- it always
# matches whatever PolicyNetwork itself is using, rather than being fixed at
# import time.
DEFAULT_DEVICE = "auto"

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
VALUE_COEF = 0.5
DEFAULT_CLIP_GRAD_NORM = 5.0
DEFAULT_MOVING_AVERAGE_WINDOW = 10
# Off by default: this is a training-dynamics change (P1 in the historical
# reports), not pure instrumentation, so it must not silently change the
# default behavior of existing callers (run_pipeline.py, train_script/, the
# hyperparameter sweep). Opt in with --normalize-advantages.
DEFAULT_NORMALIZE_ADVANTAGES = False

# Named presets for the terminal/event reward constants above, selectable at
# training time so hyperparameter sweeps can vary reward shaping without
# editing source. "default" reproduces the fixed constants exactly.
REWARD_SCHEMAS = {
    "default": {
        "terminal_win": TERMINAL_WIN_REWARD,
        "terminal_tie": TERMINAL_TIE_REWARD,
        "terminal_loss": TERMINAL_LOSS_REWARD,
        "final_pip_penalty": FINAL_PIP_PENALTY,
        "opponent_draw": OPPONENT_DRAW_REWARD,
        "learner_draw": LEARNER_DRAW_PENALTY,
        "opponent_pass": OPPONENT_PASS_REWARD,
        "learner_pass": LEARNER_PASS_PENALTY,
        "event_decay": EVENT_REWARD_DECAY,
    },
    "sparse": {
        "terminal_win": TERMINAL_WIN_REWARD,
        "terminal_tie": TERMINAL_TIE_REWARD,
        "terminal_loss": TERMINAL_LOSS_REWARD,
        "final_pip_penalty": 0.0,
        "opponent_draw": 0.0,
        "learner_draw": 0.0,
        "opponent_pass": 0.0,
        "learner_pass": 0.0,
        "event_decay": EVENT_REWARD_DECAY,
    },
    "shaped": {
        "terminal_win": TERMINAL_WIN_REWARD,
        "terminal_tie": TERMINAL_TIE_REWARD,
        "terminal_loss": TERMINAL_LOSS_REWARD,
        "final_pip_penalty": FINAL_PIP_PENALTY,
        "opponent_draw": OPPONENT_DRAW_REWARD * 2.0,
        "learner_draw": LEARNER_DRAW_PENALTY * 2.0,
        "opponent_pass": OPPONENT_PASS_REWARD * 2.0,
        "learner_pass": LEARNER_PASS_PENALTY * 2.0,
        "event_decay": EVENT_REWARD_DECAY,
    },
}
DEFAULT_REWARD_SCHEMA = "default"

# Terminal-reward discount applied per remaining real decision (1.0 = no
# discount, i.e. the previous fixed behavior).
DEFAULT_GAMMA = 1.0


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


def _event_reward_for_action(
    current_player, learner_position, action, event_stats, schema=None
):
    """Return the local event reward for draw/pass actions and update counts."""
    if schema is None:
        schema = REWARD_SCHEMAS[DEFAULT_REWARD_SCHEMA]
    if current_player != learner_position:
        if action == ("DRAW", None):
            event_stats.opponent_draws += 1
            return schema["opponent_draw"]
        if action is None:
            event_stats.opponent_passes += 1
            return schema["opponent_pass"]
    else:
        if action == ("DRAW", None):
            event_stats.learner_draws += 1
            return schema["learner_draw"]
        if action is None:
            event_stats.learner_passes += 1
            return schema["learner_pass"]
    return None


def _remaining_pips(hand):
    return sum(tile[0] + tile[1] for tile in hand)


def _terminal_reward(engine, learner_position, schema):
    """Return terminal outcome reward plus final remaining-pip penalty."""
    winner = engine.winner
    if winner == -1:
        outcome_reward = schema["terminal_tie"]
    elif winner == learner_position:
        outcome_reward = schema["terminal_win"]
    else:
        outcome_reward = schema["terminal_loss"]

    pip_penalty = schema["final_pip_penalty"] * _remaining_pips(engine.hands[learner_position])
    return outcome_reward - pip_penalty


def _finish_episode_with_rewards(learner_agent, terminal_reward, gamma=DEFAULT_GAMMA):
    """Finalize one learner trajectory into policy-gradient training samples.

    ``gamma`` discounts the terminal-reward component per remaining real
    decision, so earlier decisions in the trajectory receive a more heavily
    discounted share of the final outcome than the last one. Local event
    rewards already carry their own temporal decay and are not affected.
    """
    finished_steps = learner_agent.finish_episode(terminal_reward)
    step_count = len(finished_steps)
    samples = []
    for index, step in enumerate(finished_steps):
        remaining_after = step_count - 1 - index
        discounted_terminal = step.terminal_reward * (gamma ** remaining_after)
        raw_reward = discounted_terminal + step.local_reward
        multiplier = _choice_multiplier(step.option_count)
        policy_reward = raw_reward * multiplier
        samples.append(
            TrainingSample(
                x=step.x,
                action_index=step.action_index,
                legal_mask=step.legal_mask,
                policy_reward=policy_reward,
                raw_reward=raw_reward,
                local_reward=step.local_reward,
                terminal_reward=discounted_terminal,
                multiplier=multiplier,
                option_count=step.option_count,
            )
        )
    return samples


def _reward_signal_summary(samples, xp=None):
    """Return compact diagnostics for finalized decision rewards.

    ``reward_std`` disambiguates a falling value loss from a merely
    low-variance batch: since a value head that has not learned anything
    predicts close to the batch mean, its loss is approximately
    ``0.5 * reward_std ** 2`` — logging the standard deviation next to the
    loss makes that identity checkable instead of hidden behind a noisy
    scalar (see references/explicacoes/relatorios/relatorio_1407).

    ``xp`` should be the training run's resolved array backend (``train()``
    passes ``network.xp``); it defaults to NumPy for direct callers, which is
    fine here since this is small-scale summary math, not the training path.
    """
    if xp is None:
        xp = np
    rewards = xp.asarray([sample.policy_reward for sample in samples], dtype=float)
    local_rewards = xp.asarray([sample.local_reward for sample in samples], dtype=float)
    total = rewards.size

    good = xp.sum(rewards > REWARD_ZERO_EPSILON)
    neutral = xp.sum(xp.abs(rewards) <= REWARD_ZERO_EPSILON)
    bad = xp.sum(rewards < -REWARD_ZERO_EPSILON)

    return {
        "reward_mean": float(xp.mean(rewards)),
        "reward_std": float(xp.std(rewards)),
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


def _play_training_game(agents, learner_position, learner_agent, schema):
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
            schema,
        )
        if event_reward is not None:
            learner_agent.add_decayed_event_reward(
                event_turn=state["turn"],
                base_reward=event_reward,
                decay_lambda=schema["event_decay"],
            )

        engine.step(
            action,
            return_state=False,
            legal_actions=legal_actions,
        )

    return engine, event_stats


def _collect_self_play_steps(network, pool, schema, gamma):
    """Play one game against a frozen policy snapshot and collect learner samples."""
    learner_position = random.randint(0, 1)
    opponent_network = random.choice(pool) if pool else network

    learner = RLAgent(network, mode="training")
    opponent = RLAgent(opponent_network, mode="stochastic_evaluation")
    agents = [None, None]
    agents[learner_position] = learner
    agents[1 - learner_position] = opponent

    engine, event_stats = _play_training_game(agents, learner_position, learner, schema)
    reward = _terminal_reward(engine, learner_position, schema)
    samples = _finish_episode_with_rewards(learner, reward, gamma)
    return samples, event_stats, engine.winner, learner_position


def _collect_steps_vs_heuristic(network, schema, gamma):
    """Play one training game against the fixed heuristic agent."""
    learner_position = random.randint(0, 1)
    learner = RLAgent(network, mode="training")
    agents = [None, None]
    agents[learner_position] = learner
    agents[1 - learner_position] = StrategicAgent()

    engine, event_stats = _play_training_game(agents, learner_position, learner, schema)
    reward = _terminal_reward(engine, learner_position, schema)
    samples = _finish_episode_with_rewards(learner, reward, gamma)
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


def _load_initial_network(
    learning_rate,
    sl_weights_path,
    rl_weights_path,
    quiet=False,
    use_value_head=False,
    device=DEFAULT_DEVICE,
    sl_weights_data=None,
):
    """Load a compatible RL checkpoint or initialize from compatible SL weights.

    ``sl_weights_data`` accepts a pre-loaded mapping of SL weight arrays (see
    ``PolicyNetwork.load_from_sl``), so a caller warm-starting many runs from
    the same SL checkpoint (e.g. a hyperparameter sweep) can read it from
    disk once and reuse it, instead of every run re-reading
    ``sl_weights_path``. Unused when resuming from an existing RL checkpoint.
    """
    try:
        network = PolicyNetwork.load(
            rl_weights_path,
            learning_rate=learning_rate,
            use_value_head=use_value_head,
            device=device,
        )
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

    network = PolicyNetwork.load_from_sl(
        sl_weights_path,
        learning_rate=learning_rate,
        use_value_head=use_value_head,
        device=device,
        data=sl_weights_data,
    )
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


def _new_parallel_summary(requested_workers):
    """Return mutable aggregate metadata for all RL worker-pool phases."""
    return {
        "requested_workers": requested_workers,
        "initial_workers": None,
        "final_workers": None,
        "peak_worker_rss_mb": 0.0,
        "peak_total_children_rss_mb": 0.0,
        "min_available_memory_mb": None,
        "fallback_count": 0,
        "fallback_history": [],
        "attempted_worker_counts": [],
        "safety_capped": False,
        "memory_monitoring_available": True,
        "workers_cpu_only": True,
        "rollout_batches": 0,
        "evaluation_batches": 0,
    }


def _merge_parallel_summary(summary, run_info, *, phase, iteration):
    """Accumulate one rollout/evaluation pool run into the public summary."""
    if summary["initial_workers"] is None:
        summary["initial_workers"] = run_info.initial_workers
    summary["final_workers"] = run_info.final_workers
    summary["peak_worker_rss_mb"] = max(
        summary["peak_worker_rss_mb"],
        run_info.peak_worker_rss_mb,
    )
    summary["peak_total_children_rss_mb"] = max(
        summary["peak_total_children_rss_mb"],
        run_info.peak_total_children_rss_mb,
    )
    available = run_info.min_available_memory_mb
    if available is not None:
        current = summary["min_available_memory_mb"]
        summary["min_available_memory_mb"] = (
            available if current is None else min(current, available)
        )
    summary["fallback_count"] += run_info.fallback_count
    for item in run_info.fallback_history:
        tagged = dict(item)
        tagged["rl_phase"] = phase
        tagged["iteration"] = int(iteration)
        summary["fallback_history"].append(tagged)
    summary["attempted_worker_counts"].extend(run_info.attempted_worker_counts)
    summary["safety_capped"] = summary["safety_capped"] or run_info.safety_capped
    summary["memory_monitoring_available"] = (
        summary["memory_monitoring_available"]
        and run_info.memory_monitoring_available
    )
    summary[f"{phase}_batches"] += 1


def _add_worker_event_stats(total, values):
    """Accumulate serialized worker event counters into ``EventStats``."""
    total.opponent_draws += int(values["opponent_draws"])
    total.opponent_passes += int(values["opponent_passes"])
    total.learner_draws += int(values["learner_draws"])
    total.learner_passes += int(values["learner_passes"])


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
    use_value_head=False,
    value_coef=VALUE_COEF,
    gamma=DEFAULT_GAMMA,
    reward_schema=DEFAULT_REWARD_SCHEMA,
    clip_grad_norm=DEFAULT_CLIP_GRAD_NORM,
    normalize_advantages=DEFAULT_NORMALIZE_ADVANTAGES,
    moving_average_window=DEFAULT_MOVING_AVERAGE_WINDOW,
    seed=None,
    device=DEFAULT_DEVICE,
    sl_weights_data=None,
    workers=DEFAULT_RL_WORKERS,
    safety_config=None,
    autotune_fraction=DEFAULT_RL_AUTOTUNE_FRACTION,
    autotune_minimum_gain=DEFAULT_RL_MINIMUM_GAIN,
    worker_candidates=DEFAULT_RL_WORKER_CANDIDATES,
    status_callback=None,
):
    """Train with direct REINFORCE or an optional learned value baseline.

    ``training_opponent`` can be ``"self_play"`` for a pool of frozen previous
    policy snapshots, or ``"heuristic"`` for direct play against StrategicAgent.
    The update uses the reward assigned to each real decision directly, with no
    extra prediction target beyond the masked policy. With
    ``use_value_head=True``, the same finalized policy rewards become value
    targets and the policy update uses ``reward - V(s)`` advantages.

    ``gamma`` discounts the terminal-reward component per remaining real
    decision (``1.0`` is the previous fixed behavior: no discount).
    ``reward_schema`` selects one of the named presets in ``REWARD_SCHEMAS``
    for the terminal/event reward constants.

    Convergence-monitoring behavior (validated in
    ``references/explicacoes/relatorios/relatorio_1407``): a point-in-time
    value loss or win rate is dominated by batch noise, so ``moving_average_window``
    controls a trailing average of both, logged next to the raw values.
    ``normalize_advantages`` standardizes the policy signal per batch (mean 0,
    std 1) before the gradient step, which keeps the effective step size
    comparable across iterations regardless of reward-schema/multiplier scale;
    it is off by default (matching prior behavior) and does not affect the
    value head's regression target, which keeps learning from raw rewards.
    ``clip_grad_norm`` bounds the update's
    gradient norm. ``seed`` fixes ``random``/``numpy`` state for reproducible
    comparisons between hyperparameter configurations.

    ``device`` selects the array backend: ``"auto"`` (default) matches
    ``GPU_ENABLED`` exactly, i.e. CuPy when installed, otherwise NumPy --
    unchanged from prior behavior. ``"cpu"``/``"gpu"`` force one backend
    regardless of what's installed/enabled globally (see
    ``agents/rl_nn.py::_resolve_device``); ``"gpu"`` raises if CuPy isn't
    installed.

    ``sl_weights_data`` accepts a pre-loaded mapping of SL weight arrays so a
    caller running many training calls back-to-back (e.g. a hyperparameter
    sweep) can read ``sl_weights_path`` from disk once and pass the result to
    every call, instead of re-reading it each time. Ignored when resuming
    from an existing ``rl_weights_path`` checkpoint.

    Rollout workers are CPU-only and never update the policy. ``workers="auto"``
    benchmarks complete early iterations with 1, 2, 4, 6, ... workers, keeps
    every benchmark game in training, and stops below the configured marginal
    throughput gain. Per-game seeds and ordered parent aggregation make a
    seeded run independent of worker scheduling and worker count.
    """
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if games_per_iteration < 1:
        raise ValueError("games_per_iteration must be positive")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if evaluation_games < 1:
        raise ValueError("evaluation_games must be positive")
    if pool_interval < 1:
        raise ValueError("pool_interval must be positive")
    if max_pool_size < 0:
        raise ValueError("max_pool_size must be non-negative")
    if training_opponent not in ("self_play", "heuristic"):
        raise ValueError("training_opponent must be 'self_play' or 'heuristic'.")
    if reward_schema not in REWARD_SCHEMAS:
        raise ValueError(
            f"Unknown reward_schema {reward_schema!r}; expected one of "
            f"{sorted(REWARD_SCHEMAS)}."
        )
    if workers != "auto":
        workers = int(workers)
        if not 1 <= workers <= MAX_PARALLEL_WORKERS:
            raise ValueError(
                f"workers must be 'auto' or between 1 and {MAX_PARALLEL_WORKERS}"
            )
    safety_config = safety_config or ParallelSafetyConfig()
    schema = REWARD_SCHEMAS[reward_schema]
    effective_seed = int(seed) if seed is not None else secrets.randbits(63)
    random.seed(effective_seed)
    np.random.seed(effective_seed & 0xFFFFFFFF)

    requested_device = device
    device, device_fallback_reason = choose_safe_rl_device(device)
    # Conservative upper bound for one full 52-decision trajectory per game
    # and several matrix/gradient copies during parent-side batch assembly.
    estimated_batch_bytes = games_per_iteration * 52 * 4096
    if device_fallback_reason:
        print(
            "RL memory safety: automatic GPU selection fell back to CPU because "
            f"{device_fallback_reason}.",
            flush=True,
        )
    network = _load_initial_network(
        learning_rate,
        sl_weights_path,
        rl_weights_path,
        quiet=quiet,
        use_value_head=use_value_head,
        device=device,
        sl_weights_data=sl_weights_data,
    )
    xp = network.xp
    policy_bytes = 0
    for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
        value = getattr(network, name)
        policy_bytes += int(value.nbytes)
    shared_pool_size = max_pool_size if training_opponent == "self_play" else 0
    estimated_shared_bytes = (1 + shared_pool_size) * policy_bytes
    estimated_peak_bytes = estimated_shared_bytes + estimated_batch_bytes
    ensure_ram_available(
        estimated_peak_bytes,
        safety_config.memory_reserve_mb,
        "RL self-play and shared-policy preflight",
    )
    if not quiet:
        print_memory_report("RL self-play startup memory")
        print(
            "RL resource preflight: "
            f"requested device={requested_device!r}, selected device={device!r}, "
            f"estimated peak host allocation {estimated_peak_bytes / MIB:.1f} MiB."
        )
        print(f"RL self-play array backend: {xp.__name__} (device={network.device!r})")

    if status_callback is not None:
        emit_status = status_callback
    elif quiet:
        emit_status = lambda _message: None
    else:
        emit_status = lambda message: print(message, flush=True)

    runner = RLRolloutRunner(
        network,
        training_opponent=training_opponent,
        schema=schema,
        gamma=gamma,
        max_pool_size=shared_pool_size,
        safety=safety_config,
    )
    parallel_summary = _new_parallel_summary(workers)
    if workers == "auto":
        autotuner = RetainedRLWorkerAutotuner(
            total_iterations=iterations,
            games_per_iteration=games_per_iteration,
            safety=safety_config,
            benchmark_fraction=autotune_fraction,
            minimum_gain=autotune_minimum_gain,
            candidates=worker_candidates,
            status_callback=emit_status,
        )
        selected_workers = 1
    else:
        autotuner = None
        selected_workers, was_capped, cap_reason = cap_parallel_workers(
            workers,
            safety_config,
        )
        if was_capped:
            emit_status(
                f"Fixed RL workers reduced from {workers} to {selected_workers} "
                f"by resource preflight: {cap_reason}."
            )
            parallel_summary["safety_capped"] = True
            parallel_summary["fallback_history"].append({
                "from_workers": workers,
                "to_workers": selected_workers,
                "completed_games": 0,
                "reason": cap_reason,
                "phase": "preflight",
                "rl_phase": "rollout",
                "iteration": 0,
            })

    value_loss_window = deque(maxlen=moving_average_window)
    win_rate_window = deque(maxlen=moving_average_window)

    start_time = time.time()
    last_checkpoint_time = start_time
    configured_worker_target = None
    try:
        for iteration in range(1, iterations + 1):
            if autotuner is not None and not autotuner.finished:
                candidate = autotuner.current_workers
                capped, was_capped, cap_reason = cap_parallel_workers(
                    candidate,
                    safety_config,
                )
                if was_capped:
                    autotuner.reject_current_before_allocation(cap_reason)
                    candidate = autotuner.optimal_workers
                selected_workers = candidate
            elif autotuner is not None:
                selected_workers = autotuner.optimal_workers

            if selected_workers != configured_worker_target:
                runner.set_workers(selected_workers)
                configured_worker_target = selected_workers
            runner.sync_current(network)
            rollout_started = time.perf_counter()
            rollout_results, rollout_info = runner.collect_training_iteration(
                iteration - 1,
                games_per_iteration,
                effective_seed,
            )
            rollout_elapsed = time.perf_counter() - rollout_started
            _merge_parallel_summary(
                parallel_summary,
                rollout_info,
                phase="rollout",
                iteration=iteration,
            )
            if rollout_info.fallback_count:
                emit_status(
                    f"RL iteration {iteration} retained completed games and "
                    f"reduced workers to {rollout_info.final_workers}: "
                    f"{rollout_info.fallback_history[-1]['reason']}."
                )
            if autotuner is not None and not autotuner.finished:
                autotuner.record_iteration(
                    rollout_elapsed,
                    rollout_info,
                    iteration,
                )

            batch = []
            event_totals = EventStats()
            wins = 0
            for result in rollout_results:
                batch.extend(result["samples"])
                _add_worker_event_stats(event_totals, result["event_stats"])
                if result["winner"] == result["learner_position"]:
                    wins += 1

            win_rate_window.append(wins / games_per_iteration)

            if not batch:
                if progress_callback is not None:
                    progress_callback(iteration, iterations)
                continue

            exact_sample_bytes = 0
            for sample in batch:
                exact_sample_bytes += int(getattr(sample.x, "nbytes", 0))
                exact_sample_bytes += int(getattr(sample.legal_mask, "nbytes", 0))
            matrix_workspace_bytes = max(1, exact_sample_bytes * 4)
            if network.device == "cpu":
                ensure_ram_available(
                    matrix_workspace_bytes,
                    safety_config.memory_reserve_mb,
                    f"RL iteration {iteration} batch assembly",
                )
            else:
                effective_gpu_bytes = effective_gpu_available_bytes()
                if (
                    effective_gpu_bytes is not None
                    and effective_gpu_bytes < matrix_workspace_bytes
                ):
                    raise MemorySafetyError(
                        f"RL iteration {iteration} needs about "
                        f"{matrix_workspace_bytes / MIB:.1f} MiB of GPU workspace, "
                        f"but only {effective_gpu_bytes / MIB:.1f} MiB is effectively free. "
                        "Restart with --device cpu or a smaller games-per-iteration."
                    )

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

            value_returns = None
            policy_signal = policy_rewards
            if use_value_head:
                values = network.predict_values(x_batch)
                policy_signal = policy_rewards - values
                value_returns = policy_rewards
            else:
                network.forward(x_batch)

            if normalize_advantages:
                signal_std = float(xp.std(policy_signal))
                if signal_std > REWARD_ZERO_EPSILON:
                    policy_signal = (
                        policy_signal - float(xp.mean(policy_signal))
                    ) / signal_std

            metrics = network.backward_policy_gradient(
                action_indices,
                policy_signal,
                legal_masks=legal_masks,
                entropy_coef=entropy_coef,
                value_returns=value_returns,
                value_coef=value_coef,
                clip_grad_norm=clip_grad_norm,
            )

            if use_value_head:
                value_loss_window.append(metrics["value_loss"])

            if training_opponent == "self_play" and iteration % pool_interval == 0:
                runner.append_pool_snapshot(network)

            if iteration % log_interval == 0 and not quiet:
                reward_summary = _reward_signal_summary(batch, xp)
                win_label = (
                    "vs pool"
                    if training_opponent == "self_play"
                    else "vs heuristic"
                )
                pool_suffix = (
                    f" | pool: {len(runner.bank.pool_slots)}"
                    if training_opponent == "self_play"
                    else ""
                )
                win_rate_moving_avg = sum(win_rate_window) / len(win_rate_window)
                value_suffix = ""
                if use_value_head:
                    value_loss_moving_avg = (
                        sum(value_loss_window) / len(value_loss_window)
                    )
                    value_suffix = (
                        f" | value loss: {metrics['value_loss']:.3f}"
                        f" (avg/{len(value_loss_window)}: {value_loss_moving_avg:.3f})"
                        f" | advantage mean: {float(xp.mean(policy_signal)):+.3f}"
                    )
                print(
                    f"Iteration {iteration} | decisions: {len(batch)} | "
                    "reward mean/std/min/max: "
                    f"{reward_summary['reward_mean']:+.2f}/"
                    f"{reward_summary['reward_std']:.2f}/"
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
                    f"wins {win_label}: {wins}/{games_per_iteration}"
                    f" (avg/{len(win_rate_window)}: {win_rate_moving_avg:.1%})"
                    f"{pool_suffix} | grad: {_gradient_log_text(metrics)}"
                    f"{value_suffix}"
                )

            if iteration % checkpoint_interval == 0:
                network.save(rl_weights_path)
                runner.sync_current(network)
                evaluation_seed = game_seed(
                    effective_seed,
                    iterations * games_per_iteration + iteration,
                )
                evaluation_results, evaluation_info = (
                    runner.evaluate_current_against_heuristic(
                        evaluation_games,
                        evaluation_seed,
                    )
                )
                _merge_parallel_summary(
                    parallel_summary,
                    evaluation_info,
                    phase="evaluation",
                    iteration=iteration,
                )
                if evaluation_info.fallback_count:
                    emit_status(
                        f"RL checkpoint evaluation at iteration {iteration} "
                        f"retained completed games and reduced workers to "
                        f"{evaluation_info.final_workers}: "
                        f"{evaluation_info.fallback_history[-1]['reason']}."
                    )
                evaluation_wins = sum(
                    result["winner"] == result["learner_position"]
                    for result in evaluation_results
                )
                evaluation_draws = sum(
                    result["winner"] == -1
                    for result in evaluation_results
                )
                win_rate = evaluation_wins / evaluation_games
                draw_rate = evaluation_draws / evaluation_games
                now = time.time()
                checkpoint_elapsed = now - last_checkpoint_time
                last_checkpoint_time = now
                if not quiet:
                    print(
                        f"  [checkpoint] saved {rl_weights_path} | "
                        f"time since previous checkpoint: "
                        f"{format_duration(checkpoint_elapsed)} | "
                        f"deterministic RL vs heuristic: {win_rate:.1%} wins, "
                        f"{draw_rate:.1%} draws ({evaluation_games} games)"
                    )

            if progress_callback is not None:
                progress_callback(iteration, iterations)
    finally:
        final_runtime_workers = runner.worker_count
        runner.close()

    if autotuner is not None:
        selected_workers = autotuner.optimal_workers
        autotune_summary = autotuner.to_dict()
    else:
        selected_workers = final_runtime_workers
        autotune_summary = {
            "optimal_workers": selected_workers,
            "candidate_workers": [workers],
            "benchmark_fraction": 0.0,
            "minimum_gain": autotune_minimum_gain,
            "iterations_per_test": 0,
            "games_per_test": 0,
            "reused_iteration_count": 0,
            "reused_game_count": 0,
            "attempts": [],
        }
    parallel_summary["final_workers"] = final_runtime_workers
    network.save(rl_weights_path)
    elapsed_time = time.time() - start_time
    if not quiet:
        print(f"\nTraining complete. Total elapsed time: {format_duration(elapsed_time)}.")
        print(f"Final weights: {rl_weights_path}")

    return {
        "iterations": iterations,
        "games_per_iteration": games_per_iteration,
        "training_opponent": training_opponent,
        "learning_rate": learning_rate,
        "entropy_coef": entropy_coef,
        "use_value_head": use_value_head,
        "value_coef": value_coef if use_value_head else None,
        "gamma": gamma,
        "reward_schema": reward_schema,
        "clip_grad_norm": clip_grad_norm,
        "normalize_advantages": normalize_advantages,
        "moving_average_window": moving_average_window,
        "seed": seed,
        "effective_seed": effective_seed,
        "device": network.device,
        "requested_device": requested_device,
        "device_fallback_reason": device_fallback_reason,
        "requested_workers": workers,
        "selected_workers": selected_workers,
        "autotune": autotune_summary,
        "parallel": parallel_summary,
        "rl_weights_path": rl_weights_path,
        "duration_s": elapsed_time,
    }


def add_optional_rl_arguments(parser):
    """Add self-play hyperparameter and rollout-resource flags to ``parser``."""
    group = parser.add_argument_group("optional reinforcement-learning controls")
    group.add_argument("--iterations", type=int, default=1000, help="Training iterations.")
    group.add_argument(
        "--games-per-iteration", type=int, default=40, help="Games played per iteration."
    )
    group.add_argument(
        "--training-opponent",
        choices=("self_play", "heuristic"),
        default=TRAINING_OPPONENT,
        help="Play against a pool of frozen snapshots or the fixed heuristic agent.",
    )
    group.add_argument("--learning-rate", type=float, default=0.001)
    group.add_argument("--entropy-coef", type=float, default=0.01)
    group.add_argument("--log-interval", type=int, default=10)
    group.add_argument("--checkpoint-interval", type=int, default=50)
    group.add_argument("--pool-interval", type=int, default=10)
    group.add_argument("--max-pool-size", type=int, default=50)
    group.add_argument("--evaluation-games", type=int, default=200)
    group.add_argument("--sl-weights-path", default=SL_WEIGHTS)
    group.add_argument("--rl-weights-path", default=RL_WEIGHTS)
    group.add_argument(
        "--value-head",
        action="store_true",
        help=(
            "Train a linear V(s) baseline (the critic) and use reward-minus-value "
            "policy advantages. Direct REINFORCE (critic off) remains the default."
        ),
    )
    group.add_argument("--value-coef", type=float, default=VALUE_COEF)
    group.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help="Terminal-reward discount per remaining real decision (1.0 = no discount).",
    )
    group.add_argument(
        "--reward-schema",
        choices=tuple(REWARD_SCHEMAS),
        default=DEFAULT_REWARD_SCHEMA,
        help="Named preset for the terminal/event reward constants.",
    )
    group.add_argument(
        "--clip-grad-norm",
        type=float,
        default=DEFAULT_CLIP_GRAD_NORM,
        help="Gradient-norm clipping threshold for the policy-gradient update.",
    )
    group.add_argument(
        "--normalize-advantages",
        dest="normalize_advantages",
        action="store_true",
        default=DEFAULT_NORMALIZE_ADVANTAGES,
        help="Standardize the policy signal per batch (mean 0, std 1) before the "
        "gradient step. Off by default (matching prior behavior).",
    )
    group.add_argument(
        "--no-normalize-advantages",
        dest="normalize_advantages",
        action="store_false",
        help="Disable per-batch advantage normalization.",
    )
    group.add_argument(
        "--moving-average-window",
        type=int,
        default=DEFAULT_MOVING_AVERAGE_WINDOW,
        help="Trailing-iteration window for the value-loss/win-rate moving averages "
        "in the log (point values are noisy; use this for judging a plateau).",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fix random/numpy state for reproducible comparisons between configurations.",
    )
    group.add_argument(
        "--device",
        choices=DEVICES,
        default=DEFAULT_DEVICE,
        help="Array backend: 'auto' matches GPU_ENABLED (CuPy when installed, "
        "else NumPy) -- unchanged from prior behavior. 'cpu'/'gpu' force one "
        "backend regardless of what's installed/enabled globally.",
    )
    group.add_argument(
        "--rl-workers",
        type=parse_rl_worker_count,
        default=DEFAULT_RL_WORKERS,
        help=(
            f"CPU-only rollout workers or 'auto' for retained online tuning "
            f"(maximum {MAX_PARALLEL_WORKERS})."
        ),
    )
    group.add_argument(
        "--rl-autotune-fraction",
        type=float,
        default=DEFAULT_RL_AUTOTUNE_FRACTION,
        help="Fraction of planned iterations retained by each worker-count test.",
    )
    group.add_argument(
        "--rl-autotune-min-gain",
        type=float,
        default=DEFAULT_RL_MINIMUM_GAIN,
        help="Minimum marginal rollout-throughput gain for a larger pool.",
    )
    group.add_argument("--rl-memory-reserve-mb", type=int, default=512)
    group.add_argument("--rl-estimated-worker-mb", type=int, default=256)
    group.add_argument("--rl-max-worker-rss-mb", type=int, default=1024)
    return parser


def parse_args(argv=None):
    """Parse optional self-play training controls."""
    parser = argparse.ArgumentParser(
        description="Train the domino policy with reinforcement learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_optional_rl_arguments(parser)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train(
        iterations=args.iterations,
        games_per_iteration=args.games_per_iteration,
        training_opponent=args.training_opponent,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        pool_interval=args.pool_interval,
        max_pool_size=args.max_pool_size,
        evaluation_games=args.evaluation_games,
        sl_weights_path=args.sl_weights_path,
        rl_weights_path=args.rl_weights_path,
        use_value_head=args.value_head,
        value_coef=args.value_coef,
        gamma=args.gamma,
        reward_schema=args.reward_schema,
        clip_grad_norm=args.clip_grad_norm,
        normalize_advantages=args.normalize_advantages,
        moving_average_window=args.moving_average_window,
        seed=args.seed,
        device=args.device,
        workers=args.rl_workers,
        safety_config=ParallelSafetyConfig(
            memory_reserve_mb=args.rl_memory_reserve_mb,
            estimated_worker_mb=args.rl_estimated_worker_mb,
            max_worker_rss_mb=args.rl_max_worker_rss_mb,
        ),
        autotune_fraction=args.rl_autotune_fraction,
        autotune_minimum_gain=args.rl_autotune_min_gain,
    )


if __name__ == "__main__":
    main()
