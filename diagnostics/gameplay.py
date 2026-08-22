"""Agent construction and single-game execution for diagnostics.

This leaf module contains the gameplay operations shared by the pairwise
orchestrator and its multiprocessing runner.  It deliberately imports neither
module, keeping the diagnostics dependency graph acyclic.
"""

import contextlib
import io
import math
import time
from pathlib import Path

import numpy as np

from middleware.domino_engine import DominoEngine
from middleware.rulesets import DEFAULT_RULESET_NAME, resolve_ruleset
from utils.ruleset_paths import default_rl_weights_path, default_sl_weights_path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_AGENTS = ("rl", "neural", "heuristic", "random")
DEFAULT_WEIGHTS = {
    "rl": ROOT / "models" / "domino_rl_weights.npz",
    "neural": ROOT / "models" / "domino_sl_weights.npz",
}
VALUE_WEIGHT_NAMES = ("Wv", "bv")


def normalize_agent_name(agent_name):
    """Return the canonical diagnostics name for an agent."""
    normalized = agent_name.strip().lower()
    if normalized not in CANONICAL_AGENTS:
        raise ValueError(f"Unknown agent {agent_name!r}. Options: {CANONICAL_AGENTS}")
    return normalized


def resolve_weights_path(
    agent_name,
    weights_path=None,
    ruleset=DEFAULT_RULESET_NAME,
):
    """Return an existing checkpoint path for a checkpoint-backed agent."""
    agent_name = normalize_agent_name(agent_name)
    if agent_name not in DEFAULT_WEIGHTS:
        return None

    if weights_path is None:
        path = (
            default_rl_weights_path(ruleset)
            if agent_name == "rl"
            else default_sl_weights_path(ruleset)
        )
        if not path.is_absolute():
            path = ROOT / path
    else:
        path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {agent_name} checkpoint: {path}. "
            "Generate the model first or pass an explicit weights path."
        )
    return path


def _checkpoint_has_value_head(path):
    """Return whether an RL checkpoint contains the complete value head."""
    with np.load(path, allow_pickle=False) as weights:
        return all(name in weights.files for name in VALUE_WEIGHT_NAMES)


def create_agent(
    agent_name,
    weights_path=None,
    ruleset=DEFAULT_RULESET_NAME,
    *,
    use_opponent_suit_features=True,
):
    """Create an agent by name, importing checkpoint-backed classes only when used.

    ``use_opponent_suit_features`` reaches only the checkpoint-backed agents. A
    network must be evaluated under the encoding it trained with, and the
    heuristic reference opponent deliberately keeps its exact model either way.
    """
    agent_name = normalize_agent_name(agent_name)
    ruleset = resolve_ruleset(ruleset)

    if agent_name == "rl":
        from agents.rl_agent import RLAgent

        path = resolve_weights_path("rl", weights_path, ruleset)
        return RLAgent.load(
            str(path),
            mode="evaluation",
            use_value_head=_checkpoint_has_value_head(path),
            ruleset=ruleset,
            use_opponent_suit_features=use_opponent_suit_features,
        )
    if agent_name == "neural":
        from agents.neural_agent import NeuralAgent

        return NeuralAgent.load(
            str(resolve_weights_path("neural", weights_path, ruleset)),
            ruleset=ruleset,
            use_opponent_suit_features=use_opponent_suit_features,
        )
    if agent_name == "heuristic":
        from agents.heuristic_agent import StrategicAgent

        return StrategicAgent(ruleset=ruleset)
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
    """Update choice counters and return whether this was a real decision."""
    if is_forced_draw(legal_actions):
        stats["agent_forced_draws"] += 1
        return False

    if is_forced_pass(legal_actions):
        stats["agent_forced_passes"] += 1
        return False

    option_count = count_tile_play_options(legal_actions)
    stats["agent_choice_histogram"][str(option_count)] = (
        stats["agent_choice_histogram"].get(str(option_count), 0) + 1
    )

    if option_count >= 2:
        stats["agent_real_decision_turns"] += 1
        return True

    stats["agent_forced_tile_turns"] += 1
    return False


def _new_value_head_stats(agent):
    """Create sufficient statistics when the evaluated agent has a critic."""
    network = getattr(agent, "network", None)
    if network is None or not getattr(network, "use_value_head", False):
        return None
    return {
        "sample_count": 0,
        "finite_count": 0,
        "nonfinite_count": 0,
        "sum": 0.0,
        "sum_squares": 0.0,
        "min": None,
        "max": None,
    }


def _record_value_head_prediction(agent, stats):
    """Record V(s) from the policy forward cache without another forward pass."""
    if stats is None:
        return
    network = agent.network
    hidden = network.cache.get(network.last_hidden_activation_key)
    if hidden is None:
        raise RuntimeError("RL value diagnostics require a completed policy forward pass.")
    value = network.xp.dot(network.Wv, hidden) + network.bv
    if hasattr(value, "get"):
        value = value.get()
    scalar = float(np.asarray(value).reshape(-1)[0])
    stats["sample_count"] += 1
    if not math.isfinite(scalar):
        stats["nonfinite_count"] += 1
        return
    stats["finite_count"] += 1
    stats["sum"] += scalar
    stats["sum_squares"] += scalar * scalar
    stats["min"] = scalar if stats["min"] is None else min(stats["min"], scalar)
    stats["max"] = scalar if stats["max"] is None else max(stats["max"], scalar)


def _add_game_runtime(runtime_profile, section, started):
    if runtime_profile is None or started is None:
        return
    sections = runtime_profile.setdefault("sections_seconds", {})
    sections[section] = sections.get(section, 0.0) + (
        time.perf_counter() - started
    )


def _game_runtime_start(runtime_profile):
    return time.perf_counter() if runtime_profile is not None else None


def _play_game_unprofiled(
    agent,
    opponent,
    agent_position,
    suppress_agent_output,
    ruleset=DEFAULT_RULESET_NAME,
):
    """Profiler-free diagnostic hot path for non-sampled games."""
    agents = [None, None]
    agents[agent_position] = agent
    agents[1 - agent_position] = opponent
    engine = DominoEngine(player_count=2, ruleset=ruleset)
    choice_stats = empty_choice_stats()
    value_head_stats = _new_value_head_stats(agent)
    while not engine.game_over:
        state = engine._get_state()
        current_player = state["current_player"]
        legal_actions = engine.valid_actions(current_player)
        evaluated_real_decision = False
        if current_player == agent_position:
            evaluated_real_decision = update_choice_stats(
                choice_stats,
                legal_actions,
            )
        if suppress_agent_output:
            with contextlib.redirect_stdout(io.StringIO()):
                action = agents[current_player].choose_move(state, legal_actions)
        else:
            action = agents[current_player].choose_move(state, legal_actions)
        if evaluated_real_decision:
            _record_value_head_prediction(agent, value_head_stats)
        engine.step(
            action,
            return_state=False,
            legal_actions=legal_actions,
        )

    final_state = engine.to_dict()
    winner = final_state["winner"]
    result = "win" if winner == agent_position else "loss"
    pips = [sum(tile[0] + tile[1] for tile in hand) for hand in final_state["hands"]]
    initial_hands = final_state["initial_hands"]
    record = {
        "game": None,
        "agent_position": agent_position,
        "result": result,
        "win_reason": final_state["win_reason"],
        "turns": final_state["turn"],
        "agent_initial_hand": initial_hands[agent_position],
        "opponent_initial_hand": initial_hands[1 - agent_position],
        "agent_remaining_pips": pips[agent_position],
        "opponent_remaining_pips": pips[1 - agent_position],
        **choice_stats,
    }
    if value_head_stats is not None:
        record["_agent_value_head_stats"] = value_head_stats
    return record


def play_game(
    agent,
    opponent,
    agent_position,
    suppress_agent_output=True,
    runtime_profile=None,
    ruleset=DEFAULT_RULESET_NAME,
):
    """Play one game and return the outcome from the evaluated agent's view."""
    if runtime_profile is None:
        return _play_game_unprofiled(
            agent,
            opponent,
            agent_position,
            suppress_agent_output,
            ruleset,
        )
    section_started = _game_runtime_start(runtime_profile)
    agents = [None, None]
    agents[agent_position] = agent
    agents[1 - agent_position] = opponent

    engine = DominoEngine(player_count=2, ruleset=ruleset)
    choice_stats = empty_choice_stats()
    value_head_stats = _new_value_head_stats(agent)
    _add_game_runtime(
        runtime_profile,
        "agent_pair_and_engine_initialization",
        section_started,
    )

    while not engine.game_over:
        section_started = _game_runtime_start(runtime_profile)
        state = engine._get_state()
        current_player = state["current_player"]
        legal_actions = engine.valid_actions(current_player)
        _add_game_runtime(
            runtime_profile,
            "state_and_legal_action_generation",
            section_started,
        )

        evaluated_real_decision = False
        if current_player == agent_position:
            section_started = _game_runtime_start(runtime_profile)
            evaluated_real_decision = update_choice_stats(
                choice_stats,
                legal_actions,
            )
            _add_game_runtime(
                runtime_profile,
                "evaluated_agent_choice_statistics",
                section_started,
            )

        section_started = _game_runtime_start(runtime_profile)
        if suppress_agent_output:
            with contextlib.redirect_stdout(io.StringIO()):
                action = agents[current_player].choose_move(state, legal_actions)
        else:
            action = agents[current_player].choose_move(state, legal_actions)
        _add_game_runtime(
            runtime_profile,
            (
                "evaluated_agent_decisions"
                if current_player == agent_position
                else "opponent_agent_decisions"
            ),
            section_started,
        )
        if evaluated_real_decision:
            section_started = _game_runtime_start(runtime_profile)
            _record_value_head_prediction(agent, value_head_stats)
            _add_game_runtime(
                runtime_profile,
                "evaluated_agent_value_head_statistics",
                section_started,
            )

        section_started = _game_runtime_start(runtime_profile)
        engine.step(
            action,
            return_state=False,
            legal_actions=legal_actions,
        )
        _add_game_runtime(
            runtime_profile,
            "engine_state_transition",
            section_started,
        )

    section_started = _game_runtime_start(runtime_profile)
    final_state = engine.to_dict()
    winner = final_state["winner"]
    outcome = "win" if winner == agent_position else "loss"

    pips = [sum(tile[0] + tile[1] for tile in hand) for hand in final_state["hands"]]
    initial_hands = final_state["initial_hands"]

    result = {
        "game": None,
        "agent_position": agent_position,
        "result": outcome,
        "win_reason": final_state["win_reason"],
        "turns": final_state["turn"],
        "agent_initial_hand": initial_hands[agent_position],
        "opponent_initial_hand": initial_hands[1 - agent_position],
        "agent_remaining_pips": pips[agent_position],
        "opponent_remaining_pips": pips[1 - agent_position],
        **choice_stats,
    }
    if value_head_stats is not None:
        result["_agent_value_head_stats"] = value_head_stats
    _add_game_runtime(
        runtime_profile,
        "final_state_and_outcome_serialization",
        section_started,
    )
    return result
