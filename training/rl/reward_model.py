"""Pure reward semantics for the domino RL objective.

This module owns *what the reward means* and nothing else: no trajectory
mutation, no engine stepping, no PPO, no baselines. ``training.rl.rollout``
applies these formulas during a game and any offline analysis that needs the
same numbers imports them from here rather than reimplementing them.

The objective is decomposed into four semantically separate layers so that an
ablation on one of them cannot silently change another:

1. event semantics -- what a terminal ending or a draw/pass event *is*;
2. relative component weights -- how much each of them is worth;
3. temporal discounting -- ``gamma_f`` and ``gamma_i`` (owned by the rollout);
4. terminal/immediate mixture -- ``reward_eta`` (owned by the rollout).

The return credited to one learner decision ``t`` is

    G(t) = (1 - reward_eta) * G_T(t) + reward_eta * G_I(t)

with the terminal half

    G_T(t) = gamma_f ** k_T(t) * U_T
    U_T    = (a_E * R_E + a_B * R_B) / max(a_E, a_B)

and the immediate half

    G_I(t) = (a_D * G_D(t) + a_P * G_P(t)) / max(a_D, a_P)
    G_D(t) = sum over future draw events of gamma_i ** k_e(t) * r_D(e)
    G_P(t) = sum over future pass events of gamma_i ** k_e(t) * r_P(e)

Layers 1 and 2 live here: ``terminal_reward_components`` produces ``R_E`` and
``R_B``, ``unit_event_reward`` produces ``r_D`` and ``r_P``, and
``resolved_reward_scales`` turns the four raw weights into the normalized
scales the rollout multiplies by. Dividing each pair by its own larger member
keeps the more important component of each pair on unit scale, so ``m(Delta_p)``
answers "how decisive was this blocked result" while ``a_B / a_E`` separately
answers "how much is a blocked result worth at all".
"""

import math

from middleware.domino_engine import (
    WIN_REASON_BLOCKED_FEWEST_PIPS,
    WIN_REASON_BLOCKED_FEWEST_TILES,
    WIN_REASON_BLOCKED_LAST_VALID_PLAY,
    WIN_REASON_EMPTY_HAND,
)


# A blocked game is still a decided game, so its magnitude never collapses to
# zero: a blocked win worth nothing would be indistinguishable from a draw the
# engine cannot produce. The ceiling is reached once the pip margin saturates.
BLOCKED_REWARD_FLOOR = 0.1
BLOCKED_REWARD_CEILING = 1.0

# The three endings the engine resolves through ``_resolve_blocked_winner``.
# All of them are blocked endings; the tie-breaker that picked the winner says
# how the pips compared, not whether the game was blocked.
BLOCKED_WIN_REASONS = frozenset({
    WIN_REASON_BLOCKED_FEWEST_PIPS,
    WIN_REASON_BLOCKED_FEWEST_TILES,
    WIN_REASON_BLOCKED_LAST_VALID_PLAY,
})
KNOWN_TERMINAL_WIN_REASONS = frozenset({WIN_REASON_EMPTY_HAND}) | (
    BLOCKED_WIN_REASONS
)

# Per-event decay applied to a draw/pass reward as it is credited backwards to
# the real decisions that preceded the event.
DEFAULT_GAMMA_I = 0.90

# Terminal discount applied using the terminal metric selected by the
# reward-distance mode. The historical value was 1.0 (no discount).
DEFAULT_GAMMA_F = 0.95

# Convex mixture weight between the terminal and immediate returns of one
# decision: 0.0 trains on the terminal outcome alone, 1.0 on event shaping.
#
# 0.115 is the value the reward-architecture analysis derived and two runs
# confirmed: it restores the local half the relative weight it carried under
# the superseded reward, bringing the live local/terminal ratio back to
# ~0.29x from 2.45x. See analysis/recompensa_anterior_vs_atual.
DEFAULT_REWARD_ETA = 0.115

# Neutral defaults: equal maximum scale inside each pair, so neither ratio is
# silently preferred before an experiment chooses one. The redesign
# deliberately does not bury a preferred ratio in a private constant.
DEFAULT_TERMINAL_EMPTY_HAND_WEIGHT = 1.0
DEFAULT_TERMINAL_BLOCKED_WEIGHT = 1.0
DEFAULT_IMMEDIATE_DRAW_WEIGHT = 1.0
DEFAULT_IMMEDIATE_PASS_WEIGHT = 1.0

DRAW_EVENT = "draw"
PASS_EVENT = "pass"
EVENT_KINDS = (DRAW_EVENT, PASS_EVENT)


def validate_reward_weight(value, *, name):
    """Return one finite non-negative reward weight as a float."""
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(weight):
        raise ValueError(f"{name} must be finite")
    if weight < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return weight


def normalize_weight_pair(first, second, *, first_name, second_name):
    """Return one weight pair rescaled so its larger member is exactly 1.

    Only the ratio of a pair carries meaning, so dividing by ``max`` keeps the
    dominant component on unit scale and leaves the other one expressing the
    ratio. The pair may not be ``(0, 0)``: that would delete a whole half of
    the objective rather than reweight it.
    """
    first_weight = validate_reward_weight(first, name=first_name)
    second_weight = validate_reward_weight(second, name=second_name)
    norm = max(first_weight, second_weight)
    if norm <= 0.0:
        raise ValueError(
            f"{first_name} and {second_name} cannot both be zero"
        )
    return first_weight / norm, second_weight / norm, norm


def blocked_reward_magnitude(pip_margin, max_pip):
    """Return ``m(Delta_p)`` for one blocked ending.

        m(Delta_p) = 0.1 + 0.9 * min(Delta_p / (2 * max_pip), 1)

    The saturation margin ``S = 2 * max_pip`` is derived from the ruleset
    rather than tabulated per ruleset name, so a new ruleset needs no reward
    change. ``Delta_p == 0`` is legal and maps to the floor: the engine
    resolves a pip tie by fewest tiles or by the last valid play, which is a
    real but minimally decisive blocked win.
    """
    margin = float(pip_margin)
    if margin < 0.0:
        raise ValueError(
            f"pip_margin must be non-negative, got {pip_margin!r}; the blocked "
            "winner never holds more pips than the loser"
        )
    saturation = 2.0 * float(max_pip)
    if saturation <= 0.0:
        raise ValueError(f"max_pip must be positive, got {max_pip!r}")
    span = BLOCKED_REWARD_CEILING - BLOCKED_REWARD_FLOOR
    return BLOCKED_REWARD_FLOOR + span * min(margin / saturation, 1.0)


def remaining_pips(hand):
    """Return the pip total still held in one hand."""
    return sum(tile[0] + tile[1] for tile in hand)


def terminal_reward_components(engine, learner_position):
    """Decompose one finished game into ``R_E``/``R_B`` and its provenance.

    Exactly one of the two components is non-zero: an ending is either an
    empty hand or a block, never both. The remaining keys are the observations
    the blocked branch was derived from, kept so run diagnostics and offline
    analysis can audit a reward without recomputing it from the engine.
    """
    if not engine.game_over:
        raise ValueError("Terminal reward requested for an unfinished game.")
    winner = engine.winner
    if winner is None:
        raise ValueError("Finished game has no winner.")
    win_reason = engine.win_reason
    if win_reason not in KNOWN_TERMINAL_WIN_REASONS:
        raise ValueError(
            f"Unknown terminal win reason {win_reason!r}; expected one of "
            f"{', '.join(sorted(KNOWN_TERMINAL_WIN_REASONS))}."
        )
    sign = 1.0 if int(winner) == int(learner_position) else -1.0
    if win_reason == WIN_REASON_EMPTY_HAND:
        return {
            "win_reason": win_reason,
            "empty_hand_component": sign,
            "blocked_component": 0.0,
            "winner_final_pips": None,
            "loser_final_pips": None,
            "pip_margin": None,
            "blocked_magnitude": None,
        }
    if engine.player_count != 2:
        raise ValueError(
            "Blocked terminal reward is defined for the canonical two-player "
            f"training game, got player_count={engine.player_count}."
        )
    winner_index = int(winner)
    loser_index = 1 - winner_index
    winner_pips = remaining_pips(engine.hands[winner_index])
    loser_pips = remaining_pips(engine.hands[loser_index])
    pip_margin = loser_pips - winner_pips
    magnitude = blocked_reward_magnitude(pip_margin, engine.ruleset.max_pip)
    return {
        "win_reason": win_reason,
        "empty_hand_component": 0.0,
        "blocked_component": sign * magnitude,
        "winner_final_pips": winner_pips,
        "loser_final_pips": loser_pips,
        "pip_margin": pip_margin,
        "blocked_magnitude": magnitude,
    }


def combine_terminal_components(
    components, *, empty_hand_scale, blocked_scale
):
    """Return the undiscounted terminal utility ``U_T`` of one ending.

    The scales are already normalized by ``max(a_E, a_B)``, so this is exactly
    ``(a_E * R_E + a_B * R_B) / max(a_E, a_B)``. No temporal discounting and no
    ``reward_eta`` mixing happen here: both belong to later layers.
    """
    return (
        empty_hand_scale * components["empty_hand_component"]
        + blocked_scale * components["blocked_component"]
    )


def unit_event_reward(event_kind, *, by_learner):
    """Return ``r_D``/``r_P`` for one draw or pass event.

    Both event classes are normalized to the same unit scale, so their
    relative importance is expressed by ``a_D`` and ``a_P`` alone. Forcing the
    opponent into the event is favorable; being forced into it is not.
    """
    if event_kind not in EVENT_KINDS:
        raise ValueError(
            f"Unknown event kind {event_kind!r}; expected one of "
            f"{', '.join(EVENT_KINDS)}."
        )
    return -1.0 if by_learner else 1.0


def scaled_event_reward(
    event_kind, *, by_learner, draw_scale, pass_scale
):
    """Return one weighted unit event reward, before ``gamma_i`` decay.

    Weighting each event before it is accumulated is algebraically identical
    to accumulating ``G_D`` and ``G_P`` separately and combining them
    afterwards, because the decayed sum is linear in the event value. The
    rollout uses this form because it needs no per-step draw/pass split.
    """
    scale = draw_scale if event_kind == DRAW_EVENT else pass_scale
    return scale * unit_event_reward(event_kind, by_learner=by_learner)


def resolved_reward_scales(
    *,
    terminal_empty_hand_weight=DEFAULT_TERMINAL_EMPTY_HAND_WEIGHT,
    terminal_blocked_weight=DEFAULT_TERMINAL_BLOCKED_WEIGHT,
    immediate_draw_weight=DEFAULT_IMMEDIATE_DRAW_WEIGHT,
    immediate_pass_weight=DEFAULT_IMMEDIATE_PASS_WEIGHT,
):
    """Validate the four raw weights and derive their normalized scales.

    Both spellings are returned: the raw weights preserve exactly what the
    experiment asked for, and the derived scales are what every rollout worker
    multiplies by, so the normalization happens once per run rather than once
    per event.
    """
    empty_hand_scale, blocked_scale, terminal_norm = normalize_weight_pair(
        terminal_empty_hand_weight,
        terminal_blocked_weight,
        first_name="terminal_empty_hand_weight",
        second_name="terminal_blocked_weight",
    )
    draw_scale, pass_scale, immediate_norm = normalize_weight_pair(
        immediate_draw_weight,
        immediate_pass_weight,
        first_name="immediate_draw_weight",
        second_name="immediate_pass_weight",
    )
    return {
        "terminal_empty_hand_weight": float(terminal_empty_hand_weight),
        "terminal_blocked_weight": float(terminal_blocked_weight),
        "terminal_weight_norm": terminal_norm,
        "empty_hand_scale": empty_hand_scale,
        "blocked_scale": blocked_scale,
        "immediate_draw_weight": float(immediate_draw_weight),
        "immediate_pass_weight": float(immediate_pass_weight),
        "immediate_weight_norm": immediate_norm,
        "draw_scale": draw_scale,
        "pass_scale": pass_scale,
        "blocked_reward_floor": BLOCKED_REWARD_FLOOR,
    }
