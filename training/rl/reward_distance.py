"""Canonical reward-discount distance modes for RL trajectories."""

TURN_DISTANCE = "turn"
DECISION_DISTANCE = "decision"

DEFAULT_REWARD_DISTANCE_MODE = "turn-turn"
HISTORICAL_REWARD_DISTANCE_MODE = "turn-decision"
HISTORICAL_GAMMA_F = 1.0

REWARD_DISTANCE_MODES = (
    "turn-turn",
    "decision-decision",
    "turn-decision",
    "decision-turn",
)

_DISTANCE_METRICS_BY_MODE = {
    mode: tuple(mode.split("-", 1))
    for mode in REWARD_DISTANCE_MODES
}


def resolve_reward_distance_mode(mode):
    """Return ``(local_metric, terminal_metric)`` for one public mode."""
    try:
        return _DISTANCE_METRICS_BY_MODE[str(mode)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown reward distance mode {mode!r}; expected one of "
            f"{', '.join(REWARD_DISTANCE_MODES)}."
        ) from exc
