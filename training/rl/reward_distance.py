"""Canonical reward-discount distance modes for RL trajectories."""

TURN_DISTANCE = "turn"
DECISION_DISTANCE = "decision"

# Both clocks count the learner's own decisions. A turn clock also counts the
# opponent's actions and every draw, so the same gamma discounts far harder
# under it; keeping both halves on the decision clock makes the discount mean
# "how many choices of mine remain" rather than "how much happened".
DEFAULT_REWARD_DISTANCE_MODE = "decision-decision"
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
