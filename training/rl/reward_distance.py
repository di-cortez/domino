"""Canonical reward-discount distance modes for RL trajectories."""

TURN_DISTANCE = "turn"
DECISION_DISTANCE = "decision"

# Both clocks count engine turns: the opponent's actions and every draw count
# too, so the same gamma discounts harder than it would over the learner's own
# decisions alone, and the discount means "how much happened" rather than "how
# many choices of mine remain".
#
# The one-factor sweep measured all four modes and put `turn-turn` first,
# +0.218 pp final and +0.296 pp AUC over the `decision-decision` it replaces.
# The final-level gain sits exactly on the +/-0.22 pp reading ruler while the
# AUC gain clears its own comfortably, which is the signature of reaching the
# same level earlier rather than reaching a higher one. Under a time budget,
# arriving earlier is a real gain. See
# references/resumo_expandido/analises_agente_atual/REPORT.md.
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
