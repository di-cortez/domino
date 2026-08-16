"""Closed registry for the supported two-player draw-domino rulesets."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_RULESET_NAME = "double-six"
RULESET_NAMES = (
    "double-six",
    "double-five",
    "double-four",
    "double-three",
)


@dataclass(frozen=True)
class DominoRuleset:
    """Immutable tile-set and deal geometry for one named game variant."""

    name: str
    max_pip: int
    hand_size: int

    @property
    def pip_count(self):
        """Return the number of legal pip values, including zero."""
        return self.max_pip + 1

    @property
    def tile_count(self):
        """Return the number of unique unordered tiles in the ruleset."""
        return self.pip_count * (self.pip_count + 1) // 2

    @property
    def all_tiles(self):
        """Return tiles in the historical canonical policy/deal order."""
        return tuple(
            (left, right)
            for left in range(self.pip_count)
            for right in range(left, self.pip_count)
        )

    def initial_stock_size(self, player_count=2):
        """Return stock size after dealing all players their initial hands."""
        return self.tile_count - int(player_count) * self.hand_size


RULESETS = {
    "double-six": DominoRuleset("double-six", max_pip=6, hand_size=7),
    "double-five": DominoRuleset("double-five", max_pip=5, hand_size=6),
    "double-four": DominoRuleset("double-four", max_pip=4, hand_size=5),
    "double-three": DominoRuleset("double-three", max_pip=3, hand_size=4),
}


def resolve_ruleset(value=None):
    """Return one supported ruleset object or reject an unknown value."""
    if value is None:
        return RULESETS[DEFAULT_RULESET_NAME]
    if isinstance(value, DominoRuleset):
        registered = RULESETS.get(value.name)
        if registered != value:
            raise ValueError(
                f"Ruleset object {value!r} is not one of the registered rulesets."
            )
        return value
    if not isinstance(value, str) or value not in RULESETS:
        allowed = ", ".join(RULESET_NAMES)
        raise ValueError(f"Unknown ruleset {value!r}; choose one of: {allowed}.")
    return RULESETS[value]


def state_ruleset_name(state):
    """Read a state's ruleset, treating unnamed legacy states as double-six."""
    name = state.get("ruleset_name")
    if name is None:
        return DEFAULT_RULESET_NAME
    return resolve_ruleset(name).name


def validate_state_ruleset(state, expected_ruleset):
    """Validate that a state belongs to the receiving persistent component."""
    expected = resolve_ruleset(expected_ruleset)
    explicit_name = state.get("ruleset_name")
    if explicit_name is None and expected.name != DEFAULT_RULESET_NAME:
        raise ValueError(
            "State has no ruleset_name and can only be interpreted as "
            f"{DEFAULT_RULESET_NAME}, not {expected.name}."
        )
    actual_name = state_ruleset_name(state)
    if actual_name != expected.name:
        raise ValueError(
            f"State ruleset {actual_name!r} does not match expected ruleset "
            f"{expected.name!r}."
        )
    return expected
