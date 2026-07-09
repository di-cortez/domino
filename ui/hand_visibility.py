"""
HUD hand-visibility rules.

This logic is UI-only. The domino engine decides legal play; this mixin decides
which hands should be displayed face up for each player mode.

Current rules:

- AI vs. AI: both hands are always visible; J/K do not change anything.
- Human vs. AI: the human hand is always visible; the AI hand starts hidden and
  can be toggled with J/K.
- Human vs. human: only the current player's hand is visible; J/K are disabled
  so two people can share the same computer fairly.
"""


class HandVisibilityMixin:
    """
    Mixin used by `GameController`.

    The final class must provide:
    - `agent_types`
    - `current_state()`
    - `_set_notification(message, duration_ms=...)`
    """

    def _human_count(self):
        return sum(1 for agent_type in self.agent_types if agent_type == "human")

    def _configure_hand_visibility_for_mode(self):
        """
        Reset the base visibility state when the player modes change.

        With exactly one human, only the AI hand is user-toggleable. In every
        other mode, visibility is derived directly from the rules above.
        """
        humans = self._human_count()

        if humans == 1:
            self.user_visible_hands = [
                agent_type == "human"
                for agent_type in self.agent_types
            ]
            return

        self.user_visible_hands = [True for _agent_type in self.agent_types]

    def can_toggle_hand_visibility(self, player):
        humans = self._human_count()

        if humans != 1:
            return False

        return self.agent_types[player] != "human"

    def is_hand_visible(self, player):
        """
        Final visibility query used by the HUD.

        It combines the game mode with the user's toggle state.
        """
        humans = self._human_count()

        if humans == 0:
            return True

        if humans >= 2:
            current_player = self.current_state().get("current_player", 0)
            return player == current_player

        if self.agent_types[player] == "human":
            return True

        return self.user_visible_hands[player]

    def is_hand_hidden(self, player):
        return not self.is_hand_visible(player)

    def _toggle_hand_visibility(self, player):
        """
        Handle J/K visibility toggles.

        When the current mode does not allow toggling, the UI shows a short
        explanation instead of silently ignoring the command.
        """
        if not self.can_toggle_hand_visibility(player):
            humans = self._human_count()

            if humans == 0:
                self._set_notification("AI vs AI: hands are always visible", duration_ms=1600)
            elif humans >= 2:
                self._set_notification(
                    "Human vs human: only the current player's hand is visible",
                    duration_ms=1800,
                )
            else:
                self._set_notification("The human player's hand is always visible", duration_ms=1600)

            return

        self.user_visible_hands[player] = not self.user_visible_hands[player]

        state = "visible" if self.user_visible_hands[player] else "hidden"
        self._set_notification(f"P{player} hand: {state}", duration_ms=1400)
