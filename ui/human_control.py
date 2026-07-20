"""
Human-player interaction logic.

The domino engine already validates actions (`engine.valid_actions`) and
applies them (`engine.step`). This mixin only keeps UI interaction state:

- which hand tile is selected;
- which end the human intends to play on;
- how draw, pass, and play commands are routed;
- where the yellow arrow should be drawn over the selected tile.

Keeping this as a mixin keeps `GameController` smaller while preserving the
same public methods for the HUD and tests.
"""


END_TO_SIDE = {"left": 0, "right": 1}
SIDE_TO_END = {0: "left", 1: "right"}


class HumanControlMixin:
    """
    Mixin used by `GameController`.

    The final class must provide:
    - `engine`
    - `game_over`
    - `active_human_player()`
    - `_capture_state(info)`
    - `_update_notification()`
    - `_set_notification(message, duration_ms=...)`
    - `_register_game_over(done, info)`
    """

    def _normalize_action(self, action):
        """
        Normalize actions returned by the engine.

        Snapshots may expose tiles as lists while legal actions use tuples.
        Normalizing avoids false mismatches in UI comparisons.
        """
        if action is None:
            return None
        if action == ("DRAW", None):
            return ("DRAW", None)

        tile, side = action
        return tuple(tile), side

    def _valid_action_set(self, player):
        return {
            self._normalize_action(action)
            for action in self.engine.valid_actions(player)
        }

    def _tile_side_action(self, tile, side):
        return tuple(tile), side

    def _ends_are_equal(self):
        return len(self.engine.ends) == 2 and self.engine.ends[0] == self.engine.ends[1]

    def _valid_sides_for_tile(self, tile):
        """
        Return the end names (`left`/`right`) where a tile can legally play.

        When both board ends have the same value, left and right are logically
        equivalent. The UI displays `right` in that case for a stable selection,
        while still submitting an engine-valid action.
        """
        player = self.engine.current_player
        valid_actions_set = self._valid_action_set(player)

        if self._ends_are_equal() and self._tile_side_action(tile, 0) in valid_actions_set:
            return ["right"]

        sides = []

        for end, side in END_TO_SIDE.items():
            if self._tile_side_action(tile, side) in valid_actions_set:
                sides.append(end)

        return sides

    def _adjust_end_for_selected_tile(self, prefer_current=True):
        player = self.engine.current_player
        hand = self.engine.hands[player]

        if not hand:
            self.selected_end = "left"
            return

        self.selected_tile_index %= len(hand)
        tile = hand[self.selected_tile_index]
        sides = self._valid_sides_for_tile(tile)

        if not sides:
            return

        if prefer_current and self.selected_end in sides:
            return

        # Prefer right for equivalent ends and for stable arrow placement when
        # moving across tiles that fit on both ends.
        if "right" in sides:
            self.selected_end = "right"
        else:
            self.selected_end = sides[0]

    def _select_first_valid_move(self):
        """
        Pick the initial tile when a human turn starts.

        The first playable tile is selected when possible. If nothing fits, the
        selection still points at a hand tile so keyboard navigation remains
        predictable.
        """
        player = self.engine.current_player
        hand = self.engine.hands[player]

        if not hand:
            self.selected_tile_index = 0
            self.selected_end = "left"
            return

        valid_actions_set = self._valid_action_set(player)

        for index, tile in enumerate(hand):
            if any(self._tile_side_action(tile, side) in valid_actions_set for side in SIDE_TO_END):
                self.selected_tile_index = index
                self._adjust_end_for_selected_tile(prefer_current=False)
                return

        self.selected_tile_index = min(self.selected_tile_index, len(hand) - 1)
        self._adjust_end_for_selected_tile(prefer_current=False)

    def _sync_human_selection(self):
        """
        Keep the selected tile coherent with the live engine state.

        Player, turn, or hand changes create a new selection context. Otherwise
        only the selected index is wrapped back into the current hand length.
        """
        if not self.active_human_player():
            self._human_selection_key = None
            return

        player = self.engine.current_player
        hand = tuple(tuple(tile) for tile in self.engine.hands[player])
        key = (player, self.engine.turn, hand)

        if key != self._human_selection_key:
            self._select_first_valid_move()
            self._human_selection_key = key
            return

        if hand:
            self.selected_tile_index %= len(hand)
        else:
            self.selected_tile_index = 0

    def _navigate_human_tile(self, delta):
        """Move the hand selection circularly."""
        self._sync_human_selection()

        player = self.engine.current_player
        hand = self.engine.hands[player]

        if not hand:
            self.selected_tile_index = 0
            return

        self.selected_tile_index = (
            self.selected_tile_index + delta
        ) % len(hand)
        self._adjust_end_for_selected_tile(prefer_current=False)

    def _toggle_human_end(self):
        """
        Switch the intended play end for the selected tile.

        The end only changes when the selected tile legally fits on both sides.
        Otherwise the UI keeps the legal end and tells the user why.
        """
        player = self.engine.current_player
        hand = self.engine.hands[player]

        if not hand:
            return

        self.selected_tile_index %= len(hand)
        tile = hand[self.selected_tile_index]
        sides = self._valid_sides_for_tile(tile)

        if len(sides) < 2:
            if sides:
                self._set_notification(f"This tile only plays on the {sides[0]} end", duration_ms=1400)
            else:
                self._set_notification("This tile does not fit", duration_ms=1400)
            return

        if self.selected_end == "left":
            self.selected_end = "right"
        else:
            self.selected_end = "left"

    def _selected_human_action(self):
        """Build the action represented by the selected tile/end."""
        self._sync_human_selection()

        player = self.engine.current_player
        hand = self.engine.hands[player]

        if not hand:
            return None

        self.selected_tile_index %= len(hand)
        tile = hand[self.selected_tile_index]
        side = END_TO_SIDE[self.selected_end]

        action = self._tile_side_action(tile, side)

        if action in self._valid_action_set(player):
            return action

        if self._ends_are_equal():
            equivalent_action = self._tile_side_action(tile, 0)

            if equivalent_action in self._valid_action_set(player):
                return equivalent_action

        return action

    def selected_tile_arrow_position(self):
        """
        Return whether the HUD arrow should appear above or below the tile.

        The arrow points to the tile half that connects to the board, not merely
        to the selected left/right end. For example, with board ends 2 and 3,
        tile `(1, 2)` shows the arrow below because value 2 is connected.
        """
        if not self.active_human_player():
            return None

        player = self.engine.current_player
        hand = self.engine.hands[player]

        if not hand:
            return None

        self.selected_tile_index %= len(hand)
        tile = hand[self.selected_tile_index]

        if not self.engine.ends:
            return "below"

        if self._ends_are_equal():
            connected_value = self.engine.ends[0]
        else:
            side = END_TO_SIDE[self.selected_end]
            connected_value = self.engine.ends[side]

        if tile[1] == connected_value:
            return "below"

        if tile[0] == connected_value:
            return "above"

        return "below"

    def _execute_human_action(self, action):
        """
        Execute a human draw, pass, or tile play through the engine.

        After `step`, the controller captures a visual snapshot and updates game
        over state and notifications.
        """
        if self.game_over:
            return

        player = self.engine.current_player

        try:
            _state, done, info = self.engine.step(action, return_state=False)
        except ValueError as exc:
            print(exc)
            self._set_notification("Invalid move")
            return

        info["action"] = action
        info["acting_player"] = player

        self._capture_state(info)
        self.index = len(self.history) - 1
        self._update_notification()
        self._sync_human_selection()
        self._register_game_over(done, info)

        # If the human move gives the turn to AI, resume automatic advancement.
        if not self.game_over and not self.active_human_player():
            self.paused = False
            self._elapsed_ms = 0.0

    def _play_human_tile(self):
        action = self._selected_human_action()

        if action is None:
            self._set_notification("No selected tile")
            return

        player = self.engine.current_player
        if action not in self._valid_action_set(player):
            self._set_notification("This tile does not fit on that end")
            return

        self._execute_human_action(action)

    def _human_draw(self):
        player = self.engine.current_player

        if ("DRAW", None) not in self._valid_action_set(player):
            self._set_notification("Draw is not allowed now")
            return

        self._execute_human_action(("DRAW", None))

    def _human_pass(self):
        player = self.engine.current_player

        if None not in self._valid_action_set(player):
            self._set_notification("Pass is not allowed now")
            return

        self._execute_human_action(None)
