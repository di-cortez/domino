"""
Visual game controller.

This object sits between three layers:

- `DominoEngine`, which owns the game rules;
- `GameManager`, which asks automatic agents for moves;
- the Pygame/OpenGL UI, which needs keyboard input, pause/history control,
  notifications, and rendering state.

Interaction-specific logic is split into small helpers:

- `ui_agents.py`: menu labels and agent construction;
- `human_control.py`: selected hand tile, selected end, draw/pass/play commands;
- `hand_visibility.py`: HUD rules for showing or hiding hands.

`GameController` remains the public UI facade used by the HUD and tests.

Main keys:
    M              -> open/close settings menu
    R              -> quick restart, with confirmation while the game is live
    Space          -> pause/resume automatic advancement
    Right          -> step forward in automatic mode
    Left           -> step backward in history
    + / -          -> change speed between 1/4x, 1/2x, 1x, 2x, and 4x
    J / K          -> toggle hand visibility when the current mode allows it
    ESC            -> quit, or close the menu

When a human is playing, Left/Right select a hand tile, Up/Down switch the
target end when legal, Enter plays, D draws, and P passes.
"""

import contextlib
import copy
import io

import pygame

from ui.hand_visibility import HandVisibilityMixin
from ui.human_control import HumanControlMixin
from ui.ui_agents import (
    AGENT_TYPES as _AGENT_TYPES,
    agent_type_name,
    create_agent_by_type,
)


_SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0)
_RESTART_CONFIRM_MS = 2000


class GameController(HandVisibilityMixin, HumanControlMixin):
    """
    Control one game from the UI point of view.

    This class does not decide domino legality. The engine does that. The
    controller owns the operational state that only the UI needs:

    - visual snapshots for backward/forward history navigation;
    - pause state and automatic-advance speed;
    - routing AI turns through `GameManager`;
    - stopping at human turns until the keyboard chooses an action;
    - restarting the game while keeping selected agent types;
    - temporary notifications for the HUD.
    """

    def __init__(self, manager, engine, interval_ms=1000, agent_types=None):
        self.manager = manager
        self.engine = engine
        self.base_interval_ms = interval_ms
        self.speed_index = _SPEEDS.index(1.0)
        self.agent_types = list(agent_types) if agent_types else ["neural", "heuristic"]

        # The engine only moves forward. To support the Left key, the UI stores
        # full visual snapshots. `history_info[i]` is the engine info that
        # produced `history[i]`; the initial state uses None.
        self.history = []
        self.history_info = []
        self.index = 0

        self.paused = False
        self.game_over = False
        self.final_info = None
        self._elapsed_ms = 0.0

        # R asks for confirmation while the game is still live. During that
        # short window, the simulation pauses and restores the previous pause
        # state if the confirmation expires.
        self._restart_confirmation_ms = 0
        self._paused_before_restart_confirmation = False

        # Temporary HUD notification: {"message": str, "duration_ms": int}.
        self.notification = None

        # Minimal human-turn state. Manipulated by `HumanControlMixin`.
        self.selected_tile_index = 0
        self.selected_end = "left"
        self._human_selection_key = None

        # Base visibility state. The final query lives in HandVisibilityMixin.
        self.user_visible_hands = [True, True]

        self.menu_open = False
        self.menu_cursor = 0
        self._MENU_ITEM_COUNT = 3
        self._paused_before_menu = False

        self._configure_hand_visibility_for_mode()
        self._capture_state()
        self._sync_human_selection()

    def _capture_state(self, info=None):
        """
        Freeze the current engine state for rendering and history navigation.

        `engine._get_state()` returns the compact state used by agents. The HUD
        also needs full hands and stock, so those fields are added only to the
        visual snapshot.
        """
        state = copy.deepcopy(self.engine._get_state())

        full_state = self.engine.to_dict()
        state["hands"] = copy.deepcopy(full_state.get("hands", []))
        state["initial_hands"] = copy.deepcopy(full_state.get("initial_hands", []))
        state["drawn_tiles_by_player"] = copy.deepcopy(full_state.get("drawn_tiles_by_player", []))
        state["stock"] = copy.deepcopy(full_state.get("stock", []))

        self.history.append(state)
        self.history_info.append(info)

    @property
    def at_live_edge(self):
        """True when the UI cursor points at the newest snapshot."""
        return self.index >= len(self.history) - 1

    def current_state(self):
        """Snapshot that should be rendered in the current frame."""
        return self.history[self.index]

    def active_human_player(self):
        """
        Return whether the live turn belongs to a keyboard-controlled human.

        Historical snapshots are observation-only, even if the player at that
        old state is human.
        """
        if not self.at_live_edge or self.game_over:
            return False

        player = self.current_state().get("current_player", 0)
        return self.agent_types[player] == "human"

    def _set_notification(self, message, duration_ms=3000):
        self.notification = {
            "message": message,
            "duration_ms": duration_ms,
        }

    def _update_notification(self):
        """
        Recreate the notification associated with the current snapshot.

        Draw notifications should reappear when the user navigates back to that
        history frame, while manual notifications such as speed changes are
        controlled by `_set_notification`.
        """
        info = self.history_info[self.index]

        if info and info.get("action") == ("DRAW", None):
            player = info["acting_player"]
            name = agent_type_name(self.agent_types[player])
            self.notification = {
                "message": f"Player {player} ({name}) drew from the stock",
                "duration_ms": 3000,
            }
            return

        self.notification = None

    @property
    def speed(self):
        return _SPEEDS[self.speed_index]

    def _speed_text(self):
        labels = {
            0.25: "1/4x",
            0.5: "1/2x",
            1.0: "1x",
            2.0: "2x",
            4.0: "4x",
        }
        return labels[self.speed]

    def _current_interval_ms(self):
        """Real delay between automatic turns after applying speed."""
        return self.base_interval_ms / self.speed

    def _change_speed(self, delta):
        new_index = self.speed_index + delta
        new_index = max(0, min(len(_SPEEDS) - 1, new_index))

        if new_index == self.speed_index:
            return

        self.speed_index = new_index
        self._elapsed_ms = 0.0
        self._set_notification(f"Speed: {self._speed_text()}", duration_ms=1400)

    @staticmethod
    def _is_plus_key(key):
        return key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS)

    @staticmethod
    def _is_minus_key(key):
        return key in (pygame.K_MINUS, pygame.K_KP_MINUS)

    def _restart_confirmation_active(self):
        return self._restart_confirmation_ms > 0

    def _cancel_restart_confirmation(self, restore_pause=True):
        """
        Cancel the R-key confirmation window.

        Normal flow lets the window expire by timer, but keeping this method
        explicit makes the rule easier to change later.
        """
        if not self._restart_confirmation_active():
            return

        self._restart_confirmation_ms = 0

        if restore_pause:
            self.paused = self._paused_before_restart_confirmation

    def _restart_shortcut(self):
        """
        Implement the R key.

        - Finished games restart immediately.
        - If confirmation is already open, a second R confirms it.
        - Otherwise the game pauses for two seconds waiting for confirmation.
        """
        if self.game_over or self._restart_confirmation_active():
            self._restart_game()
            return

        self._paused_before_restart_confirmation = self.paused
        self._restart_confirmation_ms = _RESTART_CONFIRM_MS
        self.paused = True
        self._elapsed_ms = 0.0
        self._set_notification(
            "Game is still running. Press R again to restart",
            duration_ms=_RESTART_CONFIRM_MS,
        )

    def _update_restart_confirmation(self, dt_ms):
        if not self._restart_confirmation_active():
            return

        self._restart_confirmation_ms -= dt_ms

        if self._restart_confirmation_ms <= 0:
            self._restart_confirmation_ms = 0
            self.paused = self._paused_before_restart_confirmation

    def _register_game_over(self, done, info):
        """Record game-over info and print the winner once."""
        if not done:
            return

        self.game_over = True
        self.final_info = info

        if info is not None and "announced" not in info:
            print(f"Game finished! Winner: player {info.get('winner')}")
            info["announced"] = True

    def _play_turn_with_filtered_console(self):
        """
        Execute one AI turn while suppressing noisy console output.

        Neural agents print legal move diagnostics that are useful during
        demonstrations, so those lines are kept.
        """
        buffer = io.StringIO()

        try:
            with contextlib.redirect_stdout(buffer):
                return self.manager.play_turn()
        finally:
            for line in buffer.getvalue().splitlines():
                if "Possible" in line:
                    print(line)

    def _play_next_turn(self):
        """
        Advance the live game edge by one turn.

        Human turns do not call an agent; the controller only synchronizes hand
        selection and waits for keyboard input. AI turns are delegated to the
        manager.
        """
        if self.game_over:
            return

        if self.active_human_player():
            self._sync_human_selection()
            return

        try:
            done, info = self._play_turn_with_filtered_console()
            self._capture_state(info)
            self.index = len(self.history) - 1
            self._update_notification()
            self._sync_human_selection()
            self._register_game_over(done, info)
        except Exception as exc:
            print(f"Error during turn: {exc}")
            self.game_over = True

    def step_forward(self):
        if self.index < len(self.history) - 1:
            self.index += 1
            self._update_notification()
            self._sync_human_selection()
            return

        self._play_next_turn()

    def step_backward(self):
        if self.index > 0:
            self.index -= 1
            self._update_notification()
            self._sync_human_selection()

    def toggle_pause(self):
        self.paused = not self.paused
        self._elapsed_ms = 0.0

    def _open_menu(self):
        self.menu_open = True
        self.menu_cursor = 0
        self._paused_before_menu = self.paused
        self.paused = True

    def _close_menu(self):
        self.menu_open = False
        self.paused = self._paused_before_menu

    def _update_player_agent(self, player):
        if not hasattr(self.manager, "agents"):
            return

        self.manager.agents[player] = create_agent_by_type(self.agent_types[player])

    def _cycle_player_type(self, player):
        index = _AGENT_TYPES.index(self.agent_types[player])
        self.agent_types[player] = _AGENT_TYPES[(index + 1) % len(_AGENT_TYPES)]
        self._update_player_agent(player)
        self._configure_hand_visibility_for_mode()
        self._sync_human_selection()

    def _activate_menu_item(self):
        if self.menu_cursor == 0:
            self._cycle_player_type(0)
        elif self.menu_cursor == 1:
            self._cycle_player_type(1)
        elif self.menu_cursor == 2:
            self._restart_game()
            self._close_menu()

    def _restart_game(self):
        """
        Restart the game while preserving selected player modes.

        The engine is reset, agents are rebuilt, and history returns to a single
        initial snapshot.
        """
        from middleware.middleware import GameManager

        self.engine.reset()
        new_agents = [
            create_agent_by_type(agent_type)
            for agent_type in self.agent_types
        ]
        self.manager = GameManager(self.engine, new_agents)

        self.history = []
        self.history_info = []
        self.index = 0
        self.paused = False
        self.game_over = False
        self.final_info = None
        self.notification = None
        self._elapsed_ms = 0.0
        self._restart_confirmation_ms = 0
        self._paused_before_restart_confirmation = False
        self._human_selection_key = None

        self._configure_hand_visibility_for_mode()
        self._capture_state()
        self._sync_human_selection()

    def process_input(self):
        """Consume pygame events. Return False when the window should close."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type != pygame.KEYDOWN:
                continue

            if self.menu_open:
                if event.key in (pygame.K_ESCAPE, pygame.K_m):
                    self._close_menu()
                elif event.key == pygame.K_UP:
                    self.menu_cursor = (self.menu_cursor - 1) % self._MENU_ITEM_COUNT
                elif event.key == pygame.K_DOWN:
                    self.menu_cursor = (self.menu_cursor + 1) % self._MENU_ITEM_COUNT
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                    self._activate_menu_item()
                continue

            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_r:
                self._restart_shortcut()
                continue

            # While R is waiting for confirmation, ignore shortcuts that could
            # advance or change the game. The second R was handled above.
            if self._restart_confirmation_active():
                continue

            if event.key == pygame.K_m:
                self._open_menu()
            elif event.key == pygame.K_j:
                self._toggle_hand_visibility(0)
            elif event.key == pygame.K_k:
                self._toggle_hand_visibility(1)
            elif self._is_plus_key(event.key):
                self._change_speed(1)
            elif self._is_minus_key(event.key):
                self._change_speed(-1)
            elif self.active_human_player():
                self._process_human_key(event.key)
            else:
                self._process_automatic_key(event.key)

        return True

    def _process_human_key(self, key):
        """Keyboard shortcuts valid during a human turn."""
        self._sync_human_selection()

        if key == pygame.K_RIGHT:
            self._navigate_human_tile(1)
        elif key == pygame.K_LEFT:
            self._navigate_human_tile(-1)
        elif key in (pygame.K_UP, pygame.K_DOWN, pygame.K_TAB):
            self._toggle_human_end()
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._play_human_tile()
        elif key == pygame.K_d:
            self._human_draw()
        elif key == pygame.K_p:
            self._human_pass()

    def _process_automatic_key(self, key):
        """Keyboard shortcuts valid in automatic/observation mode."""
        if key == pygame.K_SPACE:
            self.toggle_pause()
        elif key == pygame.K_RIGHT:
            self.paused = True
            self.step_forward()
        elif key == pygame.K_LEFT:
            self.paused = True
            self.step_backward()

    def update(self, dt_ms):
        """
        Update timers and advance the game when allowed.

        Called once per frame by `visual_main.py`. It does not block the render
        loop; it only accumulates time and executes a turn when the configured
        interval has elapsed.
        """
        if self.notification:
            self.notification["duration_ms"] -= dt_ms
            if self.notification["duration_ms"] <= 0:
                self.notification = None

        self._update_restart_confirmation(dt_ms)

        if self.active_human_player():
            self._sync_human_selection()
            return

        if self.paused:
            return

        if self.at_live_edge and self.game_over:
            return

        self._elapsed_ms += dt_ms
        interval_ms = self._current_interval_ms()

        if self._elapsed_ms >= interval_ms:
            self._elapsed_ms -= interval_ms
            self.step_forward()
