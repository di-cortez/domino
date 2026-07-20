"""
2D HUD drawn above the OpenGL board.

This module only reads controller state. It never executes moves or mutates the
engine; all game changes go through `GameController`.
"""

import pygame

from middleware.opponent_model import ExactOpponentModel
from ui.primitives import (
    begin_2d,
    draw_domino_2d,
    draw_text,
    end_2d,
    rectangle,
    triangle,
)
from ui.ui_agents import agent_type_name


class HudRenderer:
    """Render turns, hands, stock, notifications, game over, and menu layers."""

    _NOTIFICATION_DURATION_MS = 3000
    _NOTIFICATION_FADE_MS = 1000

    _BAR_H = 38
    _HANDS_H = 74

    _DOMINO_W = 22
    _DOMINO_H = 42
    _DOMINO_GAP = 4

    _MENU_PADDING = 44
    _MENU_ITEM_H = 50
    _MENU_HEADER_H = 68
    _MENU_FOOTER_H = 30

    def __init__(self):
        self._fonts_ready = False
        self._title_font = None
        self._normal_font = None
        self._hint_font = None
        self._probability_cache = {}
        self._opponent_models = {
            0: ExactOpponentModel(record_traces=False),
            1: ExactOpponentModel(record_traces=False),
        }

    def _init_fonts(self):
        if self._fonts_ready:
            return

        pygame.font.init()

        self._title_font = pygame.font.SysFont("monospace", 26, bold=True)
        self._normal_font = pygame.font.SysFont("monospace", 20)
        self._hint_font = pygame.font.SysFont("monospace", 14)

        self._fonts_ready = True

    def render(self, state, controller, display):
        self._init_fonts()

        sw, sh = display

        begin_2d(sw, sh)

        current_player = state.get("current_player", 0)
        turn = state.get("turn", 0)

        names = [
            agent_type_name(agent_type)
            for agent_type in controller.agent_types
        ]

        self._render_top_bar(current_player, turn, names, display)
        self._render_hands_bar(state, controller, display)
        self._render_suit_probabilities(state, display)
        self._render_bottom_bar(display, controller.active_human_player())

        if controller.notification:
            self._render_notification(controller.notification, display)

        if controller.game_over and controller.final_info:
            self._render_game_over(controller, names, display)

        if controller.menu_open:
            self._render_menu(controller, display)

        end_2d()

    def _render_top_bar(self, current_player, turn, names, display):
        sw, _sh = display

        rectangle(0, 0, sw, self._BAR_H, (0, 0, 0), 0.78)

        player0_color = (80, 200, 255) if current_player == 0 else (110, 110, 110)
        player0_label = f"P0 ({names[0]})"

        if current_player == 0:
            player0_label += "  [TURN]"
            label_width, _ = self._normal_font.size(player0_label)
            rectangle(5, 4, label_width + 10, 30, (0.1, 0.4, 0.7), 0.8)

        draw_text(player0_label, self._normal_font, player0_color, 12, 9)

        turn_text = f"Turn {turn}"
        turn_width, _ = self._normal_font.size(turn_text)

        draw_text(
            turn_text,
            self._normal_font,
            (220, 220, 170),
            sw // 2 - turn_width // 2,
            9,
        )

        player1_color = (255, 185, 55) if current_player == 1 else (110, 110, 110)
        player1_label = f"P1 ({names[1]})"

        if current_player == 1:
            player1_label = "[TURN]  " + player1_label

        label_width, _ = self._normal_font.size(player1_label)

        if current_player == 1:
            rectangle(sw - label_width - 15, 4, label_width + 10, 30, (0.5, 0.28, 0.05), 0.8)

        draw_text(player1_label, self._normal_font, player1_color, sw - label_width - 10, 9)

    def _render_bottom_bar(self, display, human_active):
        sw, sh = display

        if human_active:
            hint = (
                "Left/Right: tile | Up/Down: end | Enter: play | "
                "D: draw | P: pass | M: menu | ESC: quit"
            )
        else:
            hint = "M: Menu | R: Restart | Space: Pause | Arrows: Step | +/-: speed | ESC: Quit"

        hint_width, hint_height = self._hint_font.size(hint)

        rectangle(0, sh - hint_height - 6, sw, hint_height + 6, (0, 0, 0), 0.6)

        draw_text(
            hint,
            self._hint_font,
            (150, 150, 150),
            sw // 2 - hint_width // 2,
            sh - hint_height - 3,
        )

    def _render_suit_probabilities(self, state, display):
        """Draw opponent suit-presence probabilities from both perspectives."""
        sw, sh = display
        hands = state.get("hands") or []
        if len(hands) < 2:
            return

        box_w = 38
        box_h = 20
        gap = 4
        label_h = 16
        panel_w = 7 * box_w + 6 * gap + 12
        panel_h = label_h + box_h + 12
        bottom_bar_clearance = 28
        y = sh - panel_h - bottom_bar_clearance

        self._render_probability_panel(
            player=0,
            probabilities=self._cached_probabilities_for_player(state, 0),
            x=8,
            y=y,
            panel_w=panel_w,
            panel_h=panel_h,
            box_w=box_w,
            box_h=box_h,
            gap=gap,
            align="left",
        )
        self._render_probability_panel(
            player=1,
            probabilities=self._cached_probabilities_for_player(state, 1),
            x=sw - panel_w - 8,
            y=y,
            panel_w=panel_w,
            panel_h=panel_h,
            box_w=box_w,
            box_h=box_h,
            gap=gap,
            align="right",
        )

    def _render_probability_panel(
        self,
        player,
        probabilities,
        x,
        y,
        panel_w,
        panel_h,
        box_w,
        box_h,
        gap,
        align,
    ):
        """Draw one player's opponent-suit probability panel."""
        color = (80, 200, 255) if player == 0 else (255, 185, 55)
        label = f"P{player} opp"
        label_w, _ = self._hint_font.size(label)

        rectangle(x, y, panel_w, panel_h, (0, 0, 0), 0.58)
        rectangle(x, y, panel_w, 1, (0.25, 0.45, 0.50), 0.75)

        label_x = x + 6 if align == "left" else x + panel_w - label_w - 6
        draw_text(label, self._hint_font, color, label_x, y + 4)

        row_x = x + 6
        row_y = y + 22
        for suit, probability in enumerate(probabilities):
            box_x = row_x + suit * (box_w + gap)
            rectangle(box_x, row_y, box_w, box_h, (0.05, 0.08, 0.10), 0.85)

            text = f"{float(probability):.2f}"
            text_w, text_h = self._hint_font.size(text)
            draw_text(
                text,
                self._hint_font,
                (220, 230, 220),
                box_x + box_w // 2 - text_w // 2,
                row_y + box_h // 2 - text_h // 2,
            )

    def _probability_state_for_player(self, state, player):
        """Build a private observer view for probability rendering."""
        hands = state.get("hands") or []
        initial_hands = state.get("initial_hands") or []
        drawn_tiles = state.get("drawn_tiles_by_player") or []

        return {
            "game_id": state.get("game_id"),
            "history_current_player": state.get("current_player", 0),
            "observer_player": player,
            "current_player": player,
            "ends": state.get("ends", []),
            "current_player_hand": hands[player],
            "current_player_initial_hand": (
                initial_hands[player] if player < len(initial_hands) else None
            ),
            "current_player_drawn_tiles": (
                drawn_tiles[player] if player < len(drawn_tiles) else []
            ),
            "turn": state.get("turn", 0),
            "hand_sizes": state.get("hand_sizes", [len(hand) for hand in hands]),
            "board_history": state.get("board_history", []),
            "stock_size": state.get("stock_size", len(state.get("stock", []))),
            "game_over": state.get("game_over", False),
        }

    def _cached_probabilities_for_player(self, state, player):
        """Return HUD probabilities without recomputing unchanged snapshots."""
        key = (
            state.get("game_id"),
            state.get("turn"),
            len(state.get("board_history", [])),
            player,
            tuple(tuple(tile) for tile in (state.get("hands") or [[], []])[player]),
        )
        if key in self._probability_cache:
            return self._probability_cache[key]

        try:
            perspective_state = self._probability_state_for_player(state, player)
            model = self._opponent_models[player]
            probabilities = model.update(perspective_state)
        except (KeyError, ValueError, IndexError, TypeError):
            self._opponent_models[player].reset()
            probabilities = [0.0] * 7

        if len(self._probability_cache) > 64:
            self._probability_cache.clear()
        self._probability_cache[key] = probabilities
        return probabilities

    def _draw_selection_arrow(self, x, y, w, h, position):
        center_x = x + w / 2.0

        if position == "above":
            tip_y = y - 4
            base_y = y - 14

            triangle(
                center_x,
                tip_y,
                center_x - 7,
                base_y,
                center_x + 7,
                base_y,
                (1.0, 0.82, 0.08),
                1.0,
            )
            return

        tip_y = y + h + 4
        base_y = tip_y + 10

        triangle(
            center_x,
            tip_y,
            center_x - 7,
            base_y,
            center_x + 7,
            base_y,
            (1.0, 0.82, 0.08),
            1.0,
        )

    def _render_hand(
        self,
        tiles,
        x,
        y,
        max_w,
        max_h,
        align="left",
        selected_index=None,
        arrow_position=None,
        hidden=False,
    ):
        if not tiles:
            return

        # Hands are rendered in miniature and clipped to the available width.
        # If more tiles exist than fit, the HUD shows the first tiles plus "+N".
        count = len(tiles)

        tile_width = self._DOMINO_W
        tile_height = self._DOMINO_H
        gap = self._DOMINO_GAP

        if tile_height > max_h:
            scale = max(0.72, max_h / tile_height)
            tile_width = int(self._DOMINO_W * scale)
            tile_height = int(self._DOMINO_H * scale)
            gap = max(3, int(self._DOMINO_GAP * scale))

        columns = max(1, int((max_w + gap) // (tile_width + gap)))
        capacity = min(count, columns)

        visible_tiles = tiles[:capacity]
        hidden_count = count - len(visible_tiles)

        total_width = len(visible_tiles) * tile_width + max(0, len(visible_tiles) - 1) * gap

        if align == "left":
            start_x = x
        else:
            start_x = x + max_w - total_width

        for index, tile in enumerate(visible_tiles):
            tile_x = start_x + index * (tile_width + gap)
            tile_y = y

            if hidden:
                draw_domino_2d(tile_x, tile_y, tile_width, tile_height, back=True)
            else:
                draw_domino_2d(tile_x, tile_y, tile_width, tile_height, tile=tuple(tile))

            # The arrow only appears on visible hands; hidden hands must not
            # reveal which tile a human player selected.
            if not hidden and selected_index == index:
                self._draw_selection_arrow(tile_x, tile_y, tile_width, tile_height, arrow_position)

        if hidden_count > 0:
            text = f"+{hidden_count}"
            text_width, text_height = self._hint_font.size(text)

            text_x = start_x + total_width - text_width
            text_y = y + max_h - text_height

            draw_text(text, self._hint_font, (230, 230, 230), text_x, text_y)

    def _render_hands_bar(self, state, controller, display):
        sw, _sh = display

        y = self._BAR_H

        rectangle(0, y, sw, self._HANDS_H, (0.02, 0.08, 0.05), 0.82)
        rectangle(0, y + self._HANDS_H - 1, sw, 1, (0.05, 0.20, 0.12), 0.9)

        hands = state.get("hands") or []

        hand0 = hands[0] if len(hands) > 0 else []
        hand1 = hands[1] if len(hands) > 1 else []

        margin = 12
        center_width = 184

        hand_area_width = max(210, (sw - center_width - margin * 4) // 2)

        hand_y = y + 13
        hand_h = self._HANDS_H - 18

        current_player = state.get("current_player", 0)
        human_active = controller.active_human_player()

        selected0 = None
        selected1 = None

        if human_active and current_player == 0:
            selected0 = controller.selected_tile_index
        elif human_active and current_player == 1:
            selected1 = controller.selected_tile_index

        # The controller decides whether the arrow belongs above or below the
        # tile. The HUD only draws that choice.
        arrow_position = controller.selected_tile_arrow_position()
        hidden0 = controller.is_hand_hidden(0)
        hidden1 = controller.is_hand_hidden(1)

        self._render_hand(
            hand0,
            margin,
            hand_y,
            hand_area_width,
            hand_h,
            align="left",
            selected_index=selected0,
            arrow_position=arrow_position,
            hidden=hidden0,
        )
        self._render_hand(
            hand1,
            sw - margin - hand_area_width,
            hand_y,
            hand_area_width,
            hand_h,
            align="right",
            selected_index=selected1,
            arrow_position=arrow_position,
            hidden=hidden1,
        )

        self._render_stock(state, display)

    def _render_stock(self, state, display):
        sw, _sh = display

        y = self._BAR_H

        stock_count = state.get("stock_size", len(state.get("stock", [])))

        label = "Stock"
        count_text = f"x {stock_count}"

        tile_width = 40
        tile_height = 22
        gap = 10

        label_width, label_height = self._hint_font.size(label)
        count_width, count_height = self._normal_font.size(count_text)

        total_width = label_width + gap + tile_width + gap + count_width

        start_x = sw // 2 - total_width // 2

        tile_x = start_x + label_width + gap
        tile_y = y + self._HANDS_H // 2 - tile_height // 2 + 2

        text_y = tile_y + tile_height // 2 - label_height // 2
        count_y = tile_y + tile_height // 2 - count_height // 2

        draw_text(
            label,
            self._hint_font,
            (190, 215, 190),
            start_x,
            text_y,
        )

        draw_domino_2d(tile_x, tile_y, tile_width, tile_height, back=True)

        draw_text(
            count_text,
            self._normal_font,
            (220, 230, 210),
            tile_x + tile_width + gap,
            count_y,
        )

    def _render_notification(self, notification, display):
        sw, _sh = display

        message = notification["message"]
        remaining_ms = notification["duration_ms"]

        alpha = min(1.0, remaining_ms / self._NOTIFICATION_FADE_MS)

        text_width, text_height = self._normal_font.size(message)

        pad_x = 16
        pad_y = 7

        box_x = sw // 2 - text_width // 2 - pad_x
        box_y = self._BAR_H + self._HANDS_H + 8

        box_w = text_width + pad_x * 2
        box_h = text_height + pad_y * 2

        rectangle(box_x, box_y, box_w, box_h, (0.04, 0.04, 0.04), 0.85 * alpha)
        rectangle(box_x, box_y, box_w, 2, (0.9, 0.55, 0.1), alpha)

        draw_text(
            message,
            self._normal_font,
            (255, 200, 80),
            box_x + pad_x,
            box_y + pad_y,
            alpha,
        )

    def _render_game_over(self, controller, names, display):
        sw, sh = display

        winner = controller.final_info.get("winner")

        if winner == -1:
            message = "DRAW!"
        elif winner is not None:
            message = f"Game over! Winner: P{winner} ({names[winner]})"
        else:
            message = "Game over"

        message_width, message_height = self._title_font.size(message)

        box_x = sw // 2 - message_width // 2 - 14
        box_y = sh // 2 - message_height // 2 - 10

        rectangle(box_x, box_y, message_width + 28, message_height + 20, (0, 0, 0), 0.88)
        rectangle(box_x, box_y, message_width + 28, 3, (0.85, 0.6, 0.1), 1.0)

        draw_text(
            message,
            self._title_font,
            (255, 220, 50),
            sw // 2 - message_width // 2,
            sh // 2 - message_height // 2,
        )

        subtitle = "Press M > Restart for a new game"
        subtitle_width, _ = self._hint_font.size(subtitle)

        draw_text(
            subtitle,
            self._hint_font,
            (170, 170, 170),
            sw // 2 - subtitle_width // 2,
            sh // 2 + message_height // 2 + 8,
        )

    def _render_menu(self, controller, display):
        sw, sh = display

        def type_label(agent_type):
            return f"[ {agent_type_name(agent_type)} ]"

        items = [
            f"Player 0:  {type_label(controller.agent_types[0])}  (Enter: change)",
            f"Player 1:  {type_label(controller.agent_types[1])}  (Enter: change)",
            "Restart Game",
        ]

        footer = "Arrows: navigate  |  Enter/Space: select  |  M / ESC: close"

        item_widths = [
            self._normal_font.size("> " + item)[0]
            for item in items
        ]

        title = "SETTINGS"
        title_width = self._title_font.size(title)[0]
        footer_width = self._hint_font.size(footer)[0]

        menu_width = max(*item_widths, title_width, footer_width) + self._MENU_PADDING
        menu_height = self._MENU_HEADER_H + len(items) * self._MENU_ITEM_H + self._MENU_FOOTER_H

        menu_x = (sw - menu_width) // 2
        menu_y = (sh - menu_height) // 2

        rectangle(0, 0, sw, sh, (0, 0, 0), 0.55)

        rectangle(menu_x, menu_y, menu_width, menu_height, (0.07, 0.07, 0.12), 0.97)
        rectangle(menu_x, menu_y, menu_width, 3, (0.25, 0.5, 1.0), 1.0)
        rectangle(menu_x, menu_y + menu_height - 3, menu_width, 3, (0.25, 0.5, 1.0), 1.0)

        draw_text(
            title,
            self._title_font,
            (180, 210, 255),
            menu_x + 16,
            menu_y + 14,
        )

        for index, item in enumerate(items):
            item_y = menu_y + self._MENU_HEADER_H + index * self._MENU_ITEM_H

            selected = index == controller.menu_cursor

            if selected:
                rectangle(menu_x + 8, item_y - 5, menu_width - 16, 34, (0.15, 0.35, 0.75), 0.85)

            color = (255, 255, 110) if selected else (200, 200, 210)
            prefix = "> " if selected else "  "

            draw_text(
                prefix + item,
                self._normal_font,
                color,
                menu_x + 16,
                item_y,
            )

        footer_draw_width, _ = self._hint_font.size(footer)

        draw_text(
            footer,
            self._hint_font,
            (120, 120, 145),
            menu_x + menu_width // 2 - footer_draw_width // 2,
            menu_y + menu_height - 22,
        )
