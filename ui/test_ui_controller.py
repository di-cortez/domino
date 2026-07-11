"""
Sequential unit tests for the UI/controller layer.

The file is intentionally executable without pytest so it is easy to run in
class: `python ui/test_ui_controller.py`. The tests avoid opening an OpenGL
window and focus on deterministic controller behavior.
"""

import sys
from pathlib import Path

import pygame

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.heuristic_agent import StrategicAgent
from middleware.domino_engine import DominoEngine
from middleware.middleware import GameManager
from ui.game_controller import GameController
from ui.scene_renderer import visual_chain_from_state


def _new_controller(agent_types=None, interval_ms=1000):
    engine = DominoEngine(player_count=2)
    agents = [StrategicAgent(), StrategicAgent()]
    manager = GameManager(engine, agents)

    if agent_types is None:
        agent_types = ["heuristic", "heuristic"]

    controller = GameController(
        manager,
        engine,
        interval_ms=interval_ms,
        agent_types=agent_types,
    )

    return engine, controller


def _new_controller_with_human_turn():
    engine = DominoEngine(player_count=2)
    agent_types = ["heuristic", "heuristic"]
    agent_types[engine.current_player] = "human"

    agents = [StrategicAgent(), StrategicAgent()]
    manager = GameManager(engine, agents)

    controller = GameController(
        manager,
        engine,
        interval_ms=1000,
        agent_types=agent_types,
    )

    return engine, controller


def _force_human_turn(controller, engine, player=0):
    controller.agent_types = ["heuristic", "heuristic"]
    controller.agent_types[player] = "human"
    engine.current_player = player
    controller._configure_hand_visibility_for_mode()
    controller.history = []
    controller.history_info = []
    controller.index = 0
    controller._human_selection_key = None
    controller._capture_state()
    controller._sync_human_selection()


def _prepare_human_state(controller, engine, ends, hand, player=0, stock=None):
    controller.agent_types = ["heuristic", "heuristic"]
    controller.agent_types[player] = "human"
    engine.current_player = player
    engine.ends = list(ends)
    engine.hands[player] = list(hand)
    engine.stock = list(stock) if stock is not None else list(engine.stock)
    engine.required_opening_tile = None
    engine.drew_this_turn[player] = False

    controller._configure_hand_visibility_for_mode()
    controller.history = []
    controller.history_info = []
    controller.index = 0
    controller._human_selection_key = None
    controller._capture_state()
    controller._sync_human_selection()


def _valid_actions(controller, player):
    return controller._valid_action_set(player)


def _run(name, fn):
    fn()
    print(f"OK - {name}")


def test_ai_advances_automatically():
    engine, controller = _new_controller(interval_ms=1)
    turn = engine.turn

    controller.update(1000)

    assert engine.turn > turn


def test_human_does_not_advance_automatically():
    engine, controller = _new_controller_with_human_turn()
    turn = engine.turn

    controller.update(5000)

    assert engine.turn == turn
    assert controller.active_human_player()


def test_initial_human_selection_is_valid_when_possible():
    engine, controller = _new_controller_with_human_turn()
    player = engine.current_player
    action = controller._selected_human_action()

    assert controller.selected_tile_index < len(engine.hands[player])

    if action is not None:
        assert action in _valid_actions(controller, player)


def test_human_tile_navigation_wraps():
    engine, controller = _new_controller_with_human_turn()
    player = engine.current_player
    hand_size = len(engine.hands[player])

    controller.selected_tile_index = 0
    controller._navigate_human_tile(-1)

    assert controller.selected_tile_index == hand_size - 1

    controller._navigate_human_tile(1)

    assert controller.selected_tile_index == 0


def test_tab_toggles_end():
    engine, controller = _new_controller()
    _prepare_human_state(controller, engine, ends=[1, 2], hand=[(1, 2)])

    controller._toggle_human_end()

    assert controller.selected_end == "left"

    controller._toggle_human_end()

    assert controller.selected_end == "right"


def test_navigation_updates_to_valid_end():
    engine, controller = _new_controller()
    _prepare_human_state(
        controller,
        engine,
        ends=[1, 2],
        hand=[(1, 3), (4, 2), (1, 2), (4, 5)],
    )

    assert controller.selected_tile_index == 0
    assert controller.selected_end == "left"

    controller._navigate_human_tile(1)

    assert controller.selected_tile_index == 1
    assert controller.selected_end == "right"

    controller._navigate_human_tile(1)

    assert controller.selected_tile_index == 2
    assert controller.selected_end == "right"

    controller._toggle_human_end()

    assert controller.selected_end == "left"


def test_equal_ends_display_right_but_send_valid_engine_action():
    engine, controller = _new_controller()
    _prepare_human_state(
        controller,
        engine,
        ends=[3, 3],
        hand=[(3, 5)],
    )

    assert controller.selected_end == "right"
    assert controller._selected_human_action() == ((3, 5), 0)


def test_arrow_points_to_lower_tile_half():
    engine, controller = _new_controller()
    _prepare_human_state(
        controller,
        engine,
        ends=[2, 3],
        hand=[(1, 2)],
    )

    assert controller.selected_end == "left"
    assert controller.selected_tile_arrow_position() == "below"


def test_arrow_points_to_upper_tile_half():
    engine, controller = _new_controller()
    _prepare_human_state(
        controller,
        engine,
        ends=[2, 3],
        hand=[(2, 1)],
    )

    assert controller.selected_end == "left"
    assert controller.selected_tile_arrow_position() == "above"


def test_arrow_uses_selected_end_for_two_sided_tile():
    engine, controller = _new_controller()
    _prepare_human_state(
        controller,
        engine,
        ends=[2, 3],
        hand=[(2, 3)],
    )

    assert controller.selected_end == "right"
    assert controller.selected_tile_arrow_position() == "below"


def test_enter_executes_valid_human_move():
    engine, controller = _new_controller_with_human_turn()
    turn = engine.turn

    controller._play_human_tile()

    assert engine.turn > turn
    assert len(controller.history) == 2


def test_human_draw_rejects_when_a_tile_can_play():
    engine, controller = _new_controller()
    _prepare_human_state(
        controller,
        engine,
        ends=[2, 3],
        hand=[(2, 5)],
        stock=[(0, 0)],
    )

    history_size = len(controller.history)
    controller._human_draw()

    assert len(controller.history) == history_size
    assert controller.notification["message"] == "Draw is not allowed now"


def test_d_key_executes_human_draw_when_legal():
    engine, controller = _new_controller()
    _prepare_human_state(
        controller,
        engine,
        ends=[2, 3],
        hand=[(4, 5)],
        stock=[(0, 0)],
    )

    controller._process_human_key(pygame.K_d)

    assert engine.turn == 1
    assert engine.board_history[-1] == ("DRAW", None)
    assert len(engine.hands[0]) == 2


def test_human_pass_rejects_when_draw_is_available():
    engine, controller = _new_controller()
    _prepare_human_state(
        controller,
        engine,
        ends=[2, 3],
        hand=[(4, 5)],
        stock=[(0, 0)],
    )

    controller._human_pass()

    assert controller.notification["message"] == "Pass is not allowed now"


def test_draw_notification_follows_history_cursor():
    _engine, controller = _new_controller()

    controller._capture_state({"action": ("DRAW", None), "acting_player": 0})
    controller.index = 1
    controller._update_notification()

    assert controller.notification["message"] == "Player 0 (Heuristic) drew from the stock"

    controller.step_backward()

    assert controller.notification is None

    controller.step_forward()

    assert controller.notification["message"] == "Player 0 (Heuristic) drew from the stock"


def test_visual_chain_is_rebuilt_from_board_history():
    state = {
        "board_history": [
            [[6, 6], 0],
            [[4, 6], 0],
            [[5, 6], 1],
            ["DRAW", None],
            None,
            [[4, 4], 0],
        ],
    }

    chain = visual_chain_from_state(state)

    assert [item["tile"] for item in chain] == [
        [4, 4],
        [4, 6],
        [6, 6],
        [5, 6],
    ]


def test_menu_cycle_updates_manager_agent():
    _engine, controller = _new_controller()

    controller.menu_cursor = 0
    controller._activate_menu_item()

    assert controller.agent_types[0] == "random"
    assert controller.manager.agents[0].__class__.__name__ == "RandomUIAgent"


def test_speed_has_bounds():
    _engine, controller = _new_controller()

    assert controller._speed_text() == "1x"

    for _ in range(10):
        controller._change_speed(1)

    assert controller._speed_text() == "4x"
    assert controller._current_interval_ms() == 250.0

    for _ in range(10):
        controller._change_speed(-1)

    assert controller._speed_text() == "1/4x"
    assert controller._current_interval_ms() == 4000.0


def test_restart_requires_confirmation_and_expires():
    engine, controller = _new_controller(interval_ms=1)
    turn = engine.turn

    controller._restart_shortcut()

    assert controller._restart_confirmation_active()
    assert controller.paused
    assert engine.turn == turn

    controller.update(2100)

    assert not controller._restart_confirmation_active()
    assert not controller.paused
    assert engine.turn > turn


def test_second_r_confirms_restart():
    engine, controller = _new_controller()

    controller.step_forward()
    assert engine.turn > 0

    controller._restart_shortcut()
    controller._restart_shortcut()

    assert engine.turn == 0
    assert len(controller.history) == 1
    assert not controller._restart_confirmation_active()


def test_restart_after_game_over_is_immediate():
    engine, controller = _new_controller()
    controller.game_over = True
    controller.final_info = {"winner": 0}

    controller._restart_shortcut()

    assert engine.turn == 0
    assert not controller.game_over
    assert not controller._restart_confirmation_active()


def test_ai_vs_ai_keeps_hands_visible():
    _engine, controller = _new_controller()

    assert not controller.is_hand_hidden(0)
    assert not controller.is_hand_hidden(1)

    controller._toggle_hand_visibility(0)

    assert not controller.is_hand_hidden(0)
    assert controller.notification["message"] == "AI vs AI: hands are always visible"


def test_human_vs_ai_visibility():
    engine, controller = _new_controller()
    _force_human_turn(controller, engine, player=0)

    assert not controller.is_hand_hidden(0)
    assert controller.is_hand_hidden(1)

    controller._toggle_hand_visibility(1)

    assert not controller.is_hand_hidden(1)

    controller._toggle_hand_visibility(1)

    assert controller.is_hand_hidden(1)

    controller._toggle_hand_visibility(0)

    assert not controller.is_hand_hidden(0)
    assert controller.notification["message"] == "The human player's hand is always visible"


def test_human_vs_human_shows_only_current_hand():
    engine, controller = _new_controller(agent_types=["human", "human"])
    engine.current_player = 0
    controller.history = []
    controller.history_info = []
    controller.index = 0
    controller._capture_state()

    assert not controller.is_hand_hidden(0)
    assert controller.is_hand_hidden(1)

    engine.current_player = 1
    controller.history = []
    controller.history_info = []
    controller.index = 0
    controller._capture_state()

    assert controller.is_hand_hidden(0)
    assert not controller.is_hand_hidden(1)

    controller._toggle_hand_visibility(0)

    assert controller.is_hand_hidden(0)
    assert (
        controller.notification["message"]
        == "Human vs human: only the current player's hand is visible"
    )


def main():
    tests = [
        ("AI advances automatically", test_ai_advances_automatically),
        ("human does not advance automatically", test_human_does_not_advance_automatically),
        ("initial human selection", test_initial_human_selection_is_valid_when_possible),
        ("human navigation wraps", test_human_tile_navigation_wraps),
        ("Tab toggles end", test_tab_toggles_end),
        ("navigation updates legal end", test_navigation_updates_to_valid_end),
        ("equal ends use right visually", test_equal_ends_display_right_but_send_valid_engine_action),
        ("arrow points to lower half", test_arrow_points_to_lower_tile_half),
        ("arrow points to upper half", test_arrow_points_to_upper_tile_half),
        ("arrow uses selected end value", test_arrow_uses_selected_end_for_two_sided_tile),
        ("Enter executes valid human move", test_enter_executes_valid_human_move),
        ("human draw rejection", test_human_draw_rejects_when_a_tile_can_play),
        ("D key executes draw", test_d_key_executes_human_draw_when_legal),
        ("human pass rejection", test_human_pass_rejects_when_draw_is_available),
        ("draw notification history", test_draw_notification_follows_history_cursor),
        ("visual chain rebuild", test_visual_chain_is_rebuilt_from_board_history),
        ("menu updates manager agent", test_menu_cycle_updates_manager_agent),
        ("speed bounds", test_speed_has_bounds),
        ("restart asks and expires", test_restart_requires_confirmation_and_expires),
        ("second R confirms restart", test_second_r_confirms_restart),
        ("restart after game over", test_restart_after_game_over_is_immediate),
        ("AI vs AI visibility", test_ai_vs_ai_keeps_hands_visible),
        ("human vs AI visibility", test_human_vs_ai_visibility),
        ("human vs human visibility", test_human_vs_human_shows_only_current_hand),
    ]

    for name, fn in tests:
        _run(name, fn)

    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
