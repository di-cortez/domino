"""
3D rendering for the board.

The renderer receives a visual snapshot from the controller, chooses a stable
pivot for the domino chain, and draws both branches outward from that pivot.
"""

from OpenGL.GL import *

from ui.domino_drawing import draw_tile
from ui.layout_domino import (
    calculate_branch_slots,
    inline_angle,
    split_chain_at_pivot,
)
from ui.primitives import TABLE_COLOR, rectangle
from ui.state_renderer import StateRenderer
from ui.visual_config import INITIAL_X, INITIAL_Y

_renderer_state = StateRenderer()


def draw_table():
    rectangle(-20.0, -20.0, 40.0, 40.0, TABLE_COLOR)


def _normalize_action(action):
    """Return a tuple-based action from JSON-style snapshot data."""
    if action is None:
        return None
    if action == ["DRAW", None] or action == ("DRAW", None):
        return ("DRAW", None)

    tile, side = action
    return (tuple(tile), side)


def visual_chain_from_state(state):
    """Build a left-to-right visual chain from the logical board history.

    The engine no longer stores UI-only `visual_chain` data. The renderer
    reconstructs it from the public action history so training snapshots stay
    compact while the board still has the order needed for layout.
    """
    history = state.get("board_history") or state.get("logical_board") or []
    visual_chain = []

    for index, raw_action in enumerate(history):
        action = _normalize_action(raw_action)

        if action is None or action == ("DRAW", None):
            continue

        tile, side = action
        info = {
            "id": index,
            "tile": list(tile),
        }

        if not visual_chain:
            visual_chain.append(info)
        elif side == 0:
            visual_chain.insert(0, info)
        else:
            visual_chain.append(info)

    return visual_chain


def render_scene(state):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    glTranslatef(0.0, 0.0, -15.0)

    draw_table()

    visual_chain = state.get("visual_chain") or visual_chain_from_state(state)

    if not visual_chain:
        return

    # The pivot keeps the whole chain from sliding when a tile is added at one
    # of the ends.
    pivot_index = _renderer_state.get_pivot_index(visual_chain)

    if pivot_index is None:
        return

    left_side, pivot, right_side = split_chain_at_pivot(
        visual_chain,
        pivot_index,
    )

    draw_tile(
        pivot,
        INITIAL_X,
        INITIAL_Y,
        angle=inline_angle(pivot),
        values=tuple(pivot["tile"]),
    )

    left_slots = calculate_branch_slots(
        left_side,
        pivot,
        direction=-1,
        initial_x=INITIAL_X,
        initial_y=INITIAL_Y,
    )

    right_slots = calculate_branch_slots(
        right_side,
        pivot,
        direction=1,
        initial_x=INITIAL_X,
        initial_y=INITIAL_Y,
    )

    for info, slot in left_slots:
        draw_tile(info, slot["x_pos"], slot["y_pos"], slot["angle"], slot["values"])

    for info, slot in right_slots:
        draw_tile(info, slot["x_pos"], slot["y_pos"], slot["angle"], slot["values"])
