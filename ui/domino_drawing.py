"""
3D drawing for a single domino tile.

`layout_domino.py` calculates positions and angles; this module only applies
OpenGL transformations and delegates the tile primitive.
"""

from OpenGL.GL import *

from ui.layout_domino import inline_angle
from ui.primitives import draw_domino
from ui.visual_config import TILE_SCALE


def draw_tile(info, x_pos, y_pos, angle=None, values=None):
    if values is None:
        values = tuple(info["tile"])

    if angle is None:
        angle = inline_angle(info)

    left_value, right_value = values

    glPushMatrix()

    glTranslatef(x_pos, y_pos, 0.0)
    glScalef(TILE_SCALE, TILE_SCALE, 1.0)

    if angle != 0.0:
        glRotatef(angle, 0.0, 0.0, 1.0)

    draw_domino(left_value, right_value)

    glPopMatrix()
