"""
Drawing primitives shared by the 3D board and the 2D HUD.

The functions here stay deliberately small: rectangles, lines, circles, text
textures, and basic domino drawing. Higher-level modules handle layout, game
rules, and interaction.
"""

import math

import pygame
from OpenGL.GL import *


TABLE_COLOR = (0.1, 0.5, 0.2)
TILE_COLOR = (0.95, 0.95, 0.95)
LINE_COLOR = (0.1, 0.1, 0.1)


def rectangle(x, y, w, h, color, alpha=1.0):
    r, g, b = color
    glColor4f(r, g, b, alpha)

    glBegin(GL_QUADS)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()


def rectangle_outline(x, y, w, h, color, alpha=1.0, width=1.5):
    r, g, b = color
    glColor4f(r, g, b, alpha)
    glLineWidth(width)

    glBegin(GL_LINE_LOOP)
    glVertex2f(x, y)
    glVertex2f(x + w, y)
    glVertex2f(x + w, y + h)
    glVertex2f(x, y + h)
    glEnd()


def line(x1, y1, x2, y2, color, alpha=1.0, width=1.2):
    r, g, b = color
    glColor4f(r, g, b, alpha)
    glLineWidth(width)

    glBegin(GL_LINES)
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glEnd()


def triangle(x1, y1, x2, y2, x3, y3, color, alpha=1.0):
    r, g, b = color
    glColor4f(r, g, b, alpha)

    glBegin(GL_TRIANGLES)
    glVertex2f(x1, y1)
    glVertex2f(x2, y2)
    glVertex2f(x3, y3)
    glEnd()


def circle(cx, cy, radius, color, alpha=1.0, segments=18):
    r, g, b = color
    glColor4f(r, g, b, alpha)

    glBegin(GL_POLYGON)

    for index in range(segments):
        theta = 2.0 * math.pi * index / segments
        x = cx + radius * math.cos(theta)
        y = cy + radius * math.sin(theta)
        glVertex2f(x, y)

    glEnd()


def _upload_texture(surface):
    # The HUD is small, so a temporary texture per text label is simple enough.
    data = pygame.image.tostring(surface, "RGBA", True)
    width, height = surface.get_size()

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        width,
        height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        data,
    )

    return texture_id, width, height


def draw_text(message, font, color, x, y, alpha=1.0):
    """
    Draw text in screen coordinates.

    `x` and `y` are the top-left corner. Returns `(width, height)`.
    """
    surface = font.render(message, True, color).convert_alpha()
    texture_id, width, height = _upload_texture(surface)

    glEnable(GL_TEXTURE_2D)
    glColor4f(1.0, 1.0, 1.0, alpha)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glBegin(GL_QUADS)
    glTexCoord2f(0, 1)
    glVertex2f(x, y)

    glTexCoord2f(1, 1)
    glVertex2f(x + width, y)

    glTexCoord2f(1, 0)
    glVertex2f(x + width, y + height)

    glTexCoord2f(0, 0)
    glVertex2f(x, y + height)
    glEnd()

    glDisable(GL_TEXTURE_2D)
    glDeleteTextures(1, [texture_id])

    return width, height


def begin_2d(sw, sh):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()

    glOrtho(0, sw, sh, 0, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


def end_2d():
    glDisable(GL_BLEND)

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()

    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()


def _pip_positions(value, offset_x, offset_y, y_down=False):
    """
    Return relative pip positions for one domino half.

    `y_down=True` uses screen coordinates. `False` uses the normal mathematical
    OpenGL orientation.
    """
    top = -offset_y if y_down else offset_y
    bottom = offset_y if y_down else -offset_y

    positions = {
        0: [],
        1: [(0, 0)],
        2: [(-offset_x, top), (offset_x, bottom)],
        3: [(-offset_x, top), (0, 0), (offset_x, bottom)],
        4: [
            (-offset_x, top),
            (offset_x, top),
            (-offset_x, bottom),
            (offset_x, bottom),
        ],
        5: [
            (-offset_x, top),
            (offset_x, top),
            (0, 0),
            (-offset_x, bottom),
            (offset_x, bottom),
        ],
        6: [
            (-offset_x, top),
            (offset_x, top),
            (-offset_x, 0),
            (offset_x, 0),
            (-offset_x, bottom),
            (offset_x, bottom),
        ],
    }

    if value not in positions:
        raise ValueError(f"Invalid domino value: {value}")

    return positions[value]


def draw_domino_half(value, center_x, center_y):
    pip_radius = 0.12
    offset = 0.25

    for dx, dy in _pip_positions(value, offset, offset, y_down=False):
        circle(
            center_x + dx,
            center_y + dy,
            pip_radius,
            LINE_COLOR,
        )


def draw_domino(left_value, right_value):
    """
    Draw a domino tile centered at the local origin.

    Local size before scaling:
        width = 2.0
        height = 1.0
    """
    rectangle(-1.0, -0.5, 2.0, 1.0, TILE_COLOR)
    rectangle_outline(-1.0, -0.5, 2.0, 1.0, LINE_COLOR, width=2.0)

    line(0.0, -0.5, 0.0, 0.5, LINE_COLOR, width=2.0)

    draw_domino_half(left_value, -0.5, 0.0)
    draw_domino_half(right_value, 0.5, 0.0)


def draw_domino_back_2d(x, y, w, h, alpha=1.0):
    rectangle(x, y, w, h, (0.12, 0.16, 0.22), alpha)
    rectangle_outline(x, y, w, h, (0.02, 0.02, 0.02), alpha, width=1.5)

    rectangle(x + 4, y + 4, w - 8, h - 8, (0.20, 0.34, 0.48), alpha)

    line(
        x + 7,
        y + h - 7,
        x + w - 7,
        y + 7,
        (0.65, 0.85, 1.0),
        alpha,
        width=1.4,
    )


def draw_domino_2d(x, y, w, h, tile=None, back=False, alpha=1.0):
    """
    Draw a domino in screen coordinates.

    `x` and `y` are the top-left corner. `w` and `h` are width and height. When
    `back=True`, a tile back is drawn instead of pips.
    """
    if back:
        draw_domino_back_2d(x, y, w, h, alpha)
        return

    if not tile:
        return

    left_value, right_value = tile

    vertical = h > w

    half_w = w / 2.0 if vertical else w / 4.0
    half_h = h / 4.0 if vertical else h / 2.0
    radius = max(1.5, h * 0.095)

    rectangle(x, y, w, h, (0.94, 0.94, 0.92), alpha)
    rectangle_outline(x, y, w, h, (0.02, 0.02, 0.02), alpha, width=1.5)

    if vertical:
        midpoint = y + h / 2.0

        line(x, midpoint, x + w, midpoint, (0.02, 0.02, 0.02), alpha, width=1.1)

        centers = [
            (x + w * 0.5, y + h * 0.25),
            (x + w * 0.5, y + h * 0.75),
        ]

    else:
        midpoint = x + w / 2.0

        line(midpoint, y, midpoint, y + h, (0.02, 0.02, 0.02), alpha, width=1.1)

        centers = [
            (x + w * 0.25, y + h * 0.5),
            (x + w * 0.75, y + h * 0.5),
        ]

    for px, py in _pips_2d(left_value, centers[0][0], centers[0][1], half_w, half_h):
        circle(px, py, radius, (0.03, 0.03, 0.03), alpha, segments=14)

    for px, py in _pips_2d(right_value, centers[1][0], centers[1][1], half_w, half_h):
        circle(px, py, radius, (0.03, 0.03, 0.03), alpha, segments=14)


def _pips_2d(value, center_x, center_y, half_w, half_h):
    offset_x = half_w * 0.38
    offset_y = half_h * 0.45

    return [
        (center_x + dx, center_y + dy)
        for dx, dy in _pip_positions(value, offset_x, offset_y, y_down=True)
    ]
