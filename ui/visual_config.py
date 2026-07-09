"""
Visual geometry constants for the board layout.

Changing these values affects tile size, spacing, and the area limits used by
the layout algorithm when it decides to turn a branch.
"""

INITIAL_X = 0.0
INITIAL_Y = 3.35

TILE_SCALE = 0.75
TILE_GAP = 0.10

X_LIMIT = 8.0

HORIZONTAL_TILE_WIDTH = 2.0 * TILE_SCALE
HORIZONTAL_TILE_HEIGHT = 1.0 * TILE_SCALE

VERTICAL_TILE_WIDTH = 1.0 * TILE_SCALE
VERTICAL_TILE_HEIGHT = 2.0 * TILE_SCALE

LOWER_Y_LIMIT = -5.0
UPPER_Y_LIMIT = INITIAL_Y + VERTICAL_TILE_HEIGHT / 2.0
