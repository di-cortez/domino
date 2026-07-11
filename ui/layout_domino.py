"""
Geometry for the domino chain on the board.

This module does not draw anything. It receives the renderer's left-to-right
tile chain and calculates positions, angles, and pip order while respecting the
board limits and turning branches near the edges.
"""

from ui.visual_config import (
    HORIZONTAL_TILE_HEIGHT,
    HORIZONTAL_TILE_WIDTH,
    LOWER_Y_LIMIT,
    TILE_GAP,
    UPPER_Y_LIMIT,
    VERTICAL_TILE_HEIGHT,
    VERTICAL_TILE_WIDTH,
    X_LIMIT,
)


RIGHT_PATH = [
    {"name": "top_right", "dx": 1, "dy": 0},
    {"name": "down_right", "dx": 0, "dy": -1},
    {"name": "bottom_left", "dx": -1, "dy": 0},
    {"name": "up_left", "dx": 0, "dy": 1},
    {"name": "top_right_2", "dx": 1, "dy": 0},
]

LEFT_PATH = [
    {"name": "top_left", "dx": -1, "dy": 0},
    {"name": "down_left", "dx": 0, "dy": -1},
    {"name": "bottom_right", "dx": 1, "dy": 0},
    {"name": "up_right", "dx": 0, "dy": 1},
    {"name": "top_left_2", "dx": -1, "dy": 0},
]


def get_path(direction):
    if direction == 1:
        return RIGHT_PATH

    if direction == -1:
        return LEFT_PATH

    raise ValueError(f"Invalid branch direction: {direction}")


def is_double(info):
    first, second = info["tile"]
    return first == second


def tile_values(info):
    return tuple(info["tile"])


def common_value(previous_info, current_info):
    """
    Value by which `current_info` touches `previous_info`.

    The engine usually sends `connected_value`. If it does not, the value is
    inferred from the intersection between the two tiles.
    """
    if "connected_value" in current_info:
        return current_info["connected_value"]

    previous_a, previous_b = tile_values(previous_info)
    current_a, current_b = tile_values(current_info)

    previous_values = {previous_a, previous_b}

    if current_a in previous_values:
        return current_a

    if current_b in previous_values:
        return current_b

    raise ValueError(
        "Consecutive tiles do not share a value: "
        f"{previous_info['tile']} and {current_info['tile']}"
    )


def other_value(info, connected_value):
    """Return the value on the tile that is not connected to the previous tile."""
    first, second = tile_values(info)

    if first == connected_value:
        return second

    if second == connected_value:
        return first

    raise ValueError(
        f"Connected value {connected_value} does not appear in tile {info['tile']}"
    )


def is_horizontal_move(dx, dy):
    return dx != 0 and dy == 0


def is_vertical_move(dx, dy):
    return dx == 0 and dy != 0


def angle_for_direction(info, dx, dy):
    """
    General orientation rule.

    Horizontal movement:
    - non-double tiles lie horizontally;
    - doubles stand vertically.

    Vertical movement:
    - non-double tiles stand vertically;
    - doubles lie horizontally.
    """
    if is_horizontal_move(dx, dy):
        if is_double(info):
            return 90.0

        return 0.0

    if is_vertical_move(dx, dy):
        if is_double(info):
            return 0.0

        return 90.0

    raise ValueError(f"Invalid direction: dx={dx}, dy={dy}")


def inline_angle(info):
    """Default horizontal angle used by the pivot tile."""
    return angle_for_direction(info, dx=1, dy=0)


def descending_angle(info):
    """Compatibility angle for a downward vertical segment."""
    return angle_for_direction(info, dx=0, dy=-1)


def corner_angle(info, old_segment, new_segment):
    """
    Angle for the tile that enters a corner.

    Non-doubles use the new direction. Doubles use the old direction so the
    first double after a corner remains visually crossed relative to the
    previous segment.
    """
    if is_double(info):
        return angle_for_direction(
            info,
            old_segment["dx"],
            old_segment["dy"],
        )

    return angle_for_direction(
        info,
        new_segment["dx"],
        new_segment["dy"],
    )


def width_for_angle(angle):
    if angle in (90.0, -90.0):
        return VERTICAL_TILE_WIDTH

    return HORIZONTAL_TILE_WIDTH


def height_for_angle(angle):
    if angle in (90.0, -90.0):
        return VERTICAL_TILE_HEIGHT

    return HORIZONTAL_TILE_HEIGHT


def dimensions_for_angle(angle):
    return width_for_angle(angle), height_for_angle(angle)


def axis_extent(angle, dx, dy):
    """
    Return tile size along the movement axis.

    Horizontal movement uses width; vertical movement uses height.
    """
    width, height = dimensions_for_angle(angle)

    if is_horizontal_move(dx, dy):
        return width

    if is_vertical_move(dx, dy):
        return height

    raise ValueError(f"Invalid direction: dx={dx}, dy={dy}")


def values_for_direction(previous_info, current_info, dx, dy):
    """
    Choose pip order so the connected value stays on the input side.

    dx =  1, dy =  0: moving right, connected value stays left.
    dx = -1, dy =  0: moving left, connected value stays right.
    dx =  0, dy = -1: moving down, connected value stays above.
    dx =  0, dy =  1: moving up, connected value stays below.
    """
    connected_value = common_value(previous_info, current_info)
    free_value = other_value(current_info, connected_value)

    if dx == 1 and dy == 0:
        return connected_value, free_value

    if dx == -1 and dy == 0:
        return free_value, connected_value

    if dx == 0 and dy == -1:
        # With a 90-degree rotation, the first rendered value appears below and
        # the second rendered value appears above.
        return free_value, connected_value

    if dx == 0 and dy == 1:
        return connected_value, free_value

    raise ValueError(f"Invalid direction: dx={dx}, dy={dy}")


def fits_in_area(x_pos, y_pos, angle):
    """Return whether the oriented tile bounding box fits inside board limits."""
    width, height = dimensions_for_angle(angle)

    left = x_pos - width / 2.0
    right = x_pos + width / 2.0
    bottom = y_pos - height / 2.0
    top = y_pos + height / 2.0

    if left < -X_LIMIT:
        return False

    if right > X_LIMIT:
        return False

    if bottom < LOWER_Y_LIMIT:
        return False

    if top > UPPER_Y_LIMIT:
        return False

    return True


def inline_position(
    current_x,
    current_y,
    segment,
    previous_angle,
    current_angle,
):
    """Calculate the next tile position while staying on the same segment."""
    dx = segment["dx"]
    dy = segment["dy"]

    step = (
        axis_extent(previous_angle, dx, dy) / 2.0
        + axis_extent(current_angle, dx, dy) / 2.0
        + TILE_GAP
    )

    return (
        current_x + dx * step,
        current_y + dy * step,
    )


def corner_exit_offset(previous_info, previous_angle, old_segment):
    """
    Fine-tune the exit pip position at a corner.

    For non-doubles, the connection leaves from the center of the outer half
    instead of the center of the full tile. Doubles use the center.
    """
    if is_double(previous_info):
        return 0.0

    dx = old_segment["dx"]
    dy = old_segment["dy"]

    if is_horizontal_move(dx, dy):
        return width_for_angle(previous_angle) / 4.0

    if is_vertical_move(dx, dy):
        return height_for_angle(previous_angle) / 4.0

    raise ValueError(f"Invalid segment: {old_segment}")


def corner_position(
    current_x,
    current_y,
    previous_info,
    old_segment,
    new_segment,
    previous_angle,
    current_angle,
):
    """
    Calculate the first tile position after turning a corner.

    Horizontal-to-vertical turns adjust x by the previous tile's outer half and
    move y along the new segment. Vertical-to-horizontal turns do the inverse.
    """
    old_dx = old_segment["dx"]
    old_dy = old_segment["dy"]

    new_dx = new_segment["dx"]
    new_dy = new_segment["dy"]

    current_width, current_height = dimensions_for_angle(current_angle)

    exit_offset = corner_exit_offset(
        previous_info,
        previous_angle,
        old_segment,
    )

    if is_horizontal_move(old_dx, old_dy) and is_vertical_move(new_dx, new_dy):
        new_x = current_x + old_dx * exit_offset

        y_step = (
            height_for_angle(previous_angle) / 2.0
            + current_height / 2.0
            + TILE_GAP
        )

        new_y = current_y + new_dy * y_step

        return new_x, new_y

    if is_vertical_move(old_dx, old_dy) and is_horizontal_move(new_dx, new_dy):
        x_step = (
            width_for_angle(previous_angle) / 2.0
            + current_width / 2.0
            + TILE_GAP
        )

        new_x = current_x + new_dx * x_step
        new_y = current_y + old_dy * exit_offset

        return new_x, new_y

    # Fallback for unexpected path shapes: continue on the new segment.
    return inline_position(
        current_x,
        current_y,
        new_segment,
        previous_angle,
        current_angle,
    )


def create_slot(
    x_pos,
    y_pos,
    previous_info,
    current_info,
    segment,
    segment_index,
    slot_type,
    subtype,
    angle,
):
    dx = segment["dx"]
    dy = segment["dy"]

    return {
        "slot_type": slot_type,
        "subtype": subtype,
        "segment": segment["name"],
        "segment_index": segment_index,
        "dx": dx,
        "dy": dy,
        "x_pos": x_pos,
        "y_pos": y_pos,
        "angle": angle,
        "values": values_for_direction(previous_info, current_info, dx, dy),
        "next_segment_index": segment_index,
    }


def slot_type_for_segment(segment):
    if is_vertical_move(segment["dx"], segment["dy"]):
        return "vertical"

    return "line"


def calculate_next_slot(
    current_x,
    current_y,
    previous_info,
    current_info,
    previous_angle,
    path,
    current_segment_index,
):
    """
    Try to place the tile on the current segment.

    If the new tile would exceed board limits, move to the next segment and
    create a corner slot.
    """
    current_segment = path[current_segment_index]

    current_angle = angle_for_direction(
        current_info,
        current_segment["dx"],
        current_segment["dy"],
    )

    candidate_x, candidate_y = inline_position(
        current_x,
        current_y,
        current_segment,
        previous_angle,
        current_angle,
    )

    if fits_in_area(candidate_x, candidate_y, current_angle):
        slot_type = slot_type_for_segment(current_segment)

        return create_slot(
            candidate_x,
            candidate_y,
            previous_info,
            current_info,
            current_segment,
            current_segment_index,
            slot_type=slot_type,
            subtype=None,
            angle=current_angle,
        )

    next_index = min(
        current_segment_index + 1,
        len(path) - 1,
    )

    new_segment = path[next_index]

    turn_angle = corner_angle(
        current_info,
        current_segment,
        new_segment,
    )

    corner_x, corner_y = corner_position(
        current_x,
        current_y,
        previous_info,
        current_segment,
        new_segment,
        previous_angle,
        turn_angle,
    )

    return create_slot(
        corner_x,
        corner_y,
        previous_info,
        current_info,
        new_segment,
        next_index,
        slot_type="corner",
        subtype=f"{current_segment['name']}_to_{new_segment['name']}",
        angle=turn_angle,
    )


def split_chain_at_pivot(visual_chain, pivot_index):
    """
    Split `visual_chain`, which arrives in left-to-right order.

    Before the pivot is the left side; after the pivot is the right side. To draw
    outward from the pivot, the left side must be reversed while the right side
    is already in the correct order.
    """
    pivot = visual_chain[pivot_index]

    left_side = list(reversed(visual_chain[:pivot_index]))
    right_side = visual_chain[pivot_index + 1:]

    return left_side, pivot, right_side


def calculate_branch_slots(
    tiles,
    previous_tile_info,
    direction,
    initial_x,
    initial_y,
):
    """
    Calculate slots for one branch starting from the pivot.

    Returns a list of `(tile_info, slot)` pairs.

    direction =  1 -> right side.
    direction = -1 -> left side.
    """
    result = []

    path = get_path(direction)

    current_x = initial_x
    current_y = initial_y

    previous_info = previous_tile_info
    previous_angle = inline_angle(previous_tile_info)

    current_segment_index = 0

    for current_info in tiles:
        slot = calculate_next_slot(
            current_x,
            current_y,
            previous_info,
            current_info,
            previous_angle,
            path,
            current_segment_index,
        )

        result.append((current_info, slot))

        current_x = slot["x_pos"]
        current_y = slot["y_pos"]
        previous_angle = slot["angle"]
        current_segment_index = slot["next_segment_index"]

        previous_info = current_info

    return result
