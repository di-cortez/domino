"""Condense timed full-game pair files into one comparative JSON report."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "full_game_pair_analysis.json"
MANIFEST_FILE = "full_game_pair_generation_manifest.json"
AGENT_ORDER = ("random", "heuristic", "neural")
MATCHUPS = (
    ("random", "random"),
    ("random", "heuristic"),
    ("random", "neural"),
    ("heuristic", "heuristic"),
    ("heuristic", "neural"),
    ("neural", "neural"),
)
TIMING_COMPONENTS = (
    "state",
    "legal_actions",
    "agent_decision",
    "engine_transition",
    "turn_wall",
)
GAME_TIMING_COMPONENTS = ("setup", "simulation", "game_wall")
DECISION_CLASSES = (
    "forced_draw",
    "forced_pass",
    "forced_tile",
    "voluntary_choice",
)
PIPS = tuple(range(7))
TILES = tuple((left, right) for left in PIPS for right in range(left, 7))
TILE_INDEX = {tile: index for index, tile in enumerate(TILES)}
LOCATION_ORDER = (
    "player0_hand",
    "player1_hand",
    "both_hands",
    "table",
    "stock",
)
TIMING_BIN_UPPER_BOUNDS_US = (
    0,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1_000,
    2_000,
    4_000,
    8_000,
    16_000,
    32_000,
    64_000,
    128_000,
    256_000,
    512_000,
    1_000_000,
)


def matchup_key(pair):
    """Return the stable filename/configuration key for one pairing."""
    return f"{pair[0]}_vs_{pair[1]}"


def matchup_input(input_dir, pair):
    """Return the expected raw JSONL path for one pairing."""
    return input_dir / f"{matchup_key(pair)}_full_games.jsonl"


def selected_matchups(pattern):
    """Resolve ``all`` or a comma-separated list of exact matchup keys."""
    if pattern == "all":
        return MATCHUPS
    requested = {item.strip() for item in pattern.split(",") if item.strip()}
    known = {matchup_key(pair): pair for pair in MATCHUPS}
    unknown = requested - set(known)
    if unknown:
        raise ValueError(
            f"Unknown matchups {sorted(unknown)}; choose from {sorted(known)}"
        )
    return tuple(pair for pair in MATCHUPS if matchup_key(pair) in requested)


def canonical_tile(value):
    """Return one domino as an ordered tuple suitable for counting."""
    left, right = (int(part) for part in value)
    return (left, right) if left <= right else (right, left)


def is_draw(action):
    """Return whether an action is the forced draw marker."""
    return (
        isinstance(action, (list, tuple))
        and len(action) == 2
        and action[0] == "DRAW"
    )


def is_tile_play(action):
    """Return whether an action places a domino on the table."""
    return (
        isinstance(action, (list, tuple))
        and len(action) == 2
        and isinstance(action[0], (list, tuple))
    )


def table_tiles(board_history):
    """Extract only played dominoes from a board history."""
    return [
        canonical_tile(action[0])
        for action in board_history
        if is_tile_play(action)
    ]


def new_timing_accumulator():
    """Create an exact integer-microsecond distribution accumulator."""
    return Counter()


def new_component_accumulators(components=TIMING_COMPONENTS):
    """Create one timing histogram for every requested timed component."""
    return {component: new_timing_accumulator() for component in components}


def add_timing(accumulator, value, *, context):
    """Validate and add one integer microsecond observation."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context}: timing must be a non-negative integer")
    accumulator[value] += 1


def add_component_timings(accumulators, values, *, context):
    """Add one complete component dictionary to timing histograms."""
    if set(values) != set(accumulators):
        raise ValueError(
            f"{context}: expected timing keys {sorted(accumulators)}, "
            f"found {sorted(values)}"
        )
    for component, accumulator in accumulators.items():
        add_timing(accumulator, values[component], context=f"{context}/{component}")


def new_location_accumulator():
    """Create exact counters for the composition of one physical location."""
    return {
        "state_count": 0,
        "total_tiles": 0,
        "tile_presence": [0] * len(TILES),
        "pip_containing_tiles": [0] * len(PIPS),
        "pip_endpoints": [0] * len(PIPS),
    }


def new_turn_accumulator():
    """Create state and timing counters for one absolute engine turn."""
    return {
        "states": 0,
        "ongoing_states": 0,
        "terminal_states": 0,
        "size_histograms": {
            location: Counter() for location in LOCATION_ORDER
        },
        "locations": {
            location: new_location_accumulator()
            for location in LOCATION_ORDER
        },
        "timing": new_component_accumulators(),
        "timing_by_agent": {
            agent: new_component_accumulators() for agent in AGENT_ORDER
        },
        "decision_classes": Counter(),
    }


def new_matchup_accumulator():
    """Create the complete streaming state for one matchup."""
    return {
        "turns": [],
        "durations": Counter(),
        "seat_winners": Counter(),
        "agent_winners": Counter(),
        "win_reasons": Counter(),
        "actions": Counter(),
        "opening_tiles": [0] * len(TILES),
        "game_timing": new_component_accumulators(GAME_TIMING_COMPONENTS),
        "summed_turn_wall": new_timing_accumulator(),
        "timing": new_component_accumulators(),
        "timing_by_agent": {
            agent: new_component_accumulators() for agent in AGENT_ORDER
        },
        "decision_classes": Counter(),
        "timing_by_decision_class": {
            category: new_timing_accumulator()
            for category in DECISION_CLASSES
        },
    }


def new_global_accumulator():
    """Create timing counters shared across all six matchups."""
    return {
        "timing_by_agent": {
            agent: new_component_accumulators() for agent in AGENT_ORDER
        },
        "timing_by_agent_and_turn": [],
        "timing_by_agent_and_decision_class": {
            agent: {
                category: new_timing_accumulator()
                for category in DECISION_CLASSES
            }
            for agent in AGENT_ORDER
        },
    }


def ensure_global_turn(global_accumulator, turn):
    """Ensure the cross-matchup timing table contains one absolute turn."""
    turns = global_accumulator["timing_by_agent_and_turn"]
    while len(turns) <= turn:
        turns.append({
            agent: new_component_accumulators()
            for agent in AGENT_ORDER
        })
    return turns[turn]


def add_location(accumulator, tiles):
    """Add one observed collection of unique dominoes to a location."""
    accumulator["state_count"] += 1
    accumulator["total_tiles"] += len(tiles)
    for tile in tiles:
        index = TILE_INDEX[tile]
        accumulator["tile_presence"][index] += 1
        left, right = tile
        accumulator["pip_containing_tiles"][left] += 1
        if right != left:
            accumulator["pip_containing_tiles"][right] += 1
        accumulator["pip_endpoints"][left] += 1
        accumulator["pip_endpoints"][right] += 1


def add_state(turns, turn, hands, table, stock, *, terminal):
    """Record one complete omniscient state after ``turn`` actions."""
    while len(turns) <= turn:
        turns.append(new_turn_accumulator())
    accumulator = turns[turn]
    accumulator["states"] += 1
    accumulator["terminal_states" if terminal else "ongoing_states"] += 1
    locations = {
        "player0_hand": hands[0],
        "player1_hand": hands[1],
        "both_hands": hands[0] + hands[1],
        "table": table,
        "stock": stock,
    }
    if sum(
        len(values)
        for name, values in locations.items()
        if name != "both_hands"
    ) != 28:
        raise ValueError(f"Turn {turn}: domino conservation check failed")
    for location, tiles in locations.items():
        accumulator["size_histograms"][location][len(tiles)] += 1
        add_location(accumulator["locations"][location], tiles)


def histogram_value_at(histogram, rank):
    """Return the observed integer value at one zero-based sorted rank."""
    consumed = 0
    for value in sorted(histogram):
        consumed += histogram[value]
        if rank < consumed:
            return value
    raise ValueError("rank falls outside histogram")


def histogram_quantile(histogram, fraction):
    """Return a linearly interpolated quantile from an exact histogram."""
    count = sum(histogram.values())
    if count == 0:
        return None
    position = fraction * (count - 1)
    lower_rank = math.floor(position)
    upper_rank = math.ceil(position)
    lower = histogram_value_at(histogram, lower_rank)
    upper = histogram_value_at(histogram, upper_rank)
    return lower + (upper - lower) * (position - lower_rank)


def rounded(value, digits=6):
    """Round finite JSON statistics consistently."""
    return round(float(value), digits)


def summarize_histogram(histogram, *, include_dense_histogram=True):
    """Return exact descriptive statistics for a nonempty histogram."""
    count = sum(histogram.values())
    if not count:
        return None
    minimum = min(histogram)
    maximum = max(histogram)
    weighted_sum = sum(
        value * frequency for value, frequency in histogram.items()
    )
    mean = weighted_sum / count
    variance = sum(
        frequency * (value - mean) ** 2
        for value, frequency in histogram.items()
    ) / count
    result = {
        "count": count,
        "mean": rounded(mean),
        "stddev": rounded(math.sqrt(variance)),
        "min": minimum,
        "p25": rounded(histogram_quantile(histogram, 0.25)),
        "median": rounded(histogram_quantile(histogram, 0.50)),
        "p75": rounded(histogram_quantile(histogram, 0.75)),
        "p90": rounded(histogram_quantile(histogram, 0.90)),
        "p95": rounded(histogram_quantile(histogram, 0.95)),
        "p99": rounded(histogram_quantile(histogram, 0.99)),
        "max": maximum,
    }
    if include_dense_histogram:
        result["histogram"] = {
            "first_value": minimum,
            "counts": [
                histogram.get(value, 0)
                for value in range(minimum, maximum + 1)
            ],
        }
    return result


def timing_bin_labels():
    """Return stable labels for compact logarithmic timing histograms."""
    labels = []
    lower = 0
    for upper in TIMING_BIN_UPPER_BOUNDS_US:
        labels.append(str(upper) if lower == upper else f"{lower}-{upper}")
        lower = upper + 1
    labels.append(f">{TIMING_BIN_UPPER_BOUNDS_US[-1]}")
    return labels


def timing_bin_counts(histogram):
    """Collapse exact microseconds into stable logarithmic-style bins."""
    counts = [0] * (len(TIMING_BIN_UPPER_BOUNDS_US) + 1)
    for value, frequency in histogram.items():
        index = 0
        while (
            index < len(TIMING_BIN_UPPER_BOUNDS_US)
            and value > TIMING_BIN_UPPER_BOUNDS_US[index]
        ):
            index += 1
        counts[index] += frequency
    return counts


def summarize_timing(histogram, *, include_histogram=False):
    """Summarize one microsecond distribution without emitting raw samples."""
    result = summarize_histogram(histogram, include_dense_histogram=False)
    if result is None:
        return None
    result["unit"] = "microseconds"
    if include_histogram:
        result["histogram"] = {
            "bin_labels": timing_bin_labels(),
            "counts": timing_bin_counts(histogram),
        }
    return result


def summarize_components(accumulators, *, include_histogram=False):
    """Summarize every named component, omitting empty distributions."""
    return {
        component: summary
        for component, accumulator in accumulators.items()
        if (summary := summarize_timing(
            accumulator,
            include_histogram=include_histogram,
        )) is not None
    }


def percent(numerator, denominator):
    """Return a compact percentage, including a stable zero denominator rule."""
    return rounded(100.0 * numerator / denominator) if denominator else 0.0


def ranked_tiles(counts, state_count, *, reverse, limit=5):
    """Return the most or least frequent present tiles for quick inspection."""
    candidates = [
        (count, index)
        for index, count in enumerate(counts)
        if count > 0
    ]
    candidates.sort(
        key=lambda item: ((-item[0], item[1]) if reverse else item)
    )
    return [
        {
            "tile": f"[{TILES[index][0]}:{TILES[index][1]}]",
            "states": count,
            "state_percent": percent(count, state_count),
        }
        for count, index in candidates[:limit]
    ]


def summarize_location(accumulator):
    """Return complete tile/pip frequencies plus concise rankings."""
    states = accumulator["state_count"]
    total_tiles = accumulator["total_tiles"]
    tile_counts = accumulator["tile_presence"]
    pip_tile_counts = accumulator["pip_containing_tiles"]
    endpoint_counts = accumulator["pip_endpoints"]
    return {
        "observed_states": states,
        "total_tiles_observed": total_tiles,
        "mean_tile_count": rounded(total_tiles / states),
        "tile_presence_counts": tile_counts,
        "tile_presence_percent": [
            percent(count, states) for count in tile_counts
        ],
        "pip_composition": {
            "tiles_containing_pip_counts": pip_tile_counts,
            "mean_tiles_containing_pip_per_state": [
                rounded(count / states) for count in pip_tile_counts
            ],
            "share_of_location_tiles_percent": [
                percent(count, total_tiles) for count in pip_tile_counts
            ],
            "endpoint_counts": endpoint_counts,
            "endpoint_share_percent": [
                percent(count, 2 * total_tiles) for count in endpoint_counts
            ],
        },
        "most_frequent_tiles": ranked_tiles(
            tile_counts, states, reverse=True
        ),
        "least_frequent_present_tiles": ranked_tiles(
            tile_counts, states, reverse=False
        ),
    }


def summarize_turn(turn, accumulator, game_count):
    """Convert one matchup turn accumulator to persisted form."""
    agent_timing = {
        agent: summarize_components(values)
        for agent, values in accumulator["timing_by_agent"].items()
        if sum(values["turn_wall"].values())
    }
    return {
        "turn": turn,
        "games_observed": accumulator["states"],
        "games_observed_percent": percent(accumulator["states"], game_count),
        "ongoing_states": accumulator["ongoing_states"],
        "terminal_states": accumulator["terminal_states"],
        "sizes": {
            location: summarize_histogram(histogram)
            for location, histogram in accumulator["size_histograms"].items()
        },
        "composition": {
            location: summarize_location(location_accumulator)
            for location, location_accumulator in accumulator["locations"].items()
        },
        "timing_us": summarize_components(accumulator["timing"]),
        "timing_by_agent_us": agent_timing,
        "decision_class_counts": {
            category: accumulator["decision_classes"].get(category, 0)
            for category in DECISION_CLASSES
        },
    }


def action_kind(action):
    """Classify one serialized engine action."""
    if action is None:
        return "pass"
    if is_draw(action):
        return "draw"
    if is_tile_play(action):
        return "tile_play"
    raise ValueError(f"Unknown action representation: {action!r}")


def validate_serialized_state(game_number, state, hands, table, stock):
    """Check reconstruction against the public state saved in the JSONL."""
    if list(state["hand_sizes"]) != [len(hands[0]), len(hands[1])]:
        raise ValueError(f"Game {game_number}: reconstructed hand sizes differ")
    if int(state["stock_size"]) != len(stock):
        raise ValueError(f"Game {game_number}: reconstructed stock size differs")
    if table_tiles(state["board_history"]) != table:
        raise ValueError(f"Game {game_number}: reconstructed table differs")


def apply_action(game_number, action, player, hands, table, stock):
    """Apply one recorded action to the omniscient reconstruction."""
    kind = action_kind(action)
    if kind == "draw":
        if not stock:
            raise ValueError(f"Game {game_number}: attempted draw from empty stock")
        hands[player].append(stock.pop(0))
    elif kind == "tile_play":
        tile = canonical_tile(action[0])
        try:
            hands[player].remove(tile)
        except ValueError as exc:
            raise ValueError(
                f"Game {game_number}: player {player} does not hold {tile}"
            ) from exc
        table.append(tile)
    return kind


def validate_pair_record(record, expected_matchup, game_number):
    """Validate pair identity and seat metadata before aggregating a game."""
    if record.get("matchup") != expected_matchup:
        raise ValueError(
            f"Game {game_number}: expected matchup {expected_matchup}, "
            f"found {record.get('matchup')!r}"
        )
    seats = record.get("seat_agents")
    if (
        not isinstance(seats, list)
        or len(seats) != 2
        or any(agent not in AGENT_ORDER for agent in seats)
    ):
        raise ValueError(f"Game {game_number}: invalid seat_agents {seats!r}")
    return seats


def add_turn_timing(
    accumulator,
    global_accumulator,
    turn,
    agent,
    category,
    values,
    *,
    context,
):
    """Update matchup, turn, agent, class, and cross-matchup timings."""
    turn_accumulator = accumulator["turns"][turn]
    add_component_timings(accumulator["timing"], values, context=context)
    add_component_timings(
        accumulator["timing_by_agent"][agent], values, context=context
    )
    add_component_timings(turn_accumulator["timing"], values, context=context)
    add_component_timings(
        turn_accumulator["timing_by_agent"][agent], values, context=context
    )
    add_component_timings(
        global_accumulator["timing_by_agent"][agent], values, context=context
    )
    global_turn = ensure_global_turn(global_accumulator, turn)
    add_component_timings(global_turn[agent], values, context=context)
    accumulator["decision_classes"][category] += 1
    turn_accumulator["decision_classes"][category] += 1
    add_timing(
        accumulator["timing_by_decision_class"][category],
        values["agent_decision"],
        context=context,
    )
    add_timing(
        global_accumulator["timing_by_agent_and_decision_class"][agent][category],
        values["agent_decision"],
        context=context,
    )


def analyze_matchup(input_path, pair, global_accumulator, progress_every=1_000):
    """Stream one pair JSONL and aggregate dynamics plus timing."""
    expected_matchup = matchup_key(pair)
    accumulator = new_matchup_accumulator()
    digest = hashlib.sha256()
    game_count = 0

    with input_path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            digest.update(raw_line)
            record = json.loads(raw_line)
            game_count += 1
            game_number = int(record.get("game", line_number))
            seats = validate_pair_record(record, expected_matchup, game_number)
            initial = record["initial_state"]
            hands = [
                [canonical_tile(tile) for tile in hand]
                for hand in initial["hands"]
            ]
            table = table_tiles(initial["logical_board"])
            stock = [canonical_tile(tile) for tile in initial["stock"]]
            history = record["history"]
            summed_turn_wall = 0
            opening_recorded = False

            for turn, item in enumerate(history):
                state = item["state"]
                if int(state["turn"]) != turn:
                    raise ValueError(
                        f"Game {game_number}: expected turn {turn}, "
                        f"found {state['turn']}"
                    )
                validate_serialized_state(
                    game_number, state, hands, table, stock
                )
                add_state(
                    accumulator["turns"],
                    turn,
                    hands,
                    table,
                    stock,
                    terminal=False,
                )
                player = int(state["current_player"])
                acting_agent = item["acting_agent"]
                if acting_agent != seats[player]:
                    raise ValueError(
                        f"Game {game_number}, turn {turn}: acting agent differs"
                    )
                category = item["decision_class"]
                if category not in DECISION_CLASSES:
                    raise ValueError(
                        f"Game {game_number}, turn {turn}: "
                        f"unknown decision class {category!r}"
                    )
                values = item["timing_us"]
                add_turn_timing(
                    accumulator,
                    global_accumulator,
                    turn,
                    acting_agent,
                    category,
                    values,
                    context=f"game {game_number}/turn {turn}",
                )
                summed_turn_wall += int(values["turn_wall"])
                kind = apply_action(
                    game_number,
                    item["target_action"],
                    player,
                    hands,
                    table,
                    stock,
                )
                accumulator["actions"][kind] += 1
                if kind == "tile_play" and not opening_recorded and len(table) == 1:
                    accumulator["opening_tiles"][TILE_INDEX[table[0]]] += 1
                    opening_recorded = True

            final = record["final_state"]
            final_turn = int(final["turn"])
            if final_turn != len(history):
                raise ValueError(
                    f"Game {game_number}: final turn/history length differ"
                )
            final_hands = [
                [canonical_tile(tile) for tile in hand]
                for hand in final["hands"]
            ]
            final_table = table_tiles(final["logical_board"])
            final_stock = [canonical_tile(tile) for tile in final["stock"]]
            if (
                any(
                    Counter(left) != Counter(right)
                    for left, right in zip(hands, final_hands)
                )
                or table != final_table
                or stock != final_stock
            ):
                raise ValueError(
                    f"Game {game_number}: final reconstruction differs"
                )
            add_state(
                accumulator["turns"],
                final_turn,
                final_hands,
                final_table,
                final_stock,
                terminal=True,
            )
            accumulator["durations"][final_turn] += 1
            winner = int(final["winner"])
            if winner not in (0, 1):
                raise ValueError(f"Game {game_number}: invalid winner {winner}")
            accumulator["seat_winners"][str(winner)] += 1
            accumulator["agent_winners"][seats[winner]] += 1
            accumulator["win_reasons"][str(final["win_reason"])] += 1
            add_component_timings(
                accumulator["game_timing"],
                record["timing_us"],
                context=f"game {game_number}/game timing",
            )
            add_timing(
                accumulator["summed_turn_wall"],
                summed_turn_wall,
                context=f"game {game_number}/summed turns",
            )

            if progress_every and game_count % progress_every == 0:
                print(
                    f"{expected_matchup}: analyzed {game_count:,} games",
                    flush=True,
                )

    if game_count == 0:
        raise ValueError(f"No games found in {input_path}")
    if sum(accumulator["opening_tiles"]) != game_count:
        raise ValueError(
            f"{expected_matchup}: not every game has one opening tile"
        )
    return summarize_matchup(
        input_path,
        pair,
        accumulator,
        game_count,
        digest.hexdigest(),
    )


def summarize_matchup(input_path, pair, accumulator, game_count, digest):
    """Convert one completed matchup accumulator to compact JSON data."""
    opening_tiles = accumulator["opening_tiles"]
    agent_timing = {
        agent: summarize_components(values, include_histogram=True)
        for agent, values in accumulator["timing_by_agent"].items()
        if sum(values["turn_wall"].values())
    }
    class_timing = {
        category: summarize_timing(values)
        for category, values in accumulator["timing_by_decision_class"].items()
        if values
    }
    return {
        "canonical_agents": list(pair),
        "games": game_count,
        "source": {
            "file": input_path.name,
            "bytes": input_path.stat().st_size,
            "sha256": digest,
        },
        "game_summary": {
            "duration_in_engine_turns": summarize_histogram(
                accumulator["durations"]
            ),
            "actions": dict(sorted(accumulator["actions"].items())),
            "seat_winner_counts": {
                seat: accumulator["seat_winners"].get(seat, 0)
                for seat in ("0", "1")
            },
            "agent_winner_counts": {
                agent: accumulator["agent_winners"].get(agent, 0)
                for agent in AGENT_ORDER
                if agent in pair
            },
            "win_reason_counts": dict(
                sorted(accumulator["win_reasons"].items())
            ),
            "opening_table_tile": {
                "counts": opening_tiles,
                "percent": [
                    percent(count, game_count) for count in opening_tiles
                ],
                "most_frequent": ranked_tiles(
                    opening_tiles, game_count, reverse=True, limit=10
                ),
                "least_frequent_present": ranked_tiles(
                    opening_tiles, game_count, reverse=False, limit=10
                ),
            },
        },
        "timing_summary_us": {
            "per_game": summarize_components(
                accumulator["game_timing"], include_histogram=True
            ),
            "summed_turn_wall_per_game": summarize_timing(
                accumulator["summed_turn_wall"], include_histogram=True
            ),
            "per_move": summarize_components(
                accumulator["timing"], include_histogram=True
            ),
            "per_move_by_agent": agent_timing,
            "agent_decision_by_class": class_timing,
            "decision_class_counts": {
                category: accumulator["decision_classes"].get(category, 0)
                for category in DECISION_CLASSES
            },
        },
        "turns": [
            summarize_turn(turn, turn_accumulator, game_count)
            for turn, turn_accumulator in enumerate(accumulator["turns"])
            if turn_accumulator["states"]
        ],
    }


def summarize_global_timing(global_accumulator):
    """Return cross-matchup timing views by agent, turn, and decision class."""
    by_agent = {
        agent: summarize_components(values, include_histogram=True)
        for agent, values in global_accumulator["timing_by_agent"].items()
        if sum(values["turn_wall"].values())
    }
    by_turn = []
    for turn, agents in enumerate(
        global_accumulator["timing_by_agent_and_turn"]
    ):
        summaries = {
            agent: summarize_components(values)
            for agent, values in agents.items()
            if sum(values["turn_wall"].values())
        }
        if summaries:
            by_turn.append({"turn": turn, "agents": summaries})
    by_class = {
        agent: {
            category: summary
            for category, values in categories.items()
            if (summary := summarize_timing(values)) is not None
        }
        for agent, categories in global_accumulator[
            "timing_by_agent_and_decision_class"
        ].items()
    }
    return {
        "per_move_by_agent": by_agent,
        "per_move_by_agent_and_turn": by_turn,
        "agent_decision_by_agent_and_class": by_class,
    }


def load_generation_manifest(input_dir):
    """Load the generator provenance when it exists beside the raw files."""
    path = input_dir / MANIFEST_FILE
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def validate_manifest_sources(manifest, matchups):
    """Ensure analyzed raw files match the generator's published hashes."""
    if manifest is None:
        return
    expected = {
        item["matchup"]: item for item in manifest.get("matchups", [])
    }
    for key, report in matchups.items():
        source = report["source"]
        manifest_entry = expected.get(key)
        if manifest_entry is None:
            raise ValueError(f"Generation manifest has no entry for {key}")
        if source["sha256"] != manifest_entry.get("output_sha256"):
            raise ValueError(f"Generation manifest hash differs for {key}")
        if source["bytes"] != manifest_entry.get("output_bytes"):
            raise ValueError(f"Generation manifest size differs for {key}")


def analyze_all(input_dir, pairs, progress_every=1_000):
    """Analyze selected pair files and produce one self-describing report."""
    global_accumulator = new_global_accumulator()
    matchup_reports = {}
    for index, pair in enumerate(pairs, start=1):
        key = matchup_key(pair)
        source = matchup_input(input_dir, pair)
        if not source.is_file():
            raise FileNotFoundError(f"Missing raw matchup file: {source}")
        print(f"[{index}/{len(pairs)}] Analyzing {key}", flush=True)
        matchup_reports[key] = analyze_matchup(
            source,
            pair,
            global_accumulator,
            progress_every=progress_every,
        )
    manifest = load_generation_manifest(input_dir)
    validate_manifest_sources(manifest, matchup_reports)
    return {
        "format_version": 2,
        "purpose": (
            "comparative state dynamics and wall timing for all full-game "
            "random/heuristic/neural pairings"
        ),
        "games_total": sum(item["games"] for item in matchup_reports.values()),
        "matchup_order": list(matchup_reports),
        "agent_order": list(AGENT_ORDER),
        "generation_manifest": manifest,
        "schema": {
            "state_semantics": (
                "Turn t is the complete state after t engine actions. Each "
                "game contributes turns 0 through its terminal turn; games "
                "already finished before t are absent from that turn cohort."
            ),
            "mixed_seat_semantics": (
                "Mixed matchups alternate agent types between seats every game."
            ),
            "timing_semantics": (
                "All time values are integer wall-clock microseconds measured "
                "with perf_counter_ns in one sequential process. Component "
                "means overlap only through turn_wall, which encloses the four "
                "named operations plus small Python bookkeeping overhead."
            ),
            "tile_order": [f"[{left}:{right}]" for left, right in TILES],
            "pip_order": list(PIPS),
            "location_order": list(LOCATION_ORDER),
            "timing_component_order": list(TIMING_COMPONENTS),
            "decision_class_order": list(DECISION_CLASSES),
            "tile_frequency_semantics": (
                "tile_presence_counts counts states containing that unique "
                "tile; tile_presence_percent divides it by observed_states."
            ),
            "pip_frequency_semantics": (
                "tiles_containing_pip counts a double once, while endpoint_counts "
                "counts both ends of every tile and counts a double twice."
            ),
            "size_histogram_semantics": (
                "counts[i] belongs to value first_value+i."
            ),
        },
        "global_timing_summary_us": summarize_global_timing(global_accumulator),
        "matchups": matchup_reports,
    }


def write_json_atomic(path, value, *, pretty):
    """Write the compact report atomically beside its final destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            if pretty:
                json.dump(value, stream, ensure_ascii=False, indent=2)
            else:
                json.dump(
                    value,
                    stream,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args():
    """Parse input directory, selected matchups, and report controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--matchups",
        default="all",
        help="'all' or comma-separated canonical matchup names.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Indent the output; the default is compact JSON.",
    )
    parser.add_argument(
        "--progress-every", type=int, default=1_000, metavar="GAMES"
    )
    args = parser.parse_args()
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    try:
        args.selected_matchups = selected_matchups(args.matchups)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main():
    """Analyze every selected JSONL and write one compact report."""
    args = parse_args()
    report = analyze_all(
        args.input_dir,
        args.selected_matchups,
        progress_every=args.progress_every,
    )
    write_json_atomic(args.output, report, pretty=args.pretty)
    source_bytes = sum(
        item["source"]["bytes"] for item in report["matchups"].values()
    )
    reduction = 100.0 * args.output.stat().st_size / source_bytes
    print(
        f"Saved {report['games_total']:,} games as {args.output} "
        f"({args.output.stat().st_size:,} bytes, "
        f"{reduction:.3f}% of raw sources)."
    )


if __name__ == "__main__":
    main()
