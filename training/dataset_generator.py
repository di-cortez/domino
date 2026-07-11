"""Generate supervised-learning examples from real heuristic decisions.

Only states with at least two legal tile-play actions are written. Forced draw,
pass, opening-double, and single-tile-play turns are excluded from the dataset.
"""

import json
import os
import time

from agents.heuristic_agent import StrategicAgent
from middleware.domino_engine import DominoEngine
from middleware.middleware import GameManager
from utils.runtime_status import format_duration, print_memory_report

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def _normalize_action(action):
    """Return a normalized tile-play action or None for draw/pass."""
    if action is None:
        return None
    if action == ["DRAW", None] or action == ("DRAW", None):
        return None
    if isinstance(action[0], list):
        return (tuple(action[0]), action[1])
    return action


def _legal_tile_actions_from_state(state):
    """Reconstruct legal tile-play actions from a serialized state."""
    hand = [tuple(tile) for tile in state["current_player_hand"]]
    ends = state.get("ends", [])

    if not ends:
        doubles = [tile for tile in hand if tile[0] == tile[1]]
        if doubles:
            opening_double = max(doubles, key=lambda tile: tile[0])
            return [(opening_double, 0)]
        return [(tile, 0) for tile in hand]

    left_end, right_end = ends
    actions = []

    for tile in hand:
        if left_end in tile:
            actions.append((tile, 0))
        if right_end in tile:
            actions.append((tile, 1))

    if left_end == right_end:
        actions = [(tile, 0) for tile, _side in actions]

    return list(dict.fromkeys(actions))


def _is_real_decision_state(state):
    """Return True when the player had at least two legal tile-play choices."""
    return len(_legal_tile_actions_from_state(state)) >= 2


def generate_dataset(game_count, output_file):
    """Write one JSONL row per real decision point."""
    print(f"Generating {game_count} games...")
    print_memory_report("Dataset generation startup memory")
    start_time = time.time()
    saved_turn_count = 0
    skipped_turn_count = 0

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        game_range = range(game_count)
        if tqdm is not None:
            game_range = tqdm(game_range, total=game_count, desc="Generating dataset", unit="game")

        for _i in game_range:
            engine = DominoEngine(player_count=2)
            manager = GameManager(engine, [StrategicAgent(), StrategicAgent()])
            _, game_history = manager.play_full_game()

            for turn in game_history:
                state = turn["state"]
                target_action = _normalize_action(turn["target_action"])

                if target_action is None:
                    skipped_turn_count += 1
                    continue

                if not _is_real_decision_state(state):
                    skipped_turn_count += 1
                    continue

                f.write(json.dumps(turn) + "\n")
                saved_turn_count += 1

    elapsed_time = time.time() - start_time
    print("-" * 40)
    print("GENERATION COMPLETE")
    print(f"Real decision pairs: {saved_turn_count}")
    print(f"Forced turns skipped: {skipped_turn_count}")
    print(f"Output file: {output_file}")
    print(f"Elapsed time: {format_duration(elapsed_time)}")


if __name__ == "__main__":
    generate_dataset(game_count=30000, output_file="dataset/supervised_dataset.jsonl")
