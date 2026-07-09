"""Generate supervised-learning examples from heuristic-vs-heuristic games."""

import json
import os
import time

from agents.heuristic_agent import StrategicAgent
from middleware.domino_engine import DominoEngine
from middleware.middleware import GameManager


def generate_dataset(game_count, output_file):
    """Write one JSONL row per decision point."""
    print(f"Generating {game_count} games...")
    start_time = time.time()
    saved_turn_count = 0

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(game_count):
            engine = DominoEngine(player_count=2)
            manager = GameManager(engine, [StrategicAgent(), StrategicAgent()])
            _, game_history = manager.play_full_game()

            for turn in game_history:
                f.write(json.dumps(turn) + "\n")
                saved_turn_count += 1

            if (i + 1) % 5000 == 0:
                print(f"{i + 1} games simulated... ({saved_turn_count} examples)")

    elapsed_time = time.time() - start_time
    print("-" * 40)
    print("GENERATION COMPLETE")
    print(f"State/action pairs: {saved_turn_count}")
    print(f"Output file: {output_file}")
    print(f"Elapsed time: {elapsed_time:.2f}s")


if __name__ == "__main__":
    generate_dataset(game_count=30000, output_file="dataset/supervised_dataset.jsonl")
