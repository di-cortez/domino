"""Small baseline agents and compatibility exports for the agent package."""

import random

from middleware.middleware import Agent, GameManager  # noqa: F401


class RandomAgent(Agent):
    """Choose uniformly from the legal action list."""

    def choose_move(self, state, legal_actions):
        return random.choice(legal_actions)


class GreedyAgent(Agent):
    """Play the legal tile with the largest pip sum."""

    def choose_move(self, state, legal_actions):
        tile_moves = [action for action in legal_actions if action is not None and action[0] != "DRAW"]
        if not tile_moves:
            return legal_actions[0]

        return max(tile_moves, key=lambda action: action[0][0] + action[0][1])


if __name__ == "__main__":
    from middleware.domino_engine import DominoEngine

    engine = DominoEngine(player_count=2)
    manager = GameManager(engine, [GreedyAgent(), RandomAgent()])
    manager.play_full_game()

    final_state = engine.to_dict()
    print("\n--- Game Result ---")
    if final_state["winner"] == -1:
        print("Result: draw")
    else:
        print(f"Result: player {final_state['winner']} won")

    print(f"Turns processed: {final_state['turn']}")
    print(f"Tiles left in stock: {len(final_state['stock'])}")
    print(f"Player 0 final hand size: {len(final_state['hands'][0])}")
    print(f"Player 1 final hand size: {len(final_state['hands'][1])}")
