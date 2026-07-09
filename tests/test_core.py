"""
Sequential core tests for the engine, encoder, and training history.

Run from the repository root with:

    python tests/test_core.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.encoder import DominoEncoder
from middleware.domino_engine import DominoEngine, infer_dead_suits
from middleware.middleware import GameManager


class FirstLegalAgent:
    def choose_move(self, state, legal_actions):
        return legal_actions[0]


def _run(name, fn):
    fn()
    print(f"OK - {name}")


def test_encoder_special_actions_are_stable():
    encoder = DominoEncoder()

    assert encoder.all_actions[56] == ("DRAW", None)
    assert encoder.all_actions[57] is None
    assert encoder._action_index(("DRAW", None)) == 56
    assert encoder._action_index(None) == 57


def test_encoder_accepts_list_tiles_from_json():
    encoder = DominoEncoder()

    assert encoder._action_index(([0, 6], 1)) == encoder._action_index(((0, 6), 1))


def test_engine_requires_highest_opening_double_when_present():
    engine = DominoEngine(player_count=2)
    player = engine.current_player

    engine.ends = []
    engine.hands[player] = [(0, 0), (6, 6), (1, 2)]
    engine.required_opening_tile = (6, 6)

    assert engine.valid_actions(player) == [((6, 6), 0)]


def test_infer_dead_suits_from_draw_and_pass_history():
    board_history = [((2, 3), 0), ("DRAW", None), None]

    dead_suits = infer_dead_suits(
        board_history=board_history,
        hand_sizes=[7, 7],
        current_player=0,
    )

    assert dead_suits[1] == {2, 3}
    assert dead_suits[0] == set()


def test_game_manager_training_history_excludes_visual_metadata():
    engine = DominoEngine(player_count=2)
    manager = GameManager(engine, [FirstLegalAgent(), FirstLegalAgent()])

    manager.play_turn()

    assert len(manager.training_history) == 1
    row = manager.training_history[0]

    assert "state" in row
    assert "target_action" in row
    assert "visual_chain" not in row["state"]


def main():
    tests = [
        ("encoder special actions", test_encoder_special_actions_are_stable),
        ("encoder JSON tile actions", test_encoder_accepts_list_tiles_from_json),
        ("opening double rule", test_engine_requires_highest_opening_double_when_present),
        ("dead suit inference", test_infer_dead_suits_from_draw_and_pass_history),
        ("training history shape", test_game_manager_training_history_excludes_visual_metadata),
    ]

    for name, fn in tests:
        _run(name, fn)

    print(f"\n{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
