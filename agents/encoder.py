"""Shared state and action encoder for all neural domino agents."""

import numpy as np


class DominoEncoder:
    """Map engine states to fixed-size vectors and actions to policy indices."""

    VECTOR_SIZE = 86

    def __init__(self):
        self.all_tiles = [(i, j) for i in range(7) for j in range(i, 7)]

        self.all_actions = []
        for tile in self.all_tiles:
            self.all_actions.append((tile, 0))  # 0-27: play on the left end
        for tile in self.all_tiles:
            self.all_actions.append((tile, 1))  # 28-55: play on the right end

        self.all_actions.append(("DRAW", None))  # 56
        self.all_actions.append(None)            # 57: pass
        self.action_to_index = {action: idx for idx, action in enumerate(self.all_actions)}

    def encode_state(self, state):
        """Convert a game state dictionary into an ``(86, 1)`` feature vector."""
        vector = np.zeros((self.VECTOR_SIZE, 1))

        for tile in state["current_player_hand"]:
            vector[self.all_tiles.index(tuple(tile)), 0] = 1.0

        if state["ends"]:
            vector[28 + state["ends"][0], 0] = 1.0
            vector[35 + state["ends"][1], 0] = 1.0

        for i, size in enumerate(state["hand_sizes"]):
            vector[42 + i, 0] = size / 7.0

        vector[49, 0] = state.get("stock_size", 0) / 14.0

        draw_count = 0
        for action in state.get("board_history", []):
            if action is None:
                continue
            if action == ["DRAW", None] or action == ("DRAW", None):
                draw_count += 1
                continue

            tile = tuple(action[0])
            vector[50 + self.all_tiles.index(tile), 0] = 1.0

        vector[78, 0] = draw_count / 14.0

        for value in state.get("opponent_dead_suits", []):
            vector[79 + value, 0] = 1.0

        return vector

    def _action_index(self, move):
        """Return the policy index for a move, accepting list or tuple tiles."""
        if move is None:
            return 57
        if move[0] != "DRAW" and isinstance(move[0], list):
            move = (tuple(move[0]), move[1])
        return self.action_to_index[move]

    def decode_output(self, logits, legal_actions):
        """Return the legal action with the largest masked policy score."""
        masked_q_values = np.full(58, -np.inf)
        for move in legal_actions:
            masked_q_values[self._action_index(move)] = logits[self._action_index(move), 0]

        return self.all_actions[np.argmax(masked_q_values)]

    def sample_action(self, probabilities, legal_actions):
        """Sample a legal action from the policy distribution restricted to legal moves."""
        legal_indices = [self._action_index(move) for move in legal_actions]
        legal_probs = probabilities[legal_indices, 0]

        total = legal_probs.sum()
        if total <= 0:
            legal_probs = np.ones(len(legal_indices)) / len(legal_indices)
        else:
            legal_probs = legal_probs / total

        chosen_position = np.random.choice(len(legal_indices), p=legal_probs)
        return legal_actions[chosen_position], legal_indices[chosen_position]
