"""Shared state and action encoder for all neural domino agents.

The neural policy only chooses voluntary tile plays. Forced draw/pass actions are
handled by the agent wrapper and are deliberately absent from the policy output.
"""

import numpy as np

from middleware.opponent_model import (
    ALL_TILES,
    compute_opponent_suit_probabilities,
    reconstruct_public_actions,
)


MAX_TURN = 52


class DominoEncoder:
    """Map public engine states to fixed-size vectors and tile plays to indices.

    Feature layout, total size 168:

    * 0..27: current player's hand, one bit per tile;
    * 28..55: tiles already played on the board;
    * 56..83: normalized turn when each tile was played, or 0 if unplayed;
    * 84..111: tiles played by the current player;
    * 112..139: tiles played by the opponent;
    * 140..146: left end one-hot;
    * 147..153: right end one-hot;
    * 154..155: hand sizes for player 0 and player 1, divided by 7;
    * 156: stock size divided by 14;
    * 157..158: draw counts for player 0 and player 1, divided by 14;
    * 159..160: pass counts for player 0 and player 1, divided by MAX_TURN;
    * 161..167: opponent suit-presence probabilities in [0, 1].

    The action space has 56 actions: 28 possible tiles on the left end followed
    by 28 possible tiles on the right end. Draw and pass are forced rule actions,
    not neural-policy actions.
    """

    VECTOR_SIZE = 168
    ACTION_SIZE = 56
    MAX_TURN = MAX_TURN

    HAND_OFFSET = 0
    PLAYED_OFFSET = 28
    PLAYED_TURN_OFFSET = 56
    PLAYED_BY_ME_OFFSET = 84
    PLAYED_BY_OPPONENT_OFFSET = 112
    LEFT_END_OFFSET = 140
    RIGHT_END_OFFSET = 147
    HAND_SIZE_OFFSET = 154
    STOCK_SIZE_OFFSET = 156
    DRAW_COUNT_OFFSET = 157
    PASS_COUNT_OFFSET = 159
    OPPONENT_SUIT_PROBABILITY_OFFSET = 161

    def __init__(self):
        self.all_tiles = list(ALL_TILES)

        self.all_actions = []
        for tile in self.all_tiles:
            self.all_actions.append((tile, 0))  # 0-27: play on the left end
        for tile in self.all_tiles:
            self.all_actions.append((tile, 1))  # 28-55: play on the right end

        self.action_to_index = {action: idx for idx, action in enumerate(self.all_actions)}

    def encode_state(self, state):
        """Convert a game state dictionary into a ``(168, 1)`` feature vector."""
        vector = np.zeros((self.VECTOR_SIZE, 1), dtype=float)
        current_player = state.get("current_player", 0)

        for tile in state.get("current_player_hand", []):
            tile = tuple(tile)
            vector[self.HAND_OFFSET + self.all_tiles.index(tile), 0] = 1.0

        draw_counts = [0, 0]
        pass_counts = [0, 0]

        for turn_index, entry in enumerate(reconstruct_public_actions(state)):
            action = entry.action
            actor = entry.actor

            if action is None:
                if actor < len(pass_counts):
                    pass_counts[actor] += 1
                continue

            if action == ("DRAW", None):
                if actor < len(draw_counts):
                    draw_counts[actor] += 1
                continue

            tile, _side = action
            tile_index = self.all_tiles.index(tuple(tile))
            normalized_turn = min(turn_index + 1, self.MAX_TURN) / self.MAX_TURN

            vector[self.PLAYED_OFFSET + tile_index, 0] = 1.0
            vector[self.PLAYED_TURN_OFFSET + tile_index, 0] = normalized_turn

            if actor == current_player:
                vector[self.PLAYED_BY_ME_OFFSET + tile_index, 0] = 1.0
            else:
                vector[self.PLAYED_BY_OPPONENT_OFFSET + tile_index, 0] = 1.0

        if state.get("ends"):
            left_end, right_end = state["ends"]
            vector[self.LEFT_END_OFFSET + int(left_end), 0] = 1.0
            vector[self.RIGHT_END_OFFSET + int(right_end), 0] = 1.0

        hand_sizes = state.get("hand_sizes", [])
        for i in range(min(2, len(hand_sizes))):
            vector[self.HAND_SIZE_OFFSET + i, 0] = hand_sizes[i] / 7.0

        vector[self.STOCK_SIZE_OFFSET, 0] = state.get("stock_size", 0) / 14.0

        for i in range(2):
            vector[self.DRAW_COUNT_OFFSET + i, 0] = draw_counts[i] / 14.0
            vector[self.PASS_COUNT_OFFSET + i, 0] = pass_counts[i] / self.MAX_TURN

        probabilities = compute_opponent_suit_probabilities(state)
        for suit, value in enumerate(probabilities):
            vector[self.OPPONENT_SUIT_PROBABILITY_OFFSET + suit, 0] = value

        return vector

    def is_policy_action(self, move):
        """Return True when ``move`` is a tile play represented by the network."""
        return move is not None and move[0] != "DRAW"

    def _normalize_policy_action(self, move):
        if not self.is_policy_action(move):
            raise ValueError(f"Forced action {move!r} is not part of the policy action space.")
        if isinstance(move[0], list):
            return (tuple(move[0]), move[1])
        return move

    def _action_index(self, move):
        """Return the policy index for a tile-play move, accepting list tiles."""
        move = self._normalize_policy_action(move)
        return self.action_to_index[move]

    def policy_action_mask(self, legal_actions):
        """Return a ``(56, 1)`` mask marking legal neural-policy actions."""
        mask = np.zeros((self.ACTION_SIZE, 1), dtype=float)

        for move in legal_actions:
            if self.is_policy_action(move):
                mask[self._action_index(move), 0] = 1.0

        return mask

    def decode_output(self, probabilities, legal_actions):
        """Return the legal tile play with the largest masked policy score.

        Forced draw/pass actions are not decoded here. The agent wrapper returns
        them before calling the network.
        """
        policy_actions = [move for move in legal_actions if self.is_policy_action(move)]
        if not policy_actions:
            return legal_actions[0] if legal_actions else None

        masked_scores = np.full(self.ACTION_SIZE, -np.inf)
        for move in policy_actions:
            masked_scores[self._action_index(move)] = probabilities[self._action_index(move), 0]

        return self.all_actions[int(np.argmax(masked_scores))]

    def sample_action(self, probabilities, legal_actions):
        """Sample a legal tile play from the masked policy distribution."""
        policy_actions = [move for move in legal_actions if self.is_policy_action(move)]
        if not policy_actions:
            forced_action = legal_actions[0] if legal_actions else None
            return forced_action, None

        legal_indices = [self._action_index(move) for move in policy_actions]
        legal_probs = probabilities[legal_indices, 0]

        total = legal_probs.sum()
        if total <= 0:
            legal_probs = np.ones(len(legal_indices)) / len(legal_indices)
        else:
            legal_probs = legal_probs / total

        chosen_position = np.random.choice(len(legal_indices), p=legal_probs)
        return policy_actions[chosen_position], legal_indices[chosen_position]
