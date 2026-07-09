"""Domino rules engine shared by the UI, agents, and training code."""

import copy
import random


def _is_draw(action):
    return action is not None and action[0] == "DRAW"


def infer_dead_suits(board_history, hand_sizes, current_player):
    """
    Infer values that each player likely could not play when they passed or drew.

    Every non-draw action advances the actor. A draw keeps the turn with the
    same player. Walking the action history with that rule reconstructs who
    acted at each board state.
    """
    player_count = len(hand_sizes)
    absences = {i: set() for i in range(player_count)}
    if not board_history:
        return absences

    advances = sum(1 for action in board_history if not _is_draw(action))
    actor = (current_player - advances) % player_count

    ends = None
    for action in board_history:
        if action is None or _is_draw(action):
            if ends:
                absences[actor].update(ends)
            if action is None:
                actor = (actor + 1) % player_count
            continue

        tile = tuple(action[0])
        side = action[1]
        if ends is None:
            ends = [tile[0], tile[1]]
        else:
            connected_value = ends[side]
            ends[side] = tile[1] if tile[0] == connected_value else tile[0]
        actor = (actor + 1) % player_count

    return absences


class DominoEngine:
    """Stateful two-player domino engine with draw/pass and blocked-game rules."""

    def __init__(self, player_count=2):
        self.player_count = player_count
        self.all_tiles = [(i, j) for i in range(7) for j in range(i, 7)]
        self.reset()

    def print_tiles(self):
        for tile in self.all_tiles:
            print(tile)

    def reset(self):
        """Start a new game, shuffle, deal, and choose the opening player."""
        shuffled_tiles = self.all_tiles.copy()
        random.shuffle(shuffled_tiles)

        self.hands = [
            shuffled_tiles[i * 7:(i + 1) * 7]
            for i in range(self.player_count)
        ]
        self.stock = shuffled_tiles[self.player_count * 7:]
        self.board_history = []
        self.visual_chain = []
        self.ends = []
        self.winner = None
        self.game_over = False
        self.turn = 0
        self.consecutive_passes = 0
        self.drew_this_turn = {i: False for i in range(self.player_count)}
        self.horizontal_direction = [-1, 1]

        self.current_player = 0
        highest_double = -1
        for i, hand in enumerate(self.hands):
            for tile in hand:
                if tile[0] == tile[1] and tile[0] > highest_double:
                    highest_double = tile[0]
                    self.current_player = i

        self.required_opening_tile = None
        if highest_double != -1:
            self.required_opening_tile = (highest_double, highest_double)

        return self._get_state()

    def valid_actions(self, player=None):
        """Return every legal action for ``player`` without duplicates."""
        if player is None:
            player = self.current_player

        hand = self.hands[player]

        if not self.ends:
            if self.required_opening_tile and self.required_opening_tile in hand:
                return [(self.required_opening_tile, 0)]
            return list({(tile, 0) for tile in hand})

        actions = set()
        left_end, right_end = self.ends
        for tile in hand:
            if left_end in tile:
                actions.add((tile, 0))
            if right_end in tile:
                actions.add((tile, 1))

        if left_end == right_end:
            actions = {(tile, 0) for tile, _ in actions}

        if not actions:
            if self.stock and not self.drew_this_turn[player]:
                return [("DRAW", None)]
            return [None]

        return list(actions)

    def step(self, action):
        """Apply one legal action and return ``(state, game_over, info)``."""
        if self.game_over:
            raise RuntimeError("The game is over. Call reset() before playing again.")

        hand = self.hands[self.current_player]
        available_actions = self.valid_actions(self.current_player)

        if action is None and available_actions != [None]:
            raise ValueError(
                f"Player {self.current_player} cannot pass while another action is available."
            )
        if action is not None and action not in available_actions:
            raise ValueError(f"Invalid action for the current state: {action}")

        advance_player = True

        if action == ("DRAW", None):
            drawn_tile = self.stock.pop(0)
            hand.append(drawn_tile)
            self.board_history.append(action)
            self.drew_this_turn[self.current_player] = True
            advance_player = False

        elif action is not None:
            tile, side = action
            hand.remove(tile)
            self.board_history.append(action)
            self.consecutive_passes = 0
            self.drew_this_turn[self.current_player] = False

            if self.required_opening_tile and tile == self.required_opening_tile:
                self.required_opening_tile = None

            visual_info = {
                "tile": list(tile),
                "played_side": side,
                "turn_index": self.turn,
                "orientation": "vertical" if tile[0] == tile[1] else "horizontal",
                "connected_value": None,
                "exposed_value": None,
            }

            if not self.ends:
                self.ends = [tile[0], tile[1]]
                visual_info["played_side"] = None
                self.visual_chain.append(visual_info)
            else:
                left_end, right_end = self.ends
                if side == 0:
                    connected_value = left_end
                    if tile[0] == left_end:
                        new_end = tile[1]
                    elif tile[1] == left_end:
                        new_end = tile[0]
                    else:
                        raise ValueError(f"Tile {tile} does not connect to left end {left_end}")
                    self.ends[0] = new_end
                    visual_info["connected_value"] = connected_value
                    visual_info["exposed_value"] = new_end
                    self.visual_chain.insert(0, visual_info)

                elif side == 1:
                    connected_value = right_end
                    if tile[0] == right_end:
                        new_end = tile[1]
                    elif tile[1] == right_end:
                        new_end = tile[0]
                    else:
                        raise ValueError(f"Tile {tile} does not connect to right end {right_end}")
                    self.ends[1] = new_end
                    visual_info["connected_value"] = connected_value
                    visual_info["exposed_value"] = new_end
                    self.visual_chain.append(visual_info)

        else:
            self.board_history.append(None)
            self.consecutive_passes += 1
            self.drew_this_turn[self.current_player] = False

        self.turn += 1

        if len(hand) == 0:
            self.game_over = True
            self.winner = self.current_player
        elif self.consecutive_passes >= self.player_count and not self.stock:
            self.game_over = True
            totals = [sum(tile[0] + tile[1] for tile in hand) for hand in self.hands]
            lowest_total = min(totals)
            possible_winners = [i for i, total in enumerate(totals) if total == lowest_total]
            self.winner = possible_winners[0] if len(possible_winners) == 1 else -1

        if not self.game_over and advance_player:
            self.current_player = (self.current_player + 1) % self.player_count

        return self._get_state(), self.game_over, {"winner": self.winner}

    def _get_state(self):
        """Return the compact state consumed by agents and encoders."""
        hand_sizes = [len(hand) for hand in self.hands]
        absences = infer_dead_suits(self.board_history, hand_sizes, self.current_player)

        opponent_dead_suits = set()
        for player, missing_values in absences.items():
            if player != self.current_player:
                opponent_dead_suits |= missing_values

        return {
            "ends": list(self.ends),
            "current_player_hand": [list(tile) for tile in self.hands[self.current_player]],
            "current_player": self.current_player,
            "turn": self.turn,
            "hand_sizes": hand_sizes,
            "board_history": [self._serialize_action(action) for action in self.board_history],
            "visual_chain": copy.deepcopy(self.visual_chain),
            "stock_size": len(self.stock),
            "opponent_dead_suits": sorted(opponent_dead_suits),
        }

    def _serialize_action(self, action):
        if action is None:
            return None
        if action == ("DRAW", None):
            return ["DRAW", None]
        return [list(action[0]), action[1]]

    def to_dict(self):
        """Return a JSON-serializable snapshot of the full engine state."""
        return {
            "player_count": self.player_count,
            "current_player": self.current_player,
            "ends": list(self.ends),
            "logical_board": [self._serialize_action(action) for action in self.board_history],
            "visual_chain": copy.deepcopy(self.visual_chain),
            "hands": [[list(tile) for tile in hand] for hand in self.hands],
            "stock": [list(tile) for tile in self.stock],
            "turn": self.turn,
            "game_over": self.game_over,
            "winner": self.winner,
        }
