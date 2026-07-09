"""Rule-based teacher agent used for supervised data generation."""

import random

from agents.agent import Agent
from middleware.domino_engine import infer_dead_suits


class StrategicAgent(Agent):
    """
    Score every legal tile move with a handcrafted utility function.

    The utility rewards pip disposal, early double disposal, remaining hand
    flexibility, coverage of the newly exposed end, and blocking values that an
    opponent has already shown they likely cannot play. The agent is
    deterministic by default so supervised-learning labels are stable.
    """

    def __init__(
        self,
        double_weight=3.5,
        diversity_weight=3.0,
        coverage_weight=2.0,
        blocking_weight=8.0,
        tie_break_noise=0.0,
    ):
        self.double_weight = double_weight
        self.diversity_weight = diversity_weight
        self.coverage_weight = coverage_weight
        self.blocking_weight = blocking_weight
        self.tie_break_noise = tie_break_noise

    def _infer_opponent_absences(self, state):
        """Delegate dead-suit inference to the same helper used by the engine."""
        return infer_dead_suits(
            state.get("board_history", []),
            state["hand_sizes"],
            state["current_player"],
        )

    def choose_move(self, state, legal_actions):
        if not legal_actions:
            return None

        tile_moves = [move for move in legal_actions if move is not None and move[0] != "DRAW"]
        if not tile_moves:
            return legal_actions[0]

        hand = [tuple(tile) for tile in state["current_player_hand"]]
        current_ends = state["ends"]
        current_player = state["current_player"]
        hand_sizes = state["hand_sizes"]

        opponents = [size for i, size in enumerate(hand_sizes) if i != current_player]
        urgency_factor = 2.0 if opponents and min(opponents) <= 2 else 1.0

        absences = self._infer_opponent_absences(state)
        dead_suit_values = set()
        for player, missing_values in absences.items():
            if player != current_player:
                dead_suit_values |= missing_values

        evaluations = []
        for move in tile_moves:
            tile, side = move
            tile = tuple(tile)

            remaining_hand = list(hand)
            remaining_hand.remove(tile)

            if tile[0] == tile[1]:
                pip_weight = 0
                double_bonus = tile[0] * self.double_weight
            else:
                pip_weight = tile[0] + tile[1]
                double_bonus = 0

            diversity = 0
            end_coverage = 0
            blocking = 0

            if current_ends:
                connected_value = current_ends[side]
                if tile[0] == connected_value:
                    new_end = tile[1]
                elif tile[1] == connected_value:
                    new_end = tile[0]
                else:
                    new_end = None

                if new_end is not None:
                    ends_after_move = list(current_ends)
                    ends_after_move[side] = new_end

                    diversity = sum(
                        1 for candidate in remaining_hand
                        if candidate[0] in ends_after_move or candidate[1] in ends_after_move
                    ) * self.diversity_weight
                    end_coverage = sum(
                        1 for candidate in remaining_hand if new_end in candidate
                    ) * self.coverage_weight
                    blocking = sum(
                        1 for value in ends_after_move if value in dead_suit_values
                    ) * self.blocking_weight
            else:
                remaining_numbers = set()
                for candidate in remaining_hand:
                    remaining_numbers.add(candidate[0])
                    remaining_numbers.add(candidate[1])
                diversity = len(remaining_numbers) * 2

            total_utility = (
                (pip_weight + double_bonus + blocking) * urgency_factor
                + diversity
                + end_coverage
            )
            evaluations.append((total_utility, move))

        best_utility = max(utility for utility, _ in evaluations)

        if self.tie_break_noise > 0:
            near_ties = [
                move for utility, move in evaluations
                if utility >= best_utility - self.tie_break_noise
            ]
            return random.choice(near_ties)

        for utility, move in evaluations:
            if utility == best_utility:
                return move

        return tile_moves[0]
