"""Rule-based teacher agent used for supervised data generation."""

from agents.agent import Agent
from middleware.opponent_model import ExactOpponentModel


class StrategicAgent(Agent):
    """Deterministic heuristic based on opponent suit-presence probabilities.

    The agent uses lexicographic filters instead of a weighted utility sum:

    1. prefer moves that minimize the opponent's exact response chance;
    2. among near ties, prefer moves with near-best normalized mobility;
    3. among remaining ties, prefer the highest pip sum;
    4. preserve the stable ``legal_actions`` order for exact ties.
    """

    def __init__(self, response_tolerance=0.10, mobility_tolerance=0.10):
        self.response_tolerance = float(response_tolerance)
        self.mobility_tolerance = float(mobility_tolerance)
        self.opponent_model = ExactOpponentModel(record_traces=False)

    def choose_move(self, state, legal_actions):
        if not legal_actions:
            return None

        tile_moves = [move for move in legal_actions if move is not None and move[0] != "DRAW"]
        if not tile_moves:
            return legal_actions[0]
        if len(tile_moves) == 1:
            return tile_moves[0]

        probabilities = self.opponent_model.update(state)
        state["opponent_suit_probabilities"] = probabilities

        hand = [tuple(tile) for tile in state["current_player_hand"]]
        current_ends = state.get("ends", [])

        evaluations = []
        for order, move in enumerate(tile_moves):
            tile, _side = move
            tile = tuple(tile)

            remaining_hand = list(hand)
            remaining_hand.remove(tile)

            if not remaining_hand:
                return move

            ends_after_move = self._ends_after_move(current_ends, move)
            response_probability = self.opponent_model.probability_can_play(
                ends_after_move
            )
            mobility = self._normalized_mobility(remaining_hand, ends_after_move)
            pip_sum = tile[0] + tile[1]

            evaluations.append({
                "order": order,
                "move": move,
                "response_probability": response_probability,
                "mobility": mobility,
                "pip_sum": pip_sum,
            })

        lowest_response = min(item["response_probability"] for item in evaluations)
        epsilon = 1e-9
        blocking_candidates = [
            item for item in evaluations
            if item["response_probability"] <= lowest_response + self.response_tolerance + epsilon
        ]

        best_mobility = max(item["mobility"] for item in blocking_candidates)
        mobility_candidates = [
            item for item in blocking_candidates
            if item["mobility"] >= best_mobility - self.mobility_tolerance - epsilon
        ]

        best_pip_sum = max(item["pip_sum"] for item in mobility_candidates)
        pip_candidates = [
            item for item in mobility_candidates
            if item["pip_sum"] == best_pip_sum
        ]

        return min(pip_candidates, key=lambda item: item["order"])["move"]

    def _ends_after_move(self, current_ends, move):
        """Return the board ends after applying a tile-play move."""
        tile, side = move
        tile = tuple(tile)

        if not current_ends:
            return (tile[0], tile[1])

        ends = list(current_ends)
        connected_value = ends[side]

        if tile[0] == connected_value:
            ends[side] = tile[1]
        elif tile[1] == connected_value:
            ends[side] = tile[0]
        else:
            raise ValueError(f"Move {move!r} does not connect to ends {current_ends!r}.")

        return tuple(ends)

    def _normalized_mobility(self, remaining_hand, ends_after_move):
        """Return the fraction of remaining tiles playable on the new ends."""
        if not remaining_hand:
            return 1.0

        playable_remaining_tiles = sum(
            1
            for tile in remaining_hand
            if ends_after_move[0] in tile or ends_after_move[1] in tile
        )
        return playable_remaining_tiles / len(remaining_hand)
