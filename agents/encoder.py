"""Ruleset-aware state and action encoder for neural domino agents.

The neural policy only chooses voluntary tile plays. Forced draw/pass actions are
handled by the agent wrapper and are deliberately absent from the policy output.
"""

from dataclasses import dataclass

import numpy as np

from middleware.opponent_model import (
    compute_opponent_suit_probabilities,
    reconstruct_public_actions,
)
from middleware.rulesets import (
    DEFAULT_RULESET_NAME,
    DominoRuleset,
    resolve_ruleset,
    validate_state_ruleset,
)


MAX_TURN = 52
DEFAULT_VECTOR_SIZE = 168
DEFAULT_ACTION_SIZE = 56

# One feature per logical RL opponent bucket, in ``training.rl.pool``'s registry
# order. The width is fixed at every bucket the registry defines rather than at
# the buckets one run selects, so ``--opponent-buckets`` never changes the
# policy's input size and a checkpoint stays loadable under a different
# selection. ``tests/test_opponent_bucket_features.py`` fails if the two orders
# ever drift apart; the vocabulary lives here because the encoder owns its own
# layout and ``agents`` must not import ``training``.
OPPONENT_BUCKET_FEATURE_ORDER = (
    "heuristic",
    "random",
    "recent",
    "medium_term",
    "historical_uniform",
    "champion_vs_heuristic",
    "champion_vs_learner",
)
OPPONENT_BUCKET_FEATURE_WIDTH = len(OPPONENT_BUCKET_FEATURE_ORDER)
OPPONENT_BUCKET_FEATURE_INDEX = {
    name: index for index, name in enumerate(OPPONENT_BUCKET_FEATURE_ORDER)
}


def opponent_bucket_feature_index(bucket):
    """Return the one-hot position of one opponent bucket, or ``None``.

    ``None`` is the deliberate no-bucket state: the seat's adversary belongs to
    no bucket at all. It leaves the block at zero instead of inventing a
    membership the caller cannot justify. Note that zero is not a neutral input
    to a policy that trained on a one-hot block, only an honest one, which is
    why the UI hands a human the ``heuristic`` slot instead
    (``ui.ui_agents.UI_OPPONENT_BUCKET``).
    """
    if bucket is None:
        return None
    try:
        return OPPONENT_BUCKET_FEATURE_INDEX[str(bucket)]
    except KeyError:
        raise ValueError(
            f"Unknown opponent bucket {bucket!r}; choose one of "
            + ", ".join(OPPONENT_BUCKET_FEATURE_ORDER)
            + " or None for a seat whose adversary has no bucket."
        ) from None


@dataclass(frozen=True)
class EncoderLayout:
    """Derived feature offsets for one compact domino ruleset."""

    hand: int
    played: int
    played_turn: int
    played_by_me: int
    played_by_opponent: int
    left_end: int
    right_end: int
    hand_size: int
    stock_size: int
    draw_count: int
    pass_count: int
    opponent_suit_probability: int
    opponent_bucket: int
    vector_size: int

    @classmethod
    def for_ruleset(
        cls,
        ruleset: DominoRuleset,
        *,
        use_opponent_suit_features: bool = True,
        use_opponent_bucket_features: bool = False,
    ) -> "EncoderLayout":
        """Return the offsets for one ruleset and its two optional blocks.

        Both optional blocks are trailing, so dropping either shortens the
        vector without moving any offset that precedes it. When a block is
        absent its offset equals the offset that would follow it, marking an
        empty span instead of forcing every consumer to handle ``None``. The
        opponent-bucket block sits last precisely so that a run that leaves it
        off keeps the historical layout byte for byte.
        """
        tile_count = ruleset.tile_count
        suit_count = ruleset.pip_count
        suit_blocks = 3 if use_opponent_suit_features else 2
        bucket_width = (
            OPPONENT_BUCKET_FEATURE_WIDTH if use_opponent_bucket_features else 0
        )
        opponent_bucket = 5 * tile_count + suit_blocks * suit_count + 7
        return cls(
            hand=0,
            played=tile_count,
            played_turn=2 * tile_count,
            played_by_me=3 * tile_count,
            played_by_opponent=4 * tile_count,
            left_end=5 * tile_count,
            right_end=5 * tile_count + suit_count,
            hand_size=5 * tile_count + 2 * suit_count,
            stock_size=5 * tile_count + 2 * suit_count + 2,
            draw_count=5 * tile_count + 2 * suit_count + 3,
            pass_count=5 * tile_count + 2 * suit_count + 5,
            opponent_suit_probability=5 * tile_count + 2 * suit_count + 7,
            opponent_bucket=opponent_bucket,
            vector_size=opponent_bucket + bucket_width,
        )


class DominoEncoder:
    """Map states to compact ruleset-sized features and policy actions.

    For ``T`` tiles and ``S`` pip values the feature size is ``5T + 3S + 7``.
    Its blocks are, in order: tiles in hand, tiles played, normalized play turn,
    tiles played by each side, two ``S``-wide board-end one-hots, two hand
    sizes, stock size, two draw counts, two pass counts, and ``S`` exact-model
    pip-presence probabilities. The action space is ``2T``: every tile on the
    left followed by every tile on the right. Forced draw/pass actions are not
    neural-policy actions.

    The historical double-six layout is therefore 168 features and 56 actions;
    its class-level size/offset aliases remain available for legacy callers.
    Generic code must use the ruleset-local instance attributes.

    ``use_opponent_suit_features=False`` drops the trailing exact-model block,
    giving ``5T + 2S + 7`` features and leaving the action space untouched. The
    exact opponent model is then never consulted during encoding, even when a
    caller already stored its output in the state. Double-six shrinks to 161.

    ``use_opponent_bucket_features=True`` appends one further block of
    ``OPPONENT_BUCKET_FEATURE_WIDTH`` features holding a one-hot of
    ``opponent_bucket``: which logical RL opponent bucket the seat being encoded
    is playing against. Double-six grows to 175. The block is last, so leaving
    it off reproduces the historical layout exactly, and its width is the whole
    bucket registry rather than the buckets one run selected, so the input size
    never depends on ``--opponent-buckets``.

    ``opponent_bucket`` describes the adversary of the seat this encoder serves,
    not the match as a whole: in a game between the learner and a frozen
    ``medium_term`` snapshot the learner's encoder is given ``medium_term`` and
    the snapshot's is given ``recent``, because the snapshot faces the current
    learner. ``None`` leaves the block at zero and means the adversary belongs
    to no bucket, which is the human seat in the UI.
    """

    # Legacy double-six aliases. Generic code must use the instance attributes
    # ``vector_size`` and ``action_size`` instead.
    VECTOR_SIZE = DEFAULT_VECTOR_SIZE
    ACTION_SIZE = DEFAULT_ACTION_SIZE
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
    OPPONENT_BUCKET_OFFSET = 168

    def __init__(
        self,
        ruleset=DEFAULT_RULESET_NAME,
        *,
        use_opponent_suit_features=True,
        use_opponent_bucket_features=False,
        opponent_bucket=None,
    ):
        self.ruleset = resolve_ruleset(ruleset)
        self.use_opponent_suit_features = bool(use_opponent_suit_features)
        self.use_opponent_bucket_features = bool(use_opponent_bucket_features)
        # Validated here rather than at encoding time so a misspelled bucket
        # fails once, on the agent that owns it, instead of once per decision.
        self.opponent_bucket_index = opponent_bucket_feature_index(
            opponent_bucket
        )
        self.opponent_bucket = opponent_bucket
        self.layout = EncoderLayout.for_ruleset(
            self.ruleset,
            use_opponent_suit_features=self.use_opponent_suit_features,
            use_opponent_bucket_features=self.use_opponent_bucket_features,
        )
        self.all_tiles = list(self.ruleset.all_tiles)
        self.tile_to_index = {
            tile: index for index, tile in enumerate(self.all_tiles)
        }
        self.vector_size = self.layout.vector_size
        self.action_size = 2 * self.ruleset.tile_count

        # Instance offsets keep existing consumers readable while making their
        # values ruleset-local.
        self.HAND_OFFSET = self.layout.hand
        self.PLAYED_OFFSET = self.layout.played
        self.PLAYED_TURN_OFFSET = self.layout.played_turn
        self.PLAYED_BY_ME_OFFSET = self.layout.played_by_me
        self.PLAYED_BY_OPPONENT_OFFSET = self.layout.played_by_opponent
        self.LEFT_END_OFFSET = self.layout.left_end
        self.RIGHT_END_OFFSET = self.layout.right_end
        self.HAND_SIZE_OFFSET = self.layout.hand_size
        self.STOCK_SIZE_OFFSET = self.layout.stock_size
        self.DRAW_COUNT_OFFSET = self.layout.draw_count
        self.PASS_COUNT_OFFSET = self.layout.pass_count
        self.OPPONENT_SUIT_PROBABILITY_OFFSET = (
            self.layout.opponent_suit_probability
        )
        self.OPPONENT_BUCKET_OFFSET = self.layout.opponent_bucket

        self.all_actions = []
        for tile in self.all_tiles:
            self.all_actions.append((tile, 0))
        for tile in self.all_tiles:
            self.all_actions.append((tile, 1))

        self.action_to_index = {action: idx for idx, action in enumerate(self.all_actions)}

    def encode_state(self, state):
        """Convert a compatible state into a ruleset-sized feature vector."""
        validate_state_ruleset(state, self.ruleset)
        # float32 matches the network weights and the supervised dataset, so no
        # consumer has to re-cast and no matrix is promoted during inference.
        vector = np.zeros((self.vector_size, 1), dtype=np.float32)
        current_player = state.get("current_player", 0)

        for tile in state.get("current_player_hand", []):
            tile = tuple(tile)
            vector[self.HAND_OFFSET + self.tile_to_index[tile], 0] = 1.0

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
            tile_index = self.tile_to_index[tuple(tile)]
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
            vector[self.HAND_SIZE_OFFSET + i, 0] = (
                hand_sizes[i] / self.ruleset.hand_size
            )

        initial_stock_size = self.ruleset.initial_stock_size(player_count=2)
        vector[self.STOCK_SIZE_OFFSET, 0] = (
            state.get("stock_size", 0) / initial_stock_size
        )

        for i in range(2):
            vector[self.DRAW_COUNT_OFFSET + i, 0] = (
                draw_counts[i] / initial_stock_size
            )
            vector[self.PASS_COUNT_OFFSET + i, 0] = pass_counts[i] / self.MAX_TURN

        # Ablated layouts stop here: the vector has no trailing suit block, and
        # a value already stored in the state is ignored rather than read, so
        # the exact model never influences the features it is meant to be
        # measured against.
        if self.use_opponent_suit_features:
            # Persistent agents place the exact result in the state immediately
            # before encoding. One-shot callers still reconstruct it from history.
            probabilities = state.get("opponent_suit_probabilities")
            if probabilities is None:
                probabilities = compute_opponent_suit_probabilities(state)
            if len(probabilities) != self.ruleset.pip_count:
                raise ValueError(
                    "opponent_suit_probabilities has "
                    f"{len(probabilities)} values, expected "
                    f"{self.ruleset.pip_count} for {self.ruleset.name}."
                )
            for suit, value in enumerate(probabilities):
                vector[self.OPPONENT_SUIT_PROBABILITY_OFFSET + suit, 0] = value

        # Constant for the whole game, so it comes from the encoder rather than
        # from the per-turn state: the seat's adversary cannot change mid-game,
        # and threading it through every state dict would let one stale turn
        # silently relabel the matchup. A ``None`` bucket leaves the block at
        # zero, which is the encoded form of "this adversary has no bucket".
        if (
            self.use_opponent_bucket_features
            and self.opponent_bucket_index is not None
        ):
            vector[
                self.OPPONENT_BUCKET_OFFSET + self.opponent_bucket_index, 0
            ] = 1.0

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
        """Return an ``(action_size, 1)`` legal neural-policy mask."""
        mask = np.zeros((self.action_size, 1), dtype=np.float32)

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

        masked_scores = np.full(self.action_size, -np.inf)
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
