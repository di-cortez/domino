# Middleware

The middleware layer owns rules and game flow. It does not decide strategy; that
belongs to `agents/`.

| File | Purpose |
|---|---|
| `domino_engine.py` | Stateful two-player domino engine: deal, legal actions, draw/pass, game-over rules, serialized state. |
| `middleware.py` | `Agent` protocol and `GameManager`, which asks agents for moves and records supervised-training history. |
| `opponent_model.py` | Exact two-player opponent inference using temporal slots and integer hand weights. |

## Opponent Suit Probabilities

`opponent_model.py` exports `compute_opponent_suit_probabilities(state)`, which
returns seven suit-presence probabilities from the acting player's perspective.
It replays the public action history without looking at the real opponent hand.
The meaning is direct:

- `0.0`: the observer knows the opponent does not currently hold that suit;
- `1.0`: the observer knows the opponent currently holds that suit.

The strict temporal model requires `current_player_initial_hand`,
`current_player_drawn_tiles`, and exactly two public hand sizes. Missing private
observer data raises an error; there is no snapshot fallback.

## Exact Belief Architecture

The unknown pool `U` contains exactly the opponent hand plus the stock. The
public opponent hand size is `h`.

`SlotOpponentBelief` is the initial representation. Every temporal hand slot
has its own allowed-tile mask, so a tile drawn later does not inherit negative
evidence observed before that draw. Canonical slot profiles carry positive
integer history weights. Injective assignment counts are computed with dynamic
programming, so two slots cannot contain the same tile.

`MuOpponentBelief` stores the exact posterior as `hand_mask -> mu(H)`, where
every `mu(H)` is a positive integer. It never normalizes or truncates these
weights during updates. Probabilities divide weighted integer totals only when
queried.

`HybridExactOpponentModel` always starts in `slots_exact`. At the end of the
first non-terminal public turn where `comb(|U|, h) <= 500`, it converts the
profiles to `mu(H)` with incremental hand-mask DP. Equal partial hands are
merged after every slot. The model then stays in `mu_exact` until the game ends,
even if a later draw creates more than 500 hands. `ExactOpponentModel` and
`HybridOpponentModel` are stable aliases for this exact controller.

The standard path never uses particles and never silently substitutes an
approximate posterior.

## Draw-Turn Traces

`update(state)` preserves the old API and returns only seven floats.
`update_detailed(state)` additionally returns labelled snapshots and completed
public-turn traces. For an opponent `DRAW -> PASS` turn, the stages are:

1. `after_negative_evidence`: the old hand has no tile for either board end;
2. `after_draw`: the new slot can make those suit probabilities positive again;
3. `end_turn`: the explicit pass conditions the entire new hand, restoring zero
   probability on both ends.

`DRAW -> PLAY` exposes the same three stage names. Repeating `update_detailed`
with unchanged history is idempotent. `consume_new_snapshots()` lets a UI or
logger consume only snapshots it has not seen before.

Values written to `state["opponent_suit_probabilities"]` are output only. The
persistent model never trusts that field as an input cache; processed history,
game identity, observer identity, and `MODEL_VERSION` remain the source of
truth.

`probability_can_play(ends)` is computed from the exact joint hand posterior.
`approximate_response_probability_from_marginals()` remains available only as
an explicitly approximate compatibility helper.

## Action Format

Every action uses one of these shapes:

- `(tile, side)` plays a tile on side `0` (left) or `1` (right);
- `("DRAW", None)` draws from the stock;
- `None` passes when no move or draw is legal.

The same format is consumed by `DominoEngine.step`, `DominoEngine.valid_actions`,
and `DominoEncoder`.

## Game Termination

A game ends one of two ways: a player empties their hand (that player wins),
or the game is blocked -- every player has passed consecutively
(`consecutive_passes >= player_count`) with an empty stock, decided by
comparing each hand's remaining pip total (lowest wins; a tie is a draw,
`winner == -1`).

The blocked-game check only fires when the action that just completed was a
`PASS` (`action is None`). A `DRAW` never triggers it, even if it empties the
stock and `consecutive_passes` already meets the threshold from earlier
passes: the current player keeps the turn after drawing and must still
receive the resulting `valid_actions()` (play the drawn tile if it connects,
otherwise pass) before the engine evaluates whether the game is blocked. See
`domino_final_stock_draw_bug_report.txt` for the reproduction cases this
guards against and `tests/test_core.py`'s
`test_engine_final_stock_draw_unplayable_tile_requires_pass_before_blocked_game`
/ `test_engine_final_stock_draw_playable_tile_can_be_played_immediately`.
