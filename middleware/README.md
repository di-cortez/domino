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
programming, so two slots cannot contain the same tile. Tiles with identical
eligible-slot sets are processed as one group: assigning `r` distinguishable
tiles from a group of `k` to `r` labelled slots contributes the exact falling
factorial `(k)_r`. Profiles whose slots all share one domain use `(n)_h`
directly. These exact counts are retained in a bounded, process-local LRU cache
of at most 8,192 profiles; each CPU worker owns its own cache.

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
approximate posterior. Suit, response, and integer total-weight queries are
cached only for the current immutable belief state and every evidence mutation
invalidates the dependent caches. Response probabilities are keyed by the
exact legal-tile mask, not by suit marginals.

Persistent controllers annotate public history once and then extend only the
new append-only suffix. A new game, observer change, shortened history, or
incompatible suffix safely rebuilds the exact state. This changes only history
bookkeeping; evidence order and the one-way slot-to-`mu(H)` transition boundary
remain unchanged.

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

Direct construction keeps trace recording enabled. Built-in consumers that
need only the final seven-vector construct the model with
`record_traces=False`; this skips intermediate snapshots and turn-trace
allocation while preserving the exact final inference and invoking the same
slot-to-`mu(H)` transition at the same completed public-turn boundary. Calling
`update_detailed()` on an explicitly trace-disabled model returns the exact
final result with empty trace collections.

Values written to `state["opponent_suit_probabilities"]` are output only. The
persistent model never trusts that field as an input cache; processed history,
game identity, observer identity, and `MODEL_VERSION` remain the source of
truth.

`probability_can_play(ends)` is computed from the exact joint hand posterior.
`approximate_response_probability_from_marginals()` remains available only as
an explicitly approximate compatibility helper.

The model stays on CPU. Its small irregular dictionaries, branching bitmask
operations, and arbitrary-precision integer weights are not moved to GPU. No
runtime `S_7` suit canonicalization is implemented.

## Action Format

Every action uses one of these shapes:

- `(tile, side)` plays a tile on side `0` (left) or `1` (right);
- `("DRAW", None)` draws from the stock;
- `None` passes when no move or draw is legal.

The same format is consumed by `DominoEngine.step`, `DominoEngine.valid_actions`,
and `DominoEncoder`.

## Headless Step Fast Path

`DominoEngine.step(action)` remains the fully validating public path: it
computes legal actions internally and returns the post-action state in the
stable `(state, game_over, info)` tuple. Controlled automatic loops that have
just called `valid_actions()` may instead pass that unchanged collection as
`legal_actions` and set `return_state=False`. This avoids a second legal-action
scan and a discarded post-action serialization while still checking that the
chosen action belongs to the supplied collection. The returned tuple still has
three items, with `None` in its first position.

The supplied collection is trusted internal data for that exact engine,
current player, and position. It must be computed immediately before `step`,
must not be modified by an agent, and must never come from a UI, network, or
client payload. Human actions continue to use engine-side legal-action
generation.

`python benchmarks/headless_step_benchmark.py` compares the old repeated-work
turn structure with this fast path using fixed seeds. The report includes
games/second and exact counts for legal-action scans, state snapshots, and
serialized history actions, and rejects any result-fingerprint difference.

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
