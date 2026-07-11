# Middleware

The middleware layer owns rules and game flow. It does not decide strategy; that
belongs to `agents/`.

| File | Purpose |
|---|---|
| `domino_engine.py` | Stateful two-player domino engine: deal, legal actions, draw/pass, game-over rules, serialized state. |
| `middleware.py` | `Agent` protocol and `GameManager`, which asks agents for moves and records supervised-training history. |
| `opponent_model.py` | Exact two-player opponent-suit belief model and fallback probability helpers. |

## Opponent Suit Probabilities

`opponent_model.py` exports `compute_opponent_suit_probabilities(state)`, which
returns seven suit-presence probabilities from the acting player's perspective.
For two-player states with `current_player_initial_hand` and
`current_player_drawn_tiles`, it replays the public action history without
looking at the real opponent hand. The meaning is direct:

- `0.0`: the observer knows the opponent does not currently hold that suit;
- `1.0`: the observer knows the opponent currently holds that suit.

States without the private observer fields fall back to a current-snapshot
hypergeometric estimate. That fallback is useful for old datasets and smoke
tests, but it is not treated as exact temporal inference.

## Action Format

Every action uses one of these shapes:

- `(tile, side)` plays a tile on side `0` (left) or `1` (right);
- `("DRAW", None)` draws from the stock;
- `None` passes when no move or draw is legal.

The same format is consumed by `DominoEngine.step`, `DominoEngine.valid_actions`,
and `DominoEncoder`.
