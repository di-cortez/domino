# Middleware

The middleware layer owns rules and game flow. It does not decide strategy; that
belongs to `agents/`.

| File | Purpose |
|---|---|
| `domino_engine.py` | Stateful two-player domino engine: deal, legal actions, draw/pass, game-over rules, serialized state. |
| `middleware.py` | `Agent` protocol and `GameManager`, which asks agents for moves and records supervised-training history. |

## Action Format

Every action uses one of these shapes:

- `(tile, side)` plays a tile on side `0` (left) or `1` (right);
- `("DRAW", None)` draws from the stock;
- `None` passes when no move or draw is legal.

The same format is consumed by `DominoEngine.step`, `DominoEngine.valid_actions`,
and `DominoEncoder`.
