# Agents

All playable agents expose the same `choose_move(state, legal_actions)` shape so
`GameManager` can run any pair without knowing how each decision is made.

| File | Purpose |
|---|---|
| `agent.py` | Baseline `RandomAgent` and `GreedyAgent`, plus a small runnable demo. |
| `encoder.py` | Single source of truth for state-to-vector and tile-play action encoding. |
| `heuristic_agent.py` | `StrategicAgent`, the exact-probability rule-based teacher used for supervised labels and benchmarks. |
| `nn.py` | Supervised MLP backend. Uses CuPy automatically when installed, otherwise NumPy. |
| `neural_agent.py` | Loads `models/domino_sl_weights.npz` and plays the supervised policy. |
| `rl_nn.py` | Policy-only network with masked REINFORCE gradients and entropy regularization. |
| `rl_agent.py` | Wraps `PolicyNetwork` for training trajectories or deterministic evaluation play. |

The opponent belief model lives in `middleware/opponent_model.py` because it is
shared by agents, training, diagnostics, and the UI.

## State Encoding

`DominoEncoder` produces a 168-dimensional input vector:

| Slice | Meaning |
|---|---|
| `my_hand[28]` | Tiles currently held by the acting player. |
| `played[28]` | Tiles already played on the board. |
| `played_turn[28]` | Normalized turn when each tile was played, using `MAX_TURN = 52`; zero means unplayed. |
| `played_by_me[28]` | Tiles played by the acting player. |
| `played_by_opponent[28]` | Tiles played by the opponent. |
| `left_end[7]` | One-hot encoding of the current left end. |
| `right_end[7]` | One-hot encoding of the current right end. |
| `hand_sizes[2]` | Player hand sizes divided by 7. |
| `stock_size[1]` | Stock size divided by 14. |
| `draw_count_by_player[2]` | Draw counts for players 0 and 1 divided by 14. |
| `pass_count_by_player[2]` | Pass counts for players 0 and 1 divided by `MAX_TURN`. |
| `opponent_suit_probabilities[7]` | Probability that the opponent currently holds at least one tile of each suit/value. |

The opponent probability feature is bounded in `[0, 1]`: `0.0` means the
opponent is known not to hold that suit, and `1.0` means the opponent is known
to hold it. For two-player games, the model replays public history with the
observer's private initial hand and draw history. States without those private
observer fields are rejected because exact temporal reconstruction is not
possible.

The shared exact model starts with temporal slot/cohort profiles and switches
once to integer `mu(H)` hand weights when `comb(|U|, h) <= 500`. It never uses a
particle fallback. `StrategicAgent` filters moves by the exact joint probability
that the opponent can answer the resulting ends, then by near-best normalized
mobility, then by highest pip sum, with deterministic legal-action order as the
final tie-breaker.

## Action Encoding

The neural output space now has 56 actions:

- 28 tile actions on the left end;
- 28 tile actions on the right end.

Draw, pass, and single-option tile plays are forced by the current rules
engine. They are not learned RL decisions. If a turn has fewer than two legal
tile-play actions, `RLAgent` returns the forced action directly without calling
the network or saving a trajectory step.

RL trajectory steps store the encoded state, sampled action index, legal-action
mask, decision turn, option count, and local reward accumulator. During
self-play, draw/pass events are distributed to earlier real decisions with
temporal decay. The policy-gradient backward pass uses the saved mask to
renormalize the softmax over legal actions only, so illegal actions receive no
direct policy or entropy gradient.

`PolicyNetwork` is policy-only. RL checkpoints contain only `W1`, `b1`, `W2`,
`b2`, `W3`, and `b3`; extra arrays from older checkpoints are ignored when
loading.

Because the input/output shapes changed from the old 86/58 encoder to the new
168/56 encoder, old `domino_sl_weights.npz` and `domino_rl_weights.npz`
checkpoints are not compatible. Regenerate the supervised dataset, retrain SL,
and then retrain RL.

Weights trained with the older absence-confidence feature also load by shape,
but they are semantically stale. Archive them and retrain after regenerating the
dataset with the current opponent-suit probability feature.
