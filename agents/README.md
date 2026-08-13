# Agents

All playable agents expose the same `choose_move(state, legal_actions)` shape so
`GameManager` can run any pair without knowing how each decision is made.

| File | Purpose |
|---|---|
| `agent.py` | Uniform-random `RandomAgent` baseline. |
| `encoder.py` | Single source of truth for state-to-vector and tile-play action encoding. |
| `heuristic_agent.py` | `StrategicAgent`, the exact-probability rule-based teacher used for supervised labels and benchmarks. |
| `network_architecture.py` | Hidden-layer depth/width resolution and the serializable policy dimensions. |
| `nn.py` | Per-network NumPy/CuPy float32 MLP backend with explicit `auto`/`cpu`/`gpu` selection. |
| `neural_agent.py` | Loads `models/domino_sl_weights.npz` and plays the supervised policy. |
| `rl_nn.py` | Masked PPO/REINFORCE network with entropy regularization and an optional legacy value head. |
| `rl_agent.py` | Wraps `PolicyNetwork` for training trajectories or deterministic evaluation play. |

The opponent belief model lives in `middleware/opponent_model.py` because it is
shared by agents, training, diagnostics, and the UI.

Each supervised network owns an exact `network.device` and `network.xp` instead
of relying on one module-wide array backend. Inference accepts NumPy or CuPy
inputs and converts them to that network's backend in `float32`. Training may
use host RAM, disk-backed arrays, full GPU residency, or a reusable rotating GPU
window; these storage policies live in `training/supervised/runtime.py`.
Both networks accept the same optional `weight_decay`, which applies only to
the weight matrices (`W1` through the output layer, and the RL critic `Wv`); bias vectors
are never regularized. The supervised network folds its decay into the
gradient, while `PolicyNetwork` applies a decoupled shrink after gradient
clipping so reported RL gradient norms keep describing the policy and value
gradient alone. Both regularizers default to disabled
(`DISABLED_WEIGHT_DECAY`, `DISABLED_DROPOUT_RATE`).

Optional dropout is likewise a property of the shared architecture, so both
`SupervisedNeuralNetwork` and `PolicyNetwork` accept `dropout_rate` and apply
inverted dropout to every hidden activation. It is active only in a training
forward pass (`forward(x, training=True)`); the default `training=False` keeps
gameplay, evaluation, opponent-pool snapshots, and whole-buffer PPO metrics on
the complete network. The forward pass caches one scale mask per hidden layer as `D1..D{H}`
only while dropout is active, and backpropagation reuses exactly those masks.
`PolicyNetwork` draws its masks from the host NumPy generator because that
state, unlike CuPy's, is part of the exact RL resume pair.

`GPU_ENABLED` becomes true only when CuPy imports, CUDA reports a visible
device, and a synchronized float32 allocation succeeds. `GPU_UNAVAILABLE_REASON` records why the probe failed,
allowing pipeline and standalone logs to explain a NumPy/CPU fallback instead
of claiming that an importable but unusable CuPy installation is active. See
the root README for the complete Linux driver, CuPy `[ctk]`, verification, and
troubleshooting procedure.

The hidden stack is configurable. `hidden_sizes` selects any number of hidden
layers from one upwards, of any width; `agents/network_architecture.py` owns
the defaults (256 then 128, with 128 for any deeper layer) and the CLI
resolution used by `training/supervised/training_loop.py`. Only the command line is
bounded, at `MAX_HIDDEN_LAYER_COUNT` (8), because one `--hidden<n>-size` option
has to exist per layer. Weights are named `W1..W{L}`/`b1..b{L}` for `L`
layers including the output layer, so a two-layer network keeps exactly the
historical `W1, b1, W2, b2, W3, b3` checkpoint keys and every existing
checkpoint loads unchanged. Loaders read the depth and widths back out of the
archive, so `NeuralAgent`, `PolicyNetwork`, the rollout workers, the opponent
pool, resume state, and diagnostics all follow the checkpoint instead of a
configured shape.

`rl_nn.py::PolicyNetwork` uses the same per-network resolver via a `device`
parameter (`"auto"` follows usable CuPy; `"cpu"`/`"gpu"` are explicit), so an RL run
can be pinned to CPU while supervised training elsewhere in the same process
still uses the GPU, or vice versa.

`NeuralAgent.load(..., device=...)` preserves that backend choice. CPU-only
workers set `DOMINO_FORCE_CPU=1`, so they never initialize a CUDA context.

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
final tie-breaker. `StrategicAgent`, `NeuralAgent`, and `RLAgent` use persistent
exact models with intermediate trace recording disabled
because they consume only the current seven-vector. Direct opponent-model callers
still receive traces by default.

## Action Encoding

The neural output space now has 56 actions:

- 28 tile actions on the left end;
- 28 tile actions on the right end.

Draw, pass, and single-option tile plays are forced by the current rules
engine. `NeuralAgent` and `RLAgent` return them directly
without calling the network. `StrategicAgent` also returns a single tile-play
option before running exact inference. These are not learned RL decisions, and
`RLAgent` does not save a trajectory step for them.

`RLAgent` has three explicit policy modes:

| Mode | Legal policy choice | Stores trajectory |
|---|---|---|
| `training` | Samples from the masked distribution | Yes |
| `stochastic_evaluation` | Samples from the masked distribution | No |
| `evaluation` | Selects the largest masked probability | No |

Self-play pool opponents use stochastic evaluation. UI play and diagnostics
use deterministic evaluation.

RL trajectory steps store the encoded state, sampled action index, legal-action
mask, collection-time masked `old_log_prob`, decision turn, and local reward
accumulator. During self-play, draw/pass events are distributed to earlier real
decisions with temporal decay. PPO reuses exactly that mask and log-probability;
illegal actions receive zero probability and no direct policy or entropy
gradient. A decision's reward is not weighted by its number of legal choices.

`PolicyNetwork` is policy-only by default and self-play updates it with PPO.
Optional value-head training is retained only for `--ppo-max-epochs 1`
regression runs; it adds `Wv`/`bv`, reading the last hidden activation, next to
the policy arrays.
Diagnostics detect those two arrays, load the optional head without changing
the policy decision, and obtain `V(s)` from the hidden activation already
computed for each real decision.

Because the input/output shapes changed from the old 86/58 encoder to the new
168/56 encoder, old `domino_sl_weights.npz` and `domino_rl_weights.npz`
checkpoints are not compatible. Regenerate the supervised dataset, retrain SL,
and then retrain RL.

Weights trained with the older absence-confidence feature also load by shape,
but they are semantically stale. Archive them and retrain after regenerating the
dataset with the current opponent-suit probability feature.
