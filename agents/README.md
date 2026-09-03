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
the compact first-two-layer defaults (256x128, 192x96, 128x64, or 96x48 by
ruleset, with 128 for any deeper layer) and the CLI
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

`PolicyNetwork.snapshot_parameters()` and `restore_parameters()` are inverses
that copy and write back every trainable array by name, whatever the depth and
however the critic is wired. `training/rl/ppo.py` uses them to undo a whole
diverged epoch; nothing else should need them, because a checkpoint is the
right unit for any longer-lived copy.

`_apply_gradient_step` rejects a non-finite gradient norm before writing
anything. Norm clipping cannot repair such a gradient -- `nan > clip` is False,
so a NaN bypasses clipping entirely, and `inf * (clip / inf)` is NaN -- so both
branches would otherwise poison every weight permanently and silently. The
returned metrics carry `grad_rejected`, and `optimizer_steps` is `0` for a
rejected step, so callers can tell a skipped minibatch from an applied one.

`_masked_rollout_probabilities` builds the sampling distribution over one
decision's legal actions, and which of its two paths it takes is decided by
whether the network published `cache[logits_key]`. With the logits it rebuilds
the softmax over the legal subset alone, where the maximizing legal action
contributes exactly `exp(0) = 1` and the total is therefore always at least
1.0. Without them it can only renormalize the full-support softmax, which
flushes a legal action to zero once that action sits more than `-log(tiny)`
(about 87.3 nats) below the *global* maximum -- a maximum the illegal actions
are free to hold.

Every network that reaches this must publish the cache, `PolicyNetwork` and
`training/rl/parallel.py`'s `_CPUInferencePolicy` alike. The second one is what
actually runs inside a rollout worker; while it published no cache, every
worker decision took the fallback silently, and the run
`d6_maxwr_lr032` died after 28,000,000 games on one decision whose two legal
actions had both underflowed.

`NonFinitePolicyError` (a `FloatingPointError`, so existing handlers keep
catching it) is reserved for a genuinely diverged policy: non-finite logits, or
a non-finite total on the fallback path. Underflow is a float32 limit rather
than a diverged policy, so it degrades to a uniform draw over the legal actions
and increments `agents.rl_agent.underflow_fallback_count`, warning once per
process. A run is never worth losing over one unrepresentable decision.

`GPUContextLostError` in `agents/nn.py` is unrelated to both and is not a
policy fault at all: the CUDA context has been destroyed underneath the
process, so no allocation on the device -- weights, optimizer moments, or
buffers -- still exists. `is_gpu_context_loss` tells it apart from a merely
full device by the CUDA status name, which is the only thing that separates
them, because CuPy raises the same exception types for both. Getting that
wrong in either direction is expensive: a dead context reported as exhausted
memory sends the operator hunting VRAM that is not the problem and retries
into a device that can no longer run a kernel, while exhausted memory reported
as a dead context throws away a working batch-size fallback. See
[`training/rl/README.md`](../training/rl/README.md) for the recovery path.

## State Encoding

For `T` ruleset tiles and `S` pip values, `DominoEncoder` produces
`5T + 3S + 7` inputs and `2T` outputs:

| Slice | Meaning |
|---|---|
| `my_hand[T]` | Tiles currently held by the acting player. |
| `played[T]` | Tiles already played on the board. |
| `played_turn[T]` | Normalized turn when each tile was played, using `MAX_TURN = 52`; zero means unplayed. |
| `played_by_me[T]` | Tiles played by the acting player. |
| `played_by_opponent[T]` | Tiles played by the opponent. |
| `left_end[S]` | One-hot encoding of the current left end. |
| `right_end[S]` | One-hot encoding of the current right end. |
| `hand_sizes[2]` | Player hand sizes divided by the ruleset initial hand size. |
| `stock_size[1]` | Stock size divided by the ruleset initial stock. |
| `draw_count_by_player[2]` | Draw counts divided by the ruleset initial stock. |
| `pass_count_by_player[2]` | Pass counts for players 0 and 1 divided by `MAX_TURN`. |
| `opponent_suit_probabilities[S]` | Probability that the opponent currently holds at least one tile containing each pip value. |

The opponent probability feature is bounded in `[0, 1]`: `0.0` means the
opponent is known to hold no tile with that pip value, and `1.0` means at least
one such tile is known to be present. For two-player games, the model replays public history with the
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
because they consume only the current ruleset-sized vector. Direct opponent-model callers
still receive traces by default.

## Action Encoding

The neural output space has `2T` actions:

- `T` tile actions on the left end;
- `T` tile actions on the right end.

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

Checkpoints are ruleset-specific. Loaders validate input/output dimensions and
never pad, copy, or remap policy weights between variants. Unnamed legacy
checkpoints remain double-six only.

Weights trained with the older absence-confidence feature also load by shape,
but they are semantically stale. Archive them and retrain after regenerating the
dataset with the current opponent-suit probability feature.
