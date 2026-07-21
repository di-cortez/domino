# Agents

All playable agents expose the same `choose_move(state, legal_actions)` shape so
`GameManager` can run any pair without knowing how each decision is made.

| File | Purpose |
|---|---|
| `agent.py` | Uniform-random `RandomAgent` baseline. |
| `encoder.py` | Single source of truth for state-to-vector and tile-play action encoding. |
| `heuristic_agent.py` | `StrategicAgent`, the exact-probability rule-based teacher used for supervised labels and benchmarks. |
| `nn.py` | Per-network NumPy/CuPy float32 MLP backend with explicit `auto`/`cpu`/`gpu` selection. |
| `neural_agent.py` | Loads `models/domino_sl_weights.npz` and plays the supervised policy. |
| `random_neural_agent.py` | Uses the same supervised architecture with fixed random initialization and no checkpoint. |
| `rl_nn.py` | Masked REINFORCE network with entropy regularization and an optional value head. |
| `rl_agent.py` | Wraps `PolicyNetwork` for training trajectories or deterministic evaluation play. |

The opponent belief model lives in `middleware/opponent_model.py` because it is
shared by agents, training, diagnostics, and the UI.

Each supervised network owns an exact `network.device` and `network.xp` instead
of relying on one module-wide array backend. Inference accepts NumPy or CuPy
inputs and converts them to that network's backend in `float32`. Training may
use host RAM, disk-backed arrays, full GPU residency, or a reusable rotating GPU
window; these storage policies live in `training/supervised_runtime.py`.
Optional supervised weight decay applies only to `W1`, `W2`, and `W3`; bias
vectors are never regularized.

`GPU_ENABLED` becomes true only when CuPy imports, CUDA reports a visible
device, and a synchronized float32 allocation succeeds. `GPU_UNAVAILABLE_REASON` records why the probe failed,
allowing pipeline and standalone logs to explain a NumPy/CPU fallback instead
of claiming that an importable but unusable CuPy installation is active. See
the root README for the complete Linux driver, CuPy `[ctk]`, verification, and
troubleshooting procedure.

If the `DOMINO_VRAM_LIMIT_MB` environment variable is set when `nn.py` is
first imported and CuPy is active, it caps that process's CuPy default
memory pool (`cupy.get_default_memory_pool().set_limit`) at that many
mebibytes; exceeding it raises `cupy.cuda.memory.OutOfMemoryError` instead of
growing unbounded. Unset (the default) means no limit, unchanged from prior
behavior. `train_script/run_rl_parameter_sweep.sh` sets this automatically,
sized from detected total GPU memory divided by `--jobs`, so several
concurrent training subprocesses sharing one GPU can't collectively exceed
its VRAM.

`rl_nn.py::PolicyNetwork` uses the same per-network resolver via a `device`
parameter (`"auto"` follows usable CuPy; `"cpu"`/`"gpu"` are explicit), so an RL run
can be pinned to CPU while supervised training elsewhere in the same process
still uses the GPU, or vice versa. `PolicyNetwork.load_from_sl` also accepts
a pre-loaded `data` mapping of SL weight arrays, to warm-start many networks
from the same checkpoint without re-reading it from disk each time.

`RandomNeuralAgent` is an untrained control for diagnostics. Seed `0` is local
to its network initialization, so every matchup uses the same random policy and
does not perturb the random sequence used to shuffle and play games.
`NeuralAgent.load(..., device=...)` and `RandomNeuralAgent.create(...,
device=...)` preserve that backend choice. CPU-only workers set
`DOMINO_FORCE_CPU=1`, so they never initialize a CUDA context.

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
final tie-breaker. `StrategicAgent`, `NeuralAgent`, `RandomNeuralAgent`, and
`RLAgent` use persistent exact models with intermediate trace recording disabled
because they consume only the current seven-vector. Direct opponent-model callers
still receive traces by default.

## Action Encoding

The neural output space now has 56 actions:

- 28 tile actions on the left end;
- 28 tile actions on the right end.

Draw, pass, and single-option tile plays are forced by the current rules
engine. `NeuralAgent`, `RandomNeuralAgent`, and `RLAgent` return them directly
without calling the network. `StrategicAgent` also returns a single tile-play
option before running exact inference. These are not learned RL decisions, and
`RLAgent` does not save a trajectory step for them.

`RLAgent` has three explicit policy modes:

| Mode | Legal policy choice | Stores trajectory |
|---|---|---|
| `training` | Samples from the masked distribution | Yes |
| `stochastic_evaluation` | Samples from the masked distribution | No |
| `evaluation` | Selects the largest masked probability | No |

Self-play pool opponents use stochastic evaluation. UI play, diagnostics, and
checkpoint evaluation use deterministic evaluation.

RL trajectory steps store the encoded state, sampled action index, legal-action
mask, decision turn, option count, and local reward accumulator. During
self-play, draw/pass events are distributed to earlier real decisions with
temporal decay. The policy-gradient backward pass uses the saved mask to
renormalize the softmax over legal actions only, so illegal actions receive no
direct policy or entropy gradient.

`PolicyNetwork` is policy-only by default. Optional value-head training adds a
linear `V(s)` prediction from the second hidden layer and stores `Wv`/`bv` next
to `W1`, `b1`, `W2`, `b2`, `W3`, and `b3`. Policy-only loading ignores those
extra arrays, while value-head loading initializes them to zero when absent.

Because the input/output shapes changed from the old 86/58 encoder to the new
168/56 encoder, old `domino_sl_weights.npz` and `domino_rl_weights.npz`
checkpoints are not compatible. Regenerate the supervised dataset, retrain SL,
and then retrain RL.

Weights trained with the older absence-confidence feature also load by shape,
but they are semantically stale. Archive them and retrain after regenerating the
dataset with the current opponent-suit probability feature.
