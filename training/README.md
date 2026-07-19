# Training

This folder contains the full training pipeline:

1. generate supervised examples from real heuristic decisions;
2. train a supervised neural policy;
3. refine that policy through self-play reinforcement learning.

From the repository root, `run_pipeline.py` runs the full sequence with compact
progress bars and one summary line per stage:

```bash
python run_pipeline.py
python run_pipeline.py small
python run_pipeline.py big
python run_pipeline.py huge
```

The default runner uses the same workload as the individual commands. `small`
uses one fifth of the default counts, `big` uses five times the default counts,
and `huge` uses twenty times the default counts. The scaled counts apply to
dataset games, supervised epochs, RL iterations, and diagnostic games per
matchup. RL games per iteration stay at 40 so the scale remains linear.
The default dataset workload is 10,000 heuristic-vs-heuristic games.
Diagnostics counts are per matchup. Mode labels remain compatible with older
commands, but all scales evaluate the same five agents against `random`:

| Pipeline scale | Diagnostic mode | Matchups |
|---|---|---:|
| `small` | `fast` | 5 |
| `default` | `default` | 5 |
| `big` | `complete` | 5 |
| `huge` | `complete` | 5 |

For example, `small` runs 2,000 games in each of 5 matchups, for 10,000
diagnostic games in total.

| File | Purpose |
|---|---|
| `dataset_generator.py` | Coordinates retained worker tuning, bounded SQLite aggregation, and atomic ordered JSONL output. |
| `dataset_parallel.py` | Plays deterministic dataset games in a bounded CPU-only worker pool with dynamic scheduling and memory fallback. |
| `training_loop.py` | Selects safe host/GPU storage, orchestrates retained supervised batch tuning and plateau scheduling, and saves `models/domino_sl_weights.npz`. |
| `supervised_runtime.py` | Implements CPU/GPU batch candidates, synchronized retained timing, GPU residency probes/windows, and supervised memory telemetry. |
| `self_play.py` | Loads the supervised policy or an existing RL checkpoint, orchestrates parallel rollouts, and applies parent-only policy updates. |
| `rl_parallel.py` | Shares frozen policy snapshots with deterministic CPU-only rollout workers and retains completed games across memory fallback. |

## Important Shape Change

The neural encoder now uses a 168-feature input vector and a 56-action output
space. The policy only chooses real tile-play decisions. Draw, pass, and
single-option tile plays are forced rule actions and bypass training.

The last seven input features are now opponent suit-presence probabilities:
`0.0` means known absence and `1.0` means known presence. This replaces the old
absence-confidence feature. Any encoded cache or model trained with the old
feature semantics should be treated as stale even though the array shapes still
match.

Old checkpoints trained with the previous 86-input/58-output encoder are not
compatible. After copying these files into the repo, run the pipeline again:

```bash
python -m training.dataset_generator
python -m training.training_loop
python -m training.self_play
```

## Supervised Dataset

Run:

```bash
python -m training.dataset_generator
python -m training.dataset_generator --workers auto --seed 123
python -m training.dataset_generator --workers 4 --games 5000
```

The generator records `(state, target_action)` pairs from games played by
`StrategicAgent` against itself. Engine states are already compact and do not
include rendering metadata. The command prints a startup RAM/GPU memory
snapshot, shows a progress bar, and reports total elapsed time. The standalone
command defaults to 30,000 games; the default full pipeline requests 10,000.

Automatic mode benchmarks 1, 2, 4, 6, ... CPU-only workers, capped at 20.
Every attempt generates and retains 1% of the requested games. Testing stops on
a memory/error guard or below 10% marginal gain, then the remaining absolute
game ids run with the selected count. Per-game seeds make fixed-worker and
automatic runs identical for the same `--seed`, regardless of scheduling or a
runtime fallback. Use `--help` for benchmark fractions, RAM reserve, per-worker
RSS, and estimated-worker-memory controls.

Workers serialize one compact payload per game. Only the parent writes those
payloads to a disposable SQLite database, keeping RAM bounded while results
arrive out of order. Final rows are emitted in game-id order, and the existing
JSONL is replaced atomically only after every requested game succeeds.

`StrategicAgent` now uses the exact two-player opponent model from
`middleware/opponent_model.py`. Dataset generation is therefore slower than the
old heuristic-only version, but each saved state includes the computed
`opponent_suit_probabilities` so supervised training can reuse them without
replaying the exact belief model for every row. The model keeps temporal draw
cohorts in `slots_exact`, converts once to integer `mu(H)` weights when the raw
hand bound reaches 500, and never falls back to particles.

A row is written only when the player had at least two legal tile-play choices.
The following turns are skipped:

- forced draw;
- forced pass;
- forced opening double;
- any state with only one legal tile play.

## Supervised Training

Run:

```bash
python -m training.training_loop
```

The loop:

- reads `dataset/supervised_dataset.jsonl`;
- filters out forced draw/pass examples;
- filters out single-option tile-play examples;
- scans the JSONL twice and encodes `float32` arrays without retaining decoded records;
- checks cgroup-aware host RAM before every material allocation;
- encodes states and tile-play actions with `DominoEncoder`;
- uses `dataset/supervised_dataset_encoded.npz` when the encoded dataset fits safely in RAM;
- otherwise atomically builds disk-backed `supervised_dataset_X.npy`,
  `supervised_dataset_Y.npy`, and `supervised_dataset_metadata.json` files and
  opens them read-only with `mmap`;
- splits data into training and validation sets;
- selects CPU/GPU independently with `--sl-device {auto,cpu,gpu}` (`--device`
  is a standalone alias);
- retains every batch-autotuning epoch as real training;
- keeps the complete dataset in GPU memory when safe, or rotates one reusable
  GPU window through a global per-epoch permutation when it is not;
- keeps the best validation checkpoint in memory;
- keeps only the 10 most recent archival checkpoints;
- saves `models/domino_sl_weights.npz`.

CPU batch candidates are powers of two from 1,024 through 1,048,576; GPU
candidates start at 2,048. Each candidate runs 10 complete epochs on the same
live network. Timing includes recurring data materialization/transfers and the
forward/backward update, synchronizes CUDA around each GPU epoch, excludes
validation/checkpoint/log time, and compares median **examples/second**. A
larger candidate is accepted only at a gain of at least 10%; a rejected
candidate's epochs still remain in the model. Runs shorter than 10 epochs use
the first safe device default without starting an incomplete benchmark. Use
`--sl-batch-size N` for a fixed batch or `--sl-no-batch-autotune` for the
device default.

GPU mode first probes resident example counts from 2,048 through 1,048,576
without changing weights. It preserves 512 MiB by default for batches,
activations, gradients, CUDA workspace, and fragmentation. `auto` falls back
safely to CPU when that reserve cannot be kept; explicit `gpu` fails before a
training update. Override host and GPU reserves with
`--sl-memory-reserve-mb` and `--sl-gpu-memory-reserve-mb`. The detailed command
reports the selected device, residency mode/capacity, one-time full upload,
batch results, and memory high/low watermarks. `run_pipeline.py` uses
`quiet=True`, so its existing progress bar and compact one-line SL summary do
not gain this detailed chatter.

All supervised inputs, targets, weights, activations, gradients, and new
checkpoints are `float32`. Legacy `float64` checkpoints remain loadable and are
cast on input. Archival files in `models/supervised_checkpoints/` are pruned
after every save, so repeated or large runs retain at most 10 of them.
Archival files in `models/supervised_checkpoints/` are pruned after every save,
so repeated or large training runs retain at most 10 of them.

CuPy import alone is not treated as proof of a working GPU. At startup,
`agents/nn.py` also asks the CUDA runtime for a visible device; a missing driver,
hidden device, or unusable runtime produces a documented NumPy/CPU fallback
reason. The root README's **Linux GPU setup and verification** section contains
the driver checks, CUDA 12.x/13.x installation commands, a real calculation
test, and troubleshooting steps. `run_pipeline.py` prints the selected
supervised and RL-parent backends plus free/total RAM and VRAM before dataset
generation starts.

The encoded cache is rebuilt automatically when the source JSONL file changes,
the encoder input/output dimensions change, or the feature-version tag changes.

RL self-play performs a host-memory preflight for the shared snapshot bank and
expected batch, then checks the actual workspace before each `hstack`. With
`--device auto`, less than 256 MiB of effective free VRAM causes an announced
CPU fallback; explicit `--device gpu` fails early instead. Diagnostics later
run in separate CPU-only processes and never consume training VRAM.

The Python pipeline exposes independent dataset, RL rollout, and diagnostic
worker controls. Dataset generation tunes once for its full workload. RL tunes
across complete early iterations, and diagnostics tune each matchup separately.
All three retain benchmark work and enforce the same hard limit of 20 workers.

### Supervised scheduler and controls

The normal command starts at learning rate `0.005` and enables plateau decay by
default. Validation remains every 10 epochs. The first result establishes the
global best; after five consecutive checks without strict improvement, the LR
is multiplied by `0.5` and only the LR-specific failure counter resets. Another
five failures are required for another reduction. Optional early stopping has
its own counter; its patience should normally exceed LR patience so a reduced
rate has time to help.

Enable any control independently by adding its flag:

```bash
python -m training.training_loop --weight-decay
python -m training.training_loop --early-stopping
python -m training.training_loop --lr-decay 0.7 --lr-decay-patience 8
python -m training.training_loop --no-lr-decay
python -m training.training_loop --sl-device cpu --sl-seed 123
```

Passing a flag without a value uses these defaults:

| Flag | Behavior | Default |
|---|---|---:|
| `--weight-decay [COEFFICIENT]` | Adds L2 decay to `W1`, `W2`, and `W3`, but not biases | `0.0001` |
| `--early-stopping [PATIENCE]` | Stops after this many validation checks without improvement | `5` |
| `--lr-decay [FACTOR]` | Multiplies LR after the configured consecutive failed checks | `0.5` (on) |
| `--lr-decay-patience N` | Consecutive failed validation checks before each reduction | `5` |
| `--no-lr-decay` | Disables plateau scheduling for controlled comparisons | off |
| `--sl-device` / standalone `--device` | `auto`, forced `cpu`, or required `gpu` | `auto` |
| `--sl-batch-size N` | Fixed safe batch; bypasses tuning | unset |
| `--sl-no-batch-autotune` | Uses 1,024 on CPU or 2,048 on GPU | off |
| `--sl-memory-reserve-mb N` | Free host RAM retained | `512` |
| `--sl-gpu-memory-reserve-mb N` | Effective free VRAM retained | `512` |
| `--sl-seed N` | Reproducible initialization and epoch permutations | unset |

Validation is checked every 10 epochs. The options can be combined and can
receive explicit values:

```bash
python -m training.training_loop \
  --weight-decay 0.00005 \
  --early-stopping 12 \
  --lr-decay 0.7 --lr-decay-patience 5 \
  --sl-device gpu
```

Reported training and validation losses remain cross-entropy values, allowing
loss curves to be compared with runs that do not enable weight decay.

`run_pipeline.py` accepts the same flags and forwards them only to supervised
training:

```bash
python run_pipeline.py small --weight-decay --early-stopping 12 --sl-device auto
```

## Self-Play RL

Run:

```bash
python -m training.self_play
python -m training.self_play --rl-workers auto --seed 123
python -m training.self_play --rl-workers 4 --device cpu
```

Default behavior:

- if a compatible `models/domino_rl_weights.npz` exists, resume from it;
- otherwise warm-start from a compatible `models/domino_sl_weights.npz`;
- train against a pool of frozen snapshots of the current policy;
- periodically evaluate deterministic RL play against `StrategicAgent`;
- save `models/domino_rl_weights.npz`.

The learner uses `RLAgent(..., mode="training")`: it samples from the masked
policy and stores trajectory steps. Frozen pool opponents use
`mode="stochastic_evaluation"`: they sample from their masked policies but do
not build training masks or store trajectories. This exposes the learner to
more of each snapshot's policy distribution without retaining unused opponent
experience.

### Parallel rollout generation

All games in an iteration use one immutable learner policy, so rollout work is
independent until batch aggregation. `training/rl_parallel.py` publishes the
current policy and at most `max_pool_size` opponent snapshots in a fixed-size
shared-memory ring. Workers attach NumPy views to that bank, never see the GPU,
and return finalized trajectories through a bounded dynamic queue. The parent
sorts results by game id and remains solely responsible for the gradient,
checkpoint writes, logging, and GPU allocations.

Automatic mode tests 1, 2, 4, 6, ... workers, never exceeding 20 or the number
of games per iteration. Each candidate uses and retains complete iterations
totaling about 1% of the planned iteration count. Testing stops below 10%
marginal rollout-throughput gain, on a resource cap, or when too few untrained
iterations remain. Runtime RAM pressure terminates the current pool, keeps
completed game ids, halves the worker count, and retries only unfinished games.

Per-game SplitMix64-style seeds are derived from the run seed, iteration, and
game id. Parent aggregation is ordered, so the same seed produces bit-identical
checkpoints with one or multiple workers, including after fallback. Useful
controls are:

| Flag | Meaning | Default |
|---|---|---:|
| `--rl-workers` | CPU-only rollout workers or `auto` | `auto` |
| `--rl-autotune-fraction` | Planned iteration fraction retained per candidate | `0.01` |
| `--rl-autotune-min-gain` | Required marginal throughput improvement | `0.10` |
| `--rl-memory-reserve-mb` | Host RAM that must remain free | `512` |
| `--rl-estimated-worker-mb` | Conservative worker-memory estimate for preflight | `256` |
| `--rl-max-worker-rss-mb` | Runtime RSS ceiling for one worker | `1024` |

Checkpoint evaluation games also use the selected rollout pool, but remain
deterministic and alternate the RL player position.

Checkpoint evaluation against `StrategicAgent`, diagnostics, and the UI use
`mode="evaluation"`, which always selects the highest-probability legal action
and stores no trajectory. Their results therefore avoid action-sampling noise.

The command prints startup memory, checkpoint-to-checkpoint time, and total
elapsed time. Iteration logs omit entropy and report the direct reward signal
sent to the policy gradient: reward mean/min/max, good/neutral/bad percentages,
local reward mean, raw event counts, wins, pool size, and gradient norm.

The learner trajectory stores only real decisions. Draw, pass, and single-option
tile plays are forced actions, so `RLAgent` returns them directly without
calling the network or saving a trajectory step. Each saved step carries the
legal-action mask, the decision turn, and the number of legal tile-play options.
Sampling and gradient calculation use the same masked policy distribution.

`PolicyNetwork` uses direct policy-only REINFORCE by default. Default RL
checkpoints contain only the six policy weights shared with supervised
checkpoints: `W1`, `b1`, `W2`, `b2`, `W3`, and `b3`.

Enable the optional actor-critic baseline with:

```bash
python -m training.self_play --value-head
```

This adds a linear `V(s)` head over the second hidden layer. The current
finalized policy reward is the value target, and the masked policy update uses
`reward - V(s)` as its advantage. The value-loss coefficient defaults to `0.5`
(`--value-coef`). In this mode checkpoints also contain `Wv` and `bv`.

`run_pipeline.py` forwards the same flag to RL training:

```bash
python run_pipeline.py small --value-head
```

Policy-only loading ignores `Wv`/`bv`, while value-head loading initializes
them to zero when they are absent. This permits mode changes without changing
the policy architecture, but clean comparisons should still start from the
same supervised checkpoint and use separately archived RL outputs.

### Optional RL controls

The default command preserves the original learning algorithm: no
terminal-reward discount, the original reward constants, gradient clipping at
norm `5.0`, and no advantage normalization. Rollout generation is now parallel
by default, without moving gradient updates out of the parent process:

| Flag | Meaning | Default |
|---|---|---:|
| `--gamma` | Terminal-reward discount per remaining real decision (`1.0` = no discount) | `1.0` |
| `--reward-schema` | Named preset for the terminal/event reward constants: `default` (the table below), `sparse` (win/tie/loss only, no draw/pass shaping or pip penalty), or `shaped` (doubles the draw/pass shaping rewards) | `default` |
| `--clip-grad-norm` | Gradient-norm clipping threshold for the policy-gradient update | `5.0` |
| `--normalize-advantages` / `--no-normalize-advantages` | Standardize the policy signal per batch (mean 0, std 1) before the gradient step | off |
| `--moving-average-window` | Trailing-iteration window for the value-loss/win-rate moving averages printed in the iteration log | `10` |
| `--seed` | Fix `random`/NumPy state, for reproducible comparisons between hyperparameter configurations | unset |
| `--device` | Array backend: `auto` matches `GPU_ENABLED` exactly (CuPy when installed, else NumPy); `cpu`/`gpu` force one backend regardless of what's installed/enabled globally | `auto` |

```bash
python -m training.self_play --gamma 0.97 --reward-schema shaped --seed 42
```

A point-in-time value loss or win rate is dominated by batch noise; the
iteration log always reports `reward mean/std/min/max` and a trailing moving
average of value loss and win rate next to the raw values, so a plateau can be
judged from the average rather than a single noisy line (see
`references/explicacoes/relatorios/relatorio_1407` for the methodology this
follows).

### Device selection (`--device`)

`--device auto` (the default) reproduces the original behavior exactly:
CuPy when installed, NumPy otherwise, same as `GPU_ENABLED` elsewhere in the
project. `--device cpu` or `--device gpu` force one backend for that run
regardless of what's installed, independently of the parent
`SupervisedNeuralNetwork` class used by supervised training (which is
unaffected and still always follows `GPU_ENABLED`). `--device gpu` raises a
clear error if CuPy isn't installed. This is useful because, empirically, RL
self-play is dominated by the exact opponent-hand inference in
`middleware/opponent_model.py` (>80% of iteration time, profiled) rather than
the policy network's forward/backward passes, so CuPy's per-decision
transfer/kernel-launch overhead during rollout can make GPU measurably
*slower* than CPU for this stage specifically -- `--device cpu` is worth
trying if RL training feels slow.

`training.self_play` also accepts `--iterations`, `--games-per-iteration`,
`--training-opponent`, `--learning-rate`, `--entropy-coef`, `--log-interval`,
`--checkpoint-interval`, `--pool-interval`, `--max-pool-size`,
`--evaluation-games`, `--sl-weights-path`, and `--rl-weights-path`; see
`training/self_play.py:add_optional_rl_arguments` for the authoritative
definitions, or run `python -m training.self_play --help`.

`train()` also accepts a programmatic-only `sl_weights_data` parameter (no
CLI flag): a pre-loaded mapping of SL weight arrays, for a caller that runs
many training calls back-to-back from the same SL checkpoint (e.g. a
hyperparameter sweep) and wants to read it from disk once instead of on
every call -- see `train_script/run_rl_parameter_sweep.py`, which loads it
once and reuses it across all 72 of its sweep points. `None` (the default)
reproduces the normal read-from-`sl_weights_path` behavior.

`TRAINING_OPPONENT` at the top of `self_play.py` controls the training opponent:

| Value | Meaning |
|---|---|
| `"self_play"` | Train against a rotating pool of frozen policy snapshots. |
| `"heuristic"` | Train directly against `StrategicAgent`, useful for controlled comparisons. |

The RL reward now uses a uniform terminal reward plus temporally decayed local
draw/pass shaping. For each real decision at turn `d_i`, a later event at turn
`t_e` contributes:

```text
c_e * EVENT_REWARD_DECAY ** (t_e - d_i - 1)
```

with `EVENT_REWARD_DECAY = 0.90`. An immediately following event therefore has
exponent `0` and receives the full event reward. By default (`--gamma 1.0`)
the terminal result is not discounted and is applied uniformly to every real
decision in the game; passing `--gamma` below `1.0` discounts it per
remaining real decision instead (see "Optional RL controls" above).

Reward constants (the `default` reward schema; `--reward-schema` selects an
alternate preset, see above):

| Event | Reward |
|---|---:|
| terminal win | `+0.50` |
| terminal draw | `0.0` |
| terminal loss | `-0.50` |
| opponent draw | `+0.02` |
| opponent pass | `+0.10` |
| learner draw | `-0.02` |
| learner pass | `-0.10` |
| final remaining pips | `-0.001 * remaining_pips` |

Multiple local events are summed. A learner draw/pass penalty is applied to all
earlier real decisions with the same decay rule, not just to the most recent
decision. The final pip penalty is applied to the learner's own final hand.

Each saved decision return is then multiplied by the number of tile-play options
available at that decision:

| Legal tile-play options | Multiplier |
|---:|---:|
| 2 | `1.0` |
| 3 | `2.0` |
| 4 | `5.0` |
| 5 or more | `10.0` |

The final training weight for each decision is:

```text
policy_reward = multiplier * (terminal_reward + local_reward)
```

The policy gradient uses that value directly:

```text
L = -mean(policy_reward * log pi(action | state)) - entropy_coef * entropy
```

Gradient clipping remains active in `PolicyNetwork` to limit large updates from
rare high-choice decisions.

The snapshot pool lives only in memory. Resuming from an RL checkpoint restores
the policy weights, but not the previous in-memory opponent pool.
