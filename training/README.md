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
Diagnostics counts are per matchup, and their mode depends on pipeline scale:

| Pipeline scale | Diagnostic mode | Matchups |
|---|---|---:|
| `small` | `fast` | 2 |
| `default` | `default` | 10 |
| `big` | `complete` | 15 |
| `huge` | `complete` | 15 |

For example, `small` runs 2,000 games in each of 2 matchups, for 4,000
diagnostic games in total.

| File | Purpose |
|---|---|
| `dataset_generator.py` | Simulates games and writes only real decisions to `dataset/supervised_dataset.jsonl`. |
| `training_loop.py` | Loads the JSONL dataset, trains `SupervisedNeuralNetwork`, and saves `models/domino_sl_weights.npz`. Forced draw/pass and single-option labels are skipped defensively. |
| `self_play.py` | Loads the supervised policy or an existing RL checkpoint, then trains `PolicyNetwork` from real learner decisions with reward shaping and option-count multipliers. |

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
```

The generator records `(state, target_action)` pairs from games played by
`StrategicAgent` against itself. Engine states are already compact and do not
include rendering metadata. The command prints a startup RAM/GPU memory
snapshot, shows a progress bar, and reports total elapsed time. Its default is
10,000 games.

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
- encodes states and tile-play actions with `DominoEncoder`;
- saves/loads `dataset/supervised_dataset_encoded.npz` to skip repeated JSONL encoding;
- splits data into training and validation sets;
- keeps the encoded dataset and both splits in NumPy host memory;
- trains the MLP in mini-batches of 1024 examples;
- keeps the best validation checkpoint in memory;
- keeps only the 10 most recent archival checkpoints;
- saves `models/domino_sl_weights.npz`.

`agents/nn.py` uses CuPy automatically when it is installed. Only the current
training or validation mini-batch is transferred to the GPU; complete datasets
are never copied into VRAM. GPU memory usage therefore stays proportional to
the batch size rather than the number of encoded examples. The command prints
startup memory, checkpoint-to-checkpoint time, and total elapsed time.
Archival files in `models/supervised_checkpoints/` are pruned after every save,
so repeated or large training runs retain at most 10 of them.

The encoded cache is rebuilt automatically when the source JSONL file changes,
the encoder input/output dimensions change, or the feature-version tag changes.

### Optional SL controls

The default command uses a fixed learning rate, no weight decay, and no early
stopping. It still tracks validation loss and saves the best validation weights.

Enable any control independently by adding its flag:

```bash
python -m training.training_loop --weight-decay
python -m training.training_loop --early-stopping
python -m training.training_loop --lr-decay
```

Passing a flag without a value uses these defaults:

| Flag | Enabled behavior | Default value when enabled |
|---|---|---:|
| `--weight-decay [COEFFICIENT]` | Adds L2 decay to `W1`, `W2`, and `W3`, but not biases | `0.0001` |
| `--early-stopping [PATIENCE]` | Stops after this many validation checks without improvement | `5` |
| `--lr-decay [FACTOR]` | Multiplies the learning rate after each failed validation check | `0.5` |

Validation is checked every 10 epochs. The options can be combined and can
receive explicit values:

```bash
python -m training.training_loop \
  --weight-decay 0.00005 \
  --early-stopping 8 \
  --lr-decay 0.7
```

Reported training and validation losses remain cross-entropy values, allowing
loss curves to be compared with runs that do not enable weight decay.

`run_pipeline.py` accepts the same flags and forwards them only to supervised
training:

```bash
python run_pipeline.py small --weight-decay --early-stopping --lr-decay
```

## Self-Play RL

Run:

```bash
python -m training.self_play
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

The default command reproduces the original fixed training behavior exactly:
no terminal-reward discount, the original reward constants, gradient clipping
at norm `5.0`, no advantage normalization, and unseeded randomness. Every
control below is opt-in and defaults to that behavior, so omitting all of them
changes nothing:

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
