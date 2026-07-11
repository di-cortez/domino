# Training

This folder contains the full training pipeline:

1. generate supervised examples from real heuristic decisions;
2. train a supervised neural policy;
3. refine that policy through self-play reinforcement learning.

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
snapshot, shows a progress bar, and reports total elapsed time.

`StrategicAgent` now uses the exact two-player opponent model from
`middleware/opponent_model.py`. Dataset generation is therefore slower than the
old heuristic-only version, but each saved state includes the computed
`opponent_suit_probabilities` so supervised training can reuse them without
replaying the exact belief model for every row.

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
- trains the MLP in mini-batches of 1024 examples;
- keeps the best validation checkpoint in memory;
- saves `models/domino_sl_weights.npz`.

`agents/nn.py` uses CuPy automatically when it is installed. Validation loss is
computed in batches so large datasets do not allocate a full GPU copy at once.
The command prints startup memory, checkpoint-to-checkpoint time, and total
elapsed time.

The encoded cache is rebuilt automatically when the source JSONL file changes,
the encoder input/output dimensions change, or the feature-version tag changes.

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

The command prints startup memory, checkpoint-to-checkpoint time, and total
elapsed time. Iteration logs omit entropy and keep the focus on returns, value
loss, and win counts.

The learner trajectory stores only real decisions. Draw, pass, and single-option
tile plays are forced actions, so `RLAgent` returns them directly without
calling the network or saving a trajectory step. Each saved step carries the
legal-action mask used by the RL backward pass, keeping sampling and gradient
calculation on the same masked policy distribution.

The masked-gradient change does not alter checkpoint shapes or `.npz` keys, so
existing weights still load. For clean comparisons, archive the previous RL
checkpoint and start the next long RL run from `models/domino_sl_weights.npz`.

`TRAINING_OPPONENT` at the top of `self_play.py` controls the training opponent:

| Value | Meaning |
|---|---|
| `"self_play"` | Train against a rotating pool of frozen policy snapshots. |
| `"heuristic"` | Train directly against `StrategicAgent`, useful for controlled comparisons. |

The RL reward now includes weak shaping:

| Event | Reward |
|---|---:|
| terminal win | `+1.0` |
| terminal draw | `0.0` |
| terminal loss | `-1.0` |
| opponent forced to draw | `+0.05` |
| opponent forced to pass | `+0.05` |
| final remaining pips | `-0.001 * remaining_pips` |

If the opponent draws and then passes, both intermediate rewards are applied.
The final pip penalty is applied to the learner's own final hand.

Each saved decision return is then multiplied by the number of tile-play options
available at that decision:

| Legal tile-play options | Multiplier |
|---:|---:|
| 2 | `1.0` |
| 3 | `2.0` |
| 4 | `5.0` |
| 5 or more | `10.0` |

The policy gradient still uses `clip_grad_norm=5.0` in `PolicyNetwork` to limit
large updates from rare high-choice decisions.

The snapshot pool lives only in memory. Resuming from an RL checkpoint restores
the policy weights, but not the previous in-memory opponent pool.
