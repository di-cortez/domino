# Training

This folder contains the full training pipeline:

1. generate supervised examples from heuristic-vs-heuristic games;
2. train a supervised neural policy;
3. refine that policy through self-play reinforcement learning.

| File | Purpose |
|---|---|
| `dataset_generator.py` | Simulates games and writes `dataset/supervised_dataset.jsonl`. |
| `training_loop.py` | Loads the JSONL dataset, trains `SupervisedNeuralNetwork`, and saves `models/domino_sl_weights.npz`. |
| `self_play.py` | Loads the supervised policy or an existing RL checkpoint, then trains `PolicyNetwork` with policy gradients. |

## Supervised Dataset

Run:

```bash
python -m training.dataset_generator
```

The generator records `(state, target_action)` pairs from games played by
`StrategicAgent` against itself. Rendering-only fields are removed before each
state is written.

## Supervised Training

Run:

```bash
python -m training.training_loop
```

The loop:

- reads `dataset/supervised_dataset.jsonl`;
- encodes states and actions with `DominoEncoder`;
- splits data into training and validation sets;
- trains the MLP in mini-batches;
- keeps the best validation checkpoint in memory;
- saves `models/domino_sl_weights.npz`.

`agents/nn.py` uses CuPy automatically when it is installed. Validation loss is
computed in batches so large datasets do not allocate a full GPU copy at once.

## Self-Play RL

Run:

```bash
python -m training.self_play
```

Default behavior:

- if `models/domino_rl_weights.npz` exists, resume from it;
- otherwise warm-start from `models/domino_sl_weights.npz`;
- train against a pool of frozen snapshots of the current policy;
- periodically evaluate against `StrategicAgent`;
- save `models/domino_rl_weights.npz`.

`TRAINING_OPPONENT` at the top of `self_play.py` controls the training
opponent:

| Value | Meaning |
|---|---|
| `"self_play"` | Train against a rotating pool of frozen policy snapshots. |
| `"heuristic"` | Train directly against `StrategicAgent`, useful for controlled comparisons. |

The snapshot pool lives only in memory. Resuming from an RL checkpoint restores
the policy weights, but not the previous in-memory opponent pool.
