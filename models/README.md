# Models

This directory stores generated neural-network weights. Model files are ignored
by Git and can be regenerated through the training pipeline.

| File | Contents |
|---|---|
| `domino_sl_weights.npz` | Supervised MLP weights trained from heuristic labels. Used by `NeuralAgent`. |
| `domino_sl_loss.png` | Training and validation loss curves from the latest supervised run. |
| `domino_rl_weights.npz` | RL policy weights refined by self-play. Used by `RLAgent`. |

Both NPZ files store policy arrays `W1`, `b1`, `W2`, `b2`, `W3`, and `b3` with
`numpy.savez`. RL training is policy-only by default. Runs started with
`--value-head` additionally store the training baseline arrays `Wv` and `bv`;
gameplay uses the policy arrays in either case.

`RandomNeuralAgent` does not use a model file. It creates the standard
supervised architecture directly from its fixed random initialization.

Regenerate in order:

```bash
python -m training.dataset_generator
python -m training.training_loop
python -m training.self_play --fresh-from-sl
```

Evaluate checkpoints with:

```bash
python -m diagnostics.pairwise --agent rl --opponent heuristic --weights models/domino_rl_weights.npz
```
