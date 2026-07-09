# Models

This directory stores generated neural-network weights. Model files are ignored
by Git and can be regenerated through the training pipeline.

| File | Contents |
|---|---|
| `domino_sl_weights.npz` | Supervised MLP weights trained from heuristic labels. Used by `NeuralAgent`. |
| `domino_rl_weights.npz` | RL policy/value weights refined by self-play. Used by `RLAgent`. |

Both files store policy arrays `W1`, `b1`, `W2`, `b2`, `W3`, and `b3` with
`numpy.savez`. RL checkpoints also store `Wv` and `bv`, the value baseline used
by policy-gradient training.

Regenerate in order:

```bash
python -m training.dataset_generator
python -m training.training_loop
python -m training.self_play
```

Evaluate checkpoints with:

```bash
python -m diagnostics.evaluate --agent rl --opponent heuristic --weights models/domino_rl_weights.npz
```
