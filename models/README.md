# Models

This directory stores generated neural-network weights. Model files are ignored
by Git and can be regenerated through the training pipeline.

| File | Contents |
|---|---|
| `domino_sl_weights.npz` | Supervised MLP weights trained from heuristic labels. Used by `NeuralAgent`. |
| `domino_sl_loss.png` | Training and validation loss curves from the latest supervised run. |
| `domino_rl_weights.npz` | RL policy weights refined by self-play. Used by `RLAgent`. |
| `domino_sl_standard_seed<seed>.npz` | Canonical supervised policy shared by long-run pipeline levels. |
| `domino_sl_standard_seed<seed>.meta.json` | Dataset origin/hash, architecture, training configuration, convergence, and weights hash. |
| `rl/domino_rl_<level>_seed<seed>/` | Canonical RL run state, checkpoints, pool, and diagnostics. |
| `rl/domino_rl_<small-or-default>_seed<seed>_run<id>/supervised/` | Run-local quick-profile dataset, cache, SL checkpoint, metadata, and loss plot; never reused by another invocation. |

Both NPZ files store policy arrays `W1`, `b1`, `W2`, `b2`, `W3`, and `b3` with
`numpy.savez`. RL training is policy-only by default. Runs started with
`--value-head` additionally store the training baseline arrays `Wv` and `bv`;
gameplay uses the policy arrays in either case.

Regenerate in order:

```bash
python -m training.dataset_generator
python -m training.training_loop
python -m training.self_play --fresh-from-sl
```

For normal full training, prefer the canonical command. `default` creates
isolated quick-run assets, while `big` validates and reuses the standard
seed-addressed assets automatically:

```bash
python -m training.pipeline default
python -m training.pipeline big --resume
```

A canonical RL directory contains `training_state.json` as its committed
resume marker, immutable weights/resume generations under `checkpoint_states/`,
milestone policy files under `checkpoints/`, and convenience aliases
`latest_weights.npz`, `optimizer_state.npz`, `rng_state.json`, and
`opponent_pool/pool_manifest.json`. Resume follows the immutable paths and
hashes in the marker, not `best_checkpoint.json`; `best` is monitoring-only.
Forever runs also persist `periodic_diagnostic_tuning.json`, so the periodic
RL-vs-random worker benchmark is selected once and reused after resume.

Evaluate checkpoints with:

```bash
python -m diagnostics.pairwise --agent rl --opponent heuristic --weights models/domino_rl_weights.npz
```
