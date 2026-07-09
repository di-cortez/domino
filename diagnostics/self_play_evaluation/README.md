# Self-Play Regime Evaluation

This folder compares two RL training regimes supported by
`training/self_play.py`:

- `training_opponent="self_play"`: train against a pool of frozen policy
  snapshots;
- `training_opponent="heuristic"`: train directly against `StrategicAgent`.

The script runs three evaluations with the same diagnostics machinery:

1. self-play checkpoint vs. `StrategicAgent`;
2. heuristic-trained checkpoint vs. `StrategicAgent`;
3. self-play checkpoint vs. heuristic-trained checkpoint.

## Create The Checkpoints

From the repository root:

```bash
python -c "from training.self_play import train; train(training_opponent='self_play', rl_weights_path='models/domino_rl_self_play_weights.npz')"
python -c "from training.self_play import train; train(training_opponent='heuristic', rl_weights_path='models/domino_rl_heuristic_weights.npz')"
```

Both runs should start from the same `models/domino_sl_weights.npz` file for a
fair comparison.

## Run The Comparison

```bash
python diagnostics/self_play_evaluation/compare_regimes.py -n 1000
```

Useful options:

```bash
python diagnostics/self_play_evaluation/compare_regimes.py --help
python diagnostics/self_play_evaluation/compare_regimes.py --self-play-weights models/domino_rl_self_play_weights.npz --heuristic-weights models/domino_rl_heuristic_weights.npz
```

Outputs are written under `diagnostics/results/self_play_vs_heuristic_regimes/` unless
`--output` is provided.

Use enough games for stable confidence intervals. A small direct-match win rate
near 50% with a wide interval is inconclusive, not proof that either regime is
better.
