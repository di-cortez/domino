# Training scripts

These wrappers compose the training modules documented in
[`training/README.md`](../training/README.md). The canonical Python wrapper
`python -m train_script.run_pipeline` mirrors
`python -m training.pipeline`, including `--gpi` with the supported fixed
choices. The older shell batch wrapper retains `training.self_play`'s default
of 2,000 games per iteration.

## Full batch wrapper

`run_training_pipeline.sh` runs four stages in order:

1. supervised dataset generation;
2. supervised policy training;
3. RL refinement;
4. the four supported agent-vs-random diagnostics.

The canonical and resumable entry point remains `python -m training.pipeline`.
The shell wrapper retains its historical 500,000-game RL and 50,000-game
diagnostic profile for experiments.

```bash
train_script/run_training_pipeline.sh
train_script/run_training_pipeline.sh --help
```

Common examples:

```bash
# Reuse dataset and supervised weights while varying RL settings.
train_script/run_training_pipeline.sh --skip-dataset --skip-sl \
  --rl-learning-rate 0.0005 --rl-gamma 0.97 \
  --rl-weights-file models/domino_rl_weights_lr0005_gamma097.npz

# Quick RL-stage smoke. Each iteration uses self-play's fixed default GPI.
train_script/run_training_pipeline.sh --skip-dataset --skip-sl \
  --rl-iterations 2 --rl-checkpoint-interval 1 \
  --rl-weights-file models/smoke_test.npz

# PPO actor-critic (add --rl-no-ppo for the historical update).
train_script/run_training_pipeline.sh --skip-dataset --skip-sl \
  --rl-value-head --rl-weights-file models/domino_rl_weights_critic.npz
```

Important RL options are:

| Flag | Meaning | Default |
|---|---|---:|
| `--rl-total-training-games` | Exact real-game budget | `500000` |
| `--rl-iterations` | Legacy fixed iteration budget using the default GPI | unset |
| `--rl-learning-rate` | Learning rate | `0.001` |
| `--rl-gamma` | Terminal-reward discount | `1.0` |
| `--rl-reward-schema` | `default`, `sparse`, or `shaped` | `default` |
| `--rl-workers` | CPU rollout workers or `auto` | `auto` |
| `--rl-value-head` | Enable the value head with PPO or REINFORCE | off |
| `--weight-decay` | L2 decay forwarded to both the SL and the RL stage | off |
| `--dropout` | Hidden-layer dropout forwarded to both the SL and the RL stage | off |
| `--rl-ppo` / `--rl-no-ppo` | PPO or one-update REINFORCE | PPO |
| `--hidden-layers` | Hidden policy layers, 1 to 8, used by supervised training | `2` |
| `--hidden1-size` ... `--hidden8-size` | Hidden policy widths used by supervised training | `256`, then `128` |
| `--rl-seed` | Fixed training seed | unset |
| `--rl-device` | `auto`, `cpu`, or `gpu` | `auto` |

Run `train_script/run_training_pipeline.sh --help` for dataset, supervised,
memory, checkpoint, and diagnostic controls. The wrapper intentionally has no
GPI flag.

RL rollout workers are CPU-only. With `--rl-workers auto`, worker candidates
are benchmarked sequentially and the first candidate below the required
marginal gain is rejected. Benchmark trajectories use isolated seeds and are
discarded; weights, optimizer state, RNGs, opponent pool, and real-game
counters are restored before training.

Each diagnostics run is written below
`diagnostics/results/<rl-weights-basename>/`. Existing directories are
validated against the requested model and configuration before reuse.

## Validation

For script-only changes, run at least:

```bash
bash -n train_script/run_training_pipeline.sh
train_script/run_training_pipeline.sh --help
python -m train_script.run_pipeline --help
```

Follow [`CONTRIBUTING.md`](../CONTRIBUTING.md) for Pylint and the complete
impact-matrix checks.
