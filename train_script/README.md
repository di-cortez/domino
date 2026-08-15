# Training scripts

These wrappers compose the training modules documented in
[`training/README.md`](../training/README.md). The canonical Python wrapper
`python -m train_script.run_pipeline` mirrors
`python -m training.pipeline`, including `--gpi` with the supported fixed
choices. The older shell batch wrapper retains `training.rl.cli`'s default
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

# PPO actor-critic (use --rl-ppo-max-epochs 1 for REINFORCE).
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
| `--rl-alpha` | Convex mix of local vs terminal reward (`0` = terminal only, `1` = local only) | `0.5` |
| `--rl-event-reward-decay` | Per-turn decay crediting a draw/pass event to earlier decisions | `0.90` |
| `--rl-workers` | CPU rollout workers or `auto` | `auto` |
| `--rl-value-head` | Enable the value head with PPO or REINFORCE | off |
| `--weight-decay` | L2 decay forwarded to both the SL and the RL stage | off |
| `--dropout` | Hidden-layer dropout forwarded to both the SL and the RL stage | off |
| `--rl-ppo-max-epochs` | `1` selects REINFORCE; `2`–`16` select PPO | `4` |
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

## Reward grid search

`run_reward_grid_search.sh` sweeps the three reward tunables, running one
`python -m training.pipeline forever` per grid point with a distinct
`--run-name` so each point gets its own run directory under `models/rl/`.

A parameter is swept only where it can change the reward
(`R_T = (1 - alpha) * gamma ** k * R_f + alpha * R_l`), which is 15 points:

| alpha | Swept | Points |
|---|---|---:|
| `0` | `--gamma` only; the local term is zeroed | 3 |
| `0.5` | `--gamma` x `--event-reward-decay` | 9 |
| `1` | `--event-reward-decay` only; the terminal term is zeroed | 3 |

Because `forever` has no game target, each point is capped by wall clock.
The timer starts when the pipeline prints its `Canonical RL run` banner, so
the dataset and supervised stages at the front of the first point are not
charged against any point's RL budget. The cap is delivered as SIGTERM, which
the pipeline's own shutdown flag turns into a boundary checkpoint before a
clean exit; if that does not land within `--grace`, the script escalates to a
second SIGTERM and then SIGKILL, and records the point as `hard-stopped`.
Keep `--grace` above one RL iteration so the graceful path is the one taken.

```bash
train_script/run_reward_grid_search.sh --dry-run     # plan only
train_script/run_reward_grid_search.sh               # 15 points, 2h RL each
train_script/run_reward_grid_search.sh --only 'a05_*'
```

Completed points are recorded in `grid_search_results/grid_state.tsv` and
skipped on re-invocation, so an interrupted sweep continues where it stopped;
`--force` re-runs them. Per-point pipeline output goes to
`grid_search_results/<run-name>.log`.

## Validation

For script-only changes, run at least:

```bash
bash -n train_script/run_training_pipeline.sh
bash -n train_script/run_reward_grid_search.sh
train_script/run_training_pipeline.sh --help
train_script/run_reward_grid_search.sh --dry-run
python -m train_script.run_pipeline --help
```

Follow [`CONTRIBUTING.md`](../CONTRIBUTING.md) for Pylint and the complete
impact-matrix checks.
