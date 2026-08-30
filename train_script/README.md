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

## Reward-distance grid search

`run_grid_search.sh` compares all four `--reward-distance-mode` choices on the
double-four ruleset. Each mode runs with seeds `137` and `271`, giving eight
independent `python -m training.pipeline forever` points:

| Mode | Seeds | Points |
|---|---|---:|
| `turn-turn` | `137`, `271` | 2 |
| `decision-decision` | `137`, `271` | 2 |
| `turn-decision` | `137`, `271` | 2 |
| `decision-turn` | `137`, `271` | 2 |

The numerical reward parameters retain their normal defaults. Each seed owns
one canonical double-four dataset and supervised checkpoint; its four mode
runs reuse those assets rather than regenerating or retraining them.

Because `forever` has no game target, each point is capped by wall clock.
The timer starts when the pipeline prints its `Canonical RL run` banner, so
the dataset and supervised stages at the front of the first point are not
charged against any point's RL budget. The cap is delivered as SIGTERM, which
the pipeline's own shutdown flag turns into a boundary checkpoint before a
clean exit; if that does not land within `--grace`, the script escalates to a
second SIGTERM and then SIGKILL, and records the point as `hard-stopped`.
Keep `--grace` above one RL iteration so the graceful path is the one taken.
A hard-stopped point is not considered complete and is retried on the next
invocation.

```bash
train_script/run_grid_search.sh --dry-run
train_script/run_grid_search.sh
train_script/run_grid_search.sh --only 'turn_turn_*'
train_script/run_grid_search.sh --only '*_seed137'
```

Each point receives 1h30 of RL wall time. Dataset and supervised preparation do
not consume that budget because timing starts at the canonical RL banner.
Completed points are recorded below
`grid_search_results/double_four_reward_distance/grid_state.tsv` and skipped on
re-invocation; `--force` re-runs them. Per-point pipeline output is stored next
to the state file as `<run-name>.log`.

## Opponent-bucket, PPO learning-rate, and baseline sequences

Machine-specific wrappers run three planned ablations with comparable
machine-adjusted RL wall-clock budgets. The bucket sequence uses `double-six`
and runs, in order:

1. `heuristic,recent`;
2. every current bucket except `random`;
3. `heuristic` only.

The PPO learning-rate sequence uses `double-three` and runs the default
`0.001`, followed by `0.002`, `0.004`, `0.008`, and `0.016`. No other training
option is changed. All points use separate stable run names, while the
seed-42 standard dataset and supervised checkpoint are reused where their
ruleset matches.

The baseline sequence is currently assigned to the Diego notebook and Rick
desktop. It uses `double-three`, fixes PPO learning rate at `0.01` (10x the
default), and runs six baseline choices in order:

1. `--baseline zero`;
2. `--baseline 5`;
3. `--baseline -5`;
4. `--value-head --baseline value-head-own-nn`;
5. `--value-head --baseline value-head`;
6. `--value-head --baseline value-head-no-up`.

The fixed state-calculated `lookup-table` baseline is deliberately outside
this original six-point sequence. Adding it to the implementation does not
change the identity, order, or resume state of these six experiments.

| Machine | Coefficient | Buckets per point | PPO LR per point | Baseline per point |
|---|---:|---:|---:|---:|
| Diego notebook | 1.0 | 5h | 2h | 2h |
| Rick desktop | 2.4 | 12h | 4h48 | 4h48 |
| Rick old notebook | 3.4 | 17h | 6h48 | — |
| Rick new notebook | 1.5 | 7h30 | 3h | — |

Run the available scripts assigned to a machine from the repository root:

```bash
# Diego notebook
train_script/run_bucket_tests_diego_notebook.sh
train_script/run_ppo_lr_tests_diego_notebook.sh
train_script/run_baseline_tests_diego_notebook.sh

# Rick desktop
train_script/run_bucket_tests_rick_desktop.sh
train_script/run_ppo_lr_tests_rick_desktop.sh
train_script/run_baseline_tests_rick_desktop.sh

# Rick old notebook
train_script/run_bucket_tests_rick_old_notebook.sh
train_script/run_ppo_lr_tests_rick_old_notebook.sh

# Rick new notebook
train_script/run_bucket_tests_rick_new_notebook.sh
train_script/run_ppo_lr_tests_rick_new_notebook.sh
```

Each wrapper is idempotent and resume-aware. Its state and attempt logs live
under `train_script/grid_search_results/<machine>/<experiment>/`. Running the
same command again skips completed points and resumes the interrupted one with
only its unused RL budget. `--dry-run` prints every fresh command without
starting a pipeline, and `--help` documents selection, forced restart, timing,
and forwarded pipeline options.

## Validation

For script-only changes, run at least:

```bash
bash -n train_script/run_training_pipeline.sh
bash -n train_script/run_grid_search.sh
train_script/run_training_pipeline.sh --help
train_script/run_grid_search.sh --dry-run
python -m train_script.run_pipeline --help
```

Follow [`CONTRIBUTING.md`](../CONTRIBUTING.md) for Pylint and the complete
impact-matrix checks.
