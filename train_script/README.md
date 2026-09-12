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
| `--gamma-f` | Terminal discount per selected terminal-distance unit | `0.95` |
| `--reward-eta` | Convex mix of the terminal and immediate returns (`0` = terminal only, `1` = draw/pass shaping only) | `0.115` |
| `--gamma-i` | Immediate-event discount crediting a draw/pass event to earlier decisions | `0.90` |
| `--reward-distance-mode` | Distance units in `gamma_i`/`gamma_f` order | `turn-turn` |
| `--terminal-empty-hand-weight` / `--terminal-blocked-weight` | Relative value of an empty-hand versus a blocked result; only the ratio matters | `1.0` / `1.0` |
| `--immediate-draw-weight` / `--immediate-pass-weight` | Relative value of a draw versus a pass event; only the ratio matters | `1.0` / `1.0` |
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

## Unattended runs

`run_forever_supervised.sh` wraps `python -m training.pipeline` in a restart
loop and forwards every argument verbatim:

```bash
train_script/run_forever_supervised.sh --scale forever --ruleset double-six
```

It exists because a lost CUDA context cannot be recovered inside the process.
When the GPU is reset underneath a healthy run -- an NVIDIA module reload from
an unattended driver upgrade, an Xid fault, a display-server restart, a power
or thermal event -- the policy weights and the optimizer moments die with the
device, so no handler can save the interrupted iteration. Resuming from the
last checkpoint is the entire recovery, and it costs at most
`--checkpoint-interval` iterations.

The pipeline reports that fault as exit code 70 with a single `[gpu]` line
naming the iteration and the checkpoint to resume from, instead of the
seventy-odd identical `CUDA_ERROR_LAUNCH_FAILED` tracebacks CuPy's module
destructors produce against a dead context. The supervisor restarts on it,
declines to restart on `SIGINT`/`SIGTERM`, and stops when two consecutive
attempts complete the same number of games -- a permanent failure wearing a
transient's clothes.

| Variable | Default | Meaning |
|---|---|---|
| `MAX_RESTARTS` | `100` | restart budget before giving up |
| `BACKOFF_S` | `60` | seconds to wait before each restart |
| `LOG_DIR` | `logs` | directory for one log per attempt |
| `PYTHON` | `python` | interpreter to run |

A GPU reset is a machine problem, not a training one. Diagnose it with
`sudo dmesg -T | grep -i xid`, `/var/log/apt/history.log`, and
`nvidia-smi -q -d TEMPERATURE,POWER`; the supervisor only keeps the run alive
while that is happening.

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
default), starts with `--baseline lookup-table`, and then runs the original six
baseline choices in order:

1. `--baseline zero`;
2. `--baseline 5`;
3. `--baseline -5`;
4. `--value-head --baseline value-head-own-nn`;
5. `--value-head --baseline value-head`;
6. `--value-head --baseline value-head-no-up`.

The added lookup point has its own stable machine-specific run name. It does
not change the identity, order, or resume state of the original six
experiments.

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

## Learning rates below the default

`run_lr_low_tests_diego_notebook.sh` runs two `forever` points of five hours
each on `double-six`, both on the project defaults except for the learning
rate: `0.0001` and `0.0025`. The budget, ruleset and seed match the one-factor
sweep, so the two points are directly comparable to its `control` (lr `0.01`)
and extend its ladder downwards -- `0.0001`, `0.0025`, `0.005`, `0.01`,
`0.02`, `0.03`, `0.04` under one seed.

`double-six` is not incidental: these runs keep the default `lookup-table`
baseline, which needs a packaged format-version-3 reward table, and double-six
is the only ruleset that ships one.

```bash
train_script/run_lr_low_tests_diego_notebook.sh
train_script/run_lr_low_tests_diego_notebook.sh --only '*0p0025*'
```

### Declaring points in the wrapper

A wrapper normally selects one of the point tables built into
`_sequential_rl_experiment_runner.bash` through `EXPERIMENT_KIND`. A wrapper
that tests something those tables do not cover declares its own points
instead:

```bash
EXPERIMENT_KIND="lr_low"                    # names the results directory
EXPERIMENT_PARAMETER_FLAG="--learning-rate" # the flag each value is spent on
EXPERIMENT_POINTS=(
    #  label      value   run name
    "lr_0p0001  0.0001  lr_low_0p0001_${MACHINE_SLUG}"
    "lr_0p0025  0.0025  lr_low_0p0025_${MACHINE_SLUG}"
)
EXPERIMENT_POINTS_DESCRIPTION="learning rates 0.0001 and 0.0025"
```

`EXPERIMENT_POINTS` replaces the built-in table entirely rather than extending
it. `EXPERIMENT_PARAMETER_FLAG` is what lets a brand-new `EXPERIMENT_KIND`
work without editing the runner: the value becomes that flag's argument and
the bundle tail is spelled by `training.run_artifacts.bundle_suffix`, the same
way every built-in experiment spells it. A wrapper whose `EXPERIMENT_KIND` is
already one the runner decodes -- `buckets`, `ppo_lr`, `baselines`,
`one_factor` -- can declare points without it.

## Extending the one-factor sweep

`run_one_factor_ext_tests_<machine>.sh` runs four more `forever` points on
`double-six`, each on the project defaults except for the single parameter it
tests, so all four are directly comparable to the one-factor sweep's
`control`:

| Point | Flag | Why |
|---|---|---|
| `lr_0p0007` | `--learning-rate 0.0007` | bracket the measured optimum |
| `lr_0p0008` | `--learning-rate 0.0008` | bracket the measured optimum |
| `lr_0p0009` | `--learning-rate 0.0009` | bracket the measured optimum |
| `gpi_8000` | `--gpi 8000` | continue a monotone ladder |

The sweep found the learning rate to be its most influential factor, peaking at
`0.001` (+0.677 pp against a +/-0.22 pp ruler). That peak rests on a single run
with no replicate, and its nearest measured neighbours are a factor of four
below and 2.5 above, so nothing yet separates a sharp peak from a broad
plateau. The three rates above bracket it from below at a spacing the ladder
has never had.

The GPI ladder measured 1000, 2000 and 4000 and rose monotonically without a
turning point, so it stopped at the largest value `--gpi` then accepted. `8000`
is the next step, reachable since `--gpi` gained `6000, 8000, 10000, 12000`.

There is one wrapper per machine, each carrying that machine's coefficient so
every point gets the same machine-adjusted budget the sweep gave its 49 runs:

| Machine | Coefficient | Per point | Four points |
|---|---:|---:|---:|
| Diego notebook | 1.0 | 5h00 | ~20h |
| Rick new notebook | 1.5 | 7h30 | ~30h |
| Rick desktop | 2.4 | 12h00 | ~48h |
| Rick old notebook | 3.4 | 17h00 | ~68h |

Running the same four points on more than one machine is the design, not a
duplication: 22 of the sweep's 27 levels had a replicate on a different
machine, and the recommendation these points probe rests on the one rate that
never got one. Prefer `--only` over `--time-limit` when a machine cannot afford
all four -- a shortened budget makes a point incomparable to the same point
elsewhere, while a skipped point costs nothing.

The four points themselves live in `_one_factor_ext_points.bash`, sourced by
every wrapper after it sets `MACHINE_SLUG`. They are shared rather than copied
because these runs are only worth anything as replicates of each other: a point
whose flags drifted on one machine would still run, still produce a bundle, and
silently stop being a replicate.

Each wrapper reuses `EXPERIMENT_KIND="one_factor"` -- as
`run_ppo_lr_ext_tests_diego_notebook.sh` reuses `ppo_lr` -- so the runner
decodes the mixed `lr=` and `gpi=` points without an edit, and the four rows
join that machine's existing one-factor state file rather than opening a second
one. Rows are upserted by run name, so the sweep's own rows are preserved.

```bash
export PYTHON=/path/to/the/project/venv/bin/python
train_script/run_one_factor_ext_tests_diego_notebook.sh --dry-run
train_script/run_one_factor_ext_tests_rick_desktop.sh
train_script/run_one_factor_ext_tests_rick_old_notebook.sh --only '*gpi_8000*'
```

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
