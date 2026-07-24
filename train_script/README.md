# Training scripts

These wrappers compose the training modules documented in
[`training/README.md`](../training/README.md). All reinforcement-learning
entry points use `training.self_play`'s fixed default of 2,000 games per
iteration. GPI is not a pipeline or sweep option. Use
`python -m training.self_play --gpi N` only for a direct, deliberate
self-play experiment.

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

# Historical value-head regression path; this disables PPO.
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
| `--rl-value-head` | Enable the legacy value-head path and disable PPO | off |
| `--rl-ppo` / `--rl-no-ppo` | PPO or one-update REINFORCE | PPO |
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

## RL parameter sweep

`run_rl_parameter_sweep.sh` trains points sequentially. Its grid varies only
learning rate and gamma:

- learning rate: `0.0005`, `0.001`, `0.005`;
- gamma: `1.0`, `0.97`, `0.95`, `0.92`;
- critic off and critic on for each base point;
- value coefficient: `0.25`, `0.5`, `0.75` on a seeded sample of critic-on
  base points.

GPI is fixed at the `training.self_play` default for every point and is not
present in tags, options, or the sweep grid.

```bash
train_script/run_rl_parameter_sweep.sh
train_script/run_rl_parameter_sweep.sh --resume
train_script/run_rl_parameter_sweep.sh --help
```

Every new point starts from `--sl-weights-path`. `--resume` is the only path
that continues RL history: it validates a numbered checkpoint and its paired
resume state, including the fixed GPI stored by self-play. A complete and
compatible diagnostic is reused; stale or incomplete diagnostics are rerun.

Models and diagnostics share a tag:

```text
models/rl_test/domino_rl[_critic]_default_iter002000.npz
models/rl_test/domino_rl[_critic]_lr<LR>_gamma<GAMMA>_iter002000.npz
models/rl_test/domino_rl_critic_<grid-tag>_vc<VC>_iter002000.npz
diagnostics/results/rl_test/domino_rl[_critic]_<same-tag>/
```

Useful options:

| Flag | Meaning | Default |
|---|---|---:|
| `--rl-iterations` | RL iterations per point | `2000` |
| `--sl-weights-path` | Shared supervised checkpoint | `models/domino_sl_weights.npz` |
| `--diagnostic-games` | RL-vs-random games per point | `10000` |
| `--seed` | Training, sampling, and diagnostic seed | `42` |
| `--rl-workers` | Internal rollout workers | `auto` |
| `--vc-sample-count` | Base points sampled for non-baseline value coefficients | `10` |
| `--resume` | Continue compatible points and reuse diagnostics | off |
| `--skip-report` | Skip the final comparison table | off |

Outer sweep parallelism is disabled. Optional RAM limits use a
`systemd-run --user --scope` physical-memory cgroup; optional VRAM limits use
CuPy's allocator limit. Run `--help` for the complete resource controls.

### In-process sweep

`run_rl_parameter_sweep.py` is the lower-overhead counterpart. It loads the
supervised checkpoint once and invokes self-play, diagnostics, and reporting
in one Python process. It uses the same fixed-GPI rule and produces the same
`sweep_run.json` schema.

```bash
python train_script/run_rl_parameter_sweep.py
python train_script/run_rl_parameter_sweep.py --rl-iterations 10 \
  --diagnostic-games 100 --device cpu --quiet-training
```

## Sweep comparison table

`diagnostics.rl_sweep_table` joins each point's `sweep_run.json` and
`summary.json` and writes one row per trained model:

```bash
python -m diagnostics.rl_sweep_table
python -m diagnostics.rl_sweep_table \
  --results-dir diagnostics/results/rl_test \
  --output-dir diagnostics/results/rl_sweep_table
```

Outputs are `rl_sweep_table.csv`, `rl_sweep_table.json`,
`rl_sweep_table.png`, and `rl_sweep_table.pdf`. The table compares critic,
learning rate, gamma, value coefficient, and RL-vs-random win rate. It has no
GPI columns because GPI is not varied.

## Validation

For script-only changes, run at least:

```bash
bash -n train_script/run_training_pipeline.sh
bash -n train_script/run_rl_parameter_sweep.sh
train_script/run_training_pipeline.sh --help
train_script/run_rl_parameter_sweep.sh --help
python train_script/run_rl_parameter_sweep.py --help
python -m diagnostics.rl_sweep_table --help
```

Follow [`CONTRIBUTING.md`](../CONTRIBUTING.md) for Pylint and the complete
impact-matrix checks.
