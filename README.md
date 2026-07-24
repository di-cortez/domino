# Domino: Neural vs. Heuristic

Two-player domino simulator with a Pygame/OpenGL interface, an exact
public-information opponent model, heuristic and neural agents, supervised
training, self-play reinforcement learning, and reproducible diagnostics.

All repository content is maintained in English. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) before changing code, commands, generated
artifacts, or documentation.

## Quick setup

Python 3.10 or newer is recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pygame PyOpenGL PyOpenGL-accelerate \
  matplotlib tqdm openpyxl pytest
python -m pip install -r requirements-dev.txt
```

The project runs on CPU without CuPy. For NVIDIA GPU training, follow
[`docs/GPU_SETUP.md`](docs/GPU_SETUP.md); dataset workers, rollout workers, and
diagnostic workers intentionally remain CPU-only.

## Run the visual simulator

```bash
source .venv/bin/activate
python -m ui.visual_main
```

The menu (`M`) can assign `Neural`, `Heuristic`, `Random`, `Human`, or
`RL (self-play)` to either player. Neural and RL selections need
the corresponding files in `models/`. See [`ui/README.md`](ui/README.md) for
all controls and interaction rules.

## Train the agents

Run the canonical dataset -> supervised learning -> RL -> diagnostics pipeline:

```bash
python -m training.pipeline default
# Equivalent compatibility entry point:
python -m train_script.run_pipeline default
```

`small` and `default` are isolated quick-run profiles. Without `--seed`, each
invocation chooses a fresh seed; their dataset and supervised checkpoint live
inside a unique RL run directory and are never reused. An explicit `--seed`
still enables a reproducible experiment, but the artifact namespace remains
new. Supervised training keeps the same maximum of 5,000 epochs and the
existing conservative early-stop rules.

`big`, `huge`, and `forever` are the reusable long-run profiles. They default
to seed 42 and share the compatible 100,000-game seed-addressed standard
dataset and supervised checkpoint. Metadata and SHA-256 control reuse;
incompatible files are refused unless `--rebuild-dataset`,
`--retrain-supervised`, or `--rebuild-supervised-assets` is explicit.

Pipeline levels differ primarily in exact cumulative RL games:

| Level | Dataset games | Default seed/assets | RL games | Final games/matchup | Periodic RL-vs-random |
|---|---:|---|---:|---:|---|
| `small` | 10,000 | Random, run-local | 100,000 | 10,000 | No |
| `default` | 50,000 | Random, run-local | 500,000 | 10,000 | No |
| `big` | 100,000 | 42, reusable | 2,000,000 | 1,000,000 | Every 100,000 games |
| `huge` | 100,000 | 42, reusable | 10,000,000 | 1,000,000 | Every 100,000 games |
| `forever` | 100,000 | 42, reusable | No limit | None automatically | Every 100,000 games |

RL uses a fixed GPI of 2,000. Only direct `training.self_play` experiments can
override it with `--gpi`; pipelines and sweeps do not expose GPI as a tuning
axis. Before real games begin, an isolated benchmark selects the rollout-worker
count and discards its games. Training uses masked PPO with adaptive
minibatches. Direct self-play and the finite canonical profiles retain the
four-epoch default; `forever` now allows up to 16 epochs. After each complete
epoch, a whole-buffer KL check stops the update before the next epoch when its
hard `0.015` limit is exceeded. Pass `--no-ppo` to use one
full-buffer REINFORCE update per iteration instead; that path does not build a
PPO buffer or calculate ratios, clipping, KL control, minibatches, or the
post-update full-buffer PPO evaluation. Opponent snapshots refresh every 400
cumulative real games, and checkpoint saves do not run an extra evaluation
matchup.

`big`, `huge`, and `forever` persist weights, optimizer, RNGs, counters, and
the opponent pool. Continue the same cumulative target with `--resume`, or
extend a lineage with `--resume-from`:

```bash
python -m training.pipeline big --resume
python -m training.pipeline huge \
  --resume-from models/rl/domino_rl_big_seed42
python -m training.pipeline forever --resume
python -m training.pipeline forever \
  --resume models/rl/domino_rl_forever_seed42
```

Start and later resume an unbounded policy-only REINFORCE run with:

```bash
python -m training.pipeline forever --no-ppo
python -m training.pipeline forever --no-ppo --resume
```

The algorithm is part of the exact resume identity. Repeat `--no-ppo` on every
resume command; a `reinforce_v1` run cannot be resumed as `ppo_v1`, or vice
versa. Canonical runs remain policy-only in both modes, so `--value-head` stays
limited to direct self-play experiments.

`--resume` without a value uses the level/seed default directory. For
convenience, `--resume RUN_DIR` is also accepted and is equivalent to
`--resume-from RUN_DIR`.

The `forever` periodic RL-vs-random worker autotune runs once. Its selection is
stored in `periodic_diagnostic_tuning.json` and reused at every later milestone
and after resume. The RL progress bar reports one `avg_games_s` value computed
over the full persisted RL training lineage.

The first SIGINT/SIGTERM finishes the current iteration, publishes a safe
checkpoint, and exits; `forever` never launches the final all-pairs diagnostic.

Supervised epoch counts are maximum budgets. Training stops earlier by default
after a conservative repeated-block check confirms that training loss has
saturated; use `--sl-no-training-plateau-stop` for fixed-epoch experiments.

Use `python -m training.pipeline --help` for rebuild, resume, worker, device,
PPO, and diagnostic controls.

Run stages directly when iterating on one component:

```bash
python -m training.dataset_generator --workers auto --seed 123
python -m training.training_loop --sl-device auto --sl-seed 123
python -m training.self_play --rl-workers auto --seed 123
python -m training.self_play --fresh-from-sl --rl-workers auto --seed 123
python -m diagnostics.evaluate --games 10000 --seed 123
```

The standalone self-play command preserves its historical default of
continuing a compatible RL checkpoint when one exists; `--fresh-from-sl`
forces a new RL run from the supervised checkpoint.

The long RL parameter sweep has safe resume and reporting commands documented
in [`train_script/README.md`](train_script/README.md). It varies learning rate,
gamma, and the critic value coefficient while retaining the fixed GPI.

## Diagnostics

Evaluate all supported agents against the common random baseline:

```bash
python -m diagnostics.evaluate
python -m diagnostics.evaluate --games 5000 --workers auto --seed 123
```

Evaluate one matchup:

```bash
python -m diagnostics.pairwise \
  --agent heuristic --opponent random --games 1000 --seed 123
```

Canonical diagnostic agent names are `rl`, `neural`, `heuristic`, and
`random`. Detailed output schemas and interpretation guidance live in
[`diagnostics/README.md`](diagnostics/README.md).

## Generated artifacts

Generated datasets, models, and reports are ignored by Git. Important default
locations are:

| Path | Contents |
|---|---|
| `dataset/supervised_dataset_standard_seed42.jsonl` | Canonical heuristic-labelled real decisions for seed 42. |
| `dataset/supervised_dataset_standard_seed42.meta.json` | Dataset identity, provenance, and SHA-256. |
| `models/domino_sl_standard_seed42.npz` | Canonical supervised policy. |
| `models/domino_sl_standard_seed42.meta.json` | Supervised origin, configuration, convergence, and SHA-256. |
| `models/domino_sl_standard_seed42_loss.png` | Canonical training and validation loss curves. |
| `models/rl/domino_rl_<small-or-default>_seed<seed>_run<id>/supervised/` | Non-reused dataset, cache, supervised checkpoint, metadata, and loss plot for one quick run. |
| `models/rl/domino_rl_<level>_seed42/` | Complete RL state, milestones, diagnostics, and progress curve. |
| `models/rl_test/` | Numbered parameter-sweep checkpoints and resume state. |
| `diagnostics/results/` | Pairwise, aggregate, sweep, CSV, JSON, XLSX, and plot outputs. |

Do not commit, manually edit, or casually delete generated artifacts. Long
experiments may depend on their numbered checkpoints and `.resume.npz` state.

## Tests

Run the complete suite:

```bash
python -m pytest -q
```

Useful focused checks:

```bash
python tests/test_core.py
python tests/test_parallel_dataset.py
python tests/test_parallel_diagnostics.py
python tests/test_parallel_rl.py
python ui/test_ui_controller.py
python -m compileall -q agents diagnostics middleware training ui utils \
  train_script
```

Pylint is required after every modification and currently reports without
blocking. See [`CONTRIBUTING.md`](CONTRIBUTING.md) and the staged
[`Pylint roadmap`](docs/PYLINT_ROADMAP.md).

The headless benchmark verifies both fixed-seed equivalence and throughput:

```bash
python benchmarks/headless_step_benchmark.py --games 100
```

## Repository map

| Path | Responsibility |
|---|---|
| `middleware/` | Rules engine, agent protocol, orchestration, exact opponent inference. |
| `agents/` | State/action encoding and all gameplay policies. |
| `training/` | Dataset generation, supervised training, RL, checkpoints, resume. |
| `diagnostics/` | Evaluation, metrics, reports, plots, and experiment analysis. |
| `ui/` | Visual simulator, controls, layout, rendering, and controller tests. |
| `train_script/` | Reproducible pipeline and sweep entry points. |
| `utils/` | Resource limits, runtime status, and atomic artifact helpers. |
| `tests/` | Core, parallelism, sweep, and regression tests. |

## Documentation

Start with [`docs/README.md`](docs/README.md), which indexes architecture,
setup, contribution rules, and every module README. In particular:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) explains boundaries and data
  flow;
- [`docs/GPU_SETUP.md`](docs/GPU_SETUP.md) covers CUDA/CuPy installation and
  troubleshooting;
- [`docs/PYLINT_ROADMAP.md`](docs/PYLINT_ROADMAP.md) records the permissive
  baseline and staged quality ratchet;
- [`CONTRIBUTING.md`](CONTRIBUTING.md) defines compatibility, determinism,
  testing, generated-file, and documentation requirements;
- [`AGENTS.md`](AGENTS.md) gives short instructions for coding agents.
