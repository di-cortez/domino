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
```

The project runs on CPU without CuPy. For NVIDIA GPU training, follow
[`docs/GPU_SETUP.md`](docs/GPU_SETUP.md); dataset workers, rollout workers, and
diagnostic workers intentionally remain CPU-only.

## Run the visual simulator

```bash
source .venv/bin/activate
python -m ui.visual_main
```

The menu (`M`) can assign `Neural`, `Random NN`, `Heuristic`, `Random`,
`Human`, or `RL (self-play)` to either player. Neural and RL selections need
the corresponding files in `models/`. See [`ui/README.md`](ui/README.md) for
all controls and interaction rules.

## Train the agents

Run the complete dataset -> supervised learning -> RL -> diagnostics pipeline:

```bash
python run_pipeline.py
```

By default the RL stage ignores an older `models/domino_rl_weights.npz` and
starts from the supervised checkpoint produced by that same pipeline run. Use
`python run_pipeline.py --continue-existing-rl` only when you intentionally
want to continue the existing RL checkpoint.

Supervised epoch counts are maximum budgets. Training stops earlier by default
after a conservative repeated-block check confirms that training loss has
saturated; use `--sl-no-training-plateau-stop` for fixed-epoch experiments.

The optional scale is `small`, `default`, `big`, or `huge`:

```bash
python run_pipeline.py small
python run_pipeline.py --help
```

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

The long RL parameter sweep and the controlled games-per-iteration study have
safe resume and reporting commands documented in
[`train_script/README.md`](train_script/README.md). Inspect a GPI plan without
running games:

```bash
python train_script/run_rl_games_per_iteration_sweep.py \
  --preset standard --dry-run
```

Choose a custom sweep size with `--total-training-games`, and choose the tested
batch sizes with `--games-per-iteration-values`:

```bash
python train_script/run_rl_games_per_iteration_sweep.py \
  --total-training-games 384000 \
  --games-per-iteration-values 40 80 160 320 640 960 1280 \
  --seeds 42 43 44 \
  --run-id gpi_custom
```

The total must be exactly divisible by every selected games-per-iteration
value. Use `--diagnostic-games N` separately to set final evaluation games per
opponent.

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

Canonical diagnostic agent names are `rl`, `neural`, `random_nn`, `heuristic`,
and `random`. Detailed output schemas and interpretation guidance live in
[`diagnostics/README.md`](diagnostics/README.md).

## Generated artifacts

Generated datasets, models, and reports are ignored by Git. Important default
locations are:

| Path | Contents |
|---|---|
| `dataset/supervised_dataset.jsonl` | Heuristic-labelled real decisions. |
| `dataset/supervised_dataset_encoded.npz` | Encoded supervised cache. |
| `models/domino_sl_weights.npz` | Supervised policy checkpoint. |
| `models/domino_sl_loss.png` | Training and validation loss curves from the latest supervised run. |
| `models/domino_rl_weights.npz` | Self-play policy checkpoint. |
| `models/rl_test/` | Numbered parameter-sweep checkpoints and resume state. |
| `models/rl_gpi_sweep/` | Games-per-iteration sweep checkpoints and manifests. |
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
  train_script run_pipeline.py
```

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
| `utils/` | Resource reporting and exact-update timing helpers. |
| `tests/` | Core, parallelism, sweep, and regression tests. |

## Documentation

Start with [`docs/README.md`](docs/README.md), which indexes architecture,
setup, contribution rules, and every module README. In particular:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) explains boundaries and data
  flow;
- [`docs/GPU_SETUP.md`](docs/GPU_SETUP.md) covers CUDA/CuPy installation and
  troubleshooting;
- [`CONTRIBUTING.md`](CONTRIBUTING.md) defines compatibility, determinism,
  testing, generated-file, and documentation requirements;
- [`AGENTS.md`](AGENTS.md) gives short instructions for coding agents.
