# Domino: Neural vs. Heuristic

A two-player domino research project. It pairs a rules engine and an exact
public-information opponent model with four kinds of player -- random, a
handcrafted heuristic, a supervised neural policy, and a self-play
reinforcement-learning policy -- and with the pipeline, experiment scripts and
diagnostics that train and measure them. A Pygame/OpenGL simulator lets any
pair of players, humans included, face each other.

All repository content is maintained in English. Read
[`CONTRIBUTING.md`](CONTRIBUTING.md) before changing code, commands, generated
artifacts, or documentation. Every directory owns a README with its complete
options and behavior; this page is the overview and points to them.

## The model

The canonical run trains one policy in three stages and monitors it throughout:

1. **Dataset.** The heuristic `StrategicAgent` plays itself; its real
   decisions become supervised examples
   ([`training/datagen/`](training/datagen/README.md)).
2. **Supervised pretraining.** An MLP learns to imitate those decisions, with a
   5,000-epoch budget that a conservative plateau rule normally ends earlier
   ([`training/supervised/`](training/supervised/README.md)).
3. **Self-play RL.** The pretrained policy is refined with masked PPO against an
   opponent pool, with exact checkpoints and resume
   ([`training/rl/`](training/rl/README.md)).
4. **Diagnostics.** The policy plays the random baseline periodically during
   training, and finite levels end with an all-pairs evaluation of every agent
   ([`diagnostics/`](diagnostics/README.md)).

Current defaults of a new run on the default `double-six` ruleset:

| Area | Default |
|---|---|
| Network | 168 inputs, hidden layers 256 and 128, 56 tile-play actions ([`agents/`](agents/README.md)) |
| Update | Masked PPO: clip 0.2, 512-decision minibatches, whole-buffer KL stop at 0.015; up to 16 epochs per buffer for `forever`, 4 otherwise |
| Iteration | 8,000 games per iteration (`--gpi`), learning rate 0.001, entropy coefficient 0 |
| Advantage | `batch-mean` baseline with whole-buffer normalization; no critic unless `--value-head` |
| Reward | `G = (1 - eta) G_terminal + eta G_immediate` with `eta` 0.115: terminal ±1 for an empty hand or ±m(pip margin) for a blocked game, discounted by `gamma_f` 0.95; draw/pass events discounted by `gamma_i` 0.90; both clocks count engine turns (`turn-turn`) |
| Opponents | The `random` bucket; heuristic, recent, medium-term, historical and champion buckets are optional |
| Monitoring | 100,000 RL-vs-random games every 100,000 RL games |
| Optional | KL-gated learning-rate warmup (`--warmup-lr`), background monitors (`--async-periodic-diagnostics`) |

Games always produce one winner: an empty hand wins first, and a blocked game
goes to the fewest pips, then the fewest tiles, then the most recent valid
play. The reward model, its weights and distance modes, the baselines and
critics, the warmup and every PPO control are documented in
[`training/rl/README.md`](training/rl/README.md) and
[`training/README.md`](training/README.md).

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

Everything runs on CPU without CuPy. For NVIDIA GPU training follow
[`docs/GPU_SETUP.md`](docs/GPU_SETUP.md); dataset, rollout and diagnostic
workers stay CPU-only by design.

## Play

```bash
python -m ui.visual_main
python -m ui.visual_main --ruleset double-four
```

The menu (`M`) assigns `Neural`, `Heuristic`, `Random`, `Human`, or
`RL (self-play)` to either seat; the neural choices need trained weights in
`models/`. Controls are in [`ui/README.md`](ui/README.md).

## Train

`python -m training.pipeline <level>` runs the whole sequence.
`python -m train_script.run_pipeline` is an equivalent entry point.

| Level | Dataset games | Seed and supervised assets | RL games | Resume |
|---|---:|---|---:|---|
| `small` | 10,000 | Fresh seed, run-local | 100,000 | No |
| `default` | 50,000 | Fresh seed, run-local | 500,000 | No |
| `big` | 100,000 | Seed 52, reused when compatible | 2,000,000 | Yes |
| `huge` | 100,000 | Seed 52, reused when compatible | 10,000,000 | Yes |
| `forever` | 100,000 | Seed 52, reused when compatible | Unbounded | Yes |

```bash
python -m training.pipeline small                    # quick end-to-end check
python -m training.pipeline forever --run-name baseline
python -m training.pipeline forever                  # later starts reload the active run
python -m training.pipeline forever --resume models/rl/domino_rl_forever_seed52_runbaseline
python -m training.pipeline --help
```

A long run locks its complete configuration on its first start. Resuming
reloads it; a conflicting flag is warned about and ignored, so a different
experiment needs a new `--run-name`. The first SIGINT/SIGTERM finishes the
current iteration, publishes a safe checkpoint and exits.

Each stage also runs on its own, which is faster when iterating on one
component:

```bash
python -m training.datagen.generator --workers auto --seed 123
python -m training.supervised.cli --sl-device auto --sl-seed 123
python -m training.rl.cli --fresh-from-sl --rl-workers auto --seed 123
```

Every entry point that produces games accepts `--ruleset` with `double-six`
(the default), `double-five`, `double-four`, or `double-three`. Network sizes
shrink with the deck, and datasets, checkpoints and runs are never shared
across rulesets; see the [ruleset contract](training/README.md#ruleset-contract).

## Evaluate

```bash
python -m diagnostics.evaluate --games 5000 --workers auto --seed 123
python -m diagnostics.pairwise --agent heuristic --opponent random \
  --games 1000 --seed 123
```

Agent names are `rl`, `neural`, `heuristic`, and `random`. Output schemas,
plots, statistics and the RL progress monitor are described in
[`diagnostics/README.md`](diagnostics/README.md).

## Experiments

Long comparisons run as wall-clock-budgeted sequences of `forever` runs, one
point at a time, with graceful stops and exact resume. The shared runners and
the experiments built on them -- reward-distance grid, opponent buckets,
baselines, the one-factor sweep and its extension, and the adaptive
learning-rate search -- are documented in
[`train_script/README.md`](train_script/README.md). Per-machine wrappers
(`run_*_<machine>.sh`) carry each machine's time budget; many are kept out of
Git and shared directly.

Each study of finished runs lives in its own `analysis/<study>/` directory with
a README describing its question, inputs, commands and outputs, for example
[`analysis/analise_warmup_lr/`](analysis/analise_warmup_lr/README.md) and
[`analysis/rl_performance_optimization/`](analysis/rl_performance_optimization/README.md).

## Generated artifacts

Datasets, weights, runs and reports are generated and ignored by Git. Do not
edit them by hand or delete them casually: long runs depend on their numbered
checkpoints and resume state.

| Path | Contents |
|---|---|
| `dataset/supervised_dataset_standard_seed<seed>.jsonl` | Reusable heuristic-labelled decisions, with sibling metadata and random manifest ([`dataset/`](dataset/README.md)) |
| `models/domino_sl_standard_seed<seed>.npz` | Reusable supervised policy, with metadata, random manifest and loss plot ([`models/`](models/README.md)) |
| `models/rl/domino_rl_<level>_seed<seed>[_run<name>]/` | One RL run: resume marker and state, checkpoints, opponent pool, archive |
| `models/rl/.../<date>-<number>_<machine>_<tail>/` | The run's shareable analysis bundle: `run_config.json`, periodic history, progress CSV and plots |
| `diagnostics/results/` | Evaluation and pairwise reports |
| `train_script/grid_search_results/<machine>/` | Sequence state, per-attempt logs and experiment summaries |

The bundle is named after the run's start date, its number in the experiment
log shared across machines (`XXX` until filled in, or `--run-ordinal`), the
machine, and the parameters it moves off the defaults; see
[`training/README.md`](training/README.md).

## Tests

```bash
python -m pytest -q
python -m pylint agents benchmarks diagnostics middleware tests train_script \
  training ui utils
python benchmarks/headless_step_benchmark.py --games 100
```

Name the Pylint directories explicitly: `pylint .` would also scan the virtual
environment. The required checks for each kind of change are in the impact
matrix of [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Repository map

| Path | Responsibility | README |
|---|---|---|
| `middleware/` | Rules engine, rulesets, agent protocol, game orchestration, exact opponent inference | [middleware](middleware/README.md) |
| `agents/` | State/action encoding, network backends and every player | [agents](agents/README.md) |
| `training/` | Pipeline levels, run configuration, bundles and stage entry points | [training](training/README.md) |
| `training/datagen/` | Supervised dataset generation | [datagen](training/datagen/README.md) |
| `training/supervised/` | Supervised training loop, scheduler and architecture | [supervised](training/supervised/README.md) |
| `training/rl/` | Self-play RL: PPO, rewards, baselines, warmup, opponent pool, checkpoints, resume | [rl](training/rl/README.md) |
| `training/utils/` | Seed derivation, shared regularization, encoded-feature contract | [training utils](training/utils/README.md) |
| `diagnostics/` | Evaluation, RL progress monitor, reports and plots | [diagnostics](diagnostics/README.md) |
| `train_script/` | Pipeline wrappers and experiment sequence runners | [train_script](train_script/README.md) |
| `analysis/` | One directory per study of finished runs | one README per study |
| `ui/` | Visual simulator, controls, layout and rendering | [ui](ui/README.md) |
| `utils/` | Resource limits, machine identity, runtime status, atomic artifacts, central randomness | [myrandom](utils/myrandom/README.md) |
| `dataset/`, `models/` | Generated datasets and weights | [dataset](dataset/README.md), [models](models/README.md) |
| `tests/` | Unit, parallelism, pipeline and regression tests | -- |
| `benchmarks/` | Headless engine throughput and fixed-seed equivalence | -- |
| `docs/` | Architecture, GPU setup and the Pylint roadmap | [docs](docs/README.md) |

## Documentation

Start with [`docs/README.md`](docs/README.md), which indexes every guide and
module README. In particular:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): boundaries and data flow;
- [`docs/GPU_SETUP.md`](docs/GPU_SETUP.md): CUDA/CuPy installation and
  troubleshooting;
- [`docs/PYLINT_ROADMAP.md`](docs/PYLINT_ROADMAP.md): the Pylint baseline and
  staged ratchet;
- [`CONTRIBUTING.md`](CONTRIBUTING.md): compatibility, determinism, testing,
  generated files and documentation ownership;
- [`AGENTS.md`](AGENTS.md): short instructions for coding agents.
