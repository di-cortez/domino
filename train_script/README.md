# train_script

Personal batch-training driver for the full pipeline described in
`training/README.md` and the top-level `README.md`:

1. generate supervised examples from heuristic-vs-heuristic games, using the
   `training.dataset_generator` module defaults, including retained automatic
   dataset-worker tuning;
2. train the supervised neural policy with its default retained batch tuner and
   plateau scheduler; the wrapper exposes device, memory, batch, seed, decay,
   early-stopping, and weight-decay controls;
3. refine that policy with a **BIG-scale** self-play reinforcement-learning
   run — 5x the default iteration count (1,000 x 5 = 5,000 iterations),
   matching the `big` scale in `run_pipeline.py`'s `SCALE_FACTORS`. This
   stage is fully parameterized from the command line so the script can drive
   repeated batch runs that only vary RL hyperparameters;
4. run the **five agent-vs-random diagnostics** (`diagnostics.evaluate`) —
   evaluating the exact
   RL/SL checkpoints this run used, at the same BIG scale as the RL stage
   (`run_pipeline.py` maps its `big` scale to `diagnostic_mode="complete"`
   and scales `BASE_DIAGNOSTIC_GAMES=10000` by the scale factor; this script
   mirrors that: `10,000 x 5 = 50,000` games/matchup, mode `complete`, by
   default). Results are written to a **fresh subdirectory per run**,
   `diagnostics/results/<rl-weights-basename>/`, named after the RL weights
   file this run produced or reused (e.g. `--rl-weights-file
   models/domino_rl_weights_critic_lr_0.0005.npz` writes to
   `diagnostics/results/domino_rl_weights_critic_lr_0.0005/`), so repeated
   batch runs that vary RL hyperparameters keep separate diagnostics output
   instead of overwriting a shared directory.

Stage 1 (dataset generation) is left unparameterized by this wrapper, so it
uses the standalone command's 30,000-game default and automatic worker tuner.
The Python module itself accepts `--games`, `--output`, `--workers`, `--seed`,
and memory-safety options when called directly. Stage 2 (supervised training)
forwards the controls below to `training.training_loop`; LR decay is enabled by
default with factor `0.5` and patience `5`. Use
`--skip-dataset`/`--skip-sl`/`--skip-rl`/`--skip-diagnostics` to reuse existing
artifacts across batch runs that only sweep RL hyperparameters.

## Convergence criteria (from the archived test reports)

`references/explicacoes/relatorios/{teste_1,teste_2,teste_3,relatorio_1407}`
document an earlier round of pipeline experiments that diagnosed why RL
training looked stagnant and established how to actually tell: **point-in-time
values are dominated by batch noise; judge convergence from the moving
average, not the raw log line.** Those reports' validated conclusions:

- **SL**: stop on the validation curve, not a fixed epoch budget (early
  stopping), and decay the learning rate on a validation plateau instead of
  holding it fixed. LR decay and early stopping have independent counters. The
  script forwards `--sl-early-stopping-patience`, `--sl-lr-decay-factor`,
  `--sl-lr-decay-patience`, `--sl-no-lr-decay`, and `--sl-weight-decay`.
- **RL**: log the return standard deviation next to the mean (a shrinking
  value loss is ambiguous without it — it can mean either learning or just a
  low-variance batch), and use a trailing moving average of value loss and
  win rate, not the point value, to judge a plateau. `training.self_play` now
  always logs `reward mean/std/min/max` and a `(avg/N: ...)` moving average
  next to both the value loss and the win rate; `--rl-moving-average-window`
  controls `N` (default 10, matching the reports). `--rl-clip-grad-norm` and
  `--rl-value-coef` were flagged as needing CLI exposure for any serious
  investigation of the value-loss plateau — both are now exposed.
  `--rl-normalize-advantages` (off by default, to not silently change
  training dynamics) standardizes the policy signal per batch, which the
  reports found necessary to keep the value-loss magnitude comparable across
  iterations. `--rl-seed` fixes randomness for reproducible side-by-side
  comparisons between hyperparameter configurations, per the reports'
  recommended sweep methodology.

One historical fix did **not** carry forward as a flag: the old
`SL_CHECKPOINT_EVERY == SL_EPOCHS` bug (which silenced all but one archived
SL checkpoint) is structurally impossible in the current
`training/training_loop.py` — its checkpoint interval is a fixed internal
constant, not a CLI-configurable value coupled to the epoch budget — so there
was nothing to wire through.

## Usage

From the repository root, with the project virtual environment set up as
described in the top-level `README.md`:

```bash
train_script/run_training_pipeline.sh
```

The script activates `.venv` automatically if it exists at the repository
root, then runs the three stages in order, stopping immediately if any stage
fails (`set -euo pipefail`).

Show every available option:

```bash
train_script/run_training_pipeline.sh --help
```

## Examples

Full pipeline with defaults (fresh dataset, fresh SL weights, BIG-scale RL):

```bash
train_script/run_training_pipeline.sh
```

Batch run that only varies RL hyperparameters, reusing an already-trained
supervised checkpoint:

```bash
train_script/run_training_pipeline.sh --skip-dataset --skip-sl \
    --rl-learning-rate 0.0005 --rl-gamma 0.97 --rl-reward-schema shaped \
    --rl-weights-file models/domino_rl_weights_lr0005_gamma097_shaped.npz
```

Same, with the critic (learned `V(s)` baseline) turned on:

```bash
train_script/run_training_pipeline.sh --skip-dataset --skip-sl \
    --rl-value-head --rl-weights-file models/domino_rl_weights_critic.npz
```

Quick smoke test of the RL stage alone:

```bash
train_script/run_training_pipeline.sh --skip-dataset --skip-sl \
    --rl-iterations 10 --rl-games-per-iteration 4 --rl-checkpoint-interval 5 \
    --rl-weights-file models/smoke_test.npz
```

## Options

Every RL flag forwards directly to `python -m training.self_play`, which also
accepts these same flags (see `training/self_play.py:add_optional_rl_arguments`).
This wrapper does not expose dataset-generation controls; that stage uses the
documented module defaults. Supervised controls map directly to the standalone
training module.

| Flag | Stage | Meaning | Default |
|---|---|---|---|
| `--rl-weights-file` | RL | Output weights path | `models/domino_rl_weights.npz` |
| `--rl-sl-weights-path` | RL | Input SL weights used to initialize a fresh RL run | `models/domino_sl_weights.npz` |
| `--rl-iterations` | RL | Training iterations | `5000` (BIG scale: `1000 x 5`) |
| `--rl-games-per-iteration` | RL | Games played per iteration | `40` |
| `--rl-training-opponent` | RL | `self_play` or `heuristic` | `self_play` |
| `--rl-learning-rate` | RL | Learning rate | `0.001` |
| `--rl-entropy-coef` | RL | Entropy bonus coefficient | `0.01` |
| `--rl-log-interval` | RL | Iterations between log lines | `10` |
| `--rl-checkpoint-interval` | RL | Iterations between checkpoints | `50` |
| `--rl-pool-interval` | RL | Iterations between self-play pool snapshots | `10` |
| `--rl-max-pool-size` | RL | Max frozen snapshots kept in the pool | `50` |
| `--rl-evaluation-games` | RL | Games per checkpoint evaluation | `200` |
| `--rl-value-head` | RL | Turn the critic (learned `V(s)` baseline) ON | off (direct REINFORCE) |
| `--rl-value-coef` | RL | Value-loss coefficient (only used with `--rl-value-head`) | `0.5` |
| `--rl-gamma` | RL | Terminal-reward discount per remaining real decision; `1.0` disables | `1.0` |
| `--rl-reward-schema` | RL | `default`, `sparse`, or `shaped` reward preset | `default` |
| `--rl-clip-grad-norm` | RL | Gradient-norm clipping threshold | `5.0` |
| `--rl-normalize-advantages` | RL | Standardize the policy signal per batch before the gradient step | off |
| `--rl-no-normalize-advantages` | RL | Explicitly keep advantage normalization off | (default) |
| `--rl-moving-average-window` | RL | Trailing-iteration window for value-loss/win-rate moving averages in the log | `10` |
| `--rl-seed` | RL | Fix random/numpy state for reproducible comparisons | unset |
| `--rl-device` | RL | Array backend: `auto`/`cpu`/`gpu` (see below) | `auto` |
| `--rl-workers` | RL | CPU-only rollout workers or retained automatic tuning, maximum 20 | `auto` |
| `--rl-autotune-fraction` | RL | Fraction of complete iterations retained per worker test | `0.01` |
| `--rl-autotune-min-gain` | RL | Required marginal rollout-throughput improvement | `0.10` |
| `--rl-memory-reserve-mb` | RL | Host RAM kept free while workers run | `512` |
| `--rl-estimated-worker-mb` | RL | Preflight memory estimate per worker | `256` |
| `--rl-max-worker-rss-mb` | RL | Runtime RSS ceiling for one worker | `1024` |
| `--sl-early-stopping-patience` | SL | Validation checks (every 10 epochs) without improvement before stopping | unset (off) |
| `--sl-lr-decay-factor` | SL | LR multiplier after a validation plateau | `0.5` |
| `--sl-lr-decay-patience` | SL | Consecutive failed validation checks before each LR reduction | `5` |
| `--sl-no-lr-decay` | SL | Disable the default plateau scheduler | off |
| `--sl-weight-decay` | SL | L2 penalty on the weight matrices | unset (off) |
| `--sl-device` | SL | `auto`, forced `cpu`, or required `gpu` | `auto` |
| `--sl-batch-size` | SL | Fixed mini-batch size; bypasses tuning | unset |
| `--sl-no-batch-autotune` | SL | Use the device default batch (CPU 1,024; GPU 2,048) | off |
| `--sl-memory-reserve-mb` | SL | Host RAM kept free | `512` |
| `--sl-gpu-memory-reserve-mb` | SL | Effective VRAM kept free | `512` |
| `--sl-seed` | SL | Fix initialization and epoch permutations | unset |
| `--diag-mode` | Diagnostics | Compatibility label; every value runs the same 5 agent-vs-random matchups | `complete` (BIG scale) |
| `--diag-games` | Diagnostics | Games per evaluated matchup | `50000` (BIG scale: `10000 x 5`) |
| `--diag-seed` | Diagnostics | Fix the RNG seed for the diagnostics games | unset |
| `--diag-no-pair-plots` | Diagnostics | Skip per-matchup PNG plots (the aggregate PNG and PDF are still generated) | off (plots on) |
| `--diag-output-dir` | Diagnostics | Override the output directory | `diagnostics/results/<rl-weights-basename>/` |
| `--skip-dataset` | control | Skip dataset generation (reuse an existing dataset file) | off |
| `--skip-sl` | control | Skip supervised training (reuse an existing SL weights file) | off |
| `--skip-rl` | control | Skip self-play reinforcement learning | off |
| `--skip-diagnostics` | control | Skip the agent-vs-random diagnostics stage | off |

### The critic (value-head) toggle

`--rl-value-head` is the same switch as `training.self_play --value-head`
(and `run_pipeline.py --value-head`): direct REINFORCE (no critic) is the
default; passing the flag adds a linear `V(s)` baseline and trains the policy
from `reward - V(s)` advantages instead of raw rewards. `--rl-value-coef`
only has an effect when the critic is on.

### Reward schema and gamma

`--rl-reward-schema` selects a named preset of the terminal/event reward
constants in `training/self_play.py` (`REWARD_SCHEMAS`): `default`
reproduces the original fixed constants, `sparse` zeroes out all draw/pass
shaping and the pip penalty (terminal win/tie/loss only), and `shaped`
doubles the draw/pass shaping rewards. `--rl-gamma` discounts the terminal
reward per remaining real decision in the trajectory (`gamma ** remaining`);
`1.0` reproduces the original undiscounted behavior.

This was verified end-to-end with a tiny run (`--rl-iterations 2
--rl-games-per-iteration 2`) exercising the RL stage with `--skip-dataset
--skip-sl`, both with and without `--rl-value-head`.

### RL rollout workers

`--rl-workers auto` benchmarks 1, 2, 4, 6, ... CPU-only workers over complete
early iterations. Each candidate retains about 1% of the planned iterations,
and every game contributes to the normal parent-side gradient update. Testing
stops below 10% marginal gain, on a resource guard, or at the hard limit of 20.
Use a fixed value to skip tuning.

Workers read the current policy and bounded opponent pool through shared memory;
they never update weights or access the GPU. The main process sorts trajectories
by game id, assembles the batch, updates the network, and writes checkpoints.
Stable seeds make one-worker and multi-worker runs bit-identical for the same
configuration. Runtime pressure retains completed games and retries unfinished
ones with half as many workers.

### Device selection (`--rl-device`)

`--rl-device auto` (the default) uses CuPy when available and at least 256 MiB
of effective VRAM is free; otherwise it announces a safe NumPy/CPU fallback.
`--rl-device cpu` forces host execution. `--rl-device gpu` requires a usable
GPU and fails before training when VRAM is below the safety threshold, rather
than risking an out-of-memory failure mid-batch. These controls do not change
supervised training's own backend selection. Worth trying `--rl-device cpu`
if RL training feels slow: profiling shows RL self-play is dominated by the
exact opponent-hand inference in `middleware/opponent_model.py` (>80% of
iteration time), not the policy network, so CuPy's per-decision transfer
overhead during rollout can make GPU measurably slower than CPU for this stage
specifically. Verified end-to-end with a tiny run (`--rl-device cpu`,
`--rl-iterations 3 --rl-games-per-iteration 4`): the run logged `RL self-play
array backend: numpy (device='cpu')` and completed correctly.

For Linux driver installation, the correct CUDA 12.x/13.x CuPy wheel, the
recommended `[ctk]` installation, real GPU calculation tests, and common error
diagnosis, follow **Linux GPU setup and verification** in the root README before
starting a long batch. The pipeline startup line reports both training backends
and current free/total RAM and VRAM before any workload begins.

### Supervised retained batch tuning and storage

`--sl-device auto` independently selects the supervised backend. It uses GPU
only after the 512 MiB VRAM preflight and a no-update dataset-residency probe;
automatic mode falls back to CPU, while explicit `gpu` fails before training.
The tuner tests CPU batches from 1,024 or GPU batches from 2,048, doubling up to
1,048,576 or the training-set size. Ten complete epochs per candidate update
the live network and count toward progress. Selection uses synchronized median
examples/second and requires at least 10% improvement.

When safe, encoded arrays stay in host RAM; otherwise the training module uses
an atomic disk-backed mmap cache. GPU runs upload the complete dataset once if
it fits safely, or rotate global-permutation windows through reusable buffers.
Use `--sl-batch-size N` for a fixed batch, or `--sl-no-batch-autotune` for the
device default. Detailed standalone logs show all decisions. `run_pipeline.py`
keeps per-epoch/checkpoint details suppressed, but prints the concise retained
batch tests (median time, total time, throughput, gain, decision, and selected
size) around its normal progress bar and one-line SL summary.

### Diagnostics stage and per-run output directories

Step 4 wraps `python -m diagnostics.evaluate`, the same five agent-vs-random comparisons
`run_pipeline.py` runs after RL training (`diagnostics/evaluate.py::run_all_pairs`),
passing `--rl-weights`/`--neural-weights` explicitly so it evaluates the exact
checkpoints this invocation trained or reused, rather than falling back to
`diagnostics.pairwise`'s hardcoded `models/domino_rl_weights.npz` /
`models/domino_sl_weights.npz` defaults.

With no fixed worker option supplied by this wrapper, diagnostics benchmark
1, 2, 4, 6, ... CPU-only workers independently for every matchup, retain each
benchmark game's result, stop below 10% marginal gain or on a memory guard,
and never exceed 20 workers. The aggregate JSON report records the selected
count for each matchup.

Every invocation gets its own output directory —
`diagnostics/results/<rl-weights-basename>/` (the `.npz` suffix stripped from
`--rl-weights-file`) — instead of the single shared `all_pairs/` directory
`run_pipeline.py` always writes to. A batch of runs that only vary
`--rl-weights-file` per configuration therefore keeps every configuration's
comparison table, CSV, and plots side by side instead of the next run
overwriting the last one. Override the computed path directly with
`--diag-output-dir` if needed.

This was verified end-to-end with a tiny run (`--diag-mode fast --diag-games 3
--diag-no-pair-plots --diag-seed 7`, `--rl-weights-file
models/smoke_test_rl_weights.npz`) chained after a tiny RL stage with
`--skip-dataset --skip-sl`: the diagnostics stage correctly evaluated the
just-trained checkpoint and wrote its comparison report to
`diagnostics/results/smoke_test_rl_weights/`.

### Monitored batch run example

```bash
train_script/run_training_pipeline.sh --skip-dataset \
    --sl-early-stopping-patience 12 --sl-lr-decay-factor 0.5 \
    --sl-lr-decay-patience 5 --sl-weight-decay 0.0001 \
    --rl-value-head --rl-normalize-advantages --rl-clip-grad-norm 2.0 \
    --rl-moving-average-window 10 --rl-seed 42 \
    --rl-weights-file models/domino_rl_weights_monitored.npz
```

This was verified end-to-end with a tiny run
(`--rl-iterations 6 --rl-games-per-iteration 3 --rl-moving-average-window 3`)
exercising `--rl-value-head --rl-normalize-advantages --rl-clip-grad-norm 2.0
--rl-seed 7` together, and separately confirmed that the SL flags map to the
expected `training.training_loop` arguments.

## run_rl_parameter_sweep.sh

A second, separate script: an RL-only hyperparameter sweep with a
diagnostic against the random agent, independent of the full pipeline above
(it assumes a supervised checkpoint already exists and only trains RL).

```bash
train_script/run_rl_parameter_sweep.sh
```

Trains one dedicated self-play model per sweep point. The full cross-product
grid is 3 learning rates x 4 gamma values x 3 games-per-iteration values = 36
base configurations. Every base configuration runs with the critic off and on
(72 runs). A seeded sample of 10 base configurations also runs the two
non-baseline value coefficients with the critic on (20 more runs), for **92
training+diagnostic runs** by default. Every point is diagnosed against
`random`, and the final comparative report groups the 40/80/160 game-batch
variants into columns.

`value_coef` only affects training inside
`PolicyNetwork.backward_policy_gradient`'s `use_value_head` branch, so it has
no effect on direct REINFORCE (critic off). The separate value-coefficient
axis therefore runs only with the critic on.

Baselines and the learning-rate/gamma grid values come from
`diagnostics/hyperparameter_sweep.py` (`BASELINE_LEARNING_RATE`,
`BASELINE_GAMMA`, `BASELINE_VALUE_COEF`, `DEFAULT_LR_VALUES`,
`DEFAULT_GAMMA_VALUES`). That module only exposes a single baseline value for
games-per-iteration, not a sweep tuple, so its range comes from the
historical sweep table in
`references/explicacoes/relatorios/teste_1/plano_correcao.tex` instead:
games-per-iteration in `{40, 80, 160}`. `value_coef` uses `{0.25, 0.5, 0.75}`;
the `0.5` baseline is already covered by the critic-on grid.

Naming (models and diagnostics share one name per run):

```text
models/rl_test/domino_rl[_critic]_default_iter002000.npz
models/rl_test/domino_rl[_critic]_lr<LR>_gamma<GAMMA>_gpi<GPI>_iter002000.npz
models/rl_test/domino_rl_critic_<grid-tag>_vc<VC>_iter002000.npz
diagnostics/results/rl_test/domino_rl[_critic]_<same-tag>/
```

`_critic` appears when the value head is on; the grid combination matching
every baseline value exactly is tagged `default`, and a `value_coef` sweep
point equal to the baseline is skipped (already covered by that `default`
run). Each diagnostics directory also gets a `sweep_run.json` recording the
full hyperparameter set for that run, so comparisons across runs don't depend
on parsing the folder name.

| Flag | Meaning | Default |
|---|---|---:|
| `--rl-iterations` | RL training iterations per sweep point | `2000` |
| `--sl-weights-path` | Input SL weights used to initialize every RL run | `models/domino_sl_weights.npz` |
| `--diagnostic-games` | Games in the rl-vs-random diagnostic per sweep point | `10000` |
| `--seed` | Fix random/NumPy state for both training and diagnostics | `42` |
| `--model-dir` | Output directory for RL checkpoints | `models/rl_test` |
| `--resume` | Continue incomplete training and reuse complete, compatible diagnostics | off |
| `--diag-no-plots` | Skip the per-run diagnostic PNG plots (CSV/JSON are always written) | off (plots on) |
| `--device` | Array backend for every sweep point: `auto`/`cpu`/`gpu` (see `training/README.md`) | `auto` |
| `--rl-workers` | CPU-only rollout workers inside the current point | `auto` |
| `--vc-sample-count` | Seeded base configurations used by the non-baseline value-coefficient axis | `10` |
| `--results-dir` | Output directory for per-run diagnostics subdirectories | `diagnostics/results/rl_test` |
| `--report-output-dir` | Where the final comparative table is written | `diagnostics/results/rl_sweep_table` |
| `--skip-report` | Skip the final comparative-table stage | off |
| `--jobs` | Compatibility flag; only `1` is accepted | `1` |
| `--ram-limit-mb` | Physical-memory cap for the current training subprocess | 80% of detected RAM |
| `--vram-limit-mb` | CuPy memory-pool cap for the current training subprocess | 80% of detected VRAM when device is not `cpu` |

A fixed `--seed` (default `42`) is used for every sweep point, per the
historical reports' recommendation to fix randomness when comparing
hyperparameter configurations.

### Sequential execution, interruption, and safe resume

The shell driver runs exactly one sweep point at a time. `--jobs` is retained
only for command compatibility and rejects every value except `1`. Parallelism
is exclusively inside `training.self_play`, where `--rl-workers auto` performs
the retained worker benchmark and applies the existing 20-worker and memory
limits. This avoids multiplying outer sweep processes by inner rollout pools.
Each RL point uses the compact presentation: retained worker-tuning messages,
one progress bar, and one final summary instead of iteration/checkpoint lines.

Every RL save made by this driver is iteration-numbered, for example
`domino_rl_default_iter000050.npz`. A paired
`domino_rl_default_iter000050.resume.npz` stores the configuration, checksum,
completed iteration, selected worker count, and exact historical opponent
pool. Both files are published atomically. Only the newest pool-state file is
retained because it can be much larger; all numbered policy checkpoints remain.

Press `Ctrl+C` once to stop the foreground point. Resume with the exact same
options plus `--resume`:

```bash
train_script/run_rl_parameter_sweep.sh
# Press Ctrl+C once.
train_script/run_rl_parameter_sweep.sh --resume
```

If custom options were used, repeat them unchanged. Resume scans each point for
the newest complete pair at or below the requested iteration total, verifies
its hash, seed, and computation-affecting hyperparameters, restores its policy
and opponent pool, and begins at the following absolute iteration. A lone or
corrupt file from an interrupted write is ignored. A point already at the
requested total skips training. A matching final model plus diagnostic is
checked before resumable opponent-pool state, because no computation remains
and a later automatic CPU/GPU selection change cannot alter completed output.
Its diagnostic is reused when the
configuration, numbered model path, seed, requested-game summary, complete games
CSV, and requested plots all match. New outputs record the model SHA-256;
existing outputs without that field are accepted only when their artifacts
are not older than the model checkpoint. Incomplete, stale, or incompatible
diagnostics are rerun automatically. Legacy unsuffixed model files cannot prove their completed
iteration or restore their opponent pool, so safe resume intentionally does
not treat them as complete.

### Memory limits

Memory limits are computed for the one active training subprocess and applied
by default unless overridden. The automatic value is 80% of detected system
RAM and, when GPU use is possible, 80% of detected VRAM.

- `--ram-limit-mb`: a cap on each subprocess's actual physical memory, via
  `systemd-run --user --scope -p MemoryMax=... -p MemorySwapMax=0` (a cgroup
  `memory.max` limit) wrapped around the `python -m training.self_play`
  invocation. `MemorySwapMax=0` makes an over-limit process fail fast
  (SIGKILL) instead of thrashing into swap. If `systemd-run` user scopes
  aren't usable on this system, the script prints a warning and proceeds
  without a RAM cap.

  This is deliberately **not** `ulimit -v`/`RLIMIT_AS` (virtual address
  space): that was tried first and rejected after it broke real runs.
  CUDA reserves very large virtual address ranges during initialization
  regardless of actual physical usage, so an address-space limit generous
  enough on paper (tens of GiB) still broke `import numpy.random` and
  CuPy's own CUDA context setup (`ImportError: ... failed to map segment
  from shared object`) the moment `--device auto`/`gpu` was involved --
  which is the default. The cgroup limit bounds physical residency instead,
  which doesn't fight CUDA's address-space habits, and is what
  "prevent OOM" actually needs.
- `--vram-limit-mb`: a hard cap on each subprocess's CuPy default memory
  pool (`cupy.get_default_memory_pool().set_limit`), read from the
  `DOMINO_VRAM_LIMIT_MB` environment variable by `agents/nn.py` at import
  time. This is CuPy's own allocator enforcing the limit, not an OS
  mechanism, so it doesn't share `RLIMIT_AS`'s CUDA incompatibility --
  verified directly: a 1MiB limit made a ~800MB CuPy allocation raise
  `cupy.cuda.memory.OutOfMemoryError` with `limit set to: 1,048,576 bytes`
  in the message. No-op when unset (every existing caller).

The RAM cgroup and CuPy pool implementation details remain unchanged from the
main training driver; see the flag descriptions above and the top-level
resource-safety documentation.

### Comparative table (`diagnostics.rl_sweep_table`)

The very last stage runs `python -m diagnostics.rl_sweep_table`
(`diagnostics/rl_sweep_table.py`), which discovers every subdirectory of
`diagnostics/results/` containing both `sweep_run.json` and `summary.json`
(i.e. every point this script has ever produced, not just the current
invocation -- it accumulates across repeated sweeps the same way
`diagnostics/hyperparameter_sweep.py`'s JSON log does), joins each run's
hyperparameters with its rl-vs-random win/draw/loss rates and writes:

- `diagnostics/results/rl_sweep_table/rl_sweep_table.csv`
- `diagnostics/results/rl_sweep_table/rl_sweep_table.json`
- `diagnostics/results/rl_sweep_table/rl_sweep_table.png` -- an image table,
  using the same visual style as the aggregate diagnostics report
- `diagnostics/results/rl_sweep_table/rl_sweep_table.pdf` -- the same compact
  table as a vector PDF

The CSV and JSON retain one row per trained model, including its exact
games-per-iteration value and checkpoint path. The PNG, PDF, and console table group
models that differ only in games per iteration into one row with win-rate
columns labelled `40`, `80`, and `160`. Critic, learning rate, gamma, and
value coefficient remain row dimensions. Percentage cells use red/blue
win-rate shading around the neutral 50% region.

The `40`, `80`, and `160` columns are a side-by-side comparison, not an
assumed ranking. Larger training batches usually reduce gradient noise, but
they do not guarantee a strictly higher final win rate because update count,
exploration, optimization dynamics, and statistical uncertainty still matter.
Treat differences as meaningful only when their confidence intervals and
repeated-seed results support the same conclusion.

Run it standalone at any time to rebuild the table from whatever sweep output
already exists on disk:

```bash
python -m diagnostics.rl_sweep_table
```

Verified against the real 16-run/500-game sweep this script produced: the
table correctly reproduced every run's win/draw rate, grouped and sorted as
described, and the PNG rendered without clipped columns.

## run_rl_parameter_sweep.py

An in-process counterpart to `run_rl_parameter_sweep.sh`, for when the
subprocess-per-sweep-point overhead dominates the actual training time (e.g.
a small `--rl-iterations` for a quick test). Same grid search, same
critic-off-then-on structure, same output naming and `sweep_run.json`
schema -- but as a single persistent Python process instead of one
`python -m training.self_play` subprocess per sweep point:

```bash
python train_script/run_rl_parameter_sweep.py
python train_script/run_rl_parameter_sweep.py --rl-iterations 2000 --resume
python train_script/run_rl_parameter_sweep.py --device cpu --quiet-training
```

Each subprocess in the `.sh` version pays Python interpreter startup,
re-imports numpy/cupy/agents/middleware, and re-reads the SL checkpoint from
disk -- and, on a GPU machine, re-initializes a CUDA context. The `.py`
version reads the SL checkpoint from disk exactly once
(`load_sl_weights_once`) and passes it to every
`training.self_play.train(..., sl_weights_data=...)` call via a new optional
parameter (also threaded through `agents/rl_nn.py::PolicyNetwork.load_from_sl`'s
`data` argument), and calls `training.self_play.train()`,
`diagnostics.pairwise.run_pairwise()`, and
`diagnostics.rl_sweep_table.build_report()` directly instead of spawning a
subprocess for each. `sl_weights_data` defaults to `None` everywhere it was
added, so every existing caller (the `.sh` script, `run_pipeline.py`, direct
CLI use) is unaffected.

This does not replace the shell script -- both exist, produce interchangeable
output (the same `diagnostics/results/domino_rl*/sweep_run.json` schema, so
`diagnostics/rl_sweep_table.py` reads either one's output the same way), and
can be run against the same `--model-dir`/`--results-dir`.

CLI flags mirror the shell script's: `--rl-iterations`, `--sl-weights-path`,
`--diagnostic-games`, `--seed`, `--model-dir`, `--results-dir`, `--resume`,
`--diag-no-plots`, `--device`, `--rl-workers`, `--report-output-dir`, `--skip-report`, plus
`--quiet-training` (suppresses `self_play`'s per-iteration logs and
`pairwise`'s per-matchup console summary; off by default, matching the shell
script's always-verbose behavior).

Verified end-to-end (`--rl-iterations 2 --diagnostic-games 3 --diag-no-plots
--quiet-training` against isolated `--model-dir`/`--results-dir`, never the
real ones): all 72 runs completed with the same tags/naming as the shell
script, the comparative table matched. Separately confirmed, by mocking
`numpy.load` and logging every call's path, that the SL checkpoint is read
from disk exactly once regardless of sweep size (1 vs. 4 training runs both
loaded it exactly once) -- the only other `np.load` calls are each run's own
checkpoint (a resume-probe that fails since it doesn't exist yet, and the
diagnostics stage reading back the checkpoint it just trained), which is
unrelated to the SL-weight redundancy this script fixes and happens
identically in the shell-script version.
