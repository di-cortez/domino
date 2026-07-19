# train_script

Personal batch-training driver for the full pipeline described in
`training/README.md` and the top-level `README.md`:

1. generate supervised examples from heuristic-vs-heuristic games, using the
   `training.dataset_generator` module defaults, including retained automatic
   dataset-worker tuning;
2. train the supervised neural policy, using the `training.training_loop`
   module defaults (no CLI flags exist for this stage either);
3. refine that policy with a **BIG-scale** self-play reinforcement-learning
   run — 5x the default iteration count (1,000 x 5 = 5,000 iterations),
   matching the `big` scale in `run_pipeline.py`'s `SCALE_FACTORS`. This
   stage is fully parameterized from the command line so the script can drive
   repeated batch runs that only vary RL hyperparameters;
4. run the **all-pairs diagnostics matrix** (`diagnostics.evaluate`) —
   heuristic/neural/rl vs random and in self-play — evaluating the exact
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
runs bare by default too, but accepts three opt-in convergence flags (below)
that map straight to `training.training_loop`'s own optional flags. Use
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
  holding it fixed — both already exist as `training.training_loop` flags and
  are now forwarded by this script (`--sl-early-stopping-patience`,
  `--sl-lr-decay-factor`, `--sl-weight-decay`).
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
documented module defaults. Supervised training exposes only the three SL
controls listed below.

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
| `--sl-early-stopping-patience` | SL | Validation checks (every 10 epochs) without improvement before stopping | unset (off) |
| `--sl-lr-decay-factor` | SL | LR multiplier applied on each validation check without improvement | unset (off) |
| `--sl-weight-decay` | SL | L2 penalty on the weight matrices | unset (off) |
| `--diag-mode` | Diagnostics | `default` (10 matchups), `fast` (2), or `complete` (15 matchups) | `complete` (BIG scale) |
| `--diag-games` | Diagnostics | Games per evaluated matchup | `50000` (BIG scale: `10000 x 5`) |
| `--diag-seed` | Diagnostics | Fix the RNG seed for the diagnostics games | unset |
| `--diag-no-pair-plots` | Diagnostics | Skip the per-matchup PNG plots (the aggregate table image is still generated) | off (plots on) |
| `--diag-output-dir` | Diagnostics | Override the output directory | `diagnostics/results/<rl-weights-basename>/` |
| `--skip-dataset` | control | Skip dataset generation (reuse an existing dataset file) | off |
| `--skip-sl` | control | Skip supervised training (reuse an existing SL weights file) | off |
| `--skip-rl` | control | Skip self-play reinforcement learning | off |
| `--skip-diagnostics` | control | Skip the all-pairs diagnostics stage | off |

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

### Diagnostics stage and per-run output directories

Step 4 wraps `python -m diagnostics.evaluate`, the same all-pairs matrix
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
matrix, CSV, and plots side by side instead of the next run overwriting the
last one. Override the computed path directly with `--diag-output-dir` if
needed.

This was verified end-to-end with a tiny run (`--diag-mode fast --diag-games 3
--diag-no-pair-plots --diag-seed 7`, `--rl-weights-file
models/smoke_test_rl_weights.npz`) chained after a tiny RL stage with
`--skip-dataset --skip-sl`: the diagnostics stage correctly evaluated the
just-trained checkpoint and wrote its matrix to
`diagnostics/results/smoke_test_rl_weights/`.

### Monitored batch run example

```bash
train_script/run_training_pipeline.sh --skip-dataset \
    --sl-early-stopping-patience 5 --sl-lr-decay-factor 0.5 --sl-weight-decay 0.0001 \
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

Trains one dedicated self-play checkpoint per sweep point: a **full grid
search** (cross product) over the three main hyperparameters --
`learning_rate` x `gamma` x `games_per_iteration`, 3 values each, 3x3x3 = 27
combinations -- plus a **separate** `value_coef` axis (10 values,
one-at-a-time, `learning_rate`/`gamma`/`games_per_iteration` held at
baseline; `value_coef` isn't crossed into the grid). Each point is diagnosed
against `random` (`python -m diagnostics.pairwise --agent rl --opponent
random`). Runs the whole sweep with the critic off, then again with it on,
with **every point identical between the two** so the two policies are
compared on the exact same sweep points: 72 training+diagnostics runs by
default (27 grid + 9 value_coef, per critic setting). Prints total elapsed
wall-clock time at the end.

`value_coef` only affects training inside
`PolicyNetwork.backward_policy_gradient`'s `use_value_head` branch, so it has
no effect on direct REINFORCE (critic off) -- those runs are trained and
swept anyway, for structural symmetry with the critic-on group, and are
expected to reproduce the grid's `default` checkpoint exactly (same seed,
same everything else that actually affects training).

Baselines and the learning-rate/gamma grid values come from
`diagnostics/hyperparameter_sweep.py` (`BASELINE_LEARNING_RATE`,
`BASELINE_GAMMA`, `BASELINE_VALUE_COEF`, `DEFAULT_LR_VALUES`,
`DEFAULT_GAMMA_VALUES`). That module only exposes a single baseline value for
games-per-iteration, not a sweep tuple, so its range comes from the
historical sweep table in
`references/explicacoes/relatorios/teste_1/plano_correcao.tex` instead:
games-per-iteration in `{40, 80, 160}`. `value_coef` uses 10 evenly spaced
values from 0.1 to 1.0 (baseline 0.5 included).

Naming (models and diagnostics share one name per run):

```text
models/rl_test/domino_rl[_critic]_default.npz                       (learning_rate/gamma/games_per_iteration all at baseline)
models/rl_test/domino_rl[_critic]_lr<LR>_gamma<GAMMA>_gpi<GPI>.npz  (every other grid point)
models/rl_test/domino_rl[_critic]_value_coef_<VC>.npz               (value_coef axis)
diagnostics/results/domino_rl[_critic]_<same tag>/
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
| `--diagnostic-games` | Games in the rl-vs-random diagnostic per sweep point | `500` |
| `--seed` | Fix random/NumPy state for both training and diagnostics | `42` |
| `--model-dir` | Output directory for RL checkpoints | `models/rl_test` |
| `--resume` | Skip training a checkpoint that already exists on disk; still (re)run its diagnostics | off |
| `--diag-no-plots` | Skip the per-run diagnostic PNG plots (CSV/JSON are always written) | off (plots on) |
| `--device` | Array backend for every sweep point: `auto`/`cpu`/`gpu` (see `training/README.md`) | `auto` |
| `--results-dir` | Output directory for per-run diagnostics subdirectories | `diagnostics/results` |
| `--report-output-dir` | Where the final comparative table is written | `diagnostics/results/rl_sweep_table` |
| `--skip-report` | Skip the final comparative-table stage | off |
| `--jobs` | Run up to N sweep points at once as background subprocesses | `1` (sequential) |
| `--ram-limit-mb` | Per-subprocess physical-memory cap in MiB (see below) | auto when `--jobs` > 1, else unset |
| `--vram-limit-mb` | Per-subprocess CuPy memory-pool cap in MiB (see below) | auto when `--jobs` > 1 and device isn't `cpu`, else unset |

A fixed `--seed` (default `42`) is used for every sweep point, per the
historical reports' recommendation to fix randomness when comparing
hyperparameter configurations. `--resume` matters because a full sweep is
expensive (72 runs by default): rerunning after an interruption skips
checkpoints that already exist instead of retraining them.

The naming scheme, `--resume` behavior, and `sweep_run.json` contents were
verified end-to-end with a tiny run (`--rl-iterations 2 --diagnostic-games 3
--diag-no-plots`) before the sweep became a full grid search with a
separately-swept `value_coef` axis; the grid/value_coef run-count math and
tag generation were checked by simulating the loop logic standalone (no
training invoked), and `diagnostics/rl_sweep_table.py`'s updated sort
(directly on the numeric hyperparameters rather than parsing the tag string)
was checked against synthetic grid/value_coef records.

### Concurrency (`--jobs`) and memory limits

Sweep points are fully independent -- each writes to its own unique
`--rl-weights-path`/diagnostics directory and only *reads* the shared SL
checkpoint -- so running several at once is safe. `--jobs N` launches up to
N sweep points at once as background `python -m training.self_play`
subprocesses; `--jobs 1` (the default) is unchanged from before this existed:
fully sequential, same terminal output. Measured on a 20-core/32GB machine
with one shared GPU: `--jobs 4` cut a 58-point tiny sweep from 6m13s to
3m11s (~1.95x, sub-linear because each batch of 4 waits for its slowest
member -- see below).

Concurrent output doesn't interleave: each background job's full output goes
to `diagnostics/results/_sweep_logs/<run-name>.log` instead of the terminal,
with only concise `started`/`finished`/`FAILED` lines printed live. Unlike
`--jobs 1` (where `set -e` aborts the whole script on the first failure,
unchanged), a failed point under `--jobs > 1` doesn't stop the rest of the
sweep -- other points keep running, failures are collected and printed as a
summary at the end, and the script exits nonzero if any occurred, so
automation can detect it while still getting every other point's results.

The job pool is batch-based, not a rolling pool: it launches up to `--jobs`
points, waits for that whole batch, then launches the next. Simpler and
portable across bash versions (a rolling pool needs `wait -n`), at the cost
of a batch waiting on its slowest point before the next one starts -- part of
why the measured speedup above is ~2x rather than ~4x at `--jobs 4`.

**Memory limits**, auto-computed from detected system RAM / GPU memory
divided by `--jobs`, **always applied by default** (not only when `--jobs >
1`) as a general OOM backstop, unless set explicitly with
`--ram-limit-mb`/`--vram-limit-mb`. 80% of the detected total is used, so
headroom stays free for the rest of the system. Verified on this machine: 20
CPU cores, 31779MiB (~31GiB) system RAM, one NVIDIA RTX 3050 with 6144MiB
(6GiB) VRAM, 2047MiB (2GiB) swap -- giving, at the default `--jobs 1`, a
~25.4GiB RAM / ~4.8GiB VRAM ceiling for that single process. That's
deliberately generous relative to real observed usage (a few hundred MiB per
process) -- it's a backstop against a bug badly leaking memory, not a routine
constraint, and re-verified not to interfere with a normal run (see below).

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

Verified end-to-end: the job pool's concurrency and failure-isolation logic
was checked in isolation first (fake sleep-based jobs, confirmed N=3
concurrent jobs finish in ~1/3 the sequential time and that one deliberately
failing job doesn't stop the others). The RAM limit's `RLIMIT_AS` rejection
was reproduced and diagnosed from an actual failed real run before switching
to the cgroup approach, which was then verified to (a) not break a real
`training.self_play` invocation under a generous limit and (b) actually
enforce a tight one (`MemoryMax=50M` + `MemorySwapMax=0` correctly SIGKILLed
a process touching 1.6GB). The full `--jobs 4` sweep (58 points) then ran
end-to-end with zero failures and correct `sweep_run.json`/`summary.json`
output, timed against an equivalent `--jobs 1` run for the speedup figure
above.

### Comparative table (`diagnostics.rl_sweep_table`)

The very last stage runs `python -m diagnostics.rl_sweep_table`
(`diagnostics/rl_sweep_table.py`), which discovers every subdirectory of
`diagnostics/results/` containing both `sweep_run.json` and `summary.json`
(i.e. every point this script has ever produced, not just the current
invocation -- it accumulates across repeated sweeps the same way
`diagnostics/hyperparameter_sweep.py`'s JSON log does), joins each run's
hyperparameters with its rl-vs-random win/draw/loss rates into one row, and
writes:

- `diagnostics/results/rl_sweep_table/rl_sweep_table.csv`
- `diagnostics/results/rl_sweep_table/rl_sweep_table.json`
- `diagnostics/results/rl_sweep_table/rl_sweep_table.png` -- an image table,
  mirroring `diagnostics/evaluate.py`'s all-pairs matrix output
  (`_matrix_rows`/`_save_matrix_csv`/`plot_all_pairs_table`)

Rows are grouped critic-off then critic-on, then by which parameter was
varied, then sorted by that parameter's value, with the win-rate column
shaded the same way `plot_all_pairs_table` shades its win-rate matrix (light
blue >= 60%, light orange <= 40%). It also prints an aligned console table.
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
`--diag-no-plots`, `--device`, `--report-output-dir`, `--skip-report`, plus
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
