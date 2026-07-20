# Domino - Neural vs Heuristic

## Repository Maintenance Policy

**Keep everything in this repository in English.** This requirement applies to
source code, filenames, directory names, variables, functions, classes, command
line options, log messages, comments, docstrings, tests, generated report
labels, and documentation.

All code must remain clearly documented with useful comments and docstrings,
especially around non-obvious algorithms, resource safeguards, concurrency,
and public interfaces. Update every affected README whenever behavior,
commands, configuration, outputs, or architecture change. A code change is not
complete until its documentation is current.

Interactive two-player domino simulator with a 3D OpenGL board, a rule-based
heuristic agent, a supervised neural agent, and an RL agent refined by self-play.
An untrained neural baseline is also available for measuring whether learned
checkpoints improve on their random initialization.

The project is organized so the game rules, agents, training code, diagnostics,
and visual UI can be changed independently.

## Project Structure

| Path | Purpose |
|---|---|
| `middleware/` | Game rules and orchestration (`DominoEngine`, `GameManager`). |
| `agents/` | Baseline, heuristic, supervised neural, and RL agents. |
| `training/` | Dataset generation, supervised training, and self-play training. |
| `diagnostics/` | Agent-vs-agent evaluation, metrics, CSV output, and plots. |
| `ui/` | Pygame/OpenGL visual simulator, HUD, input controller, and UI tests. |
| `dataset/` | Generated supervised-learning datasets. |
| `models/` | Generated `.npz` neural-network checkpoints. |
| `tests/` | Core non-UI tests. |

## Main Modules

| Module | Description |
|---|---|
| `middleware/domino_engine.py` | Domino rules: shuffle/deal, legal actions, draw/pass logic, game-over detection, state snapshots. |
| `middleware/middleware.py` | `Agent` protocol and `GameManager`, which connects the engine to the selected agents. |
| `agents/encoder.py` | Shared state/action encoder: 168 input features and 56 tile-play policy actions. Forced draw/pass actions bypass the network. |
| `agents/heuristic_agent.py` | `StrategicAgent`, the rule-based teacher used for dataset generation and benchmark evaluation. |
| `agents/neural_agent.py` | `NeuralAgent`, which loads supervised weights and plays with action masking. |
| `agents/random_neural_agent.py` | `RandomNeuralAgent`, a reproducible untrained neural-policy baseline. |
| `agents/rl_agent.py` | `RLAgent`, which plays a `PolicyNetwork` in training or evaluation mode. |
| `agents/nn.py` | NumPy/CuPy MLP for supervised learning. CuPy is selected automatically when available. |
| `agents/rl_nn.py` | Policy-only network with masked REINFORCE gradients for self-play RL. |
| `training/dataset_generator.py` | Coordinates deterministic dataset generation, retained worker tuning, bounded SQLite aggregation, and atomic JSONL output. |
| `training/dataset_parallel.py` | Plays independent heuristic-vs-heuristic dataset games in a bounded CPU-only worker pool. |
| `training/training_loop.py` | Trains supervised weights, skips forced labels, and saves the best validation checkpoint. |
| `training/supervised_runtime.py` | Retained CPU/GPU batch tuning, GPU dataset residency/windows, and supervised memory telemetry. |
| `training/self_play.py` | Refines the RL policy with direct REINFORCE, parallel rollout orchestration, and parent-only gradient updates. |
| `training/rl_parallel.py` | Generates deterministic RL trajectories in CPU-only workers backed by a bounded shared policy-snapshot bank. |
| `utils/exact_update_timing.py` | Aggregates per-stage CPU time spent in `ExactOpponentModel.update()` across the pipeline parent and worker processes. |
| `diagnostics/evaluate.py` | Evaluates all five supported agents against the common random baseline. |
| `diagnostics/pairwise.py` | Helper for evaluating one agent against another and writing `summary.json`, `games.csv`, and plots. |
| `diagnostics/hyperparameter_sweep.py` | Trains an RL checkpoint per sweep point (one hyperparameter varied at a time, critic on and off) and appends its diagnostics to a single JSON log. |
| `diagnostics/rl_sweep_table.py` | Joins sweep hyperparameters with rl-vs-random results; raw CSV/JSON keep every model, while the compact PNG pivots games-per-iteration into 40/80/160 win-rate columns. |
| `ui/visual_main.py` | Starts the visual simulator. |

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pygame PyOpenGL PyOpenGL-accelerate numpy matplotlib tqdm
```

### Linux GPU setup and verification

The neural networks use NVIDIA CUDA through CuPy. Dataset generation, RL
rollout workers, and diagnostics intentionally remain CPU-only; supervised
training and the RL parent process use the GPU when it is available and safe.

1. Verify that Linux sees an NVIDIA GPU and that the driver is loaded:

   ```bash
   lspci | grep -i nvidia
   nvidia-smi
   ```

   If `nvidia-smi` is missing or fails, install the recommended NVIDIA driver
   using the distribution package manager and reboot. On Ubuntu/Linux Mint:

   ```bash
   sudo apt update
   sudo apt install ubuntu-drivers-common
   sudo ubuntu-drivers install
   sudo reboot
   ```

   Do not install Python GPU packages until `nvidia-smi` works. The `CUDA
   Version` shown by `nvidia-smi` is the newest CUDA runtime supported by the
   loaded driver; it does not prove that a system CUDA Toolkit or `nvcc` is
   installed.

2. Activate this repository's environment and verify that `python` and `pip`
   point into `.venv`:

   ```bash
   source .venv/bin/activate
   which python
   python -m pip --version
   ```

3. Install exactly one CuPy wheel family. The `[ctk]` extra installs the CUDA
   runtime libraries inside `.venv`, so a system-wide CUDA Toolkit and `nvcc`
   are not required. Choose the wheel from the CUDA major version supported by
   the driver:

   ```bash
   # Driver supports CUDA 13.x
   python -m pip install "cupy-cuda13x[ctk]"

   # Driver supports CUDA 12.x
   python -m pip install "cupy-cuda12x[ctk]"
   ```

   If a matching system CUDA Toolkit is already installed and `nvcc --version`
   works, the smaller installation without `[ctk]` can be used instead. Never
   install `cupy`, `cupy-cuda12x`, and `cupy-cuda13x` together; their modules
   conflict. The authoritative package table is in the
   [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).

4. Test the CUDA runtime, a real GPU calculation, and device memory:

   ```bash
   python - <<'PY'
   import cupy as cp

   print("CuPy:", cp.__version__)
   print("CUDA runtime:", cp.cuda.runtime.runtimeGetVersion())
   print("CUDA driver:", cp.cuda.runtime.driverGetVersion())
   print("GPU count:", cp.cuda.runtime.getDeviceCount())
   print("GPU:", cp.cuda.runtime.getDeviceProperties(0)["name"])
   values = cp.arange(1_000_000, dtype=cp.float32)
   print("GPU calculation:", float(cp.sum(values * values).get()))
   free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
   print(f"VRAM: {free_bytes / 1024**2:.1f} MiB free / "
         f"{total_bytes / 1024**2:.1f} MiB total")
   PY
   ```

5. Verify the backend selected by this project:

   ```bash
   python - <<'PY'
   from agents.nn import GPU_ENABLED, GPU_UNAVAILABLE_REASON
   print("GPU enabled:", GPU_ENABLED)
   print("Fallback reason:", GPU_UNAVAILABLE_REASON)
   PY

   python -m training.training_loop
   ```

   The training command must begin with `CuPy available. Training on GPU.`.
   Use `watch -n 1 nvidia-smi` in another terminal to observe VRAM and GPU
   utilization during a real run.

Common failures:

- `ModuleNotFoundError: cupy`: CuPy was installed into a different Python;
  reactivate `.venv` and use `python -m pip`, not bare `pip` or `sudo pip`.
- `cudaErrorInsufficientDriver`: update the NVIDIA driver or install a CuPy
  wheel/runtime compatible with the current driver.
- missing `libcudart`, NVRTC, cuBLAS, or cuSPARSE libraries: reinstall with the
  `[ctk]` extra, or correctly set `CUDA_PATH` and `LD_LIBRARY_PATH` for an
  existing system Toolkit.
- warnings about multiple CuPy installations: run
  `python -m pip list | grep -i cupy`, uninstall every conflicting CuPy
  package, then install exactly one matching wheel.
- an intentional CPU selection: check `echo "$DOMINO_FORCE_CPU"` and
  `echo "$CUDA_VISIBLE_DEVICES"`; unset them for normal parent-process GPU use.

See NVIDIA's [Linux CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
for driver/Toolkit installation and its
[driver compatibility matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html)
for the meaning of the CUDA version reported by `nvidia-smi`.

Every pipeline run now prints one early `Pipeline compute resources` line with
the supervised and RL-parent backend, the CPU-only worker policy, free/total
system RAM, and free/total GPU VRAM. Individual stages retain their more
detailed memory snapshots.

## Run the Visual Simulator

```bash
source .venv/bin/activate
python -m ui.visual_main
```

Default players are:

- Player 0: `Neural`
- Player 1: `Heuristic`

Use the in-game menu (`M`) to cycle either player through `Neural`,
`Random NN`, `Heuristic`, `Random`, `Human`, and `RL (self-play)`.

## Visual Controls

| Key | Action |
|---|---|
| `Space` | Pause/resume automatic play. |
| `Right` | Step one turn forward and pause. |
| `Left` | Step one snapshot backward in history. |
| `+` / `-` | Change automatic speed between `1/4x`, `1/2x`, `1x`, `2x`, and `4x`. |
| `J` / `K` | Toggle player 0/player 1 hand visibility when the current mode allows it. |
| `M` | Open/close the settings menu. |
| `R` | Restart. Live games require pressing `R` twice within two seconds. |
| `ESC` | Close the menu or quit. |

During a human turn:

| Key | Action |
|---|---|
| `Left` / `Right` | Select a hand tile. |
| `Up` / `Down` / `Tab` | Switch the target end when the selected tile fits both ends. |
| `Enter` | Play the selected tile. |
| `D` | Draw from the stock when legal. |
| `P` | Pass when legal. |

## Neural Encoding Update

The current neural policy uses a 168-feature public-information vector and a
56-action output space. The network only chooses real tile-play decisions: 28
possible tiles on the left end and 28 possible tiles on the right end. Draw,
pass, and single-option tile plays are forced by the rules engine, so
`NeuralAgent`, `RandomNeuralAgent`, and `RLAgent` return them directly without a
network call.

The input vector includes current hand, played tiles, normalized play turn, who
played each tile, board ends, hand sizes, stock size, draw/pass counts, and
`opponent_suit_probabilities[7]`. Those final seven values estimate the chance
that the opponent currently holds each suit/value. `0.0` means known absence;
`1.0` means known presence.

Opponent inference is exact. It starts with temporal slot/cohort domains, then
converts once to integer `mu(H)` hand weights at the first non-terminal turn end
where `comb(|U|, h) <= 500`. Drawn slots retain only evidence observed after
their creation. The exact path has no particle fallback, and the heuristic uses
the joint hand posterior rather than combining suit marginals independently.
Slot assignment counting groups tiles with identical eligible-slot sets and
uses exact falling factorials, backed by a bounded 8,192-entry process-local
cache. Persistent built-in consumers incrementally annotate appended public
history and disable intermediate traces when they need only the final vector;
direct model construction retains detailed traces by default. The exact model
remains CPU-based and keeps arbitrary-precision integer weights throughout.

Old checkpoints trained with the previous 86-input/58-output encoder are not
compatible. Regenerate the dataset, retrain supervised learning, and then retrain
RL before using `Neural` or `RL` in the UI or diagnostics.

Checkpoints trained with the newer 168/56 shape but the older absence-confidence
feature still load by shape, but their final seven inputs now have the opposite
meaning. Archive those weights and retrain for clean results.

## Training Pipeline

Run the complete sequence with compact progress bars:

```bash
python run_pipeline.py
python run_pipeline.py small
python run_pipeline.py big
python run_pipeline.py huge
```

With no argument, the runner uses the normal defaults: 10,000 dataset games,
1,000 supervised epochs, 1,000 RL iterations, and 10,000 diagnostic games per
matchup. `small` uses one fifth of those counts, `big` uses five times those
counts, and `huge` uses twenty times those counts. RL keeps 40 games per
iteration so the scale changes total RL iterations linearly. Historical mode
labels remain (`fast`, `default`, and `complete`), but every pipeline scale now
uses the same five matchups: each of `rl`, `neural`, `random_nn`, `heuristic`,
and `random` against `random`. Diagnostic game counts are specified per
matchup.

Each full pipeline run appends a timing record to
`../exact_opponent_model_timing.json`, next to the repository. The report keeps
separate entries for dataset generation, supervised training, RL self-play,
and diagnostics. For every stage it records elapsed wall time and splits
aggregate process CPU time into `ExactOpponentModel.update()` and everything
else, including call counts and percentages. Aggregate CPU is the parent CPU
plus all reaped worker CPU, so parallel worker times are intentionally summed.
The report also includes the aggregate wall duration of exact-update calls as
informational data; do not subtract it from stage wall time because concurrent
worker calls overlap. The timing hook is inactive when modules are run outside
`run_pipeline.py`, and it does not add console log lines.

Dataset generation, RL rollouts, and diagnostics automatically benchmark
CPU-only worker counts 1, 2, 4, 6, ... up to the hard limit of 20. Dataset and
diagnostic attempts retain 1% game slices. RL attempts retain complete early
iterations totaling about 1% of the planned iterations per candidate, so every
tested trajectory still contributes to a gradient update. Testing stops on a
memory/error guard or below 10% marginal gain. Override the tuners with
`--dataset-workers N`, `--rl-workers N`, or `--diagnostic-workers N`; the
corresponding `--*-memory-reserve-mb` options control their RAM reserves.

Generate supervised data:

```bash
python -m training.dataset_generator
python -m training.dataset_generator --workers auto --seed 123
```

Dataset generation uses a bounded dynamic queue and writes completed games to a
temporary SQLite database in the parent process. It emits the final JSONL in
stable game-id order, atomically replaces the old dataset only after success,
shows a progress bar, and prints total elapsed time.

Train the supervised neural agent:

```bash
python -m training.training_loop
```

Supervised training uses safe host RAM when possible and an atomic disk-backed
`.npy`/`mmap` cache when the encoded dataset is too large. In GPU mode it keeps
the full dataset resident when safe or reuses a bounded rotating GPU window.
It benchmarks power-of-two CPU/GPU mini-batches for 10 retained epochs each,
selects on synchronized median examples/second, and stops at the first gain
below 10% or memory guard. Every completed benchmark epoch remains trained.
The pipeline prints each batch candidate's median epoch time, total benchmark
time, throughput, marginal gain, and the final selected size alongside its
compact supervised progress bar.

The dataset encoder uses a two-pass preallocated `float32` representation and
checks cgroup-aware RAM headroom before loading/encoding. RL also validates host
workspace and effective free VRAM before large batch assembly; automatic device
selection falls back to CPU when VRAM is below its safety minimum.

Supervised weights and intermediates are `float32`; legacy `float64` checkpoints
are cast safely. Plateau LR decay is enabled by default: validation runs every
10 epochs and five consecutive failures multiply LR by `0.5`. Early stopping
and LR scheduling use independent counters. See `training/README.md` for
device, fixed-batch, reserve, seed, scheduler, and disable flags; the canonical
pipeline device flag is `--sl-device`.

Refine the RL agent:

```bash
python -m training.self_play
python -m training.self_play --rl-workers auto --seed 123
python -m training.self_play --rl-workers 4 --device cpu
```

Self-play reports startup memory, checkpoint-to-checkpoint time, and total
elapsed time. Iteration logs omit entropy and show reward mean/min/max,
good/neutral/bad percentages, local reward mean, draw/pass event counts, wins,
pool size, and gradient norm.

All games inside an iteration see the same frozen policy. CPU-only workers
generate trajectories from shared-memory policy snapshots; the main process
orders results by game id, assembles the batch, performs the only gradient
update, and remains the only process allowed to use the GPU. Stable per-game
seeds make fixed worker counts, autotuning, scheduling, and memory fallback
produce identical seeded trajectories. Checkpoint evaluation is parallelized
through the same safe pool.

Headless dataset, RL, and diagnostic turns reuse the legal-action collection
already computed for the agent and skip the discarded post-action engine
snapshot. `DominoEngine.step(action)` keeps its original validating, full-state
defaults; the precomputed collection is used only by controlled internal loops
for the exact current position. Human actions still ask the engine to compute
legal actions itself.

Run the reproducible structural and throughput comparison with:

```bash
python benchmarks/headless_step_benchmark.py
python benchmarks/headless_step_benchmark.py --games 500 --rollout-games 200 \
  --json-output /tmp/headless_step_benchmark.json
```

It compares the legacy-equivalent discarded-state loop with the optimized
path for random, heuristic, neural, and RL agents against random, plus actual
RL rollout generation when checkpoints are available. It reports games per
second, legal-action calls, state snapshots, serialized history actions, and
requires identical fixed-seed result fingerprints.

The learner samples actions and stores its trajectory. Frozen self-play pool
opponents also sample actions but do not store trajectories. Checkpoint
evaluation, diagnostics, and UI play remain deterministic.

The default RL update is policy-only REINFORCE. It applies local draw/pass events to all
earlier real decisions with `EVENT_REWARD_DECAY = 0.90`, using
`k = event_turn - decision_turn - 1`, then adds the uniform terminal result and
multiplies by the number of legal tile-play options. Add `--value-head` to
`training.self_play` or `run_pipeline.py` to learn `V(s)` and train the policy
from `reward - V(s)` advantages. Default checkpoints contain only policy arrays;
value-head checkpoints also contain `Wv` and `bv`.

Generated files:

| File | Generated by |
|---|---|
| `dataset/supervised_dataset.jsonl` | `training.dataset_generator` |
| `dataset/supervised_dataset_encoded.npz` | `training.training_loop` cache for encoded `X/Y` arrays |
| `dataset/supervised_dataset_X.npy`, `supervised_dataset_Y.npy`, `supervised_dataset_metadata.json` | Disk-backed fallback cache from `training.training_loop` |
| `models/domino_sl_weights.npz` | `training.training_loop` |
| `models/domino_rl_weights.npz` | `training.self_play` |

## Diagnostics

Run all five agents against `random`, with 10,000 games per matchup by default:

```bash
python -m diagnostics.evaluate
python -m diagnostics.evaluate fast
python -m diagnostics.evaluate complete
python -m diagnostics.evaluate complete -n 5000
python -m diagnostics.evaluate fast --workers auto --seed 123
```

Supported names are `rl`, `neural`, `random_nn`, `heuristic`, and `random`.
`random_nn` uses the supervised network architecture with reproducible random
initial weights and no checkpoint. The old `greedy` baseline is no longer part
of diagnostics. `random_nn` is included in every automatic diagnostic mode.

The full diagnostic writes to `diagnostics/results/all_pairs/` unless `--output`
is provided. Pair evaluation uses a progress bar. The aggregate report records
`selected_workers_by_matchup`, retained tuning details for every matchup, and
`duration_s`; pair summaries also include `duration_s`:

- `all_pairs_table.png`
- `all_pairs_table.pdf`
- `choice_opportunities.png`
- `all_pairs_matrix.csv`
- `all_pairs_summary.json`
- `pairs/<agent>_vs_<opponent>/summary.json`
- `pairs/<agent>_vs_<opponent>/games.csv`
- `pairs/<agent>_vs_<opponent>/*.png`

For a single matchup, call the pairwise helper directly:

```bash
python -m diagnostics.pairwise --agent heuristic --opponent random
python -m diagnostics.pairwise --agent rl --opponent neural
```

## Tests

Run the core tests:

```bash
python tests/test_core.py
python tests/test_parallel_dataset.py
python tests/test_parallel_diagnostics.py
python tests/test_parallel_rl.py
```

Run the UI/controller tests:

```bash
python ui/test_ui_controller.py
```

Compile-check all Python packages:

```bash
python -m compileall agents middleware training diagnostics ui tests
```

## VS Code

1. Open this repository folder in VS Code.
2. Select the interpreter at `.venv/bin/python`.
3. Open an integrated terminal from the repository root.
4. Activate the environment with `source .venv/bin/activate`.
5. Run the commands above with `python -m ...`.
