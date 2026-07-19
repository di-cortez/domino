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
| `training/self_play.py` | Refines the RL policy with direct REINFORCE and decayed draw/pass reward shaping. |
| `diagnostics/evaluate.py` | Runs the upper-triangle all-pairs diagnostic matrix. |
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

For GPU acceleration, install a CuPy build that matches the local CUDA runtime.
For CUDA 12:

```bash
python -m pip install cupy-cuda12x
```

Training and diagnostics commands print a startup RAM/GPU memory snapshot.
`training/training_loop.py` also prints whether it is using CPU or GPU at
startup.

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
iteration so the scale changes total RL iterations linearly. Pipeline diagnostic
modes are selected automatically: `small` uses `fast` (2 matchups), the default
pipeline uses `default` (10), and `big`/`huge` use `complete` (15). Diagnostic
game counts are always specified per matchup.

Dataset generation and diagnostics automatically benchmark CPU-only worker
counts 1, 2, 4, 6, ... up to the hard limit of 20. Dataset attempts each use
and retain 1% of all requested dataset games. Diagnostics tune every matchup
independently because agent combinations have different costs; every attempt
uses and retains 1% of that matchup's games. Testing stops on a memory/error
guard or below 10% marginal gain. Override either tuner with
`--dataset-workers N` or `--diagnostic-workers N`; the corresponding
`--*-memory-reserve-mb` options control their RAM reserves.

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

Supervised training keeps the complete encoded dataset in RAM and transfers
only mini-batches of 1024 examples to the GPU. It reports startup memory,
checkpoint-to-checkpoint time, and total elapsed time.

The dataset encoder uses a two-pass preallocated `float32` representation and
checks cgroup-aware RAM headroom before loading/encoding. RL also validates host
workspace and effective free VRAM before large batch assembly; automatic device
selection falls back to CPU when VRAM is below its safety minimum.

Weight decay, early stopping, and learning-rate decay are optional. The default
keeps the learning rate fixed; see `training/README.md` for the three flags and
their configurable values. The same flags work with `run_pipeline.py`.

Refine the RL agent:

```bash
python -m training.self_play
```

Self-play reports startup memory, checkpoint-to-checkpoint time, and total
elapsed time. Iteration logs omit entropy and show reward mean/min/max,
good/neutral/bad percentages, local reward mean, draw/pass event counts, wins,
pool size, and gradient norm.

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
| `models/domino_sl_weights.npz` | `training.training_loop` |
| `models/domino_rl_weights.npz` | `training.self_play` |

## Diagnostics

Run diagnostics with 10,000 games per selected matchup. With no mode argument,
the historical four-agent upper triangle runs 10 matchups:

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
of diagnostics. The `random_nn` agent enters the automatic matrix only in
`complete` mode, but remains available to the pairwise helper in every mode.

The full diagnostic writes to `diagnostics/results/all_pairs/` unless `--output`
is provided. Pair evaluation uses a progress bar. The aggregate report records
`selected_workers_by_matchup`, retained tuning details for every matchup, and
`duration_s`; pair summaries also include `duration_s`:

- `all_pairs_table.png`
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
