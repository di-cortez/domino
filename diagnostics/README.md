# Diagnostics

Diagnostics compare every supported agent with the same random baseline over
many two-player domino games and write compact metrics, CSV data, and plots.

## Supported Agents

| Name | Implementation |
|---|---|
| `rl` | `RLAgent` loaded from the RL self-play checkpoint and used in evaluation mode. |
| `neural` | `NeuralAgent` loaded from supervised-learning weights. |
| `random_nn` | `RandomNeuralAgent` using the supervised architecture with fixed, untrained random weights. |
| `heuristic` | `StrategicAgent`, the handcrafted rule-based agent. |
| `random` | Uniform random legal move. |

## Diagnostic Workload

| Command | Matchups |
|---|---:|
| `python -m diagnostics.evaluate` | 5: every supported agent vs `random`. |

The command uses 10,000 games per matchup by default. Set the explicit count
with `-n`/`--games`:

```bash
python -m diagnostics.evaluate --games 5000
```

Diagnostics use CPU-only multiprocessing by default. Immediately before each
matchup, an independent online benchmark tries 1, 2, 4, 6, 8, 10, ... workers
(never more than 20), stopping on an error/memory guard or when marginal
throughput gain is below 10%. Each attempt plays 1% of that matchup's requested
games, and those games remain in that matchup's final result. Computationally
different matchups may therefore select different worker counts. Stable
matchup and absolute-game seeds ensure that scheduling and worker fallback do
not alter results.

Each matchup displays a `tqdm` progress bar. The command also prints a cgroup-
aware RAM/GPU memory snapshot and writes elapsed seconds as `duration_s`.

Useful options:

```bash
python -m diagnostics.evaluate --help
python -m diagnostics.evaluate --seed 123
python -m diagnostics.evaluate --games 5000 --seed 123
python -m diagnostics.evaluate --no-pair-plots
python -m diagnostics.evaluate --output /tmp/domino_all_pairs
python -m diagnostics.evaluate --neural-weights models/domino_sl_weights.npz
python -m diagnostics.evaluate --rl-weights models/domino_rl_weights.npz
python -m diagnostics.evaluate --workers 4
python -m diagnostics.evaluate --workers auto --autotune-fraction 0.01
python -m diagnostics.evaluate --memory-reserve-mb 1024
```

Worker subprocesses cannot see the GPU, use a bounded dynamic job queue, and
return records to the parent for aggregation/writing. RAM is checked before a
pool starts and while it runs. Under pressure, unfinished game ids are retried
with half as many workers while completed records are kept. Output directories
are replaced atomically only after all files and plots have been produced.

Inside each diagnostic game, the headless engine loop reuses the fresh,
unchanged legal-action collection already supplied to the acting agent and
does not serialize a post-action state that would be discarded. The final
`engine.to_dict()` record, seeded deals, agent choices, validation, terminal
rules, and diagnostic outputs remain unchanged. This trusted collection is an
internal optimization and is not accepted from diagnostic CLI or external
payloads.

The output folder defaults to `diagnostics/results/all_pairs/`.
Reusing that folder replaces the aggregate report and removes pair folders that
do not belong to the selected plan, keeping its contents internally consistent.

| File or folder | Contents |
|---|---|
| `all_pairs_table.png` | Metadata-rich one-row comparison of all five agents against random. |
| `all_pairs_table.pdf` | Vector PDF version of the same aggregate comparison. |
| `choice_opportunities.png` | Aggregate histogram of draw/pass/choice opportunities across all evaluated matchups. |
| `all_pairs_matrix.csv` | One row per evaluated matchup. |
| `all_pairs_summary.json` | Full aggregate report with `selected_workers_by_matchup`, per-matchup retained autotuning reports, accumulated choice-opportunity stats, `duration_s`, and all pairwise summaries. |
| `pairs/<agent>_vs_<opponent>/` | Standard pairwise artifacts for each matchup. |

The aggregate PNG/PDF header records games per matchup, total games,
elapsed evaluation time, seed, selected workers, checkpoint names, neural
architectures, parameter counts, and whether the RL checkpoint contains a
value head. It also reports the 95% worst-case percentage margin of error as
`sqrt(0.9604 / n)`, rounded to two significant digits, where `n` is the games
per matchup.

Win-rate cells use red-to-blue intensity bands at five-percentage-point
intervals: `<30%`, `30–35%`, ..., `65–70%`, and `≥70%`. This makes both weak
and strong deviations from 50% visible without changing the underlying
numeric percentages.

## Pairwise Helper

Use the helper directly when only one matchup is needed:

```bash
python -m diagnostics.pairwise --agent heuristic --opponent random
python -m diagnostics.pairwise --agent rl --opponent neural
python -m diagnostics.pairwise --agent neural --opponent random_nn
python -m diagnostics.pairwise --agent heuristic --opponent random -j 4
```

The evaluated agent alternates between player 0 and player 1 to reduce
first-player bias.

By default, pairwise files are written under
`diagnostics/results/pairwise/<agent>_vs_<opponent>/`:

| File | Contents |
|---|---|
| `summary.json` | Win/draw/loss rates, Wilson 95% confidence interval, position split, mean turns, remaining pips, choice-opportunity totals, and `duration_s`. |
| `games.csv` | Compact one-row-per-game data with position, result, turns, initial hands as JSON arrays, and final pip counts. |
| `cumulative_rates.png` | Win/draw/loss rates over time. |
| `result_distribution.png` | Final result counts. |
| `wins_by_position.png` | Win rate as player 0 vs. player 1. |
| `game_lengths.png` | Turn-count histogram. |
| `choice_opportunities.png` | Histogram of draw/pass/choice opportunities for the evaluated agent. |

## Interpretation

Small samples are noisy. Prefer at least several hundred games when comparing
two checkpoints. If confidence intervals overlap heavily, the result should be
treated as inconclusive.

The `self_play_evaluation/` subfolder contains a helper script for comparing
two RL training regimes: pure self-play and direct training against the
heuristic agent.

## RL Hyperparameter Sweep

`hyperparameter_sweep.py` trains a fresh RL checkpoint per sweep point (one
axis of `training.self_play`'s hyperparameters varied at a time: learning
rate, reward schema, or gamma, holding the other two at their baseline), runs
the whole sweep once with the actor-critic value head on and once off, and
benchmarks `heuristic`, `neural`, and each freshly trained `rl` checkpoint
against `random` and in self-play:

```bash
python -m diagnostics.hyperparameter_sweep
python -m diagnostics.hyperparameter_sweep --rl-iterations 300 --diagnostic-games 1000
```

Every point loads the same supervised checkpoint and explicitly ignores any
older RL file at its target path, keeping the comparison centered on the
tested hyperparameters rather than prior RL history.

Every record — the exact RL hyperparameters used plus every matchup's
win/draw/loss rates — is appended to a single JSON array on disk
(`--output`, default `diagnostics/results/hyperparameter_sweep.json`), so
repeated invocations accumulate a growing log instead of overwriting it.
Trained checkpoints are written under `--checkpoint-dir` (default
`models/hyperparameter_sweep/`). Run `python -m diagnostics.hyperparameter_sweep
--help` for the full flag list.

### RL Sweep Comparative Table

`rl_sweep_table.py` is the counterpart for `train_script/run_rl_parameter_sweep.sh`
(a separate, bash-driven sweep — see `train_script/README.md`): that script
writes one `sweep_run.json` (hyperparameters) + `summary.json` (rl-vs-random
results) pair per sweep point under `diagnostics/results/<run_name>/`. This
module discovers every such pair and joins its data. Raw CSV/JSON keep one row
per trained model. The console, PNG, and PDF reduce clutter by grouping runs that
differ only in games per iteration into one row, with win-rate percentage
columns labelled `40`, `80`, and `160`:

```bash
python -m diagnostics.rl_sweep_table
python -m diagnostics.rl_sweep_table --results-dir diagnostics/results --output-dir /tmp/report
```

Output defaults to `diagnostics/results/rl_sweep_table/` and includes
`rl_sweep_table.csv`, `rl_sweep_table.json`, `rl_sweep_table.png`, and
`rl_sweep_table.pdf`. `train_script/run_rl_parameter_sweep.sh` invokes this
automatically as its final stage (`--skip-report` to opt out). On a shell
`--resume`, the same module validates the saved matchup, seed, requested game
count, result totals and rates, complete games CSV, requested plots, sweep
metadata, and numbered model identity before an existing diagnostic is reused.
New metadata includes the model SHA-256 checksum; older output uses conservative
artifact timestamps for backward-compatible validation.
