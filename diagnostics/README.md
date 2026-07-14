# Diagnostics

Diagnostics compare agents over many two-player domino games and write compact
metrics, CSV data, and plots. Three modes control the number of matchups without
changing the number of games played inside each selected matchup.

## Supported Agents

| Name | Implementation |
|---|---|
| `rl` | `RLAgent` loaded from the RL self-play checkpoint and used in evaluation mode. |
| `neural` | `NeuralAgent` loaded from supervised-learning weights. |
| `random_nn` | `RandomNeuralAgent` using the supervised architecture with fixed, untrained random weights. |
| `heuristic` | `StrategicAgent`, the handcrafted rule-based agent. |
| `random` | Uniform random legal move. |

The old `greedy` baseline is no longer available in diagnostics. The pairwise
helper still accepts the legacy alias `sl` for `neural`, but new commands and
reports use `neural`.

## Diagnostic Modes

The mode is an optional positional argument:

| Command | Matchups |
|---|---:|
| `python -m diagnostics.evaluate` | 10: the historical upper triangle for `rl`, `neural`, `heuristic`, and `random`. |
| `python -m diagnostics.evaluate fast` | 2: `rl` vs `random` and `heuristic` vs `random`. |
| `python -m diagnostics.evaluate complete` | 15: the full upper triangle including `random_nn`. |

Every mode uses 10,000 games per matchup by default. Change that count with
`-n`:

```bash
python -m diagnostics.evaluate fast -n 5000
python -m diagnostics.evaluate complete -n 5000
```

Each matchup displays a `tqdm` progress bar. The command also prints a startup
RAM/GPU memory snapshot and writes elapsed seconds as `duration_s`.

Useful options:

```bash
python -m diagnostics.evaluate --help
python -m diagnostics.evaluate --seed 123
python -m diagnostics.evaluate complete --seed 123
python -m diagnostics.evaluate --no-pair-plots
python -m diagnostics.evaluate --output /tmp/domino_all_pairs
python -m diagnostics.evaluate --neural-weights models/domino_sl_weights.npz
python -m diagnostics.evaluate --rl-weights models/domino_rl_weights.npz
```

The output folder defaults to `diagnostics/results/all_pairs/`.
Reusing that folder replaces the aggregate report and removes pair folders that
do not belong to the selected mode, keeping its contents internally consistent.

| File or folder | Contents |
|---|---|
| `all_pairs_table.png` | Triangular image table with one win-rate number per evaluated matchup. |
| `choice_opportunities.png` | Aggregate histogram of draw/pass/choice opportunities across all evaluated matchups. |
| `all_pairs_matrix.csv` | One row per evaluated matchup. |
| `all_pairs_summary.json` | Full aggregate report with accumulated choice-opportunity stats, `duration_s`, and all pairwise summaries. |
| `pairs/<agent>_vs_<opponent>/` | Standard pairwise artifacts for each matchup. |

## Pairwise Helper

Use the helper directly when only one matchup is needed:

```bash
python -m diagnostics.pairwise --agent heuristic --opponent random
python -m diagnostics.pairwise --agent rl --opponent neural
python -m diagnostics.pairwise --agent neural --opponent random_nn
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

Older `compact_to_enumerated_counts.png`, `first_stock_draw_turns.png`, and
`first_stock_draw_final_state_counts.png` files are obsolete. A new diagnostic
run removes those names from its output folders.

## Interpretation

Small samples are noisy. Prefer at least several hundred games when comparing
two checkpoints. If confidence intervals overlap heavily, the result should be
treated as inconclusive.

The `self_play_evaluation/` subfolder contains a helper script for comparing
two RL training regimes: pure self-play and direct training against the
heuristic agent.
