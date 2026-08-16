# Full-game pair dynamics and timing

This directory is a self-contained analysis of all unordered pairings among
the repository's `random`, `heuristic`, and `neural` players. It does not write
to datasets, training checkpoints, or model-run directories.

## Experiment design

- Six matchups, with 10,000 complete games in each one.
- All matchups reuse seeds 42 through 10,041, so their deals are paired.
- Mixed matchups swap player 0/player 1 on alternating games.
- Games run sequentially in one process to avoid worker contention in the
  wall-time measurements.
- The copied neural policy runs on CPU, matching rollout-worker inference.
- The network is loaded and warmed before timing begins.
- Every move stores non-negative integer microseconds from
  `time.perf_counter_ns()` for state construction, legal-action generation,
  agent decision, engine transition, and the complete turn.
- Timing covers simulation code, not JSON encoding or file I/O, unless the
  field explicitly says `end_to_end`.

The policy provenance is in `neural_policy_source.json`. The copy used by the
experiment is `neural_policy_weights.npz`; keeping it here makes the analysis
reproducible even after the training run is cleaned.

## Scripts

From the repository root:

```bash
.venv/bin/python analysis/full_game_pair_dynamics/generate_full_game_pairs.py
.venv/bin/python analysis/full_game_pair_dynamics/analyze_full_games.py
.venv/bin/python analysis/full_game_pair_dynamics/plot_full_game_analysis.py
```

The generator accepts `--games`, `--base-seed`, `--weights`, `--output-dir`,
and `--matchups`. The analyzer accepts `--input-dir`, `--output`, `--matchups`,
and `--pretty`. All scripts default to this directory.

## Artifacts

- `*_full_games.jsonl`: one complete, timed history per game and matchup.
- `full_game_pair_generation_manifest.json`: seeds, machine/timer provenance,
  policy identity, hashes, file sizes, and raw throughput.
- `full_game_pair_analysis.json`: compact validated statistics for state
  dynamics and timings by matchup, turn, agent, and decision type.
- `01_*.png` through `13_*.png`: comparative state and timing figures.

The report's terminal state is a separate state at turn `t`; only turns with an
action have move timings. A turn's `turn_wall` encloses the four named timed
components plus a small amount of Python bookkeeping. Agent timing includes
the complete `choose_move()` path, including exact-opponent-model work when the
agent actually evaluates a voluntary choice.
