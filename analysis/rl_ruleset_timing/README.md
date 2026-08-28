# RL ruleset timing analysis

This directory contains a fixed-seed, subprocess-isolated timing and early
learning comparison of the four supported rulesets. It focuses exclusively on
reinforcement learning: no dataset generation or supervised training is part
of the measured runs.

The benchmark matches the important canonical `forever` RL defaults: GPI
2,000, 10 CPU rollout workers, GPU updates, 16 PPO epochs, and the
`heuristic,recent` opponent pool. Every ruleset is trained from a deterministic
random policy with its ruleset-default architecture for 60,000 games (30
iterations). Two repetitions use the same game seed, and their execution order
is reversed in the second pass to reduce warm-cache and time-of-run bias.

Every iteration policy plus the exact zero-game initial policy is retained.
All 31 policies are byte-identical between repetitions. One copy of each is
then evaluated deterministically against the same fixed panel of 10,000
random-opponent games, which supplies the learning curves without perturbing
training randomness or timing.

Commands:

```bash
./.venv/bin/python analysis/rl_ruleset_timing/benchmark_30_iterations.py
./.venv/bin/python analysis/rl_ruleset_timing/evaluate_learning_curves.py
./.venv/bin/python analysis/rl_ruleset_timing/analyze.py
```

Raw weights, metrics, profiles, logs, and manifests stay below
`raw_30_iterations/`; compact pairwise summaries stay below
`curve_diagnostics/`. The earlier 12-iteration pilot remains preserved in
`raw/`. The analysis script writes compact CSV/JSON summaries, twelve figures,
and `REPORT.md` in this directory. Original model and dataset directories are
never read or modified.
