# RL ruleset timing report

## Conclusion

The compact rulesets are faster, and their end-to-end time is almost perfectly explained by the number of trainable decisions per game (Pearson r = 0.993 across the four rulesets). Double-three is 2.80x faster than double-six. It is not 6.75x faster like the parameter count might suggest because it still produces 1/2.58 as many trainable decisions and pays the same 16-epoch PPO orchestration and GPU-launch structure.

One non-monotonic hotspot is real: a double-four learner decision spends about 322.9 us updating the exact opponent model, 2.08x the double-six cost per call. This makes double-four rollout scale worse than its decision count predicts.

The learning curves do not show compact rulesets learning faster per fixed iteration. After 30 iterations, double-three gained 2.39 win-rate points versus random while double-six gained 4.37. Double-three is nevertheless the most efficient per wall minute and per trainable decision because each iteration is much cheaper.

## Benchmark controls

- 4 rulesets x 2 subprocess-isolated repetitions; 60,000 games and 30 iterations per repetition.
- Fixed seed 20,260,828, GPI 2,000, 10 CPU rollout workers, GPU policy updates, and 16 PPO epochs.
- Default `heuristic,recent` opponent buckets; no decision restarts, diagnostics, dataset generation, or supervised training.
- Random ruleset-default networks were initialized from the fixed seed; all 31 policies from iteration zero through 30 were byte-identical between repetitions.
- Worker deep profiling sampled exactly 1,875/60,000 games per run (one in 32); the normal hot path remains uninstrumented for the other games.
- Every retained checkpoint was evaluated deterministically against the same fixed panel of 10,000 random-opponent games.
- Retaining every policy required an experimental numbered checkpoint after every iteration; capture-adjusted efficiency removes 29 of those 30 serializations while retaining the normal final save.

## Interpretation limits

These are controlled RL smoke benchmarks, not forecasts of final playing strength. They isolate the cost of the training loop by excluding dataset generation, supervised learning, and diagnostics. A mature run has a larger recent-opponent pool and a more developed policy, so absolute throughput can move somewhat; the ruleset geometry, PPO work per collected decision, and exact-model representation effect measured here remain the relevant mechanisms. Two repetitions and alternating execution order keep the comparison stable, but a four-point correlation should be read as a strong engineering clue rather than a statistical law.

## Measured totals

| Ruleset | Parameters | Decisions/game | Median time | Games/s | Rollout | PPO | PPO share |
|---|---:|---:|---:|---:|---:|---:|---:|
| double-three | 12,356 | 1.528 | 47.21 s | 1271.5 | 9.43 s | 33.60 s | 71.2% |
| double-four | 22,750 | 2.244 | 81.14 s | 739.8 | 26.78 s | 48.98 s | 60.4% |
| double-five | 47,754 | 3.000 | 103.88 s | 577.7 | 30.28 s | 65.43 s | 63.0% |
| double-six | 83,384 | 3.948 | 132.13 s | 454.1 | 35.84 s | 84.20 s | 63.7% |

## Early learning curves versus random

| Ruleset | Initial | Iteration 30 | Gain | Gain 0-10 / 10-20 / 20-30 | adjusted pp/min | pp/100k decisions | Time to +1 pp | Time to +2 pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| double-three | 49.86% | 52.25% | +2.39 pp | +0.69 / +1.13 / +0.57 | 3.17 | 2.61 | 23.6 s | 34.5 s |
| double-four | 49.40% | 52.05% | +2.65 pp | +0.75 / +1.05 / +0.85 | 2.03 | 1.97 | 34.5 s | 64.4 s |
| double-five | 50.42% | 53.60% | +3.18 pp | +1.25 / +1.28 / +0.65 | 1.93 | 1.77 | 33.9 s | 58.2 s |
| double-six | 50.42% | 54.79% | +4.37 pp | +1.99 / +1.36 / +1.02 | 2.12 | 1.84 | 21.7 s | 47.0 s |

The first hope is not supported in iteration units: double-six improves most rapidly per iteration, and reaches +1 point in only five iterations. A fixed 2,000-game iteration is not equal work across rulesets: double-six collected about 236,890 trainable decisions over the experiment, versus 91,671 for double-three, and therefore performed many more optimizer steps.

The compact ruleset does recover an efficiency advantage after normalizing the work. Double-three gains 3.17 points per capture-adjusted training minute and 2.61 points per 100,000 decisions. The other rulesets produce only 1.93-2.12 points per minute and 1.77-1.97 points per 100,000 decisions. It is the fastest to reach a +2-point relative improvement (34.5 seconds), although double-six narrowly reaches +1 point first (21.7 versus 23.6 seconds).

Saving all 31 policies added an artificial 1.93, 2.82, 5.14, and 8.47 seconds to D3 through D6, respectively. The adjusted per-minute comparison removes the 29 intermediate saves a normal 30-iteration invocation would not make. Threshold times above are the directly observed conservative values, including those saves; this choice cannot create the double-three efficiency advantage.

There is an early flattening hint, not an asymptote measurement. Double-three gains +1.13 points in iterations 10-20 but only +0.57 in iterations 20-30; its fitted late slope falls to 0.038 points per iteration. Double-five also flattens in the last third. Double-four and double-six still have clearer positive late slopes. These 30 iterations are enough to reject an orders-of-magnitude learning-speed advantage, but not enough to estimate final ceilings or prove convergence.

## What explains the scaling

1. **Decision density dominates.** Double-three has 1.528 trainable decisions/game versus 3.948 for double-six. PPO time falls from 84.20 s to 33.60 s, almost exactly with this ratio.
2. **PPO is the majority cost in every ruleset.** It consumes 60-71% of invocation time. The optimizer steps scale with buffer decisions, while the full buffer is evaluated once after every epoch. That evaluation alone consumes about 36% of PPO time in all four rulesets.
3. **The GPU does not scale with parameter count alone.** These networks are all small; fixed kernel launches, mask validation, host transfers, minibatch materialization, and 16 epoch-level evaluations remain. Double-three has only 14.8% of double-six parameters but needs 39.6% of its PPO time.
4. **Generic orchestration is small.** Excluding the explicitly measured experimental per-iteration checkpoint capture, match planning, metrics, archive work, buffer preflight, and final writes together are a small fraction. There is no sign that a large double-six-only operation is accidentally running unchanged in every compact ruleset.

## The double-four exact-model anomaly

The fixed representation threshold is `SWITCH_TO_MU_MAX_HANDS = 500` in `middleware/opponent_model.py:35`. At the end of a non-terminal turn, `_maybe_switch_to_mu()` converts when the raw hidden-hand upper bound is at most 500 (`middleware/opponent_model.py:1314-1345`). Initial upper bounds are:

| Ruleset | Initial hidden-hand upper bound | Immediate relationship to threshold | Exact-model us/learner call | Exact-model policy share |
|---|---:|---|---:|---:|
| double-three | 15 | at/below 500 | 88.1 | 35.6% |
| double-four | 252 | at/below 500 | 322.9 | 67.9% |
| double-five | 5,005 | above 500 | 205.2 | 58.9% |
| double-six | 116,280 | above 500 | 155.3 | 52.6% |

Double-three converts to only 15 hidden hands and is cheap. Double-four can convert early to as many as 252 hands and then repeatedly filters/sums that dictionary. Double-five and double-six initially exceed the threshold, so they remain in the slot representation until the state becomes smaller. This explains why per-call exact inference peaks at double-four rather than increasing monotonically with ruleset size.

This is the clearest candidate for a future optimization experiment: keep double-four in the slot representation longer (or choose the representation from measured operation cost rather than hand count), then verify probability traces and fixed-seed RL weights remain identical. No production change was made here.

## PPO observations

- All 240 benchmark iterations completed 16 epochs; KL early stops: 0; gradient-clipped iterations: 0.
- `training/rl/ppo.py:847-923` performs minibatch updates and then a full-buffer evaluation after every epoch so KL stopping and final metrics use the complete buffer.
- Removing or reducing that evaluation would save substantial time but would weaken or redesign PPO control; it is not dead work under the current algorithm.
- Reducing PPO epochs would also accelerate every ruleset, but it changes the optimization budget rather than fixing a ruleset-specific inefficiency.

## Recommended next measurements before changing code

1. A controlled double-four model-only benchmark comparing the current threshold 500 with a threshold below 252, while asserting exact probabilities at every public action.
2. A short fixed-seed RL A/B for double-four after that change, requiring byte-identical trajectories/rewards and comparing rollout time only.
3. Separately investigate whether PPO full-buffer evaluation can share forward results with the last epoch without changing KL semantics.
4. Do not optimize generic session/reporting code first; it is too small to matter.

## Artifacts

- `raw_30_iterations/`: all eight isolated runs, 248 retained policy checkpoints, metrics, summaries, and logs.
- `curve_diagnostics/` and `learning_curve_raw.jsonl`: the 124 fixed-panel RL-vs-random evaluations.
- `run_summary.csv`: one row per repetition.
- `ruleset_summary.csv`: medians, ranges, and coefficients of variation.
- `learning_curve.csv` and `learning_curve_summary.csv`: point-level and ruleset-level learning results.
- `top_level_sections.csv`, `ppo_sections.csv`, and `rollout_sections.csv`: long-form profiles.
- `analysis_summary.json`: machine-readable conclusions and validation.
- Figures `01` through `12`: timing, scaling, rollout, PPO, learning curves, and efficiency comparisons.
