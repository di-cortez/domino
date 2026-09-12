# RL performance optimization

Measurements behind the staged PPO and pipeline speedups proposed in
`references/atualizacoes/atualizacoes_1209/run_time/`. Every stage removes
work whose result was discarded, unnecessary synchronization, or unnecessary
waiting; none may change the learning algorithm, rewards, seeds, opponent
allocation, minibatches, epoch budget, or the whole-buffer KL stop.

Each stage is one commit, measured against the commit before it.

## Tools

All three tools take the checkout under test from `PYTHONPATH` and never add
their own repository to `sys.path`, so the same file measures a reference
`git worktree` and the candidate tree. They write only where their output
arguments point.

| File | Purpose |
|---|---|
| `ppo_harness.py capture` | Plays real learner games once with a copied SL checkpoint and freezes the decisions. |
| `ppo_harness.py run` | Restores the frozen policy before every measured iteration and applies `update_from_samples`, recording per-array SHA-256, every deterministic PPO metric, and timings. |
| `ppo_harness.py compare` | Exact comparison first; otherwise per-metric absolute/relative errors and per-array parameter errors. |
| `ppo_harness.py timing` | Alternates checkouts in fresh processes (order reversed on odd repetitions), one untimed warm-up update per process, and reports medians and quartiles. |
| `correctness_suite.sh` | Five variants on GPU plus three CPU cases, reference versus candidate. |
| `training_smoke.py` / `smoke_suite.sh` | Real `train()` runs, uninterrupted and stopped/resumed, fingerprinting final weights, deterministic metrics rows, and the warmup trace. |
| `save_stage_results.py` | Copies one stage's compact evidence into `results/<stage>/` with scratch paths redacted. |

Harness variants:

| Variant | Learning rate | Entropy | Baseline | Epochs | Dropout | Purpose |
|---|---:|---:|---|---:|---:|---|
| `production` | 0.0008 | 0 | batch-mean | 16 | 0 | The measured forever runs |
| `hot_lr` | 0.03 | 0 | batch-mean | 16 | 0 | KL stop and norm clipping |
| `entropy_shared_critic` | 0.003 | 0.01 | value-head | 8 | 0 | Entropy gradient, shared critic |
| `own_critic_dropout` | 0.003 | 0 | value-head-own-nn | 8 | 0.1 | Separate critic, dropout draws |
| `no_up_entropy_dropout` | 0.003 | 0.02 | value-head-no-up | 4 | 0.1 | Stopped critic, both regularizers |

Training smokes (double-three, SL seed 42, 2,000 GPI, 8 iterations, stopped
after 3, four CPU rollout workers):

| Smoke | Device | Settings | Exercises |
|---|---|---|---|
| `gpu_warmup` | GPU | lr 0.01, `--warmup-lr` exponent 3, hold 2, cooldown 1, default KL threshold | Warmup promotions read PPO's `max_approx_kl` |
| `gpu_hot_critic` | GPU | lr 0.05, value-head, entropy 0.01 | KL stops at epochs 1, 5, 7, 8; one at 0.01522 against the 0.015 limit |
| `cpu_own_critic` | CPU | lr 0.05, value-head-own-nn, dropout 0.1, 6 epochs, 6 iterations | CPU path, separate critic |

## Reproducibility baseline

Commit `36a8160`. Python 3.10.12, NumPy 2.2.6, CuPy 14.1.1 (CUDA runtime
12.9, driver 535.309.01), NVIDIA GeForce RTX 3050 6GB Laptop GPU, Intel Core
i7-13650HX (20 logical CPUs). Workload buffers: double-six 2,000 games versus
random, 8,327 decisions; double-three 2,000 games, 3,201 decisions.

Before any change, two independent processes of the reference agreed byte for
byte in every harness case, every smoke, and every full-versus-split pair.

Timings were taken while an unrelated long training run occupied the same
machine, so absolute values are inflated (the reference optimizer step costs
2.24 ms here against 1.45 ms in that run's own profile). Only the interleaved
ratios are meaningful, and the whole-buffer evaluation is bimodal (about 20 ms
or 36 ms per call) depending on whether that run's own GPU work overlaps, which
hides small changes in the total update time. The optimizer-step time per step
is the steadier signal for stages B and C. From stage C on, `--cpu-affinity
0-11` pins the measured processes to the performance cores of this hybrid CPU.

## Results

| Stage | Correctness | Median PPO update (production, GPU) | Notes |
|---|---|---|---|
| B: discard unused minibatch metrics | Byte-identical in all harness cases and smokes, warmup trace included | 1.213 s -> 0.920 s (0.76x); optimizer step 2.24 ms -> 1.43 ms | Removes 7 policy and 2 critic host transfers per optimizer step |
| C: exact-zero entropy fast path | Byte-identical in all harness cases and smokes, warmup trace included | Unpinned 0.995 s -> 0.977 s, step 1.53 -> 1.32 ms; pinned 0.811 s -> 0.760 s, step 1.49 -> 1.44 ms (q1 1.40 -> 1.35) | About eight fewer kernel launches per step; the total is within the evaluation's noise |
| D: skip the redundant unmasked softmax | Byte-identical in all harness cases and smokes, warmup trace included | Pinned median 0.757 s -> 0.788 s (noise), q1 0.744 -> 0.725 s, min 0.736 -> 0.703 s; step 1.41 -> 1.29 ms; evaluation min 20.1 -> 19.2 ms | Five fewer kernel launches per forward in every optimizer step and evaluation partition |
| E: 4,096-decision whole-buffer evaluation | Weights, optimizer counters, KL stops, metrics rows, and warmup promotions identical; metrics within the tolerances below | 0.897 s -> 0.567 s (0.63x); evaluation 24.6 -> 6.9 ms per call; peak CuPy pool 11 -> 40 MiB | Sweep: 2,048 gives 0.632 s, 8,192 gives 0.534 s at 49 MiB; 4,096 captures 91% of the 8,192 gain |
| F: device-side evaluation accumulators | Byte-identical to E in all harness cases and smokes (reference: stage E smokes); at 512 partitions byte-identical to D | Alone at 512: evaluation median 29.8 -> 18.5 ms, q1 20.4 -> 18.2, min 18.6 -> 17.9. After E: update median 0.491 -> 0.477 s, q1 0.474 -> 0.444, min 0.441 -> 0.440; evaluation median 6.6 -> 4.4 ms, min 3.9 -> 4.1 | Mostly removes contention-sensitive spread; the kernel launches, not the transfers, dominate what remains |
| G1: validate decisions once per buffer | Byte-identical to F in all harness cases and smokes | Update median 0.519 -> 0.430 s, min 0.423 -> 0.382 s; step median 1.47 -> 1.22 ms, min 1.21 -> 1.07 ms; evaluation min 3.89 -> 2.88 ms | Two blocking host transfers fewer per optimizer step and per evaluation partition |
| G2: sliced evaluation partitions | Byte-identical to G1 in all harness cases and smokes; a strided-view variant was byte-identical too | Evaluation min 2.70 -> 2.40 ms, median 3.32 -> 2.94 ms; update min 0.364 -> 0.355 s | Views measured 2.37 ms, no faster than contiguous slices, so the contiguous layout of a gathered batch is kept |

### Combined PPO stages B-G2

`36a8160` against `65df615` on the same workloads, pinned and interleaved
(`results/combined_b_to_g2/`):

| Workload | Median PPO update | Optimizer steps | Whole-buffer evaluation | Peak CuPy pool |
|---|---|---|---|---|
| GPI 2,000 (8,327 decisions), 8 process pairs | 1.295 s -> 0.420 s (0.32x) | 0.639 s -> 0.311 s; 2.49 -> 1.21 ms per step | 0.592 s -> 0.047 s; 37.0 -> 2.9 ms per call | 11 -> 40 MiB |
| GPI 8,000 (33,204 decisions), 5 process pairs | 4.891 s -> 1.632 s (0.33x) | 2.554 s -> 1.304 s; 2.46 -> 1.25 ms per step | 2.097 s -> 0.132 s; 131 -> 8.3 ms per call | 29 -> 58 MiB |

Weights, optimizer counters, and KL stop epochs are identical to `36a8160` in
every harness case; the metric differences are exactly stage E's. The training
smokes keep identical weights and metrics rows against `36a8160`; only the
warmup trace's full-precision KL differs, by stage E's amount, with every
promotion on the same iteration. The percentages of the individual stages must
not be added: each one changes what the next one is measured against.

Per-stage evidence lives in `results/<stage>/`.

### Stage E numerical tolerances

The same code at 512 decisions per partition is byte-identical to stage D, so
every difference below comes from partitioning alone: a GPU (or BLAS) float32
matrix product may round a column differently when the batch shape around it
changes, and each partition sum is float32 before the float64 total. Largest
differences observed against stage D across all harness cases:

| Statistic | Largest absolute | Largest relative |
|---|---:|---:|
| `approx_kl` | 7.0e-8 (`hot_lr`, KL 0.092) | 1.2e-4 (at KL 2.4e-6) |
| `clip_fraction` | 1.2e-4 (one decision of 8,327) | 4.0e-3 |
| `ratio_max` / `ratio_min` | 1.4e-4 / 9.8e-6 | 2.2e-5 / 1.9e-5 |
| `explained_variance` | 1.8e-7 | 9.5e-5 (at values near 0.002) |
| `policy_loss` | 8.4e-8 | 1.9e-5 |
| `legal_logit_deficit_max` | 2.5e-5 | 5.4e-6 |
| `value_std` | 1.9e-7 | 7.2e-6 |
| `entropy`, `ratio_mean`, `value_loss`, `value_mean` | below 1.3e-7 | below 2.4e-7 |

No KL stop epoch changed. In the warmup smoke, per-iteration `max_approx_kl`
moved by at most 8e-11 and every promotion landed on the same iteration
(`warmup_trace_difference.json`). A stop or promotion decided within about
1e-7 of its threshold could still flip; none was observed.
