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

Per-stage evidence lives in `results/<stage>/`.
