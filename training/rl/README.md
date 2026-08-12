# Reinforcement learning

Refines the supervised policy through on-policy RL against a snapshot pool or
the heuristic player, with masked PPO by default.
Owned by `training/rl/`.

| File | Purpose |
|---|---|
| `training_loop.py` | Orchestrates the exact-budget on-policy training lifecycle and delegates its specialized phases. |
| `config.py` / `cli.py` | Validate side-effect-free RL options and own the standalone/canonical shared argument definitions. |
| `constants.py` | Fixed worker-autotuning implementation invariants shared by RL modules. |
| `pool.py` | Separates durable opponent identities and bucket retention from physical shared-memory slots. |
| `matchmaking.py` | Tracks smoothed difficulty evidence and builds immutable exact-GPI match plans. |
| `checkpoint_archive.py` | Stores a bounded, progressively thinned history independently of exact-resume checkpoints. |
| `rollout.py` | Finalizes rewards and trajectories and plays one CPU-only self-play or heuristic-opponent training game. |
| `iteration.py` / `session.py` | Run one update and prepare fresh/resumed training state respectively. |
| `resume.py` | Loads compatible policies and atomically saves, validates, and restores exact numbered RL resume pairs. |
| `reporting.py` | Owns iteration summaries, durable metrics JSONL writes, worker metadata aggregation, and cumulative RL runtime profiles. |
| `ppo.py` | Builds immutable decision buffers, selects minibatches, manages GPU/RAM storage, and performs KL-limited PPO epochs. |
| `adaptive_tuning.py` | Selects rollout workers with isolated seed streams, state restoration, safety checks, and `adaptive_tuning.json`. |
| `parallel.py` | Shares frozen policy snapshots with deterministic CPU-only rollout workers and retains completed real games across memory fallback. |


Run:

```bash
python -m training.rl.cli
python -m training.rl.cli --compact
python -m training.rl.cli --rl-workers auto --seed 123
python -m training.rl.cli --rl-workers 4 --device cpu
python -m training.rl.cli --gpi 1000
python -m training.rl.cli --fresh-from-sl
python -m training.rl.cli --fresh-from-sl --opponent-buckets random --difficulty-weight 0
python -m training.rl.cli --fresh-from-sl --opponent-buckets heuristic,random,recent
python -m training.rl.cli --dropout 0.1 --weight-decay
```

Default behavior:

- if a compatible `models/domino_rl_weights.npz` exists, resume from it;
- otherwise warm-start from a compatible `models/domino_sl_weights.npz`;
- train against the fixed heuristic and the 200 most recent frozen learner
  snapshots, splitting half the games uniformly and half by measured
  difficulty;
- use a fixed GPI of 2,000 and select rollout workers with isolated,
  discarded benchmarks;
- update the policy with masked PPO minibatches for at most four epochs;
- save `models/domino_rl_weights.npz`.

That compatibility-first policy is the default for the standalone module.
Pass `--fresh-from-sl` to ignore an older RL checkpoint and start from the SL
weights, or `--continue-existing-rl` to state the historical behavior
explicitly. A new canonical pipeline run always starts from its compatible
seed-addressed supervised checkpoint. Canonical continuation is deliberately
separate and complete: use `--resume` or `--resume-from`, never the
weights-only `--continue-existing-rl` path.
When starting fresh, the old RL file is ignored but kept intact until the new
checkpoint is atomically saved, so an interrupted run does not erase a usable
model.

The learner uses `RLAgent(..., mode="training")`: it samples from the masked
policy and stores trajectory steps. Frozen pool opponents use
`mode="stochastic_evaluation"`: they sample from their masked policies but do
not build training masks or store trajectories. This exposes the learner to
more of each snapshot's policy distribution without retaining unused opponent
experience.

## Parallel rollout generation

All games in an iteration use one immutable learner policy, so rollout work is
independent until batch aggregation. Before workers start, `matchmaking.py`
allocates the exact GPI across selected buckets and members and deterministically
orders one immutable assignment per absolute game ID. Workers attach NumPy
views to a physical shared-policy bank, never see the GPU, and execute those
assignments without selecting opponents. Memory fallback retries each unfinished
assignment unchanged. The parent sorts results by game ID and remains solely
responsible for performance evidence, gradients, admission, checkpoints,
logging, and GPU allocations.

`heuristic` contains only `StrategicAgent` and consumes no policy-bank slot.
`random` contains only the uniform `RandomAgent`, likewise consumes no
policy-bank slot, and is available for explicit experiments without joining
the default bucket selection.
`recent` begins with a frozen copy of the initial learner, admits every
non-empty completed update, and retains the latest 200 snapshots with logical
FIFO eviction. The current mutable learner is never a bucket member.

`--opponent-buckets` accepts any non-empty combination of `heuristic`, `random`,
and `recent` as a comma-separated selection. Input order is canonicalized,
while duplicates, unknown names, and an empty selection are rejected.
`--difficulty-weight` controls the exact
convex allocation: `0` is entirely uniform, `0.5` is half uniform and half
difficulty-based, and `1` is entirely difficulty-based. GPI remains the single
game budget; matchmaking never adds evaluation games.

GPI is never autotuned. Canonical pipelines and direct RL training accept
`--gpi` with choices `100, 200, 400, 600, 800, 1000, 2000`, defaulting to
`2000`.

Worker tuning tests 1, 2, 4, 6, ... workers, never exceeding 20, on exactly 1%
of the real game budget per candidate. Starting from the one-worker baseline,
each larger candidate must improve throughput by at least 10% over the
previously accepted candidate; the first smaller gain stops tuning and is not
selected. Benchmark games use independent deterministic seed streams and are
discarded. Weights, optimizer, RNG, opponent pool, and real counters are
restored and verified before training begins. Results are saved as
`adaptive_tuning.json`.

Runtime RAM pressure during real rollout generation terminates the current
pool, keeps completed game ids, halves the worker count, and retries only
unfinished games.

Per-game SplitMix64-style seeds are derived from the run seed, iteration, and
game id. Parent aggregation is ordered, so the same seed produces bit-identical
checkpoints with one or multiple workers, including after fallback. Useful
controls are:

| Flag | Meaning | Default |
|---|---|---:|
| `--gpi` | Fixed positive number of games per RL iteration | `2000` |
| `--opponent-buckets` | Named active bucket selection | `heuristic,recent` |
| `--difficulty-weight` | Uniform/difficulty allocation mixture in `[0, 1]` | `0.5` |
| `--rl-workers` | CPU-only rollout workers or `auto` | `auto` |
| `--retune-workers` | Explicitly rerun saved worker tuning on resume | off |
| `--rl-memory-reserve-mb` | Host RAM that must remain free | `512` |
| `--rl-estimated-worker-mb` | Conservative worker-memory estimate for preflight | `256` |
| `--rl-max-worker-rss-mb` | Runtime RSS ceiling for one worker | `1024` |

## Numbered checkpoints and exact resume

Direct-module calls keep the existing single-file checkpoint behavior unless
`--numbered-checkpoints` is requested. The canonical pipeline always uses
interruption-safe numbered pairs. Each save adds the absolute completed iteration
to the name, such as `model_iter000050.npz`, and atomically publishes a paired
`model_iter000050.resume.npz`. The state file contains a SHA-256 checksum,
every computation-affecting RL/PPO setting, completed real games, optimizer
state, RNGs, supervised-checkpoint hash, fixed GPI, tuned workers, rolling logs,
and the exact opponent-policy pool. The newest pool state replaces the previous
one to bound disk use; numbered policy files remain available.

To continue manually, pass only the matching pair. The checkpoint supplies the
completed iteration, game budget, algorithm, hyperparameters, worker policy,
and all other training settings:

```bash
python -m training.rl.cli \
  --resume-weights-path models/example_iter000500.npz \
  --resume-state-file models/example_iter000500.resume.npz
```

Resume validates the checksum and configuration before loading anything,
restores optimizer/RNG/pool state, continues at the next absolute game id, and
reuses the saved GPI/workers without rerunning their autotune, and ignores any
conflicting training options. Loading an ordinary `.npz` through
`--continue-existing-rl` restores only weights and cannot reconstruct the
former in-memory pool.

Diagnostics and the UI use `mode="evaluation"`, which always selects the
highest-probability legal action and stores no trajectory. Their results
therefore avoid action-sampling noise. Checkpoints are saved without running
an auxiliary matchup during training.

The command prints startup memory, tuning throughput, checkpoint-to-checkpoint
time, and total elapsed time. Every ten iterations it aggregates PPO decisions,
requested/effective minibatches, optimizer steps, epochs, KL stops, clipping,
entropy, gradient norms, buffer location/bytes, rollout time, and update time.
Every iteration is also appended to `<weights>_training_metrics.jsonl`.

That JSONL uses compact schema version 2. Its first line is a header object with
the ordered column list, complete training configuration, and canonical
configuration SHA-256. Each later line is an array in that column order. The
format preserves full JSON numeric precision and every PPO iteration, including
KL values, KL early-stop state, completed epochs, minibatch sizes, optimizer
steps, entropy, clipping, and gradient norms, while avoiding repeated field
names and duplicate aliases in every row. Exact resume validates the header
hash and truncates any uncommitted tail beyond the checkpoint.

Canonical runs additionally maintain
`models/rl/<run>/diagnostics/runtime_profile.json`. The report is written
atomically after every RL segment and periodic RL-vs-random diagnostic. It
contains one session per pipeline process plus cumulative totals. RL timing is
split into initialization/resume, adaptive tuning, runner setup, policy sync,
rollout execution and parent aggregation, reward and buffer preparation, PPO,
pool refresh, checkpoint I/O, metrics, callbacks, and shutdown. PPO is split
again into storage preparation, minibatch materialization, optimizer work,
synchronization, whole-buffer evaluation, KL control, and cleanup. The worker
side of rollouts is split into rules/state generation, learner and opponent
decisions, reward shaping, engine transitions, and episode finalization. Each
RL policy decision is split again into exact-opponent-model update, encoding,
network inference, legal-action selection, and trajectory recording. Worker
totals are exact summed CPU-seconds, so they are intentionally distinct from
(and can exceed) parent wall time when several workers overlap. Per-turn
subphases use a deterministic 1-in-32 game sample; the JSON records both its
coverage and sampled CPU denominator. This avoids making per-turn profiling a
measurable bottleneck while still retaining thousands of sampled games in a
normal checkpoint window. Optimizer steps and
whole-buffer evaluation are also split at their existing GPU synchronization
boundaries; the profiler does not add extra synchronizations merely to improve
attribution. Existing long runs start their fine-grained coverage at the game
counter where the profiler was introduced; earlier games are recorded as
unprofiled instead of being estimated.

Pass `--compact` to suppress iteration and checkpoint lines while retaining
worker-autotuning messages, one absolute iteration progress bar, and one final
summary.

The learner trajectory stores only real decisions. Draw, pass, and single-option
tile plays are forced actions, so `RLAgent` returns them directly without
calling the network or saving a trajectory step. Each saved step carries the
legal-action mask and decision turn. Sampling and gradient calculation use the
same masked policy distribution.

`PolicyNetwork` uses masked PPO by default with the critic disabled. At the
start of an iteration, one policy is frozen for all rollouts. Every real
decision stores its legal mask and collection-time probability metadata; draw,
pass, and single-choice plays never enter the buffer. PPO uses `old_log_prob`
to compare the updated policy with the rollout policy. Advantages use the
existing finalized reward and are normalized once over the complete iteration.

With `--ppo-max-epochs 1`, the same fresh on-policy decisions go directly to the
full-buffer policy-gradient update. Exactly one optimizer step is attempted per
non-empty iteration, and none of `training/rl/ppo.py`'s buffer, ratio, clipped
surrogate, KL, minibatch, or whole-buffer evaluation paths run. Checkpoint,
optimizer, RNG, adaptive-tuning, opponent-pool, and canonical resume behavior
remain unchanged apart from recording `reinforce_v1` instead of `ppo_v1`.

The canonical contiguous buffer stays in RAM. If it fits within 70% of reported
free VRAM and a dry first-minibatch workspace probe succeeds, a complete GPU
copy is retained across epochs. Otherwise minibatches stream from RAM. No
fallback restarts a partially applied epoch.

Requested minibatches are fixed by the implementation as
`clamp(ceil(actual_games / 125), 4, 16)`, further capped to keep roughly 128
decisions per minibatch. PPO implementation constants live in `ppo.py`; the
1% worker benchmark fraction and 10% worker acceptance gain live in
`constants.py`. None are experiment or CLI parameters. Each epoch visits every
decision exactly once with a deterministic new permutation. PPO uses clip
epsilon `0.2`, reports target KL `0.01` as an informational reference, and
does not start another epoch after the
completed epoch's whole-buffer approximate KL exceeds `0.015`. Direct
standalone and finite canonical profiles run at most four epochs by default;
the canonical `forever` profile runs at most 16. An explicit
`--ppo-max-epochs` overrides the profile default within the supported 1–16
range.

Enable the optional PPO actor-critic with:

```bash
python -m training.pipeline forever --value-head --run-name critic
# Or directly:
python -m training.rl.cli --value-head
```

This adds a linear `V(s)` head over the last hidden layer. The current
finalized policy reward is the value target, and the masked policy update uses
`reward - V(s)` as its advantage. PPO saves the collection-time values, uses a
clipped value loss, and evaluates value loss, clipping, prediction moments, and
explained variance after each epoch. The value-loss coefficient defaults to
`0.5` (`--value-coef`). Checkpoints contain `Wv` and `bv`. Combining
`--ppo-max-epochs 1 --value-head` keeps the one-full-buffer actor-critic path.

Policy-only loading ignores `Wv`/`bv`, while value-head loading initializes
them to zero when they are absent. This permits mode changes without changing
the policy architecture, but clean comparisons should still start from the
same supervised checkpoint and use separately archived RL outputs.

## Optional RL controls

The default command uses PPO, the original reward constants, no terminal-reward
discount, gradient clipping at norm `5.0`, and whole-buffer advantage
normalization. Rollouts remain parallel while all updates stay in the parent:

| Flag | Meaning | Default |
|---|---|---:|
| `--fresh-from-sl` / `--continue-existing-rl` | Force initialization from SL or allow a compatible existing RL checkpoint | continue existing RL (standalone); canonical continuation uses `--resume` |
| `--gamma` | Terminal-reward discount per remaining real decision (`1.0` = no discount) | `1.0` |
| `--reward-schema` | Named preset for the terminal/event reward constants: `default` (the table below), `sparse` (win/loss only, no draw/pass shaping or pip penalty), or `shaped` (doubles the draw/pass shaping rewards) | `default` |
| `--ppo-max-epochs` | `1` selects one-update REINFORCE; `2`–`16` select masked PPO | `4` standalone/finite, `16` forever |
| `--value-head` | Train a linear critic with PPO or REINFORCE | off |
| `--weight-decay [COEFFICIENT]` | Decoupled L2 shrink on every weight matrix and `Wv` after clipping; shared with supervised training | off (`0.0001`) |
| `--dropout [RATE]` | Hidden-layer dropout shared with supervised training | off (`0.1`) |
| `--value-coef` | Critic loss coefficient when the value head is enabled | `0.5` |
| `--normalize-advantages` / `--no-normalize-advantages` | Standardize once over the complete iteration buffer | on for PPO |
| `--total-training-games` | Exact real-game budget; final iteration may be partial | `100000` |
| `--gpi` | Fixed positive number of games per RL iteration | `2000` |
| `--moving-average-window` | Trailing-iteration window for the value-loss/win-rate moving averages printed in the iteration log | `10` |
| `--seed` | Fix `random`/NumPy state, for reproducible comparisons between hyperparameter configurations | unset |
| `--device` | Array backend: `auto` matches `GPU_ENABLED` exactly (CuPy when installed, else NumPy); `cpu`/`gpu` force one backend regardless of what's installed/enabled globally | `auto` |

```bash
python -m training.rl.cli --gamma 0.97 --reward-schema shaped --seed 42
```

A point-in-time value loss or win rate is dominated by batch noise; the
iteration log always reports `reward mean/std/min/max` and a trailing moving
average of value loss and win rate next to the raw values, so a plateau can be
judged from the average rather than a single noisy line. In detailed
`training.rl.cli` output, a value-head run also prints the count and
mean/std/min/max of the pre-update `V(s)` predictions for every displayed
iteration. These are the same predictions used in `reward - V(s)`; the report
reuses that forward pass rather than evaluating the buffer again.

## Device selection (`--device`)

`--device auto` (the default) reproduces the original behavior exactly:
CuPy when installed, NumPy otherwise, same as `GPU_ENABLED` elsewhere in the
project. `--device cpu` or `--device gpu` force one backend for that run
regardless of what's installed, independently of the parent
`SupervisedNeuralNetwork` class used by supervised training (which is
unaffected and still always follows `GPU_ENABLED`). `--device gpu` raises a
clear error if CuPy isn't installed. Rollout workers remain CPU-only, while
PPO updates may use the GPU. Consult the run's cumulative runtime profile
before changing devices: it records CPU/GPU optimizer-call counts and separates
rollout rules, exact-model work, inference, buffer transfer, backpropagation,
parameter updates, and metric transfers.

`training.rl.cli` also accepts `--iterations`, `--total-training-games`,
`--gpi`, `--opponent-buckets`, `--difficulty-weight`, `--learning-rate`,
`--entropy-coef`, `--log-interval`, `--checkpoint-interval`,
`--sl-weights-path`, and `--rl-weights-path`; see
`training/rl/cli.py:add_optional_rl_arguments` for the authoritative
definitions, or run `python -m training.rl.cli --help`.

The former GPI-autotune options (`--games-per-iteration`, `--adaptive-gpi`,
`--no-adaptive-gpi`, `--gpi-candidates`, `--gpi-benchmark-games-target`,
`--retune-gpi`, and `--retune-all`) were removed. The former iteration-based
`--pool-interval` and auxiliary `--evaluation-games` options were also removed.
Existing numbered resume states that contain `pool_interval` are rejected rather
than silently reinterpreted as a game count; start a new run instead.

The former binary opponent flags (`--training-opponent` and
`--max-pool-size`) were replaced by named buckets and the one public allocation
weight. Bucket capacity, difficulty calibration/smoothing, retention, archive
cadence, and integer tie-breaking are internal versioned policies and appear in
metrics/resume manifests rather than as flags.

Every tenth completed update also writes the same admitted policy identity to
`checkpoint_archive/`. This archive is independent of exact resume: it has a
1-GiB internal limit and deterministically thins old history in exponentially
widening tiers while retaining dense recent coverage, its baseline, newest,
and any pinned entries. Resume reconciles away archive descendants newer than
the selected exact checkpoint.

The RL reward now uses a uniform terminal reward plus temporally decayed local
draw/pass shaping. For each real decision at turn `d_i`, a later event at turn
`t_e` contributes:

```text
c_e * EVENT_REWARD_DECAY ** (t_e - d_i - 1)
```

with `EVENT_REWARD_DECAY = 0.90`. An immediately following event therefore has
exponent `0` and receives the full event reward. By default (`--gamma 1.0`)
the terminal result is not discounted and is applied uniformly to every real
decision in the game; passing `--gamma` below `1.0` discounts it per
remaining real decision instead (see "Optional RL controls" above).

Reward constants (the `default` reward schema; `--reward-schema` selects an
alternate preset, see above):

| Event | Reward |
|---|---:|
| terminal win | `+0.50` |
| terminal loss | `-0.50` |
| opponent draw | `+0.02` |
| opponent pass | `+0.10` |
| learner draw | `-0.02` |
| learner pass | `-0.10` |
| final remaining pips | `-0.001 * remaining_pips` |

Multiple local events are summed. A learner draw/pass penalty is applied to all
earlier real decisions with the same decay rule, not just to the most recent
decision. The final pip penalty is applied to the learner's own final hand. The
number of legal choices does not rescale a decision's return. The final
training weight for every decision is:

```text
policy_reward = terminal_reward + local_reward
```

PPO uses that value as the pre-normalization advantage. For one decision:

```text
ratio = exp(new_log_prob - old_log_prob)
surrogate = min(ratio * advantage, clip(ratio, 0.8, 1.2) * advantage)
L = -mean(surrogate) - entropy_coef * mean(entropy)
```

Gradient clipping remains active in `PolicyNetwork` to limit large updates.

The snapshot pool normally lives only in memory. The opt-in numbered resume
state described above is the exception: it serializes and restores that pool
for exact interruption recovery.
## Host and device memory

RL training performs a host-memory preflight for the shared snapshot bank and
expected batch, then checks the actual workspace before each `hstack`. With
`--device auto`, less than 256 MiB of effective free VRAM causes an announced
CPU fallback; explicit `--device gpu` fails early instead. Diagnostics later
run in separate CPU-only processes and never consume training VRAM.
The policy architecture is selected once for supervised training and adopted
unchanged by this stage; see
[`../supervised/README.md`](../supervised/README.md) for the
`--hidden-layers`/`--hidden<n>-size` controls. The shared
`--weight-decay`/`--dropout` flags are documented in
[`../utils/README.md`](../utils/README.md).
