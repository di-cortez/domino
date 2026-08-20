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
| `champion_evaluation.py` | Races candidate snapshots against the fixed heuristic on shared seed panels and ranks the survivors. |
| `checkpoint_archive.py` | Stores a bounded, progressively thinned history independently of exact-resume checkpoints. |
| `rollout.py` | Finalizes rewards and trajectories and plays one CPU-only self-play or heuristic-opponent training game. |
| `restarts.py` | Defines immutable same-iteration opponent-decision restart records. |
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
python -m training.rl.cli --ruleset double-three --fresh-from-sl
python -m training.rl.cli --fresh-from-sl
python -m training.rl.cli --fresh-from-sl --opponent-buckets random --difficulty-weight 0
python -m training.rl.cli --fresh-from-sl --opponent-buckets heuristic,random,recent
python -m training.rl.cli --fresh-from-sl --opponent-buckets heuristic,recent,medium_term
python -m training.rl.cli --fresh-from-sl \
  --opponent-buckets heuristic,recent,medium_term,historical_uniform,champion_vs_heuristic
python -m training.rl.cli --fresh-from-sl \
  --opponent-buckets heuristic,recent,medium_term,historical_uniform,champion_vs_heuristic,champion_vs_learner
python -m training.rl.cli --fresh-from-sl --opponent-decision-restarts
python -m training.rl.cli --dropout 0.1 --weight-decay
```

Default behavior:

- if a compatible `models/domino_rl_weights.npz` exists, resume from it;
- otherwise warm-start from a compatible `models/domino_sl_weights.npz`;
- if `--fresh-from-sl` is requested but that file does not exist, create a
  seeded random policy using the selected ruleset's default hidden sizes;
- train against the fixed heuristic and the 200 most recent frozen learner
  snapshots, splitting half the games uniformly and half by measured
  difficulty;
- use a fixed GPI of 2,000 and select rollout workers with isolated,
  discarded benchmarks;
- update the policy with masked PPO minibatches for at most four epochs;
- save `models/domino_rl_weights.npz`.

`--ruleset` accepts the four registry names and defaults to double-six.
Compact standalone paths include the name (for example
`models/domino_rl_double-three_weights.npz`). The ruleset is durable resume
identity and is propagated to every engine, learner, heuristic, snapshot,
worker, pool member, and checkpoint. Resume cannot switch it, and policy
weights are never converted across variants.

That compatibility-first policy is the default for the standalone module.
Pass `--fresh-from-sl` to ignore an older RL checkpoint and start from the SL
weights, or `--continue-existing-rl` to state the historical behavior
explicitly. A missing SL path now means a clean random-policy start rather than
an error; `--seed` makes those initial random weights reproducible. A new
canonical pipeline run always starts from its compatible
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
The three neural buckets cover disjoint chronological regions, so a policy is
never played from two bands at once.

`recent` begins with a frozen copy of the initial learner, admits every
non-empty completed update, and retains the latest 200 snapshots with logical
FIFO eviction. The current mutable learner is never a bucket member.

`medium_term` is a delayed archive-backed window rather than an immediate
milestone bucket. At every archive cadence boundary it selects the 200 newest
archive milestones no newer than `N - 200` completed iterations, in
oldest-to-newest order. A fresh milestone therefore joins no band on the
iteration it is produced; it waits until the recent band has moved past it. At
GPI 2,000 the band is a nominal four-million-game horizon (3,980,000 games
between its oldest and newest members when full), starting roughly 400,000
games behind the present.

`historical_uniform` represents everything older than both bands: archive
records no newer than `N - 2200` completed iterations. When more than 200
qualify it keeps exactly 200 deterministic representatives spaced uniformly in
completed-game coordinates, not in record rank, so a thinned era widens its own
gaps instead of borrowing resolution from a denser one. The oldest and newest
eligible records are always kept, targets are compared with integer arithmetic,
and ties resolve toward the older checkpoint. Unlike `medium_term`, its span
grows with the run, so no fixed-cadence nominal horizon applies to it.

Identities in the delayed bands are rehydrated from the checkpoint archive by
stable checkpoint ID, with the recorded hash, weight names, and shapes verified
before the policy reaches the shared bank. Both band memberships are reconciled
as one transaction: incoming weights are fully validated in ordinary host
memory before any logical state changes, and a slot is released only once no
bucket still references its identity. A checkpoint that leaves `medium_term`
for `historical_uniform` in the same refresh keeps its durable identity, its
bank slot, and its accumulated difficulty evidence.

The two champion buckets are not chronological. Each holds up to 200 snapshots
selected for measured strength, so they are the buckets that may intentionally
share identities with `recent`, `medium_term`, `historical_uniform`, and with
each other. An overlapping identity is not duplicated storage: it keeps one
record, one bank slot, one difficulty tracker, and simply receives matchmaking
games through each membership it holds. The three historical bands remain
pairwise disjoint among themselves, so pool observability reports
`forbidden_historical_overlap_counts` (always zero) separately from
`champion_overlap_counts` (expected to be non-zero).

`champion_vs_heuristic` selects for strength against the fixed heuristic.

Its members are chosen by a racing event, not by a cadence. Every successful
post-update snapshot that joins `recent` also joins a pending candidate list;
once that list holds 50 distinct snapshots, the next iteration races them
against the heuristic in four rounds:

    50 candidates x   500 games -> keep 40
    40 candidates x   500 games -> keep 30
    30 candidates x   500 games -> keep 20
    20 candidates x 2,000 games -> keep  5

which is exactly 100,000 evaluation games per completed event. Every candidate
in a round faces the identical seeded panel of deals and plays exactly half its
games in each seat, so the comparison is of play rather than of luck, and each
round ranks on its own games alone: a lucky screening round is never carried
forward. The final five are therefore ranked on their final 2,000 games, and
only that final win rate is stored as the champion score. All five winners are
admitted unconditionally, even when every one of them is weaker than every
incumbent; if the bucket is full, the five incumbents with the lowest stored
heuristic win rate leave first. Eviction chooses only among the incumbents that
were already there, never from the union of old and new. The candidate list is
consumed in the same transaction that admits the winners, so consecutive events
never share a candidate.

`champion_vs_learner` uses the same candidate stream, the same funnel, and the
same 100,000-game cost, but races each candidate against the **current
post-update learner**, frozen for the whole event. Because its target moves
between events, an old admission win rate says nothing about present strength:
65% against the learner of iteration 500 is not comparable with 55% against the
learner of iteration 5,000. It therefore stores no durable admission score. When
its bucket is full, the incumbents that leave are the ones the learner currently
finds easiest, read from the existing decayed `OpponentPerformanceTracker`
difficulty, ties broken by opponent ID. That is the one place the two champion
buckets genuinely differ: a fixed target admits a comparable stored score, a
moving target does not.

> **Not implemented yet.** `champion_vs_learner` is registered, selectable, and
> reserves its 200 bank slots, but no racing event runs against the learner
> target yet, so the bucket stays permanently empty and receives no games.
> Selecting it today only costs memory. This note goes away when the evaluator
> lands.

Both champion buckets start empty, so like the delayed bands they are configured
but unavailable during warm-up and receive no training games until their first
event completes at the fiftieth successful update. Each owns its own pending
candidate list and its own event index; a snapshot joins every selected champion
queue, and committing one bucket's event clears only that bucket's queue.

Racing games are evaluation games. They never enter the GPI budget, the
`bucket_results` metrics rows, the difficulty evidence, or the PPO buffers, and
they are timed in their own `champion_evaluation` runtime-profile section
rather than inside rollout time. A completed event prints its own console
summary regardless of `--log-interval`. Exact resume restores the champion
membership, the stored win rates, the pending candidate list, and the completed
event index, alongside the uniform rotation anchors, so a resumed run races the
same candidates on the same seed panels a single uninterrupted run would have.

The delayed bands are genuinely empty during warm-up, and empty buckets are not
padded with duplicated recent policies. Matchmaking distinguishes *configured*
buckets from *available* ones and redistributes the complete GPI budget across
the available buckets, giving an empty bucket zero games and a `[0, 0, 0]`
metrics row until the archive makes its band real.

`--opponent-buckets` accepts any non-empty combination of `heuristic`, `random`,
`recent`, `medium_term`, `historical_uniform`, `champion_vs_heuristic`, and
`champion_vs_learner` as a comma-separated selection. Input order is canonicalized, while duplicates,
unknown names, and an empty selection are rejected. At least one bucket that is
available from the first iteration (`heuristic`, `random`, or `recent`) is
required, because the archive-backed bands cannot bootstrap their own training
history. Selecting either champion bucket additionally requires `recent`,
because their candidates are the snapshots `recent` is already holding.
Selecting all five neural buckets reserves 1,000 opponent slots conservatively,
without compressing the overlap the champion buckets are allowed to have; at the
default architecture that is one shared segment of about 318 MB.
`--difficulty-weight` controls the exact
convex allocation: `0` is entirely uniform, `0.5` is half uniform and half
difficulty-based, and `1` is entirely difficulty-based. GPI remains the single
game budget; matchmaking never adds evaluation games.

Both components split their budget across available buckets by the same
largest-remainder rule. Inside a bucket they differ. The difficulty component
weights each member by its own measured difficulty, so hard opponents keep
concentrating games. The uniform component gives every member
`floor(bucket_games / members)` games and hands the remainder to consecutive
members walking forward from a persisted anchor, so the identities receiving
the extra game rotate across iterations instead of always being the same
stable-order prefix. Without that rotation a 200-member bucket receiving 66
uniform games would serve the same 66 identities forever and never reach the
rest.

The anchor is the last opponent ID that received an extra game, held per
bucket, and it is exact-resume state. When the anchored identity has left the
bucket the rotation continues at that ID's insertion point, so FIFO eviction
and archive rebalancing move the cursor forward instead of resetting it. A
bucket whose share divides evenly has no remainder and does not advance its
anchor, and an unavailable bucket never advances at all. Rotation is committed
only after the complete match plan validates. The per-iteration console line
reports each bucket's current anchor after `uniform rotation after`.

Allocation fixes counts only. The complete assignment list is still shuffled
once with a stable seed after every count is final.

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

With `--opponent-decision-restarts`, the normal GPI games and match plan remain
unchanged. During them, workers capture every exact pre-action state where the
source opponent has two or more legal tile placements. After all normal games,
the same frozen learner continues once from each state in the source opponent's
seat while the exact source counterpart takes the old learner seat. Normal and
restart decisions are concatenated and updated once. Restart episodes are
ephemeral: they never increment GPI progress, win rate, difficulty evidence,
pool cadence, or checkpoint cadence. Their counts, decisions, and wall time are
reported separately and persisted cumulatively for exact resume.

| Flag | Meaning | Default |
|---|---|---:|
| `--gpi` | Fixed positive number of games per RL iteration | `2000` |
| `--opponent-buckets` | Named active bucket selection | `heuristic,recent` |
| `--difficulty-weight` | Uniform/difficulty allocation mixture in `[0, 1]` | `0.5` |
| `--opponent-decision-restarts` | Add one same-iteration continuation from every genuine opponent tile-choice state | off |
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

That JSONL uses compact schema version 6. Its first line is a header object with
the ordered metric columns, complete training configuration, canonical
configuration SHA-256, ordered opponent-bucket names, bucket-result columns,
and nominal uniform/difficulty budgets. Each later line is an array in metric
column order. Its `bucket_results` value is itself a name-free numeric matrix:
one `[games, wins, losses]` row per header bucket. Per-opponent identities,
difficulty evidence, and allocations belong to exact resume state and are not
duplicated into every historical metrics row.

The format preserves full JSON numeric precision and every PPO iteration,
including KL values, KL early-stop state, completed epochs, minibatch sizes,
optimizer steps, entropy, clipping, and gradient norms. Exact resume validates
the header hash and stream-copies only committed rows through the selected
checkpoint into a replacement file. It never materializes the complete JSONL
or its decoded per-iteration objects merely to truncate an uncommitted tail.

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

`DRAW` in this paragraph means buying a tile from the stock. There is no drawn
game in the current ruleset: every terminal state has player 0 or player 1 as
its winner, including blocked games. Matchmaking difficulty evidence and its
resume state consequently store only learner wins and learner losses. Any
other terminal winner value is rejected as an engine-contract violation.

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
remain unchanged apart from recording `reinforce_v1` instead of
`ppo_v2_decision_minibatches`.

The canonical contiguous buffer stays in RAM. If it fits within 70% of reported
free VRAM and a dry first-minibatch workspace probe succeeds, a complete GPU
copy is retained across epochs. Otherwise minibatches stream from RAM. No
fallback restarts a partially applied epoch.

Requested minibatches are based only on the combined decision count and fixed
as `min(ceil(decisions / 512), 256)`. Every optimizer minibatch targets 512
decisions. A final remainder of 256–511 decisions is retained; a remainder
below 256 is omitted from that epoch's optimizer steps. The full buffer,
including an omitted optimizer tail, is still used by the post-epoch KL
evaluation. Every epoch uses a fresh deterministic permutation, so omitted
decisions normally change between epochs. PPO implementation constants live
in `ppo.py`; the 1% worker benchmark fraction and 10% worker acceptance gain live in
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
| `--alpha` | Convex mix of the two reward components per decision: `0` trains on the terminal outcome alone, `1` on local draw/pass shaping alone | `0.5` |
| `--event-reward-decay` | Per-turn decay crediting a draw/pass event back to the real decisions preceding it (`0` credits only the immediately preceding decision) | `0.90` |
| `--opponent-decision-restarts` | Continue once from every same-iteration pre-action opponent state with at least two tile plays; the learner swaps seats and all decisions join one update | off |
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
python -m training.rl.cli --gamma 0.97 --alpha 0.3 --event-reward-decay 0.8 --seed 42
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

The iteration-zero frozen policy is the archive baseline. Every successful
update at an absolute multiple of ten iterations writes the same admitted
policy identity to `checkpoint_archive/`. This archive is independent of exact
resume: it has a 1-GiB internal limit and deterministically thins old history in
exponentially widening tiers while retaining dense recent coverage, its
baseline, newest, and any pinned entries. The pin set is the union of every
active `medium_term` and `historical_uniform` member plus the staging
milestones that have not entered `medium_term` yet — the ones inside the recent
band width, and any archived identity `recent` still holds after iterations
that produced no trainable decisions. Without those staging pins a tight byte
limit could thin a milestone during the exact window between its archive write
and its delayed admission. Pins are published only after both band memberships
are final, so a checkpoint moving from `medium_term` to `historical_uniform` is
never briefly unpinned. Exact resume restores all active weights from its own
state and then rebuilds archive pins, so it never depends on a thinned archive
file. Resume also reconciles away archive descendants newer than the selected
exact checkpoint.

Resume states written before the disjoint bands (versions 10 and 11) are
rejected rather than reinterpreted. A state that selected the superseded
overlapping `medium_term` cannot be converted without silently changing what
its saved memberships mean, and no migration is implemented for the other
selections either; start a new run.

The RL reward now uses a uniform terminal reward plus temporally decayed local
draw/pass shaping. For each real decision at turn `d_i`, a later event at turn
`t_e` contributes:

```text
c_e * EVENT_REWARD_DECAY ** (t_e - d_i - 1)
```

with `EVENT_REWARD_DECAY = 0.90` by default, tunable through
`--event-reward-decay`. An immediately following event therefore has exponent
`0` and receives the full event reward.

The two components are then combined into the total reward of each decision:

```text
R_T = (1 - ALPHA) * DEFAULT_GAMMA ** k * R_f + ALPHA * R_l
```

where `R_f` is the terminal reward, `k` the number of real decisions still
remaining after this one, and `R_l` the summed decayed local event reward.
By default (`--gamma 1.0`) the terminal result is not discounted and is
applied uniformly to every real decision in the game; passing `--gamma` below
`1.0` discounts it per remaining real decision instead. `--alpha` trades the
two components off against each other: `0` trains on the terminal outcome
alone and `1` on local shaping alone (see "Optional RL controls" above).

Reward constants:

| Event | Reward |
|---|---:|
| terminal win | `+1.00` |
| terminal loss | `-1.00` |
| opponent draw | `+0.20` |
| opponent pass | `+0.10` |
| learner draw | `-0.20` |
| learner pass | `-0.10` |
| final remaining pips | `-0.05 * remaining_pips` |

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
