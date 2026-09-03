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
| `baseline.py` / `reward_lookup_tables/` | Select policy-gradient baselines and load the fixed state-conditioned unit-reward artifacts. |
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

The learner target is published into the shared bank's current-policy region
immediately before the event and left alone until it finishes, so all 100,000
games of one event face exactly the same weights. Both seats play in
deterministic evaluation mode.

Both champion buckets start empty, so like the delayed bands they are configured
but unavailable during warm-up and receive no training games until their first
event completes at the fiftieth successful update. Each owns its own pending
candidate list and its own event index; a snapshot joins every selected champion
queue, and committing one bucket's event clears only that bucket's queue.

All five winners of a learner event are admitted unconditionally too, exactly
as for the heuristic bucket, so a weak generation still displaces older
champions and the bucket keeps tracking the run. Eviction chooses only among the
incumbents that were already there.

Racing games are evaluation games. They never enter the GPI budget, the
`bucket_results` metrics rows, the difficulty evidence, or the PPO buffers, and
each event is timed in its own `champion_vs_heuristic_evaluation` or
`champion_vs_learner_evaluation` runtime-profile section rather than inside
rollout time. The parallel summary counts the two separately as well, so an
iteration that runs both reports two 100,000-game costs rather than one opaque
200,000. A completed event prints its own console summary regardless of
`--log-interval`, labelled with the direction of the number it reports:
`candidate win rate vs current learner` is the candidate's, which is the
opposite direction from the tracker's `estimated_win_rate`.

Exact resume restores both pending candidate lists and both completed event
indices independently, both champion memberships, the heuristic score map, the
opponent performance evidence, and the uniform rotation anchors. The learner
bucket has no durable admission score to restore. A resumed run therefore races
the same candidates on the same seed panels a single uninterrupted run would
have, and evicts the same incumbents.

The delayed bands are genuinely empty during warm-up, and empty buckets are not
padded with duplicated recent policies. Matchmaking distinguishes *configured*
buckets from *available* ones and redistributes the complete GPI budget across
the available buckets, giving an empty bucket zero games and a `[0, 0, 0]`
metrics row until the archive makes its band real.

`--opponent-buckets` accepts any non-empty combination of `heuristic`, `random`,
`recent`, `medium_term`, `historical_uniform`, `champion_vs_heuristic`, and
`champion_vs_learner` as a comma-separated selection, and defaults to
`heuristic,recent`. Input order is canonicalized, while duplicates,
unknown names, and an empty selection are rejected. At least one bucket that is
available from the first iteration (`heuristic`, `random`, or `recent`) is
required, because the archive-backed bands cannot bootstrap their own training
history. Selecting either champion bucket additionally requires `recent`,
because their candidates are the snapshots `recent` is already holding.
Selecting all five neural buckets reserves 1,000 opponent slots conservatively,
without compressing the overlap the champion buckets are allowed to have; at the
default architecture that is about 318 MB. The whole bank is **one** shared
memory segment whatever the capacity, not one per slot: a POSIX segment costs
two file descriptors in every process that maps it, so a per-slot layout would
need roughly 2,000 descriptors at full capacity and exceed the common 1,024
limit in the parent and again in every worker.
A canonical run selecting every bucket looks like:

```bash
python -m training.pipeline forever \
  --opponent-buckets \
  heuristic,recent,medium_term,historical_uniform,champion_vs_heuristic,champion_vs_learner \
  --difficulty-weight 0.5
```

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

The default command uses PPO, turn-based local and terminal reward discounting,
gradient clipping at norm `5.0`, and whole-buffer advantage normalization.
Rollouts remain parallel while all updates stay in the parent:

| Flag | Meaning | Default |
|---|---|---:|
| `--fresh-from-sl` / `--continue-existing-rl` | Force initialization from SL or allow a compatible existing RL checkpoint | continue existing RL (standalone); canonical continuation uses `--resume` |
| `--gamma-f` | Terminal/final-reward discount using the mode's second distance metric | `0.95` |
| `--reward-eta` | Convex mix of the terminal and immediate returns per decision: `0` trains on the terminal outcome alone, `1` on draw/pass shaping alone | `0.5` |
| `--gamma-i` | Local/immediate-event discount using the mode's first distance metric | `0.90` |
| `--reward-distance-mode` | Distance units in `gamma_i`/`gamma_f` order: `turn-turn`, `decision-decision`, `turn-decision`, or `decision-turn` | `turn-turn` |
| `--terminal-empty-hand-weight` | Weight `a_E` of the empty-hand terminal component; only the `a_E`/`a_B` ratio matters | `1.0` |
| `--terminal-blocked-weight` | Weight `a_B` of the blocked terminal component; `a_E` and `a_B` cannot both be zero | `1.0` |
| `--immediate-draw-weight` | Weight `a_D` of draw events; only the `a_D`/`a_P` ratio matters | `1.0` |
| `--immediate-pass-weight` | Weight `a_P` of pass events; `a_D` and `a_P` cannot both be zero | `1.0` |
| `--opponent-decision-restarts` | Continue once from every same-iteration pre-action opponent state with at least two tile plays; the learner swaps seats and all decisions join one update | off |
| `--ppo-max-epochs` | `1` selects one-update REINFORCE; `2`–`16` select masked PPO | `4` standalone/finite, `16` forever |
| `--value-head` | Train a linear critic with PPO or REINFORCE | off |
| `--baseline` | Term subtracted from each return: zero, constant, batch mean, fixed lookup, or one of three critic wirings | configuration-dependent historical default |
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
python -m training.rl.cli --gamma-f 0.97 --reward-eta 0.3 --gamma-i 0.8 --reward-distance-mode decision-turn --seed 42
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

The RL reward combines a discounted terminal return with temporally discounted
draw/pass event shaping. A "decision" is exactly one compressed PPO trajectory
step: a learner tile choice with at least two legal tile actions. Forced tile
plays, draws, passes, and opponent actions are not decisions. For decision `t`:

```text
G(t) = (1 - eta) * G_T(t) + eta * G_I(t)
```

with the terminal half

```text
G_T(t) = gamma_f ** k_T(t) * U_T
U_T    = (a_E * R_E + a_B * R_B) / max(a_E, a_B)
```

and the immediate half

```text
G_I(t) = (a_D * G_D(t) + a_P * G_P(t)) / max(a_D, a_P)
G_D(t) = sum over later draw events of gamma_i ** k_e(t) * r_D(e)
G_P(t) = sum over later pass events of gamma_i ** k_e(t) * r_P(e)
```

`G_D` and `G_P` already contain their `gamma_i`-discounted event sums, so `eta`
mixes two returns that are each complete. The four layers stay separate on
purpose: `a_E`/`a_B` and `a_D`/`a_P` say what a result is worth, `gamma_f` and
`gamma_i` say how far back it is credited, and `eta` says how the two
subsystems trade off. No one of them is used to compensate for another.

The single `--reward-distance-mode` flag selects both distances. Its first word
always controls `gamma_i`; its second always controls `gamma_f`:

| Mode | Local `gamma_i` distance | Terminal `gamma_f` distance |
|---|---|---|
| `turn-turn` | engine turns between the decision and event | engine turns between the decision and terminal state |
| `decision-decision` | later real learner decisions before the event | later real learner decisions before termination |
| `turn-decision` | engine turns | later real learner decisions |
| `decision-turn` | later real learner decisions | engine turns |

For turn distance, an event or terminal state immediately after a decision has
distance zero: `event_turn - decision_turn - 1` or
`terminal_turn - decision_turn - 1`. Decision distance counts only later
compressed learner decisions. Multiple local events are summed.

The new-run defaults are:

```text
gamma_i = 0.90
gamma_f = 0.95
eta = 0.50
reward_distance_mode = turn-turn
terminal_empty_hand_weight = 1.0
terminal_blocked_weight    = 1.0
immediate_draw_weight      = 1.0
immediate_pass_weight      = 1.0
```

Thus both components use actual engine time by default. `reward_eta` trades the
components off: `0` trains on terminal outcome alone and `1` on local shaping
alone. Runs/checkpoints created before `reward_distance_mode` existed are
unambiguously restored as `turn-decision`, with the historical defaults
`gamma_i=0.90`, `gamma_f=1.0`, and `reward_eta=0.50`; resume never silently
changes their objective.

Runs and checkpoints created before the reward redesign are a different case
and are **rejected** rather than restored. They record none of the four weights
because their terminal reward was the binary outcome minus `0.05` per pip of
the learner's own hand, an objective the empty-hand/blocked decomposition
replaces. Filling the missing weights with today's defaults would make such a
run look identical to a new one, so `RLTrainingConfiguration.from_mapping`
raises instead. Start a new RL run; an existing supervised checkpoint is
unaffected and can still seed it.

### Reward components

Every component is normalized before any weight is applied, so a weight
expresses importance and nothing else.

| Component | Value | Applies to |
|---|---:|---|
| `R_E` empty-hand win | `+1` | the game ended because a hand emptied |
| `R_E` empty-hand loss | `-1` | the game ended because a hand emptied |
| `R_B` blocked win | `+m(dp)` | the game ended blocked |
| `R_B` blocked loss | `-m(dp)` | the game ended blocked |
| `r_D` opponent draw | `+1` | one draw event |
| `r_D` learner draw | `-1` | one draw event |
| `r_P` opponent pass | `+1` | one pass event |
| `r_P` learner pass | `-1` | one pass event |

The two terminal components are mutually exclusive: an empty-hand ending has
`R_B = 0` and a blocked ending has `R_E = 0`. The blocked magnitude is

```text
m(dp) = 0.1 + 0.9 * min(dp / (2 * max_pip), 1)
dp    = loser_final_pips - winner_final_pips >= 0
```

so it lies in `[0.1, 1]` and saturates at a margin of `2 * max_pip`, taken from
the ruleset rather than a table (`double-three` 6, `double-four` 8,
`double-five` 10, `double-six` 12). `dp = 0` is legal and maps to the `0.1`
floor: the engine compares pip totals first, so a blocked win resolved by
`blocked_fewest_tiles` or `blocked_last_valid_play` is a real but minimally
decisive win. All three blocked win reasons are blocked endings.

There is no longer a per-pip penalty on the learner's own final hand. The pip
story now lives entirely in the blocked margin, where it is zero-sum between
the seats, instead of being applied to both ending kinds against one seat only.

### Reward weights

| Flag | Symbol | Meaning |
|---|---|---|
| `--terminal-empty-hand-weight` | `a_E` | value of an empty-hand result |
| `--terminal-blocked-weight` | `a_B` | value of a blocked result |
| `--immediate-draw-weight` | `a_D` | value of one draw event |
| `--immediate-pass-weight` | `a_P` | value of one pass event |

Each pair is divided by its own larger member, so only the ratio inside a pair
reaches training: `--terminal-empty-hand-weight 2 --terminal-blocked-weight 1`
is the same objective as `1` and `0.5`. The weights are finite and
non-negative, they need not sum to one, and a pair may not be `(0, 0)`. A zero
on one side deletes that component only.

`G_I(t)` is deliberately not renormalized to `[-1, 1]`: one decision may be
followed by several draw and pass events, and the normalization is of event
values, not of the accumulated trajectory.

The optional `--baseline lookup-table` evaluates source-controlled histograms
conditioned only on the two hand sizes at each genuine learner decision. The
signed `empty_hand` and `blocked` unit components use the terminal clock and
`gamma_f`; signed pass/draw unit counts use the local clock and `gamma_i`. The
run's normalized scales and `reward_eta` are applied at runtime, so the same
artifact serves every reward configuration. The artifact checksum is part of
exact resume identity. See `reward_lookup_tables/README.md` for cell
resolution, artifact ownership, and why the packaged version 2 tables must be
rebuilt before this baseline can be selected.

A learner draw/pass penalty is applied to all earlier real decisions with the
selected decay rule, not just to the most recent decision. The number of legal
choices does not rescale a decision's return. The final training weight is:

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

## Numerical stability guards

The policy is trained in float32 by plain SGD, so a single non-finite value can
end a run that has already cost tens of millions of games. Five guards make
that impossible to do silently. None of them is a tunable: they are module
constants deliberately kept out of `rl_config`, because `rl_config` is hashed
into the run identity and any key added to it makes every existing run
unresumable.

Two distinct failures live here and must not be conflated. A **diverged
learner** loses its weights to NaN or infinity. An **unsamplable rollout
policy** keeps perfectly finite weights and simply becomes too confident for
float32 to represent on a non-maximizing action. The first four guards address
the former; the last addresses the latter, and no observable of the former
gives any warning of the latter.

| Guard | Where | Effect |
|---|---|---|
| Non-finite gradient rejection | `PolicyNetwork._apply_gradient_step` | A NaN or infinite gradient norm skips the optimizer step instead of being clipped. Clipping cannot repair either: `nan > clip` is False, so a NaN bypasses clipping, and `inf * (clip / inf)` is NaN. |
| Log-ratio bound | `PolicyNetwork.backward_ppo`, `PPO_LOG_RATIO_LIMIT = 20.0` | Bounds `log(pi_new / pi_old)` in the gradient path. The rollout floors the behavior probability at the smallest float32, so an unbounded ratio reaches `exp(87)`, and `exp(87) * advantage` overflows float32 into an infinite `active_weights`, where every illegal action then computes `0 * inf = NaN`. |
| Epoch rollback | `training/rl/ppo.py` epoch loop | If `evaluate_full_buffer` cannot evaluate the updated policy, the whole epoch is undone from a snapshot taken before its first minibatch and the iteration reports `diverged_epoch`. |
| Rollout diagnosis | `agents/rl_agent.py`, `NonFinitePolicyError` | Names the non-finite logits and counts them, instead of accusing the legal mask. After the max-subtraction `sum(exp(logits - max))` is at least 1.0 for every finite input, so the mask can never be the cause. |
| Masked rollout sampling | `_CPUInferencePolicy.cache`, `_masked_rollout_probabilities` | The worker-side policy publishes its logits, so the agent builds the sampling distribution over the legal subset instead of renormalizing the full-support softmax. The latter flushes a legal action to zero once it sits more than 87.3 nats below the global maximum, and a decision whose legal actions have all flushed has no mass left to sample. |

Reporting paths are deliberately **not** bounded. `evaluate_full_buffer` and
`log_ratio_statistics` keep the true, unclamped ratio, KL and clip fraction,
because those statistics are what a learning-rate study compares across runs.

`diverged_epoch` is separate from `stopped_by_kl` on purpose. A rolled-back
epoch is not a KL stop, and reporting it as one would corrupt the trust-region
statistics of the run.

### Metrics

The iteration log gains a numerical-health line:

```text
  PPO/10: policy weight max|W| 1.5925 avg/1.5925 max
  PPO/10: legal logit deficit 1.69 avg/1.91 max nats (float32 limit 87.34)
```

`policy_weight_max_abs` is `max(abs(W))` over the policy weights. The float32
forward pass overflows as a function of the largest weight times the largest
activation, so it is the direct precursor of a non-finite policy; the gradient
norm does not show it, because a run can drift for millions of games with
clipping never firing once.

`legal_logit_deficit_max` is `max(global logit) - max(legal logit)`, the exact
crash-proximity signal for the rollout sampler, and it is the one observable
that moves when a policy sharpens. `d6_maxwr_lr032` held `max|W|` between 1.97
and 2.66 for its whole 28,000,000-game life while this reached 84.5 against a
float32 limit of 87.34 -- so the weight signal alone gives no warning at all. A
fresh policy sits near 1.7. The threshold is `-log(tiny)`, a property of
float32 rather than a tuned number: past it a legal action's contribution to a
full-support softmax is denormal and heading for zero.

Three warning lines follow, printed **only** when they fire, so silence is the
healthy state:

| Warning | Meaning |
|---|---|
| `N minibatch(es) had a non-finite gradient and were skipped` | The gradient guard absorbed a state that would previously have written NaN into every weight. |
| `rolled back diverged epoch N` | The policy could not be evaluated on the data that trained it, and the epoch was undone. Not a KL stop. |
| `legal logit deficit N nats exceeds the float32 limit` | The policy is sharp enough that a full-support softmax can no longer represent some legal action. Sampling is unaffected while the rollout policy publishes its logits cache; it is a trend to watch, not a failure. |

Both warnings are emitted as `[numerical]` status messages rather than as part
of the iteration report, because the canonical pipeline runs the reporter with
`quiet=True` and only status messages survive that. They also ignore
`--log-interval`: a numerical failure is reported when it happens, not up to
ten iterations later. In a `python -m training.pipeline forever` log they look
like this:

```text
  [numerical] iteration 9632: 2 minibatch(es) had a non-finite gradient and were skipped; the weights were left untouched.
  [numerical] iteration 9632: rolled back diverged epoch 3 (PPO full-buffer metrics produced NaN/Inf.). The policy is the one that collected this buffer; training continues.
```

These five are deliberately **not** columns of `training_metrics.jsonl`. That
file is a fixed 59-column schema with a version that `_prepare_metrics_file`
re-validates on every resume, so adding columns would make every existing run
unresumable. The durable per-checkpoint history of `max(abs(W))` comes from
`analysis/rl_weight_health.py` instead, which reconstructs it from the saved
`.npz` files and therefore also works on runs that predate this change.

### Triaging a diverged run

`analysis/rl_weight_health.py` reports, per saved checkpoint, whether it is
finite and how large its largest weight is. It is read-only and never writes
into a run directory.

```bash
python -m analysis.rl_weight_health models/rl/<run directory>
python -m analysis.rl_weight_health models/rl/<run directory> --per-layer
```

For reference, every run in this repository -- the seven-point `double-three`
learning-rate grid from `lr = 0.001` to `lr = 0.064`, and eleven `double-six`
runs -- keeps `max_abs_weight` between 1.49 and 2.09 for its whole life, and
the largest gradient norm ever recorded across that grid is 2.90, well under
the clip at 5.0. A run outside those bands is the anomaly.

## GPU context loss

The guards above all assume the process still owns a working device. A
separate failure mode does not: the CUDA context can be destroyed underneath a
perfectly healthy run by an NVIDIA module reload from an unattended driver
upgrade, an Xid fault, a display-server restart, or a power or thermal event.
CuPy then reports `cudaErrorLaunchFailure: unspecified launch failure`.

It is not a training fault, and it must not be read as one. CUDA surfaces an
asynchronous failure at the next synchronization, so the traceback blames
whichever line happened to copy a scalar to the host -- in the
`d6_maxwr_lradaptativev4` run that was `_reward_signal_summary`, which is only
the first GPU touch of an iteration after the CPU rollout.

Nothing about the interrupted iteration can be saved. The policy weights and
the Adam moments live on the dead device, so a checkpoint cannot even be
written; recovery is entirely the last checkpoint, at most
`checkpoint_interval` iterations back.

| Piece | Where | Effect |
|---|---|---|
| Fault classification | `agents/nn.py`, `is_gpu_context_loss` | Separates a dead context from a full one by the CUDA status name. Both arrive as the same CuPy exception types, so `_is_backend_memory_error` used to rename a lost context into "exhausted memory at fixed batch N" and retry into a device that can no longer run a kernel. |
| Diagnosis | `training/rl/training_loop.py`, `GPUContextLostError` | Names the iteration, the games completed, and the checkpoint to resume from, and points at `dmesg`, the driver upgrade log, and `nvidia-smi`. The worker pool still closes through the existing `finally`. |
| Hard exit | `training/pipeline.py`, `run_cli`, exit code 70 | Leaves through `os._exit` on purpose. Unwinding runs CuPy's module destructors against the dead context, which turned one diagnosis into 76 identical `CUDA_ERROR_LAUNCH_FAILED` tracebacks in the log this was written from, and can hang instead of merely printing them. `main` itself stays exception-based so callers and tests keep ordinary control flow. |
| Restart | `train_script/run_forever_supervised.sh` | Resumes the run. Exit code 70 is the signal it keys on; it declines to restart on `SIGINT`/`SIGTERM` and stops when two consecutive attempts complete the same number of games. |

A GPU reset is a machine problem. Diagnose it with `sudo dmesg -T | grep -i
xid`, `grep -i nvidia /var/log/apt/history.log`, and `nvidia-smi -q -d
TEMPERATURE,POWER`; `HW Slowdown: Active` points at sustained-load cooling,
where capping the clocks buys stability at a small throughput cost.
