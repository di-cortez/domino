# Training

This folder contains the full training pipeline:

1. generate supervised examples from real heuristic decisions;
2. train a supervised neural policy;
3. refine that policy through self-play reinforcement learning.

From the repository root, the canonical pipeline runs the full sequence:

```bash
python -m training.pipeline small
python -m training.pipeline default
python -m training.pipeline big
python -m training.pipeline huge
python -m training.pipeline forever
```

`python -m train_script.run_pipeline` is an equivalent compatibility entry
point; the wrapper now lives with the other executable training drivers.

The quick profiles (`small` and `default`) deliberately do not reuse
supervised artifacts. Unless `--seed` is explicit, each invocation gets a new
seed, unique run directory, and run-local dataset/SL checkpoint. The long-run
profiles (`big`, `huge`, and `forever`) retain the stable seed-42,
seed-addressed 100,000-game assets and metadata-validated reuse:

| Level | Dataset games | Default seed/assets | Cumulative RL games | Final games/matchup | Periodic monitor | Resume |
|---|---:|---|---:|---:|---|---|
| `small` | 10,000 | Random, run-local | 100,000 | 10,000 | No | Not exposed |
| `default` | 50,000 | Random, run-local | 500,000 | 10,000 | No | Not exposed |
| `big` | 100,000 | 42, reusable | 2,000,000 | 1,000,000 | 100,000 RL games | Yes |
| `huge` | 100,000 | 42, reusable | 10,000,000 | 1,000,000 | 100,000 RL games | Yes |
| `forever` | 100,000 | 42, reusable | Unbounded | None | 100,000 RL games | Yes |

Direct self-play and the finite canonical profiles keep the four-epoch PPO
default. The `forever` PPO profile uses a maximum budget of 16 epochs per
on-policy buffer; its unchanged whole-buffer KL guard can finish each update
earlier. The first `forever` invocation locks the complete run configuration.
Later invocations reload it automatically, so the epoch budget and other
training arguments do not need to be repeated and conflicting explicit values
are rejected.

The reusable standard assets are built from 100,000 heuristic games with a
maximum supervised budget of 5,000 epochs (the convergence/plateau stopping
rules can finish earlier). They are
`dataset/supervised_dataset_standard_seed<seed>.jsonl` and
`models/domino_sl_standard_seed<seed>.npz`. Their sibling `meta.json` files
record structural versions, configuration, provenance, convergence fields,
and SHA-256. Presence alone is never enough for reuse. An incompatible asset
stops the run unless one of these explicit replacement controls is supplied:

```bash
python -m training.pipeline big --rebuild-dataset
python -m training.pipeline big --retrain-supervised
python -m training.pipeline big --rebuild-supervised-assets
```

Quick-run assets are retained for provenance under
`models/rl/domino_rl_<small-or-default>_seed<seed>_run<id>/supervised/`, but a
later invocation never selects them as inputs. Both quick and long-run
profiles retain the 5,000-epoch maximum; their plateau rules normally finish
earlier.

RL output lives at `models/rl/domino_rl_<level>_seed<seed>/`. `big`, `huge`,
and `forever` publish immutable exact resume generations plus the convenience
aliases `latest_weights.npz`, `optimizer_state.npz`, `rng_state.json`, and
`opponent_pool/pool_manifest.json`. `training_state.json` is the commit marker;
resume restores policy, optimizer, RNG state, opponent pool/order, adaptive
selection, algorithm-specific update history, and cumulative counters. Examples:

The marker advances at the normal numbered-checkpoint interval, not only at a
100,000-game diagnostic boundary. Superseded non-milestone latest payloads are
pruned only after the replacement marker is durable. Numbered policy
checkpoints and full milestone resume states each retain a rolling window of
the five newest generations; milestone policy weights remain available for
the complete diagnostic history and best-checkpoint pointer.

```bash
python -m training.pipeline big --resume
python -m training.pipeline huge \
  --resume-from models/rl/domino_rl_big_seed42

# First start: choose and lock the forever configuration.
python -m training.pipeline forever --seed 42 --gpi 2000 \
  --ppo-max-epochs 16 --run-name baseline

# Later starts: the active run and all locked arguments are loaded automatically.
python -m training.pipeline forever
```

PPO remains the default. To run `forever` with the policy-only historical
REINFORCE update and later resume that exact run:

```bash
python -m training.pipeline forever --no-ppo --run-name reinforce
python -m training.pipeline forever
```

`reinforce_v1` performs one update over the complete iteration buffer. It does
not construct a PPO buffer or calculate PPO ratios, clipping, KL control,
minibatches, or the post-update full-buffer evaluation. The algorithm is an
immutable resume field and is reloaded from the saved run configuration;
changing between `ppo_v1` and `reinforce_v1` is rejected before training. Both
algorithms support the optional value head, which remains off by default. A
canonical PPO actor-critic run can be started and resumed with:

```bash
python -m training.pipeline forever --value-head --run-name critic
python -m training.pipeline forever
```

The optional value accepted by `--resume` is a convenience alias for
`--resume-from`. In `forever`, diagnostic-worker autotuning is performed once,
persisted in `periodic_diagnostic_tuning.json`, and reused at subsequent
100,000-game monitors and after resume. Progress exposes a single cumulative
`avg_games_s` rate across the persisted RL training lineage.

`forever` has no percentage or target. SIGINT/SIGTERM stops admission of a new
iteration, lets an in-flight iteration finish, atomically publishes state, and
exits without an automatic all-pairs evaluation. GPI is never autotuned. The
canonical pipeline and direct self-play expose `--gpi` with choices
`100, 200, 400, 600, 800, 1000, 2000`; the default is 2,000. Worker autotuning
is unchanged. A boundary iteration is shortened so a periodic or final target
is never exceeded.

On the first `forever` start, `run_config.json` stores every locked argument,
the full canonical configuration, its SHA-256, ruleset version, optional run
name, supervised origin, and start-machine metadata. The active-run pointer
lets a later bare `python -m training.pipeline forever` find that run. Supply a
new `--run-name` (or use `--restart-rl`) when intentionally starting a distinct
configuration. A checkpoint is accepted only when its configuration hash
matches the run configuration.

| File | Purpose |
|---|---|
| `dataset_generator.py` | Coordinates retained worker tuning, bounded SQLite aggregation, and atomic ordered JSONL output. |
| `dataset_parallel.py` | Plays deterministic dataset games in a bounded CPU-only worker pool with dynamic scheduling and memory fallback. |
| `training_loop.py` | Selects safe host/GPU storage, validates the fixed supervised batch, orchestrates plateau scheduling, and saves `models/domino_sl_weights.npz` plus its loss graph. |
| `supervised_runtime.py` | Implements supervised batch/workspace safety, GPU residency probes/windows, and supervised memory telemetry. |
| `self_play.py` | Orchestrates the exact-budget on-policy training lifecycle and delegates its specialized phases. |
| `rl_config.py` / `rl_cli.py` | Validate side-effect-free RL options and own the standalone/canonical shared argument definitions. |
| `rl_rollout.py` | Finalizes rewards and trajectories and plays one CPU-only self-play or heuristic-opponent training game. |
| `rl_resume.py` | Loads compatible policies and atomically saves, validates, and restores exact numbered RL resume pairs. |
| `rl_reporting.py` | Owns iteration summaries, durable metrics JSONL writes, worker metadata aggregation, and cumulative RL runtime profiles. |
| `ppo.py` | Builds immutable decision buffers, selects minibatches, manages GPU/RAM storage, and performs KL-limited PPO epochs. |
| `adaptive_tuning.py` | Selects rollout workers with isolated seed streams, state restoration, safety checks, and `adaptive_tuning.json`. |
| `rl_parallel.py` | Shares frozen policy snapshots with deterministic CPU-only rollout workers and retains completed real games across memory fallback. |
| `canonical_assets.py` | Names, validates, hashes, and records reusable standard dataset/SL assets. |
| `canonical_run.py` | Publishes and validates complete atomic RL generations and lineage. |
| `pipeline.py` | Owns canonical levels, exact game boundaries, periodic diagnostics, resume, and safe shutdown. |

## Important Shape Change

The neural encoder now uses a 168-feature input vector and a 56-action output
space. The policy only chooses real tile-play decisions. Draw, pass, and
single-option tile plays are forced rule actions and bypass training.

The last seven input features are now opponent suit-presence probabilities:
`0.0` means known absence and `1.0` means known presence. This replaces the old
absence-confidence feature. Any encoded cache or model trained with the old
feature semantics should be treated as stale even though the array shapes still
match.

Old checkpoints trained with the previous 86-input/58-output encoder are not
compatible. After copying these files into the repo, run the pipeline again:

```bash
python -m training.dataset_generator
python -m training.training_loop
python -m training.self_play
```

## Supervised Dataset

Run:

```bash
python -m training.dataset_generator
python -m training.dataset_generator --workers auto --seed 123
python -m training.dataset_generator --workers 4 --games 5000
```

The generator records `(state, target_action)` pairs from games played by
`StrategicAgent` against itself. Engine states are already compact and do not
include rendering metadata. The command prints a startup RAM/GPU memory
snapshot, shows a progress bar, and reports total elapsed time. The standalone
command defaults to 30,000 games; the canonical pipeline requests 100,000.

Automatic mode benchmarks 1, 2, 4, 6, ... CPU-only workers, capped at 20.
Every attempt generates and retains 1% of the requested games. Testing stops on
a memory/error guard or below 10% marginal gain, then the remaining absolute
game ids run with the selected count. A central `utils.myrandom.SeedPlan`
derives one NumPy `PCG64` stream from the root seed and absolute game id. The
worker receives the plan and reconstructs that game's generator on demand; it
does not seed process-global Python, NumPy, or CuPy state. Fixed-worker and
automatic runs therefore produce byte-identical JSONL for the same `--seed`,
regardless of scheduling or a runtime fallback. Use `--help` for benchmark
fractions, RAM reserve, per-worker RSS, and estimated-worker-memory controls.

Each successful generation also writes
`<dataset-stem>.random_manifest.json` beside the JSONL. It records the root
seed, NumPy bit generator, derivation scheme, and registered namespaces. There
is no large per-game seed file: every game stream is reproducibly calculated
from `(root seed, DATASET_GAME, absolute game id)`.

Workers serialize one compact payload per game. Only the parent writes those
payloads to a disposable SQLite database, keeping RAM bounded while results
arrive out of order. Final rows are emitted in game-id order, and the existing
JSONL is replaced atomically only after every requested game succeeds.

Automatic dataset and RL game loops use the engine's trusted headless step
path. They reuse the unchanged legal-action collection already shown to the
agent and skip the post-action state snapshot that the loop would discard.
The pre-action state used for supervised examples, policy encoding, opponent
inference, trajectories, and event rewards is unchanged. Public/default engine
calls still generate their own legal actions and return a full state.

`StrategicAgent` now uses the exact two-player opponent model from
`middleware/opponent_model.py`. Dataset generation is therefore slower than the
old heuristic-only version, but each saved state includes the computed
`opponent_suit_probabilities` so supervised training can reuse them without
replaying the exact belief model for every row. The model keeps temporal draw
cohorts in `slots_exact`, converts once to integer `mu(H)` weights when the raw
hand bound reaches 500, and never falls back to particles.

A row is written only when the player had at least two legal tile-play choices.
The following turns are skipped:

- forced draw;
- forced pass;
- forced opening double;
- any state with only one legal tile play.

## Supervised Training

Run:

```bash
python -m training.training_loop
```

The loop:

- reads `dataset/supervised_dataset.jsonl`;
- filters out forced draw/pass examples;
- filters out single-option tile-play examples;
- scans the JSONL twice and encodes `float32` arrays without retaining decoded records;
- checks cgroup-aware host RAM before every material allocation;
- encodes states and tile-play actions with `DominoEncoder`;
- uses `dataset/supervised_dataset_encoded.npz` when the encoded dataset fits safely in RAM;
- otherwise atomically builds disk-backed `supervised_dataset_X.npy`,
  `supervised_dataset_Y.npy`, and `supervised_dataset_metadata.json` files and
  opens them read-only with `mmap`;
- splits data into training and validation sets;
- selects CPU/GPU independently with `--sl-device {auto,cpu,gpu}` (`--device`
  is a standalone alias);
- uses a fixed mini-batch of 8,192 examples by default;
- keeps the complete dataset in GPU memory when safe, or rotates one reusable
  GPU window through a global per-epoch permutation when it is not;
- keeps the best validation checkpoint in memory;
- stops automatically after conservative repeated blocks confirm that
  training loss has saturated;
- keeps only the 10 most recent archival checkpoints;
- saves `models/domino_sl_weights.npz`;
- writes `models/domino_sl_loss.png`, with one training-loss value per epoch
  and the validation-loss values already computed at validation intervals.

The loss graph uses only metrics collected by the current supervised run; it
does not run extra games or include win-rate data. A custom weights path such
as `models/experiment.npz` produces the sibling
`models/experiment_loss.png`. The PNG is replaced atomically after it is
rendered, so a plotting failure does not destroy the previous graph. Its lower
limit sits slightly below the terminal training loss and its upper limit is
the maximum observed loss, making the learned change visible instead of
spending most of the plot on the unused interval down to zero.

The supervised mini-batch is fixed at 8,192 by default and receives a memory
preflight before the first update. `--sl-batch-size N` can select one of the
former power-of-two candidate sizes from 1,024 through 1,048,576. The selected
size is capped to the number of training examples for small datasets.

GPU mode first probes resident example counts from 2,048 through 1,048,576
without changing weights. It preserves 512 MiB by default for batches,
activations, gradients, CUDA workspace, and fragmentation. `auto` falls back
safely to CPU when that reserve cannot be kept; explicit `gpu` fails before a
training update. Override host and GPU reserves with
`--sl-memory-reserve-mb` and `--sl-gpu-memory-reserve-mb`. The detailed command
reports the selected device, residency mode/capacity, one-time full upload,
fixed batch, and memory high/low watermarks. `train_script/run_pipeline.py` uses
`quiet=True`, so it continues suppressing per-epoch, checkpoint, scheduler, and
memory-detail chatter.

All supervised inputs, targets, weights, activations, gradients, and new
checkpoints are `float32`. Legacy `float64` checkpoints remain loadable and are
cast on input. Archival files in `models/supervised_checkpoints/` are pruned
after every save, so repeated or large runs retain at most 10 of them.
Archival files in `models/supervised_checkpoints/` are pruned after every save,
so repeated or large training runs retain at most 10 of them.

CuPy import alone is not treated as proof of a working GPU. At startup,
`agents/nn.py` also asks the CUDA runtime for a visible device; a missing driver,
hidden device, or unusable runtime produces a documented NumPy/CPU fallback
reason. The root README's **Linux GPU setup and verification** section contains
the driver checks, CUDA 12.x/13.x installation commands, a real calculation
test, and troubleshooting steps. `train_script/run_pipeline.py` prints the selected
supervised and RL-parent backends plus free/total RAM and VRAM before dataset
generation starts.

The encoded cache is rebuilt automatically when the source JSONL file changes,
the encoder input/output dimensions change, or the feature-version tag changes.

RL self-play performs a host-memory preflight for the shared snapshot bank and
expected batch, then checks the actual workspace before each `hstack`. With
`--device auto`, less than 256 MiB of effective free VRAM causes an announced
CPU fallback; explicit `--device gpu` fails early instead. Diagnostics later
run in separate CPU-only processes and never consume training VRAM.

The Python pipeline exposes independent dataset, RL rollout, and diagnostic
worker controls. Dataset generation tunes once for its full workload. RL tunes
across complete early iterations, and diagnostics tune each matchup separately.
All three retain benchmark work and enforce the same hard limit of 20 workers.

### Supervised scheduler and controls

The normal command starts at learning rate `0.005` and treats the requested
epoch count as a maximum. Automatic training-loss stopping is enabled by
default. It compares medians of non-overlapping 25-epoch blocks. A block counts
as saturated when its relative improvement over the previous block is below
`0.001` (0.1%). The run stops only after four consecutive saturated blocks and
never before epoch 100. A genuine improvement resets the counter.

Validation remains every 10 epochs, and validation-based LR decay is also on
by default. The first validation result establishes the global best; after
five consecutive checks without strict improvement, the LR is multiplied by
`0.5` and only the LR-specific failure counter resets. Another five failures
are required for another reduction. Optional validation early stopping has its
own counter; its patience should normally exceed LR patience so a reduced rate
has time to help. Whichever enabled stopping rule triggers first ends the run,
and the summary records `training loss plateau`, `validation loss plateau`, or
`epoch limit`.

Enable any control independently by adding its flag:

```bash
python -m training.training_loop --weight-decay
python -m training.training_loop --dropout
python -m training.training_loop --early-stopping
python -m training.training_loop --lr-decay 0.7 --lr-decay-patience 8
python -m training.training_loop --no-lr-decay
python -m training.training_loop --sl-no-training-plateau-stop
python -m training.training_loop --sl-device cpu --sl-seed 123
```

The supervised controls use these defaults:

| Flag | Behavior | Default |
|---|---|---:|
| `--weight-decay [COEFFICIENT]` | Adds L2 decay to the weight matrices, but not biases | off (`0.0001`) |
| `--dropout [RATE]` | Inverted dropout on every hidden layer | off (`0.1`) |
| `--early-stopping [PATIENCE]` | Stops after this many validation checks without improvement | `5` |
| `--lr-decay [FACTOR]` | Multiplies LR after the configured consecutive failed checks | `0.5` (on) |
| `--lr-decay-patience N` | Consecutive failed validation checks before each reduction | `5` |
| `--no-lr-decay` | Disables plateau scheduling for controlled comparisons | off |
| `--sl-no-training-plateau-stop` | Disables automatic training-loss saturation stopping | off |
| `--sl-training-plateau-window N` | Epochs in each non-overlapping median-loss block | `25` |
| `--sl-training-plateau-patience N` | Consecutive saturated blocks required to stop | `4` |
| `--sl-training-plateau-min-epochs N` | Minimum total epochs before this stop is allowed | `100` |
| `--sl-training-plateau-min-relative-improvement F` | Block improvement below this fraction counts as saturated | `0.001` |
| `--sl-device` / standalone `--device` | `auto`, forced `cpu`, or required `gpu` | `auto` |
| `--sl-batch-size N` | Fixed safe batch; power of two from 1,024 through 1,048,576 | `8,192` |
| `--hidden-layers N` | Number of hidden policy layers, 1 to 8 | `2` |
| `--hidden1-size N` ... `--hidden8-size N` | Width of one hidden layer | `256`, then `128` |
| `--sl-memory-reserve-mb N` | Free host RAM retained | `512` |
| `--sl-gpu-memory-reserve-mb N` | Effective free VRAM retained | `512` |
| `--sl-seed N` | Reproducible initialization and epoch permutations | unset |

Validation is checked every 10 epochs. The options can be combined and can
receive explicit values:

```bash
python -m training.training_loop \
  --weight-decay 0.00005 \
  --early-stopping 12 \
  --lr-decay 0.7 --lr-decay-patience 5 \
  --sl-device gpu
```

Reported training and validation losses remain cross-entropy values, allowing
loss curves to be compared with runs that do not enable weight decay.

The canonical pipeline accepts the same supervised controls:

```bash
python -m training.pipeline small \
  --weight-decay --early-stopping 12 --sl-device auto
```

### Shared regularization

Both regularizers are off by default; omitting their flags reproduces the
historical update rules exactly. Each flag takes an optional coefficient and
falls back to its default when passed bare, so `--dropout` and `--dropout 0.1`
are equivalent.

| Flag | Coefficient | Applies to | Behavior |
|---|---:|---|---|
| *(omitted)* | `0.0` | — | No regularization |
| `--weight-decay [COEFFICIENT]` | `0.0001` | supervised **and** RL | L2 decay on the weight matrices; biases are never decayed |
| `--dropout [RATE]` | `0.1` | supervised **and** RL | Inverted dropout on every hidden layer |

Each regularizer is one flag for both stages: the supervised policy and the RL
policy are the same architecture, and the RL run starts from the supervised
checkpoint, so a single coefficient keeps the two stages comparable. The two
flags are independent — requesting one never enables the other.

The two stages differ only in how the decay is applied. Supervised training
folds it into the gradient, matching the historical `--weight-decay` behavior.
The RL update applies it as a decoupled shrink after gradient clipping, so a
clipped step still decays and the logged gradient norms keep describing the
policy and value gradient alone.

Dropout is applied to training forward passes only. Validation, periodic and
final diagnostics, opponent-pool snapshots, rollout play, and the whole-buffer
PPO metrics all evaluate the complete network. The masks are drawn from the
host NumPy generator, whose state is part of the exact RL resume pair, so a
dropout run stays resumable on CPU and GPU alike.

One consequence is worth knowing before enabling dropout with PPO. The rollout
log-probabilities are recorded without dropout while the update evaluates a
thinned network, so the first-epoch importance ratios are no longer exactly
one; expect higher `approx_kl` and clip fractions, and consider raising
`--ppo-stop-kl` if updates stop early.

Runs and canonical assets created before these controls existed record no
regularization field. That absence is read as the disabled value, so existing
supervised checkpoints stay reusable and existing RL runs stay resumable
without a rebuild.

Because both flags change the supervised weights, they are part of the
seed-addressed asset identity. On `big`, `huge`, and `forever`, changing either
against existing reusable assets is refused until the rebuild is explicit:

```bash
python -m training.pipeline big --dropout 0.1 --retrain-supervised
```

## Self-Play RL

Run:

```bash
python -m training.self_play
python -m training.self_play --compact
python -m training.self_play --rl-workers auto --seed 123
python -m training.self_play --rl-workers 4 --device cpu
python -m training.self_play --gpi 1000
python -m training.self_play --fresh-from-sl
python -m training.self_play --dropout 0.1 --weight-decay
```

Default behavior:

- if a compatible `models/domino_rl_weights.npz` exists, resume from it;
- otherwise warm-start from a compatible `models/domino_sl_weights.npz`;
- train against a pool of frozen snapshots of the current policy;
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

### Parallel rollout generation

All games in an iteration use one immutable learner policy, so rollout work is
independent until batch aggregation. `training/rl_parallel.py` publishes the
current policy and at most `max_pool_size` opponent snapshots in a fixed-size
shared-memory ring. Workers attach NumPy views to that bank, never see the GPU,
and return finalized trajectories through a bounded dynamic queue. The parent
sorts results by game id and remains solely responsible for the gradient,
checkpoint writes, logging, and GPU allocations.

Opponent snapshots are published once per iteration, after the batch has been
played and the policy updated. Every iteration shares one immutable learner
policy, so the snapshot cadence is exactly the iteration size: `--gpi` sets it,
and the pool therefore spans `max_pool_size * gpi` training games. At the
defaults that is 50 snapshots covering 100,000 games.

GPI is never autotuned. Canonical pipelines and direct self-play accept
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
| `--rl-workers` | CPU-only rollout workers or `auto` | `auto` |
| `--retune-workers` | Explicitly rerun saved worker tuning on resume | off |
| `--rl-memory-reserve-mb` | Host RAM that must remain free | `512` |
| `--rl-estimated-worker-mb` | Conservative worker-memory estimate for preflight | `256` |
| `--rl-max-worker-rss-mb` | Runtime RSS ceiling for one worker | `1024` |

### Numbered checkpoints and exact resume

Direct-module calls keep the existing single-file checkpoint behavior unless
`--numbered-checkpoints` is requested. The canonical pipeline always uses
interruption-safe numbered pairs. Each save adds the absolute completed iteration
to the name, such as `model_iter000050.npz`, and atomically publishes a paired
`model_iter000050.resume.npz`. The state file contains a SHA-256 checksum,
every computation-affecting RL/PPO setting, completed real games, optimizer
state, RNGs, supervised-checkpoint hash, fixed GPI, tuned workers, rolling logs,
and the exact opponent-policy pool. The newest pool state replaces the previous
one to bound disk use; numbered policy files remain available.

To continue manually, pass the matching pair and its completed iteration while
keeping the original training configuration and total target:

```bash
python -m training.self_play --iterations 2000 --numbered-checkpoints \
  --rl-weights-path models/example.npz --start-iteration 500 \
  --resume-weights-path models/example_iter000500.npz \
  --resume-state-file models/example_iter000500.resume.npz --seed 42
```

Resume validates the checksum and configuration before loading anything,
restores optimizer/RNG/pool state, continues at the next absolute game id, and
requires the same fixed GPI while reusing saved workers without rerunning their
autotune.
This is a true continuation. Loading an ordinary `.npz` through the legacy
path restores only weights and cannot reconstruct the former in-memory pool.

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
summary. The parameter-sweep shell enables this presentation automatically.

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

With `--no-ppo`, the same fresh on-policy decisions go directly to the legacy
full-buffer policy-gradient update. Exactly one optimizer step is attempted per
non-empty iteration, and none of `training/ppo.py`'s buffer, ratio, clipped
surrogate, KL, minibatch, or whole-buffer evaluation paths run. Checkpoint,
optimizer, RNG, adaptive-tuning, opponent-pool, and canonical resume behavior
remain unchanged apart from recording `reinforce_v1` instead of `ppo_v1`.

The canonical contiguous buffer stays in RAM. If it fits within 70% of reported
free VRAM and a dry first-minibatch workspace probe succeeds, a complete GPU
copy is retained across epochs. Otherwise minibatches stream from RAM. No
fallback restarts a partially applied epoch.

Requested minibatches are fixed by the implementation as
`clamp(ceil(actual_games / 125), 4, 16)`, further capped to keep roughly 128
decisions per minibatch. The 4/16 bounds, 1% worker benchmark fraction, 10%
worker acceptance gain, and 70% GPU-buffer safety fraction are centralized in
`training/rl_constants.py` and are not experiment or CLI parameters. Each epoch visits every
decision exactly once with a deterministic new permutation. PPO uses clip
epsilon `0.2`, reports target KL `0.01` as an informational reference, and
does not start another epoch after the
completed epoch's whole-buffer approximate KL exceeds `0.015`. Direct
self-play and finite canonical profiles run at most four epochs by default;
the canonical `forever` profile runs at most 16. An explicit
`--ppo-max-epochs` overrides the profile default within the supported 1–16
range.

Enable the optional PPO actor-critic with:

```bash
python -m training.pipeline forever --value-head --run-name critic
# Or directly:
python -m training.self_play --value-head
```

This adds a linear `V(s)` head over the last hidden layer. The current
finalized policy reward is the value target, and the masked policy update uses
`reward - V(s)` as its advantage. PPO saves the collection-time values, uses a
clipped value loss, and evaluates value loss, clipping, prediction moments, and
explained variance after each epoch. The value-loss coefficient defaults to
`0.5` (`--value-coef`). Checkpoints contain `Wv` and `bv`. Combining
`--no-ppo --value-head` keeps the historical one-full-buffer actor-critic path.

Policy-only loading ignores `Wv`/`bv`, while value-head loading initializes
them to zero when they are absent. This permits mode changes without changing
the policy architecture, but clean comparisons should still start from the
same supervised checkpoint and use separately archived RL outputs.

### Optional RL controls

The default command uses PPO, the original reward constants, no terminal-reward
discount, gradient clipping at norm `5.0`, and whole-buffer advantage
normalization. Rollouts remain parallel while all updates stay in the parent:

| Flag | Meaning | Default |
|---|---|---:|
| `--fresh-from-sl` / `--continue-existing-rl` | Force initialization from SL or allow a compatible existing RL checkpoint | continue existing RL (standalone); canonical continuation uses `--resume` |
| `--gamma` | Terminal-reward discount per remaining real decision (`1.0` = no discount) | `1.0` |
| `--reward-schema` | Named preset for the terminal/event reward constants: `default` (the table below), `sparse` (win/loss only, no draw/pass shaping or pip penalty), or `shaped` (doubles the draw/pass shaping rewards) | `default` |
| `--clip-grad-norm` | Gradient-norm clipping threshold for the policy-gradient update | `5.0` |
| `--ppo` / `--no-ppo` | Masked PPO or historical one-update REINFORCE regression | PPO |
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
python -m training.self_play --gamma 0.97 --reward-schema shaped --seed 42
```

A point-in-time value loss or win rate is dominated by batch noise; the
iteration log always reports `reward mean/std/min/max` and a trailing moving
average of value loss and win rate next to the raw values, so a plateau can be
judged from the average rather than a single noisy line. In detailed
`training.self_play` output, a value-head run also prints the count and
mean/std/min/max of the pre-update `V(s)` predictions for every displayed
iteration. These are the same predictions used in `reward - V(s)`; the report
reuses that forward pass rather than evaluating the buffer again.

### Hidden-layer depth and width

The policy architecture defaults to `168 -> 256 -> 128 -> 56`. Both the number
of hidden layers and each width come from `agents/network_architecture.py` and
are selected once for supervised training and the canonical pipeline:

| Flag | Meaning | Default |
|---|---|---:|
| `--hidden-layers N` | Hidden layers, from 1 to 8 | `2` |
| `--hidden1-size N` | Width of hidden layer 1 | `256` |
| `--hidden2-size N` | Width of hidden layer 2 | `128` |
| `--hidden3-size N` ... `--hidden8-size N` | Width of hidden layer 3 through 8 | `128` |

An omitted width falls back to the default for that position, which keeps the
historical 256 and 128 for the first two layers and uses 128 for every deeper
layer. Sizing a layer the requested depth does not have is rejected instead of
silently ignored, so `--hidden-layers 2 --hidden3-size 64` is an error.

```bash
# Unchanged default: two layers, 168 -> 256 -> 128 -> 56.
python -m training.pipeline forever --run-name baseline

# Four layers, sized 512, 256, 128 (defaulted), and 64.
python -m training.pipeline forever --hidden-layers 4 \
  --hidden1-size 512 --hidden2-size 256 --hidden4-size 64 \
  --retrain-supervised --run-name deep
```

The RL stage has no separate architecture flags: it adopts whatever depth and
widths the supervised checkpoint stores. Weight arrays stay named `W1..W{L}`
and `b1..b{L}` for `L` layers including the output layer, so a two-layer
checkpoint keeps exactly its historical `W1, b1, W2, b2, W3, b3` keys and every
pre-existing artifact loads unchanged.

Architecture is part of supervised compatibility metadata and the immutable
RL resume identity. A run cannot resume with different dimensions, and changing
depth or width against reusable assets requires `--retrain-supervised`.

Only the command line stops at eight layers, because one `--hidden<n>-size`
option has to exist per layer. Nothing in the networks, checkpoints, resume
state, or metadata is limited to that depth, so a programmatic experiment can
build any `n >= 1`:

```python
from agents.rl_nn import PolicyNetwork

network = PolicyNetwork(hidden_sizes=(512, 384, 256, 192, 128, 96, 64, 48, 32))
```

The learning-curve footer in `rl_vs_random_progress.png` reports the depth and
every width of the run it plots, for example
`hidden 4 layers 512x256x128x64`. That row shrinks its font when a deep
architecture would otherwise collide with the checkpoint block on its left.

### Device selection (`--device`)

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

`training.self_play` also accepts `--iterations`, `--total-training-games`,
`--gpi`, `--training-opponent`, `--learning-rate`, `--entropy-coef`, `--log-interval`,
`--checkpoint-interval`, `--max-pool-size`,
`--sl-weights-path`, and `--rl-weights-path`; see
`training/rl_cli.py:add_optional_rl_arguments` for the authoritative
definitions, or run `python -m training.self_play --help`.

The former GPI-autotune options (`--games-per-iteration`, `--adaptive-gpi`,
`--no-adaptive-gpi`, `--gpi-candidates`, `--gpi-benchmark-games-target`,
`--retune-gpi`, and `--retune-all`) were removed. The former iteration-based
`--pool-interval` and auxiliary `--evaluation-games` options were also removed.
Existing numbered resume states that contain `pool_interval` are rejected rather
than silently reinterpreted as a game count; start a new run instead.

`--pool-refresh-games` was removed as well, with no replacement flag: it set a
cumulative-game threshold that was tested once per iteration, so every `--gpi`
at or above it produced exactly one snapshot per iteration and the option had no
effect. The cadence is now that behavior, stated directly. Unlike
`pool_interval`, a stored `pool_refresh_games` never rejects a resume: existing
runs and numbered resume states keep the field, it is ignored when comparing
configurations, and they continue with the current cadence. Snapshot timing is
unchanged for every `--gpi` of 400 or more, including the default 2,000;
`--gpi 100` and `--gpi 200` now snapshot every 100 or 200 games instead of
every 400.

`TRAINING_OPPONENT` at the top of `self_play.py` controls the training opponent:

| Value | Meaning |
|---|---|
| `"self_play"` | Train against a rotating pool of frozen policy snapshots. |
| `"heuristic"` | Train directly against `StrategicAgent`, useful for controlled comparisons. |

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
