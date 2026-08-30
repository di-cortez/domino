# Training

This folder contains the full training pipeline:

1. generate supervised examples from real heuristic decisions;
2. train a supervised neural policy;
3. refine that policy through self-play reinforcement learning.

## Game-result invariant

The current two-player ruleset has exactly two terminal outcomes: player 0
wins or player 1 wins. A blocked game is resolved by remaining pip total, then
hand size, then the most recent valid tile play, so a game-result draw is
impossible. Training, diagnostics, difficulty estimates, and persisted RL
state must therefore contain only wins and losses; a missing or non-binary
winner is an engine-contract error.

This is separate from the `DRAW` action, which means taking a tile from the
stock. Tile draws, passes, their event counters, and optional RL reward shaping
remain valid gameplay concepts and must not be confused with a drawn game.

## Ruleset contract

All stages accept `--ruleset` with exactly `double-six`, `double-five`,
`double-four`, or `double-three`; double-six is the default. The compact
policy dimensions are:

| Ruleset | Tiles | Hand | Stock | Input | Hidden defaults | Output |
|---|---:|---:|---:|---:|---:|---:|
| `double-six` | 28 | 7 | 14 | 168 | 256 x 128 | 56 |
| `double-five` | 21 | 6 | 9 | 130 | 192 x 96 | 42 |
| `double-four` | 15 | 5 | 5 | 97 | 128 x 64 | 30 |
| `double-three` | 10 | 4 | 2 | 69 | 96 x 48 | 20 |

The `Input` column is the default layout. `--no-opponent-suit-features` removes
the trailing exact-model block (`-S` features) and `--opponent-bucket-features`
appends a 7-wide one-hot of the opponent bucket the agent is facing, so
double-six ranges over 161, 168, and 175 inputs. Both flags change the input
size, so checkpoints and supervised assets are not interchangeable across them
and each non-default regime claims its own asset suffix.

Dataset/cache/model metadata records the canonical name. Compact canonical
asset and run names include it; historical double-six names are unchanged.
Resume locks the ruleset, and cross-ruleset checkpoint loading is rejected.
There is no weight padding, remapping, or transfer-learning path.

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
| `small` | 10,000 | Random, run-local | 100,000 | 10,000 | 100,000 RL games, 10,000 games/point | Not exposed |
| `default` | 50,000 | Random, run-local | 500,000 | 10,000 | 100,000 RL games, 10,000 games/point | Not exposed |
| `big` | 100,000 | 42, reusable | 2,000,000 | 1,000,000 | 100,000 RL games | Yes |
| `huge` | 100,000 | 42, reusable | 10,000,000 | 1,000,000 | 100,000 RL games | Yes |
| `forever` | 100,000 | 42, reusable | Unbounded | None | 100,000 RL games | Yes |

Direct self-play and the finite canonical profiles keep the four-epoch PPO
default. The `forever` PPO profile uses a maximum budget of 16 epochs per
on-policy buffer; its unchanged whole-buffer KL guard can finish each update
earlier. The first `forever` invocation locks the complete run configuration.
Later invocations reload it automatically, so the epoch budget and other
training arguments do not need to be repeated. Conflicting explicit values are
reported in one warning, ignored, and replaced from `run_config.json`.

The reusable standard assets are built from 100,000 heuristic games with a
maximum supervised budget of 5,000 epochs (the convergence/plateau stopping
rules can finish earlier). They are
`dataset/supervised_dataset_standard_seed<seed>.jsonl` and
`models/domino_sl_standard_seed<seed>.npz`. Their sibling `meta.json` files
record separately the artifact identity, source-dataset creation parameters,
model architecture, SL hyperparameters and fixed TP policy, training result,
resolved execution choices, structural contracts, repository provenance, and
SHA-256. They deliberately omit creation timestamps and machine-local paths,
so equal runs can publish equal metadata in different checkouts.
Presence alone is never enough for reuse. An incompatible asset stops the run
unless one of these explicit replacement controls is supplied:

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

RL output lives at `models/rl/domino_rl_<level>_seed<seed>/`. Its compact
analysis bundle (`run_config.json`, periodic JSONL, progress CSV and progress
PNG) lives together under `run_compact_diagnostics/`. `big`, `huge`,
and `forever` publish immutable exact resume generations plus the convenience
aliases `latest_weights.npz`, `optimizer_state.npz`, `rng_state.json`, and
`opponent_pool/pool_manifest.json`. `training_state.json` is the commit marker;
resume restores policy, optimizer, RNG state, opponent identities/bucket order,
difficulty evidence, adaptive selection, algorithm-specific update history,
and cumulative counters. The independent `checkpoint_archive/` keeps a bounded,
progressively thinned policy history. The optional `medium_term` opponent
bucket references 200 of its ten-iteration milestones without duplicating
weights and pins those active records against thinning. Examples:

The marker advances at the normal numbered-checkpoint interval, not only at a
100,000-game diagnostic boundary. Superseded non-milestone latest payloads are
pruned only after the replacement marker is durable. Numbered policy
checkpoints and full milestone resume states each retain a rolling window of
the five newest generations; milestone policy weights remain available for
the complete diagnostic history and best-checkpoint pointer.

```bash
python -m training.pipeline big --resume
python -m training.pipeline huge \
  --resume models/rl/domino_rl_huge_seed42

# First start: choose and lock the forever configuration.
python -m training.pipeline forever --seed 42 --gpi 2000 \
  --ppo-max-epochs 16 --run-name baseline

# Later starts: the active run and all locked arguments are loaded automatically.
python -m training.pipeline forever
```

PPO remains the default. To run `forever` with the policy-only
REINFORCE update and later resume that exact run:

```bash
python -m training.pipeline forever --ppo-max-epochs 1 --run-name reinforce
python -m training.pipeline forever
```

`reinforce_v1` performs one update over the complete iteration buffer. It does
not construct a PPO buffer or calculate PPO ratios, clipping, KL control,
minibatches, or the post-update full-buffer evaluation. The algorithm is an
immutable resume field and is reloaded from the saved run configuration;
an attempted algorithm override during resume is warned about and ignored.
Both algorithms support the optional value head, which remains off by default. A
canonical PPO actor-critic run can be started and resumed with:

```bash
python -m training.pipeline forever --value-head --run-name critic
python -m training.pipeline forever
```

### Opponent-bucket input features

`--opponent-bucket-features` appends a one-hot to the policy input naming which
opponent bucket the agent is playing against, so the policy can condition on the
kind of opponent it faces. The block is 7 wide — one slot per bucket in
`training/rl/pool.py`'s registry, not per bucket `--opponent-buckets` selects —
so the input size never depends on the selection and double-six grows from 168
to 175 features.

| Position | Bucket | double-six encoding |
| ---: | --- | --- |
| 0 | `heuristic` | `1000000` |
| 1 | `random` | `0100000` |
| 2 | `recent` | `0010000` |
| 3 | `medium_term` | `0001000` |
| 4 | `historical_uniform` | `0000100` |
| 5 | `champion_vs_heuristic` | `0000010` |
| 6 | `champion_vs_learner` | `0000001` |

```bash
python -m training.pipeline forever --opponent-bucket-features \
  --opponent-buckets heuristic,random,recent --run-name knows_opponent
```

The one-hot names *that seat's* adversary, not the match, so the two players in
one game normally see different values. A learner drawn against a frozen
`medium_term` snapshot is given `medium_term`; the snapshot is given `recent`,
because what it faces is the current learner. The bucket comes from the match
assignment rather than from the opponent's kind, because several buckets share
one kind: every neural bucket dispatches as a policy snapshot, and the two
champion buckets deliberately overlap the chronological bands.

Outside the RL rollout the same rule fills the block. The supervised dataset is
`StrategicAgent` against `StrategicAgent`, so pretraining encodes `heuristic` —
a bucket RL reuses, rather than a pattern it would never show again. Champion
racing gives the candidate `heuristic` or `recent` by target and the learner
seat `champion_vs_learner`; the pairwise diagnostics map each agent's
counterpart onto the bucket it stands for. An all-zero block is the explicit
no-bucket state, reserved for an adversary with no bucket at all.

The UI is not one of those cases. A human belongs to no bucket, but since the
block is one-hot in every training vector, all-zero is a pattern the policy has
never seen rather than a neutral input, so `ui.ui_agents.UI_OPPONENT_BUCKET`
hands it the `heuristic` slot — the bucket pretraining is encoded with. That is
a stand-in for an input the policy can read, not a claim that a human plays like
`StrategicAgent`, and no bucket value makes the conditioning transfer to human
play.

The flag changes the input size, so its checkpoints and supervised assets are
not interchangeable with runs that omit it, and it claims its own asset suffix
(`_bucket`). Note that it is not distinguishable from the suit ablation by size
alone: `--no-opponent-suit-features --opponent-bucket-features` restores exactly
168 double-six inputs with seven different trailing features. A shape check
cannot catch that swap, so the flag is recorded in the durable resume
configuration as well.

### Policy-gradient baseline

`--baseline` selects the only term subtracted from a return before the policy
gradient. Advantage normalization only rescales, so the baseline alone decides
what is subtracted:

| Flag | Baseline `b` | Advantage |
| --- | --- | --- |
| `--baseline zero` | `0` | `R` |
| `--baseline 2` | `2` | `R - 2` |
| `--baseline batch-mean` | `mean(R)` over the iteration buffer | `R - mean(R)` |
| `--baseline lookup-table` | Fixed `E[R | learner tiles, opponent tiles]` | `R - lookup(state)` |
| `--baseline value-head` | `V(s)`, shared trunk, critic trains it | `R - V(s)` |
| `--baseline value-head-no-up` | `V(s)`, shared trunk, critic does not train it | `R - V(s)` |
| `--baseline value-head-own-nn` | `V(s)` from a separate network | `R - V(s)` |

```bash
python -m training.pipeline forever --baseline zero --run-name no_baseline
python -m training.pipeline forever --baseline 2 --run-name const2
python -m training.pipeline forever --baseline lookup-table \
  --run-name fixed_lookup
python -m training.pipeline forever --value-head --baseline batch-mean \
  --run-name critic_trained_not_used
```

Leaving the flag unset keeps the behavior that predates it, so the default is
numerically unchanged: the critic when `--value-head` is on, otherwise the batch
mean whenever advantage normalization is on and no baseline at all when it is
off.

#### Fixed hand-size lookup

`lookup-table` reads one source-controlled artifact for the active ruleset and
conditions only on `(learner_hand_size, opponent_hand_size)` before the sampled
action. It evaluates final outcome and terminal-pip histograms with `gamma_f`,
pass/draw histograms with `gamma_i`, and selects the two clocks through
`--reward-distance-mode`. The current terminal, pip, pass, and draw magnitudes
are then applied before terminal/local components are combined with
`reward_eta`. The artifacts therefore contain unit counts rather than one
experiment's reward hyperparameters.

Unsupported cells use the documented deterministic boundary/diagonal rules in
`training/rl/reward_lookup_tables/README.md`. The artifact SHA-256 is persisted
with exact resume state. The table is a state baseline only: it never replaces
the sampled trajectory return and does not alter reward generation.

#### The constant is a bare number

A constant baseline is spelled as the number itself: `--baseline 2` subtracts
2, `--baseline -0.5` subtracts -0.5. No baseline name parses as a number, so
this can never shadow one.

**The number is the constant's value, never a position in the table above.**
`--baseline 2` is the constant 2 and not `batch-mean`; `--baseline 3` is the
constant 3 and not `value-head`. This is the one real trap in the grammar.

`--baseline 0` is the constant zero. It produces the same gradient as
`--baseline zero` but stays a distinct request: the two record different
`locked_arguments` and therefore different `configuration_sha256`, which is what
lets a later audit tell which one a run actually asked for.

Anything after the number is an error rather than silently absorbed, so
`--baseline 2 forever` is refused instead of running at the wrong pipeline
level. Put positional arguments before the flag. The older
`--baseline constant 2` spelling still parses, because that is how the value is
persisted and every existing run has to stay resumable.

#### The three critics

`value-head`, `value-head-no-up` and `value-head-own-nn` subtract exactly the
same thing -- one `V(s)` per decision -- and differ only in how the critic is
attached to the policy. Together they separate the critic's worth *as a
baseline* from its effect *on the shared representation*:

| Kind | Shared trunk | Critic shapes the trunk |
| --- | --- | --- |
| `value-head` | yes | yes |
| `value-head-no-up` | yes | no |
| `value-head-own-nn` | no | no (there is no shared trunk) |

```bash
python -m training.pipeline forever --value-head --baseline value-head-no-up \
  --run-name critic_reads_only
python -m training.pipeline forever --value-head --baseline value-head-own-nn \
  --run-name critic_own_network
```

`value-head-no-up` keeps the linear head over the policy's last hidden
activation and keeps training it, but its loss stops at `Wv`/`bv`. The head
still learns exactly as fast: those gradients read the hidden activation as
data, not as a node to differentiate through, so cutting the trunk contribution
cannot change them. What changes is the hidden stack, which is then trained by
the policy gradient alone.

`value-head-own-nn` gives the critic its own network, mirroring the policy's
hidden widths with a single linear output and sharing no weights at all. It
roughly doubles the trained parameters and adds one forward and one backward
pass per minibatch, so a per-iteration time comparison against the other kinds
is not like-for-like. Its weights are stored in the checkpoint under a
`critic.` prefix; a frozen opponent in the pool never carries them, because an
opponent only acts and never evaluates.

Every value-head kind requires `--value-head`. That flag still means "build and
train a critic", and `--baseline` chooses which one, so
`--value-head --baseline batch-mean` keeps working as the way to pay the
critic's cost without using its output.

One reporting caveat: gradient norms are summed over every gradient in the
update. `value-head-no-up` drops the term the critic used to inject into the
trunk, and `value-head-own-nn` adds a whole second network's gradients, so
reported `grad norm` is not comparable across the three kinds.

#### `batch-mean` versus the previous behavior

`batch-mean` is not a new estimator — it is the name for the baseline the
project has always used without calling it one. As derived in
[`ppo_sem_critico.tex`](../references/explicacoes/ppo_with_out_critic/ppo_sem_critico.tex),
the old `normalize_advantages` subtracted the buffer mean and divided by the
buffer standard deviation in a single step, so the advantage reaching the
clipped objective was already `(R - mu_B) / (sigma_B + eps)`. The mean was a
baseline in every mathematical sense; it simply arrived as a side effect of a
variance-reduction step instead of as a choice.

What changed is that the two operations are factored apart:

```
before:  advantage = (R - mu_B) / (sigma_B + eps)     # one inseparable step
after:   advantage = (R - b)    / (sigma   + eps)     # b chosen, then scaled
```

`--baseline batch-mean` with normalization on is therefore bit-for-bit the
previous default. The denominator does not move for the three state-independent
kinds: `zero`, `constant` and `batch-mean` subtract the same value from every
decision, and a standard deviation is invariant under a constant shift, so
`sigma(R - b)` equals `sigma(R)` exactly. `lookup-table` and the critic kinds
can change the scale because their baselines differ per decision.

Two combinations that were previously unreachable now are:

- **Centered but unscaled.** Turning normalization off used to remove the
  centering with it, leaving the raw return. `--baseline batch-mean
  --no-normalize-advantages` now gives `R - mu_B` without the division.
- **A zero-variance iteration keeps its offset.** It used to collapse to all
  zeros because the mean was always removed. It now keeps whatever offset the
  selected baseline implies — still zero for `batch-mean`, but not for
  `constant`.

One existing configuration does change numerically. With `--value-head` the
advantage was `(R - V - mean(R - V)) / sigma(R - V)`; that extra centering was
an artifact of normalization doing double duty, and the advantage is now
`(R - V) / sigma(R - V)`.

The critic head and the baseline are independent. `--baseline value-head`
requires `--value-head`; every other choice may be combined with it, which keeps
training `V(s)` through the value loss without subtracting it — the combination
that separates the cost of training the critic from the effect of using it.

The positional pipeline level must come before `--baseline`, because
`constant` takes its value as a following token. The selected baseline is an
immutable resume field, reloaded from the saved run configuration like the
algorithm.

The optional value accepted by `--resume` is a convenience alias for
`--resume-from`. In `forever`, diagnostic-worker autotuning is performed once,
persisted in `periodic_diagnostic_tuning.json`, and reused at subsequent
100,000-game monitors and after resume. Progress exposes a single cumulative
`avg_games_s` rate across the persisted history of that run.

`forever` has no percentage or target. SIGINT/SIGTERM stops admission of a new
iteration, lets an in-flight iteration finish, atomically publishes state, and
exits without an automatic all-pairs evaluation. GPI is never autotuned. The
canonical pipeline and direct self-play expose `--gpi` with choices
`100, 200, 400, 600, 800, 1000, 2000`; the default is 2,000. Worker autotuning
is unchanged. A boundary iteration is shortened so a periodic or final target
is never exceeded.

On the first `forever` start, `run_compact_diagnostics/run_config.json` stores every locked argument,
the full canonical configuration, its SHA-256, ruleset version, optional run
name, supervised origin, and start-machine metadata. The active-run pointer
lets a later bare `python -m training.pipeline forever` find that run. Supply a
new `--run-name` (or use `--restart-rl`) when intentionally starting a distinct
configuration. A checkpoint is accepted only when its configuration hash
matches the run configuration. Resume never accepts training or asset
overrides: explicit values are listed in a warning and replaced by the saved
ones. A repository commit mismatch is also warned about but does not block the
continuation, and the run keeps its original commit as provenance.

## Layout

The package is organized by stage. Each subpackage owns its own README with the
detailed commands and behavior for that stage.

| Path | Owns | README |
|---|---|---|
| `utils/` | Cross-stage helpers: seed derivation, shared CLI controls, the encoded-feature contract. | [`utils/README.md`](utils/README.md) |
| `datagen/` | Deterministic supervised dataset generation. | [`datagen/README.md`](datagen/README.md) |
| `supervised/` | Supervised policy training and its memory/residency safety. | [`supervised/README.md`](supervised/README.md) |
| `rl/` | Self-play reinforcement learning, PPO, rollouts, and exact resume. | [`rl/README.md`](rl/README.md) |

Three modules stay at the package root because they own orchestration across
all three stages rather than belonging to any one of them:

| File | Purpose |
|---|---|
| `pipeline.py` | Owns canonical levels, exact game boundaries, periodic diagnostics, resume, and safe shutdown. |
| `canonical_assets.py` | Names, validates, hashes, and records reusable standard dataset/SL assets. |
| `canonical_run.py` | Publishes and validates complete atomic RL generations against their immutable run configuration. |

### Import layering

The subpackages form strict layers. Each one imports only downward, which keeps
the graph acyclic and stops an RL run from pulling in the supervised trainer:

- `training/utils/` imports nothing else from `training/`.
- `datagen/`, `supervised/`, and `rl/` never import each other. Each may import
  `training/utils/`.
- `canonical_assets.py` and `canonical_run.py` may import `training/utils/` and
  `rl/resume.py`; they do not import `supervised/`.
- Only `pipeline.py` imports across all three stages.

Preserve these rules when adding a module. A helper that two stages need
belongs in `training/utils/`, not in whichever stage happened to define it
first.

### Stage entry points

| Stage | Command |
|---|---|
| Dataset generation | `python -m training.datagen.generator` |
| Supervised training | `python -m training.supervised.cli` |
| Self-play RL | `python -m training.rl.cli` |
| Full canonical pipeline | `python -m training.pipeline <level>` |

## Compact encoder shape

For `T` tiles and `S` pip values the encoder uses `5T + 3S + 7` inputs and
`2T` policy outputs. The policy only chooses real tile-play decisions; draw,
pass, and single-option tile plays bypass training. Its final `S` features are
exact opponent pip-presence probabilities in `[0, 1]`. Legacy records without
an explicit name are interpreted as double-six only.

## Worker controls

The Python pipeline exposes independent dataset, RL rollout, and diagnostic
worker controls. Dataset generation tunes once for its full workload. RL tunes
across complete early iterations, and diagnostics tune each matchup separately.
All three retain benchmark work and enforce the same hard limit of 20 workers.
