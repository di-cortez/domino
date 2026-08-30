# Architecture

## System boundaries

The project has five primary layers with one-way ownership of core contracts:

```text
UI and experiment entry points
          |
          v
agents and training/diagnostics orchestration
          |
          v
middleware game rules and public state
          |
          +----> exact opponent inference
          |
          v
generated datasets, checkpoints, and reports
```

`middleware/` owns legality and state transitions. Agents may choose from legal
actions but never redefine the rules. Training and diagnostics create games
through the same engine and agents used by the UI. Generated artifacts are
outputs of those layers, not inputs to source control.

`middleware.rulesets` is the dependency-free, closed source of truth for
double-six, double-five, double-four, and double-three geometry. One immutable
ruleset name flows downward through the engine, exact opponent domain, encoder,
agents, workers, artifacts, diagnostics, and UI. Persistent states and files
record that identity; missing legacy identity means double-six only.

## Rules and orchestration

`middleware.domino_engine.DominoEngine` owns dealing, turns, board ends, stock,
draw/pass behavior, blocked-game resolution, winner selection, legal actions,
and serialized state. The stable action forms are:

- `(tile, side)` for a play on the left (`0`) or right (`1`) end;
- `("DRAW", None)` for a legal stock draw;
- `None` for a legal pass.

Games never end in a draw. Emptying a hand wins immediately. A blocked game is
resolved by fewest remaining pips, then fewest remaining tiles, then the most
recent valid tile play among any players still tied.

`middleware.middleware.Agent` is the minimal `choose_move(state,
legal_actions)` protocol. `GameManager` connects an engine to player agents and
records public game history used by supervised data generation.

The normal `DominoEngine.step(action)` path validates legal actions and returns
`(state, game_over, info)`. Controlled headless loops may reuse the unchanged
legal-action collection they just obtained and request no post-action state.
That fast path is internal: UI, network, or external payloads must use the
validating path. The fixed-seed equivalence benchmark is
`benchmarks/headless_step_benchmark.py`.

## Public-information opponent model

`middleware.opponent_model.HybridExactOpponentModel` reconstructs public
actions and maintains an exact two-player belief without reading the actual
opponent hand. It begins with temporal hand-slot domains so a later draw does
not inherit earlier negative evidence. At the first eligible non-terminal turn
boundary it converts once to integer-weighted exact hand masks `mu(H)` and
never falls back to particles.

The model returns one pip-presence probability per value in the active ruleset:
`0.0` means known absence and `1.0` known presence. Response probabilities use
the exact joint hand posterior, not independent suit marginals. The model stays
on CPU because its workload is irregular branching over bitmasks and
arbitrary-precision integer weights.

The middleware README is the source of truth for evidence ordering, trace
stages, cache invalidation, and the slot-to-hand transition.

## Agents and neural contracts

All concrete agents inherit `middleware.middleware.Agent`:

| Agent | Policy |
|---|---|
| `RandomAgent` | Uniform legal action. |
| `StrategicAgent` | Deterministic exact-belief heuristic and supervised teacher. |
| `NeuralAgent` | Supervised MLP checkpoint with legal-action masking. |
| `RLAgent` | Supervised-initialized policy refined by on-policy self-play. |

`DominoEncoder` is shared by supervised and RL paths. For `T` tiles and `S` pip
values it produces `5T + 3S + 7` public-information features and maps real tile
decisions to `2T` outputs. Draw, pass, and single-option tile plays are forced
by the engine and bypass neural inference and policy-gradient sampling.

Two trailing feature blocks are optional, and each is last in the layout while
it is present, so toggling either one moves no offset before it:

| Block | Flag | Width | double-six |
|---|---|---|---|
| Exact-model suit presence | `--no-opponent-suit-features` removes it | `S` | 168 to 161 |
| Opponent bucket one-hot | `--opponent-bucket-features` adds it | 7 | 168 to 175 |

The opponent-bucket block holds a one-hot of which logical opponent bucket the
seat being encoded is playing against, so the policy can condition on the kind
of opponent it faces. Its width is every bucket in `training/rl/pool.py`'s
`BUCKET_REGISTRY`, not the subset `--opponent-buckets` selects, so the input
size never depends on the bucket selection and a checkpoint stays loadable
under a different one. The vocabulary is duplicated in `agents/encoder.py`
because `agents` must not import `training`;
`tests/test_opponent_bucket_features.py` fails if the two orders drift apart.

The block names *that seat's* adversary rather than the match, so the two
players in one game normally receive different values: a learner facing a
frozen `medium_term` snapshot is given `medium_term`, and the snapshot is given
`recent`, because what it faces is the current learner. Outside the RL rollout
the value comes from the same rule. The supervised dataset is `StrategicAgent`
against `StrategicAgent`, so every example is encoded as `heuristic`, keeping
pretraining inside a distribution RL reproduces; champion racing gives the
candidate `heuristic` or `recent` by target, and the learner seat
`champion_vs_learner`; the pairwise diagnostics map each agent's counterpart
through `diagnostics.gameplay.opponent_bucket_for_agent`. An all-zero block is
the explicit no-bucket state, reserved for an adversary that belongs to no
bucket at all.

The UI does not use it. A human belongs to no bucket, but the block is one-hot
in every vector the policy trained on, so all-zero is a pattern it has never
seen rather than a neutral input. `ui.ui_agents.UI_OPPONENT_BUCKET` hands the
`heuristic` slot instead — the bucket the supervised dataset is encoded with,
and therefore the one pattern every checkpoint has seen from its first gradient
step. It is a stand-in, not a claim about how the human plays. The UI reads the
layout off the checkpoint's input width, since it has no run configuration:
only the widened vector is decided that way, and a default-width vector keeps
the historical layout.

The two flags are independent, and for double-six they are not distinguishable
by size: dropping the suit block while adding the bucket block restores exactly
168 inputs with seven different trailing features. A checkpoint shape check
cannot see that swap, which is why `use_opponent_bucket_features` is recorded in
the durable resume configuration and why each non-default regime claims its own
supervised-asset suffix (`_nosuit`, `_bucket`).

`SupervisedNeuralNetwork` is a float32 compact MLP. Defaults are
`168-256-128-56`, `130-192-96-42`, `97-128-64-30`, and `69-96-48-20` from
double-six through double-three. The hidden-layer count and every width have a single
source of truth in `agents/network_architecture.py` and are configurable from
the supervised and canonical pipeline CLIs with `--hidden-layers` and
`--hidden1-size` through `--hidden8-size`; the networks themselves accept any
depth, and only the command line is bounded. `PolicyNetwork` extends it with
masked policy-gradient updates and an optional training-only value head over
the last hidden activation. Policy checkpoints store `W1`, `b1`, ..., `W{L}`,
`b{L}` for `L` layers including the output layer, so the default two-layer
network still stores exactly `W1`, `b1`, `W2`, `b2`, `W3`, and `b3`;
critic-enabled checkpoints also store `Wv` and `bv`. Every loader reads the
architecture back out of the checkpoint. Current code still loads compatible
float64 arrays by casting them to float32.

## Training data flow

```text
StrategicAgent vs StrategicAgent games
          |
          v
quick run-local or standard_seed<seed> dataset + metadata/hash
          |
          v
encoded float32 cache -> supervised MLP -> run-local or standard SL + metadata
                              |                              |
                              v                              v
                         loss PNG                  adaptive on-policy RL
                                             frozen rollouts -> selected update
                                                    /                 \
                                             masked PPO       full-buffer REINFORCE
                                                               |
                                                               v
                                               level/seed RL run directory
                                                  |             |
                                                  v             v
                                      exact resume state   periodic monitor
```

Dataset generation retains only real policy decisions and writes deterministic
game-id order through a bounded SQLite aggregation stage. The encoded cache is
rebuilt when the dataset metadata or encoder contract changes.
Each absolute dataset game derives its own NumPy `PCG64` stream from the root
`SeedPlan`; worker scheduling, autotuning, and fallback therefore cannot change
the deal. The small seed-plan manifest is persisted beside the JSONL, while
per-game streams are calculated on demand instead of being stored in a seed
table.

Supervised training can keep encoded arrays in host RAM, use atomic disk-backed
memory maps, and place all or rotating windows of data on the GPU. It saves the
best validation checkpoint atomically and renders the training/validation loss
history already collected during that run. The epoch count is a maximum
budget: repeated low-improvement blocks of training loss can stop a saturated
run early. Supervised optimization uses a fixed, memory-checked batch of 8,192
examples by default. One run-level `SeedPlan` owns independent PCG64 streams for
weight initialization, coordinate-derived epoch permutations, and sequential
dropout. Random arrays are generated on the host and transferred when the
selected backend is CuPy, and a manifest beside the final weights records the
root seed and derivation contract.

RL uses fresh on-policy trajectories: all games in an iteration observe the
same frozen policy. The default update stores masked collection-time
log-probabilities, subtracts a baseline, rescales advantages once over the
complete decision buffer, and runs masked PPO in deterministic minibatches.
Direct and finite canonical runs default to at most four epochs; `forever`
defaults to 16, with a whole-buffer KL guard after every completed epoch.

`training/rl/baseline.py` owns the only term subtracted from a return, selected
by `--baseline`: `zero`, `constant VALUE`, `batch-mean`, `lookup-table`, or one
of the three value-head wirings. The lookup kind reads versioned unit-reward
histograms from `training/rl/reward_lookup_tables/`, conditions only on learner
and opponent hand sizes before the action, and applies the run's gamma, reward
magnitudes, distance clocks, and eta at evaluation time.
Advantage normalization only rescales; centering belongs to the baseline alone,
because a normalizer that also removed the batch mean would silently reimpose
`batch-mean` and make `zero` and `constant` unobservable. An unset flag keeps
the behavior that predates it — the critic when its head is on, otherwise the
batch mean whenever normalization is on and no baseline at all when it is off —
so the default is numerically unchanged. The critic head and the baseline are
independent: `value-head` requires `--value-head`, while every other choice may
be combined with it to keep training `V(s)` without subtracting it. The
selected baseline is recorded in `locked_arguments` rather than `rl_config`,
which is compared as an immutable run key, so runs created before the flag
existed stay resumable. With the value head PPO also optimizes a clipped critic
loss. The optional `reinforce_v1` update instead applies one
full-buffer policy-gradient step and skips PPO buffer construction, ratios,
clipping, KL control, minibatches, and post-update full-buffer evaluation.
There is no replay buffer or cross-iteration reuse in either mode. Decision
returns are not rescaled by the number of legal choices. A checksummed
`.resume.npz` preserves the selected algorithm, policy, optimizer, RNG,
adaptive selections, counters, and pool.

Opponent buckets are ownership boundaries, not just names. `training/rl/pool.py`
owns durable identity, bucket definitions, membership, and the physical-bank
mapping; `training/rl/checkpoint_archive.py` owns the disk-backed files,
hashes, thinning, and pin state; `training/rl/matchmaking.py` owns deterministic
integer allocation, including the persisted per-bucket anchor that rotates
which members receive the uniform component's remainder games. The three historical neural buckets cover disjoint chronological regions:
`recent` holds the latest 200 learner snapshots, `medium_term` the newest 200
archive milestones already behind that band, and `historical_uniform` up to 200
uniform representatives of everything older. The two champion buckets,
`champion_vs_heuristic` and `champion_vs_learner`, are selected by strength
rather than by age and may therefore overlap all three -- and each other -- on
purpose, giving one identity extra matchmaking weight without a second weight
copy. They share the racing mechanics and the candidate stream but not the
retention evidence: the fixed heuristic target admits a stored score that stays
comparable across events, while the moving learner target does not, so a full
learner bucket evicts by current decayed difficulty instead. `training/rl/champion_evaluation.py` owns only the racing mechanics that
choose those champions -- the fixed stage table, the shared seed panels, seat
balance, and deterministic ranking -- and never touches membership, which
remains the pool's; `iteration.py` runs the event between the archive refresh
and performance-tracker reconciliation, so its 100,000 evaluation games are
timed separately and stay out of GPI counters, metrics rows, and PPO buffers.
The two delayed bands are derived
from archive metadata at every cadence boundary rather than admitted at update
time, so `iteration.py` publishes the archived milestone first, reconciles both
memberships as one transaction, then republishes the archive pin union and only
afterwards reconciles performance trackers. `pool.py` never imports the archive
class: archive records and a weight-loader callback are passed in, keeping the
dependency pointing one way.

`training.pipeline` owns canonical orchestration. `small` and `default` are
ephemeral profiles: they choose a random seed by default, build 10,000- and
50,000-game datasets respectively, and place non-reused supervised assets in a
unique run namespace. `big`, `huge`, and `forever` default to seed 42 and reuse
the same compatible 100,000-game standard dataset and supervised checkpoint.
The RL budget is cumulative games, with shortened final and milestone
iterations. Canonical `big`, `huge`, and `forever` state uses immutable payload
generations and an atomic `training_state.json` marker so resume restores weights,
optimizer, RNGs, pool order/provenance, adaptive choices, and counters.
`forever` also persists a canonical configuration and SHA-256 on first start.
Later bare invocations select the active run, reload its locked arguments, and
resume its latest committed state; a conflicting explicit argument is rejected.
The start machine is captured once as run provenance.
Each canonical RL process also appends a session to
`diagnostics/runtime_profile.json` inside the run directory. That atomic report
keeps fine-grained RL/PPO and periodic RL-vs-random timing cumulative across
`forever` resumes without placing timers inside opponent-model inference.

## Parallelism and device policy

Dataset games, RL rollouts, and diagnostic games are independent CPU work.
Their bounded worker pools use stable per-game seeds, preserve game-id ordering,
and reduce worker counts after resource or execution failures. RL worker
autotunes use separate seed streams and discard every benchmark trajectory.
The process running supervised or RL network updates is the only process
allowed to use CuPy/GPU.

This boundary prevents multiple worker processes from creating competing CUDA
contexts and keeps deterministic seeded results independent of scheduling.
See `GPU_SETUP.md` for the installation and runtime selection policy.

## Diagnostics and reports

`diagnostics.pairwise` alternates the evaluated agent between player positions,
writes one record per game, summarizes win/loss and choice opportunities,
and can generate plots. `diagnostics.evaluate` runs the four canonical agents
against `random` and atomically replaces the aggregate output directory only
after all requested artifacts are complete. For an RL checkpoint with `Wv` and
`bv`, evaluation also aggregates `V(s)` over real decision states from the
policy forward cache; it does not add a second network forward or affect the
chosen action.

`diagnostics.rl_progress` owns the canonical RL learning curve. It evaluates
RL versus random on a fixed periodic seed namespace, appends deduplicated JSONL
points, and derives CSV/PNG reports. Final all-pairs evaluation uses a distinct
holdout namespace. Diagnostic execution preserves parent training RNG state
and never mutates the checkpoint or training schedule.

Every new canonical pipeline initializes RL from its selected supervised
checkpoint, independent of an older RL output. Canonical
`--resume`/`--resume-from` restores exact numbered state.
Direct `training.rl.cli` calls continue an existing compatible RL checkpoint
by default, with `--fresh-from-sl` available for controlled new runs.

## UI

The UI asks the engine for legal actions and submits selected actions; it does
not own game rules. `GameController` coordinates snapshots, speed, pause,
history, menu changes, and human input. Layout code computes geometry without
drawing, while the renderer and HUD translate snapshots into OpenGL/Pygame
output. Persistent opponent models feed the two probability rows in the HUD.

## Compatibility boundaries

Treat these as persistent external contracts unless a change is explicitly
approved and documented:

- action shapes and `DominoEngine.step` return shape;
- ruleset-derived encoder/action dimensions and checkpoint array names;
- double-six's historical 168-feature/56-action layout and filenames;
- ruleset identity in new datasets, checkpoints, runs, and resume state;
- float64 checkpoint loading and optional value-head arrays;
- deterministic seed-to-game mapping;
- numbered checkpoint plus `.resume.npz` pairing and validation;
- atomic replacement of datasets, checkpoints, diagnostics, and reports.

See `CONTRIBUTING.md` for the required impact analysis and tests.
