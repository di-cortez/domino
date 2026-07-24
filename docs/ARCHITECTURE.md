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

## Rules and orchestration

`middleware.domino_engine.DominoEngine` owns dealing, turns, board ends, stock,
draw/pass behavior, blocked-game resolution, winner selection, legal actions,
and serialized state. The stable action forms are:

- `(tile, side)` for a play on the left (`0`) or right (`1`) end;
- `("DRAW", None)` for a legal stock draw;
- `None` for a legal pass.

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

The model returns seven suit-presence probabilities for the acting player:
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

`DominoEncoder` is shared by supervised and RL paths. It produces 168
public-information features and maps real tile decisions to 56 outputs: 28
tiles times two board ends. Draw, pass, and single-option tile plays are forced
by the engine and bypass neural inference and policy-gradient sampling.

`SupervisedNeuralNetwork` is a float32 MLP with policy shape
`168 -> 256 -> 128 -> 56`. `PolicyNetwork` extends it with masked
policy-gradient updates and an optional training-only value head. Policy
checkpoints store `W1`, `b1`, `W2`, `b2`, `W3`, and `b3`; critic-enabled
checkpoints also store `Wv` and `bv`. Current code still loads compatible
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

Supervised training can keep encoded arrays in host RAM, use atomic disk-backed
memory maps, and place all or rotating windows of data on the GPU. It saves the
best validation checkpoint atomically and renders the training/validation loss
history already collected during that run. The epoch count is a maximum
budget: after batch-size tuning is complete, repeated low-improvement blocks
of training loss can stop a saturated run early.

RL uses fresh on-policy trajectories: all games in an iteration observe the
same frozen policy. The default update stores masked collection-time
log-probabilities, normalizes advantages once over the complete decision
buffer, and runs masked PPO in deterministic minibatches. Direct and finite
canonical runs default to at most four epochs; `forever` defaults to 16, with
a whole-buffer KL guard after every completed epoch. The optional policy-only
`reinforce_v1` update instead applies one
full-buffer policy-gradient step and skips PPO buffer construction, ratios,
clipping, KL control, minibatches, and post-update full-buffer evaluation.
There is no replay buffer or cross-iteration reuse in either mode. Decision
returns are not rescaled by the number of legal choices. Opponent-pool
snapshots refresh by cumulative training-game thresholds, and a checksummed
`.resume.npz` preserves the selected algorithm, policy, optimizer, RNG,
adaptive selections, counters, and pool.

`training.pipeline` owns canonical orchestration. `small` and `default` are
ephemeral profiles: they choose a random seed by default, build 10,000- and
50,000-game datasets respectively, and place non-reused supervised assets in a
unique run namespace. `big`, `huge`, and `forever` default to seed 42 and reuse
the same compatible 100,000-game standard dataset and supervised checkpoint.
The RL budget is cumulative games, with shortened final and milestone
iterations. Canonical `big`, `huge`, and `forever` state uses immutable payload
generations and an atomic `training_state.json` marker so resume restores weights,
optimizer, RNGs, pool order/provenance, adaptive choices, and counters.
Each canonical RL process also appends a session to
`diagnostics/runtime_profile.json` inside the run directory. That atomic report
keeps fine-grained RL/PPO and periodic RL-vs-random timing cumulative across
`forever` resumes without placing timers inside opponent-model inference.

## Parallelism and device policy

Dataset games, RL rollouts, and diagnostic games are independent CPU work.
Their bounded worker pools use stable per-game seeds, preserve game-id ordering,
and reduce worker counts after resource or execution failures. RL GPI/worker
autotunes use separate seed streams and discard every benchmark trajectory.
The process running supervised or RL network updates is the only process
allowed to use CuPy/GPU.

This boundary prevents multiple worker processes from creating competing CUDA
contexts and keeps deterministic seeded results independent of scheduling.
See `GPU_SETUP.md` for the installation and runtime selection policy.

## Diagnostics and reports

`diagnostics.pairwise` alternates the evaluated agent between player positions,
writes one record per game, summarizes win/draw/loss and choice opportunities,
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

Parameter and games-per-iteration sweeps train points sequentially while each
point can use internal rollout workers. Their manifests, fingerprints, hashes,
numbered checkpoints, metrics, and diagnostic artifacts support conservative
resume. Report builders consume those immutable run artifacts to create CSV,
JSON, XLSX, PNG, and PDF outputs.

Every new canonical pipeline and parameter-sweep point initializes RL from its
selected supervised checkpoint, independent of an older RL output. Canonical
`--resume`/`--resume-from` and sweep resume restore exact numbered state.
Direct `training.self_play` calls continue an existing compatible RL checkpoint
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
- the 168-feature/56-action encoder and checkpoint array names;
- float64 checkpoint loading and optional value-head arrays;
- deterministic seed-to-game mapping;
- numbered checkpoint plus `.resume.npz` pairing and validation;
- existing generated sweep layouts that report/resume tools read;
- atomic replacement of datasets, checkpoints, diagnostics, and reports.

See `CONTRIBUTING.md` for the required impact analysis and tests.
