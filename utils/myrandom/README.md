# Central randomness package

`utils.myrandom` is the single authority being adopted for randomness in the
Domino project. Supervised dataset generation and supervised optimization are
integrated consumers. RL, diagnostics, general agents, and UI code have not yet
been migrated.

## Contract

- Every reproducible stream uses `numpy.random.Generator` with `PCG64`.
- A run owns one immutable root seed and one `SeedPlan`.
- Logical streams use registered `RandomNamespace` values.
- Coordinates identify work, such as game, epoch, iteration, or minibatch IDs.
- Stream derivation depends only on root seed, namespace, and coordinates; it
  never depends on creation order or worker scheduling.
- No module-global NumPy generator is used.
- Operating-system entropy remains isolated in `entropy.py` for fresh root
  seeds and collision-resistant technical identifiers.
- A small manifest records the root seed and derivation contract. Per-game seed
  files are unnecessary because every stream can be derived on demand.
- Sequential generator state can be stored as JSON when reconstruction from
  coordinates is not sufficient.

## Intended use

```python
from utils.myrandom import RandomNamespace, SeedPlan

plan = SeedPlan(root_seed=42)
game_rng = plan.generator(RandomNamespace.RL_GAME, 12_345)

tiles = list(range(28))
game_rng.shuffle(tiles)
position = int(game_rng.integers(0, 2))
```

Creating streams in another order produces the same stream for game 12,345.
Workers should derive generators from stable game IDs, not worker IDs.

At the beginning of a future run, write one static manifest:

```python
plan.write_manifest(run_directory / "random_manifest.json")
```

Generator state persistence is reserved for genuinely sequential streams:

```python
from utils.myrandom import generator_from_state, generator_state

saved = generator_state(game_rng)
restored = generator_from_state(saved)
```

## Current migration boundary

Dataset games derive `RandomNamespace.DATASET_GAME` streams from their absolute
game IDs. Supervised optimization derives independent initialization, epoch
shuffle, and dropout streams. Both stages write a root manifest beside their
primary artifact. Other project areas continue using their historical
randomness until their own migration is performed.

## Repository migration inventory

This inventory was refreshed by static inspection on 2026-08-09, after
supervised optimization became the second `utils.myrandom` consumer. It covers
every Python module containing `random`, every direct standard-library `random`
import, NumPy/CuPy/backend random APIs, and `secrets`. String-only references to
the random baseline are separated from calls that consume entropy.

Current counts overlap by design:

- 51 Python modules contain `random` in source, identifiers, strings, comments,
  or docstrings.
- 16 modules import the standard-library `random` package directly.
- 21 modules access `numpy.random`, `cupy.random`, `xp.random`, or their legacy
  global state APIs.
- 10 modules import `secrets`.
- Including the two `secrets`-only modules, the broad inventory covers 53
  Python files.

The dataset migration deliberately leaves old APIs elsewhere. A file importing
`utils.myrandom` is not evidence that it consumes global NumPy state: migrated
callers receive explicit `numpy.random.Generator` instances from `SeedPlan`.

### Completed: supervised dataset generation

| Module | Current behavior |
|---|---|
| `training/datagen/generator.py` | Creates one `SeedPlan` from the explicit seed or `fresh_root_seed()`, writes the root manifest, and sends the plan plus absolute game IDs to workers. |
| `training/datagen/parallel.py` | Reconstructs `RandomNamespace.DATASET_GAME/<game_id>` for each game. It no longer imports or seeds standard-library `random`, NumPy globals, or CuPy. |
| `middleware/domino_engine.py` | Accepts an optional generator exposing `shuffle`; dataset games pass their explicit NumPy generator. The non-migrated default path still calls `random.shuffle`. |
| `training/canonical_assets.py` | Records `PCG64` and the namespace/coordinate derivation scheme in canonical dataset identity. |

The root manifest is static. Per-game streams are calculated on demand, and
the temporary SQLite aggregation stores no per-game seed column. Consequently,
worker count, scheduling, autotuning retention, and memory fallback cannot
change a game's deal.

### Completed: supervised optimization

| Module | Current behavior |
|---|---|
| `training/supervised/training_loop.py` | Creates one `SeedPlan` from `--sl-seed` or `fresh_root_seed()`, injects it into the network and data plan, writes `<weights-stem>.random_manifest.json`, and reports the effective root seed. It no longer seeds NumPy or CuPy globals. |
| `training/supervised/runtime.py` | Derives `SUPERVISED_SHUFFLE/<epoch>` on the host for every storage mode. A fully GPU-resident run transfers those indices to CuPy instead of drawing a second backend-specific permutation. |
| `agents/nn.py` | The injected-plan path derives initialization and sequential dropout generators from separate namespaces, produces their random arrays with NumPy `Generator`, and transfers arrays to the selected backend. The legacy no-plan path remains for non-migrated RL and inference construction. |
| `training/pipeline.py` | Includes the bit generator and derivation scheme in canonical supervised identity, so legacy globally-seeded weights are not silently reused under the new seed meaning. |

The shuffle stream is coordinate-derived, so generating epoch 10 never depends
on whether an earlier epoch generator was constructed. Initialization and
dropout are separate streams, so changing one responsibility cannot perturb the
other. Dropout is sequential within a complete supervised invocation; partial
supervised resume is intentionally unsupported.

### Standard-library `random` imports still to migrate

There are no `from random import ...` statements. Every remaining case uses
`import random` and qualified calls.

| Module | Use and migration concern |
|---|---|
| `agents/agent.py` | `RandomAgent` selects a legal action with `random.choice`; it should receive an explicit agent/action stream. |
| `agents/neural_agent.py` | Epsilon-greedy exploration tests NumPy global state and selects with `random.choice`; both decisions belong to one explicit policy stream. |
| `benchmarks/headless_step_benchmark.py` | Reseeds Python and NumPy globals for fixed comparisons; benchmark game IDs should derive benchmark streams instead. |
| `diagnostics/parallel_runner.py` | Reseeds Python and NumPy globals once per diagnostic game; this is the central diagnostics migration point. |
| `diagnostics/rl_progress.py` | Saves and restores both global RNG states around periodic evaluation; explicit diagnostic streams should make this unnecessary. |
| `middleware/domino_engine.py` | The compatibility/default deal path still uses `random.shuffle`; dataset calls already bypass it with an injected generator. |
| `training/rl/adaptive_tuning.py` | Snapshots and restores global Python/NumPy states so benchmarks do not perturb training. Independent namespaces should replace this protection. |
| `training/rl/parallel.py` | Reseeds Python and NumPy globals inside each rollout worker. Future RL games should derive streams from absolute game identity. |
| `training/rl/resume.py` | Persists and restores Python and NumPy global states. Future resume should store only genuinely sequential named-generator states. |
| `training/rl/rollout.py` | Uses `random.randint` for learner position and `random.choice` for the opponent pool. These need separate stable RL namespaces. |
| `training/rl/training_loop.py` | Chooses a fallback root seed with `secrets`, then seeds both Python and NumPy globals. It should construct the RL `SeedPlan` instead. |
| `ui/ui_agents.py` | `RandomUIAgent` uses `random.choice`; interactive runs need a UI-owned explicit generator. |
| `tests/test_adaptive_tuning.py` | Exercises the old global-state snapshot contract; replace as adaptive tuning migrates. |
| `tests/test_exact_model_optimization.py` | Uses local/global Python generators to synthesize reproducible test cases and actions. Local explicit NumPy test streams can replace them. |
| `tests/test_headless_engine_step.py` | Still seeds Python/NumPy globals for non-dataset diagnostics and RL compatibility checks; the dataset part already uses `SeedPlan`. |
| `tests/test_parallel_dataset.py` | Seeds globals only to prove that migrated dataset generation leaves them untouched; retain this regression test until global APIs disappear. |

### NumPy, CuPy, and backend random APIs still to migrate

Package-internal construction in `utils/myrandom/generators.py`,
`seed_plan.py`, and `serialization.py` is intentional. The remaining runtime
callers are migration work:

| Module | Use and migration concern |
|---|---|
| `agents/encoder.py` | `np.random.choice` samples a legal policy action. Pass an explicit policy generator. |
| `agents/neural_agent.py` | `np.random.rand` controls epsilon exploration. Share the explicit stream used for the exploratory choice. |
| `agents/nn.py` | The non-migrated fallback still exposes `xp.random.RandomState/randn`, `xp.random.random`, and `host_np.random.permutation` to RL/inference and low-level callers that do not inject a `SeedPlan`. The supervised entry point always injects one. |
| `agents/rl_nn.py` | Generates RL dropout masks with host `np.random.random` and copies them to the selected backend. An explicit resumable RL-dropout generator should replace the global host state. |
| `benchmarks/headless_step_benchmark.py` | Calls `np.random.seed` beside Python seeding; derive each benchmark game instead. |
| `diagnostics/parallel_runner.py` | Calls `np.random.seed` per game; derive each diagnostic game instead. |
| `diagnostics/rl_progress.py` | Uses `np.random.get_state/set_state`; separate streams should remove this coupling. |
| `training/rl/adaptive_tuning.py` | Saves/restores NumPy global state; replace with isolated autotune namespaces. |
| `training/rl/ppo.py` | A local `np.random.RandomState(seed)` permutes decisions into minibatches. Migrate to `SeedPlan.generator(PPO_MINIBATCH, iteration, epoch)`. |
| `training/rl/parallel.py` | Calls `np.random.seed` in rollout workers; use per-game RL streams. |
| `training/rl/resume.py` | Serializes the legacy NumPy global state; replace with named sequential generator snapshots only where coordinate derivation is insufficient. |
| `training/rl/training_loop.py` | Seeds the NumPy global state at startup; construct and distribute an RL seed plan instead. |

Tests with direct legacy NumPy state are
`tests/test_adaptive_tuning.py`, `tests/test_core.py`,
`tests/test_headless_engine_step.py`, `tests/test_parallel_dataset.py`, and
`tests/test_supervised_autotuning.py`. Package tests in
`utils/myrandom/tests/test_seed_plan.py` intentionally inspect NumPy global
state only to prove that `SeedPlan` derivation does not consume it.

### `secrets` inventory

`secrets` has two distinct responsibilities. Root-seed creation affects the
stochastic experiment; temporary-name generation does not.

Seed-affecting uses:

- `diagnostics/evaluate.py`: fallback 63-bit root seed for a complete
  evaluation.
- `diagnostics/pairwise.py`: fallback 63-bit root seed for one pair.
- `training/pipeline.py`: fallback seed for ephemeral pipeline levels.
- `training/rl/training_loop.py`: fallback 63-bit RL root seed.
- `utils/myrandom/entropy.py`: the intended central `fresh_root_seed()` owner.

Technical identifiers only:

- `diagnostics/runtime_profile.py`: collision-resistant temporary profile name.
- `training/rl/adaptive_tuning.py`: atomic-write temporary name.
- `training/rl/reporting.py`: atomic-write temporary name.
- `training/rl/resume.py`: temporary checkpoint names.
- `utils/artifacts.py`: shared atomic-write temporary names.

The technical `token_hex` calls do not affect model parameters, games, or
training order. They can migrate to `utils.myrandom.unique_token()` without any
experiment-level seed contract. The fallback `randbits` calls must instead
produce one root seed and then hand control to `SeedPlan`.

### Random-baseline names that do not themselves draw randomness

The following modules mainly carry agent names, matchup names, report keys,
paths, CLI text, or test expectations. The actual draw occurs in another
module:

- `diagnostics/gameplay.py`: random-agent selection and construction; the
  agent itself owns the random draw.
- `diagnostics/plots.py`: random-baseline filters, labels, and plot text.
- `tests/test_canonical_pipeline.py`: configuration, namespace, and mocked
  fallback-seed expectations.
- `tests/test_parallel_diagnostics.py`: matchup names and aggregate result keys.
- `tests/test_parallel_rl.py` and `tests/test_ppo.py`: network `random_seed`
  arguments.
- `tests/test_runtime_profile.py`: RL-vs-random profile keys and artifact paths.
- `train_script/run_pipeline.py`: compact random-baseline diagnostic text.
- `training/canonical_run.py`: persisted `uniform_random`/opponent-selection
  metadata.
- `training/rl/cli.py`: seed help text.
- `ui/test_ui_controller.py`: verifies that the `random` UI choice constructs
  `RandomUIAgent`.

`diagnostics/evaluate.py`, `diagnostics/pairwise.py`,
`diagnostics/parallel_runner.py`, `diagnostics/rl_progress.py`,
`benchmarks/headless_step_benchmark.py`, and `training/pipeline.py` have both
random-baseline naming and real entropy/state responsibilities, so they are
listed in the actionable sections above.

### Current end-to-end randomness flows

Dataset generation, now migrated:

```text
explicit/fresh root seed
    -> SeedPlan
    -> DATASET_GAME + absolute game_id
    -> independent PCG64 Generator
    -> DominoEngine shuffle
    -> ordered parent SQLite aggregation
```

RL, not yet migrated:

```text
pipeline/self_play fallback or explicit seed
    -> global random.seed + np.random.seed
    -> worker reseeding
    -> Python learner-position and opponent-pool choices
    -> NumPy policy sampling and host dropout
    -> local legacy RandomState PPO permutation
    -> global Python/NumPy state persisted by resume
```

Supervised optimization, now migrated:

```text
explicit/fresh root seed
    -> SeedPlan
    +-> SUPERVISED_INITIALIZATION -> host PCG64 weights -> CPU/GPU backend
    +-> SUPERVISED_SHUFFLE + epoch -> host permutation -> CPU/GPU indices
    +-> SUPERVISED_DROPOUT -> sequential host masks -> CPU/GPU backend
    -> supervised random manifest beside final weights
```

Diagnostics, not yet migrated:

```text
explicit/secrets root seed
    -> SplitMix-style per-game scalar seeds
    -> worker global Python/NumPy reseeding
    -> random baseline choices and NumPy policy sampling
    -> save/restore parent global state for periodic monitoring
```

### Recommended migration order

1. Keep the dataset contract as the reference implementation: root manifest,
   stable namespace, absolute work identity, and no per-item seed table.
2. Migrate diagnostics next because games already have absolute IDs and their
   workers closely resemble dataset workers. Inject generators into random
   agents and policy sampling instead of reseeding processes.
3. Migrate RL rollouts by assigning distinct namespaces to deal, learner
   position, opponent selection, and action sampling. Identity must use
   absolute cumulative game ID, never worker ID or completion order.
4. Migrate PPO minibatch permutations using iteration/epoch coordinates.
5. Migrate supervised initialization, epoch shuffle, and dropout as independent
   streams. Host-generated NumPy arrays may be transferred to GPU during this
   organization-first migration.
6. Replace legacy resume payloads with the package's named-generator snapshots
   only for truly sequential streams; coordinate-derived streams need no saved
   state.
7. Route technical temporary tokens and every remaining fallback root seed
   through `utils.myrandom.entropy`.
8. Update benchmarks and tests last, retaining regressions that prove worker
   scheduling and diagnostics cannot perturb training streams.

The final repository rule remains: only `utils.myrandom` constructs generators
or obtains operating-system entropy; callers receive explicit streams, CuPy
does not own random state, and no worker invents a seed formula.
