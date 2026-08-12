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
training arguments do not need to be repeated. Conflicting explicit values are
reported in one warning, ignored, and replaced from `run_config.json`.

The reusable standard assets are built from 100,000 heuristic games with a
maximum supervised budget of 5,000 epochs (the convergence/plateau stopping
rules can finish earlier). They are
`dataset/supervised_dataset_standard_seed<seed>.jsonl` and
`models/domino_sl_standard_seed<seed>.npz`. Their sibling `meta.json` files
record structural versions, configuration, provenance, convergence fields,
and SHA-256. They deliberately omit creation timestamps and machine-local
paths, so equal runs can publish equal metadata in different checkouts.
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
  --resume models/rl/domino_rl_huge_seed42

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
an attempted algorithm override during resume is warned about and ignored.
Both algorithms support the optional value head, which remains off by default. A
canonical PPO actor-critic run can be started and resumed with:

```bash
python -m training.pipeline forever --value-head --run-name critic
python -m training.pipeline forever
```

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

On the first `forever` start, `run_config.json` stores every locked argument,
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
| Supervised training | `python -m training.supervised.training_loop` |
| Self-play RL | `python -m training.rl.self_play` |
| Full canonical pipeline | `python -m training.pipeline <level>` |

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
python -m training.datagen.generator
python -m training.supervised.training_loop
python -m training.rl.self_play
```

## Worker controls

The Python pipeline exposes independent dataset, RL rollout, and diagnostic
worker controls. Dataset generation tunes once for its full workload. RL tunes
across complete early iterations, and diagnostics tune each matchup separately.
All three retain benchmark work and enforce the same hard limit of 20 workers.
