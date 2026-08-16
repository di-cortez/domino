# Dataset generation

Deterministic supervised examples generated from heuristic self-play.
Owned by `training/datagen/`; the canonical pipeline drives this stage
through `training/pipeline.py`.

| File | Purpose |
|---|---|
| `generator.py` | Coordinates retained worker tuning, bounded SQLite aggregation, and atomic ordered JSONL output. |
| `parallel.py` | Plays deterministic dataset games in a bounded CPU-only worker pool with dynamic scheduling and memory fallback. |


Run:

```bash
python -m training.datagen.generator
python -m training.datagen.generator --workers auto --seed 123
python -m training.datagen.generator --workers 4 --games 5000
python -m training.datagen.generator --ruleset double-three --games 1000 --seed 123
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
Workers receive only the canonical ruleset name and resolve the immutable game
geometry locally. Every row includes that name and initial hand size; compact
outputs use ruleset-specific default filenames.

Each successful generation also writes
`<dataset-stem>.random_manifest.json` beside the JSONL. It records the root
seed, NumPy bit generator, derivation scheme, and registered namespaces. There
is no large per-game seed file: every game stream is reproducibly calculated
from `(root seed, DATASET_GAME, absolute game id)`.

Workers serialize one compact payload per game. Only the parent writes those
payloads to a disposable SQLite database, keeping RAM bounded while results
arrive out of order. Final rows are emitted in game-id order, and the existing
JSONL is replaced atomically only after every requested game succeeds. Dataset
generation has no partial-resume mode: an interrupted temporary database is
discarded, and the next invocation either reuses a previously complete
metadata-validated dataset or generates the requested dataset from the start.

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
