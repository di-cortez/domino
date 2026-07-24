# Contributing

## Repository language

Keep all repository content in English: source, filenames, identifiers, CLI
options, messages, comments, docstrings, tests, report labels, and Markdown.
Generated user data is outside this rule, but generators and schemas remain in
English.

## Before changing code

1. Read the root README, `docs/ARCHITECTURE.md`, this file, and the README owned
   by the affected module.
2. Inspect `git status` and the existing diff. Preserve unrelated and
   uncommitted work.
3. Run a baseline proportional to the change and record its result.
4. Identify contracts, generated artifacts, deterministic behavior, and resume
   paths touched by the change.

Do not start a long training run or sweep as routine validation. Prefer unit
tests, fixed-seed micro-runs, `--dry-run`, `--report-only`, and small temporary
diagnostics.

## Impact matrix

Use this table to choose the minimum required validation; combine rows when a
change crosses multiple areas.

| Changed area | Required checks | Documentation owner |
|---|---|---|
| Rules, state, or actions | Core tests, headless regression tests, fixed-seed smoke | `middleware/README.md` |
| Opponent inference | Exact-model tests, cache/trace tests, agent regressions | `middleware/README.md` and architecture |
| Encoder or checkpoint shape | Core, supervised, RL, load/save compatibility | `agents/README.md`, `training/README.md`, `models/README.md` |
| Supervised training | Supervised autotuning tests, CPU smoke, compile | `training/README.md` |
| RL or resume | Core/parallel RL tests, checkpoint/resume smoke, seed comparison | `training/README.md` |
| Diagnostics | Diagnostic tests, CLI help, four-game temporary pairwise smoke | `diagnostics/README.md` |
| Shell pipeline or sweep | `bash -n`, help, parser tests, dry-run/report-only | `train_script/README.md` |
| GPI sweep/report | GPI tests, dry-run, report rebuild, CSV/XLSX when relevant | `train_script/README.md` |
| UI/controller | UI controller tests; visual run when rendering changed | `ui/README.md` or `ui/ui_workflow.md` |
| Documentation only | Link check, command/help verification, language/path search | Owning document |

## Compatibility policy

Compatibility is evidence-based, not the indefinite retention of every alias.
Before changing a public or persisted contract, search source, tests, docs,
scripts, and experiment readers. Preserve a compatibility path when existing
artifacts or supported commands require it. Remove a legacy shim when it has no
consumer, obscures ownership, or pretends to provide behavior it no longer
has; report intentional breakage with the replacement command or symbol.

The following need explicit review:

- CLI names and accepted values;
- importable classes, functions, and module attributes;
- action/state shapes and encoder dimensions;
- NPZ array names and float32/float64 loading;
- seed mapping and ordering;
- checkpoint/resume metadata and hashes;
- existing diagnostic and sweep directory layouts;
- CSV/JSON fields consumed by current report builders.

Do not silently reinterpret old state or checkpoints. Fail with a useful
message when safe conversion cannot be proven.

## Determinism and parallelism

Fixed seeds must describe game identity, not worker scheduling. A seeded run
should produce the same ordered records for one worker, multiple workers,
autotuning, and safe worker fallback. Parent processes own final ordering,
artifact writes, GPU contexts, and network updates.

When changing parallel code:

- keep queues and retained samples bounded;
- retain completed game ids across safe fallback;
- make partial worker failures visible;
- preserve CPU-only worker isolation from CUDA;
- compare fixed-seed fingerprints or records across worker counts;
- never use process completion order as report order.

## Artifact and write safety

Datasets, encoded caches, `.npz` weights, resume state, diagnostic results, and
sweep reports are generated artifacts. They are ignored by Git and may contain
hours of user work.

- Do not commit or manually edit them.
- Do not delete broad generated directories as cleanup.
- Write to a sibling temporary path and atomically replace the final artifact.
- A resumable RL checkpoint is a validated numbered weights file plus its
  matching `.resume.npz`; a lone file is not a safe continuation point.
- Reusing a diagnostic requires validating configuration, seed, requested
  games, completed records, model identity, and expected plots/files.
- Report builders must keep incomplete points explicit rather than converting
  missing measurements to zero.

## Tests and command checks

Run Pylint after every repository modification, including source,
configuration, tests, scripts, and documentation:

```bash
python -m pylint agents benchmarks diagnostics middleware tests train_script \
  training ui utils
```

This requirement applies while `exit-zero = true`: Pylint is initially a
report-only technical-debt inventory, so contributors must review its output
even though findings do not fail the command. Do not perform unrelated
score-driven refactors. Install the development tools with
`python -m pip install -r requirements-dev.txt`; the baseline and tightening
sequence are maintained in [`docs/PYLINT_ROADMAP.md`](docs/PYLINT_ROADMAP.md).

The standard full validation is:

```bash
python -m pylint agents benchmarks diagnostics middleware tests train_script \
  training ui utils
python -m pytest -q
python -m unittest discover -s tests -v
python -m compileall -q agents diagnostics middleware training ui utils \
  train_script
bash -n train_script/run_training_pipeline.sh
bash -n train_script/run_rl_parameter_sweep.sh
bash -n train_script/run_rl_test_diagnostics.sh
git diff --check
```

Run `shellcheck` for changed shell scripts when it is installed. Verify `--help`
for every changed CLI. A diagnostic smoke should use a temporary directory and
only four random-vs-random games:

```bash
python -m diagnostics.pairwise --agent random --opponent random \
  --games 4 --seed 1 --workers 1 --no-plots --output /tmp/domino_pair_smoke
```

If a relevant test cannot run, state which one and why. Lack of GPU hardware is
not a reason to skip CPU behavior and device-selection tests.

## Documentation ownership

- Root `README.md`: quick setup, first run, common commands, generated paths,
  test entry point, and links only.
- `docs/ARCHITECTURE.md`: stable cross-module boundaries and data flow.
- `docs/GPU_SETUP.md`: GPU installation and troubleshooting only.
- Module READMEs: complete options, algorithms, artifacts, and operational
  behavior owned by that module.
- `CONTRIBUTING.md`: contributor process and validation policy.
- `AGENTS.md`: short agent-facing pointer to this policy.

Update documentation in the same change as behavior. Do not copy a large
section into multiple owners; link to the canonical document.

## Definition of done

A change is complete when:

- behavior matches the request and architectural boundary;
- existing work is preserved;
- legacy decisions cite consumer evidence and compatibility risk;
- deterministic and persisted contracts are preserved or the break is explicit;
- generated writes remain atomic and resumable artifacts remain validated;
- affected tests, helps, syntax checks, smoke runs, and link checks pass;
- documentation owned by every affected module is current;
- `git diff --check` is clean;
- the handoff lists changed/removed files, intentional breaks, tests run, tests
  not run, and the final diff/status.
