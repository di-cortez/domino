# Pylint Quality Ratchet

Pylint starts in report-only mode. The deliberately permissive limits in
`pyproject.toml` expose correctness and extreme structural debt without
blocking development. Every repository modification must still run the full
Pylint command from `CONTRIBUTING.md`; `exit-zero = true` changes the exit code,
not that contributor requirement.

Do not refactor code merely to improve the score. Correctness comes first, and
each limit is tightened only after the repository passes the preceding value.
This makes quality a ratchet: it can improve gradually but cannot regress.

## Current baseline

Stage 1 is recorded against the whole tracked Python repository. The baseline
date, revision, Pylint version, total count, and finding types are updated when
the initial report is generated.

<!-- PYLINT_BASELINE_START -->
- Recorded: `2026-07-23T21:38:07Z`
- Source state: revision `00921a1` plus the current Pylint/pipeline working-tree
  change
- Pylint: `4.0.6` with Astroid `4.0.4`, Python `3.12.3`
- Scope: `agents benchmarks diagnostics middleware tests train_script training
  ui utils`
- Total findings: **12**

| Symbol | Count | Initial classification |
|---|---:|---|
| `undefined-variable` | 2 | Correctness/false-positive review |
| `cyclic-import` | 1 | Architecture |
| `too-complex` | 1 | Structure |
| `too-many-arguments` | 3 | Structure |
| `too-many-branches` | 1 | Structure |
| `too-many-locals` | 3 | Structure |
| `too-many-statements` | 1 | Structure |

No enabled `used-before-assignment`, `dangerous-default-value`,
`duplicate-key`, `bare-except`, `unreachable`, `too-many-lines`,
`too-many-return-statements`, or `too-many-nested-blocks` findings were
reported. The two `undefined-variable` findings are both the imported Pygame
constants `DOUBLEBUF` and `OPENGL` in `ui/visual_main.py`; stage 2/3 must decide
whether they are real defects or import-related false positives before any
suppressions are added. Stage 1 is complete; no baseline finding was refactored
as part of recording it.
<!-- PYLINT_BASELINE_END -->

## Ten-stage tightening plan

1. **Record the baseline.** Run Pylint on the whole repository and save the
   number and types of findings. Do not refactor merely to improve the score.

2. **Fix correctness findings first.** Eliminate `undefined-variable`,
   `used-before-assignment`, `duplicate-key`, `bare-except`, `unreachable`, and
   dangerous mutable defaults.

3. **Remove false positives.** Replace wildcard imports where practical. Use
   narrowly scoped suppressions only when Pylint is demonstrably wrong.

4. **Make correctness checks blocking.** Keep structural checks informational,
   but configure CI to fail on the correctness rules from stage 2.

5. **Reduce extreme module size.** Lower `max-module-lines` from `2500` to
   `2000`, then `1500`. Split modules by responsibility, not arbitrarily.

6. **Reduce extreme function complexity.** Lower `max-complexity` from `50` to
   `40`, then `30`. Prioritize orchestration functions such as training
   pipelines.

7. **Reduce function state and interfaces.** Move toward:

   ```toml
   max-args = 15
   max-locals = 40
   ```

   Introduce configuration dataclasses instead of passing large collections of
   unrelated arguments.

8. **Reduce control-flow size.** Move toward:

   ```toml
   max-branches = 30
   max-statements = 120
   max-returns = 12
   max-nested-blocks = 8
   ```

9. **Resolve architectural findings.** Fix cyclic imports and move shared
   types/constants into lower-level modules. Afterward, make all currently
   enabled checks blocking by removing `exit-zero = true`.

10. **Adopt the long-term target.** A reasonable final configuration for this
    project is approximately:

    ```toml
    [tool.pylint.design]
    max-args = 8
    max-branches = 15
    max-complexity = 15
    max-locals = 25
    max-returns = 8
    max-statements = 60
    max-nested-blocks = 5

    [tool.pylint.format]
    max-module-lines = 1000
    ```

Only tighten a limit after the repository passes the previous value.
