# Coding Agent Instructions

Work from the repository root and keep every repository artifact in English.

Read `CONTRIBUTING.md`, `docs/ARCHITECTURE.md`, and the affected module README
before editing. `CONTRIBUTING.md` is the authoritative maintenance and
validation policy; this file is only its short agent-facing entry point.

Inspect `git status` and the current diff first. Preserve user work and avoid
editing generated datasets, checkpoints, resume state, and diagnostic reports.

Respect ownership boundaries:

- `middleware/` owns game rules and state;
- `agents/` owns policy choices and encoding;
- `training/` owns learning and checkpoints;
- `diagnostics/` owns evaluation and reports;
- `ui/` owns interaction and rendering;
- module READMEs own detailed commands and behavior.

Keep fixed-seed results independent of worker scheduling. Worker processes stay
CPU-only; parent processes own GPU use, ordering, updates, and final writes.

Preserve checkpoint keys, safe float64 loading, numbered resume pairs, hashes,
existing sweep layouts, and atomic output replacement unless the task
explicitly authorizes a documented compatibility break.

Do not start long training or sweeps for validation. Use focused tests,
fixed-seed micro-runs, dry-runs, report-only paths, and temporary outputs.

Run the impact-matrix checks from `CONTRIBUTING.md`, verify changed CLI help and
shell syntax, update the owning documentation, and finish with
`git diff --check` plus a concise test/diff report.
