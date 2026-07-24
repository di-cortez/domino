# Documentation

This index separates project-wide concepts from module-owned operational
details. Keep the root [`README.md`](../README.md) focused on setup and common
commands; put deeper material in the owner listed below.

## Project-wide guides

| Document | Owner of |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Boundaries, data flow, state/action contracts, concurrency, and artifact flow. |
| [`GPU_SETUP.md`](GPU_SETUP.md) | NVIDIA driver, CUDA/CuPy installation, verification, device policy, and troubleshooting. |
| [`PYLINT_ROADMAP.md`](PYLINT_ROADMAP.md) | Current baseline and the staged Pylint quality ratchet. |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Repository policy, compatibility, determinism, test matrix, documentation ownership, and definition of done. |
| [`../AGENTS.md`](../AGENTS.md) | Short instructions for automated coding agents. |

## Module documentation

| Document | Source of truth for |
|---|---|
| [`../middleware/README.md`](../middleware/README.md) | Rules, game state, action format, headless step contract, and exact opponent inference. |
| [`../agents/README.md`](../agents/README.md) | Agent implementations, 168-feature encoder, 56 policy actions, and checkpoint shapes. |
| [`../training/README.md`](../training/README.md) | Dataset loading, supervised training, RL algorithm, devices, parallel rollout, checkpointing, and resume. |
| [`../diagnostics/README.md`](../diagnostics/README.md) | Supported agents, evaluation commands, output schemas, plots, statistics, and report interpretation. |
| [`../train_script/README.md`](../train_script/README.md) | Shell pipeline, fixed-GPI RL parameter sweep, budgets, safe resume, and report locations. |
| [`../ui/README.md`](../ui/README.md) | Simulator startup, controls, HUD, agent menu, and UI tests. |
| [`../ui/ui_workflow.md`](../ui/ui_workflow.md) | Controller, human-turn, visibility, layout, and rendering sequence. |
| [`../dataset/README.md`](../dataset/README.md) | Generated supervised dataset and encoded-cache formats. |
| [`../models/README.md`](../models/README.md) | Generated policy checkpoint names and stored arrays. |
| [`../diagnostics/self_play_evaluation/README.md`](../diagnostics/self_play_evaluation/README.md) | Comparison helper for two RL training regimes. |

## Where changes belong

- A command or flag change belongs in the README for the command's module.
- A state, action, or checkpoint contract change belongs in the owning module
  README and, if it crosses boundaries, in `ARCHITECTURE.md`.
- GPU installation or device-selection troubleshooting belongs only in
  `GPU_SETUP.md`; other documents should link to it.
- Contributor process and validation requirements belong only in
  `CONTRIBUTING.md`.
- The root README should contain only the shortest successful path and links to
  the detailed owner.

Generated reports are outputs, not documentation sources. Do not edit a CSV,
JSON, XLSX, PNG, PDF, dataset, or checkpoint to describe a code change.
