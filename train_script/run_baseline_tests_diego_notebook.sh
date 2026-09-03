#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACHINE_SLUG="diego_notebook"
MACHINE_LABEL="Diego notebook"
TIME_COEFFICIENT="1.0"
EXPERIMENT_KIND="baselines"
RULESET="double-three"
RL_TIME_LIMIT=7200
INCLUDE_LOOKUP_BASELINE=1

# shellcheck source=train_script/_sequential_rl_experiment_runner.bash
source "$SCRIPT_DIR/_sequential_rl_experiment_runner.bash"
run_rl_experiment_sequence "$@"
