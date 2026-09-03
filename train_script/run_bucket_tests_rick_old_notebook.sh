#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACHINE_SLUG="rick_old_notebook"
MACHINE_LABEL="Rick old notebook"
TIME_COEFFICIENT="3.4"
EXPERIMENT_KIND="buckets"
RULESET="double-six"
RL_TIME_LIMIT=61200

# shellcheck source=train_script/_sequential_rl_experiment_runner.bash
source "$SCRIPT_DIR/_sequential_rl_experiment_runner.bash"
run_rl_experiment_sequence "$@"
