#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACHINE_SLUG="rick_desktop"
MACHINE_LABEL="Rick desktop"
TIME_COEFFICIENT="2.4"
EXPERIMENT_KIND="baselines"
RULESET="double-three"
RL_TIME_LIMIT=17280
INCLUDE_LOOKUP_BASELINE=1

# shellcheck source=train_script/_sequential_rl_experiment_runner.bash
source "$SCRIPT_DIR/_sequential_rl_experiment_runner.bash"
run_rl_experiment_sequence "$@"
