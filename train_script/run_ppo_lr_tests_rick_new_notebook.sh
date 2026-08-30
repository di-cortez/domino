#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACHINE_SLUG="rick_new_notebook"
MACHINE_LABEL="Rick new notebook"
TIME_COEFFICIENT="1.5"
EXPERIMENT_KIND="ppo_lr"
RULESET="double-three"
RL_TIME_LIMIT=10800

# shellcheck source=train_script/_sequential_rl_experiment_runner.bash
source "$SCRIPT_DIR/_sequential_rl_experiment_runner.bash"
run_rl_experiment_sequence "$@"
