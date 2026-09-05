#!/usr/bin/env bash
# One-factor sweep over the project defaults, on the Diego notebook.
#
# Twenty-three `forever` runs of five hours each. Every run uses the project
# defaults except for the single parameter it tests, so any difference against
# the `control` run is attributable to that one parameter.
#
# The defaults these runs sit on (training/rl/config.py, reward_model.py,
# reward_distance.py, pool.py, pipeline.py):
#
#   seed 52 | reward_eta 0.115 | gamma_f 0.95 | gamma_i 0.90
#   opponent bucket `random` | baseline `lookup-table`
#   reward distance `decision-decision` | entropy coefficient 0
#   learning rate 0.01 | gpi 2000
#
# The requested grid asks for six values that already *are* the default --
# bucket `random`, baseline `lookup-table`, distance `decision-decision`,
# entropy 0, lr 0.01, gpi 2000. Under one seed those six are the same run, so
# they are collapsed into the single `control` point rather than costing five
# hours each to recompute identical weights. The same applies to the reward
# weight pairs: their (1,1) reference is the default, so `control` is that
# point for both groups too.
#
#   control                    everything at the default
#   bucket_heuristic           --opponent-buckets heuristic
#   baseline_zero              --baseline zero
#   baseline_batch_mean        --baseline batch-mean
#   distance_turn_turn         --reward-distance-mode turn-turn
#   distance_turn_decision     --reward-distance-mode turn-decision
#   distance_decision_turn     --reward-distance-mode decision-turn
#   entropy_0p01               --entropy-coef 0.01
#   entropy_0p1                --entropy-coef 0.1
#   lr_0p005 .. lr_0p04        --learning-rate 0.005 / 0.02 / 0.03 / 0.04
#   gpi_1000, gpi_4000         --gpi 1000 / 4000
#
# Terminal reward pair (a_E, a_B) -- empty hand against blocked:
#   terminal_aE1_aB0           --terminal-empty-hand-weight 1 --terminal-blocked-weight 0
#   terminal_aE0_aB1           --terminal-empty-hand-weight 0 --terminal-blocked-weight 1
#   terminal_aE1_aB2           --terminal-empty-hand-weight 1 --terminal-blocked-weight 2
#   terminal_aE2_aB1           --terminal-empty-hand-weight 2 --terminal-blocked-weight 1
#
# Immediate reward pair (a_P, a_D) -- pass against draw, in that order:
#   immediate_aP1_aD0          --immediate-pass-weight 1 --immediate-draw-weight 0
#   immediate_aP0_aD1          --immediate-pass-weight 0 --immediate-draw-weight 1
#   immediate_aP1_aD2          --immediate-pass-weight 1 --immediate-draw-weight 2
#   immediate_aP2_aD1          --immediate-pass-weight 2 --immediate-draw-weight 1
#
# Each pair is normalized by its own larger member, so only the ratio inside a
# pair carries meaning and neither pair may be (0, 0).
#
# RULESET is `double-six` because the `lookup-table` baseline needs a packaged
# format-version-3 reward table, and double-six is the only ruleset that has
# one. Pointing this at double-three requires rebuilding that artifact first --
# see training/rl/reward_lookup_tables/README.md.
#
# Each run's analysis bundle is named after the parameter it tests, so a
# directory copied out of its run still says which point it is:
#
#   models/rl/domino_rl_forever_seed52_runone_factor_lr_0p02_diego_notebook/
#   └── 20260904-XXX_diego_notebook_lr_0p02/
#
# A decimal point is written `p`, so `lr_0p02` and `lr_0p2` stay distinct.
# Substitute the `XXX` ordinal by hand once a run has stopped; the bundle stays
# discoverable because nothing keys on the directory's name.
#
# Usage:
#   ./run_one_factor_tests_diego_notebook.sh                 # run the sequence
#   ./run_one_factor_tests_diego_notebook.sh --help          # options
#   ./run_one_factor_tests_diego_notebook.sh --only 'one_factor_lr_*'
#   ./run_one_factor_tests_diego_notebook.sh --rl-time-limit 3h
#
# The sequence is resumable: each point records its consumed RL time, and
# re-running continues an interrupted point instead of restarting it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACHINE_SLUG="diego_notebook"
MACHINE_LABEL="Diego notebook"
TIME_COEFFICIENT="1.0"
EXPERIMENT_KIND="one_factor"
RULESET="double-six"
# Five hours of RL wall-clock time per run, as requested.
RL_TIME_LIMIT=18000

# shellcheck source=train_script/_sequential_rl_experiment_runner.bash
source "$SCRIPT_DIR/_sequential_rl_experiment_runner.bash"
run_rl_experiment_sequence "$@"
