#!/usr/bin/env bash
# Run every PPO harness variant on two checkouts and compare the payloads.
#
# Usage:
#   correctness_suite.sh REFERENCE_CHECKOUT CANDIDATE_CHECKOUT OUTPUT_DIR \
#     DOUBLE_SIX_BUFFER DOUBLE_SIX_WEIGHTS DOUBLE_THREE_BUFFER DOUBLE_THREE_WEIGHTS
#
# PYTHON selects the interpreter (default: python). Timings are not compared;
# use ``ppo_harness.py timing`` for those.
set -euo pipefail

if [[ $# -ne 7 ]]; then
  sed -n '2,9p' "$0"
  exit 2
fi

reference=$1
candidate=$2
output=$3
six_buffer=$4
six_weights=$5
three_buffer=$6
three_weights=$7
python=${PYTHON:-python}
harness="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ppo_harness.py"

mkdir -p "$output"

run_pair() {
  local name=$1
  shift
  local label
  for label in reference candidate; do
    local checkout=$reference
    [[ $label == candidate ]] && checkout=$candidate
    PYTHONPATH=$checkout "$python" "$harness" run "$@" \
      --output "$output/${name}_${label}.json" --save-parameters >/dev/null
  done
  PYTHONPATH=$candidate "$python" "$harness" compare \
    "$output/${name}_reference.json" "$output/${name}_candidate.json" \
    --output "$output/${name}_comparison.json" --oneline "$name"
}

for variant in production hot_lr entropy_shared_critic own_critic_dropout no_up_entropy_dropout; do
  run_pair "gpu_six_${variant}" --device gpu --variant "$variant" --iterations 2 \
    --buffer "$six_buffer" --weights "$six_weights"
done
run_pair "gpu_three_production" --device gpu --variant production --iterations 2 \
  --buffer "$three_buffer" --weights "$three_weights"
run_pair "cpu_six_hot_lr" --device cpu --variant hot_lr --iterations 1 --max-epochs 4 \
  --buffer "$six_buffer" --weights "$six_weights"
run_pair "cpu_six_entropy_shared_critic" --device cpu --variant entropy_shared_critic \
  --iterations 1 --max-epochs 2 --buffer "$six_buffer" --weights "$six_weights"
run_pair "cpu_three_own_critic_dropout" --device cpu --variant own_critic_dropout \
  --iterations 1 --max-epochs 3 --buffer "$three_buffer" --weights "$three_weights"
