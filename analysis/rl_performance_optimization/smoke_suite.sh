#!/usr/bin/env bash
# Run the three fixed-seed training smokes for one checkout.
#
# Usage:
#   smoke_suite.sh CHECKOUT OUTPUT_DIR DOUBLE_THREE_SL_WEIGHTS
#
# Each smoke trains uninterrupted and stopped/resumed runs and writes a
# fingerprint. Compare two suites with:
#   training_smoke.py compare REF/<name>/fingerprint.json CAND/<name>/fingerprint.json
set -euo pipefail

if [[ $# -ne 3 ]]; then
  sed -n '2,9p' "$0"
  exit 2
fi

checkout=$1
output=$2
weights=$3
python=${PYTHON:-python}
smoke="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/training_smoke.py"

mkdir -p "$output"

# GPU learner, KL-gated warmup at its default threshold: the ladder decisions
# are part of the fingerprint.
PYTHONPATH=$checkout "$python" "$smoke" run --output-dir "$output/gpu_warmup" \
  --weights "$weights" --learning-rate 0.01 --warmup

# GPU learner hot enough to stop PPO on KL at varying epochs, with a shared
# critic and a non-zero entropy coefficient.
PYTHONPATH=$checkout "$python" "$smoke" run --output-dir "$output/gpu_hot_critic" \
  --weights "$weights" --learning-rate 0.05 --baseline value-head \
  --entropy-coef 0.01

# CPU learner on a seeded random policy with a separate critic and dropout.
PYTHONPATH=$checkout "$python" "$smoke" run --output-dir "$output/cpu_own_critic" \
  --device cpu --learning-rate 0.05 --baseline value-head-own-nn \
  --dropout 0.1 --ppo-max-epochs 6 --iterations 6
