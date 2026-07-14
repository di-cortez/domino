#!/usr/bin/env bash
#
# Run the full domino training pipeline in order:
#   1. generate supervised examples from heuristic-vs-heuristic games;
#   2. train the supervised neural policy;
#   3. refine that policy with self-play reinforcement learning.
#
# Each stage is a thin wrapper around the matching `python -m training.*`
# module, which already accepts all of the flags below directly. This script
# only chains the three stages and forwards a consistent set of parameters
# between them (e.g. the same dataset/weights paths feed the next stage).
#
# Usage:
#   train_script/run_training_pipeline.sh [options]
#
# Run with --help for the full list of options.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ------------------------------------------------------------------
# Defaults (mirrors the defaults already declared in the Python modules)
# ------------------------------------------------------------------

# teste_3 setup: fresh dataset and a model trained from scratch under the
# P1/P2 corrections (weight decay, LR decay, gamma, advantage normalization,
# flattened choice multipliers). The _1407 files are kept untouched as the
# archived baseline for side-by-side comparison.
# Continuation, leg 2: the 30k-game leg on _teste3_2 was halted at epoch 1200
# (validation plateaued near 0.305 with the LR ground down by plateau decay).
# This leg regenerates a fresh 30k-game dataset and resumes the same weights
# on it, so the model sees new gradient signal instead of grinding out
# diminishing returns on data it has already fit.
GAMES=30000
DATASET_FILE="dataset/supervised_dataset_teste3_3.jsonl"

SL_WEIGHTS_FILE="models/domino_sl_weights_teste3.npz"
SL_CACHE_FILE="dataset/supervised_dataset_encoded_teste3_3.npz"
SL_EPOCHS=1200
SL_BATCH_SIZE=1024
SL_LEARNING_RATE=0.005
# Kept well below SL_EARLY_STOPPING_PATIENCE * 10 epochs so at least a couple
# of archived checkpoints exist even if training stops early (see teste_1).
SL_CHECKPOINT_EVERY=50
SL_CHECKPOINT_DIR="models/supervised_checkpoints_teste3_3"
SL_EARLY_STOPPING_PATIENCE=5
SL_WEIGHT_DECAY=0.0001
SL_LR_DECAY_FACTOR=0.5

RL_WEIGHTS_FILE="models/domino_rl_weights_teste3.npz"
RL_ITERATIONS=800
RL_GAMES_PER_ITERATION=80
RL_TRAINING_OPPONENT="self_play"
RL_LEARNING_RATE=0.001
RL_ENTROPY_COEF=0.01
RL_LOG_INTERVAL=10
RL_CHECKPOINT_INTERVAL=50
RL_POOL_INTERVAL=10
RL_MAX_POOL_SIZE=50
RL_EVALUATION_GAMES=200
RL_VALUE_COEF=0.5
RL_CLIP_GRAD_NORM=5.0
RL_GAMMA=0.99
RL_NORMALIZE_ADVANTAGES=1

SKIP_DATASET=0
SKIP_SL=0
SKIP_RL=0

usage() {
    cat <<EOF
Run the domino training pipeline: dataset generation -> supervised training -> self-play RL.

Usage: $(basename "$0") [options]

Dataset generation:
  --games N                    Number of heuristic-vs-heuristic games to simulate (default: $GAMES)
  --dataset-file PATH          Output JSONL dataset path (default: $DATASET_FILE)

Supervised training:
  --sl-weights-file PATH       Output SL weights path (default: $SL_WEIGHTS_FILE)
  --sl-cache-file PATH         Encoded dataset cache path (default: $SL_CACHE_FILE)
  --sl-epochs N                Training epochs (default: $SL_EPOCHS)
  --sl-batch-size N            Mini-batch size (default: $SL_BATCH_SIZE)
  --sl-learning-rate F         Learning rate (default: $SL_LEARNING_RATE)
  --sl-checkpoint-every N      Epochs between checkpoints (default: $SL_CHECKPOINT_EVERY)
  --sl-checkpoint-dir PATH     Checkpoint directory (default: $SL_CHECKPOINT_DIR)
  --sl-early-stopping-patience N  Validation checks (every 10 epochs) without
                               improvement before stopping early; 0 disables
                               (default: $SL_EARLY_STOPPING_PATIENCE)
  --sl-weight-decay F          L2 penalty on the weight matrices; 0 disables (default: $SL_WEIGHT_DECAY)
  --sl-lr-decay-factor F       LR multiplier applied on each validation check
                               without improvement; 1 disables (default: $SL_LR_DECAY_FACTOR)

Self-play reinforcement learning:
  --rl-weights-file PATH       Output RL weights path (default: $RL_WEIGHTS_FILE)
  --rl-iterations N            Training iterations (default: $RL_ITERATIONS)
  --rl-games-per-iteration N   Games played per iteration (default: $RL_GAMES_PER_ITERATION)
  --rl-training-opponent NAME  "self_play" or "heuristic" (default: $RL_TRAINING_OPPONENT)
  --rl-learning-rate F         Learning rate (default: $RL_LEARNING_RATE)
  --rl-entropy-coef F          Entropy bonus coefficient (default: $RL_ENTROPY_COEF)
  --rl-log-interval N          Iterations between log lines (default: $RL_LOG_INTERVAL)
  --rl-checkpoint-interval N   Iterations between checkpoints (default: $RL_CHECKPOINT_INTERVAL)
  --rl-pool-interval N         Iterations between self-play pool snapshots (default: $RL_POOL_INTERVAL)
  --rl-max-pool-size N         Max frozen snapshots kept in the pool (default: $RL_MAX_POOL_SIZE)
  --rl-evaluation-games N      Games per checkpoint evaluation (default: $RL_EVALUATION_GAMES)
  --rl-value-coef F            Value-loss coefficient in the actor-critic update (default: $RL_VALUE_COEF)
  --rl-clip-grad-norm F        Gradient-norm clipping threshold (default: $RL_CLIP_GRAD_NORM)
  --rl-gamma F                 Terminal-reward discount per remaining decision (default: $RL_GAMMA)
  --rl-no-normalize-advantages Disable per-batch advantage normalization

Stage control:
  --skip-dataset               Skip dataset generation (reuse an existing dataset file)
  --skip-sl                    Skip supervised training (reuse an existing SL weights file)
  --skip-rl                    Skip self-play reinforcement learning

  -h, --help                   Show this help message and exit

Examples:
  # Full pipeline with defaults
  train_script/run_training_pipeline.sh

  # Quick smoke test
  train_script/run_training_pipeline.sh --games 200 --sl-epochs 20 --rl-iterations 10

  # Re-run only self-play RL against an existing SL checkpoint
  train_script/run_training_pipeline.sh --skip-dataset --skip-sl
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --games) GAMES="$2"; shift 2 ;;
        --dataset-file) DATASET_FILE="$2"; shift 2 ;;
        --sl-weights-file) SL_WEIGHTS_FILE="$2"; shift 2 ;;
        --sl-cache-file) SL_CACHE_FILE="$2"; shift 2 ;;
        --sl-epochs) SL_EPOCHS="$2"; shift 2 ;;
        --sl-batch-size) SL_BATCH_SIZE="$2"; shift 2 ;;
        --sl-learning-rate) SL_LEARNING_RATE="$2"; shift 2 ;;
        --sl-checkpoint-every) SL_CHECKPOINT_EVERY="$2"; shift 2 ;;
        --sl-checkpoint-dir) SL_CHECKPOINT_DIR="$2"; shift 2 ;;
        --sl-early-stopping-patience) SL_EARLY_STOPPING_PATIENCE="$2"; shift 2 ;;
        --sl-weight-decay) SL_WEIGHT_DECAY="$2"; shift 2 ;;
        --sl-lr-decay-factor) SL_LR_DECAY_FACTOR="$2"; shift 2 ;;
        --rl-weights-file) RL_WEIGHTS_FILE="$2"; shift 2 ;;
        --rl-iterations) RL_ITERATIONS="$2"; shift 2 ;;
        --rl-games-per-iteration) RL_GAMES_PER_ITERATION="$2"; shift 2 ;;
        --rl-training-opponent) RL_TRAINING_OPPONENT="$2"; shift 2 ;;
        --rl-learning-rate) RL_LEARNING_RATE="$2"; shift 2 ;;
        --rl-entropy-coef) RL_ENTROPY_COEF="$2"; shift 2 ;;
        --rl-log-interval) RL_LOG_INTERVAL="$2"; shift 2 ;;
        --rl-checkpoint-interval) RL_CHECKPOINT_INTERVAL="$2"; shift 2 ;;
        --rl-pool-interval) RL_POOL_INTERVAL="$2"; shift 2 ;;
        --rl-max-pool-size) RL_MAX_POOL_SIZE="$2"; shift 2 ;;
        --rl-evaluation-games) RL_EVALUATION_GAMES="$2"; shift 2 ;;
        --rl-value-coef) RL_VALUE_COEF="$2"; shift 2 ;;
        --rl-clip-grad-norm) RL_CLIP_GRAD_NORM="$2"; shift 2 ;;
        --rl-gamma) RL_GAMMA="$2"; shift 2 ;;
        --rl-no-normalize-advantages) RL_NORMALIZE_ADVANTAGES=0; shift ;;
        --skip-dataset) SKIP_DATASET=1; shift ;;
        --skip-sl) SKIP_SL=1; shift ;;
        --skip-rl) SKIP_RL=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
    echo "Activated virtual environment at .venv"
else
    echo "No .venv found at repository root; using the interpreter already on PATH."
fi

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Neither 'python' nor 'python3' was found on PATH." >&2
    exit 1
fi

section() {
    echo
    echo "==================================================================="
    echo "$1"
    echo "==================================================================="
}

if [[ "$SKIP_DATASET" -eq 1 ]]; then
    section "Step 1/3: dataset generation (skipped)"
else
    section "Step 1/3: generating supervised dataset ($GAMES games -> $DATASET_FILE)"
    "$PYTHON_BIN" -u -m training.dataset_generator \
        --games "$GAMES" \
        --output-file "$DATASET_FILE"
fi

if [[ "$SKIP_SL" -eq 1 ]]; then
    section "Step 2/3: supervised training (skipped)"
else
    section "Step 2/3: training supervised policy ($SL_EPOCHS epochs -> $SL_WEIGHTS_FILE)"
    "$PYTHON_BIN" -u -m training.training_loop \
        --dataset-file "$DATASET_FILE" \
        --weights-file "$SL_WEIGHTS_FILE" \
        --cache-file "$SL_CACHE_FILE" \
        --epochs "$SL_EPOCHS" \
        --batch-size "$SL_BATCH_SIZE" \
        --learning-rate "$SL_LEARNING_RATE" \
        --checkpoint-every "$SL_CHECKPOINT_EVERY" \
        --checkpoint-dir "$SL_CHECKPOINT_DIR" \
        --early-stopping-patience "$SL_EARLY_STOPPING_PATIENCE" \
        --weight-decay "$SL_WEIGHT_DECAY" \
        --lr-decay-factor "$SL_LR_DECAY_FACTOR"
fi

if [[ "$SKIP_RL" -eq 1 ]]; then
    section "Step 3/3: self-play reinforcement learning (skipped)"
else
    section "Step 3/3: refining RL policy by self-play ($RL_ITERATIONS iterations -> $RL_WEIGHTS_FILE)"
    NORMALIZE_FLAG="--normalize-advantages"
    if [[ "$RL_NORMALIZE_ADVANTAGES" -eq 0 ]]; then
        NORMALIZE_FLAG="--no-normalize-advantages"
    fi
    "$PYTHON_BIN" -u -m training.self_play \
        --iterations "$RL_ITERATIONS" \
        --games-per-iteration "$RL_GAMES_PER_ITERATION" \
        --training-opponent "$RL_TRAINING_OPPONENT" \
        --learning-rate "$RL_LEARNING_RATE" \
        --entropy-coef "$RL_ENTROPY_COEF" \
        --log-interval "$RL_LOG_INTERVAL" \
        --checkpoint-interval "$RL_CHECKPOINT_INTERVAL" \
        --pool-interval "$RL_POOL_INTERVAL" \
        --max-pool-size "$RL_MAX_POOL_SIZE" \
        --evaluation-games "$RL_EVALUATION_GAMES" \
        --sl-weights-path "$SL_WEIGHTS_FILE" \
        --rl-weights-path "$RL_WEIGHTS_FILE" \
        --value-coef "$RL_VALUE_COEF" \
        --clip-grad-norm "$RL_CLIP_GRAD_NORM" \
        --gamma "$RL_GAMMA" \
        "$NORMALIZE_FLAG"
fi

section "Training pipeline complete"
