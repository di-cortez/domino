#!/usr/bin/env bash
#
# Batch-training driver: run the full domino pipeline in order:
#   1. generate the supervised dataset with README/module defaults;
#   2. train the supervised policy with README/module defaults;
#   3. refine an RL policy with the wrapper's historical 500,000-game
#      experiment profile, with RL
#      hyperparameters overridable from the command line so the same script
#      can drive repeated batch runs that only vary the RL stage;
#   4. compare all four supported agents with the random baseline over the
#      wrapper's historical 50,000-game profile, writing results to a subdirectory of
#      `diagnostics/results/` named after the RL weights file this run
#      produced (or reused), so repeated batch runs that vary RL
#      hyperparameters keep separate diagnostics output instead of
#      overwriting a shared `all_pairs/` directory.
#
# Stage 1 keeps the dataset-generator defaults. Stage 2 forwards supervised
# scheduler, backend, batch, seed, and memory controls while retaining the
# training module's dataset/epoch/output defaults:
#
#   python -m training.dataset_generator
#   python -m training.training_loop
#
# Stage 3 wraps `python -m training.self_play`, which does accept CLI flags
# (exact games, fixed GPI, learning rate, reward schema, gamma,
# value-head/critic toggle, ...). Stage 4 wraps `python -m diagnostics.evaluate`,
# passing the RL/SL weights this run used so the report evaluates the correct
# checkpoints rather than falling back to `diagnostics.pairwise`'s hardcoded
# default paths. This script only chains the four stages and forwards flags
# through; see `training/self_play.py add_optional_rl_arguments` and
# `diagnostics/evaluate.py` for the authoritative flag definitions.
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
# Defaults
# ------------------------------------------------------------------

# Historical experiment profile. The canonical `big` pipeline is separate and
# trains 2,000,000 cumulative RL games.
BASE_RL_TOTAL_TRAINING_GAMES=100000
BIG_SCALE_FACTOR=5
RL_TOTAL_TRAINING_GAMES=$((BASE_RL_TOTAL_TRAINING_GAMES * BIG_SCALE_FACTOR))
RL_ITERATIONS=""

RL_WEIGHTS_FILE="models/domino_rl_weights.npz"
RL_SL_WEIGHTS_PATH="models/domino_sl_weights.npz"
RL_GPI=2000
RL_TRAINING_OPPONENT="self_play"
RL_LEARNING_RATE=0.001
RL_ENTROPY_COEF=0.01
RL_LOG_INTERVAL=10
RL_CHECKPOINT_INTERVAL=50
RL_POOL_REFRESH_GAMES=400
RL_MAX_POOL_SIZE=50
RL_VALUE_HEAD=0
RL_VALUE_COEF=0.5
RL_GAMMA=1.0
RL_REWARD_SCHEMA="default"

# Convergence monitoring uses trailing averages because a point-in-time value
# loss or win rate is dominated by batch noise. Gradient clipping and optional
# advantage normalization stabilize comparisons, while a seed makes
# side-by-side hyperparameter runs reproducible.
RL_CLIP_GRAD_NORM=5.0
RL_NORMALIZE_ADVANTAGES="auto"
RL_MOVING_AVERAGE_WINDOW=10
RL_SEED=""

# Array backend: "auto" (default) matches GPU_ENABLED exactly (CuPy when
# installed, else NumPy) -- unchanged from prior behavior. "cpu"/"gpu" force
# one backend regardless of what's installed/enabled globally.
RL_DEVICE="auto"
RL_WORKERS="auto"
RL_AUTOTUNE_FRACTION=0.01
RL_AUTOTUNE_MIN_GAIN=0.10
RL_MEMORY_RESERVE_MB=512
RL_ESTIMATED_WORKER_MB=256
RL_MAX_WORKER_RSS_MB=1024
RL_PPO=1

# SL convergence, device, memory, and retained-batch controls. Plateau decay is
# enabled by default and has an independent counter from optional early
# stopping.
SL_EARLY_STOPPING_PATIENCE=""
SL_LR_DECAY_FACTOR=0.5
SL_LR_DECAY_PATIENCE=5
SL_NO_LR_DECAY=0
SL_TRAINING_PLATEAU_STOP=1
SL_TRAINING_PLATEAU_WINDOW=25
SL_TRAINING_PLATEAU_PATIENCE=4
SL_TRAINING_PLATEAU_MIN_EPOCHS=100
SL_TRAINING_PLATEAU_MIN_RELATIVE_IMPROVEMENT=0.001
SL_WEIGHT_DECAY=""
SL_DEVICE="auto"
SL_BATCH_SIZE=""
SL_BATCH_AUTOTUNE=1
SL_MEMORY_RESERVE_MB=512
SL_GPU_MEMORY_RESERVE_MB=512
SL_SEED=""

# Historical diagnostics profile. Canonical `big` and `huge` use 1,000,000
# games per matchup.
BASE_DIAGNOSTIC_GAMES=10000
DIAG_GAMES=$((BASE_DIAGNOSTIC_GAMES * BIG_SCALE_FACTOR))
DIAG_SEED=""
DIAG_PAIR_PLOTS=1
DIAG_OUTPUT_DIR=""

SKIP_DATASET=0
SKIP_SL=0
SKIP_RL=0
SKIP_DIAGNOSTICS=0

usage() {
    cat <<EOF
Run the domino training pipeline: dataset generation -> supervised training
(both with README/module defaults) -> the wrapper's historical self-play RL
profile ($BASE_RL_TOTAL_TRAINING_GAMES x ${BIG_SCALE_FACTOR} =
$RL_TOTAL_TRAINING_GAMES real games by default) -> four agent-vs-random diagnostics
($BASE_DIAGNOSTIC_GAMES x ${BIG_SCALE_FACTOR} = $DIAG_GAMES games per matchup by default),
written to diagnostics/results/<rl-weights-basename>/.

For canonical levels, shared seed-addressed supervised assets, complete resume,
periodic monitoring, and forever mode, use: python -m training.pipeline --help

Usage: $(basename "$0") [options]

Dataset generation runs with no extra flags (see training/README.md):
dataset -> dataset/supervised_dataset.jsonl (30,000 games). Supervised
training runs with no extra flags by default too (-> models/domino_sl_weights.npz,
up to 2,000 epochs with automatic training-loss plateau stopping), unless one
of the SL convergence flags below is passed.

Self-play reinforcement learning ($RL_TOTAL_TRAINING_GAMES exact real games by
default; fixed GPI and discarded worker tuning; all forwarded to
training.self_play):
  --rl-weights-file PATH       Output RL weights path (default: $RL_WEIGHTS_FILE)
  --rl-sl-weights-path PATH    Input SL weights used to initialize a fresh RL run (default: $RL_SL_WEIGHTS_PATH)
  --rl-total-training-games N  Exact real-game budget (default: $RL_TOTAL_TRAINING_GAMES)
  --rl-iterations N            Legacy fixed iteration budget; implies iterations x GPI games
  --rl-gpi N                   Fixed games per iteration (default: $RL_GPI)
  --rl-training-opponent NAME  "self_play" or "heuristic" (default: $RL_TRAINING_OPPONENT)
  --rl-learning-rate F         Learning rate (default: $RL_LEARNING_RATE)
  --rl-entropy-coef F          Entropy bonus coefficient (default: $RL_ENTROPY_COEF)
  --rl-log-interval N          Iterations between log lines (default: $RL_LOG_INTERVAL)
  --rl-checkpoint-interval N   Iterations between checkpoints (default: $RL_CHECKPOINT_INTERVAL)
  --rl-pool-refresh-games N    Training games between self-play pool snapshots (default: $RL_POOL_REFRESH_GAMES)
  --rl-max-pool-size N         Max frozen snapshots kept in the pool (default: $RL_MAX_POOL_SIZE)
  --rl-value-head              Turn the legacy critic ON; implies --rl-no-ppo
  --rl-value-coef F            Value-loss coefficient, only used when --rl-value-head is set (default: $RL_VALUE_COEF)
  --rl-gamma F                 Terminal-reward discount per remaining real decision, 1.0 = no discount (default: $RL_GAMMA)
  --rl-reward-schema NAME      "default", "sparse", or "shaped" reward preset (default: $RL_REWARD_SCHEMA)
  --rl-workers N|auto          CPU-only rollout workers with isolated tuning (default: $RL_WORKERS, maximum 20)
  --rl-autotune-fraction F     Discarded real-budget fraction per worker candidate (default: $RL_AUTOTUNE_FRACTION)
  --rl-autotune-min-gain F     Minimum marginal worker-throughput gain (default: $RL_AUTOTUNE_MIN_GAIN)
  --rl-memory-reserve-mb N     Host RAM kept free during rollouts (default: $RL_MEMORY_RESERVE_MB)
  --rl-estimated-worker-mb N   Preflight RAM estimate per worker (default: $RL_ESTIMATED_WORKER_MB)
  --rl-max-worker-rss-mb N     Runtime RSS ceiling for one worker (default: $RL_MAX_WORKER_RSS_MB)

RL convergence monitoring:
  --rl-clip-grad-norm F         Gradient-norm clipping threshold (default: $RL_CLIP_GRAD_NORM)
  --rl-ppo                      Use masked PPO with minibatches (default)
  --rl-no-ppo                   Use historical one-update REINFORCE for regression
  --rl-normalize-advantages     Normalize once over the complete decision buffer (PPO default)
  --rl-no-normalize-advantages  Explicitly disable normalization
  --rl-moving-average-window N  Trailing-iteration window for value-loss/win-rate moving averages in the log (default: $RL_MOVING_AVERAGE_WINDOW)
  --rl-seed N                   Fix random/numpy state, for reproducible comparisons between configurations
  --rl-device {auto,cpu,gpu}    Array backend; "auto" matches GPU_ENABLED (default: $RL_DEVICE)

SL training controls:
  --sl-early-stopping-patience N  Validation checks (every 10 epochs) without improvement before stopping
  --sl-lr-decay-factor F          LR multiplier after a validation plateau (default: $SL_LR_DECAY_FACTOR)
  --sl-lr-decay-patience N        Failed validation checks before LR decay (default: $SL_LR_DECAY_PATIENCE)
  --sl-no-lr-decay                Disable the default supervised LR schedule
  --sl-no-training-plateau-stop   Disable automatic training-loss saturation stopping
  --sl-training-plateau-window N  Epochs per non-overlapping loss block (default: $SL_TRAINING_PLATEAU_WINDOW)
  --sl-training-plateau-patience N  Consecutive saturated blocks before stopping (default: $SL_TRAINING_PLATEAU_PATIENCE)
  --sl-training-plateau-min-epochs N  Minimum total epochs before stopping (default: $SL_TRAINING_PLATEAU_MIN_EPOCHS)
  --sl-training-plateau-min-relative-improvement F  Improvement threshold (default: $SL_TRAINING_PLATEAU_MIN_RELATIVE_IMPROVEMENT)
  --sl-weight-decay F              L2 penalty on the weight matrices
  --sl-device {auto,cpu,gpu}       Supervised array backend (default: $SL_DEVICE)
  --sl-batch-size N                Fixed mini-batch size; disables autotuning
  --sl-no-batch-autotune           Use the device default mini-batch size
  --sl-memory-reserve-mb N         Host RAM reserve (default: $SL_MEMORY_RESERVE_MB)
  --sl-gpu-memory-reserve-mb N     GPU VRAM reserve (default: $SL_GPU_MEMORY_RESERVE_MB)
  --sl-seed N                      Fix supervised initialization and shuffling

Agent-vs-random diagnostics (forwarded to diagnostics.evaluate):
  --diag-games N                Games per evaluated matchup (default: $DIAG_GAMES)
  --diag-seed N                 Fix the RNG seed for the diagnostics games (default: unset)
  --diag-no-pair-plots          Skip per-matchup PNG plots (the aggregate PNG and PDF are still generated)
  --diag-output-dir PATH        Override the diagnostics output directory (default: diagnostics/results/<rl-weights-basename>/)

Stage control:
  --skip-dataset                Skip dataset generation (reuse an existing dataset file)
  --skip-sl                     Skip supervised training (reuse an existing SL weights file)
  --skip-rl                     Skip self-play reinforcement learning
  --skip-diagnostics            Skip the agent-vs-random diagnostics stage

  -h, --help                   Show this help message and exit

Examples:
  # Full historical wrapper profile: dataset + SL + RL + diagnostics
  train_script/run_training_pipeline.sh

  # Batch run: only vary RL hyperparameters, reuse existing dataset/SL weights;
  # diagnostics land in diagnostics/results/domino_rl_weights_lr0005_gamma097_shaped/
  train_script/run_training_pipeline.sh --skip-dataset --skip-sl \\
      --rl-learning-rate 0.0005 --rl-gamma 0.97 --rl-reward-schema shaped \\
      --rl-weights-file models/domino_rl_weights_lr0005_gamma097_shaped.npz

  # Same, with the critic (value head) turned on
  train_script/run_training_pipeline.sh --skip-dataset --skip-sl \\
      --rl-value-head --rl-weights-file models/domino_rl_weights_critic.npz
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rl-weights-file) RL_WEIGHTS_FILE="$2"; shift 2 ;;
        --rl-sl-weights-path) RL_SL_WEIGHTS_PATH="$2"; shift 2 ;;
        --rl-total-training-games) RL_TOTAL_TRAINING_GAMES="$2"; shift 2 ;;
        --rl-iterations) RL_ITERATIONS="$2"; shift 2 ;;
        --rl-gpi) RL_GPI="$2"; shift 2 ;;
        --rl-training-opponent) RL_TRAINING_OPPONENT="$2"; shift 2 ;;
        --rl-learning-rate) RL_LEARNING_RATE="$2"; shift 2 ;;
        --rl-entropy-coef) RL_ENTROPY_COEF="$2"; shift 2 ;;
        --rl-log-interval) RL_LOG_INTERVAL="$2"; shift 2 ;;
        --rl-checkpoint-interval) RL_CHECKPOINT_INTERVAL="$2"; shift 2 ;;
        --rl-pool-refresh-games) RL_POOL_REFRESH_GAMES="$2"; shift 2 ;;
        --rl-max-pool-size) RL_MAX_POOL_SIZE="$2"; shift 2 ;;
        --rl-value-head) RL_VALUE_HEAD=1; RL_PPO=0; shift ;;
        --rl-value-coef) RL_VALUE_COEF="$2"; shift 2 ;;
        --rl-gamma) RL_GAMMA="$2"; shift 2 ;;
        --rl-reward-schema) RL_REWARD_SCHEMA="$2"; shift 2 ;;
        --rl-clip-grad-norm) RL_CLIP_GRAD_NORM="$2"; shift 2 ;;
        --rl-normalize-advantages) RL_NORMALIZE_ADVANTAGES=1; shift ;;
        --rl-no-normalize-advantages) RL_NORMALIZE_ADVANTAGES=0; shift ;;
        --rl-ppo) RL_PPO=1; shift ;;
        --rl-no-ppo) RL_PPO=0; shift ;;
        --rl-moving-average-window) RL_MOVING_AVERAGE_WINDOW="$2"; shift 2 ;;
        --rl-seed) RL_SEED="$2"; shift 2 ;;
        --rl-device) RL_DEVICE="$2"; shift 2 ;;
        --rl-workers) RL_WORKERS="$2"; shift 2 ;;
        --rl-autotune-fraction) RL_AUTOTUNE_FRACTION="$2"; shift 2 ;;
        --rl-autotune-min-gain) RL_AUTOTUNE_MIN_GAIN="$2"; shift 2 ;;
        --rl-memory-reserve-mb) RL_MEMORY_RESERVE_MB="$2"; shift 2 ;;
        --rl-estimated-worker-mb) RL_ESTIMATED_WORKER_MB="$2"; shift 2 ;;
        --rl-max-worker-rss-mb) RL_MAX_WORKER_RSS_MB="$2"; shift 2 ;;
        --sl-early-stopping-patience) SL_EARLY_STOPPING_PATIENCE="$2"; shift 2 ;;
        --sl-lr-decay-factor) SL_LR_DECAY_FACTOR="$2"; shift 2 ;;
        --sl-lr-decay-patience) SL_LR_DECAY_PATIENCE="$2"; shift 2 ;;
        --sl-no-lr-decay) SL_NO_LR_DECAY=1; shift ;;
        --sl-no-training-plateau-stop) SL_TRAINING_PLATEAU_STOP=0; shift ;;
        --sl-training-plateau-window) SL_TRAINING_PLATEAU_WINDOW="$2"; shift 2 ;;
        --sl-training-plateau-patience) SL_TRAINING_PLATEAU_PATIENCE="$2"; shift 2 ;;
        --sl-training-plateau-min-epochs) SL_TRAINING_PLATEAU_MIN_EPOCHS="$2"; shift 2 ;;
        --sl-training-plateau-min-relative-improvement) SL_TRAINING_PLATEAU_MIN_RELATIVE_IMPROVEMENT="$2"; shift 2 ;;
        --sl-weight-decay) SL_WEIGHT_DECAY="$2"; shift 2 ;;
        --sl-device) SL_DEVICE="$2"; shift 2 ;;
        --sl-batch-size) SL_BATCH_SIZE="$2"; shift 2 ;;
        --sl-no-batch-autotune) SL_BATCH_AUTOTUNE=0; shift ;;
        --sl-memory-reserve-mb) SL_MEMORY_RESERVE_MB="$2"; shift 2 ;;
        --sl-gpu-memory-reserve-mb) SL_GPU_MEMORY_RESERVE_MB="$2"; shift 2 ;;
        --sl-seed) SL_SEED="$2"; shift 2 ;;
        --diag-games) DIAG_GAMES="$2"; shift 2 ;;
        --diag-seed) DIAG_SEED="$2"; shift 2 ;;
        --diag-no-pair-plots) DIAG_PAIR_PLOTS=0; shift ;;
        --diag-output-dir) DIAG_OUTPUT_DIR="$2"; shift 2 ;;
        --skip-dataset) SKIP_DATASET=1; shift ;;
        --skip-sl) SKIP_SL=1; shift ;;
        --skip-rl) SKIP_RL=1; shift ;;
        --skip-diagnostics) SKIP_DIAGNOSTICS=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

# Prefer the repository interpreter without depending on a user's shell or
# machine-specific virtual-environment location.
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
    echo "Using virtual environment at .venv"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Neither 'python' nor 'python3' was found on PATH." >&2
    exit 1
fi

if [[ -z "$DIAG_OUTPUT_DIR" ]]; then
    RL_WEIGHTS_BASENAME="$(basename "$RL_WEIGHTS_FILE")"
    RL_WEIGHTS_BASENAME="${RL_WEIGHTS_BASENAME%.npz}"
    DIAG_OUTPUT_DIR="diagnostics/results/$RL_WEIGHTS_BASENAME"
fi

"$PYTHON_BIN" - "$RL_DEVICE" "$SL_DEVICE" <<'PY'
import sys

from utils.runtime_status import pipeline_compute_report

print(pipeline_compute_report(sys.argv[1], sys.argv[2]))
PY

section() {
    echo
    echo "==================================================================="
    echo "$1"
    echo "==================================================================="
}

if [[ "$SKIP_DATASET" -eq 1 ]]; then
    section "Step 1/4: dataset generation (skipped)"
else
    section "Step 1/4: generating supervised dataset (README defaults -> dataset/supervised_dataset.jsonl)"
    "$PYTHON_BIN" -u -m training.dataset_generator
fi

if [[ "$SKIP_SL" -eq 1 ]]; then
    section "Step 2/4: supervised training (skipped)"
else
    SL_EXTRA_ARGS=(
        --lr-decay-patience "$SL_LR_DECAY_PATIENCE"
        --sl-device "$SL_DEVICE"
        --sl-memory-reserve-mb "$SL_MEMORY_RESERVE_MB"
        --sl-gpu-memory-reserve-mb "$SL_GPU_MEMORY_RESERVE_MB"
        --sl-training-plateau-window "$SL_TRAINING_PLATEAU_WINDOW"
        --sl-training-plateau-patience "$SL_TRAINING_PLATEAU_PATIENCE"
        --sl-training-plateau-min-epochs "$SL_TRAINING_PLATEAU_MIN_EPOCHS"
        --sl-training-plateau-min-relative-improvement "$SL_TRAINING_PLATEAU_MIN_RELATIVE_IMPROVEMENT"
    )
    if [[ -n "$SL_EARLY_STOPPING_PATIENCE" ]]; then
        SL_EXTRA_ARGS+=(--early-stopping "$SL_EARLY_STOPPING_PATIENCE")
    fi
    if [[ "$SL_NO_LR_DECAY" -eq 1 ]]; then
        SL_EXTRA_ARGS+=(--no-lr-decay)
    else
        SL_EXTRA_ARGS+=(--lr-decay "$SL_LR_DECAY_FACTOR")
    fi
    if [[ "$SL_TRAINING_PLATEAU_STOP" -eq 0 ]]; then
        SL_EXTRA_ARGS+=(--sl-no-training-plateau-stop)
    fi
    if [[ -n "$SL_WEIGHT_DECAY" ]]; then
        SL_EXTRA_ARGS+=(--weight-decay "$SL_WEIGHT_DECAY")
    fi
    if [[ -n "$SL_BATCH_SIZE" ]]; then
        SL_EXTRA_ARGS+=(--sl-batch-size "$SL_BATCH_SIZE")
    elif [[ "$SL_BATCH_AUTOTUNE" -eq 0 ]]; then
        SL_EXTRA_ARGS+=(--sl-no-batch-autotune)
    fi
    if [[ -n "$SL_SEED" ]]; then
        SL_EXTRA_ARGS+=(--sl-seed "$SL_SEED")
    fi

    section "Step 2/4: training supervised policy (${SL_EXTRA_ARGS[*]} -> models/domino_sl_weights.npz)"
    "$PYTHON_BIN" -u -m training.training_loop "${SL_EXTRA_ARGS[@]}"
fi

if [[ "$SKIP_RL" -eq 1 ]]; then
    section "Step 3/4: self-play reinforcement learning (skipped)"
else
    section "Step 3/4: historical RL profile ($RL_TOTAL_TRAINING_GAMES exact real games -> $RL_WEIGHTS_FILE)"
    VALUE_HEAD_FLAG=()
    if [[ "$RL_VALUE_HEAD" -eq 1 ]]; then
        VALUE_HEAD_FLAG=(--value-head)
    fi
    NORMALIZE_ARGS=()
    if [[ "$RL_NORMALIZE_ADVANTAGES" == "1" ]]; then
        NORMALIZE_ARGS=(--normalize-advantages)
    elif [[ "$RL_NORMALIZE_ADVANTAGES" == "0" ]]; then
        NORMALIZE_ARGS=(--no-normalize-advantages)
    fi
    BUDGET_ARGS=(--total-training-games "$RL_TOTAL_TRAINING_GAMES" --gpi "$RL_GPI")
    if [[ -n "$RL_ITERATIONS" ]]; then
        BUDGET_ARGS=(--iterations "$RL_ITERATIONS" --gpi "$RL_GPI")
    fi
    PPO_FLAG="--ppo"
    if [[ "$RL_PPO" -eq 0 ]]; then
        PPO_FLAG="--no-ppo"
    fi
    RL_SEED_ARGS=()
    if [[ -n "$RL_SEED" ]]; then
        RL_SEED_ARGS+=(--seed "$RL_SEED")
    fi
    "$PYTHON_BIN" -u -m training.self_play \
        "${BUDGET_ARGS[@]}" \
        --training-opponent "$RL_TRAINING_OPPONENT" \
        --learning-rate "$RL_LEARNING_RATE" \
        --entropy-coef "$RL_ENTROPY_COEF" \
        --log-interval "$RL_LOG_INTERVAL" \
        --checkpoint-interval "$RL_CHECKPOINT_INTERVAL" \
        --pool-refresh-games "$RL_POOL_REFRESH_GAMES" \
        --max-pool-size "$RL_MAX_POOL_SIZE" \
        --sl-weights-path "$RL_SL_WEIGHTS_PATH" \
        --rl-weights-path "$RL_WEIGHTS_FILE" \
        --adaptive-tuning-path "${RL_WEIGHTS_FILE%.npz}_adaptive_tuning.json" \
        --fresh-from-sl \
        --value-coef "$RL_VALUE_COEF" \
        --gamma "$RL_GAMMA" \
        --reward-schema "$RL_REWARD_SCHEMA" \
        --clip-grad-norm "$RL_CLIP_GRAD_NORM" \
        --moving-average-window "$RL_MOVING_AVERAGE_WINDOW" \
        --device "$RL_DEVICE" \
        --rl-workers "$RL_WORKERS" \
        --rl-autotune-fraction "$RL_AUTOTUNE_FRACTION" \
        --rl-autotune-min-gain "$RL_AUTOTUNE_MIN_GAIN" \
        --rl-memory-reserve-mb "$RL_MEMORY_RESERVE_MB" \
        --rl-estimated-worker-mb "$RL_ESTIMATED_WORKER_MB" \
        --rl-max-worker-rss-mb "$RL_MAX_WORKER_RSS_MB" \
        "$PPO_FLAG" \
        "${NORMALIZE_ARGS[@]}" \
        "${RL_SEED_ARGS[@]}" \
        "${VALUE_HEAD_FLAG[@]}"
fi

if [[ "$SKIP_DIAGNOSTICS" -eq 1 ]]; then
    section "Step 4/4: agent-vs-random diagnostics (skipped)"
else
    section "Step 4/4: agent-vs-random diagnostics ($DIAG_GAMES games/matchup -> $DIAG_OUTPUT_DIR/)"
    DIAG_EXTRA_ARGS=()
    if [[ -n "$DIAG_SEED" ]]; then
        DIAG_EXTRA_ARGS+=(--seed "$DIAG_SEED")
    fi
    if [[ "$DIAG_PAIR_PLOTS" -eq 0 ]]; then
        DIAG_EXTRA_ARGS+=(--no-pair-plots)
    fi
    "$PYTHON_BIN" -u -m diagnostics.evaluate \
        --games "$DIAG_GAMES" \
        --output "$DIAG_OUTPUT_DIR" \
        --rl-weights "$RL_WEIGHTS_FILE" \
        --neural-weights "$RL_SL_WEIGHTS_PATH" \
        "${DIAG_EXTRA_ARGS[@]}"
fi

section "Training pipeline complete"
