#!/usr/bin/env bash
#
# Personal batch-training driver: run the full domino pipeline in order:
#   1. generate the supervised dataset with README/module defaults;
#   2. train the supervised policy with README/module defaults;
#   3. refine an RL policy with a "BIG" self-play run (5x the default
#      iteration count used by run_pipeline.py's "big" scale), with RL
#      hyperparameters overridable from the command line so the same script
#      can drive repeated batch runs that only vary the RL stage;
#   4. run the all-pairs diagnostics matrix (heuristic/neural/rl vs random and
#      in self-play), mirroring `run_pipeline.py`'s diagnostics stage at the
#      same BIG scale, writing results to a subdirectory of
#      `diagnostics/results/` named after the RL weights file this run
#      produced (or reused), so repeated batch runs that vary RL
#      hyperparameters keep separate diagnostics output instead of
#      overwriting a shared `all_pairs/` directory.
#
# Stage 1 and 2 are intentionally NOT parameterized here: neither
# `training.dataset_generator` nor `training.training_loop` currently exposes
# a dataset-size/epoch-count/output-path CLI (see their module docstrings and
# `training/README.md`), so this script just runs them exactly as
# `README.md` documents:
#
#   python -m training.dataset_generator
#   python -m training.training_loop
#
# Stage 3 wraps `python -m training.self_play`, which does accept CLI flags
# (iterations, games-per-iteration, learning rate, reward schema, gamma,
# value-head/critic toggle, ...). Stage 4 wraps `python -m diagnostics.evaluate`,
# passing the RL/SL weights this run used so the matrix evaluates the correct
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

# Mirrors run_pipeline.py: BASE_RL_ITERATIONS=1000, "big" scale = 5x, RL
# games-per-iteration stays fixed at 40 across every scale.
BASE_RL_ITERATIONS=1000
BIG_SCALE_FACTOR=5
RL_ITERATIONS=$((BASE_RL_ITERATIONS * BIG_SCALE_FACTOR))

RL_WEIGHTS_FILE="models/domino_rl_weights.npz"
RL_SL_WEIGHTS_PATH="models/domino_sl_weights.npz"
RL_GAMES_PER_ITERATION=40
RL_TRAINING_OPPONENT="self_play"
RL_LEARNING_RATE=0.001
RL_ENTROPY_COEF=0.01
RL_LOG_INTERVAL=10
RL_CHECKPOINT_INTERVAL=50
RL_POOL_INTERVAL=10
RL_MAX_POOL_SIZE=50
RL_EVALUATION_GAMES=200
RL_VALUE_HEAD=0
RL_VALUE_COEF=0.5
RL_GAMMA=1.0
RL_REWARD_SCHEMA="default"

# Convergence-monitoring controls, validated in
# references/explicacoes/relatorios/{teste_1,teste_2,teste_3,relatorio_1407}:
# a point-in-time value loss or win rate is dominated by batch noise, so
# judge a plateau from the moving average, not the raw value; clip-grad-norm
# and normalize-advantages bound/stabilize the gradient step so that
# comparison is meaningful; seed fixes randomness for reproducible
# side-by-side comparisons between hyperparameter configurations.
RL_CLIP_GRAD_NORM=5.0
RL_NORMALIZE_ADVANTAGES=0
RL_MOVING_AVERAGE_WINDOW=10
RL_SEED=""

# Array backend: "auto" (default) matches GPU_ENABLED exactly (CuPy when
# installed, else NumPy) -- unchanged from prior behavior. "cpu"/"gpu" force
# one backend regardless of what's installed/enabled globally.
RL_DEVICE="auto"

# SL convergence controls (unset by default so a bare run matches
# README/module defaults exactly). Early stopping and LR-decay-by-plateau are
# the validated SL convergence-determination mechanisms from the same
# reports: stop/slow down on the validation curve, not on a fixed epoch
# budget.
SL_EARLY_STOPPING_PATIENCE=""
SL_LR_DECAY_FACTOR=""
SL_WEIGHT_DECAY=""

# Diagnostics stage: mirrors run_pipeline.py's diagnostics logic
# (diagnostics/evaluate.py::run_all_pairs). run_pipeline.py maps its "big"
# scale to diagnostic_mode="complete" and scales BASE_DIAGNOSTIC_GAMES=10000
# by the same scale factor as the other stages; since this script always runs
# the BIG-scale RL stage, it mirrors that mapping by default.
BASE_DIAGNOSTIC_GAMES=10000
DIAG_MODE="complete"
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
(both with README/module defaults) -> a BIG-scale self-play RL run ($BASE_RL_ITERATIONS
x ${BIG_SCALE_FACTOR} = $RL_ITERATIONS iterations by default, matching run_pipeline.py's
"big" scale) -> the all-pairs diagnostics matrix at the same BIG scale
($BASE_DIAGNOSTIC_GAMES x ${BIG_SCALE_FACTOR} = $DIAG_GAMES games per matchup by default,
mode "$DIAG_MODE"), written to diagnostics/results/<rl-weights-basename>/.

Usage: $(basename "$0") [options]

Dataset generation runs with no extra flags (see training/README.md):
dataset -> dataset/supervised_dataset.jsonl (10,000 games). Supervised
training runs with no extra flags by default too (-> models/domino_sl_weights.npz,
1,000 epochs), unless one of the SL convergence flags below is passed.

Self-play reinforcement learning (all forwarded to training.self_play):
  --rl-weights-file PATH       Output RL weights path (default: $RL_WEIGHTS_FILE)
  --rl-sl-weights-path PATH    Input SL weights used to initialize a fresh RL run (default: $RL_SL_WEIGHTS_PATH)
  --rl-iterations N            Training iterations (default: $RL_ITERATIONS, i.e. the BIG scale)
  --rl-games-per-iteration N   Games played per iteration (default: $RL_GAMES_PER_ITERATION)
  --rl-training-opponent NAME  "self_play" or "heuristic" (default: $RL_TRAINING_OPPONENT)
  --rl-learning-rate F         Learning rate (default: $RL_LEARNING_RATE)
  --rl-entropy-coef F          Entropy bonus coefficient (default: $RL_ENTROPY_COEF)
  --rl-log-interval N          Iterations between log lines (default: $RL_LOG_INTERVAL)
  --rl-checkpoint-interval N   Iterations between checkpoints (default: $RL_CHECKPOINT_INTERVAL)
  --rl-pool-interval N         Iterations between self-play pool snapshots (default: $RL_POOL_INTERVAL)
  --rl-max-pool-size N         Max frozen snapshots kept in the pool (default: $RL_MAX_POOL_SIZE)
  --rl-evaluation-games N      Games per checkpoint evaluation (default: $RL_EVALUATION_GAMES)
  --rl-value-head              Turn the critic (learned V(s) baseline) ON; off by default (direct REINFORCE)
  --rl-value-coef F            Value-loss coefficient, only used when --rl-value-head is set (default: $RL_VALUE_COEF)
  --rl-gamma F                 Terminal-reward discount per remaining real decision, 1.0 = no discount (default: $RL_GAMMA)
  --rl-reward-schema NAME      "default", "sparse", or "shaped" reward preset (default: $RL_REWARD_SCHEMA)

RL convergence monitoring (see references/explicacoes/relatorios/relatorio_1407):
  --rl-clip-grad-norm F         Gradient-norm clipping threshold (default: $RL_CLIP_GRAD_NORM)
  --rl-normalize-advantages     Standardize the policy signal per batch before the gradient step; off by default
  --rl-no-normalize-advantages  Explicitly keep advantage normalization off (default)
  --rl-moving-average-window N  Trailing-iteration window for value-loss/win-rate moving averages in the log (default: $RL_MOVING_AVERAGE_WINDOW)
  --rl-seed N                   Fix random/numpy state, for reproducible comparisons between configurations
  --rl-device {auto,cpu,gpu}    Array backend; "auto" matches GPU_ENABLED (default: $RL_DEVICE)

SL convergence monitoring (unset by default -> bare README invocation):
  --sl-early-stopping-patience N  Validation checks (every 10 epochs) without improvement before stopping
  --sl-lr-decay-factor F          LR multiplier applied on each validation check without improvement
  --sl-weight-decay F              L2 penalty on the weight matrices

All-pairs diagnostics (forwarded to diagnostics.evaluate, mirrors run_pipeline.py):
  --diag-mode NAME              "default" (10 matchups), "fast" (2), or "complete" (15 matchups) (default: $DIAG_MODE)
  --diag-games N                Games per evaluated matchup (default: $DIAG_GAMES)
  --diag-seed N                 Fix the RNG seed for the diagnostics games (default: unset)
  --diag-no-pair-plots          Skip the per-matchup PNG plots (the aggregate table image is still generated)
  --diag-output-dir PATH        Override the diagnostics output directory (default: diagnostics/results/<rl-weights-basename>/)

Stage control:
  --skip-dataset                Skip dataset generation (reuse an existing dataset file)
  --skip-sl                     Skip supervised training (reuse an existing SL weights file)
  --skip-rl                     Skip self-play reinforcement learning
  --skip-diagnostics            Skip the all-pairs diagnostics stage

  -h, --help                   Show this help message and exit

Examples:
  # Full pipeline: default dataset + SL, BIG-scale RL with defaults, then diagnostics
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
        --rl-value-head) RL_VALUE_HEAD=1; shift ;;
        --rl-value-coef) RL_VALUE_COEF="$2"; shift 2 ;;
        --rl-gamma) RL_GAMMA="$2"; shift 2 ;;
        --rl-reward-schema) RL_REWARD_SCHEMA="$2"; shift 2 ;;
        --rl-clip-grad-norm) RL_CLIP_GRAD_NORM="$2"; shift 2 ;;
        --rl-normalize-advantages) RL_NORMALIZE_ADVANTAGES=1; shift ;;
        --rl-no-normalize-advantages) RL_NORMALIZE_ADVANTAGES=0; shift ;;
        --rl-moving-average-window) RL_MOVING_AVERAGE_WINDOW="$2"; shift 2 ;;
        --rl-seed) RL_SEED="$2"; shift 2 ;;
        --rl-device) RL_DEVICE="$2"; shift 2 ;;
        --sl-early-stopping-patience) SL_EARLY_STOPPING_PATIENCE="$2"; shift 2 ;;
        --sl-lr-decay-factor) SL_LR_DECAY_FACTOR="$2"; shift 2 ;;
        --sl-weight-decay) SL_WEIGHT_DECAY="$2"; shift 2 ;;
        --diag-mode) DIAG_MODE="$2"; shift 2 ;;
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

# Prefer a repo-local .venv for portability; fall back to the pre-provisioned
# environment at amb_virtual (has cupy/pygame already installed), then to
# whatever's already on PATH.
DEFAULT_VIRTUAL_ENV="/home/diego/CCO/amb_virtual"
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
    echo "Activated virtual environment at .venv"
elif [[ -f "$DEFAULT_VIRTUAL_ENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$DEFAULT_VIRTUAL_ENV/bin/activate"
    echo "Activated virtual environment at $DEFAULT_VIRTUAL_ENV"
else
    echo "No .venv at repository root and no environment found at $DEFAULT_VIRTUAL_ENV; using the interpreter already on PATH."
fi

if command -v python >/dev/null 2>&1; then
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
    SL_EXTRA_ARGS=()
    if [[ -n "$SL_EARLY_STOPPING_PATIENCE" ]]; then
        SL_EXTRA_ARGS+=(--early-stopping "$SL_EARLY_STOPPING_PATIENCE")
    fi
    if [[ -n "$SL_LR_DECAY_FACTOR" ]]; then
        SL_EXTRA_ARGS+=(--lr-decay "$SL_LR_DECAY_FACTOR")
    fi
    if [[ -n "$SL_WEIGHT_DECAY" ]]; then
        SL_EXTRA_ARGS+=(--weight-decay "$SL_WEIGHT_DECAY")
    fi

    if [[ ${#SL_EXTRA_ARGS[@]} -eq 0 ]]; then
        section "Step 2/4: training supervised policy (README defaults -> models/domino_sl_weights.npz)"
        "$PYTHON_BIN" -u -m training.training_loop
    else
        section "Step 2/4: training supervised policy (convergence controls: ${SL_EXTRA_ARGS[*]} -> models/domino_sl_weights.npz)"
        "$PYTHON_BIN" -u -m training.training_loop "${SL_EXTRA_ARGS[@]}"
    fi
fi

if [[ "$SKIP_RL" -eq 1 ]]; then
    section "Step 3/4: self-play reinforcement learning (skipped)"
else
    section "Step 3/4: BIG-scale RL self-play ($RL_ITERATIONS iterations -> $RL_WEIGHTS_FILE)"
    VALUE_HEAD_FLAG=()
    if [[ "$RL_VALUE_HEAD" -eq 1 ]]; then
        VALUE_HEAD_FLAG=(--value-head)
    fi
    NORMALIZE_FLAG="--no-normalize-advantages"
    if [[ "$RL_NORMALIZE_ADVANTAGES" -eq 1 ]]; then
        NORMALIZE_FLAG="--normalize-advantages"
    fi
    RL_SEED_ARGS=()
    if [[ -n "$RL_SEED" ]]; then
        RL_SEED_ARGS+=(--seed "$RL_SEED")
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
        --sl-weights-path "$RL_SL_WEIGHTS_PATH" \
        --rl-weights-path "$RL_WEIGHTS_FILE" \
        --value-coef "$RL_VALUE_COEF" \
        --gamma "$RL_GAMMA" \
        --reward-schema "$RL_REWARD_SCHEMA" \
        --clip-grad-norm "$RL_CLIP_GRAD_NORM" \
        --moving-average-window "$RL_MOVING_AVERAGE_WINDOW" \
        --device "$RL_DEVICE" \
        "$NORMALIZE_FLAG" \
        "${RL_SEED_ARGS[@]}" \
        "${VALUE_HEAD_FLAG[@]}"
fi

if [[ "$SKIP_DIAGNOSTICS" -eq 1 ]]; then
    section "Step 4/4: all-pairs diagnostics (skipped)"
else
    section "Step 4/4: all-pairs diagnostics ($DIAG_MODE mode, $DIAG_GAMES games/matchup -> $DIAG_OUTPUT_DIR/)"
    DIAG_EXTRA_ARGS=()
    if [[ -n "$DIAG_SEED" ]]; then
        DIAG_EXTRA_ARGS+=(--seed "$DIAG_SEED")
    fi
    if [[ "$DIAG_PAIR_PLOTS" -eq 0 ]]; then
        DIAG_EXTRA_ARGS+=(--no-pair-plots)
    fi
    "$PYTHON_BIN" -u -m diagnostics.evaluate "$DIAG_MODE" \
        --games "$DIAG_GAMES" \
        --output "$DIAG_OUTPUT_DIR" \
        --rl-weights "$RL_WEIGHTS_FILE" \
        --neural-weights "$RL_SL_WEIGHTS_PATH" \
        "${DIAG_EXTRA_ARGS[@]}"
fi

section "Training pipeline complete"
