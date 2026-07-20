#!/usr/bin/env bash
#
# Re-run the rl-vs-random diagnostics for every RL checkpoint already present
# in models/rl_test -- no training. Companion to
# train_script/run_rl_parameter_sweep.sh for the case where the sweep was
# halted early: every successfully written checkpoint gets a fresh
# diagnostics matchup (default: 10000 games, overwriting the old, smaller
# per-run results directories under diagnostics/results/rl_test), followed by
# the same comparative-table stage (diagnostics.rl_sweep_table).
#
# Each run's hyperparameters are reconstructed from the checkpoint filename,
# which encodes them exactly (see the naming section of
# run_rl_parameter_sweep.sh):
#   domino_rl[_critic]_default.npz
#   domino_rl[_critic]_lr<LR>_gamma<GAMMA>_gpi<GPI>[_vc<VC>].npz
# and written to <results-dir>/<name>/sweep_run.json in the same format the
# sweep script produces, so diagnostics.rl_sweep_table works unchanged.
#
# The per-run results directory of each checkpoint is deleted and recreated
# before its diagnostic so no stale file survives the overwrite. Nothing else
# under --results-dir is touched -- in particular _sweep_logs (the training
# logs of the original sweep) is preserved; this script's own per-run logs go
# to <results-dir>/_diag_logs.
#
# Parallelization mirrors run_rl_parameter_sweep.sh: a batch-based pool of up
# to --jobs background subprocesses, each writing to its own unique output
# directory. Diagnostics only *read* the checkpoints, so runs are independent.
#
# Usage:
#   train_script/run_rl_test_diagnostics.sh [options]
#
# Run with --help for the full list of options.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_START_EPOCH=$(date +%s)

# ------------------------------------------------------------------
# Defaults (matching run_rl_parameter_sweep.sh where applicable)
# ------------------------------------------------------------------

DIAGNOSTIC_GAMES=10000
SEED=42
MODEL_DIR="models/rl_test"
RESULTS_DIR="diagnostics/results/rl_test"
REPORT_OUTPUT_DIR="diagnostics/results/rl_sweep_table"
DIAG_PLOTS=1
JOBS=20
SKIP_REPORT=0

# Metadata only (recorded in each sweep_run.json; diagnostics don't use it).
# These mirror the sweep script's defaults, under which the checkpoints in
# models/rl_test were trained.
RL_ITERATIONS=2000
SL_WEIGHTS_PATH="models/domino_sl_weights.npz"

# Baseline hyperparameters (diagnostics/hyperparameter_sweep.py BASELINE_*):
# the "default" tag stands for exactly these values, and value_coef defaults
# to the baseline whenever the filename carries no _vc suffix.
BASELINE_LEARNING_RATE=0.001
BASELINE_GAMMA=1.0
BASELINE_GAMES_PER_ITERATION=40
BASELINE_VALUE_COEF=0.5

usage() {
    cat <<EOF
Re-run the rl-vs-random diagnostics ($DIAGNOSTIC_GAMES games each) for every
checkpoint (*.npz) in $MODEL_DIR, overwriting each checkpoint's per-run
results directory under $RESULTS_DIR (everything else there, including
_sweep_logs, is left alone), then rebuild the comparative table with
diagnostics.rl_sweep_table. Training is never run; hyperparameters are
reconstructed from the checkpoint filenames.

Usage: $(basename "$0") [options]

Options:
  --diagnostic-games N    Games in the rl-vs-random diagnostic per checkpoint (default: $DIAGNOSTIC_GAMES)
  --seed N                Fix random/numpy state for the diagnostics (default: $SEED)
  --model-dir PATH        Directory scanned for RL checkpoints (default: $MODEL_DIR)
  --results-dir PATH      Output directory for per-run diagnostics subdirectories (default: $RESULTS_DIR)
  --report-output-dir PATH  Where the final comparative table is written (default: $REPORT_OUTPUT_DIR)
  --diag-no-plots         Skip the per-run diagnostic PNG plots (CSV/JSON are always written)
  --skip-report           Skip the final comparative-table stage
  --jobs N                Run up to N diagnostics at once as background subprocesses (default: $JOBS)
  --rl-iterations N       Metadata: iterations the checkpoints were trained for (default: $RL_ITERATIONS)
  --sl-weights-path PATH  Metadata: SL weights the RL runs were initialized from (default: $SL_WEIGHTS_PATH)
  -h, --help              Show this help message and exit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --diagnostic-games) DIAGNOSTIC_GAMES="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --report-output-dir) REPORT_OUTPUT_DIR="$2"; shift 2 ;;
        --diag-no-plots) DIAG_PLOTS=0; shift ;;
        --skip-report) SKIP_REPORT=1; shift ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --rl-iterations) RL_ITERATIONS="$2"; shift 2 ;;
        --sl-weights-path) SL_WEIGHTS_PATH="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
    echo "--jobs must be a positive integer, got: $JOBS" >&2
    exit 1
fi

RUN_LOG_DIR="$RESULTS_DIR/_diag_logs"

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

section() {
    echo
    echo "==================================================================="
    echo "$1"
    echo "==================================================================="
}

format_duration() {
    local total_seconds="$1"
    local hours=$(( total_seconds / 3600 ))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$(( total_seconds % 60 ))
    if [[ "$hours" -gt 0 ]]; then
        printf '%dh %dm %ds' "$hours" "$minutes" "$seconds"
    elif [[ "$minutes" -gt 0 ]]; then
        printf '%dm %ds' "$minutes" "$seconds"
    else
        printf '%ds' "$seconds"
    fi
}

# write_run_metadata NAME TAG CRITIC LR GAMMA GPI VC MODEL_PATH DIAG_DIR
# (same format as run_rl_parameter_sweep.sh, so diagnostics.rl_sweep_table
# consumes it unchanged)
write_run_metadata() {
    local name="$1" tag="$2" critic="$3" lr="$4" gamma="$5" gpi="$6" vc="$7" model_path="$8" diag_dir="$9"
    local critic_bool="false"
    if [[ "$critic" -eq 1 ]]; then critic_bool="true"; fi
    cat > "$diag_dir/sweep_run.json" <<EOF
{
  "run_name": "$name",
  "varied_parameter": "$tag",
  "critic_enabled": $critic_bool,
  "learning_rate": $lr,
  "gamma": $gamma,
  "games_per_iteration": $gpi,
  "value_coef": $vc,
  "rl_iterations": $RL_ITERATIONS,
  "seed": $SEED,
  "diagnostic_games": $DIAGNOSTIC_GAMES,
  "sl_weights_path": "$SL_WEIGHTS_PATH",
  "model_path": "$model_path"
}
EOF
}

# run_point NAME TAG CRITIC LR GAMMA GPI VC MODEL_PATH
run_point() {
    local name="$1" tag="$2" critic="$3" lr="$4" gamma="$5" gpi="$6" vc="$7" model_path="$8"
    local diag_dir="$RESULTS_DIR/${name}"

    section "[$name] diagnostics: rl vs random ($DIAGNOSTIC_GAMES games) -> $diag_dir/"
    # Overwrite: drop the whole previous per-run directory so no stale file
    # (old plots, old summary) survives next to the new results.
    rm -rf "$diag_dir"
    mkdir -p "$diag_dir"
    DIAG_EXTRA_ARGS=()
    if [[ "$DIAG_PLOTS" -eq 0 ]]; then
        DIAG_EXTRA_ARGS+=(--no-plots)
    fi
    "$PYTHON_BIN" -u -m diagnostics.pairwise \
        --agent rl --opponent random \
        --weights "$model_path" \
        --games "$DIAGNOSTIC_GAMES" \
        --seed "$SEED" \
        --output "$diag_dir" \
        "${DIAG_EXTRA_ARGS[@]}"

    write_run_metadata "$name" "$tag" "$critic" "$lr" "$gamma" "$gpi" "$vc" "$model_path" "$diag_dir"
}

# ------------------------------------------------------------------
# Concurrent job pool (same batch-based scheme as run_rl_parameter_sweep.sh)
# ------------------------------------------------------------------

JOB_PIDS=()
JOB_NAMES=()
FAILED_NAMES=()

wait_for_batch() {
    local i pid name
    for i in "${!JOB_PIDS[@]}"; do
        pid="${JOB_PIDS[$i]}"
        name="${JOB_NAMES[$i]}"
        if wait "$pid"; then
            echo "[$name] finished"
        else
            echo "[$name] FAILED -- see $RUN_LOG_DIR/${name}.log" >&2
            FAILED_NAMES+=("$name")
        fi
    done
    JOB_PIDS=()
    JOB_NAMES=()
}

# launch_point NAME TAG CRITIC LR GAMMA GPI VC MODEL_PATH
launch_point() {
    if [[ "$JOBS" -le 1 ]]; then
        run_point "$@"
        return
    fi

    local name="$1"
    if [[ "${#JOB_PIDS[@]}" -ge "$JOBS" ]]; then
        wait_for_batch
    fi

    local log_file="$RUN_LOG_DIR/${name}.log"
    echo "[$name] started in background (log: $log_file)"
    ( run_point "$@" ) >"$log_file" 2>&1 &
    JOB_PIDS+=("$!")
    JOB_NAMES+=("$name")
}

# ------------------------------------------------------------------
# Discover checkpoints and reconstruct their hyperparameters
# ------------------------------------------------------------------

# The sweep script now writes each checkpoint into its own directory
# ($MODEL_DIR/<name>/<name>.npz, next to a <name>.json hyperparameter
# record); older sweeps wrote flat $MODEL_DIR/<name>.npz files. Scan both
# layouts so either generation of checkpoints gets its diagnostics.
shopt -s nullglob
MODEL_PATHS=("$MODEL_DIR"/*/*.npz "$MODEL_DIR"/*.npz)
shopt -u nullglob
if [[ "${#MODEL_PATHS[@]}" -eq 0 ]]; then
    echo "No *.npz checkpoints found in $MODEL_DIR (or its per-model subdirectories)" >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"
if [[ "$JOBS" -gt 1 ]]; then
    mkdir -p "$RUN_LOG_DIR"
fi

section "RL sweep diagnostics rerun: ${#MODEL_PATHS[@]} checkpoints in $MODEL_DIR, $DIAGNOSTIC_GAMES games each vs random (jobs: $JOBS)"

for model_path in "${MODEL_PATHS[@]}"; do
    name="$(basename "$model_path" .npz)"

    # Strip the domino_rl[_critic]_ prefix; the remainder is the tag.
    critic=0
    if [[ "$name" == domino_rl_critic_* ]]; then
        critic=1
        tag="${name#domino_rl_critic_}"
    elif [[ "$name" == domino_rl_* ]]; then
        tag="${name#domino_rl_}"
    else
        echo "[$name] SKIPPED: filename doesn't match domino_rl[_critic]_<tag>.npz" >&2
        FAILED_NAMES+=("$name")
        continue
    fi

    # Decode the tag into hyperparameter values.
    if [[ "$tag" =~ ^default(_vc([0-9.]+))?$ ]]; then
        lr="$BASELINE_LEARNING_RATE"
        gamma="$BASELINE_GAMMA"
        gpi="$BASELINE_GAMES_PER_ITERATION"
        vc="${BASH_REMATCH[2]:-$BASELINE_VALUE_COEF}"
    elif [[ "$tag" =~ ^lr([0-9.]+)_gamma([0-9.]+)_gpi([0-9]+)(_vc([0-9.]+))?$ ]]; then
        lr="${BASH_REMATCH[1]}"
        gamma="${BASH_REMATCH[2]}"
        gpi="${BASH_REMATCH[3]}"
        vc="${BASH_REMATCH[5]:-$BASELINE_VALUE_COEF}"
    else
        echo "[$name] SKIPPED: unrecognized tag '$tag'" >&2
        FAILED_NAMES+=("$name")
        continue
    fi

    launch_point "$name" "$tag" "$critic" "$lr" "$gamma" "$gpi" "$vc" "$model_path"
done

# Reap any still-running jobs from the final (possibly partial) batch.
wait_for_batch

if [[ "${#FAILED_NAMES[@]}" -gt 0 ]]; then
    echo "WARNING: ${#FAILED_NAMES[@]} checkpoint(s) failed or were skipped: ${FAILED_NAMES[*]}" >&2
fi

if [[ "$SKIP_REPORT" -eq 1 ]]; then
    section "Comparative table (skipped)"
else
    section "Comparative table: parsing sweep_run.json + summary.json for every run -> $REPORT_OUTPUT_DIR/"
    "$PYTHON_BIN" -u -m diagnostics.rl_sweep_table \
        --results-dir "$RESULTS_DIR" \
        --output-dir "$REPORT_OUTPUT_DIR"
fi

section "Diagnostics rerun complete: ${#MODEL_PATHS[@]} checkpoints -> $RESULTS_DIR/"

RUN_END_EPOCH=$(date +%s)
ELAPSED_SECONDS=$((RUN_END_EPOCH - RUN_START_EPOCH))
echo "Total elapsed execution time: $(format_duration "$ELAPSED_SECONDS") (${ELAPSED_SECONDS}s)"

if [[ "${#FAILED_NAMES[@]}" -gt 0 ]]; then
    exit 1
fi
