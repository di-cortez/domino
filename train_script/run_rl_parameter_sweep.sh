#!/usr/bin/env bash
#
# RL-only hyperparameter sweep: train a dedicated self-play RL checkpoint for
# every point in a full grid search over the three main hyperparameters
# (learning_rate x gamma x games_per_iteration, 3x3x3 = 27 combinations),
# then diagnose it against the random agent. value_coef is swept separately
# (one-at-a-time, holding learning_rate/gamma/games_per_iteration at
# baseline) since it only affects the actor-critic value head. The whole
# sweep runs twice: once with the critic off, once on, with every point
# identical between the two, so the two policies can be compared directly.
#
# Baselines and the learning-rate/gamma sweep values mirror
# diagnostics/hyperparameter_sweep.py: BASELINE_LEARNING_RATE, BASELINE_GAMMA,
# BASELINE_VALUE_COEF, DEFAULT_LR_VALUES, DEFAULT_GAMMA_VALUES,
# DEFAULT_RL_GAMES_PER_ITERATION. That module only exposes
# DEFAULT_RL_GAMES_PER_ITERATION as a single baseline value, not a sweep
# tuple, so this script supplies its candidate range from the historical
# sweep table in references/explicacoes/relatorios/teste_1/plano_correcao.tex
# (Section 2): games-per-iteration in {40, 80, 160}. value_coef uses 10
# evenly spaced values from 0.1 to 1.0 (baseline 0.5 included).
#
# value_coef has no effect on direct REINFORCE (critic off) -- it only enters
# the gradient inside PolicyNetwork.backward_policy_gradient's use_value_head
# branch -- so its critic-off runs are expected to reproduce the grid's
# baseline ("default") checkpoint exactly (same seed, same everything else).
# It is still swept for both critic settings, for structural symmetry.
#
# Every sweep point is one training run (per the baseline
# training-opponent/reward-schema/etc. defaults in training/self_play.py)
# followed by one diagnostics matchup against the random agent
# (diagnostics.pairwise). The grid combination that matches every baseline
# value exactly is tagged "default" rather than spelling out all three
# values; a value_coef sweep point equal to the baseline is skipped -- it is
# already covered by that "default" run.
#
# Naming (models and diagnostics share one name per run):
#   models/rl_test/domino_rl[_critic]_default.npz                       (lr/gamma/gpi all at baseline)
#   models/rl_test/domino_rl[_critic]_lr<LR>_gamma<GAMMA>_gpi<GPI>.npz  (every other grid point)
#   models/rl_test/domino_rl[_critic]_value_coef_<VC>.npz               (value_coef axis)
#   diagnostics/results/domino_rl[_critic]_<same tag>/
#
# Reports total elapsed wall-clock time at the very end.
#
# Usage:
#   train_script/run_rl_parameter_sweep.sh [options]
#
# Run with --help for the full list of options.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SWEEP_START_EPOCH=$(date +%s)

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------

RL_ITERATIONS=10
SL_WEIGHTS_PATH="models/domino_sl_weights.npz"
DIAGNOSTIC_GAMES=100
SEED=42
MODEL_DIR="models/rl_test"
RESUME=0
DIAG_PLOTS=1
# Array backend for every sweep point: "auto" (default) matches GPU_ENABLED
# exactly (CuPy when installed, else NumPy); "cpu"/"gpu" force one backend.
DEVICE="auto"

# Concurrency: 1 (default) runs every sweep point sequentially, exactly as
# before. --jobs N > 1 runs up to N sweep points at once as background
# `python -m training.self_play` subprocesses -- safe because every sweep
# point writes to its own unique --rl-weights-path/diagnostics directory and
# only *reads* the shared SL checkpoint. RAM_LIMIT_MB/VRAM_LIMIT_MB, left
# empty here, are auto-computed from this machine's detected system RAM / GPU
# memory divided by --jobs -- always, not only when --jobs > 1, as a general
# OOM backstop (see the "Concurrency and memory limits" block below for the
# exact formula and enforcement mechanism).
JOBS=5
RAM_LIMIT_MB=""
VRAM_LIMIT_MB=""
RUN_LOG_DIR=""  # computed from RESULTS_DIR after argument parsing, below

# Final comparative-table stage: parses every sweep point's sweep_run.json +
# summary.json under diagnostics/results/ (diagnostics.rl_sweep_table) and
# writes a combined CSV/JSON/PNG table.
RESULTS_DIR="diagnostics/results/rl_test"
REPORT_OUTPUT_DIR="diagnostics/results/rl_sweep_table"
SKIP_REPORT=0

# Baselines: diagnostics/hyperparameter_sweep.py BASELINE_* constants.
BASELINE_LEARNING_RATE=0.001
BASELINE_GAMMA=1.0
BASELINE_GAMES_PER_ITERATION=40  # DEFAULT_RL_GAMES_PER_ITERATION
BASELINE_VALUE_COEF=0.5

# Grid-search values (3x3x3 = 27 combinations): DEFAULT_LR_VALUES /
# DEFAULT_GAMMA_VALUES from diagnostics/hyperparameter_sweep.py.
# games-per-iteration is not a tuple there (see header comment above) -- its
# range comes from the historical report instead.
LR_VALUES=(0.0005 0.001 0.005)
GAMMA_VALUES=(1.0 0.97 0.9)
GAMES_PER_ITERATION_VALUES=(40 80 160)

# value_coef axis (swept separately, one-at-a-time, not part of the grid):
# exactly 10 values, evenly spaced from 0.1 to 1.0, baseline (0.5) included.
VALUE_COEF_VALUES=(0.25 0.5 0.75)

usage() {
    cat <<EOF
Train a dedicated RL self-play checkpoint ($RL_ITERATIONS iterations) per
sweep point: a full grid search over learning_rate x gamma x
games_per_iteration (3x3x3 = 27 combinations), plus a separate value_coef
sweep (10 values, one-at-a-time). Runs the full sweep with the critic off,
then again with it on -- every point identical between the two. Reports
total elapsed wall-clock time at the end.

Usage: $(basename "$0") [options]

Grid search axes (cross product, 27 combinations):
  learning_rate        ${LR_VALUES[*]} (baseline: $BASELINE_LEARNING_RATE)
  gamma                 ${GAMMA_VALUES[*]} (baseline: $BASELINE_GAMMA)
  games_per_iteration   ${GAMES_PER_ITERATION_VALUES[*]} (baseline: $BASELINE_GAMES_PER_ITERATION)

Separate axis (learning_rate/gamma/games_per_iteration held at baseline):
  value_coef            ${VALUE_COEF_VALUES[*]} (baseline: $BASELINE_VALUE_COEF; no effect when critic is off)

Options:
  --rl-iterations N       RL training iterations per sweep point (default: $RL_ITERATIONS)
  --sl-weights-path PATH  Input SL weights used to initialize every RL run (default: $SL_WEIGHTS_PATH)
  --diagnostic-games N    Games in the rl-vs-random diagnostic per sweep point (default: $DIAGNOSTIC_GAMES)
  --seed N                Fix random/numpy state for both training and diagnostics (default: $SEED)
  --model-dir PATH        Output directory for RL checkpoints (default: $MODEL_DIR)
  --resume                Skip training a checkpoint that already exists on disk; still (re)run its diagnostics
  --diag-no-plots         Skip the per-run diagnostic PNG plots (CSV/JSON are always written)
  --device {auto,cpu,gpu} Array backend for every sweep point; "auto" matches GPU_ENABLED (default: $DEVICE)
  --results-dir PATH      Output directory for per-run diagnostics subdirectories (default: $RESULTS_DIR)
  --report-output-dir PATH  Where the final comparative table is written (default: $REPORT_OUTPUT_DIR)
  --skip-report           Skip the final comparative-table stage

Concurrency and memory limits:
  --jobs N                Run up to N sweep points at once as background subprocesses (default: $JOBS, sequential)
  --ram-limit-mb N        Per-subprocess physical-memory cap in MiB, enforced via a systemd-run --scope cgroup if available (default: auto from detected system RAM / --jobs)
  --vram-limit-mb N       Per-subprocess CuPy memory-pool cap in MiB, hard-enforced by CuPy (default: auto from detected GPU memory / --jobs, when device isn't cpu)

  -h, --help              Show this help message and exit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rl-iterations) RL_ITERATIONS="$2"; shift 2 ;;
        --sl-weights-path) SL_WEIGHTS_PATH="$2"; shift 2 ;;
        --diagnostic-games) DIAGNOSTIC_GAMES="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --diag-no-plots) DIAG_PLOTS=0; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --report-output-dir) REPORT_OUTPUT_DIR="$2"; shift 2 ;;
        --skip-report) SKIP_REPORT=1; shift ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --ram-limit-mb) RAM_LIMIT_MB="$2"; shift 2 ;;
        --vram-limit-mb) VRAM_LIMIT_MB="$2"; shift 2 ;;
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

if [[ -z "$RUN_LOG_DIR" ]]; then
    RUN_LOG_DIR="$RESULTS_DIR/_sweep_logs"
fi

# ------------------------------------------------------------------
# Concurrency and memory limits
# ------------------------------------------------------------------
#
# Memory limits are computed and applied by default -- always, not only
# when --jobs > 1 -- as a general OOM backstop (e.g. against a runaway leak
# in a single sequential run), sized from this machine's actual specs
# (verified: 20 CPU cores, 31779MiB system RAM, one NVIDIA RTX 3050 with
# 6144MiB VRAM) divided by --jobs, at 80% of the detected total so headroom
# stays free for the rest of the system. At the default --jobs 1 that's a
# deliberately generous single-process ceiling (~25.4GiB RAM / ~4.8GiB VRAM
# here) -- not a routine constraint (observed real usage is a few hundred
# MiB per process), just a ceiling a bug would have to badly blow through.
# An explicit --ram-limit-mb/--vram-limit-mb always wins over the
# auto-computed value.
#
# RAM is capped with `systemd-run --user --scope -p MemoryMax=... -p
# MemorySwapMax=0`, a cgroup memory.max limit on actual physical memory --
# NOT `ulimit -v`/RLIMIT_AS (virtual address space). RLIMIT_AS was tried
# first and rejected: CUDA reserves very large virtual address ranges during
# initialization regardless of physical usage, so an address-space limit
# generous enough on paper (tens of GiB) still broke `import numpy.random`
# and cupy's own CUDA context setup in practice. The cgroup limit bounds
# what the process actually resides in RAM, which is what matters here, and
# doesn't fight CUDA's address-space habits. MemorySwapMax=0 makes an
# over-limit process fail fast (SIGKILL) instead of thrashing into this
# machine's small (2GiB) swap.
USE_CGROUP_RAM_LIMIT=0
if [[ -z "$RAM_LIMIT_MB" ]]; then
    if [[ -r /proc/meminfo ]]; then
        total_ram_mb=$(awk '/MemTotal/ {print int($2 / 1024)}' /proc/meminfo)
        RAM_LIMIT_MB=$(( (total_ram_mb * 80 / 100) / JOBS ))
        echo "Auto RAM limit per job: ${RAM_LIMIT_MB}MiB (80% of ${total_ram_mb}MiB system RAM / $JOBS job(s))"
    else
        echo "Warning: /proc/meminfo not readable; no RAM limit will be applied. Pass --ram-limit-mb to set one explicitly." >&2
    fi
fi
if [[ -n "$RAM_LIMIT_MB" ]]; then
    if command -v systemd-run >/dev/null 2>&1 && \
       systemd-run --user --scope -p MemoryMax=1G --quiet -- true >/dev/null 2>&1; then
        USE_CGROUP_RAM_LIMIT=1
    else
        echo "Warning: systemd-run user scopes are unavailable here, so RAM_LIMIT_MB=${RAM_LIMIT_MB} cannot be enforced. Proceeding without a RAM cap." >&2
    fi
fi
if [[ -z "$VRAM_LIMIT_MB" && "$DEVICE" != "cpu" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        total_vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [[ -n "$total_vram_mb" ]]; then
            VRAM_LIMIT_MB=$(( (total_vram_mb * 80 / 100) / JOBS ))
            echo "Auto VRAM limit per job: ${VRAM_LIMIT_MB}MiB (80% of ${total_vram_mb}MiB total GPU memory / $JOBS job(s))"
        fi
    fi
fi

# Read by agents/nn.py: a hard cap enforced by CuPy's own memory-pool
# allocator (cupy.get_default_memory_pool().set_limit), not an OS-level
# mechanism, so it doesn't share RLIMIT_AS's incompatibility with CUDA.
export DOMINO_VRAM_LIMIT_MB="$VRAM_LIMIT_MB"

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

mkdir -p "$MODEL_DIR"
if [[ "$JOBS" -gt 1 ]]; then
    mkdir -p "$RUN_LOG_DIR"
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

# run_point TAG CRITIC LR GAMMA GPI VC
run_point() {
    local tag="$1" critic="$2" lr="$3" gamma="$4" gpi="$5" vc="$6"
    local name="domino_rl"
    if [[ "$critic" -eq 1 ]]; then
        name="${name}_critic"
    fi
    name="${name}_${tag}"

    local model_path="$MODEL_DIR/${name}.npz"
    local diag_dir="$RESULTS_DIR/${name}"
    local critic_label="off"
    if [[ "$critic" -eq 1 ]]; then
        critic_label="on"
    fi

    if [[ "$RESUME" -eq 1 && -f "$model_path" ]]; then
        section "[$name] training skipped (--resume, $model_path already exists)"
    else
        section "[$name] RL training: $RL_ITERATIONS iterations, lr=$lr gamma=$gamma games/iter=$gpi value_coef=$vc critic=$critic_label -> $model_path"
        VALUE_HEAD_FLAG=()
        if [[ "$critic" -eq 1 ]]; then
            VALUE_HEAD_FLAG=(--value-head)
        fi
        RUN_PREFIX=()
        if [[ "$USE_CGROUP_RAM_LIMIT" -eq 1 ]]; then
            RUN_PREFIX=(systemd-run --user --scope -p "MemoryMax=${RAM_LIMIT_MB}M" -p "MemorySwapMax=0" --quiet --)
        fi
        "${RUN_PREFIX[@]}" "$PYTHON_BIN" -u -m training.self_play \
            --iterations "$RL_ITERATIONS" \
            --games-per-iteration "$gpi" \
            --learning-rate "$lr" \
            --gamma "$gamma" \
            --value-coef "$vc" \
            --sl-weights-path "$SL_WEIGHTS_PATH" \
            --rl-weights-path "$model_path" \
            --seed "$SEED" \
            --device "$DEVICE" \
            "${VALUE_HEAD_FLAG[@]}"
    fi

    section "[$name] diagnostics: rl vs random ($DIAGNOSTIC_GAMES games) -> $diag_dir/"
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
# Concurrent job pool (--jobs > 1 only)
# ------------------------------------------------------------------
#
# Batch-based, not a rolling pool: launch up to $JOBS points, wait for that
# whole batch, then launch the next. Simpler and more portable across bash
# versions than a rolling pool (which needs `wait -n`), at the minor cost of
# a batch waiting on its slowest point before the next one starts.
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

# launch_point TAG CRITIC LR GAMMA GPI VC
launch_point() {
    if [[ "$JOBS" -le 1 ]]; then
        run_point "$1" "$2" "$3" "$4" "$5" "$6"
        return
    fi

    local tag="$1" critic="$2"
    local name="domino_rl"
    if [[ "$critic" -eq 1 ]]; then
        name="${name}_critic"
    fi
    name="${name}_${tag}"

    if [[ "${#JOB_PIDS[@]}" -ge "$JOBS" ]]; then
        wait_for_batch
    fi

    local log_file="$RUN_LOG_DIR/${name}.log"
    echo "[$name] started in background (log: $log_file)"
    ( run_point "$1" "$2" "$3" "$4" "$5" "$6" ) >"$log_file" 2>&1 &
    JOB_PIDS+=("$!")
    JOB_NAMES+=("$name")
}

# ------------------------------------------------------------------
# Sweep plan (echoed up front so a run can be sanity-checked before it burns
# hours of RL training)
# ------------------------------------------------------------------

grid_point_count=$((${#LR_VALUES[@]} * ${#GAMMA_VALUES[@]} * ${#GAMES_PER_ITERATION_VALUES[@]}))
total_points=0
for critic in 0 1; do
    total_points=$((total_points + grid_point_count))
    for vc in "${VALUE_COEF_VALUES[@]}"; do
        [[ "$vc" == "$BASELINE_VALUE_COEF" ]] || total_points=$((total_points + 1))
    done
done

section "RL parameter sweep: $total_points training+diagnostics runs ($RL_ITERATIONS iterations each: $grid_point_count-point grid x 2 critic settings, plus the value_coef axis)"

# ------------------------------------------------------------------
# Sweep
# ------------------------------------------------------------------

for CRITIC in 0 1; do
    # Full grid search: every combination of learning_rate x gamma x
    # games_per_iteration, run for both critic settings.
    for LR in "${LR_VALUES[@]}"; do
        for GAMMA in "${GAMMA_VALUES[@]}"; do
            for GPI in "${GAMES_PER_ITERATION_VALUES[@]}"; do
                if [[ "$LR" == "$BASELINE_LEARNING_RATE" && "$GAMMA" == "$BASELINE_GAMMA" && "$GPI" == "$BASELINE_GAMES_PER_ITERATION" ]]; then
                    TAG="default"
                else
                    TAG="lr${LR}_gamma${GAMMA}_gpi${GPI}"
                fi
                launch_point "$TAG" "$CRITIC" "$LR" "$GAMMA" "$GPI" "$BASELINE_VALUE_COEF"
            done
        done
    done

    # value_coef axis: separate from the grid, learning_rate/gamma/
    # games_per_iteration held at baseline. A value equal to the baseline
    # duplicates the grid's "default" point, so it's skipped.
    for VC in "${VALUE_COEF_VALUES[@]}"; do
        if [[ "$VC" == "$BASELINE_VALUE_COEF" ]]; then
            continue
        fi
        launch_point "value_coef_${VC}" "$CRITIC" "$BASELINE_LEARNING_RATE" "$BASELINE_GAMMA" "$BASELINE_GAMES_PER_ITERATION" "$VC"
    done
done

# Reap any still-running jobs from the final (possibly partial) batch --
# a no-op sequentially (--jobs 1), since JOB_PIDS never gets populated then.
wait_for_batch

if [[ "${#FAILED_NAMES[@]}" -gt 0 ]]; then
    echo "WARNING: ${#FAILED_NAMES[@]} sweep point(s) failed: ${FAILED_NAMES[*]}" >&2
fi

if [[ "$SKIP_REPORT" -eq 1 ]]; then
    section "Comparative table (skipped)"
else
    section "Comparative table: parsing sweep_run.json + summary.json for every run -> $REPORT_OUTPUT_DIR/"
    "$PYTHON_BIN" -u -m diagnostics.rl_sweep_table \
        --results-dir "$RESULTS_DIR" \
        --output-dir "$REPORT_OUTPUT_DIR"
fi

section "RL parameter sweep complete: $total_points runs -> $MODEL_DIR/, diagnostics/results/domino_rl*/"

SWEEP_END_EPOCH=$(date +%s)
ELAPSED_SECONDS=$((SWEEP_END_EPOCH - SWEEP_START_EPOCH))
echo "Total elapsed execution time: $(format_duration "$ELAPSED_SECONDS") (${ELAPSED_SECONDS}s)"

if [[ "${#FAILED_NAMES[@]}" -gt 0 ]]; then
    exit 1
fi
