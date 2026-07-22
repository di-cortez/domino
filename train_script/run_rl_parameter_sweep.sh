#!/usr/bin/env bash
#
# RL-only hyperparameter sweep: train a dedicated self-play RL checkpoint for
# every point in a full grid search over the three main hyperparameters
# (learning_rate x gamma x games_per_iteration, 3x4x3 = 36 combinations),
# then diagnose it against the random agent. The full grid runs for both
# policies -- critic off (direct REINFORCE) and critic on (actor-critic) at
# the baseline value_coef -- with every base combination identical between
# the two, so the two policies can be compared directly.
#
# value_coef only affects the actor-critic value head -- it enters the
# gradient exclusively inside PolicyNetwork.backward_policy_gradient's
# use_value_head branch and has no effect on direct REINFORCE -- so it is
# swept with the critic ON only. Its sweep: VC_SAMPLE_COUNT base combinations
# are drawn at random (seeded with --seed, so reproducible) from the grid,
# and each sampled combination is trained once per value_coef value. The
# same sampled combinations are reused for every value_coef value; a
# value_coef equal to the baseline is skipped because the critic-on grid run
# of that combination already covers it.
#
# Loop structure: one pass over the full cross-product grid; per base
# combination, run it critic-off, then critic-on at baseline value_coef,
# then (if the combination was sampled for the value_coef axis) once more
# per remaining value_coef value, before moving to the next combination.
#
# Baselines and the learning-rate/gamma sweep values mirror
# diagnostics/hyperparameter_sweep.py: BASELINE_LEARNING_RATE, BASELINE_GAMMA,
# BASELINE_VALUE_COEF, DEFAULT_LR_VALUES, DEFAULT_GAMMA_VALUES,
# DEFAULT_RL_GAMES_PER_ITERATION. That module exposes only one baseline rather
# than a sweep tuple, so this script preserves the established
# games-per-iteration comparison {40, 80, 160}.
#
# Every sweep point is one training run (per the baseline
# training-opponent/reward-schema/etc. defaults in training/self_play.py)
# followed by one diagnostics matchup against the random agent
# (diagnostics.pairwise). The grid combination that matches every baseline
# value exactly is tagged "default" rather than spelling out all three
# values.
#
# Naming (models and diagnostics share one name per run):
#   models/rl_test/domino_rl[_critic]_default_iter002000.npz
#   models/rl_test/domino_rl[_critic]_lr<LR>_gamma<GAMMA>_gpi<GPI>_iter002000.npz
#   models/rl_test/domino_rl_critic_<grid tag>_vc<VC>_iter002000.npz
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

RL_ITERATIONS=2000
SL_WEIGHTS_PATH="models/domino_sl_weights.npz"
DIAGNOSTIC_GAMES=10000
SEED=42
MODEL_DIR="models/rl_test"
RESUME=0
DIAG_PLOTS=1
# Array backend for every sweep point: "auto" (default) matches GPU_ENABLED
# exactly (CuPy when installed, else NumPy); "cpu"/"gpu" force one backend.
DEVICE="auto"
# Sweep points run sequentially. Parallelism belongs inside self_play's
# CPU-only rollout pool, where game ids, memory fallback, and retained worker
# autotuning are already controlled centrally.
RL_WORKERS=auto

RAM_LIMIT_MB=""
VRAM_LIMIT_MB=""

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

# Grid-search values (3x4x3 = 36 combinations): DEFAULT_LR_VALUES /
# DEFAULT_GAMMA_VALUES from diagnostics/hyperparameter_sweep.py.
# Games-per-iteration is not a tuple there, so this driver owns the established
# 40/80/160 comparison.
LR_VALUES=(0.0005 0.001 0.005)
GAMMA_VALUES=(1.0 0.97 0.95 0.92)
GAMES_PER_ITERATION_VALUES=(40 80 160)

# value_coef axis (critic-on only; value_coef has no effect with the critic
# off). Swept over VC_SAMPLE_COUNT base combinations drawn at random from the
# grid -- the same sampled combinations for every value_coef value. The
# baseline (0.5) is included in the list but skipped during the axis sweep:
# the critic-on grid run already covers it.
VALUE_COEF_VALUES=(0.25 0.5 0.75)
VC_SAMPLE_COUNT=10

usage() {
    cat <<EOF
Train a dedicated RL self-play checkpoint ($RL_ITERATIONS iterations) per
sweep point. The full learning_rate x gamma x games_per_iteration grid
(3x4x3 = 36 combinations) runs for both policies -- critic off and critic on
at the baseline value_coef -- with every base combination identical between
the two. The value_coef axis runs with the critic ON only (it has no effect
otherwise): $VC_SAMPLE_COUNT base combinations are sampled at random (seeded)
from the grid, and each sampled combination is trained once per value_coef
value, reusing the exact same combinations for every value. Reports total
elapsed wall-clock time at the end.

Usage: $(basename "$0") [options]

Grid search axes (cross product, 36 combinations, both critic settings):
  learning_rate        ${LR_VALUES[*]} (baseline: $BASELINE_LEARNING_RATE)
  gamma                 ${GAMMA_VALUES[*]} (baseline: $BASELINE_GAMMA)
  games_per_iteration   ${GAMES_PER_ITERATION_VALUES[*]} (baseline: $BASELINE_GAMES_PER_ITERATION)

value_coef axis (critic on only, over $VC_SAMPLE_COUNT sampled grid combinations):
  value_coef            ${VALUE_COEF_VALUES[*]} (baseline: $BASELINE_VALUE_COEF, covered by the critic-on grid run)

Options:
  --rl-iterations N       RL training iterations per sweep point (default: $RL_ITERATIONS)
  --sl-weights-path PATH  Input SL weights used to initialize every RL run (default: $SL_WEIGHTS_PATH)
  --diagnostic-games N    Games in the rl-vs-random diagnostic per sweep point (default: $DIAGNOSTIC_GAMES)
  --seed N                Fix random/numpy state for both training and diagnostics (default: $SEED)
  --model-dir PATH        Output directory for RL checkpoints (default: $MODEL_DIR)
  --resume                Continue incomplete training and reuse complete, compatible diagnostics
  --diag-no-plots         Skip the per-run diagnostic PNG plots (CSV/JSON are always written)
  --device {auto,cpu,gpu} Array backend for every sweep point; "auto" matches GPU_ENABLED (default: $DEVICE)
  --rl-workers N|auto     CPU-only rollout workers inside the current sweep point (default: $RL_WORKERS)
  --vc-sample-count N     Number of grid combinations sampled for the value_coef axis (default: $VC_SAMPLE_COUNT)
  --results-dir PATH      Output directory for per-run diagnostics subdirectories (default: $RESULTS_DIR)
  --report-output-dir PATH  Where the final comparative table is written (default: $REPORT_OUTPUT_DIR)
  --skip-report           Skip the final comparative-table stage

Sequential execution and memory limits:
  Outer sweep parallelism is disabled; points run one at a time.
  --ram-limit-mb N        Active subprocess physical-memory cap in MiB, enforced via a systemd-run --scope cgroup if available (default: 80% of detected system RAM)
  --vram-limit-mb N       Active subprocess CuPy memory-pool cap in MiB, hard-enforced by CuPy (default: 80% of detected GPU memory when device isn't cpu)

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
        --rl-workers) RL_WORKERS="$2"; shift 2 ;;
        --vc-sample-count) VC_SAMPLE_COUNT="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --report-output-dir) REPORT_OUTPUT_DIR="$2"; shift 2 ;;
        --skip-report) SKIP_REPORT=1; shift ;;
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

if ! [[ "$VC_SAMPLE_COUNT" =~ ^[0-9]+$ ]]; then
    echo "--vc-sample-count must be a non-negative integer, got: $VC_SAMPLE_COUNT" >&2
    exit 1
fi

# ------------------------------------------------------------------
# Sequential execution and memory limits
# ------------------------------------------------------------------
#
# Memory limits remain a general OOM backstop for the one active sweep point.
# They default to 80% of the detected total so headroom remains for the system;
# explicit --ram-limit-mb/--vram-limit-mb values always take precedence.
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
        RAM_LIMIT_MB=$(( total_ram_mb * 80 / 100 ))
        echo "Auto RAM limit for the active job: ${RAM_LIMIT_MB}MiB (80% of ${total_ram_mb}MiB system RAM)"
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
            VRAM_LIMIT_MB=$(( total_vram_mb * 80 / 100 ))
            echo "Auto VRAM limit for the active job: ${VRAM_LIMIT_MB}MiB (80% of ${total_vram_mb}MiB total GPU memory)"
        fi
    fi
fi

# Read by agents/nn.py: a hard cap enforced by CuPy's own memory-pool
# allocator (cupy.get_default_memory_pool().set_limit), not an OS-level
# mechanism, so it doesn't share RLIMIT_AS's incompatibility with CUDA.
export DOMINO_VRAM_LIMIT_MB="$VRAM_LIMIT_MB"

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

mkdir -p "$MODEL_DIR"

# ------------------------------------------------------------------
# value_coef axis: seeded random sample of grid combinations
# ------------------------------------------------------------------
#
# Draw VC_SAMPLE_COUNT base combinations (without replacement) from the full
# lr x gamma x gpi grid, seeded with --seed so the sample is reproducible and
# identical for every value_coef value. Python's random.Random is used
# instead of shuf because shuf offers no portable way to seed. Values travel
# as strings end to end, so they come back byte-identical to the bash arrays.
declare -A VC_SAMPLED_LOOKUP=()
if [[ "$VC_SAMPLE_COUNT" -gt 0 ]]; then
    while IFS= read -r combo; do
        VC_SAMPLED_LOOKUP["$combo"]=1
    done < <("$PYTHON_BIN" -c '
import itertools, random, sys
seed, count = int(sys.argv[1]), int(sys.argv[2])
lrs, gammas, gpis = (arg.split() for arg in sys.argv[3:6])
combos = list(itertools.product(lrs, gammas, gpis))
for combo in random.Random(seed).sample(combos, min(count, len(combos))):
    print(" ".join(combo))
' "$SEED" "$VC_SAMPLE_COUNT" "${LR_VALUES[*]}" "${GAMMA_VALUES[*]}" "${GAMES_PER_ITERATION_VALUES[*]}")
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

# validate_resume_pair WEIGHTS STATE GPI LR GAMMA VC CRITIC
validate_resume_pair() {
    "$PYTHON_BIN" - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$SEED" "$DEVICE" <<'PY'
import inspect
import sys

from training.self_play import (
    _resume_configuration,
    _validate_resume_configuration,
    load_resume_state,
    train,
)
from utils.resource_limits import choose_safe_rl_device

try:
    weights_path, state_path, gpi, learning_rate, gamma, value_coef, critic, seed, requested_device = sys.argv[1:]
    metadata, _pool = load_resume_state(weights_path, state_path)
    parameters = inspect.signature(train).parameters
    default = lambda name: parameters[name].default
    expected = _resume_configuration(
        games_per_iteration=int(gpi),
        training_opponent=default("training_opponent"),
        learning_rate=float(learning_rate),
        entropy_coef=default("entropy_coef"),
        pool_interval=default("pool_interval"),
        max_pool_size=default("max_pool_size"),
        use_value_head=bool(int(critic)),
        value_coef=float(value_coef),
        gamma=float(gamma),
        reward_schema=default("reward_schema"),
        clip_grad_norm=default("clip_grad_norm"),
        normalize_advantages=default("normalize_advantages"),
        effective_seed=int(seed),
        device=choose_safe_rl_device(requested_device)[0],
    )
    _validate_resume_configuration(metadata, expected)
except Exception as exc:
    print(f"Resume pair validation failed: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

# find_latest_resume_checkpoint BASE_PATH GPI LR GAMMA VC CRITIC
# Sets LAST_RESUME_ITERATION/WEIGHTS/STATE to the newest complete, compatible
# pair at or below RL_ITERATIONS. A weights file without its validated state is
# intentionally ignored after a sudden interruption.
find_latest_resume_checkpoint() {
    local base_path="$1" gpi="$2" lr="$3" gamma="$4" vc="$5" critic="$6"
    local base_stem="${base_path%.npz}"
    local state_path weights_path filename digits iteration
    LAST_RESUME_ITERATION=0
    LAST_RESUME_WEIGHTS=""
    LAST_RESUME_STATE=""

    shopt -s nullglob
    for state_path in "${base_stem}"_iter*.resume.npz; do
        weights_path="${state_path%.resume.npz}.npz"
        [[ -f "$weights_path" ]] || continue
        filename="${weights_path##*/}"
        if [[ "$filename" =~ _iter([0-9]+)\.npz$ ]]; then
            digits="${BASH_REMATCH[1]}"
            iteration=$((10#$digits))
        else
            continue
        fi
        if [[ "$iteration" -gt "$RL_ITERATIONS" ]]; then
            continue
        fi
        if ! validate_resume_pair \
            "$weights_path" "$state_path" "$gpi" "$lr" "$gamma" "$vc" "$critic"; then
            echo "Warning: ignoring invalid resume pair $weights_path / $state_path" >&2
            continue
        fi
        if [[ "$iteration" -gt "$LAST_RESUME_ITERATION" ]]; then
            LAST_RESUME_ITERATION="$iteration"
            LAST_RESUME_WEIGHTS="$weights_path"
            LAST_RESUME_STATE="$state_path"
        fi
    done
    shopt -u nullglob
}

# write_run_metadata NAME TAG CRITIC LR GAMMA GPI VC MODEL_PATH DIAG_DIR
write_run_metadata() {
    local name="$1" tag="$2" critic="$3" lr="$4" gamma="$5" gpi="$6" vc="$7" model_path="$8" diag_dir="$9"
    local critic_bool="false"
    local model_sha256
    if [[ "$critic" -eq 1 ]]; then critic_bool="true"; fi
    model_sha256=$("$PYTHON_BIN" - "$model_path" <<'PY'
import sys

from diagnostics.rl_sweep_table import file_sha256

print(file_sha256(sys.argv[1]))
PY
)
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
  "model_path": "$model_path",
  "model_sha256": "$model_sha256"
}
EOF
}

# reusable_diagnostics NAME TAG CRITIC LR GAMMA GPI VC MODEL_PATH DIAG_DIR
# Returns success only when every requested diagnostic artifact is complete
# and belongs to this exact sweep configuration and numbered model checkpoint.
reusable_diagnostics() {
    "$PYTHON_BIN" - \
        "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" \
        "$RL_ITERATIONS" "$SEED" "$DIAGNOSTIC_GAMES" "$SL_WEIGHTS_PATH" \
        "$DIAG_PLOTS" <<'PY'
import sys

from diagnostics.rl_sweep_table import validate_reusable_sweep_diagnostic

(
    name,
    tag,
    critic,
    learning_rate,
    gamma,
    games_per_iteration,
    value_coef,
    model_path,
    diagnostic_dir,
    rl_iterations,
    seed,
    diagnostic_games,
    sl_weights_path,
    diagnostic_plots,
) = sys.argv[1:]
expected = {
    "run_name": name,
    "varied_parameter": tag,
    "critic_enabled": bool(int(critic)),
    "learning_rate": float(learning_rate),
    "gamma": float(gamma),
    "games_per_iteration": int(games_per_iteration),
    "value_coef": float(value_coef),
    "rl_iterations": int(rl_iterations),
    "seed": int(seed),
    "diagnostic_games": int(diagnostic_games),
    "sl_weights_path": sl_weights_path,
    "model_path": model_path,
}
valid, _reason = validate_reusable_sweep_diagnostic(
    diagnostic_dir,
    expected,
    model_path,
    require_plots=bool(int(diagnostic_plots)),
)
raise SystemExit(0 if valid else 1)
PY
}

# run_point TAG CRITIC LR GAMMA GPI VC
run_point() {
    local tag="$1" critic="$2" lr="$3" gamma="$4" gpi="$5" vc="$6"
    local name="domino_rl"
    if [[ "$critic" -eq 1 ]]; then
        name="${name}_critic"
    fi
    name="${name}_${tag}"

    local model_base_path="$MODEL_DIR/${name}.npz"
    local model_path
    local final_model_path
    local diag_dir="$RESULTS_DIR/${name}"
    local critic_label="off"
    if [[ "$critic" -eq 1 ]]; then
        critic_label="on"
    fi

    # A compatible final diagnostic proves that this exact numbered final
    # model completed the point. Check it before loading resumable pool state:
    # no further computation remains, so a CPU/GPU selection change cannot
    # invalidate or alter the already completed model and diagnostic.
    printf -v final_model_path '%s_iter%06d.npz' \
        "${model_base_path%.npz}" "$RL_ITERATIONS"
    if [[ "$RESUME" -eq 1 ]] && reusable_diagnostics \
        "$name" "$tag" "$critic" "$lr" "$gamma" "$gpi" "$vc" \
        "$final_model_path" "$diag_dir"; then
        section "[$name] training already complete at iteration $RL_ITERATIONS (--resume: $final_model_path)"
        section "[$name] diagnostics already complete ($DIAGNOSTIC_GAMES games; --resume: $diag_dir/summary.json)"
        return
    fi

    LAST_RESUME_ITERATION=0
    LAST_RESUME_WEIGHTS=""
    LAST_RESUME_STATE=""
    if [[ "$RESUME" -eq 1 ]]; then
        find_latest_resume_checkpoint \
            "$model_base_path" "$gpi" "$lr" "$gamma" "$vc" "$critic"
    fi

    if [[ "$LAST_RESUME_ITERATION" -eq "$RL_ITERATIONS" ]]; then
        model_path="$LAST_RESUME_WEIGHTS"
        section "[$name] training already complete at iteration $RL_ITERATIONS (--resume: $model_path)"
    else
        RESUME_ARGS=(--numbered-checkpoints)
        FRESH_START_ARGS=(--fresh-from-sl)
        if [[ "$LAST_RESUME_ITERATION" -gt 0 ]]; then
            FRESH_START_ARGS=()
            RESUME_ARGS+=(
                --start-iteration "$LAST_RESUME_ITERATION"
                --resume-weights-path "$LAST_RESUME_WEIGHTS"
                --resume-state-file "$LAST_RESUME_STATE"
            )
            section "[$name] resuming RL at iteration $LAST_RESUME_ITERATION/$RL_ITERATIONS, lr=$lr gamma=$gamma games/iter=$gpi value_coef=$vc critic=$critic_label"
        else
            section "[$name] RL training: $RL_ITERATIONS iterations, lr=$lr gamma=$gamma games/iter=$gpi value_coef=$vc critic=$critic_label"
        fi
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
            --rl-weights-path "$model_base_path" \
            --seed "$SEED" \
            --device "$DEVICE" \
            --rl-workers "$RL_WORKERS" \
            --compact \
            "${FRESH_START_ARGS[@]}" \
            "${RESUME_ARGS[@]}" \
            "${VALUE_HEAD_FLAG[@]}"
        model_path="$final_model_path"
    fi

    if [[ "$RESUME" -eq 1 ]] && reusable_diagnostics \
        "$name" "$tag" "$critic" "$lr" "$gamma" "$gpi" "$vc" \
        "$model_path" "$diag_dir"; then
        section "[$name] diagnostics already complete ($DIAGNOSTIC_GAMES games; --resume: $diag_dir/summary.json)"
    else
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
    fi
}

# ------------------------------------------------------------------
# Sweep plan (echoed up front so a run can be sanity-checked before it burns
# hours of RL training)
# ------------------------------------------------------------------

grid_point_count=$((${#LR_VALUES[@]} * ${#GAMMA_VALUES[@]} * ${#GAMES_PER_ITERATION_VALUES[@]}))
vc_extra_values=0
for vc in "${VALUE_COEF_VALUES[@]}"; do
    [[ "$vc" == "$BASELINE_VALUE_COEF" ]] || vc_extra_values=$((vc_extra_values + 1))
done
total_points=$((grid_point_count * 2 + ${#VC_SAMPLED_LOOKUP[@]} * vc_extra_values))

section "RL parameter sweep: $total_points training+diagnostics runs ($RL_ITERATIONS iterations each: $grid_point_count-point grid x 2 critic settings, plus ${#VC_SAMPLED_LOOKUP[@]} sampled combinations x $vc_extra_values non-baseline value_coef values, critic on)"
if [[ "${#VC_SAMPLED_LOOKUP[@]}" -gt 0 ]]; then
    echo "value_coef axis combinations (lr gamma gpi), sampled with seed $SEED:"
    printf '  %s\n' "${!VC_SAMPLED_LOOKUP[@]}" | sort
fi

# ------------------------------------------------------------------
# Sweep
# ------------------------------------------------------------------

# One pass over the full cross-product grid. Per base combination: run it
# critic-off, then critic-on at the baseline value_coef -- the exact same
# base combinations for both policies -- then, if the combination was
# sampled for the value_coef axis, once more per remaining value_coef value
# (critic on only: value_coef has no effect with the critic off). The
# baseline value_coef is skipped there because the critic-on grid run just
# above already covers it.
for LR in "${LR_VALUES[@]}"; do
    for GAMMA in "${GAMMA_VALUES[@]}"; do
        for GPI in "${GAMES_PER_ITERATION_VALUES[@]}"; do
            if [[ "$LR" == "$BASELINE_LEARNING_RATE" && "$GAMMA" == "$BASELINE_GAMMA" && "$GPI" == "$BASELINE_GAMES_PER_ITERATION" ]]; then
                TAG="default"
            else
                TAG="lr${LR}_gamma${GAMMA}_gpi${GPI}"
            fi

            run_point "$TAG" 0 "$LR" "$GAMMA" "$GPI" "$BASELINE_VALUE_COEF"
            run_point "$TAG" 1 "$LR" "$GAMMA" "$GPI" "$BASELINE_VALUE_COEF"

            if [[ -n "${VC_SAMPLED_LOOKUP["$LR $GAMMA $GPI"]:-}" ]]; then
                for VC in "${VALUE_COEF_VALUES[@]}"; do
                    if [[ "$VC" == "$BASELINE_VALUE_COEF" ]]; then
                        continue
                    fi
                    run_point "${TAG}_vc${VC}" 1 "$LR" "$GAMMA" "$GPI" "$VC"
                done
            fi
        done
    done
done

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
