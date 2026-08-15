#!/usr/bin/env bash
#
# Reward-shaping grid search over the three RL reward tunables.
#
# Each grid point is one `python3 -m training.pipeline forever` invocation with
# its own `--run-name`, so every point gets a separate run directory under
# `models/rl/` and never resumes another point's checkpoints. The `forever`
# level reuses seed-addressed supervised assets, so the dataset and the
# supervised policy are built once by the first point and shared by the rest.
#
# The sweep follows the identifiability of each parameter under
#
#     R_T = (1 - alpha) * gamma ** k * R_f + alpha * R_l
#
# so a parameter is swept only where it can actually change the reward:
#
#   alpha = 0    the local term vanishes -> sweep gamma only          (3 points)
#   alpha = 1    the terminal term vanishes -> sweep decay only       (3 points)
#   alpha = 0.5  both terms are live -> sweep gamma x decay           (9 points)
#
# for 15 points total. The parameter that is inert at a given alpha is still
# passed explicitly, pinned to PINNED_GAMMA/PINNED_DECAY, so every run_config
# records a complete and reproducible reward configuration.
#
# TIME LIMIT
#
# `forever` never stops on its own, so each point is capped by wall clock. The
# cap covers the RL stage only: the timer starts when the pipeline prints its
# "Canonical RL run" banner, so dataset and supervised work at the front of the
# first point is not charged against any point's RL budget.
#
# The cap is enforced with the pipeline's own graceful-shutdown path. The first
# SIGTERM sets its ShutdownFlag, which lets the current RL iteration finish and
# then publishes a safe boundary checkpoint before exiting 0; rollout workers
# ignore SIGTERM, so only the parent reacts. Because that waits for an
# iteration boundary, a point can overrun the cap by up to one iteration, and
# the script escalates to keep the bound hard:
#
#   1. SIGTERM        -> graceful stop at the next iteration boundary
#   2. SIGTERM again  -> KeyboardInterrupt inside the pipeline (after --grace)
#   3. SIGKILL        -> process group killed (after another --grace)
#
# Only step 1 is guaranteed to leave a resumable checkpoint, so keep --grace
# comfortably above one RL iteration. A point that needed step 2 or 3 is
# recorded as "hard-stopped" in the summary.
#
# The script is idempotent: completed points are skipped on re-invocation, so
# an interrupted sweep can be continued by running it again.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RL_TIME_LIMIT=7200
GRACE_SECONDS=900
BANNER_TIMEOUT=21600
POLL_SECONDS=5
GAMMA_VALUES=(0.8 0.9 1)
DECAY_VALUES=(0.8 0.9 1)
ALPHA_VALUES=(0 0.5 1)
PINNED_GAMMA=1
PINNED_DECAY=0.9
RESULTS_DIR="$REPO_ROOT/train_script/grid_search_results"
DRY_RUN=0
FORCE=0
ONLY_PATTERN=""
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: train_script/run_reward_grid_search.sh [options] [-- extra pipeline args]

Grid search over gamma, alpha, and event-reward-decay, one
\`training.pipeline forever\` run per point, each capped at a fixed amount of
RL wall-clock time.

Options:
  --time-limit DURATION   RL wall clock per point; accepts plain seconds or a
                          30m/2h/1d suffix (default: ${RL_TIME_LIMIT}s = 2h)
  --grace DURATION        Wait after each shutdown escalation step before the
                          next one (default: ${GRACE_SECONDS}s). Must exceed one
                          RL iteration or points will be hard-stopped.
  --results-dir DIR       Logs and sweep state (default: $RESULTS_DIR)
  --only PATTERN          Run only points whose run name matches this shell
                          pattern, e.g. --only 'a05_*'
  --force                 Re-run points already marked completed
  --dry-run               Print the plan and the exact commands, run nothing
  -h, --help              Show this help

Environment:
  PYTHON                  Interpreter to use. Defaults to \$REPO_ROOT/.venv/bin/python
                          when present, else python3.

Examples:
  # Full 15-point sweep, 2 hours of RL per point
  train_script/run_reward_grid_search.sh

  # See the plan without running anything
  train_script/run_reward_grid_search.sh --dry-run

  # Short smoke test of the whole harness
  train_script/run_reward_grid_search.sh --time-limit 5m --grace 2m --only 'a0_*'

  # Forward extra flags to every pipeline invocation
  train_script/run_reward_grid_search.sh -- --rl-workers 4
EOF
}

# Accept 90, 30m, 2h, or 1d so the cap reads naturally at the call site.
parse_duration() {
    local value="$1" number unit
    number="${value%[smhd]}"
    unit="${value#"$number"}"
    if [[ ! "$number" =~ ^[0-9]+$ ]]; then
        echo "Invalid duration: $value" >&2
        exit 1
    fi
    case "$unit" in
        ""|s) echo "$number" ;;
        m) echo $((number * 60)) ;;
        h) echo $((number * 3600)) ;;
        d) echo $((number * 86400)) ;;
        *) echo "Invalid duration unit in: $value" >&2; exit 1 ;;
    esac
}

format_duration() {
    local total="$1"
    printf '%dh%02dm%02ds' $((total / 3600)) $((total % 3600 / 60)) $((total % 60))
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --time-limit) RL_TIME_LIMIT="$(parse_duration "$2")"; shift 2 ;;
        --grace) GRACE_SECONDS="$(parse_duration "$2")"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --only) ONLY_PATTERN="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

# Prefer the repository interpreter without depending on a user's shell or
# machine-specific virtual-environment location.
if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "No interpreter found; set PYTHON to the one with the project installed." >&2
    exit 1
fi

# 0.8 -> 08, 0.5 -> 05, 1 -> 1. Run names accept only letters, digits, '-'
# and '_', so the decimal point cannot survive into the label.
label_for() {
    local value="$1"
    echo "${value//./}"
}

# Emits one "alpha gamma decay run_name" row per grid point.
build_grid() {
    local alpha gamma decay name
    for alpha in "${ALPHA_VALUES[@]}"; do
        case "$alpha" in
            0)
                for gamma in "${GAMMA_VALUES[@]}"; do
                    decay="$PINNED_DECAY"
                    name="a$(label_for "$alpha")_g$(label_for "$gamma")_d$(label_for "$decay")"
                    echo "$alpha $gamma $decay $name"
                done
                ;;
            1)
                for decay in "${DECAY_VALUES[@]}"; do
                    gamma="$PINNED_GAMMA"
                    name="a$(label_for "$alpha")_g$(label_for "$gamma")_d$(label_for "$decay")"
                    echo "$alpha $gamma $decay $name"
                done
                ;;
            *)
                for gamma in "${GAMMA_VALUES[@]}"; do
                    for decay in "${DECAY_VALUES[@]}"; do
                        name="a$(label_for "$alpha")_g$(label_for "$gamma")_d$(label_for "$decay")"
                        echo "$alpha $gamma $decay $name"
                    done
                done
                ;;
        esac
    done
}

STATE_FILE="$RESULTS_DIR/grid_state.tsv"

# The running point lives in its own session, so it would survive this script
# being interrupted. Interrupting the sweep must stop the point it is running,
# and it should stop it the same graceful way the budget does.
CURRENT_PID=""
CURRENT_PGID=""

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$CURRENT_PID" ]] && kill -0 "$CURRENT_PID" 2>/dev/null; then
        echo >&2
        echo "Sweep interrupted; asking the running point to checkpoint and stop." >&2
        kill -TERM "$CURRENT_PID" 2>/dev/null || true
        if ! wait_for_exit "$CURRENT_PID" "$GRACE_SECONDS"; then
            echo "  Still running; killing the process group." >&2
            kill -KILL "-$CURRENT_PGID" 2>/dev/null || true
        fi
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

point_completed() {
    local name="$1"
    [[ -f "$STATE_FILE" ]] || return 1
    awk -F'\t' -v want="$name" '
        $1 == want && ($2 == "graceful" || $2 == "hard-stopped" || $2 == "finished") {
            found = 1
        }
        END { exit !found }
    ' "$STATE_FILE"
}

record_point() {
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$1" "$2" "$3" "$4" "$5" "$(date -Is)" >>"$STATE_FILE"
}

# Runs one grid point under the wall-clock cap and leaves the outcome in
# POINT_STATUS. It cannot report through stdout in a command substitution: the
# subshell that would create hides the child PID from the interrupt trap.
run_point() {
    local alpha="$1" gamma="$2" decay="$3" name="$4" log="$5"
    local cmd=(
        "$PYTHON_BIN" -u -m training.pipeline forever
        --alpha "$alpha"
        --gamma "$gamma"
        --event-reward-decay "$decay"
        --run-name "$name"
    )
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    # setsid puts the pipeline in its own process group so the SIGKILL backstop
    # can reap the rollout workers with it instead of orphaning them.
    setsid "${cmd[@]}" >"$log" 2>&1 &
    local pid=$!
    local pgid
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    [[ -n "$pgid" ]] || pgid="$pid"
    CURRENT_PID="$pid"
    CURRENT_PGID="$pgid"

    # Phase 1: wait for the RL stage to start before charging the budget.
    local waited=0 rl_started=0
    while kill -0 "$pid" 2>/dev/null; do
        if grep -q "Canonical RL run" "$log" 2>/dev/null; then
            rl_started=1
            break
        fi
        if (( waited >= BANNER_TIMEOUT )); then
            echo "  RL stage did not start within $(format_duration "$BANNER_TIMEOUT"); stopping." >&2
            break
        fi
        sleep "$POLL_SECONDS"
        waited=$((waited + POLL_SECONDS))
    done

    if (( rl_started )); then
        echo "  RL stage started after $(format_duration "$waited"); budget $(format_duration "$RL_TIME_LIMIT") begins now." >&2
    fi

    # Phase 2: hold until the pipeline exits on its own or the budget runs out.
    local rl_elapsed=0
    while kill -0 "$pid" 2>/dev/null && (( rl_elapsed < RL_TIME_LIMIT )); do
        sleep "$POLL_SECONDS"
        rl_elapsed=$((rl_elapsed + POLL_SECONDS))
    done

    if ! kill -0 "$pid" 2>/dev/null; then
        if wait "$pid" 2>/dev/null; then
            POINT_STATUS="finished"
        else
            POINT_STATUS="failed"
        fi
        CURRENT_PID=""
        return
    fi

    # Phase 3: escalate until it is gone, giving the graceful path first refusal.
    echo "  Budget reached; requesting a graceful stop at the next iteration boundary." >&2
    kill -TERM "$pid" 2>/dev/null || true
    if wait_for_exit "$pid" "$GRACE_SECONDS"; then
        wait "$pid" 2>/dev/null || true
        POINT_STATUS="graceful"
        CURRENT_PID=""
        return
    fi

    echo "  Still running after $(format_duration "$GRACE_SECONDS"); interrupting." >&2
    kill -TERM "$pid" 2>/dev/null || true
    if wait_for_exit "$pid" "$GRACE_SECONDS"; then
        wait "$pid" 2>/dev/null || true
        POINT_STATUS="hard-stopped"
        CURRENT_PID=""
        return
    fi

    echo "  Unresponsive; killing the process group." >&2
    kill -KILL "-$pgid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    POINT_STATUS="hard-stopped"
    CURRENT_PID=""
}

wait_for_exit() {
    local pid="$1" limit="$2" waited=0
    while (( waited < limit )); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep "$POLL_SECONDS"
        waited=$((waited + POLL_SECONDS))
    done
    ! kill -0 "$pid" 2>/dev/null
}

mapfile -t GRID < <(build_grid)

if [[ -n "$ONLY_PATTERN" ]]; then
    FILTERED=()
    for row in "${GRID[@]}"; do
        read -r _alpha _gamma _decay name <<<"$row"
        # Unquoted on purpose: ONLY_PATTERN is matched as a shell pattern.
        # shellcheck disable=SC2053
        if [[ "$name" == $ONLY_PATTERN ]]; then
            FILTERED+=("$row")
        fi
    done
    GRID=(${FILTERED[@]+"${FILTERED[@]}"})
fi

if [[ ${#GRID[@]} -eq 0 ]]; then
    echo "No grid points selected." >&2
    exit 1
fi

echo "======================================================================"
echo "Reward-shaping grid search"
echo "======================================================================"
echo "Interpreter:      $PYTHON_BIN"
echo "Points:           ${#GRID[@]}"
echo "RL cap per point: $(format_duration "$RL_TIME_LIMIT")"
echo "Shutdown grace:   $(format_duration "$GRACE_SECONDS") per escalation step"
echo "Results:          $RESULTS_DIR"
echo "Worst-case total: $(format_duration $(( ${#GRID[@]} * (RL_TIME_LIMIT + 2 * GRACE_SECONDS) )))"
echo "----------------------------------------------------------------------"
printf '%-14s %-7s %-7s %-7s\n' "RUN NAME" "ALPHA" "GAMMA" "DECAY"
for row in "${GRID[@]}"; do
    read -r alpha gamma decay name <<<"$row"
    printf '%-14s %-7s %-7s %-7s\n' "$name" "$alpha" "$gamma" "$decay"
done
echo "----------------------------------------------------------------------"

if (( DRY_RUN )); then
    echo "Dry run; commands that would be issued:"
    for row in "${GRID[@]}"; do
        read -r alpha gamma decay name <<<"$row"
        echo "  $PYTHON_BIN -u -m training.pipeline forever --alpha $alpha" \
             "--gamma $gamma --event-reward-decay $decay --run-name $name" \
             "${EXTRA_ARGS[*]+${EXTRA_ARGS[*]}}"
    done
    exit 0
fi

mkdir -p "$RESULTS_DIR"
[[ -f "$STATE_FILE" ]] || printf 'run_name\tstatus\talpha\tgamma\tevent_reward_decay\tfinished_at\n' >"$STATE_FILE"

index=0
for row in "${GRID[@]}"; do
    read -r alpha gamma decay name <<<"$row"
    index=$((index + 1))

    if (( ! FORCE )) && point_completed "$name"; then
        echo "[$index/${#GRID[@]}] $name -- already completed, skipping (--force to redo)"
        continue
    fi

    log="$RESULTS_DIR/$name.log"
    echo
    echo "[$index/${#GRID[@]}] $name  alpha=$alpha gamma=$gamma event-reward-decay=$decay"
    echo "  Log: $log"
    started=$(date +%s)
    POINT_STATUS="failed"
    run_point "$alpha" "$gamma" "$decay" "$name" "$log"
    elapsed=$(( $(date +%s) - started ))
    echo "  Status: $POINT_STATUS after $(format_duration "$elapsed")"
    record_point "$name" "$POINT_STATUS" "$alpha" "$gamma" "$decay"
done

echo
echo "======================================================================"
echo "Sweep complete"
echo "======================================================================"
echo "State:    $STATE_FILE"
echo "Logs:     $RESULTS_DIR"
echo "Run dirs: models/rl/domino_rl_forever_seed42_run<RUN_NAME>"
echo
column -t -s $'\t' "$STATE_FILE" 2>/dev/null || cat "$STATE_FILE"
