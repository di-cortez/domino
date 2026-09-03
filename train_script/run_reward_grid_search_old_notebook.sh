#!/usr/bin/env bash
#
# Old-notebook reward-shaping grid search over the three RL reward tunables.
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
#   alpha = 0.5  both terms are live -> sweep gamma x decay           (9 points)
#
# This old-notebook profile runs these 9 points. The 6 alpha=0/1 points belong
# to the desktop profile. Every reward parameter is live at alpha=0.5, and
# every run_config records the complete reward configuration explicitly.
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
# The script is idempotent and resume-aware.  Its TSV state contains one
# current row per grid point, including accumulated RL wall time.  Completed
# points are skipped, while an interrupted point automatically reloads its
# canonical checkpoint and receives only the unspent part of its original
# wall-clock budget.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RL_TIME_LIMIT=25200
GRACE_SECONDS=900
BANNER_TIMEOUT="${GRID_BANNER_TIMEOUT_SECONDS:-21600}"
POLL_SECONDS="${GRID_POLL_SECONDS:-5}"
GAMMA_VALUES=(0.8 0.9 1)
DECAY_VALUES=(0.8 0.9 1)
ALPHA_VALUES=(0.5)
PINNED_GAMMA=1
PINNED_DECAY=0.9
RESULTS_DIR="$REPO_ROOT/train_script/grid_search_results_old_notebook"
RUN_ROOT="${GRID_RUN_ROOT:-$REPO_ROOT/models/rl}"
DRY_RUN=0
FORCE=0
TIME_LIMIT_EXPLICIT=0
ONLY_PATTERN=""
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: train_script/run_reward_grid_search_old_notebook.sh [options] [-- extra pipeline args]

Grid search over gamma, alpha, and event-reward-decay, one
\`training.pipeline forever\` run per point, each capped at a fixed amount of
RL wall-clock time.

Options:
  --time-limit DURATION   RL wall clock per point; accepts plain seconds or a
                          30m/2h/1d suffix (default: ${RL_TIME_LIMIT}s = 7h)
  --grace DURATION        Wait after each shutdown escalation step before the
                          next one (default: ${GRACE_SECONDS}s). Must exceed one
                          RL iteration or points will be hard-stopped.
  --results-dir DIR       Logs and sweep state (default: $RESULTS_DIR)
  --only PATTERN          Run only points whose run name matches this shell
                          pattern, e.g. --only 'a05_*'
  --force                 Archive and restart selected points with a fresh
                          full time budget
  --dry-run               Print the plan and the exact commands, run nothing
  -h, --help              Show this help

Environment:
  PYTHON                  Interpreter to use. Defaults to \$REPO_ROOT/.venv/bin/python
                          when present, else python3.

Examples:
  # Old-notebook 9-point sweep, 7 hours of RL per point. Running this same
  # command again resumes the interrupted point and continues the pending ones.
  train_script/run_reward_grid_search_old_notebook.sh

  # See the plan without running anything
  train_script/run_reward_grid_search_old_notebook.sh --dry-run

  # Short smoke test of the whole harness
  train_script/run_reward_grid_search_old_notebook.sh --time-limit 5m --grace 2m --only 'a05_*'

  # Forward extra flags to every pipeline invocation
  train_script/run_reward_grid_search_old_notebook.sh -- --rl-workers 4
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
        --time-limit)
            RL_TIME_LIMIT="$(parse_duration "$2")"
            TIME_LIMIT_EXPLICIT=1
            shift 2
            ;;
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

if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ || ! "$BANNER_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
    echo "GRID_POLL_SECONDS and GRID_BANNER_TIMEOUT_SECONDS must be positive integers." >&2
    exit 1
fi

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
STATE_HEADER=$'run_name\tstatus\talpha\tgamma\tevent_reward_decay\tbudget_seconds\trl_elapsed_seconds\tattempts\tpid\tpgid\tlast_outcome\tupdated_at'
LEGACY_STATE_HEADER=$'run_name\tstatus\talpha\tgamma\tevent_reward_decay\tfinished_at'

# Current-point fields are globals because the signal trap must persist an
# exact interrupted state even when Ctrl+C lands inside sleep/wait.
CURRENT_NAME=""
CURRENT_ALPHA=""
CURRENT_GAMMA=""
CURRENT_DECAY=""
CURRENT_BUDGET=0
CURRENT_ELAPSED_BASE=0
CURRENT_ATTEMPT=0
CURRENT_PID=""
CURRENT_PGID=""
CURRENT_RL_STARTED_AT=0
POINT_STATUS="failed"
POINT_OUTCOME="not-started"
POINT_ELAPSED=0

state_row() {
    local name="$1"
    awk -F'\t' -v want="$name" '$1 == want { print; found = 1; exit } END { exit !found }' "$STATE_FILE"
}

state_upsert() {
    local name="$1" status="$2" alpha="$3" gamma="$4" decay="$5"
    local budget="$6" elapsed="$7" attempts="$8" pid="$9" pgid="${10}"
    local outcome="${11}" updated_at="${12}" temporary
    temporary="$(mktemp "$RESULTS_DIR/.grid_state.XXXXXX")"
    awk -F'\t' -v OFS='\t' \
        -v name="$name" -v status="$status" -v alpha="$alpha" \
        -v gamma="$gamma" -v decay="$decay" -v budget="$budget" \
        -v elapsed="$elapsed" -v attempts="$attempts" -v pid="$pid" \
        -v pgid="$pgid" -v outcome="$outcome" -v updated="$updated_at" '
        NR == 1 { print; next }
        $1 == name {
            print name, status, alpha, gamma, decay, budget, elapsed,
                  attempts, pid, pgid, outcome, updated
            found = 1
            next
        }
        { print }
        END {
            if (!found) {
                print name, status, alpha, gamma, decay, budget, elapsed,
                      attempts, pid, pgid, outcome, updated
            }
        }
    ' "$STATE_FILE" >"$temporary"
    chmod 0644 "$temporary"
    mv "$temporary" "$STATE_FILE"
}

migrate_or_create_state() {
    local header temporary now
    if [[ ! -f "$STATE_FILE" ]]; then
        printf '%s\n' "$STATE_HEADER" >"$STATE_FILE"
        return
    fi
    IFS= read -r header <"$STATE_FILE"
    [[ "$header" == "$STATE_HEADER" ]] && return
    if [[ "$header" != "$LEGACY_STATE_HEADER" ]]; then
        echo "Unsupported grid state schema in $STATE_FILE" >&2
        exit 1
    fi
    now="$(date -Is)"
    temporary="$(mktemp "$RESULTS_DIR/.grid_state.migrate.XXXXXX")"
    awk -F'\t' -v OFS='\t' -v header="$STATE_HEADER" \
        -v budget="$RL_TIME_LIMIT" -v now="$now" '
        NR == 1 { print header; next }
        {
            old_status = $2
            completed = (old_status == "graceful" ||
                         old_status == "hard-stopped" ||
                         old_status == "finished")
            status = completed ? "completed" : "failed"
            elapsed = completed ? budget : 0
            updated = ($6 == "" ? now : $6)
            print $1, status, $3, $4, $5, budget, elapsed, 1,
                  "-", "-", "legacy-" old_status, updated
        }
    ' "$STATE_FILE" >"$temporary"
    chmod 0644 "$temporary"
    mv "$temporary" "$STATE_FILE"
    echo "Migrated legacy grid state to the resumable timing schema."
}

ensure_state_rows() {
    local row alpha gamma decay name
    for row in "${ALL_GRID[@]}"; do
        read -r alpha gamma decay name <<<"$row"
        if ! state_row "$name" >/dev/null 2>&1; then
            state_upsert "$name" pending "$alpha" "$gamma" "$decay" \
                "$RL_TIME_LIMIT" 0 0 - - never-started "$(date -Is)"
        fi
    done
}

run_directory_for() {
    printf '%s/domino_rl_forever_seed42_run%s\n' "$RUN_ROOT" "$1"
}

current_elapsed() {
    local elapsed="$CURRENT_ELAPSED_BASE"
    if (( CURRENT_RL_STARTED_AT > 0 )); then
        elapsed=$((elapsed + $(date +%s) - CURRENT_RL_STARTED_AT))
    fi
    printf '%s\n' "$elapsed"
}

persist_current() {
    local status="$1" outcome="$2" elapsed pid pgid
    [[ -n "$CURRENT_NAME" ]] || return 0
    elapsed="$(current_elapsed)"
    pid="${CURRENT_PID:--}"
    pgid="${CURRENT_PGID:--}"
    state_upsert "$CURRENT_NAME" "$status" "$CURRENT_ALPHA" \
        "$CURRENT_GAMMA" "$CURRENT_DECAY" "$CURRENT_BUDGET" "$elapsed" \
        "$CURRENT_ATTEMPT" "$pid" "$pgid" "$outcome" "$(date -Is)"
}

wait_for_exit() {
    local pid="$1" limit="$2" waited=0
    while (( waited < limit )); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep "$POLL_SECONDS"
        waited=$((waited + POLL_SECONDS))
        persist_current running stopping
    done
    ! kill -0 "$pid" 2>/dev/null
}

clear_current() {
    CURRENT_NAME=""
    CURRENT_ALPHA=""
    CURRENT_GAMMA=""
    CURRENT_DECAY=""
    CURRENT_BUDGET=0
    CURRENT_ELAPSED_BASE=0
    CURRENT_ATTEMPT=0
    CURRENT_PID=""
    CURRENT_PGID=""
    CURRENT_RL_STARTED_AT=0
}

cleanup() {
    local status=$? outcome="interrupted-before-process-start" elapsed
    trap - EXIT INT TERM
    set +e
    if [[ -n "$CURRENT_NAME" ]]; then
        if [[ -n "$CURRENT_PID" ]] && kill -0 "$CURRENT_PID" 2>/dev/null; then
            echo >&2
            echo "Sweep interrupted; asking $CURRENT_NAME to checkpoint and stop." >&2
            kill -TERM "$CURRENT_PID" 2>/dev/null
            if wait_for_exit "$CURRENT_PID" "$GRACE_SECONDS"; then
                wait "$CURRENT_PID" 2>/dev/null
                outcome="interrupted-graceful-checkpoint"
            else
                echo "  Still running; killing the process group." >&2
                kill -KILL "-$CURRENT_PGID" 2>/dev/null
                wait "$CURRENT_PID" 2>/dev/null
                outcome="interrupted-hard-stop"
            fi
        fi
        elapsed="$(current_elapsed)"
        state_upsert "$CURRENT_NAME" interrupted "$CURRENT_ALPHA" \
            "$CURRENT_GAMMA" "$CURRENT_DECAY" "$CURRENT_BUDGET" "$elapsed" \
            "$CURRENT_ATTEMPT" - - "$outcome" "$(date -Is)"
        echo "  Saved grid progress: $(format_duration "$elapsed") / $(format_duration "$CURRENT_BUDGET")." >&2
    fi
    exit "$status"
}

process_matches_point() {
    local pid="$1" name="$2" command
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    command="$(ps -o args= -p "$pid" 2>/dev/null || true)"
    [[ "$command" == *"training.pipeline forever"* && "$command" == *"--run-name $name"* ]]
}

recover_stale_running_rows() {
    local name row status alpha gamma decay budget elapsed attempts pid pgid outcome updated
    mapfile -t running_names < <(awk -F'\t' 'NR > 1 && $2 == "running" { print $1 }' "$STATE_FILE")
    for name in "${running_names[@]}"; do
        row="$(state_row "$name")"
        IFS=$'\t' read -r _ status alpha gamma decay budget elapsed attempts \
            pid pgid outcome updated <<<"$row"
        if process_matches_point "$pid" "$name"; then
            echo "Grid point $name is still running as PID $pid; refusing a duplicate sweep." >&2
            exit 1
        fi
        state_upsert "$name" interrupted "$alpha" "$gamma" "$decay" \
            "$budget" "$elapsed" "$attempts" - - stale-run-recovered "$(date -Is)"
    done
}

reset_selected_for_force() {
    local row alpha gamma decay name
    for row in "${GRID[@]}"; do
        read -r alpha gamma decay name <<<"$row"
        state_upsert "$name" pending "$alpha" "$gamma" "$decay" \
            "$RL_TIME_LIMIT" 0 0 - - force-reset "$(date -Is)"
    done
}

apply_explicit_budget_to_pending() {
    local row alpha gamma decay name saved status budget elapsed attempts pid pgid outcome updated
    (( TIME_LIMIT_EXPLICIT )) || return 0
    for row in "${GRID[@]}"; do
        read -r alpha gamma decay name <<<"$row"
        saved="$(state_row "$name")"
        IFS=$'\t' read -r _ status _ _ _ budget elapsed attempts pid pgid outcome updated <<<"$saved"
        if [[ "$status" == pending && "$elapsed" == 0 ]]; then
            state_upsert "$name" pending "$alpha" "$gamma" "$decay" \
                "$RL_TIME_LIMIT" 0 "$attempts" - - budget-set "$(date -Is)"
        fi
    done
}

report_state_summary() {
    local row name saved status budget elapsed remaining
    local completed=0 resumable=0 pending=0 failed=0
    local completed_names=() remaining_names=()
    for row in "${GRID[@]}"; do
        read -r _ _ _ name <<<"$row"
        saved="$(state_row "$name")"
        IFS=$'\t' read -r _ status _ _ _ budget elapsed _ _ _ _ _ <<<"$saved"
        case "$status" in
            completed)
                completed=$((completed + 1))
                completed_names+=("$name")
                ;;
            interrupted)
                resumable=$((resumable + 1))
                remaining_names+=("$name")
                ;;
            failed)
                failed=$((failed + 1))
                remaining_names+=("$name")
                ;;
            *)
                pending=$((pending + 1))
                remaining_names+=("$name")
                ;;
        esac
    done
    echo "Grid state: $completed/${#GRID[@]} completed; $resumable resumable; $pending pending; $failed failed."
    if (( completed > 0 )); then
        echo "Completed: ${completed_names[*]}"
    fi
    if (( ${#remaining_names[@]} > 0 )); then
        echo "Remaining: ${remaining_names[*]}"
    fi
    for name in "${remaining_names[@]}"; do
        saved="$(state_row "$name")"
        IFS=$'\t' read -r _ status _ _ _ budget elapsed _ _ _ _ _ <<<"$saved"
        if [[ "$status" == interrupted || "$status" == failed ]]; then
            remaining=$((budget > elapsed ? budget - elapsed : 0))
            echo "Next resumable $name: $(format_duration "$elapsed") used, $(format_duration "$remaining") remaining."
            break
        fi
    done
}

# Runs one grid point and updates POINT_STATUS/POINT_OUTCOME/POINT_ELAPSED.
run_point() {
    local alpha="$1" gamma="$2" decay="$3" name="$4" log="$5"
    local budget="$6" elapsed_before="$7" attempt="$8" mode="$9"
    local cmd=(
        "$PYTHON_BIN" -u -m training.pipeline forever
        --alpha "$alpha"
        --gamma "$gamma"
        --event-reward-decay "$decay"
        --run-name "$name"
    )
    case "$mode" in
        resume) cmd+=(--resume) ;;
        restart) cmd+=(--restart-rl) ;;
    esac
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    CURRENT_NAME="$name"
    CURRENT_ALPHA="$alpha"
    CURRENT_GAMMA="$gamma"
    CURRENT_DECAY="$decay"
    CURRENT_BUDGET="$budget"
    CURRENT_ELAPSED_BASE="$elapsed_before"
    CURRENT_ATTEMPT="$attempt"
    CURRENT_RL_STARTED_AT=0
    POINT_STATUS="failed"
    POINT_OUTCOME="launch-failed"
    POINT_ELAPSED="$elapsed_before"

    # Each attempt owns a fresh log, preventing an old banner from starting the
    # new timer before the resumed process actually reaches RL.
    setsid "${cmd[@]}" >"$log" 2>&1 &
    CURRENT_PID=$!
    CURRENT_PGID="$(ps -o pgid= -p "$CURRENT_PID" 2>/dev/null | tr -d ' ' || true)"
    [[ -n "$CURRENT_PGID" ]] || CURRENT_PGID="$CURRENT_PID"
    persist_current running launching

    local banner_started now waited=0 rl_started=0 child_status
    banner_started="$(date +%s)"
    while kill -0 "$CURRENT_PID" 2>/dev/null; do
        if grep -q "Canonical RL run" "$log" 2>/dev/null; then
            rl_started=1
            break
        fi
        now="$(date +%s)"
        waited=$((now - banner_started))
        if (( waited >= BANNER_TIMEOUT )); then
            echo "  RL stage did not start within $(format_duration "$BANNER_TIMEOUT"); stopping." >&2
            kill -TERM "$CURRENT_PID" 2>/dev/null || true
            wait_for_exit "$CURRENT_PID" "$GRACE_SECONDS" || true
            wait "$CURRENT_PID" 2>/dev/null || true
            POINT_STATUS="failed"
            POINT_OUTCOME="rl-banner-timeout"
            POINT_ELAPSED="$elapsed_before"
            return
        fi
        sleep "$POLL_SECONDS"
        persist_current running waiting-for-rl
    done

    if (( ! rl_started )); then
        if wait "$CURRENT_PID" 2>/dev/null; then child_status=0; else child_status=$?; fi
        POINT_STATUS="failed"
        POINT_OUTCOME="exited-before-rl-$child_status"
        POINT_ELAPSED="$elapsed_before"
        return
    fi

    CURRENT_RL_STARTED_AT="$(date +%s)"
    persist_current running training
    echo "  RL stage started after $(format_duration "$waited"); $(format_duration "$elapsed_before") already used of $(format_duration "$budget")."

    while kill -0 "$CURRENT_PID" 2>/dev/null; do
        POINT_ELAPSED="$(current_elapsed)"
        (( POINT_ELAPSED >= budget )) && break
        sleep "$POLL_SECONDS"
        persist_current running training
    done

    POINT_ELAPSED="$(current_elapsed)"
    if ! kill -0 "$CURRENT_PID" 2>/dev/null; then
        if wait "$CURRENT_PID" 2>/dev/null; then child_status=0; else child_status=$?; fi
        POINT_STATUS=$([[ "$child_status" == 0 ]] && echo interrupted || echo failed)
        POINT_OUTCOME="exited-before-budget-$child_status"
        return
    fi

    echo "  Budget reached; requesting a graceful stop at the next iteration boundary."
    kill -TERM "$CURRENT_PID" 2>/dev/null || true
    if wait_for_exit "$CURRENT_PID" "$GRACE_SECONDS"; then
        wait "$CURRENT_PID" 2>/dev/null || true
        POINT_STATUS="completed"
        POINT_OUTCOME="budget-graceful-checkpoint"
        POINT_ELAPSED="$(current_elapsed)"
        return
    fi

    echo "  Still running after $(format_duration "$GRACE_SECONDS"); interrupting."
    kill -TERM "$CURRENT_PID" 2>/dev/null || true
    if wait_for_exit "$CURRENT_PID" "$GRACE_SECONDS"; then
        wait "$CURRENT_PID" 2>/dev/null || true
        POINT_STATUS="completed"
        POINT_OUTCOME="budget-hard-stop"
        POINT_ELAPSED="$(current_elapsed)"
        return
    fi

    echo "  Unresponsive; killing the process group."
    kill -KILL "-$CURRENT_PGID" 2>/dev/null || true
    wait "$CURRENT_PID" 2>/dev/null || true
    POINT_STATUS="completed"
    POINT_OUTCOME="budget-killed"
    POINT_ELAPSED="$(current_elapsed)"
}

mapfile -t ALL_GRID < <(build_grid)
GRID=("${ALL_GRID[@]}")

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

for extra_arg in "${EXTRA_ARGS[@]}"; do
    case "$extra_arg" in
        --resume|--resume=*|--resume-from|--resume-from=*|--restart-rl|--run-name|--run-name=*|--alpha|--alpha=*|--gamma|--gamma=*|--event-reward-decay|--event-reward-decay=*)
            echo "The grid owns resume, run-name, and reward parameters; remove forwarded argument $extra_arg." >&2
            exit 1
            ;;
    esac
done

echo "======================================================================"
echo "Reward-shaping grid search - old notebook"
echo "======================================================================"
echo "Interpreter:      $PYTHON_BIN"
echo "Points selected:  ${#GRID[@]}"
echo "New-point budget: $(format_duration "$RL_TIME_LIMIT")"
echo "Shutdown grace:   $(format_duration "$GRACE_SECONDS") per escalation step"
echo "Results:          $RESULTS_DIR"

if (( DRY_RUN )); then
    echo "Dry run; commands that would be issued for a fresh grid:"
    for row in "${GRID[@]}"; do
        read -r alpha gamma decay name <<<"$row"
        echo "  $PYTHON_BIN -u -m training.pipeline forever --alpha $alpha" \
             "--gamma $gamma --event-reward-decay $decay --run-name $name" \
             "${EXTRA_ARGS[*]+${EXTRA_ARGS[*]}}"
    done
    exit 0
fi

mkdir -p "$RESULTS_DIR"
if ! command -v flock >/dev/null 2>&1; then
    echo "run_reward_grid_search.sh requires flock to prevent concurrent sweeps." >&2
    exit 1
fi
exec 9>"$RESULTS_DIR/.grid_search.lock"
if ! flock -n 9; then
    echo "Another reward grid search is already using $RESULTS_DIR." >&2
    exit 1
fi

migrate_or_create_state
ensure_state_rows
recover_stale_running_rows
apply_explicit_budget_to_pending
if (( FORCE )); then
    reset_selected_for_force
fi
report_state_summary
trap cleanup EXIT INT TERM

index=0
for row in "${GRID[@]}"; do
    read -r alpha gamma decay name <<<"$row"
    index=$((index + 1))
    saved="$(state_row "$name")"
    IFS=$'\t' read -r _ status _ _ _ budget elapsed attempts _ _ outcome _ <<<"$saved"

    if [[ "$status" == completed ]]; then
        continue
    fi
    if (( elapsed >= budget )); then
        state_upsert "$name" completed "$alpha" "$gamma" "$decay" \
            "$budget" "$elapsed" "$attempts" - - recovered-complete "$(date -Is)"
        continue
    fi

    run_dir="$(run_directory_for "$name")"
    mode=fresh
    if (( FORCE )) && [[ -d "$run_dir" ]]; then
        mode=restart
    elif [[ -f "$run_dir/training_state.json" ]]; then
        mode=resume
    elif [[ "$status" == interrupted || "$status" == failed ]] && (( elapsed > 0 )); then
        echo "  $name has no resumable checkpoint; restarting its full budget."
        elapsed=0
        budget="$RL_TIME_LIMIT"
        mode=$([[ -d "$run_dir" ]] && echo restart || echo fresh)
    fi

    attempt=$((attempts + 1))
    log="$RESULTS_DIR/$name.log"
    remaining=$((budget - elapsed))
    echo
    echo "[$index/${#GRID[@]}] $name: $mode, $(format_duration "$remaining") remaining"
    echo "  alpha=$alpha gamma=$gamma event-reward-decay=$decay"
    echo "  Log: $log"
    run_point "$alpha" "$gamma" "$decay" "$name" "$log" \
        "$budget" "$elapsed" "$attempt" "$mode"
    state_upsert "$name" "$POINT_STATUS" "$alpha" "$gamma" "$decay" \
        "$budget" "$POINT_ELAPSED" "$attempt" - - "$POINT_OUTCOME" "$(date -Is)"
    echo "  $POINT_STATUS: $(format_duration "$POINT_ELAPSED") / $(format_duration "$budget") RL time."
    clear_current
    if [[ "$POINT_STATUS" != completed ]]; then
        echo "Point $name did not complete ($POINT_OUTCOME). Run this same command to retry or resume it." >&2
        exit 1
    fi
done

echo
report_state_summary
echo "State: $STATE_FILE"
