#!/usr/bin/env bash
#
# Reward-distance grid search for the double-four ruleset.
#
# The grid contains all four public `--reward-distance-mode` values and two
# deterministic seeds, for eight independent `training.pipeline forever` runs.
# The two seeds are deliberately different from the canonical seed 42. Each
# seed builds its canonical dataset and supervised checkpoint once; the four
# reward-distance runs with that seed reuse those exact assets.
#
# `forever` has no game target, so the timer starts only when the pipeline
# prints its "Canonical RL run" banner. Dataset generation and supervised
# training are therefore outside the 1h30 RL budget. At the deadline the script
# first requests the pipeline's graceful iteration-boundary checkpoint, then
# escalates only if the configured grace periods expire.
#
# Completed points are recorded in grid_state.tsv and skipped when the script
# is run again. A point interrupted before it was recorded is resumed
# automatically by the canonical pipeline from its existing run directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RL_TIME_LIMIT=5400
GRACE_SECONDS=900
BANNER_TIMEOUT=21600
POLL_SECONDS=5
RULESET="double-four"
REWARD_DISTANCE_MODES=(
    turn-turn
    decision-decision
    turn-decision
    decision-turn
)
RL_SEEDS=(137 271)
RESULTS_DIR="$REPO_ROOT/train_script/grid_search_results/double_four_reward_distance"
DRY_RUN=0
FORCE=0
ONLY_PATTERN=""
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage: train_script/run_grid_search.sh [options] [-- extra pipeline args]

Run all four reward-distance modes with seeds 137 and 271 on double-four.
Each seed's dataset and supervised checkpoint are generated once and reused by
its four RL runs.

Options:
  --time-limit DURATION   RL wall clock per point; accepts plain seconds or a
                          30m/2h/1d suffix (default: 1h30m)
  --grace DURATION        Wait after each shutdown escalation step (default:
                          15m). Keep this above one RL iteration.
  --results-dir DIR       Logs and sweep state (default: $RESULTS_DIR)
  --only PATTERN          Run only matching run names, for example
                          --only 'turn_turn_*' or --only '*_seed137'
  --force                 Give completed points another full RL budget
  --dry-run               Print the plan and exact commands without running
  -h, --help              Show this help

Environment:
  PYTHON                  Interpreter to use. Defaults to
                          $REPO_ROOT/.venv/bin/python when present, else python3.

Examples:
  train_script/run_grid_search.sh
  train_script/run_grid_search.sh --dry-run
  train_script/run_grid_search.sh --time-limit 5m --grace 2m --only 'turn_turn_*'
  train_script/run_grid_search.sh -- --rl-workers 4
EOF
}

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
    printf '%dh%02dm%02ds' \
        $((total / 3600)) $((total % 3600 / 60)) $((total % 60))
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

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "No interpreter found; set PYTHON to the project interpreter." >&2
    exit 1
fi

mode_label() {
    local mode="$1"
    echo "${mode//-/_}"
}

# Emit one "mode seed run_name" row. Seed-major ordering ensures that the
# first point builds one seed's assets and its next three points immediately
# reuse them before the second seed is prepared.
build_grid() {
    local seed mode name
    for seed in "${RL_SEEDS[@]}"; do
        for mode in "${REWARD_DISTANCE_MODES[@]}"; do
            name="$(mode_label "$mode")_seed${seed}"
            echo "$mode $seed $name"
        done
    done
}

STATE_FILE="$RESULTS_DIR/grid_state.tsv"
CURRENT_PID=""
CURRENT_PGID=""

wait_for_exit() {
    local pid="$1" limit="$2" waited=0
    while (( waited < limit )); do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep "$POLL_SECONDS"
        waited=$((waited + POLL_SECONDS))
    done
    ! kill -0 "$pid" 2>/dev/null
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ -n "$CURRENT_PID" ]] && kill -0 "$CURRENT_PID" 2>/dev/null; then
        echo >&2
        echo "Grid interrupted; asking the running point to checkpoint." >&2
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
        $1 == want && ($2 == "graceful" || $2 == "finished") {
            found = 1
        }
        END { exit !found }
    ' "$STATE_FILE"
}

record_point() {
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$1" "$2" "$3" "$4" "$(date -Is)" >>"$STATE_FILE"
}

run_point() {
    local mode="$1" seed="$2" name="$3" log="$4"
    local cmd=(
        "$PYTHON_BIN" -u -m training.pipeline forever
    )
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi
    # Grid identity comes last so forwarded resource/reporting controls cannot
    # accidentally replace the ruleset, seed, mode, or run name.
    cmd+=(
        --ruleset "$RULESET"
        --seed "$seed"
        --reward-distance-mode "$mode"
        --run-name "$name"
    )

    setsid "${cmd[@]}" >"$log" 2>&1 &
    local pid=$!
    local pgid
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    [[ -n "$pgid" ]] || pgid="$pid"
    CURRENT_PID="$pid"
    CURRENT_PGID="$pgid"

    local waited=0 rl_started=0
    while kill -0 "$pid" 2>/dev/null; do
        if grep -q "Canonical RL run" "$log" 2>/dev/null; then
            rl_started=1
            break
        fi
        if (( waited >= BANNER_TIMEOUT )); then
            echo "  RL did not start within $(format_duration "$BANNER_TIMEOUT")." >&2
            break
        fi
        sleep "$POLL_SECONDS"
        waited=$((waited + POLL_SECONDS))
    done

    if (( rl_started )); then
        echo "  RL started after $(format_duration "$waited"); " \
             "budget $(format_duration "$RL_TIME_LIMIT") begins now." >&2
    fi

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

    echo "  Budget reached; requesting a graceful boundary checkpoint." >&2
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

mapfile -t GRID < <(build_grid)

if [[ -n "$ONLY_PATTERN" ]]; then
    FILTERED=()
    for row in "${GRID[@]}"; do
        read -r _mode _seed name <<<"$row"
        # Unquoted deliberately: ONLY_PATTERN is a shell pattern.
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
echo "Reward-distance grid search"
echo "======================================================================"
echo "Interpreter:       $PYTHON_BIN"
echo "Ruleset:           $RULESET"
echo "Points:            ${#GRID[@]}"
echo "Seeds:             ${RL_SEEDS[*]}"
echo "Assets:            one dataset + SL checkpoint per seed"
echo "RL cap per point:  $(format_duration "$RL_TIME_LIMIT")"
echo "Shutdown grace:    $(format_duration "$GRACE_SECONDS") per step"
echo "Results:           $RESULTS_DIR"
echo "Nominal RL total:  $(format_duration $(( ${#GRID[@]} * RL_TIME_LIMIT )))"
echo "----------------------------------------------------------------------"
printf '%-31s %-22s %-10s\n' "RUN NAME" "DISTANCE MODE" "SEED"
for row in "${GRID[@]}"; do
    read -r mode seed name <<<"$row"
    printf '%-31s %-22s %-10s\n' "$name" "$mode" "$seed"
done
echo "----------------------------------------------------------------------"

if (( DRY_RUN )); then
    echo "Dry run; commands that would be issued:"
    for row in "${GRID[@]}"; do
        read -r mode seed name <<<"$row"
        printf '  %q -u -m training.pipeline forever' "$PYTHON_BIN"
        if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
            printf ' %q' "${EXTRA_ARGS[@]}"
        fi
        printf ' --ruleset %q --seed %q --reward-distance-mode %q --run-name %q' \
            "$RULESET" "$seed" "$mode" "$name"
        printf '\n'
    done
    exit 0
fi

mkdir -p "$RESULTS_DIR"
if [[ ! -f "$STATE_FILE" ]]; then
    printf 'run_name\tstatus\treward_distance_mode\tseed\tfinished_at\n' \
        >"$STATE_FILE"
fi

index=0
for row in "${GRID[@]}"; do
    read -r mode seed name <<<"$row"
    index=$((index + 1))

    if (( ! FORCE )) && point_completed "$name"; then
        echo "[$index/${#GRID[@]}] $name -- completed, skipping (--force to redo)"
        continue
    fi

    log="$RESULTS_DIR/$name.log"
    echo
    echo "[$index/${#GRID[@]}] $name  mode=$mode seed=$seed"
    echo "  Log: $log"
    started=$(date +%s)
    POINT_STATUS="failed"
    run_point "$mode" "$seed" "$name" "$log"
    elapsed=$(( $(date +%s) - started ))
    echo "  Status: $POINT_STATUS after $(format_duration "$elapsed")"
    record_point "$name" "$POINT_STATUS" "$mode" "$seed"
done

echo
echo "======================================================================"
echo "Grid complete"
echo "======================================================================"
echo "State: $STATE_FILE"
echo "Logs:  $RESULTS_DIR"
echo "Run directories: models/rl/domino_rl_double-four_forever_seed*_run*"
echo
column -t -s $'\t' "$STATE_FILE" 2>/dev/null || cat "$STATE_FILE"
