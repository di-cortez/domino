# Shared implementation for machine-specific bucket, PPO-LR, and baseline sequences.
#
# This file is sourced by the public wrappers beside it.  It deliberately
# has no shebang and is not an entry point: keeping the lifecycle, resume state,
# signal handling, and wall-clock accounting here prevents the machine profiles
# from drifting apart.

run_rl_experiment_sequence() {
    : "${MACHINE_SLUG:?MACHINE_SLUG must be set by the wrapper}"
    : "${MACHINE_LABEL:?MACHINE_LABEL must be set by the wrapper}"
    : "${TIME_COEFFICIENT:?TIME_COEFFICIENT must be set by the wrapper}"
    : "${EXPERIMENT_KIND:?EXPERIMENT_KIND must be set by the wrapper}"
    : "${RULESET:?RULESET must be set by the wrapper}"
    : "${RL_TIME_LIMIT:?RL_TIME_LIMIT must be set by the wrapper}"

    local script_dir repo_root
    script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
    repo_root="$(cd "$script_dir/.." && pwd)"
    cd "$repo_root"

    local grace_seconds=900
    local banner_timeout="${SEQUENCE_BANNER_TIMEOUT_SECONDS:-21600}"
    local poll_seconds="${SEQUENCE_POLL_SECONDS:-5}"
    local results_dir="${SEQUENCE_RESULTS_DIR:-$repo_root/train_script/grid_search_results/$MACHINE_SLUG/$EXPERIMENT_KIND}"
    local run_root="${SEQUENCE_RUN_ROOT:-$repo_root/models/rl}"
    local dry_run=0
    local force=0
    local time_limit_explicit=0
    local only_pattern=""
    local -a extra_args=()

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

    usage() {
        local point_count point_description
        case "$EXPERIMENT_KIND" in
            buckets)
                point_count=3
                point_description="heuristic+recent, all buckets except random, and heuristic only"
                ;;
            ppo_lr)
                point_count=5
                point_description="normal, 2x, 4x, 8x, and 16x PPO learning rate"
                ;;
            baselines)
                point_count=6
                point_description="zero, constants +5/-5, and the three critic layouts at PPO LR 0.01"
                ;;
            *)
                echo "Unsupported EXPERIMENT_KIND: $EXPERIMENT_KIND" >&2
                return 1
                ;;
        esac
        cat <<EOF
Usage: ${BASH_SOURCE[1]#"$repo_root/"} [options] [-- extra pipeline args]

Run the $point_count $EXPERIMENT_KIND experiments for $MACHINE_LABEL
sequentially: $point_description.  Each point receives
$(format_duration "$RL_TIME_LIMIT") of RL wall-clock time by default.

Options:
  --time-limit DURATION   Override the RL wall clock per point; accepts seconds
                          or an s/m/h/d suffix
  --grace DURATION        Wait after each shutdown escalation step
                          (default: $(format_duration "$grace_seconds"))
  --results-dir DIR       Override logs and sequence state directory
  --only PATTERN          Select run names matching a shell pattern
  --force                 Restart selected points with a fresh full budget
  --dry-run               Print the plan and exact commands, run nothing
  -h, --help              Show this help

Running the same wrapper again skips completed points and resumes an interrupted
point from its exact canonical checkpoint with only its unused time budget.

Environment:
  PYTHON                  Interpreter to use; defaults to .venv/bin/python
  SEQUENCE_RUN_ROOT       Override the canonical models/rl root
  SEQUENCE_POLL_SECONDS   State-update polling interval (default: 5)
  SEQUENCE_BANNER_TIMEOUT_SECONDS
                          Maximum wait for dataset/SL preparation (default: 21600)
EOF
    }

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --time-limit)
                RL_TIME_LIMIT="$(parse_duration "$2")"
                time_limit_explicit=1
                shift 2
                ;;
            --grace)
                grace_seconds="$(parse_duration "$2")"
                shift 2
                ;;
            --results-dir)
                results_dir="$2"
                shift 2
                ;;
            --only)
                only_pattern="$2"
                shift 2
                ;;
            --force) force=1; shift ;;
            --dry-run) dry_run=1; shift ;;
            -h|--help) usage; return 0 ;;
            --) shift; extra_args=("$@"); break ;;
            *) echo "Unknown option: $1" >&2; usage >&2; return 1 ;;
        esac
    done

    if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ||
          ! "$banner_timeout" =~ ^[1-9][0-9]*$ ]]; then
        echo "SEQUENCE_POLL_SECONDS and SEQUENCE_BANNER_TIMEOUT_SECONDS must be positive integers." >&2
        return 1
    fi

    local python_bin
    if [[ -n "${PYTHON:-}" ]]; then
        python_bin="$PYTHON"
    elif [[ -x "$repo_root/.venv/bin/python" ]]; then
        python_bin="$repo_root/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        python_bin="python3"
    else
        echo "No interpreter found; set PYTHON to the project interpreter." >&2
        return 1
    fi

    local -a all_points=()
    if [[ "$EXPERIMENT_KIND" == "buckets" ]]; then
        all_points=(
            "heuristic_recent heuristic,recent bucket_heuristic_recent_${MACHINE_SLUG}"
            "all_except_random heuristic,recent,medium_term,historical_uniform,champion_vs_heuristic,champion_vs_learner bucket_all_except_random_${MACHINE_SLUG}"
            "heuristic_only heuristic bucket_heuristic_only_${MACHINE_SLUG}"
        )
    elif [[ "$EXPERIMENT_KIND" == "ppo_lr" ]]; then
        all_points=(
            "normal 0.001 ppo_lr_1x_${MACHINE_SLUG}"
            "2x 0.002 ppo_lr_2x_${MACHINE_SLUG}"
            "4x 0.004 ppo_lr_4x_${MACHINE_SLUG}"
            "8x 0.008 ppo_lr_8x_${MACHINE_SLUG}"
            "16x 0.016 ppo_lr_16x_${MACHINE_SLUG}"
        )
    elif [[ "$EXPERIMENT_KIND" == "baselines" ]]; then
        all_points=(
            "zero zero baseline_zero_${MACHINE_SLUG}"
            "constant_plus_5 5 baseline_constant_plus_5_${MACHINE_SLUG}"
            "constant_minus_5 -5 baseline_constant_minus_5_${MACHINE_SLUG}"
            "critic_separate_network value-head-own-nn baseline_value_head_own_nn_${MACHINE_SLUG}"
            "critic_updates_shared_trunk value-head baseline_value_head_${MACHINE_SLUG}"
            "critic_head_only value-head-no-up baseline_value_head_no_up_${MACHINE_SLUG}"
        )
    else
        echo "Unsupported EXPERIMENT_KIND: $EXPERIMENT_KIND" >&2
        return 1
    fi

    local -a points=("${all_points[@]}")
    if [[ -n "$only_pattern" ]]; then
        local -a filtered=()
        local row label value name
        for row in "${points[@]}"; do
            read -r label value name <<<"$row"
            # Unquoted intentionally: only_pattern is a shell pattern.
            # shellcheck disable=SC2053
            if [[ "$name" == $only_pattern ]]; then
                filtered+=("$row")
            fi
        done
        points=("${filtered[@]}")
    fi
    if [[ ${#points[@]} -eq 0 ]]; then
        echo "No experiment points selected." >&2
        return 1
    fi

    local extra_arg
    for extra_arg in "${extra_args[@]}"; do
        case "$extra_arg" in
            --resume|--resume=*|--resume-from|--resume-from=*|--restart-rl|\
            --run-name|--run-name=*|--ruleset|--ruleset=*|--seed|--seed=*|\
            --artifact-root|--artifact-root=*|--learning-rate|--learning-rate=*|\
            --opponent-buckets|--opponent-buckets=*|--baseline|--baseline=*|\
            --value-head)
                echo "The sequence owns identity and tested parameters; remove forwarded argument $extra_arg." >&2
                return 1
                ;;
        esac
    done

    local state_file="$results_dir/sequence_state.tsv"
    local state_header=$'run_name\tstatus\texperiment\tparameter_label\tparameter_value\truleset\tbudget_seconds\trl_elapsed_seconds\tattempts\tpid\tpgid\tlast_outcome\tupdated_at'

    state_row() {
        local wanted="$1"
        awk -F'\t' -v want="$wanted" \
            '$1 == want { print; found = 1; exit } END { exit !found }' \
            "$state_file"
    }

    state_upsert() {
        local name="$1" status="$2" label="$3" value="$4"
        local budget="$5" elapsed="$6" attempts="$7" pid="$8"
        local pgid="$9" outcome="${10}" updated_at="${11}" temporary
        temporary="$(mktemp "$results_dir/.sequence_state.XXXXXX")"
        awk -F'\t' -v OFS='\t' \
            -v name="$name" -v status="$status" -v experiment="$EXPERIMENT_KIND" \
            -v label="$label" -v value="$value" -v ruleset="$RULESET" \
            -v budget="$budget" -v elapsed="$elapsed" -v attempts="$attempts" \
            -v pid="$pid" -v pgid="$pgid" -v outcome="$outcome" \
            -v updated="$updated_at" '
            NR == 1 { print; next }
            $1 == name {
                print name, status, experiment, label, value, ruleset, budget,
                      elapsed, attempts, pid, pgid, outcome, updated
                found = 1
                next
            }
            { print }
            END {
                if (!found) {
                    print name, status, experiment, label, value, ruleset,
                          budget, elapsed, attempts, pid, pgid, outcome, updated
                }
            }
        ' "$state_file" >"$temporary"
        chmod 0644 "$temporary"
        mv "$temporary" "$state_file"
    }

    initialize_state() {
        local header
        if [[ ! -f "$state_file" ]]; then
            printf '%s\n' "$state_header" >"$state_file"
            return
        fi
        IFS= read -r header <"$state_file"
        if [[ "$header" != "$state_header" ]]; then
            echo "Unsupported sequence state schema in $state_file" >&2
            exit 1
        fi
    }

    ensure_state_rows() {
        local row label value name
        for row in "${all_points[@]}"; do
            read -r label value name <<<"$row"
            if ! state_row "$name" >/dev/null 2>&1; then
                state_upsert "$name" pending "$label" "$value" \
                    "$RL_TIME_LIMIT" 0 0 - - never-started "$(date -Is)"
            fi
        done
    }

    run_directory_for() {
        local name="$1" ruleset_part=""
        if [[ "$RULESET" != "double-six" ]]; then
            ruleset_part="${RULESET}_"
        fi
        printf '%s/domino_rl_%sforever_seed42_run%s\n' \
            "$run_root" "$ruleset_part" "$name"
    }

    local current_name=""
    local current_label=""
    local current_value=""
    local current_budget=0
    local current_elapsed_base=0
    local current_attempt=0
    local current_pid=""
    local current_pgid=""
    local current_rl_started_at=0
    local point_status="failed"
    local point_outcome="not-started"
    local point_elapsed=0

    current_elapsed() {
        local elapsed="$current_elapsed_base"
        if (( current_rl_started_at > 0 )); then
            elapsed=$((elapsed + $(date +%s) - current_rl_started_at))
        fi
        printf '%s\n' "$elapsed"
    }

    persist_current() {
        local status="$1" outcome="$2" elapsed pid pgid
        [[ -n "$current_name" ]] || return 0
        elapsed="$(current_elapsed)"
        pid="${current_pid:--}"
        pgid="${current_pgid:--}"
        state_upsert "$current_name" "$status" "$current_label" \
            "$current_value" "$current_budget" "$elapsed" \
            "$current_attempt" "$pid" "$pgid" "$outcome" "$(date -Is)"
    }

    wait_for_exit() {
        local pid="$1" limit="$2" waited=0
        while (( waited < limit )); do
            kill -0 "$pid" 2>/dev/null || return 0
            sleep "$poll_seconds"
            waited=$((waited + poll_seconds))
            persist_current running stopping
        done
        ! kill -0 "$pid" 2>/dev/null
    }

    clear_current() {
        current_name=""
        current_label=""
        current_value=""
        current_budget=0
        current_elapsed_base=0
        current_attempt=0
        current_pid=""
        current_pgid=""
        current_rl_started_at=0
    }

    cleanup_sequence() {
        local status=$? outcome="interrupted-before-process-start" elapsed
        trap - EXIT INT TERM
        set +e
        if [[ -n "${current_name:-}" ]]; then
            if [[ -n "$current_pid" ]] && kill -0 "$current_pid" 2>/dev/null; then
                echo >&2
                echo "Sequence interrupted; asking $current_name to checkpoint and stop." >&2
                kill -TERM "$current_pid" 2>/dev/null
                if wait_for_exit "$current_pid" "$grace_seconds"; then
                    wait "$current_pid" 2>/dev/null
                    outcome="interrupted-graceful-checkpoint"
                else
                    echo "  Still running; killing the process group." >&2
                    kill -KILL "-$current_pgid" 2>/dev/null
                    wait "$current_pid" 2>/dev/null
                    outcome="interrupted-hard-stop"
                fi
            fi
            elapsed="$(current_elapsed)"
            state_upsert "$current_name" interrupted "$current_label" \
                "$current_value" "$current_budget" "$elapsed" \
                "$current_attempt" - - "$outcome" "$(date -Is)"
            echo "  Saved progress: $(format_duration "$elapsed") / $(format_duration "$current_budget")." >&2
        fi
        exit "$status"
    }

    process_matches_point() {
        local pid="$1" name="$2" command run_dir
        [[ "$pid" =~ ^[0-9]+$ ]] || return 1
        kill -0 "$pid" 2>/dev/null || return 1
        command="$(ps -o args= -p "$pid" 2>/dev/null || true)"
        run_dir="$(run_directory_for "$name")"
        [[ "$command" == *"training.pipeline forever"* ]] || return 1
        [[ "$command" == *"--run-name $name"* ||
           "$command" == *"--resume $run_dir"* ]]
    }

    recover_stale_running_rows() {
        local name saved status _experiment label value ruleset budget elapsed
        local attempts pid pgid outcome updated
        local -a running_names=()
        mapfile -t running_names < <(
            awk -F'\t' 'NR > 1 && $2 == "running" { print $1 }' "$state_file"
        )
        for name in "${running_names[@]}"; do
            saved="$(state_row "$name")"
            IFS=$'\t' read -r _ status _experiment label value ruleset budget \
                elapsed attempts pid pgid outcome updated <<<"$saved"
            if process_matches_point "$pid" "$name"; then
                echo "Point $name is still running as PID $pid; refusing a duplicate sequence." >&2
                exit 1
            fi
            state_upsert "$name" interrupted "$label" "$value" "$budget" \
                "$elapsed" "$attempts" - - stale-run-recovered "$(date -Is)"
        done
    }

    reset_selected_for_force() {
        local row label value name
        for row in "${points[@]}"; do
            read -r label value name <<<"$row"
            state_upsert "$name" pending "$label" "$value" \
                "$RL_TIME_LIMIT" 0 0 - - force-reset "$(date -Is)"
        done
    }

    apply_explicit_budget_to_pending() {
        local row label value name saved status budget elapsed attempts
        (( time_limit_explicit )) || return 0
        for row in "${points[@]}"; do
            read -r label value name <<<"$row"
            saved="$(state_row "$name")"
            IFS=$'\t' read -r _ status _ _ _ _ budget elapsed attempts _ _ _ _ \
                <<<"$saved"
            if [[ "$status" == pending && "$elapsed" == 0 ]]; then
                state_upsert "$name" pending "$label" "$value" \
                    "$RL_TIME_LIMIT" 0 "$attempts" - - budget-set "$(date -Is)"
            fi
        done
    }

    report_state_summary() {
        local row name saved status budget elapsed remaining
        local completed=0 resumable=0 pending=0 failed=0
        local -a completed_names=() remaining_names=()
        for row in "${points[@]}"; do
            read -r _ _ name <<<"$row"
            saved="$(state_row "$name")"
            IFS=$'\t' read -r _ status _ _ _ _ budget elapsed _ _ _ _ _ \
                <<<"$saved"
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
        echo "Sequence state: $completed/${#points[@]} completed; $resumable resumable; $pending pending; $failed failed."
        if (( completed > 0 )); then
            echo "Completed: ${completed_names[*]}"
        fi
        if (( ${#remaining_names[@]} > 0 )); then
            echo "Remaining: ${remaining_names[*]}"
        fi
        for name in "${remaining_names[@]}"; do
            saved="$(state_row "$name")"
            IFS=$'\t' read -r _ status _ _ _ _ budget elapsed _ _ _ _ _ \
                <<<"$saved"
            if [[ "$status" == interrupted || "$status" == failed ]]; then
                remaining=$((budget > elapsed ? budget - elapsed : 0))
                echo "Next resumable $name: $(format_duration "$elapsed") used, $(format_duration "$remaining") remaining."
                break
            fi
        done
    }

    append_tested_parameter() {
        local -n command_ref="$1"
        local value="$2"
        if [[ "$EXPERIMENT_KIND" == "buckets" ]]; then
            command_ref+=(--opponent-buckets "$value")
        elif [[ "$EXPERIMENT_KIND" == "ppo_lr" ]]; then
            command_ref+=(--learning-rate "$value")
        elif [[ "$EXPERIMENT_KIND" == "baselines" ]]; then
            command_ref+=(--learning-rate 0.01)
            case "$value" in
                zero|5|-5)
                    command_ref+=(--baseline "$value")
                    ;;
                value-head|value-head-no-up|value-head-own-nn)
                    command_ref+=(--value-head --baseline "$value")
                    ;;
                *)
                    echo "Unsupported baseline experiment value: $value" >&2
                    return 1
                    ;;
            esac
        else
            echo "Unsupported EXPERIMENT_KIND: $EXPERIMENT_KIND" >&2
            return 1
        fi
    }

    print_command() {
        local -a command=("$@")
        printf '  '
        printf '%q ' "${command[@]}"
        printf '\n'
    }

    run_point() {
        local label="$1" value="$2" name="$3" log="$4"
        local budget="$5" elapsed_before="$6" attempt="$7" mode="$8"
        local run_dir
        run_dir="$(run_directory_for "$name")"
        local -a command
        if [[ "$mode" == "resume" ]]; then
            command=("$python_bin" -u -m training.pipeline forever --resume "$run_dir")
        else
            command=(
                "$python_bin" -u -m training.pipeline forever
                --ruleset "$RULESET"
                --run-name "$name"
            )
            append_tested_parameter command "$value"
            if [[ "$mode" == "restart" ]]; then
                command+=(--restart-rl)
            fi
            if [[ ${#extra_args[@]} -gt 0 ]]; then
                command+=("${extra_args[@]}")
            fi
        fi

        current_name="$name"
        current_label="$label"
        current_value="$value"
        current_budget="$budget"
        current_elapsed_base="$elapsed_before"
        current_attempt="$attempt"
        current_rl_started_at=0
        point_status="failed"
        point_outcome="launch-failed"
        point_elapsed="$elapsed_before"

        setsid "${command[@]}" >"$log" 2>&1 &
        current_pid=$!
        current_pgid="$(ps -o pgid= -p "$current_pid" 2>/dev/null | tr -d ' ' || true)"
        [[ -n "$current_pgid" ]] || current_pgid="$current_pid"
        persist_current running launching

        local banner_started now waited=0 rl_started=0 child_status
        banner_started="$(date +%s)"
        while kill -0 "$current_pid" 2>/dev/null; do
            if grep -q "Canonical RL run" "$log" 2>/dev/null; then
                rl_started=1
                break
            fi
            now="$(date +%s)"
            waited=$((now - banner_started))
            if (( waited >= banner_timeout )); then
                echo "  RL stage did not start within $(format_duration "$banner_timeout"); stopping." >&2
                kill -TERM "$current_pid" 2>/dev/null || true
                wait_for_exit "$current_pid" "$grace_seconds" || true
                wait "$current_pid" 2>/dev/null || true
                point_status="failed"
                point_outcome="rl-banner-timeout"
                point_elapsed="$elapsed_before"
                return
            fi
            sleep "$poll_seconds"
            persist_current running waiting-for-rl
        done

        if (( ! rl_started )); then
            if wait "$current_pid" 2>/dev/null; then
                child_status=0
            else
                child_status=$?
            fi
            point_status="failed"
            point_outcome="exited-before-rl-$child_status"
            point_elapsed="$elapsed_before"
            return
        fi

        current_rl_started_at="$(date +%s)"
        persist_current running training
        echo "  RL stage started after $(format_duration "$waited"); $(format_duration "$elapsed_before") already used of $(format_duration "$budget")."

        while kill -0 "$current_pid" 2>/dev/null; do
            point_elapsed="$(current_elapsed)"
            (( point_elapsed >= budget )) && break
            sleep "$poll_seconds"
            persist_current running training
        done

        point_elapsed="$(current_elapsed)"
        if ! kill -0 "$current_pid" 2>/dev/null; then
            if wait "$current_pid" 2>/dev/null; then
                child_status=0
            else
                child_status=$?
            fi
            if [[ "$child_status" == 0 ]]; then
                point_status="interrupted"
            else
                point_status="failed"
            fi
            point_outcome="exited-before-budget-$child_status"
            return
        fi

        echo "  Budget reached; requesting a graceful stop at the next iteration boundary."
        kill -TERM "$current_pid" 2>/dev/null || true
        if wait_for_exit "$current_pid" "$grace_seconds"; then
            wait "$current_pid" 2>/dev/null || true
            point_status="completed"
            point_outcome="budget-graceful-checkpoint"
            point_elapsed="$(current_elapsed)"
            return
        fi

        echo "  Still running after $(format_duration "$grace_seconds"); interrupting."
        kill -TERM "$current_pid" 2>/dev/null || true
        if wait_for_exit "$current_pid" "$grace_seconds"; then
            wait "$current_pid" 2>/dev/null || true
            point_status="completed"
            point_outcome="budget-hard-stop"
            point_elapsed="$(current_elapsed)"
            return
        fi

        echo "  Unresponsive; killing the process group."
        kill -KILL "-$current_pgid" 2>/dev/null || true
        wait "$current_pid" 2>/dev/null || true
        point_status="completed"
        point_outcome="budget-killed"
        point_elapsed="$(current_elapsed)"
    }

    echo "======================================================================"
    echo "$EXPERIMENT_KIND sequence - $MACHINE_LABEL"
    echo "======================================================================"
    echo "Interpreter:      $python_bin"
    echo "Ruleset:          $RULESET"
    echo "Time coefficient: $TIME_COEFFICIENT (Diego notebook = 1.0)"
    echo "Points selected:  ${#points[@]}"
    echo "RL budget/point:  $(format_duration "$RL_TIME_LIMIT")"
    echo "Shutdown grace:   $(format_duration "$grace_seconds") per escalation step"
    echo "Results:          $results_dir"

    if (( dry_run )); then
        echo "Dry run; fresh commands:"
        local row label value name
        for row in "${points[@]}"; do
            read -r label value name <<<"$row"
            local -a command=(
                "$python_bin" -u -m training.pipeline forever
                --ruleset "$RULESET"
                --run-name "$name"
            )
            append_tested_parameter command "$value"
            command+=("${extra_args[@]}")
            print_command "${command[@]}"
        done
        return 0
    fi

    mkdir -p "$results_dir"
    if ! command -v flock >/dev/null 2>&1; then
        echo "The sequence runner requires flock to prevent duplicate runs." >&2
        return 1
    fi
    exec 9>"$results_dir/.sequence.lock"
    if ! flock -n 9; then
        echo "Another sequence is already using $results_dir." >&2
        return 1
    fi

    initialize_state
    ensure_state_rows
    recover_stale_running_rows
    apply_explicit_budget_to_pending
    if (( force )); then
        reset_selected_for_force
    fi
    report_state_summary
    trap cleanup_sequence EXIT INT TERM

    local index=0 saved status budget elapsed attempts outcome
    local run_dir mode attempt log remaining
    local row label value name
    for row in "${points[@]}"; do
        read -r label value name <<<"$row"
        index=$((index + 1))
        saved="$(state_row "$name")"
        IFS=$'\t' read -r _ status _ _ _ _ budget elapsed attempts _ _ outcome _ \
            <<<"$saved"

        if [[ "$status" == completed ]]; then
            continue
        fi
        if (( elapsed >= budget )); then
            state_upsert "$name" completed "$label" "$value" "$budget" \
                "$elapsed" "$attempts" - - recovered-complete "$(date -Is)"
            continue
        fi

        run_dir="$(run_directory_for "$name")"
        mode="fresh"
        if (( force )) && [[ -d "$run_dir" ]]; then
            mode="restart"
        elif [[ -f "$run_dir/training_state.json" ]]; then
            mode="resume"
        elif [[ "$status" == interrupted || "$status" == failed ]] &&
             (( elapsed > 0 )); then
            echo "  $name has no resumable checkpoint; restarting its full budget."
            elapsed=0
            budget="$RL_TIME_LIMIT"
            if [[ -d "$run_dir" ]]; then
                mode="restart"
            else
                mode="fresh"
            fi
        fi

        attempt=$((attempts + 1))
        log="$results_dir/$name.attempt${attempt}.log"
        remaining=$((budget - elapsed))
        echo
        echo "[$index/${#points[@]}] $name: $mode, $(format_duration "$remaining") remaining"
        echo "  $label: $value"
        echo "  Log: $log"
        run_point "$label" "$value" "$name" "$log" "$budget" \
            "$elapsed" "$attempt" "$mode"
        state_upsert "$name" "$point_status" "$label" "$value" "$budget" \
            "$point_elapsed" "$attempt" - - "$point_outcome" "$(date -Is)"
        echo "  $point_status: $(format_duration "$point_elapsed") / $(format_duration "$budget") RL time."
        clear_current
        if [[ "$point_status" != completed ]]; then
            echo "Point $name did not complete ($point_outcome). Run this wrapper again to retry or resume it." >&2
            # This is an ordinary handled failure, not an asynchronous shell
            # exit.  Remove the trap while the function-local state still
            # exists; otherwise `set -u` would make EXIT look up locals that
            # have already gone out of scope after this return.
            trap - EXIT INT TERM
            return 1
        fi
    done

    echo
    report_state_summary
    echo "State: $state_file"
    clear_current
    trap - EXIT INT TERM
}
