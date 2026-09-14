# Shared driver for the adaptive one-factor learning-rate search.
#
# Sourced by the `run_lr_factor_search_<machine>.sh` wrappers after they set
# MACHINE_SLUG, MACHINE_LABEL, TIME_COEFFICIENT and RL_TIME_LIMIT. It has no
# shebang and is not an entry point. Optional settings:
#
#   LR_SEARCH_CONTROL_RUN=1      run the project default without `--warmup-lr`
#                                first, outside the decision, on a machine that
#                                has not measured it yet (default 0)
#   LR_SEARCH_REFERENCE_RUN=NAME a completed LR_default warmup run made earlier,
#   LR_SEARCH_REFERENCE_RESULTS_DIR=DIR
#                                with the sequence results directory that ran
#                                it: the search scores it as its LR_default
#                                instead of launching that run again
#   LR_SEARCH_FIRST_ORDINAL=N    bundle number of the first run (default 101)
#
# THE SEARCH
#
# Starting from the project default learning rate, runs are tried in the order
#
#   1. LR   2. 2 LR   3. LR/2   4. 4 LR   5. LR/4   6. 8 LR   7. LR/8
#
# with `--warmup-lr` and every other setting on the project defaults, so only
# the learning rate moves. After each finished run,
# `python -m train_script.lr_factor_search next` scores every finished run by
# its sustained final level -- mean win rate against `random` over the last
# fifth of its horizon, the last hour at the Diego notebook's 5 hours -- and
# replays the decision:
#
#   - a run is worse when it falls more than 0.22 pp (the one-factor sweep's
#     reading ruler) below the best run so far; within the ruler it is a tie;
#   - each direction counts its own worse runs and stops after the second;
#   - the search ends when both directions stop or run out of factors, and the
#     best run tested is the recommended default.
#
# Every warmup run takes part in the decision. The control run, when enabled, is
# the only one outside it, and the search starts right after it.
#
# Every run is launched with `--run-ordinal`, numbered in launch order, so its
# bundle is `<date>-101_<machine>_...`, then 102, and so on; a skipped
# candidate and a reference run take no number.
#
# Each run goes through `_sequential_rl_experiment_runner.bash`, which owns the
# wall-clock budget, graceful stops and exact resume. Everything the search
# decides is recomputed from the finished runs, so re-running the wrapper
# continues an interrupted run and then the search itself.
#
# The summary is rewritten after every decision to
# `grid_search_results/<machine>/lr_factor_search/lr_factor_search_summary.md`.

run_lr_factor_search() {
    : "${MACHINE_SLUG:?MACHINE_SLUG must be set by the wrapper}"
    : "${MACHINE_LABEL:?MACHINE_LABEL must be set by the wrapper}"
    : "${TIME_COEFFICIENT:?TIME_COEFFICIENT must be set by the wrapper}"
    : "${RL_TIME_LIMIT:?RL_TIME_LIMIT must be set by the wrapper}"

    local script_dir repo_root
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "$script_dir/.." && pwd)"
    cd "$repo_root" || return 1

    # The runner names its results directory after EXPERIMENT_KIND; this search
    # keeps its own, so its state never mixes with a `combined` sequence.
    RULESET="double-six"
    EXPERIMENT_KIND="combined"
    local control_run="${LR_SEARCH_CONTROL_RUN:-0}"
    local reference_run="${LR_SEARCH_REFERENCE_RUN:-}"
    local reference_results_dir="${LR_SEARCH_REFERENCE_RESULTS_DIR:-}"
    local first_ordinal="${LR_SEARCH_FIRST_ORDINAL:-101}"
    if [[ "$control_run" != 0 && "$control_run" != 1 ]] ||
       [[ ! "$first_ordinal" =~ ^[0-9]+$ ]]; then
        echo "LR_SEARCH_CONTROL_RUN must be 0 or 1 and LR_SEARCH_FIRST_ORDINAL a number." >&2
        return 1
    fi
    if [[ -n "$reference_run" && -z "$reference_results_dir" ]]; then
        echo "LR_SEARCH_REFERENCE_RUN needs LR_SEARCH_REFERENCE_RESULTS_DIR." >&2
        return 1
    fi
    export SEQUENCE_RESULTS_DIR="${SEQUENCE_RESULTS_DIR:-$repo_root/train_script/grid_search_results/$MACHINE_SLUG/lr_factor_search}"

    local python_bin
    if [[ -n "${PYTHON:-}" ]]; then
        python_bin="$PYTHON"
    elif [[ -x "$repo_root/.venv/bin/python" ]]; then
        python_bin="$repo_root/.venv/bin/python"
    else
        python_bin="python3"
    fi

    local dry_run=0 report_only=0
    local -a runner_args=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run) dry_run=1; shift ;;
            --report) report_only=1; shift ;;
            --only|--only=*|--force)
                echo "The search chooses its runs; $1 is not available here." >&2
                return 1
                ;;
            --run-ordinal|--run-ordinal=*)
                echo "The search numbers its runs from LR_SEARCH_FIRST_ORDINAL; remove $1." >&2
                return 1
                ;;
            -h|--help)
                local setup_text=""
                if (( control_run )); then
                    setup_text+=$'Before it: the project default without --warmup-lr, outside the decision.\n'
                fi
                if [[ -n "$reference_run" ]]; then
                    setup_text+="LR_default is the existing run $reference_run, not launched again."$'\n'
                fi
                cat <<EOF
Usage: ${BASH_SOURCE[1]#"$repo_root/"} [--dry-run] [--report] [runner options] [-- extra pipeline args]

Adaptive one-factor learning-rate search on $MACHINE_LABEL: LR, 2 LR, LR/2,
4 LR, LR/4, 8 LR, LR/8 around the project default, all with --warmup-lr, each
with $((RL_TIME_LIMIT / 3600))h$(printf '%02d' $((RL_TIME_LIMIT % 3600 / 60)))m of RL wall clock.
A direction stops after two runs more than 0.22 pp below the best run so far.
${setup_text}Bundles are numbered in launch order from $first_ordinal.

  --dry-run    Print the plan, the current decisions and the next command
  --report     Print the search summary and exit
  --time-limit, --grace, --results-dir
               Forwarded to the sequence runner

Re-running the wrapper resumes an interrupted run and then the search.
Summary: $SEQUENCE_RESULTS_DIR/lr_factor_search_summary.md
EOF
                return 0
                ;;
            --results-dir)
                SEQUENCE_RESULTS_DIR="$2"
                runner_args+=("$1" "$2")
                shift 2
                ;;
            *) runner_args+=("$1"); shift ;;
        esac
    done

    local -a identity_args=(
        --machine-slug "$MACHINE_SLUG"
        --first-run-ordinal "$first_ordinal"
    )
    (( control_run )) && identity_args+=(--control-run)
    [[ -n "$reference_run" ]] && identity_args+=(--reference-run "$reference_run")
    local -a search_args=(
        "${identity_args[@]}"
        --results-dir "$SEQUENCE_RESULTS_DIR"
        --run-root "${SEQUENCE_RUN_ROOT:-$repo_root/models/rl}"
    )
    [[ -n "$reference_run" ]] &&
        search_args+=(--reference-results-dir "$reference_results_dir")

    if (( dry_run )); then
        echo "Learning-rate search plan for $MACHINE_LABEL:"
        "$python_bin" -m train_script.lr_factor_search plan "${identity_args[@]}" || return 1
        echo
    fi
    if (( report_only )); then
        "$python_bin" -m train_script.lr_factor_search report "${search_args[@]}"
        return $?
    fi

    local next status next_kind next_label next_value next_name next_ordinal
    local previous_name="" argument
    local -a read_only=() point_args=()
    (( dry_run )) && read_only=(--read-only)
    while true; do
        set +e
        next="$("$python_bin" -m train_script.lr_factor_search next "${read_only[@]}" "${search_args[@]}")"
        status=$?
        set -e
        if (( status == 3 )); then
            echo
            "$python_bin" -m train_script.lr_factor_search report "${read_only[@]}" "${search_args[@]}"
            return 0
        fi
        if (( status != 0 )) || [[ -z "$next" ]]; then
            echo "The learning-rate search could not decide its next run." >&2
            return 1
        fi
        read -r next_kind next_label next_value next_name next_ordinal <<<"$next"
        if [[ "$next_name" == "$previous_name" ]]; then
            echo "The search asked for $next_name again after the runner finished it; stopping." >&2
            return 1
        fi
        previous_name="$next_name"
        # One point per runner call: the next point depends on this one's result.
        # The control run is a `one_factor` point (`default`); a
        # search run is a `combined` one (`lr=<rate>+warmup=on`).
        EXPERIMENT_KIND="$next_kind"
        EXPERIMENT_POINTS=("$next_label $next_value $next_name")
        # The bundle number reaches the pipeline as a forwarded argument, after
        # any the operator forwarded; a resumed run keeps the one it started with.
        point_args=("${runner_args[@]}")
        for argument in "${runner_args[@]}"; do
            [[ "$argument" == -- ]] && break
        done
        if [[ "${argument:-}" == -- ]]; then
            point_args+=(--run-ordinal "$next_ordinal")
        else
            point_args+=(-- --run-ordinal "$next_ordinal")
        fi
        EXPERIMENT_POINTS_DESCRIPTION="the next learning rate of the adaptive search"
        if (( dry_run )); then
            "$python_bin" -m train_script.lr_factor_search report --read-only "${search_args[@]}"
            echo
            run_rl_experiment_sequence --dry-run "${point_args[@]}"
            return $?
        fi
        echo
        echo "Learning-rate search: next run $next_name ($next_label: $next_value, Nº $next_ordinal)"
        run_rl_experiment_sequence "${point_args[@]}" || return $?
    done
}
