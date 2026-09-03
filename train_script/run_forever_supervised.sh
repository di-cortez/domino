#!/usr/bin/env bash
#
# Restart the canonical pipeline after a fatal but recoverable process death.
#
# The motivating failure is a lost CUDA context (see
# references/atualizacoes/atualizacoes_3108/DOMINO_RL_CUDA_CONTEXT_LOSS_RECOVERY_PLAN.md):
# the GPU is reset underneath a healthy process by a driver reload, an Xid
# fault, or a power event. No in-process handler can undo that, because the
# policy weights and the optimizer moments live on the dead device. Resuming
# from the last checkpoint is the whole recovery, and a forever run rehydrates
# its own configuration from the active-run pointer, so re-issuing the same
# command is a complete resume.
#
# Every argument is forwarded verbatim to `python -m training.pipeline`.
#
#   train_script/run_forever_supervised.sh --scale forever --ruleset double-six
#
# Environment:
#   MAX_RESTARTS  restart budget before giving up          (default 100)
#   BACKOFF_S     seconds to wait before each restart      (default 60)
#   LOG_DIR       directory for per-attempt logs           (default logs)
#   PYTHON        interpreter to run                       (default python)

set -u

MAX_RESTARTS="${MAX_RESTARTS:-100}"
BACKOFF_S="${BACKOFF_S:-60}"
LOG_DIR="${LOG_DIR:-logs}"
PYTHON="${PYTHON:-python}"

# Matches GPU_CONTEXT_LOST_EXIT_CODE in training/pipeline.py.
GPU_CONTEXT_LOST_EXIT_CODE=70

mkdir -p "$LOG_DIR"

attempt=0
previous_games=-1

while true; do
    log="${LOG_DIR}/forever_$(date +%Y%m%d_%H%M%S).log"
    echo "[supervisor] attempt ${attempt} -> ${log}"
    "$PYTHON" -u -m training.pipeline "$@" >>"$log" 2>&1
    status=$?

    if (( status == 0 )); then
        echo "[supervisor] pipeline finished cleanly"
        exit 0
    fi

    # SIGINT and SIGTERM are the operator asking the run to stop. Restarting
    # there would make the run impossible to end.
    if (( status == 130 || status == 143 )); then
        echo "[supervisor] interrupted by the operator (exit ${status}); not restarting"
        exit "$status"
    fi

    if (( status == GPU_CONTEXT_LOST_EXIT_CODE )); then
        echo "[supervisor] the CUDA context was lost; the run resumes from its last checkpoint"
    else
        echo "[supervisor] unrecognized exit ${status}; see ${log}"
    fi

    # A restart that makes no progress is a permanent failure wearing a
    # transient's clothes -- a corrupt checkpoint, a GPU that no longer
    # initializes, a full disk. Stop instead of spinning on it.
    # The highest game count the log mentions, from either the progress line
    # ("RL training: 26886000 games") or a resume banner ("... and 26800000
    # real games"). Commas are stripped because the context-loss diagnosis
    # formats its count with thousands separators.
    games="$(grep -oE '[0-9][0-9,]*( real)? games' "$log" \
             | tr -d ', ' | grep -oE '^[0-9]+' | sort -n | tail -1)"
    games="${games:-0}"
    if (( games > 0 && games == previous_games )); then
        echo "[supervisor] no progress since the last restart (${games} games); stopping"
        exit "$status"
    fi
    (( games > 0 )) && previous_games="$games"

    attempt=$(( attempt + 1 ))
    if (( attempt > MAX_RESTARTS )); then
        echo "[supervisor] restart budget of ${MAX_RESTARTS} exhausted"
        exit "$status"
    fi

    # Recorded next to the failure so a recurring fault leaves a temperature
    # and memory trail across attempts.
    command -v nvidia-smi >/dev/null 2>&1 && \
        nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total \
            --format=csv,noheader >>"$log" 2>&1

    echo "[supervisor] restarting in ${BACKOFF_S}s (${attempt}/${MAX_RESTARTS})"
    sleep "$BACKOFF_S"
done
