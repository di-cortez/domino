"""Exact-game-budget reinforcement-learning orchestration.

The policy controls only real tile-play decisions. Independent games run in
deterministic CPU workers; only the parent process updates weights, writes
checkpoints, and uses the GPU. Session preparation and exact resume restoration
live in :mod:`training.rl.session`; one iteration lives in
:mod:`training.rl.iteration`.
"""

from pathlib import Path
import time

from agents.nn import GPUContextLostError, is_gpu_context_loss
from training.rl.iteration import _reinforce_policy_update, run_iteration
from training.rl.reporting import build_training_summary
from training.rl.resume import (
    _atomic_network_save,
    _save_numbered_resume_checkpoint,
    _training_state_payload,
    resume_state_path,
)
from training.rl.session import prepare_training_session


def _ensure_final_checkpoint(session, actual_final_iteration):
    """Save the exact final state when no scheduled checkpoint already did."""
    context = session.context
    state = session.state
    execution = context.execution
    final_resume_path = resume_state_path(state.final_weights_path)
    if not execution.numbered_checkpoints or (
        state.last_saved_iteration == actual_final_iteration
        and state.final_weights_path.is_file()
        and final_resume_path.is_file()
    ):
        return
    training_state = _training_state_payload(
        win_rate_window=state.win_rate_window,
        value_loss_window=state.value_loss_window,
        ppo_window=state.ppo_window,
        total_decision_samples=state.total_decision_samples,
        total_normal_decision_samples=state.total_normal_decision_samples,
        total_restart_decision_samples=state.total_restart_decision_samples,
        total_restart_episodes=state.total_restart_episodes,
        total_restart_duration_s=state.total_restart_duration_s,
        policy_updates_completed=state.policy_updates_completed,
        clipped_iteration_count=state.clipped_iteration_count,
        total_rollout_duration_s=state.total_rollout_duration_s,
        total_update_duration_s=state.total_update_duration_s,
        elapsed_rl_seconds=(
            context.restored_elapsed_rl_seconds
            + time.perf_counter() - context.training_perf_started
        ),
    )
    state.final_weights_path, _state_path = _save_numbered_resume_checkpoint(
        context.network,
        context.runner,
        context.resources.rl_weights_path,
        actual_final_iteration,
        context.resume_configuration,
        context.runner.worker_count,
        state.completed_training_games,
        context.adaptive_tuning,
        training_state,
    )
    state.last_saved_iteration = actual_final_iteration


def _context_loss_message(state, iteration):
    """Return the operator-facing diagnosis of one lost CUDA context.

    Names the recovery point explicitly because the traceback alone reads like
    a training fault: CUDA reports an asynchronous failure at the next
    synchronization, so the line that raises is whichever one happened to copy
    a scalar to the host, not the one at fault. Nothing about ``iteration`` can
    be salvaged -- the weights and the optimizer moments live on the dead
    device -- so the last written checkpoint is the whole recovery.
    """
    return (
        f"The CUDA context died during RL iteration {iteration} "
        f"({state.completed_training_games:,} games completed, "
        f"{state.completed_this_invocation:,} of them in this process). This "
        "is not a training fault: the policy weights were healthy and the "
        "checkpoint at "
        f"iteration {state.last_saved_iteration} is intact. The GPU was reset "
        "underneath the process. Check `sudo dmesg -T | grep -i xid`, a driver "
        "upgrade in /var/log/apt/history.log, and `nvidia-smi -q -d "
        "TEMPERATURE,POWER`. Resume with the same command; the run loses only "
        "the iterations since that checkpoint."
    )


def train(training=None, resources=None, execution=None):
    """Train an exact game budget using three typed option groups."""
    session = prepare_training_session(training, resources, execution)
    context = session.context
    state = session.state
    stopped_by_shutdown = False
    finalization_started = None
    try:
        for local_iteration in range(1, session.iterations_to_run + 1):
            iteration = session.start_iteration + local_iteration
            try:
                if run_iteration(context, state, iteration):
                    stopped_by_shutdown = True
                    break
            except Exception as exc:
                if not is_gpu_context_loss(exc):
                    raise
                raise GPUContextLostError(
                    _context_loss_message(state, iteration)
                ) from exc

        finalization_started = time.perf_counter()
        if (
            state.completed_training_games != context.invocation_target_games
            and not stopped_by_shutdown
        ):
            raise AssertionError(
                f"RL completed {state.completed_training_games} games, expected "
                f"{context.invocation_target_games}."
            )
        actual_final_iteration = (
            session.start_iteration
            + state.completed_iterations_this_invocation
        )
        _ensure_final_checkpoint(session, actual_final_iteration)
    finally:
        context.metrics_stream.close()
        final_runtime_workers = context.runner.worker_count
        opponent_count = context.runner.opponent_pool.size
        unique_neural_opponent_count = (
            context.runner.opponent_pool.unique_neural_opponent_count
        )
        bucket_sizes = context.runner.opponent_pool.bucket_sizes()
        context.runner.close()

    context.parallel_summary["final_workers"] = final_runtime_workers
    if not context.execution.numbered_checkpoints:
        _atomic_network_save(context.network, context.resources.rl_weights_path)
        state.final_weights_path = Path(context.resources.rl_weights_path)
    elapsed_time = time.time() - session.start_time
    session.reporter.complete(elapsed_time, state.final_weights_path)
    session.runtime_profile.add(
        "final_checkpoint_shutdown_and_summary",
        time.perf_counter() - finalization_started,
    )
    runtime_profile_delta = session.runtime_profile.finish(
        games=state.completed_this_invocation,
        iterations=state.completed_iterations_this_invocation,
        decisions=(
            state.total_decision_samples
            - session.decision_samples_at_invocation_start
        ),
        optimizer_steps=(
            context.network.optimizer_step_count
            - session.optimizer_steps_at_invocation_start
        ),
    )
    return build_training_summary(
        session,
        actual_final_iteration=actual_final_iteration,
        stopped_by_shutdown=stopped_by_shutdown,
        final_runtime_workers=final_runtime_workers,
        opponent_count=opponent_count,
        unique_neural_opponent_count=unique_neural_opponent_count,
        bucket_sizes=bucket_sizes,
        runtime_profile_delta=runtime_profile_delta,
        elapsed_time=elapsed_time,
    )
