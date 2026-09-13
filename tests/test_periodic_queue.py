"""Asynchronous periodic diagnostics: the durable queue, its worker, the pipeline.

The contract under test is that measuring a periodic point beside training
changes when it is published and nothing else: the same checkpoint and seed
give the same games, wins, and training window as the synchronous path, no
point is lost or duplicated across failures and restarts, a task for an
abandoned future checkpoint is never published, and no worker outlives the
process that owns it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import numpy as np
import pytest

from agents.encoder import DominoEncoder
from agents.rl_nn import PolicyNetwork
from diagnostics import periodic_queue
from diagnostics.parallel_runner import ParallelSafetyConfig
from diagnostics.periodic_queue import (
    COMPLETED,
    FAILED,
    PENDING,
    PeriodicDiagnosticFailed,
    PeriodicDiagnosticQueue,
    build_task,
)
from diagnostics.rl_progress import (
    periodic_point_identity,
    read_periodic_history,
    run_periodic_diagnostic,
)
import training.pipeline as pipeline
from training.run_artifacts import periodic_diagnostics_path
from utils.artifacts import file_sha256

ROOT = Path(__file__).resolve().parents[1]
RULESET = "double-three"
SAFETY = ParallelSafetyConfig(
    memory_reserve_mb=0,
    estimated_worker_mb=1,
    max_worker_rss_mb=2048,
)


def _identity(games, *, diagnostic_games=40):
    return {
        "rl_games": games,
        "configuration_sha256": None,
        "diagnostic_seed": 11,
        "diagnostic_games": diagnostic_games,
        "opponent": "random",
        "ruleset_name": "double-six",
    }


def _task(run_dir, games, checkpoint, **overrides):
    checkpoint = Path(checkpoint)
    task = build_task(
        run_dir=run_dir,
        identity=overrides.pop("identity", _identity(games)),
        pipeline_level="forever",
        seed=7,
        rl_iterations=games // 100,
        checkpoint_path=checkpoint,
        checkpoint_sha256=file_sha256(checkpoint),
        rl_elapsed_seconds=float(games),
        training_window={},
        workers=1,
        safety_config=SAFETY,
    )
    task.update(overrides)
    return task


def _fake_worker(tmp_path, body):
    """Return a worker command running ``body`` with the task path as ``path``.

    The fake writes the same result envelope the real worker writes, so the
    queue logic is exercised without playing games.
    """
    script = tmp_path / "fake_worker.py"
    script.write_text(
        textwrap.dedent(
            """
            import json, os, sys, time
            from pathlib import Path
            sys.path.insert(0, {root!r})
            from diagnostics import periodic_queue
            from utils.artifacts import atomic_write_json
            path = Path(sys.argv[1])
            task = json.loads(path.read_text())

            def succeed(wins=3):
                atomic_write_json(
                    periodic_queue.result_path(task["run_dir"], task["rl_games"]),
                    {{
                        "format": periodic_queue.RESULT_FORMAT,
                        "format_version": periodic_queue.QUEUE_FORMAT_VERSION,
                        "identity": task["identity"],
                        "checkpoint_sha256": task["checkpoint_sha256"],
                        "row": {{"rl_games": task["rl_games"], "wins": wins}},
                        "measure_seconds": 0.0,
                        "measure_sections_seconds": {{}},
                        "pairwise_runtime_profile": {{"sections_seconds": {{}}}},
                        "selected_workers": 1,
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "finished_at": "2026-01-01T00:00:00+00:00",
                    }},
                )
            """
        ).format(root=str(ROOT))
        + textwrap.dedent(body),
        encoding="utf-8",
    )
    return [sys.executable, str(script)]


def _checkpoint(tmp_path, name="weights.npz"):
    path = tmp_path / name
    path.write_bytes(b"immutable milestone weights")
    return path


def _run_until(queue, condition, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        queue.pump()
        if condition():
            return
        time.sleep(0.05)
    raise AssertionError("the queue did not reach the expected state in time")


# -- the durable queue -------------------------------------------------------


def test_enqueue_is_idempotent_and_refuses_a_different_point(tmp_path):
    queue = PeriodicDiagnosticQueue(tmp_path, worker_command=["false"])
    checkpoint = _checkpoint(tmp_path)

    first = queue.enqueue(_task(tmp_path, 100, checkpoint))
    again = queue.enqueue(_task(tmp_path, 100, checkpoint))

    assert again["enqueued_at"] == first["enqueued_at"]
    assert queue.outstanding() == 1
    with pytest.raises(ValueError, match="different periodic diagnostic"):
        queue.enqueue(_task(
            tmp_path, 100, checkpoint,
            identity=_identity(100, diagnostic_games=99),
        ))


def test_tasks_run_one_at_a_time_and_publish_in_checkpoint_order(tmp_path):
    marker = tmp_path / "running"
    queue = PeriodicDiagnosticQueue(
        tmp_path,
        poll_seconds=0.05,
        worker_command=_fake_worker(tmp_path, f"""
            flag = Path({str(marker)!r})
            if flag.exists():
                raise SystemExit("two workers overlapped")
            flag.write_text("x")
            time.sleep(0.3)
            flag.unlink()
            succeed(wins=task["rl_games"] // 100)
        """),
    )
    checkpoint = _checkpoint(tmp_path)
    for games in (300, 100, 200):
        queue.enqueue(_task(tmp_path, games, checkpoint))

    published = []
    _run_until(
        queue,
        lambda: (
            published.extend(
                item.result["row"]["wins"] for item in queue.completed()
                if not queue.acknowledge(item.task)
            )
            or queue.outstanding() == 0
        ),
    )

    assert published == [1, 2, 3]
    assert not list(periodic_queue.queue_dir(tmp_path).glob("*"))


def test_a_task_failing_twice_stops_and_is_retried_by_the_next_session(tmp_path):
    queue = PeriodicDiagnosticQueue(
        tmp_path,
        poll_seconds=0.05,
        worker_command=_fake_worker(tmp_path, "raise SystemExit(5)"),
    )
    queue.enqueue(_task(tmp_path, 100, _checkpoint(tmp_path)))

    with pytest.raises(PeriodicDiagnosticFailed, match="failed 2 times"):
        _run_until(queue, lambda: False, timeout=20.0)
    (task,) = queue.tasks()
    assert task["state"] == FAILED
    assert task["attempts"] == 2
    assert Path(task["last_error"].rpartition("log: ")[2]).is_file()

    restarted = PeriodicDiagnosticQueue(tmp_path, worker_command=["false"])
    restarted.reconcile(restored_rl_games=100)
    (task,) = restarted.tasks()
    assert task["state"] == PENDING
    assert task["attempts"] == 0


def test_reconcile_discards_future_tasks_and_repairs_interrupted_ones(tmp_path):
    queue = PeriodicDiagnosticQueue(tmp_path, worker_command=["false"])
    checkpoint = _checkpoint(tmp_path)
    running = queue.enqueue(_task(tmp_path, 100, checkpoint, state="running"))
    queue.enqueue(_task(tmp_path, 200, checkpoint, state=COMPLETED))
    queue.enqueue(_task(tmp_path, 300, checkpoint, state=COMPLETED))
    # Only the future task has a readable result; the one at 200 lost its.
    periodic_queue.result_path(tmp_path, 300).write_text("{}", encoding="utf-8")
    periodic_queue.result_path(tmp_path, 999).write_text("{}", encoding="utf-8")

    queue.reconcile(restored_rl_games=250)

    tasks = {task["rl_games"]: task for task in queue.tasks()}
    assert set(tasks) == {100, 200}
    assert tasks[100]["state"] == PENDING
    assert tasks[100]["attempts"] == running["attempts"]
    assert tasks[200]["state"] == PENDING
    assert not periodic_queue.result_path(tmp_path, 300).exists()
    assert not periodic_queue.result_path(tmp_path, 999).exists()


def test_a_completed_result_survives_a_restart_and_publishes_once(tmp_path):
    queue = PeriodicDiagnosticQueue(
        tmp_path,
        poll_seconds=0.05,
        worker_command=_fake_worker(tmp_path, "succeed(wins=9)"),
    )
    queue.enqueue(_task(tmp_path, 100, _checkpoint(tmp_path)))
    _run_until(queue, lambda: bool(queue.completed()))

    # The training process stops before publishing; a new one takes over.
    restarted = PeriodicDiagnosticQueue(tmp_path, worker_command=["false"])
    restarted.reconcile(restored_rl_games=100)
    (ready,) = restarted.completed()
    assert ready.result["row"]["wins"] == 9
    restarted.pump()
    assert restarted.running_games() is None
    restarted.acknowledge(ready.task)
    assert restarted.outstanding() == 0


def test_stop_ends_the_worker_and_returns_its_attempt(tmp_path):
    started = tmp_path / "started"
    queue = PeriodicDiagnosticQueue(
        tmp_path,
        poll_seconds=0.05,
        worker_command=_fake_worker(tmp_path, f"""
            Path({str(started)!r}).write_text(str(os.getpid()))
            time.sleep(600)
        """),
    )
    queue.enqueue(_task(tmp_path, 100, _checkpoint(tmp_path)))
    _run_until(queue, started.exists)
    pid = int(started.read_text())

    began = time.monotonic()
    queue.stop(grace_seconds=5.0)

    assert time.monotonic() - began < 6.0
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    (task,) = queue.tasks()
    assert task["state"] == PENDING
    assert task["attempts"] == 0


def test_blocking_time_is_charged_to_the_newest_task(tmp_path):
    queue = PeriodicDiagnosticQueue(tmp_path, worker_command=["false"])
    checkpoint = _checkpoint(tmp_path)
    queue.enqueue(_task(tmp_path, 100, checkpoint))
    queue.enqueue(_task(tmp_path, 200, checkpoint))

    queue.add_blocking_seconds(1.5)
    queue.add_blocking_seconds(0.25)

    tasks = {task["rl_games"]: task for task in queue.tasks()}
    assert tasks[100]["blocking_seconds"] == 0.0
    assert tasks[200]["blocking_seconds"] == 1.75


# -- the real worker ----------------------------------------------------------


def _policy_checkpoint(path, *, seed=5):
    encoder = DominoEncoder()
    network = PolicyNetwork(
        input_size=encoder.vector_size,
        hidden_sizes=(12, 6),
        output_size=encoder.action_size,
        random_seed=seed,
        device="cpu",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{
        name: np.asarray(getattr(network, name)) for name in network.weight_names
    })
    return path


def _row_without_timing(row):
    return {
        key: value
        for key, value in row.items()
        if key not in {
            "created_at",
            "diagnostic_seconds",
            "rl_elapsed_seconds",
            "progress_elapsed_seconds",
            "runtime_profile_delta",
            "diagnostic_selected_workers",
        }
    }


@pytest.mark.parametrize("workers", [1, 3])
def test_the_worker_measures_exactly_what_the_synchronous_path_records(
    tmp_path, workers
):
    checkpoint = _policy_checkpoint(tmp_path / "checkpoints" / "games_0000000100_weights.npz")
    synchronous_dir = tmp_path / "synchronous"
    expected, appended = run_periodic_diagnostic(
        run_dir=synchronous_dir,
        pipeline_level="forever",
        seed=7,
        rl_games=100,
        rl_iterations=1,
        checkpoint_path=checkpoint,
        diagnostic_games=24,
        rl_elapsed_seconds=12.5,
        workers=2,
        safety_config=SAFETY,
    )
    assert appended

    run_dir = tmp_path / "asynchronous"
    identity = periodic_point_identity(
        run_dir, seed=7, rl_games=100, diagnostic_games=24
    )
    task = build_task(
        run_dir=run_dir,
        identity=identity,
        pipeline_level="forever",
        seed=7,
        rl_iterations=1,
        checkpoint_path=checkpoint,
        checkpoint_sha256=file_sha256(checkpoint),
        rl_elapsed_seconds=12.5,
        training_window={},
        workers=workers,
        safety_config=SAFETY,
    )
    queue = PeriodicDiagnosticQueue(run_dir, poll_seconds=0.1)
    queue.enqueue(task)
    _run_until(queue, lambda: bool(queue.completed()), timeout=120.0)
    (ready,) = queue.completed()

    assert ready.result["selected_workers"] == workers
    measured = dict(ready.result["row"])
    stored = read_periodic_history(periodic_diagnostics_path(synchronous_dir))[0]
    assert _row_without_timing(measured) == {
        key: value
        for key, value in _row_without_timing(expected).items()
        if key in measured
    }
    assert measured["wins"] == stored["wins"]


def test_the_worker_refuses_a_checkpoint_that_changed_after_queueing(tmp_path):
    checkpoint = _policy_checkpoint(tmp_path / "weights.npz")
    run_dir = tmp_path / "run"
    task = _task(
        run_dir,
        100,
        checkpoint,
        identity=periodic_point_identity(
            run_dir, seed=7, rl_games=100, diagnostic_games=8
        ),
        diagnostic_games=8,
    )
    queue = PeriodicDiagnosticQueue(run_dir)
    queue.enqueue(task)
    _policy_checkpoint(checkpoint, seed=6)
    path = periodic_queue.task_path(run_dir, 100)

    completed = subprocess.run(
        [sys.executable, "-m", "diagnostics.periodic_worker", str(path)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        check=False,
    )

    assert completed.returncode != 0
    assert "changed after it was queued" in completed.stderr
    assert not periodic_queue.result_path(run_dir, 100).exists()


def _children(pid):
    children = set()
    for task_dir in Path(f"/proc/{pid}/task").glob("*"):
        text = (task_dir / "children").read_text().split()
        children.update(int(value) for value in text)
    return children


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="uses /proc")
def test_terminating_the_worker_also_ends_its_game_workers(tmp_path):
    checkpoint = _policy_checkpoint(tmp_path / "weights.npz")
    run_dir = tmp_path / "run"
    task = _task(
        run_dir,
        100,
        checkpoint,
        identity=periodic_point_identity(
            run_dir, seed=7, rl_games=100, diagnostic_games=400_000
        ),
        diagnostic_games=400_000,
        workers=2,
    )
    queue = PeriodicDiagnosticQueue(run_dir, poll_seconds=0.1)
    queue.enqueue(task)
    queue.pump()
    worker_pid = queue.tasks()[0]["worker_pid"]
    deadline = time.monotonic() + 60.0
    game_workers = set()
    while time.monotonic() < deadline and len(game_workers) < 2:
        time.sleep(0.2)
        game_workers = _children(worker_pid)
    assert game_workers, "the diagnostic never started its game workers"

    queue.stop(grace_seconds=30.0)

    time.sleep(0.5)
    for pid in game_workers | {worker_pid}:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    (stopped,) = queue.tasks()
    assert stopped["state"] == PENDING
    assert not periodic_queue.result_path(run_dir, 100).exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="uses prctl")
def test_the_worker_exits_when_its_training_process_dies(tmp_path):
    checkpoint = _policy_checkpoint(tmp_path / "weights.npz")
    run_dir = tmp_path / "run"
    task = _task(
        run_dir,
        100,
        checkpoint,
        identity=periodic_point_identity(
            run_dir, seed=7, rl_games=100, diagnostic_games=400_000
        ),
        diagnostic_games=400_000,
        workers=1,
    )
    PeriodicDiagnosticQueue(run_dir).enqueue(task)
    pid_file = tmp_path / "worker.pid"
    owner = subprocess.Popen(  # pylint: disable=consider-using-with
        [
            sys.executable,
            "-c",
            textwrap.dedent(f"""
                import sys, time
                sys.path.insert(0, {str(ROOT)!r})
                from pathlib import Path
                from diagnostics.periodic_queue import PeriodicDiagnosticQueue
                queue = PeriodicDiagnosticQueue(Path({str(run_dir)!r}))
                queue.pump()
                Path({str(pid_file)!r}).write_text(str(queue.tasks()[0]["worker_pid"]))
                time.sleep(600)
            """),
        ],
        cwd=str(ROOT),
    )
    deadline = time.monotonic() + 60.0
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    worker_pid = int(pid_file.read_text())
    time.sleep(3.0)  # let the worker install its parent-death signal

    owner.send_signal(signal.SIGKILL)
    owner.wait()

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            os.kill(worker_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.2)
    else:
        os.kill(worker_pid, signal.SIGKILL)
        pytest.fail("the worker outlived its training process")


# -- the canonical pipeline ----------------------------------------------------


# double-three plays about 1.6 learner decisions per game, so 200 games give
# PPO the 256 decisions it needs for a real update between two points.
GPI = 200


def _run_pipeline(root, *extra, total=3 * GPI):
    root.mkdir(parents=True, exist_ok=True)
    encoder = DominoEncoder(RULESET)
    supervised = root / "sl.npz"
    if not supervised.exists():
        network = PolicyNetwork(
            input_size=encoder.vector_size,
            hidden_sizes=(16, 8),
            output_size=encoder.action_size,
            random_seed=3,
            device="cpu",
        )
        np.savez(supervised, **{
            name: np.asarray(getattr(network, name))
            for name in network.weight_names
        })
    args = pipeline.parse_args([
        "big",
        "--artifact-root", str(root),
        "--seed", "7",
        "--ruleset", RULESET,
        "--total-training-games", str(total),
        "--gpi", str(GPI),
        "--rl-workers", "1",
        "--device", "cpu",
        "--learning-rate", "0.5",
        "--ppo-max-epochs", "2",
        "--periodic-diagnostic-games", "200",
        "--periodic-diagnostic-every-games", str(GPI),
        "--hidden-layers", "2",
        "--hidden1-size", "16",
        "--hidden2-size", "8",
        "--diagnostic-workers", "2",
        "--diagnostic-memory-reserve-mb", "0",
        "--rl-memory-reserve-mb", "0",
        "--skip-final-diagnostic",
        *extra,
    ])
    config = pipeline._build_config(args.scale)  # pylint: disable=protected-access
    pipeline.validate_args(args, config)
    assets = {
        "paths": SimpleNamespace(weights=supervised),
        "weights_metadata": {"artifact": {"sha256": file_sha256(supervised)}},
    }
    return pipeline.run_rl_pipeline(root, config, args, assets)


def _history(result):
    return read_periodic_history(periodic_diagnostics_path(result["run_dir"]))


def test_asynchronous_pipeline_records_what_the_synchronous_one_records(
    tmp_path, monkeypatch
):
    synchronous = _run_pipeline(tmp_path / "sync")
    segments = []
    train = pipeline.rl_training_loop.train

    def observed_train(training, resources, execution):
        # The pipeline trains into <run>/checkpoint_states/training.npz.
        run_dir = Path(resources.rl_weights_path).parents[1]
        segments.append([
            task["state"] for task in PeriodicDiagnosticQueue(run_dir).tasks()
        ])
        return train(training, resources, execution)

    monkeypatch.setattr(pipeline.rl_training_loop, "train", observed_train)
    asynchronous = _run_pipeline(
        tmp_path / "async",
        "--async-periodic-diagnostics",
        "--async-diagnostic-workers", "3",
    )

    # The first segment's rollout-worker autotune runs on a quiet machine: the
    # point-zero task is queued but not started until the first milestone.
    assert segments[0] == [PENDING]
    assert any("running" in states for states in segments[1:])

    expected = _history(synchronous)
    actual = _history(asynchronous)
    assert [row["rl_games"] for row in actual] == [0, GPI, 2 * GPI, 3 * GPI]
    # The flag is a locked argument, and locked arguments are part of a run's
    # configuration hash, exactly as --diagnostic-workers already is.
    assert expected[0]["configuration_sha256"] != actual[0]["configuration_sha256"]
    for left, right in zip(expected, actual):
        left = {**_row_without_timing(left), "configuration_sha256": None}
        right = {**_row_without_timing(right), "configuration_sha256": None}
        assert left == right
    assert len({row["wins"] for row in actual}) > 1, "the points are indistinct"
    # Diagnostics never touch training: the final weights are byte-identical.
    with np.load(synchronous["rl_weights_path"]) as left:
        with np.load(asynchronous["rl_weights_path"]) as right:
            assert left.files == right.files
            for name in left.files:
                assert left[name].tobytes() == right[name].tobytes(), name
    run_dir = Path(asynchronous["run_dir"])
    state = json.loads((run_dir / "training_state.json").read_text())
    assert state["last_periodic_diagnostic_game"] == 3 * GPI
    assert not list(periodic_queue.queue_dir(run_dir).glob("task_*"))
    profile = json.loads(
        (run_dir / "diagnostics" / "runtime_profile.json").read_text()
    )
    diagnostics = profile["cumulative"]["rl_vs_random_diagnostics"]
    assert diagnostics["async_execution_count"] == 4
    assert diagnostics["games"] == 800
    # Locked into the run, like every other non-operational argument.
    config = json.loads(
        next(run_dir.glob("*/run_config.json")).read_text()
    )
    assert config["locked_arguments"]["async_periodic_diagnostics"] is True


def test_a_resumed_pipeline_publishes_a_finished_result_without_measuring_it(
    tmp_path, monkeypatch
):
    root = tmp_path / "run"
    first = _run_pipeline(root, "--async-periodic-diagnostics", total=2 * GPI)
    run_dir = Path(first["run_dir"])
    history_before = _history(first)
    assert [row["rl_games"] for row in history_before] == [0, GPI, 2 * GPI]

    # Rebuild the state of a process that died after the worker finished the
    # last point but before publishing it: the task and its result are on disk,
    # the history does not have the point yet.
    path = periodic_diagnostics_path(run_dir)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    measured = dict(history_before[-1])
    queue = PeriodicDiagnosticQueue(run_dir)
    identity = periodic_point_identity(
        run_dir, seed=7, rl_games=2 * GPI, diagnostic_games=200
    )
    checkpoint = run_dir / "checkpoints" / f"games_{2 * GPI:010d}_weights.npz"
    task = build_task(
        run_dir=run_dir,
        identity=identity,
        pipeline_level="big",
        seed=7,
        rl_iterations=2,
        checkpoint_path=checkpoint,
        checkpoint_sha256=file_sha256(checkpoint),
        rl_elapsed_seconds=measured["rl_elapsed_seconds"],
        training_window={},
        workers=2,
        safety_config=SAFETY,
    )
    task["state"] = COMPLETED
    queue.enqueue(task)
    from utils.artifacts import atomic_write_json  # pylint: disable=import-outside-toplevel

    atomic_write_json(periodic_queue.result_path(run_dir, 2 * GPI), {
        "format": periodic_queue.RESULT_FORMAT,
        "format_version": periodic_queue.QUEUE_FORMAT_VERSION,
        "identity": identity,
        "checkpoint_sha256": task["checkpoint_sha256"],
        "row": {
            key: measured[key]
            for key in (
                "format_version", "pipeline_level", "seed", "rl_games",
                "rl_iterations", "configuration_sha256", "opponent",
                "diagnostic_games", "wins", "diagnostic_seed",
                "diagnostic_seed_namespace", "diagnostic_seconds",
                "rl_elapsed_seconds", "created_at",
            )
        },
        "measure_seconds": 0.5,
        "measure_sections_seconds": {},
        "pairwise_runtime_profile": {"sections_seconds": {}},
        "selected_workers": 2,
        "started_at": task["enqueued_at"],
        "finished_at": task["enqueued_at"],
    })

    def refuse(*_args, **_kwargs):
        raise AssertionError("a finished result was measured again")

    monkeypatch.setattr(PeriodicDiagnosticQueue, "_start", refuse)
    resumed = _run_pipeline(
        root, "--async-periodic-diagnostics", "--resume", str(run_dir),
        total=2 * GPI,
    )

    history_after = _history(resumed)
    assert [row["rl_games"] for row in history_after] == [0, GPI, 2 * GPI]
    assert history_after[-1]["wins"] == measured["wins"]
    assert not list(periodic_queue.queue_dir(run_dir).glob("*_games_*"))


def _points_with_fake_worker(tmp_path, body):
    queue = PeriodicDiagnosticQueue(
        tmp_path,
        poll_seconds=0.05,
        worker_command=_fake_worker(tmp_path, body),
    )
    return pipeline._AsyncPeriodicPoints(  # pylint: disable=protected-access
        args=SimpleNamespace(),
        run_dir=tmp_path,
        level="forever",
        runtime_profiler=None,
        queue=queue,
    )


def test_a_full_backlog_holds_training_and_charges_the_wait(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ASYNC_OUTSTANDING_LIMIT", 2)
    points = _points_with_fake_worker(tmp_path, "time.sleep(0.4); succeed()")
    queue = points.queue
    checkpoint = _checkpoint(tmp_path)
    for games in (100, 200, 300):
        queue.enqueue(_task(tmp_path, games, checkpoint))
    acknowledged = []

    def publish():
        for item in queue.completed():
            acknowledged.append(item.task["rl_games"])
            queue.acknowledge(item.task)

    began = time.monotonic()
    points.wait_for_capacity(publish=publish, shutdown=lambda: False)
    waited = time.monotonic() - began

    assert acknowledged == [100, 200]
    (newest,) = queue.tasks()
    assert newest["rl_games"] == 300
    assert 0.5 < newest["blocking_seconds"] <= waited + 0.01
    queue.stop(grace_seconds=5.0)


def test_a_shutdown_ends_the_wait_and_leaves_the_backlog_queued(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "ASYNC_OUTSTANDING_LIMIT", 1)
    points = _points_with_fake_worker(tmp_path, "time.sleep(600)")
    queue = points.queue
    queue.enqueue(_task(tmp_path, 100, _checkpoint(tmp_path)))
    calls = []

    points.wait_for_capacity(
        publish=lambda: None,
        shutdown=lambda: calls.append(1) or len(calls) > 3,
    )
    points.stop(publish=lambda: None)

    (task,) = queue.tasks()
    assert task["state"] == PENDING
    assert task["blocking_seconds"] > 0.0


def test_queue_wait_is_measured_between_iso_timestamps():
    seconds = pipeline._seconds_between(  # pylint: disable=protected-access
        "2026-09-12T10:00:00+00:00", "2026-09-12T10:00:02.500000+00:00"
    )
    assert seconds == 2.5
    assert pipeline._seconds_between(None, "x") == 0.0  # pylint: disable=protected-access
