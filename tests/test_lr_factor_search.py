"""The adaptive one-factor learning-rate search: order, scoring, decisions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import textwrap

import pytest

from train_script import lr_factor_search as search
from train_script.lr_factor_search import (
    DOWN,
    UP,
    candidates,
    decide,
    learning_rate_text,
    search_ordinals,
    sustained_final,
)

ROOT = Path(__file__).resolve().parents[1]


def _orders(state, verdict):
    return [e.candidate.order for e in state.evaluations if e.verdict == verdict]


# -- order -------------------------------------------------------------------


def test_candidates_alternate_up_and_down_in_powers_of_two():
    ordered = candidates(0.001)

    assert [c.learning_rate for c in ordered] == [
        0.001, 0.002, 0.0005, 0.004, 0.00025, 0.008, 0.000125,
    ]
    assert [c.label for c in ordered] == [
        "lr_x1", "lr_x2", "lr_d2", "lr_x4", "lr_d4", "lr_x8", "lr_d8",
    ]
    assert [c.order for c in ordered] == list(range(1, 8))
    assert ordered[4].run_name("diego_notebook") == "lr_search_0p00025_diego_notebook"
    assert ordered[6].runner_value() == "lr=0.000125+warmup=on"


def test_rates_are_never_spelled_in_scientific_notation():
    assert learning_rate_text(1e-05) == "0.00001"
    assert candidates(0.0001)[6].run_name("m") == "lr_search_0p0000125_m"


# -- scoring ------------------------------------------------------------------


def _history(points):
    return [
        {"progress_elapsed_seconds": hours * 3600.0, "win_rate": win / 100.0}
        for hours, win in points
    ]


def test_the_score_is_the_mean_of_the_last_fifth_of_the_horizon():
    points = [(0.0, 62.0)] + [(h / 2, 64.0 + h / 10) for h in range(1, 11)]
    scored = sustained_final(_history(points), 0.2)

    # Horizon 5 h; the window starts at 4 h: points at 4.0, 4.5 and 5.0 h.
    assert scored["window_points"] == 3
    assert scored["final_sustained_percent"] == pytest.approx((64.8 + 64.9 + 65.0) / 3)
    assert scored["horizon_hours"] == 5.0
    assert 62.0 < scored["auc_percent"] < 65.0


def test_a_run_with_too_few_final_points_is_not_scored():
    with pytest.raises(ValueError, match="final window"):
        sustained_final(_history([(0.0, 62.0), (1.0, 63.0), (5.0, 64.0)]), 0.2)


# -- decisions ------------------------------------------------------------------


def test_the_first_run_is_the_default_and_nothing_else_waits_on_it():
    state = decide(candidates(0.001), {})

    assert state.next_candidate.order == 1
    assert _orders(state, "pending") == [2, 3, 4, 5, 6, 7]


def test_two_worse_runs_stop_a_direction_and_skip_its_remaining_factors():
    ordered = candidates(0.001)
    # x1 66.0 | x2 65.5 worse | /2 66.1 tie above | x4 65.0 worse -> up stops
    scores = {1: 66.0, 2: 65.5, 3: 66.1, 4: 65.0}

    state = decide(ordered, scores)

    assert state.worse_counts == {UP: 2, DOWN: 0}
    assert state.stopped == {UP: True, DOWN: False}
    assert state.best.order == 3
    assert state.next_candidate.order == 5
    assert _orders(state, "skipped") == [6]
    assert _orders(state, "pending") == [7]


def test_a_difference_within_the_ruler_is_a_tie_not_a_worse_run():
    ordered = candidates(0.001)
    scores = {1: 66.00, 2: 65.79, 3: 65.77}

    state = decide(ordered, scores, margin_pp=0.22)

    verdicts = {e.candidate.order: e.verdict for e in state.evaluations}
    assert verdicts[2] == "tie_below"
    assert verdicts[3] == "worse"
    assert state.worse_counts == {UP: 0, DOWN: 1}


def test_worse_is_judged_against_the_best_known_when_the_run_finished():
    ordered = candidates(0.001)
    # /2 becomes the best only after x2 was judged, so x2 stays a tie.
    scores = {1: 66.0, 2: 65.9, 3: 66.5, 4: 66.2, 5: 66.9, 6: 66.0, 7: 66.3}

    state = decide(ordered, scores)
    verdicts = {e.candidate.order: e.verdict for e in state.evaluations}

    assert verdicts == {
        1: "reference", 2: "tie_below", 3: "better", 4: "worse",
        5: "better", 6: "worse", 7: "worse",
    }
    assert state.done
    assert state.best.learning_rate == 0.00025


def test_the_search_ends_when_both_directions_have_stopped():
    ordered = candidates(0.001)
    scores = {1: 67.0, 2: 66.0, 3: 66.0, 4: 66.0, 5: 66.0}

    state = decide(ordered, scores)

    assert state.done
    assert state.stopped == {UP: True, DOWN: True}
    assert _orders(state, "skipped") == [6, 7]
    assert state.best.order == 1


def test_bundle_numbers_follow_launch_order_and_skip_nothing():
    ordered = candidates(0.001)
    # Up stops at x4, so x8 is never launched and /8 takes the next number.
    state = decide(ordered, {1: 66.0, 2: 65.5, 3: 66.1, 4: 65.0, 5: 66.0})

    assert search_ordinals(state, 101) == {1: 101, 2: 102, 3: 103, 4: 104, 5: 105, 7: 106}
    assert search_ordinals(state, 101, unlaunched={1}) == {
        2: 101, 3: 102, 4: 103, 5: 104, 7: 105,
    }


# -- runs made for the test ------------------------------------------------------


def make_run(run_root, pipeline_argv, final, periodic_points=11):
    """Create a finished canonical run the way the pipeline records one.

    Its win rate climbs to ``final`` over 5 hours of progress, so its sustained
    final level is ``final - 0.01``.
    """
    # pylint: disable=import-outside-toplevel,protected-access
    from diagnostics.rl_progress import append_periodic_point
    from training import pipeline
    from training.canonical_run import create_run_config
    from training.run_artifacts import periodic_diagnostics_path

    args = pipeline.parse_args(pipeline_argv)
    run_dir = Path(run_root) / f"domino_rl_forever_seed52_run{args.run_name}"
    config = create_run_config(
        run_dir, root=Path(run_root), pipeline_level="forever", seed=52,
        target_rl_games=None, supervised_weights_path="sl.npz",
        supervised_weights_sha256="f" * 64, ppo_config=pipeline._ppo_config(args),
        rl_config=pipeline._rl_config(args), diagnostic_config={"periodic_games": 1000},
        run_name=args.run_name, locked_arguments=pipeline._locked_run_arguments(args),
        network_architecture=pipeline._network_architecture(args),
        bundle_suffix=args.bundle_suffix,
        run_ordinal=args.run_ordinal, machine_slug_override="test_machine",
        machine={"cpu_model": "Test", "logical_cpu_count": 4, "ram_total_bytes": 1,
                 "gpu_name": None, "vram_total_bytes": None, "rl_device": "cpu"},
    )
    last = periodic_points - 1
    for index in range(periodic_points):
        win = 62.58 if index == 0 else final - (last - index) * 0.01
        append_periodic_point(periodic_diagnostics_path(run_dir), {
            "format_version": 5, "pipeline_level": "forever", "seed": 52,
            "rl_games": index * 1000, "rl_iterations": index,
            "configuration_sha256": config["configuration_sha256"],
            "opponent": "random", "diagnostic_games": 100000,
            "wins": round(win * 1000), "diagnostic_seed": 7,
            "diagnostic_seed_namespace": "periodic_rl_vs_random",
            "diagnostic_seconds": 0.0,
            "rl_elapsed_seconds": index * 18000.0 / last,
            "created_at": "2026-09-13T00:00:00+00:00",
        })
    (run_dir / "training_state.json").write_text("{}")
    return run_dir


def _write_state(results_dir, rows):
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "sequence_state.tsv").write_text(
        "run_name\tstatus\n" + "".join(f"{name}\t{status}\n" for name, status in rows),
        encoding="utf-8",
    )


def _evaluate(tmp_path, **options):
    return search.evaluate_search(
        machine_slug="m", results_dir=tmp_path / "results", run_root=tmp_path / "runs",
        seed=52, default_learning_rate=0.001, **options,
    )


def test_the_control_run_comes_first_and_the_search_follows_it(tmp_path):
    assert _evaluate(tmp_path).next_run == (
        "combined", "lr_x1", "lr=0.001+warmup=on", "lr_search_0p001_m", 101,
    )
    runs = _evaluate(tmp_path, control_run=True)
    assert runs.next_run == ("one_factor", "default", "default", "lr_search_default_m", 101)
    assert runs.ordinals == {1: 102}


def test_a_reference_run_stands_in_for_the_default_and_takes_no_number(tmp_path):
    reference = tmp_path / "earlier"
    make_run(tmp_path / "runs", [
        "forever", "--ruleset", "double-six", "--run-name", "earlier_warmup",
        "--warmup-lr", "--bundle-suffix", "warmup_true",
    ], final=66.0)
    _write_state(reference, [("earlier_warmup", "completed")])

    runs = _evaluate(tmp_path, reference_run="earlier_warmup", reference_results_dir=reference)

    assert runs.names[1] == "earlier_warmup"
    assert runs.state.best.order == 1
    assert runs.details[1]["final_sustained_percent"] == pytest.approx(65.99)
    assert runs.next_run == (
        "combined", "lr_x2", "lr=0.002+warmup=on", "lr_search_0p002_m", 101,
    )


def test_a_reference_run_with_other_settings_is_refused(tmp_path):
    reference = tmp_path / "earlier"
    make_run(tmp_path / "runs", [
        "forever", "--ruleset", "double-six", "--run-name", "earlier_gpi",
        "--warmup-lr", "--gpi", "12000",
    ], final=66.0)
    _write_state(reference, [("earlier_gpi", "completed")])

    with pytest.raises(ValueError, match="cannot stand in.*gpi"):
        _evaluate(tmp_path, reference_run="earlier_gpi", reference_results_dir=reference)


def test_an_unfinished_reference_run_is_refused(tmp_path):
    reference = tmp_path / "earlier"
    _write_state(reference, [("earlier_warmup", "interrupted")])

    with pytest.raises(ValueError, match="has not completed"):
        _evaluate(tmp_path, reference_run="earlier_warmup", reference_results_dir=reference)


def test_a_changed_search_definition_is_refused(tmp_path):
    config = {"default_learning_rate": 0.001, "margin_pp": 0.22}
    search.load_or_lock_config(tmp_path, config)
    search.load_or_lock_config(tmp_path, dict(config))

    with pytest.raises(ValueError, match="different search"):
        search.load_or_lock_config(tmp_path, {**config, "default_learning_rate": 0.002})


# -- end to end through the real shell runner ----------------------------------


FAKE_PYTHON = textwrap.dedent("""\
    #!{python}
    import json, os, signal, sys, time
    argv = sys.argv[1:]
    if argv[:3] != ["-u", "-m", "training.pipeline"]:
        os.execv({python!r}, [{python!r}, *argv])
    sys.path.insert(0, {root!r})
    pipeline_argv = argv[3:]
    with open(os.environ["FAKE_LOG"], "a") as log:
        log.write(" ".join(pipeline_argv) + "\\n")
    print("Canonical RL run", flush=True)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    deadline = time.time() + 120
    while not stop and time.time() < deadline:
        time.sleep(0.1)
    from tests.test_lr_factor_search import make_run
    name = pipeline_argv[pipeline_argv.index("--run-name") + 1]
    make_run(os.environ["SEQUENCE_RUN_ROOT"], pipeline_argv,
             json.loads(os.environ["FAKE_FINALS"])[name])
    print("Canonical pipeline finished", flush=True)
""")

SEARCH_FINALS = {
    # x2 and x4 are worse, /2 is the best, /4 and /8 are worse: the up
    # direction stops before x8 is ever run, the down direction after /8.
    "lr_search_0p001_test_machine": 66.0,
    "lr_search_0p002_test_machine": 65.5,
    "lr_search_0p0005_test_machine": 66.4,
    "lr_search_0p004_test_machine": 65.0,
    "lr_search_0p00025_test_machine": 65.9,
    "lr_search_0p008_test_machine": 70.0,
    "lr_search_0p000125_test_machine": 65.8,
    # The control run scores far above everything and still decides nothing.
    "lr_search_default_test_machine": 80.0,
}
LAUNCHED_SEARCH = [
    "lr_search_0p001_test_machine", "lr_search_0p002_test_machine",
    "lr_search_0p0005_test_machine", "lr_search_0p004_test_machine",
    "lr_search_0p00025_test_machine", "lr_search_0p000125_test_machine",
]


@pytest.mark.parametrize("setup", ["search_only", "control_run", "reference_run"])
def test_the_wrapper_runs_the_search_through_the_real_runner(tmp_path, setup):
    fake = tmp_path / "python"
    fake.write_text(FAKE_PYTHON.format(python=sys.executable, root=str(ROOT)))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    settings = ""
    expected_names = list(LAUNCHED_SEARCH)
    if setup == "control_run":
        settings = "LR_SEARCH_CONTROL_RUN=1"
        expected_names.insert(0, "lr_search_default_test_machine")
    elif setup == "reference_run":
        # The default warmup run was made earlier, elsewhere, under its own name.
        make_run(tmp_path / "runs", [
            "forever", "--ruleset", "double-six", "--run-name", "earlier_warmup",
            "--warmup-lr", "--bundle-suffix", "warmup_true",
        ], final=66.0)
        _write_state(tmp_path / "earlier", [("earlier_warmup", "completed")])
        settings = (
            'LR_SEARCH_REFERENCE_RUN="earlier_warmup"\n'
            f'LR_SEARCH_REFERENCE_RESULTS_DIR="{tmp_path / "earlier"}"'
        )
        expected_names.pop(0)
    wrapper = tmp_path / "run_search.sh"
    wrapper.write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail
        MACHINE_SLUG="test_machine"
        MACHINE_LABEL="Test machine"
        TIME_COEFFICIENT="1.0"
        RL_TIME_LIMIT=2
    """) + settings + textwrap.dedent(f"""
        source {ROOT}/train_script/_sequential_rl_experiment_runner.bash
        source {ROOT}/train_script/_lr_factor_search.bash
        run_lr_factor_search --grace 20s "$@"
    """))
    results = tmp_path / "results"
    environment = {
        **os.environ,
        "PYTHON": str(fake),
        "SEQUENCE_RUN_ROOT": str(tmp_path / "runs"),
        "SEQUENCE_RESULTS_DIR": str(results),
        "SEQUENCE_POLL_SECONDS": "1",
        "FAKE_LOG": str(tmp_path / "fake.log"),
        "FAKE_FINALS": json.dumps(SEARCH_FINALS),
    }
    # Both sourced files locate the repository from their own path, so the
    # wrapper itself can live anywhere.
    completed = subprocess.run(
        ["bash", str(wrapper)],
        env=environment, capture_output=True, text=True, timeout=300, check=False,
    )

    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    launched = [line.split() for line in (tmp_path / "fake.log").read_text().splitlines()]
    names = [line[line.index("--run-name") + 1] for line in launched]
    assert names == expected_names
    ordinals = [int(line[line.index("--run-ordinal") + 1]) for line in launched]
    assert ordinals == list(range(101, 101 + len(names)))
    warmups = ["--warmup-lr" in line for line in launched]
    assert warmups == [name != "lr_search_default_test_machine" for name in names]

    bundles = sorted(
        path.name.split("-", 1)[1]
        for path in (tmp_path / "runs").glob("*/[0-9]*-[0-9]*_test_machine_*")
    )
    assert bundles[:2] == {
        "search_only": ["101_test_machine_lr0p001_warmup", "102_test_machine_lr0p002_warmup"],
        "control_run": ["101_test_machine_control", "102_test_machine_lr0p001_warmup"],
        "reference_run": ["101_test_machine_lr0p002_warmup", "102_test_machine_lr0p0005_warmup"],
    }[setup]
    assert len(bundles) == len(names)

    summary = json.loads((results / "lr_factor_search_summary.json").read_text())
    assert summary["done"] is True
    assert summary["best"]["learning_rate"] == 0.0005
    assert summary["stopped"] == {"up": True, "down": True}
    verdicts = [row["verdict"] for row in summary["evaluations"]]
    assert verdicts == ["reference", "worse", "better", "worse", "worse", "skipped", "worse"]
    assert (summary["control_run"] is not None) == (setup == "control_run")
    assert summary["evaluations"][0]["reference_run"] == (setup == "reference_run")

    # Running the finished search again launches nothing and reports.
    again = subprocess.run(
        ["bash", str(wrapper)],
        env=environment, capture_output=True, text=True, timeout=120, check=False,
    )
    assert again.returncode == 0, again.stderr[-2000:]
    assert len((tmp_path / "fake.log").read_text().splitlines()) == len(launched)
    assert "Busca encerrada" in again.stdout


def test_a_completed_run_that_cannot_be_scored_stops_with_its_reason(tmp_path, capsys):
    results = tmp_path / "results"
    results.mkdir()
    name = candidates(0.001)[0].run_name("m")
    (results / "sequence_state.tsv").write_text(
        "run_name\tstatus\n" f"{name}\tcompleted\n", encoding="utf-8"
    )

    status = search.main([
        "next", "--machine-slug", "m", "--results-dir", str(results),
        "--run-root", str(tmp_path / "runs"),
    ])

    assert status == 2
    assert name in capsys.readouterr().err
