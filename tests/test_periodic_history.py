"""Contracts for the compact periodic-diagnostic record and its older layouts.

Three formats flow through one reader, so every consumer downstream is
version-blind. These tests pin that: a run written under any of them keeps its
curve, its derived statistics and its rebuilt CSV.

    v3  header with `columns` plus positional arrays, nine outcome fields
    v4  one self-describing JSON object per line, 39 fields
    v5  header with `fields` plus positional arrays, 37 fields, rounded per
        category -- the format specified in
        references/atualizacoes/atualizacao_0409
"""

from __future__ import annotations

import json
import zipfile

import pytest

from training.canonical_run import RUN_FORMAT_VERSION
from diagnostics.rl_progress import (
    FORMAT_VERSION,
    HISTORY_CHECKPOINT_BASE,
    HISTORY_RECORD_TYPE,
    LEGACY_V3_DATA_FIELDS,
    RECORDED_PLACES,
    V5_FIELD_ORDER,
    _point_key,
    append_periodic_point,
    archive_periodic_history,
    checkpoint_path_for_record,
    read_periodic_history,
    summarize_training_window,
)

STATIC = {
    "pipeline_level": "small",
    "ruleset_name": "double-three",
    "seed": 42,
    "opponent": "random",
    "diagnostic_games": 100,
    "diagnostic_seed": 7,
    "diagnostic_seed_namespace": "periodic_rl_vs_random",
    "configuration_sha256": "cfg",
}


def _row(rl_games, *, wins=60, **extra):
    row = {
        **STATIC,
        "rl_games": rl_games,
        "rl_iterations": rl_games // 2,
        "wins": wins,
        "diagnostic_seconds": 0.5,
        "rl_elapsed_seconds": float(rl_games),
        "created_at": "2026-09-04T00:00:00.123456+00:00",
    }
    row.update(extra)
    return row


def _history(tmp_path):
    return tmp_path / "run_compact_diagnostics" / "periodic_diagnostics.jsonl"


def _write_legacy(path, rows, *, version):
    """Write a file in one of the superseded layouts."""
    header = {
        "record_type": HISTORY_RECORD_TYPE,
        "format_version": version,
        "checkpoint_path_base": HISTORY_CHECKPOINT_BASE,
        "static": dict(STATIC),
    }
    if version == 3:
        header["columns"] = list(LEGACY_V3_DATA_FIELDS)
    lines = [json.dumps(header)]
    for row in rows:
        stored = dict(row)
        stored.setdefault("checkpoint_path", f"games_{row['rl_games']:010d}_weights.npz")
        stored.setdefault("selected_workers", 4)
        if version == 3:
            lines.append(
                json.dumps([stored.get(n) for n in LEGACY_V3_DATA_FIELDS])
            )
        else:
            lines.append(json.dumps({
                "cumulative_games": stored["rl_games"],
                "iteration": stored["rl_iterations"],
                "timestamp": stored["created_at"],
                "wins": stored["wins"],
                "rl_elapsed_seconds": stored["rl_elapsed_seconds"],
                "diagnostic_seconds": stored["diagnostic_seconds"],
                "checkpoint_path": stored["checkpoint_path"],
                "selected_workers": stored["selected_workers"],
            }))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------
# The older layouts keep working
# ----------------------------------------------------------------------


@pytest.mark.parametrize("version", [3, 4])
def test_a_legacy_file_reads_with_every_derived_value(tmp_path, version):
    path = _history(tmp_path)
    _write_legacy(
        path,
        [_row(0, wins=60, checkpoint_sha256="aa" * 32),
         _row(100, wins=70, checkpoint_sha256="bb" * 32)],
        version=version,
    )
    rows = read_periodic_history(path)

    assert [row["rl_games"] for row in rows] == [0, 100]
    assert rows[0]["win_rate"] == pytest.approx(0.60)
    assert rows[1]["losses"] == 30
    assert rows[1]["ci95_win_rate_low"] < 0.70 < rows[1]["ci95_win_rate_high"]
    # A legacy record's stored path is passed through, so an archived run keeps
    # naming exactly what it named before.
    assert rows[1]["checkpoint_path"].endswith("games_0000000100_weights.npz")


def test_a_legacy_record_and_its_v5_equivalent_share_one_identity(tmp_path):
    """Deduplication must not regress across the format change."""
    path = _history(tmp_path)
    _write_legacy(path, [_row(0)], version=3)
    assert _point_key(read_periodic_history(path)[0]) == _point_key(_row(0))


# ----------------------------------------------------------------------
# The v5 layout
# ----------------------------------------------------------------------


def test_v5_writes_a_field_header_and_positional_arrays(tmp_path):
    path = _history(tmp_path)
    append_periodic_point(path, _row(0))
    append_periodic_point(path, _row(100, wins=70))

    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header["format_version"] == FORMAT_VERSION
    assert header["fields"] == list(V5_FIELD_ORDER)
    for line in lines[1:]:
        values = json.loads(line)
        assert isinstance(values, list)
        assert len(values) == len(V5_FIELD_ORDER)


def test_v5_no_longer_stores_the_checkpoint_or_the_worker_count():
    for gone in ("checkpoint_path", "checkpoint_sha256", "selected_workers"):
        assert gone not in V5_FIELD_ORDER


def test_the_measured_checkpoint_is_reconstructed_from_the_game_count(tmp_path):
    """Dropping the stored path must not lose which weights were measured."""
    run_dir = tmp_path
    assert checkpoint_path_for_record(run_dir, {"rl_games": 100_000}).endswith(
        "checkpoints/games_0000100000_weights.npz"
    )
    # A record that still carries the stored path keeps it verbatim.
    assert checkpoint_path_for_record(
        run_dir, {"rl_games": 100, "checkpoint_path": "/somewhere/old.npz"}
    ) == "/somewhere/old.npz"


def test_the_initial_point_names_the_supervised_policy(tmp_path):
    """At zero games the run has not produced a checkpoint of its own."""
    bundle = tmp_path / "run_compact_diagnostics"
    bundle.mkdir(parents=True)
    (bundle / "run_config.json").write_text(
        json.dumps({
            "format_version": RUN_FORMAT_VERSION,
            "supervised_weights_path": "/models/domino_sl.npz",
        }),
        encoding="utf-8",
    )
    assert checkpoint_path_for_record(
        tmp_path, {"rl_games": 0}
    ) == "/models/domino_sl.npz"


# ----------------------------------------------------------------------
# Rounding
# ----------------------------------------------------------------------


def test_each_category_is_written_at_its_own_precision(tmp_path):
    path = _history(tmp_path)
    append_periodic_point(path, _row(
        100,
        wins=70,
        max_kl=0.004381234,
        clip_fraction=0.051234567,
        entropy=0.322042399,
        epochs_completed=16.0,
        R_E_mean=-0.027248999, R_E_min=-0.999, R_E_max=0.999,
        baseline_mean=0.257411234, baseline_std=0.021631234,
        G_D_mean=0.241516789,
        window_iterations=50,
    ))
    values = dict(zip(
        V5_FIELD_ORDER,
        json.loads(path.read_text(encoding="utf-8").splitlines()[1]),
    ))

    # Five places for the trust region: two would turn 0.00438 into 0.00 and
    # delete the measurement entirely.
    assert values["max_kl"] == 0.00438
    assert values["clip_fraction"] == 0.05123
    assert values["entropy"] == 0.32204
    assert values["R_E_mean"] == -0.02725
    assert values["G_D_mean"] == 0.24152
    # A terminal component's extremes only ever reach +/-1.
    assert values["R_E_min"] == -1.0
    assert values["R_E_max"] == 1.0
    # The deviation of a near-constant baseline needs the extra place.
    assert values["baseline_std"] == 0.021631
    assert values["baseline_mean"] == 0.25741
    # Integers stay integers.
    assert values["epochs_completed"] == 16
    assert isinstance(values["epochs_completed"], int)
    assert values["window_iterations"] == 50
    assert values["cumulative_games"] == 100
    assert values["wins"] == 70
    # Seconds to two places, hours to four, win rate to three.
    assert values["rl_elapsed_seconds"] == 100.0
    assert values["diagnostic_seconds"] == 0.5
    assert values["elapsed_hours"] == round((100.0 + 0.5) / 3600.0, 4)
    assert values["diagnostic_win_rate"] == 70.0
    # The timestamp loses its sub-second part.
    assert values["timestamp"] == "2026-09-04T00:00:00+00:00"


def test_the_rounding_table_covers_every_non_integer_field():
    """A new statistic must be given a precision, not silently written raw."""
    integers = {
        "cumulative_games", "iteration", "wins", "epochs_completed",
        "window_first_iteration", "window_iterations", "window_games",
        "window_decisions", "window_restart_decisions",
    }
    uncovered = [
        name for name in V5_FIELD_ORDER
        if name not in integers
        and name != "timestamp"
        and name not in RECORDED_PLACES
    ]
    assert uncovered == []


def test_an_unmeasured_statistic_reads_back_as_none(tmp_path):
    """The initial diagnostic has no training window behind it."""
    path = _history(tmp_path)
    append_periodic_point(path, _row(0))
    stored = json.loads(path.read_text(encoding="utf-8").splitlines()[1])
    values = dict(zip(V5_FIELD_ORDER, stored))
    assert values["max_kl"] is None
    assert values["R_E_mean"] is None
    assert read_periodic_history(path)[0]["max_kl"] is None


# ----------------------------------------------------------------------
# Behaviour that must survive every format change
# ----------------------------------------------------------------------


def test_deduplication_still_rejects_a_repeated_measurement(tmp_path):
    path = _history(tmp_path)
    _first, appended = append_periodic_point(path, _row(100))
    assert appended
    _second, appended = append_periodic_point(path, _row(100))
    assert not appended
    assert len(read_periodic_history(path)) == 1


def test_a_corrupt_final_line_is_tolerated(tmp_path):
    path = _history(tmp_path)
    append_periodic_point(path, _row(100))
    with open(path, "a", encoding="utf-8") as stream:
        stream.write("[100, ")
    assert len(read_periodic_history(path)) == 1


def test_a_run_without_a_metrics_trace_still_records_its_win_rate(tmp_path):
    assert summarize_training_window(
        tmp_path, first_iteration=1, last_iteration=50
    ) == {}
    path = _history(tmp_path)
    append_periodic_point(path, _row(100, wins=70))
    assert read_periodic_history(path)[0]["win_rate"] == pytest.approx(0.70)


# ----------------------------------------------------------------------
# The archive
# ----------------------------------------------------------------------


def test_the_finished_history_is_archived_beside_itself(tmp_path):
    path = _history(tmp_path)
    for games in range(0, 500_000, 100_000):
        append_periodic_point(path, _row(games))

    archive = archive_periodic_history(tmp_path)

    assert archive is not None and archive.name == "periodic_diagnostics.zip"
    # The JSONL stays in place: nothing that reads it learns about the archive.
    assert path.is_file()
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == ["periodic_diagnostics.jsonl"]
        assert bundle.read("periodic_diagnostics.jsonl") == path.read_bytes()
    assert archive.stat().st_size < path.stat().st_size


def test_archiving_a_run_with_no_history_is_a_no_op(tmp_path):
    assert archive_periodic_history(tmp_path) is None


def test_archiving_twice_refreshes_rather_than_leaving_a_stale_copy(tmp_path):
    path = _history(tmp_path)
    append_periodic_point(path, _row(0))
    archive_periodic_history(tmp_path)
    append_periodic_point(path, _row(100_000))
    archive = archive_periodic_history(tmp_path)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.read("periodic_diagnostics.jsonl") == path.read_bytes()
