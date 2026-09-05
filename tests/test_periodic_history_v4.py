"""Contracts for the v4 periodic-diagnostic record and its v3 compatibility.

Two eras flow through one reader, so every consumer downstream is
version-blind. These tests pin that: a v3 file keeps its curve, its derived
statistics and its rebuilt CSV, while a v4 file additionally carries the
training-window summary.
"""

from __future__ import annotations

import json

import pytest

from diagnostics.rl_progress import (
    FORMAT_VERSION,
    HISTORY_CHECKPOINT_BASE,
    HISTORY_DATA_FIELDS,
    HISTORY_RECORD_TYPE,
    LEGACY_HISTORY_VERSION,
    LEGACY_V3_DATA_FIELDS,
    _point_key,
    append_periodic_point,
    read_periodic_history,
    summarize_training_window,
    v4_recorded_field_names,
)


def _row(rl_games, *, iteration=None, wins=60, **extra):
    row = {
        "pipeline_level": "small",
        "seed": 42,
        "rl_games": rl_games,
        "rl_iterations": iteration if iteration is not None else rl_games // 2,
        "checkpoint_path": f"checkpoints/games_{rl_games:010d}_weights.npz",
        "configuration_sha256": "cfg",
        "opponent": "random",
        "diagnostic_games": 100,
        "wins": wins,
        "diagnostic_seed": 7,
        "diagnostic_seed_namespace": "periodic_rl_vs_random",
        "diagnostic_seconds": 0.5,
        "rl_elapsed_seconds": float(rl_games),
        "selected_workers": 2,
        "created_at": "2026-09-04T00:00:00+00:00",
        "ruleset_name": "double-three",
    }
    row.update(extra)
    return row


def _write_v3(path, rows):
    """Write a file in the superseded positional layout."""
    header = {
        "record_type": HISTORY_RECORD_TYPE,
        "format_version": LEGACY_HISTORY_VERSION,
        "checkpoint_path_base": HISTORY_CHECKPOINT_BASE,
        "columns": list(LEGACY_V3_DATA_FIELDS),
        "static": {
            "pipeline_level": "small",
            "ruleset_name": "double-three",
            "seed": 42,
            "opponent": "random",
            "diagnostic_games": 100,
            "diagnostic_seed": 7,
            "diagnostic_seed_namespace": "periodic_rl_vs_random",
            "configuration_sha256": "cfg",
        },
    }
    lines = [json.dumps(header)]
    for row in rows:
        stored = dict(row)
        stored["checkpoint_path"] = stored["checkpoint_path"].split("/")[-1]
        lines.append(
            json.dumps([stored.get(name) for name in LEGACY_V3_DATA_FIELDS])
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _history(tmp_path):
    return tmp_path / "run_compact_diagnostics" / "periodic_diagnostics.jsonl"


# ----------------------------------------------------------------------
# v3 keeps working
# ----------------------------------------------------------------------


def test_a_v3_file_still_reads_with_every_derived_value(tmp_path):
    path = _history(tmp_path)
    _write_v3(path, [
        _row(0, wins=60, checkpoint_sha256="aa" * 32),
        _row(100, wins=70, checkpoint_sha256="bb" * 32),
    ])
    rows = read_periodic_history(path)

    assert [row["rl_games"] for row in rows] == [0, 100]
    assert rows[0]["win_rate"] == pytest.approx(0.60)
    assert rows[1]["losses"] == 30
    assert rows[1]["ci95_win_rate_low"] < 0.70 < rows[1]["ci95_win_rate_high"]
    # A v3 record's hash is passed through, so older runs stay auditable.
    assert rows[0]["checkpoint_sha256"] == "aa" * 32
    # The window statistics simply are not there.
    assert "R_E_mean" not in rows[0]


def test_a_v3_record_and_its_v4_equivalent_share_one_identity(tmp_path):
    """Deduplication must not regress across the format change."""
    path = _history(tmp_path)
    _write_v3(path, [_row(0, checkpoint_sha256="aa" * 32)])
    legacy = read_periodic_history(path)[0]
    modern = _row(0)
    assert _point_key(legacy) == _point_key(modern)


# ----------------------------------------------------------------------
# v4 round trip
# ----------------------------------------------------------------------


def test_v4_records_are_self_describing_objects(tmp_path):
    path = _history(tmp_path)
    append_periodic_point(path, _row(0))
    append_periodic_point(path, _row(100, wins=70))

    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    assert header["format_version"] == FORMAT_VERSION
    assert "columns" not in header
    for line in lines[1:]:
        record = json.loads(line)
        assert isinstance(record, dict)
        assert set(v4_recorded_field_names()).issubset(record)
        assert "checkpoint_sha256" not in record


def test_v4_stores_the_window_statistics_it_was_given(tmp_path):
    path = _history(tmp_path)
    append_periodic_point(path, _row(
        100,
        window_iterations=50,
        window_decisions=1234,
        R_E_mean=-0.02,
        R_E_std=0.99,
        G_D_mean=0.24,
        baseline_mean=0.25,
        entropy=0.068,
        max_kl=0.0037,
    ))
    stored = json.loads(path.read_text(encoding="utf-8").splitlines()[1])
    assert stored["window_iterations"] == 50
    assert stored["R_E_mean"] == -0.02
    assert stored["max_kl"] == 0.0037
    # An unmeasured statistic is omitted rather than written as a null.
    assert "R_B_mean" not in stored

    restored = read_periodic_history(path)[0]
    assert restored["G_D_mean"] == 0.24
    assert restored["entropy"] == 0.068


def test_a_run_without_a_metrics_trace_still_records_its_win_rate(tmp_path):
    """The diagnostic must never be lost to a missing metrics file."""
    assert summarize_training_window(
        tmp_path, first_iteration=1, last_iteration=50
    ) == {}

    path = _history(tmp_path)
    append_periodic_point(path, _row(100, wins=70))
    rows = read_periodic_history(path)
    assert rows[0]["win_rate"] == pytest.approx(0.70)


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
        stream.write('{"rl_games": ')
    assert len(read_periodic_history(path)) == 1
