"""Focused tests for fixed unit-component reward lookup construction."""

from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_fixed_signed_reward_lookup import (
    CellAccumulator,
    _normalized_cell,
    _runtime_tables,
    accumulate_sample,
    build_ruleset,
    dense_normalized_histogram,
    eligibility_threshold,
    evaluate_histogram,
)
from reward_lookup_common import LOOKUP_FORMAT, LOOKUP_FORMAT_VERSION, file_sha256


def _event(kind, turn_distance=0, decision_distance=0, decision_turn=2):
    actor, action_kind, reward = {
        "opponent_pass": ("opponent", "pass", 0.1),
        "neural_pass": ("neural", "pass", -0.1),
        "opponent_draw": ("opponent", "draw", 0.2),
        "neural_draw": ("neural", "draw", -0.2),
    }[kind]
    return {
        "kind": kind,
        "actor": actor,
        "base_reward": reward,
        "event_turn": decision_turn + turn_distance + 1,
        "turn_distance": turn_distance,
        "decision_distance": decision_distance,
        "action_kind": action_kind,
    }


def _sample(
    result="win",
    *,
    turn_distance=0,
    decision_distance=0,
    events=None,
    decision_turn=2,
):
    sign = 1.0 if result == "win" else -1.0
    return {
        "game": 1,
        "neural_seat": 0,
        "result": result,
        "decision_index": 0,
        "decision_turn": decision_turn,
        "action": [[0, 0], 0],
        "tile_option_count": 2,
        "future_local_events": list(events or []),
        "future_local_reward_sum_undiscounted": 0.0,
        "terminal": {
            "outcome_reward": sign,
            "remaining_pips": 0 if result == "win" else 3,
            "remaining_pip_penalty": 0.0 if result == "win" else -0.15,
            "base_reward": sign + (0.0 if result == "win" else -0.15),
            "terminal_turn": decision_turn + turn_distance + 1,
            "turn_distance": turn_distance,
            "decision_distance": decision_distance,
        },
    }


def _accumulate(*samples, track_direct=False):
    accumulator = CellAccumulator(track_direct=track_direct)
    for sample in samples:
        accumulate_sample(accumulator, sample)
    return accumulator


def test_immediate_terminal_signs_and_symmetric_cancellation():
    win = _accumulate(_sample("win"))
    loss = _accumulate(_sample("loss"))
    cancelled = _accumulate(_sample("win"), _sample("loss"))

    assert _normalized_cell(win, "2,3")["final"]["turn"] == [1.0]
    assert _normalized_cell(loss, "2,3")["final"]["decision"] == [-1.0]
    assert _normalized_cell(cancelled, "2,3")["final"]["turn"] == []
    assert _normalized_cell(cancelled, "2,3")["final"]["decision"] == []


def test_pips_are_unit_counts_not_the_scaled_penalty():
    win = _normalized_cell(_accumulate(_sample("win")), "2,3")
    loss = _normalized_cell(_accumulate(_sample("loss")), "2,3")
    mixed = _normalized_cell(
        _accumulate(_sample("win"), _sample("loss")),
        "2,3",
    )

    assert win["pips"]["turn"] == []
    assert loss["pips"]["turn"] == [3.0]
    assert loss["pips"]["decision"] == [3.0]
    assert mixed["pips"]["turn"] == [1.5]
    assert mixed["pips"]["decision"] == [1.5]


def test_pips_follow_both_distance_clocks_and_direct_evaluation():
    normalized = _normalized_cell(
        _accumulate(
            _sample("loss", turn_distance=5, decision_distance=2),
            track_direct=True,
        ),
        "2,3",
    )

    assert normalized["pips"]["turn"] == [0.0] * 5 + [3.0]
    assert normalized["pips"]["decision"] == [0.0, 0.0, 3.0]
    assert evaluate_histogram(normalized["pips"]["turn"], 0.5) == pytest.approx(
        3.0 * 0.5**5
    )
    assert evaluate_histogram(
        normalized["pips"]["decision"], 0.5
    ) == pytest.approx(3.0 * 0.5**2)


def test_pass_draw_signs_and_multiple_events_accumulate():
    events = [
        _event("opponent_pass", 0, 0),
        _event("opponent_pass", 2, 1),
        _event("neural_pass", 2, 1),
        _event("opponent_draw", 1, 0),
        _event("neural_draw", 3, 0),
        _event("neural_draw", 4, 2),
    ]
    accumulator = _accumulate(_sample(events=events))

    assert accumulator.histograms["pass"]["turn"] == Counter({0: 1, 2: 0})
    assert accumulator.histograms["pass"]["decision"] == Counter({0: 1, 1: 0})
    assert accumulator.histograms["draw"]["turn"] == Counter({1: 1, 3: -1, 4: -1})
    assert accumulator.histograms["draw"]["decision"] == Counter({0: 0, 2: -1})
    normalized = _normalized_cell(accumulator, "2,3")
    assert normalized["pass"]["turn"] == [1.0]
    assert normalized["draw"]["decision"] == [0.0, 0.0, -1.0]


def test_turn_and_decision_clocks_move_bins_but_not_signed_totals():
    events = [
        _event("opponent_pass", 4, 0),
        _event("opponent_pass", 7, 1),
    ]
    normalized = _normalized_cell(
        _accumulate(
            _sample(
                turn_distance=9,
                decision_distance=2,
                events=events,
            ),
            track_direct=True,
        ),
        "2,3",
    )
    assert normalized["final"]["turn"] == [0.0] * 9 + [1.0]
    assert normalized["final"]["decision"] == [0.0, 0.0, 1.0]
    assert normalized["pips"]["turn"] == []
    assert normalized["pips"]["decision"] == []
    assert sum(normalized["pass"]["turn"]) == 2.0
    assert sum(normalized["pass"]["decision"]) == 2.0


def test_eligibility_boundary_and_missing_cells_have_no_placeholders():
    assert eligibility_threshold(153_697) == 769
    assert eligibility_threshold(233_262) == 1_167
    assert eligibility_threshold(314_774) == 1_574
    assert eligibility_threshold(421_010) == 2_106

    eligible = _accumulate(_sample())
    omitted = _accumulate(_sample())
    tables = _runtime_tables(
        {"2,3": eligible, "3,2": omitted},
        {"2,3"},
    )
    for component in ("final", "pips", "pass", "draw"):
        for clock in ("turn", "decision"):
            assert set(tables[component][clock]) == {"2,3"}


@pytest.mark.parametrize("gamma", (0.0, 0.5, 0.9, 0.95, 1.0))
def test_histogram_evaluation_matches_direct_trajectory_return(gamma):
    counter = {0: 1, 1: -2, 3: 4, 7: -1}
    histogram = dense_normalized_histogram(Counter(counter), denominator=2)
    direct = sum(sign * gamma ** exponent for exponent, sign in counter.items()) / 2
    assert evaluate_histogram(histogram, gamma) == pytest.approx(direct)


def test_distances_match_the_production_rl_return_code():
    from agents.rl_agent import RLAgent, TrajectoryStep
    from training.rl.rollout import _finish_episode_with_rewards

    agent = object.__new__(RLAgent)
    agent.trajectory = [
        TrajectoryStep(np.zeros((1, 1)), 0, np.ones((1, 1)), decision_turn=2),
        TrajectoryStep(np.zeros((1, 1)), 0, np.ones((1, 1)), decision_turn=5),
    ]
    agent.add_decayed_event_reward(7, 1.0, 0.5, "turn")
    assert [step.local_reward for step in agent.trajectory] == pytest.approx([
        0.5 ** 4,
        0.5 ** 1,
    ])
    agent.add_decayed_event_reward(7, 1.0, 0.5, "decision")
    assert [step.local_reward for step in agent.trajectory] == pytest.approx([
        0.5 ** 4 + 0.5 ** 1,
        0.5 ** 1 + 0.5 ** 0,
    ])

    terminal_agent = object.__new__(RLAgent)
    terminal_agent.trajectory = [
        TrajectoryStep(np.zeros((1, 1)), 0, np.ones((1, 1)), decision_turn=2),
        TrajectoryStep(np.zeros((1, 1)), 0, np.ones((1, 1)), decision_turn=5),
    ]
    turn_samples = _finish_episode_with_rewards(
        terminal_agent,
        terminal_reward=1.0,
        gamma_f=0.5,
        reward_eta=0.0,
        terminal_turn=8,
        reward_distance_mode="turn-turn",
    )
    assert [sample.terminal_reward for sample in turn_samples] == pytest.approx([
        evaluate_histogram([0.0] * 5 + [1.0], 0.5),
        evaluate_histogram([0.0, 0.0, 1.0], 0.5),
    ])


def _write_tiny_derived_fixture(root):
    root.mkdir(parents=True)
    ruleset = "double-three"
    counts = {"2,3": 4, "3,2": 996}
    summary = {
        "games": 1_000,
        "decisions": 1_000,
        "cells": 2,
        "neural_wins": 1_000,
        "neural_losses": 0,
    }
    manifest = {
        "format": LOOKUP_FORMAT,
        "format_version": LOOKUP_FORMAT_VERSION,
        "ruleset_name": ruleset,
        "output_file": f"{ruleset}_reward_lookup_samples.json.gz",
        "output_bytes": 0,
        "output_sha256": "fixture",
        "source_raw_manifest_sha256": "fixture",
        "summary": summary,
        "cell_sample_counts": counts,
    }
    cells = {}
    for key, count in counts.items():
        agent_size, opponent_size = (int(value) for value in key.split(","))
        cells[key] = {
            "neural_hand_size": agent_size,
            "opponent_hand_size": opponent_size,
            "sample_count": count,
            "samples": [_sample() for _ in range(count)],
        }
    payload = {
        "format": LOOKUP_FORMAT,
        "format_version": LOOKUP_FORMAT_VERSION,
        "ruleset_name": ruleset,
        "matchup": "neural_vs_heuristic",
        "key_fields": ["neural_hand_size", "opponent_hand_size"],
        "key_encoding": "neural_hand_size,opponent_hand_size",
        "action_is_part_of_key": False,
        "summary": summary,
        "cells": cells,
    }
    sample_path = root / manifest["output_file"]
    with sample_path.open("wb") as raw_stream:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_stream,
            mtime=0,
        ) as stream:
            stream.write(
                json.dumps(payload, separators=(",", ":")).encode("utf-8")
            )
    manifest["output_bytes"] = sample_path.stat().st_size
    manifest["output_sha256"] = file_sha256(sample_path)
    manifest_path = root / f"{ruleset}_reward_lookup_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return ruleset


def test_end_to_end_output_is_deterministic_fixed_and_count_free(tmp_path):
    derived = tmp_path / "derived"
    output = tmp_path / "fixed"
    ruleset = _write_tiny_derived_fixture(derived)
    first = build_ruleset(derived, output, ruleset)
    first_hash = file_sha256(output / first["output_file"])
    second = build_ruleset(derived, output, ruleset, force=True)
    assert file_sha256(output / second["output_file"]) == first_hash

    with gzip.open(output / first["output_file"], "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    assert payload["format"] == "domino_fixed_signed_reward_lookup"
    assert payload["format_version"] == 2
    assert payload["component_semantics"]["pips"] == (
        "nonnegative_terminal_remaining_pip_count"
    )
    assert "ruleset_decisions" not in payload
    assert "eligibility_threshold" not in payload
    for component in ("final", "pips", "pass", "draw"):
        for clock in ("turn", "decision"):
            assert set(payload["tables"][component][clock]) == {"3,2"}
