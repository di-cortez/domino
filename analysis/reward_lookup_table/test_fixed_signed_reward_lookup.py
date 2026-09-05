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


# ``m(Delta_p) = 0.1 + 0.9 * min(Delta_p / (2 * max_pip), 1)`` at a pip margin
# of three under ``double-three``, which is the ruleset every fixture uses.
BLOCKED_MAGNITUDE = 0.55


def _event(kind, turn_distance=0, decision_distance=0, decision_turn=2):
    actor, action_kind, unit_reward = {
        "opponent_pass": ("opponent", "pass", 1.0),
        "neural_pass": ("neural", "pass", -1.0),
        "opponent_draw": ("opponent", "draw", 1.0),
        "neural_draw": ("neural", "draw", -1.0),
    }[kind]
    return {
        "kind": kind,
        "actor": actor,
        "unit_reward": unit_reward,
        "event_turn": decision_turn + turn_distance + 1,
        "turn_distance": turn_distance,
        "decision_distance": decision_distance,
        "action_kind": action_kind,
    }


def _terminal(result, ending, turn_distance, decision_distance, decision_turn):
    """Build one unit terminal decomposition for a win or a loss.

    Exactly one of the two components is non-zero, which is the invariant the
    builder enforces: an ending is either an empty hand or a block.
    """
    sign = 1.0 if result == "win" else -1.0
    blocked = ending == "blocked"
    return {
        "win_reason": "blocked_fewest_pips" if blocked else "empty_hand",
        "learner_won": result == "win",
        "empty_hand_component": 0.0 if blocked else sign,
        "blocked_component": sign * BLOCKED_MAGNITUDE if blocked else 0.0,
        "winner_final_pips": 3 if blocked else 0,
        "loser_final_pips": 6 if blocked else 3,
        "pip_margin": 3 if blocked else None,
        "blocked_magnitude": BLOCKED_MAGNITUDE if blocked else None,
        "terminal_turn": decision_turn + turn_distance + 1,
        "turn_distance": turn_distance,
        "decision_distance": decision_distance,
    }


def _sample(
    result="win",
    *,
    ending="empty_hand",
    turn_distance=0,
    decision_distance=0,
    events=None,
    decision_turn=2,
):
    return {
        "game": 1,
        "neural_seat": 0,
        "result": result,
        "decision_index": 0,
        "decision_turn": decision_turn,
        "action": [[0, 0], 0],
        "tile_option_count": 2,
        "future_local_events": list(events or []),
        "future_local_unit_sum_undiscounted": 0.0,
        "terminal": _terminal(
            result,
            ending,
            turn_distance,
            decision_distance,
            decision_turn,
        ),
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

    assert _normalized_cell(win, "2,3")["empty_hand"]["turn"] == [1.0]
    assert _normalized_cell(loss, "2,3")["empty_hand"]["decision"] == [-1.0]
    assert _normalized_cell(cancelled, "2,3")["empty_hand"]["turn"] == []
    assert _normalized_cell(cancelled, "2,3")["empty_hand"]["decision"] == []


def test_each_ending_populates_exactly_one_terminal_component():
    empty_hand = _normalized_cell(_accumulate(_sample("win")), "2,3")
    blocked = _normalized_cell(
        _accumulate(_sample("win", ending="blocked")),
        "2,3",
    )

    assert empty_hand["empty_hand"]["turn"] == [1.0]
    assert empty_hand["blocked"]["turn"] == []
    assert blocked["empty_hand"]["turn"] == []
    assert blocked["blocked"]["turn"] == [pytest.approx(BLOCKED_MAGNITUDE)]


def test_blocked_component_stores_the_signed_margin_utility_not_a_sign():
    loss = _normalized_cell(
        _accumulate(_sample("loss", ending="blocked")),
        "2,3",
    )
    mixed = _normalized_cell(
        _accumulate(
            _sample("win", ending="blocked"),
            _sample("loss", ending="empty_hand"),
        ),
        "2,3",
    )

    assert loss["blocked"]["turn"] == [pytest.approx(-BLOCKED_MAGNITUDE)]
    assert loss["blocked"]["decision"] == [pytest.approx(-BLOCKED_MAGNITUDE)]
    # One blocked win and one empty-hand loss populate different components
    # and are each halved by the two-sample cell denominator.
    assert mixed["blocked"]["turn"] == [pytest.approx(BLOCKED_MAGNITUDE / 2)]
    assert mixed["empty_hand"]["turn"] == [pytest.approx(-0.5)]


def test_blocked_follows_both_distance_clocks_and_direct_evaluation():
    normalized = _normalized_cell(
        _accumulate(
            _sample(
                "loss",
                ending="blocked",
                turn_distance=5,
                decision_distance=2,
            ),
            track_direct=True,
        ),
        "2,3",
    )

    assert normalized["blocked"]["turn"] == [0.0] * 5 + [
        pytest.approx(-BLOCKED_MAGNITUDE)
    ]
    assert normalized["blocked"]["decision"] == [
        0.0,
        0.0,
        pytest.approx(-BLOCKED_MAGNITUDE),
    ]
    assert evaluate_histogram(
        normalized["blocked"]["turn"], 0.5
    ) == pytest.approx(-BLOCKED_MAGNITUDE * 0.5**5)
    assert evaluate_histogram(
        normalized["blocked"]["decision"], 0.5
    ) == pytest.approx(-BLOCKED_MAGNITUDE * 0.5**2)


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
    assert normalized["empty_hand"]["turn"] == [0.0] * 9 + [1.0]
    assert normalized["empty_hand"]["decision"] == [0.0, 0.0, 1.0]
    assert normalized["blocked"]["turn"] == []
    assert normalized["blocked"]["decision"] == []
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
    for component in ("empty_hand", "blocked", "pass", "draw"):
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
    from training.rl.reward_model import DRAW_EVENT
    from training.rl.rollout import _finish_episode_with_rewards

    agent = object.__new__(RLAgent)
    agent.trajectory = [
        TrajectoryStep(np.zeros((1, 1)), 0, np.ones((1, 1)), decision_turn=2),
        TrajectoryStep(np.zeros((1, 1)), 0, np.ones((1, 1)), decision_turn=5),
    ]
    agent.add_decayed_event_reward(7, 1.0, 0.5, "turn", event_kind=DRAW_EVENT)
    assert [step.local_reward for step in agent.trajectory] == pytest.approx([
        0.5 ** 4,
        0.5 ** 1,
    ])
    agent.add_decayed_event_reward(7, 1.0, 0.5, "decision", event_kind=DRAW_EVENT)
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
        terminal_utility=1.0,
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
    assert payload["format_version"] == 3
    assert payload["component_semantics"]["blocked"] == (
        "signed_blocked_terminal_margin_utility"
    )
    assert "ruleset_decisions" not in payload
    assert "eligibility_threshold" not in payload
    for component in ("empty_hand", "blocked", "pass", "draw"):
        for clock in ("turn", "decision"):
            assert set(payload["tables"][component][clock]) == {"3,2"}
