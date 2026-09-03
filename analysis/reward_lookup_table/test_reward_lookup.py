"""Focused deterministic tests for the reward lookup analysis pipeline."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_reward_lookup import main as build_main
from generate_raw_histories import main as generate_main
from reward_lookup_common import (
    deserialize_action,
    discover_checkpoints,
    file_sha256,
    read_gzip_json_lines,
)


def _write_test_checkpoints(directory, rulesets=("double-three",)):
    """Create tiny deterministic policies instead of requiring analysis data."""
    from agents.encoder import DominoEncoder
    from agents.rl_nn import PolicyNetwork

    directory.mkdir(parents=True, exist_ok=True)
    for index, ruleset_name in enumerate(rulesets):
        encoder = DominoEncoder(ruleset_name)
        network = PolicyNetwork(
            input_size=encoder.vector_size,
            output_size=encoder.action_size,
            hidden_sizes=(4,),
            random_seed=100 + index,
            device="cpu",
        )
        network.save(str(directory / f"{ruleset_name}.npz"))
    return directory


def _generate(raw_root, workers, weights_dir):
    generate_main([
        "--games", "12",
        "--workers", str(workers),
        "--chunk-games", "4",
        "--job-games", "2",
        "--rulesets", "double-three",
        "--weights-dir", str(weights_dir),
        "--raw-root", str(raw_root),
        "--force",
    ])


def _artifact_hashes(root):
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_all_four_default_checkpoints_are_discovered(tmp_path):
    weights_dir = _write_test_checkpoints(
        tmp_path / "weights",
        ("double-three", "double-four", "double-five", "double-six"),
    )
    checkpoints = discover_checkpoints(weights_dir)
    assert set(checkpoints) == {
        "double-three", "double-four", "double-five", "double-six"
    }
    assert all(not item["has_value_head"] for item in checkpoints.values())


def test_raw_is_worker_invariant_and_lookup_keeps_hand_size_only_key(tmp_path):
    raw_one = tmp_path / "raw-one"
    raw_three = tmp_path / "raw-three"
    weights_dir = _write_test_checkpoints(tmp_path / "weights")
    _generate(raw_one, workers=1, weights_dir=weights_dir)
    _generate(raw_three, workers=3, weights_dir=weights_dir)
    assert _artifact_hashes(raw_one) == _artifact_hashes(raw_three)

    output = tmp_path / "derived"
    build_main([
        "--raw-root", str(raw_one),
        "--output-root", str(output),
        "--rulesets", "double-three",
    ])
    lookup_path = output / "double-three_reward_lookup_samples.json.gz"
    with gzip.open(lookup_path, "rt", encoding="utf-8") as stream:
        lookup = json.load(stream)
    assert lookup["key_fields"] == [
        "neural_hand_size", "opponent_hand_size"
    ]
    assert lookup["action_is_part_of_key"] is False
    assert sum(
        cell["sample_count"] for cell in lookup["cells"].values()
    ) == lookup["summary"]["decisions"]
    assert all(
        set(key.split(",")) <= {"1", "2", "3", "4", "5"}
        for key in lookup["cells"]
    )
    first_sample = next(
        sample
        for cell in lookup["cells"].values()
        for sample in cell["samples"]
    )
    assert "action" in first_sample
    assert "future_local_events" in first_sample
    assert "terminal" in first_sample


def test_event_sourced_history_replays_to_exact_final_state(tmp_path):
    raw_root = tmp_path / "raw"
    weights_dir = _write_test_checkpoints(tmp_path / "weights")
    _generate(raw_root, workers=2, weights_dir=weights_dir)
    manifest_path = raw_root / "double-three" / "manifest.json"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    first_chunk = manifest_path.parent / manifest["chunks"][0]["file"]
    record = next(read_gzip_json_lines(first_chunk))

    from middleware.domino_engine import DominoEngine
    from utils.myrandom import RandomNamespace, SeedPlan

    game_index = int(record["game_index"])
    configuration = manifest["configuration"]
    generator = SeedPlan(
        int(configuration["base_seed"])
    ).generator(
        RandomNamespace.DIAGNOSTIC_GAME,
        "double-three",
        game_index,
    )
    engine = DominoEngine(rng=generator, ruleset="double-three")
    engine.game_id = game_index + 1
    assert engine.to_dict() == record["initial_state"]

    for turn in record["turns"]:
        assert [len(hand) for hand in engine.hands] == turn["pre"]["hand_sizes"]
        legal_actions = engine.valid_actions(engine.current_player)
        assert [
            deserialize_action(action) for action in turn["legal_actions"]
        ] == legal_actions
        action = deserialize_action(turn["action"])
        if action == ("DRAW", None):
            assert list(engine.stock[0]) == turn["drawn_tile"]
        engine.step(action, return_state=False, legal_actions=legal_actions)
        assert [len(hand) for hand in engine.hands] == turn["post"]["hand_sizes"]
    assert engine.to_dict() == record["final_state"]
