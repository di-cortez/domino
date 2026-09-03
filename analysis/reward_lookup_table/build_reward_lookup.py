"""Build hand-size keyed future-reward sample lists from full raw histories."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reward_lookup_common import (
    LOOKUP_FORMAT,
    LOOKUP_FORMAT_VERSION,
    RAW_FORMAT,
    RAW_FORMAT_VERSION,
    atomic_write_json,
    canonical_json_sha256,
    file_sha256,
    parse_ruleset_selection,
    read_gzip_json_lines,
)


DEFAULT_RAW_ROOT = SCRIPT_DIR / "raw"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "derived"


def _load_raw_manifest(raw_root, ruleset_name):
    path = raw_root / ruleset_name / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing complete raw manifest: {path}")
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    configuration = manifest.get("configuration", {})
    if configuration.get("format") != RAW_FORMAT:
        raise ValueError(f"Unexpected raw format in {path}")
    if int(configuration.get("format_version", -1)) != RAW_FORMAT_VERSION:
        raise ValueError(f"Unsupported raw format version in {path}")
    if configuration.get("ruleset_name") != ruleset_name:
        raise ValueError(f"Raw ruleset mismatch in {path}")
    if manifest.get("status") != "complete":
        raise ValueError(f"Raw generation is not complete in {path}")
    expected_hash = canonical_json_sha256(configuration)
    if manifest.get("configuration_sha256") != expected_hash:
        raise ValueError(f"Raw configuration hash mismatch in {path}")
    return path, manifest


def _validate_game_record(record, ruleset_name):
    turns = record["turns"]
    if [int(turn["turn"]) for turn in turns] != list(range(len(turns))):
        raise ValueError(f"Non-contiguous turns in game {record.get('game')}")
    final_state = record["final_state"]
    if final_state["ruleset_name"] != ruleset_name:
        raise ValueError(f"Game {record.get('game')} has the wrong ruleset")
    if int(final_state["turn"]) != len(turns):
        raise ValueError(f"Final turn mismatch in game {record.get('game')}")
    if final_state["winner"] not in (0, 1):
        raise ValueError(f"Game {record.get('game')} has no unique winner")
    expected_decisions = list(range(int(record["neural_real_decisions"])))
    actual_decisions = [
        int(turn["neural_decision_index"])
        for turn in turns
        if turn["neural_decision_index"] is not None
    ]
    if actual_decisions != expected_decisions:
        raise ValueError(f"Neural decision chronology mismatch in game {record.get('game')}")


def _future_local_events(turns, decision_turn, decision_index):
    events = []
    for turn in turns:
        event = turn["local_event_for_neural"]
        if event is None or int(turn["turn"]) <= int(decision_turn):
            continue
        decision_distance = (
            int(turn["neural_decisions_before"])
            - int(decision_index)
            - 1
        )
        turn_distance = int(turn["turn"]) - int(decision_turn) - 1
        if decision_distance < 0 or turn_distance < 0:
            raise ValueError("Future local reward has a negative distance")
        events.append({
            "kind": event["kind"],
            "actor": event["actor"],
            "unit_reward": float(event["unit_reward"]),
            "event_turn": int(turn["turn"]),
            "turn_distance": int(turn_distance),
            "decision_distance": int(decision_distance),
        })
    return events


def _sample_from_decision(record, turn):
    neural_seat = int(record["neural_seat"])
    opponent_seat = 1 - neural_seat
    hand_sizes = turn["pre"]["hand_sizes"]
    decision_index = int(turn["neural_decision_index"])
    decision_turn = int(turn["turn"])
    local_events = _future_local_events(
        record["turns"],
        decision_turn,
        decision_index,
    )
    terminal_component = record["raw_rewards_for_neural"]["terminal"]
    terminal_turn = int(record["final_state"]["turn"])
    terminal_decision_distance = (
        int(record["neural_real_decisions"]) - decision_index - 1
    )
    terminal_turn_distance = terminal_turn - decision_turn - 1
    if terminal_decision_distance < 0 or terminal_turn_distance < 0:
        raise ValueError("Terminal reward has a negative distance")
    sample = {
        "game": int(record["game"]),
        "neural_seat": neural_seat,
        "result": record["result"],
        "decision_index": decision_index,
        "decision_turn": decision_turn,
        # The action remains auditable in each observation but deliberately is
        # not part of the lookup key requested for this experiment.
        "action": turn["action"],
        "tile_option_count": int(turn["tile_option_count"]),
        "future_local_events": local_events,
        "future_local_unit_sum_undiscounted": float(
            sum(event["unit_reward"] for event in local_events)
        ),
        "terminal": {
            "win_reason": terminal_component["win_reason"],
            "learner_won": bool(terminal_component["learner_won"]),
            "empty_hand_component": float(
                terminal_component["empty_hand_component"]
            ),
            "blocked_component": float(
                terminal_component["blocked_component"]
            ),
            "winner_final_pips": int(terminal_component["winner_final_pips"]),
            "loser_final_pips": int(terminal_component["loser_final_pips"]),
            "pip_margin": (
                None
                if terminal_component["pip_margin"] is None
                else int(terminal_component["pip_margin"])
            ),
            "blocked_magnitude": (
                None
                if terminal_component["blocked_magnitude"] is None
                else float(terminal_component["blocked_magnitude"])
            ),
            "terminal_turn": terminal_turn,
            "turn_distance": int(terminal_turn_distance),
            "decision_distance": int(terminal_decision_distance),
        },
    }
    key = (int(hand_sizes[neural_seat]), int(hand_sizes[opponent_seat]))
    return key, sample


def _write_lookup_json_gz(destination, header, cell_files, counts):
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as raw_stream:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=6,
                fileobj=raw_stream,
                mtime=0,
            ) as compressed:
                compressed.write(b"{")
                first = True
                for key, value in header.items():
                    if not first:
                        compressed.write(b",")
                    first = False
                    compressed.write(json.dumps(key).encode("utf-8"))
                    compressed.write(b":")
                    compressed.write(
                        json.dumps(
                            value,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8")
                    )
                compressed.write(b',"cells":{')
                first_cell = True
                for key in sorted(counts):
                    if not first_cell:
                        compressed.write(b",")
                    first_cell = False
                    key_text = f"{key[0]},{key[1]}"
                    compressed.write(json.dumps(key_text).encode("utf-8"))
                    compressed.write(b":{")
                    compressed.write(
                        (
                            f'"neural_hand_size":{key[0]},'
                            f'"opponent_hand_size":{key[1]},'
                            f'"sample_count":{counts[key]},'
                            '"samples":['
                        ).encode("utf-8")
                    )
                    with cell_files[key].open("rb") as samples:
                        first_sample = True
                        for line in samples:
                            line = line.strip()
                            if not line:
                                continue
                            if not first_sample:
                                compressed.write(b",")
                            first_sample = False
                            compressed.write(line)
                    compressed.write(b"]}")
                compressed.write(b"}}")
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_ruleset(args, ruleset_name):
    manifest_path, raw_manifest = _load_raw_manifest(args.raw_root, ruleset_name)
    destination = args.output_root / f"{ruleset_name}_reward_lookup_samples.json.gz"
    sidecar = args.output_root / f"{ruleset_name}_reward_lookup_manifest.json"
    if (destination.exists() or sidecar.exists()) and not args.force:
        raise FileExistsError(
            f"Derived output already exists for {ruleset_name}; pass --force to replace it."
        )
    args.output_root.mkdir(parents=True, exist_ok=True)

    counts = {}
    result_counts = {"win": 0, "loss": 0}
    game_count = 0
    decision_count = 0
    with tempfile.TemporaryDirectory(
        prefix=f".{ruleset_name}-lookup-",
        dir=args.output_root,
    ) as temporary_name:
        temporary_dir = Path(temporary_name)
        cell_paths = {}
        streams = {}
        try:
            for chunk in raw_manifest["chunks"]:
                chunk_path = manifest_path.parent / chunk["file"]
                if file_sha256(chunk_path) != chunk["compressed_sha256"]:
                    raise ValueError(f"Raw chunk checksum mismatch: {chunk_path}")
                for record in read_gzip_json_lines(chunk_path):
                    _validate_game_record(record, ruleset_name)
                    game_count += 1
                    result_counts[record["result"]] += 1
                    for turn in record["turns"]:
                        if turn["neural_decision_index"] is None:
                            continue
                        key, sample = _sample_from_decision(record, turn)
                        if key not in streams:
                            path = temporary_dir / f"cell-{key[0]}-{key[1]}.jsonl"
                            cell_paths[key] = path
                            streams[key] = path.open("wb")
                            counts[key] = 0
                        payload = json.dumps(
                            sample,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode("utf-8")
                        streams[key].write(payload + b"\n")
                        counts[key] += 1
                        decision_count += 1
        finally:
            for stream in streams.values():
                stream.close()

        expected = raw_manifest["summary"]
        if game_count != int(expected["games"]):
            raise ValueError("Derived game count differs from raw manifest")
        if decision_count != int(expected["neural_real_decisions"]):
            raise ValueError("Derived decision count differs from raw manifest")

        header = {
            "format": LOOKUP_FORMAT,
            "format_version": LOOKUP_FORMAT_VERSION,
            "ruleset_name": ruleset_name,
            "matchup": "neural_vs_heuristic",
            "key_fields": ["neural_hand_size", "opponent_hand_size"],
            "key_encoding": "neural_hand_size,opponent_hand_size",
            "action_is_part_of_key": False,
            "sample_semantics": (
                "Every genuine neural decision keeps its observed action, all "
                "later unit draw/pass events, and the unit terminal "
                "empty-hand/blocked decomposition. No reward weight, gamma_i, "
                "gamma_f, or reward_eta has been applied."
            ),
            "distance_fields": {
                "turn_distance": "intervening engine actions",
                "decision_distance": "intervening genuine neural decisions",
            },
            "reward_semantics": raw_manifest["configuration"][
                "reward_semantics"
            ],
            "neural_policy": raw_manifest["configuration"]["neural_policy"],
            "source": {
                "raw_manifest_sha256": file_sha256(manifest_path),
                "raw_configuration_sha256": raw_manifest[
                    "configuration_sha256"
                ],
                "raw_chunk_count": len(raw_manifest["chunks"]),
            },
            "summary": {
                "games": int(game_count),
                "decisions": int(decision_count),
                "cells": len(counts),
                "neural_wins": int(result_counts["win"]),
                "neural_losses": int(result_counts["loss"]),
            },
        }
        _write_lookup_json_gz(destination, header, cell_paths, counts)

    output_manifest = {
        "format": LOOKUP_FORMAT,
        "format_version": LOOKUP_FORMAT_VERSION,
        "ruleset_name": ruleset_name,
        "output_file": destination.name,
        "output_bytes": destination.stat().st_size,
        "output_sha256": file_sha256(destination),
        "source_raw_manifest_sha256": file_sha256(manifest_path),
        "summary": header["summary"],
        "cell_sample_counts": {
            f"{key[0]},{key[1]}": int(counts[key])
            for key in sorted(counts)
        },
    }
    atomic_write_json(sidecar, output_manifest)
    print(
        f"{ruleset_name}: {decision_count:,} decision samples in "
        f"{len(counts)} hand-size cells -> {destination}"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rulesets", default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    args.raw_root = args.raw_root.resolve()
    args.output_root = args.output_root.resolve()
    available = tuple(
        path.name
        for path in args.raw_root.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    ) if args.raw_root.is_dir() else ()
    try:
        args.selected_rulesets = parse_ruleset_selection(
            args.rulesets,
            available=available,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    for ruleset_name in args.selected_rulesets:
        build_ruleset(args, ruleset_name)


if __name__ == "__main__":
    main()
