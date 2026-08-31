"""Shared contracts for neural-versus-heuristic reward lookup analysis."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
RAW_FORMAT = "domino_neural_heuristic_full_history"
RAW_FORMAT_VERSION = 1
LOOKUP_FORMAT = "domino_hand_size_reward_lookup_samples"
LOOKUP_FORMAT_VERSION = 1
DEFAULT_RULESET_ORDER = (
    "double-three",
    "double-four",
    "double-five",
    "double-six",
)
_WEIGHT_PATTERN = re.compile(r"^W([1-9][0-9]*)$")


def file_sha256(path):
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value):
    """Hash one JSON-compatible value independently of presentation."""
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path, value):
    """Write indented JSON beside its destination and publish atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_deterministic_gzip_lines(path, lines, compresslevel=6):
    """Atomically write deterministic gzip members from UTF-8 JSON lines."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.tmp"
    )
    uncompressed_digest = hashlib.sha256()
    line_count = 0
    try:
        with temporary.open("wb") as raw_stream:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=int(compresslevel),
                fileobj=raw_stream,
                mtime=0,
            ) as compressed:
                for line in lines:
                    payload = line.encode("utf-8")
                    if not payload.endswith(b"\n"):
                        payload += b"\n"
                    compressed.write(payload)
                    uncompressed_digest.update(payload)
                    line_count += 1
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "bytes": destination.stat().st_size,
        "compressed_sha256": file_sha256(destination),
        "records": line_count,
        "uncompressed_sha256": uncompressed_digest.hexdigest(),
    }


def read_gzip_json_lines(path):
    """Yield decoded JSON objects from a gzip JSONL file."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc


def serialize_action(action):
    """Return the engine's stable JSON action representation."""
    if action is None:
        return None
    if action == ("DRAW", None):
        return ["DRAW", None]
    tile, side = action
    return [[int(tile[0]), int(tile[1])], int(side)]


def deserialize_action(value):
    """Restore one serialized engine action."""
    if value is None:
        return None
    if value == ["DRAW", None]:
        return ("DRAW", None)
    return (tuple(int(pip) for pip in value[0]), int(value[1]))


def action_kind(action):
    """Classify one engine action without inspecting policy internals."""
    if action is None:
        return "pass"
    if action == ("DRAW", None):
        return "draw"
    return "tile_play"


def tile_option_count(legal_actions):
    """Count legal policy tile actions, excluding draw and pass."""
    return sum(
        action is not None and action != ("DRAW", None)
        for action in legal_actions
    )


def decision_class(action, option_count):
    """Separate genuine neural choices from forced engine actions."""
    kind = action_kind(action)
    if kind == "draw":
        return "forced_draw"
    if kind == "pass":
        return "forced_pass"
    if int(option_count) <= 1:
        return "forced_tile"
    return "voluntary_choice"


def raw_reward_schema():
    """Read the exact current RL reward constants without applying discounts."""
    from training.rl.rollout import (  # Imported lazily for CPU worker safety.
        FINAL_PIP_PENALTY,
        LEARNER_DRAW_PENALTY,
        LEARNER_PASS_PENALTY,
        OPPONENT_DRAW_REWARD,
        OPPONENT_PASS_REWARD,
        TERMINAL_LOSS_REWARD,
        TERMINAL_WIN_REWARD,
    )

    return {
        "terminal_win": float(TERMINAL_WIN_REWARD),
        "terminal_loss": float(TERMINAL_LOSS_REWARD),
        "final_pip_penalty_per_pip": -float(FINAL_PIP_PENALTY),
        "neural_draw": float(LEARNER_DRAW_PENALTY),
        "opponent_draw": float(OPPONENT_DRAW_REWARD),
        "neural_pass": float(LEARNER_PASS_PENALTY),
        "opponent_pass": float(OPPONENT_PASS_REWARD),
        "gamma_i_applied": False,
        "gamma_f_applied": False,
        "reward_eta_applied": False,
    }


def local_event_for_action(action, acting_seat, neural_seat, reward_schema):
    """Return one raw local event from the neural policy's perspective."""
    kind = action_kind(action)
    actor = "neural" if int(acting_seat) == int(neural_seat) else "opponent"
    if kind not in {"draw", "pass"}:
        return None
    event_kind = f"{actor}_{kind}"
    return {
        "kind": event_kind,
        "actor": actor,
        "action_kind": kind,
        "base_reward": float(reward_schema[event_kind]),
    }


def terminal_reward_for_neural(final_state, neural_seat, reward_schema):
    """Return terminal outcome and remaining-pip components separately."""
    neural_seat = int(neural_seat)
    neural_won = int(final_state["winner"]) == neural_seat
    outcome = float(
        reward_schema["terminal_win"]
        if neural_won
        else reward_schema["terminal_loss"]
    )
    remaining_pips = sum(
        int(tile[0]) + int(tile[1])
        for tile in final_state["hands"][neural_seat]
    )
    pip_penalty = (
        float(reward_schema["final_pip_penalty_per_pip"])
        * remaining_pips
    )
    return {
        "outcome_reward": outcome,
        "remaining_pips": int(remaining_pips),
        "remaining_pip_penalty": float(pip_penalty),
        "base_reward": float(outcome + pip_penalty),
    }


def _checkpoint_structure(path):
    """Inspect a policy checkpoint without constructing a compute backend."""
    with np.load(path, allow_pickle=False) as data:
        layer_indices = sorted(
            int(match.group(1))
            for key in data.files
            if (match := _WEIGHT_PATTERN.match(key))
        )
        if not layer_indices or layer_indices != list(
            range(1, max(layer_indices) + 1)
        ):
            raise ValueError(f"Checkpoint {path} has non-contiguous W layers")
        last_index = layer_indices[-1]
        first_weight = np.asarray(data["W1"])
        last_weight = np.asarray(data[f"W{last_index}"])
        return {
            "input_size": int(first_weight.shape[1]),
            "output_size": int(last_weight.shape[0]),
            "hidden_sizes": [
                int(np.asarray(data[f"W{index}"]).shape[0])
                for index in layer_indices[:-1]
            ],
            "layer_count": int(last_index),
            "dtype": str(first_weight.dtype),
            "has_value_head": "Wv" in data.files and "bv" in data.files,
            "optimizer_step_count": (
                int(np.asarray(data["optimizer_step_count"]).item())
                if "optimizer_step_count" in data.files
                else None
            ),
            "training_algorithm": (
                str(np.asarray(data["rl_training_algorithm"]).item())
                if "rl_training_algorithm" in data.files
                else None
            ),
        }


def expected_default_checkpoint_shapes():
    """Return default encoder input/output dimensions for every ruleset."""
    from agents.encoder import DominoEncoder

    return {
        name: (
            int(DominoEncoder(name).vector_size),
            int(DominoEncoder(name).action_size),
        )
        for name in DEFAULT_RULESET_ORDER
    }


def discover_checkpoints(directory=SCRIPT_DIR):
    """Map exactly one default-layout checkpoint to each matching ruleset."""
    directory = Path(directory)
    expected = expected_default_checkpoint_shapes()
    discovered = {}
    for path in sorted(directory.glob("*.npz")):
        structure = _checkpoint_structure(path)
        matches = [
            ruleset_name
            for ruleset_name, dimensions in expected.items()
            if dimensions == (
                structure["input_size"],
                structure["output_size"],
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot infer one ruleset for {path.name}: "
                f"input/output={structure['input_size']}/{structure['output_size']}"
            )
        ruleset_name = matches[0]
        if ruleset_name in discovered:
            raise ValueError(
                f"Multiple checkpoints match {ruleset_name}: "
                f"{discovered[ruleset_name]['path'].name}, {path.name}"
            )
        discovered[ruleset_name] = {
            "path": path,
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            **structure,
            "assumed_encoder": {
                "use_opponent_suit_features": True,
                "use_opponent_bucket_features": False,
            },
        }
    return discovered


def parse_ruleset_selection(value, available=None):
    """Resolve ``all`` or a comma-separated ruleset list in canonical order."""
    if available is None:
        available = DEFAULT_RULESET_ORDER
    available = tuple(available)
    if value == "all":
        requested = set(available)
    else:
        requested = {item.strip() for item in value.split(",") if item.strip()}
    unknown = requested - set(DEFAULT_RULESET_ORDER)
    if unknown:
        raise ValueError(f"Unknown rulesets: {sorted(unknown)}")
    missing = requested - set(available)
    if missing:
        raise ValueError(f"Missing checkpoints for rulesets: {sorted(missing)}")
    return tuple(name for name in DEFAULT_RULESET_ORDER if name in requested)
