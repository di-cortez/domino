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
# Version 2 records the unit semantic components of the redesigned reward
# instead of the old scalar magnitudes, so a version 1 corpus cannot be read as
# if its ``base_reward`` fields meant the same thing.
RAW_FORMAT_VERSION = 2
LOOKUP_FORMAT = "domino_hand_size_reward_lookup_samples"
LOOKUP_FORMAT_VERSION = 2
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


def raw_reward_semantics():
    """Describe what a raw corpus records, which is facts and unit signs only.

    The raw histories deliberately carry no experimental weight: an experiment
    chooses ``a_E``/``a_B``/``a_D``/``a_P``, ``gamma_f``, ``gamma_i`` and
    ``reward_eta`` at training time, so baking any of them into a corpus would
    make the corpus answer one experiment instead of every experiment. What is
    stored is the unit semantics of ``training.rl.reward_model``.
    """
    return {
        "model": "training.rl.reward_model unit semantic components",
        "terminal_components": ("empty_hand_component", "blocked_component"),
        "local_event_component": "unit_reward",
        "unit_event_reward": {"opponent": 1.0, "neural": -1.0},
        "weights_applied": False,
        "gamma_i_applied": False,
        "gamma_f_applied": False,
        "reward_eta_applied": False,
    }


def local_event_for_action(action, acting_seat, neural_seat):
    """Return one raw local event from the neural policy's perspective.

    ``unit_reward`` is ``r_D``/``r_P`` from the reward model: forcing the
    opponent into a draw or pass is ``+1``, being forced into one is ``-1``.
    The draw/pass weights are runtime configuration and are not applied here.
    """
    from training.rl.reward_model import (  # Imported lazily for worker safety.
        unit_event_reward,
    )

    kind = action_kind(action)
    actor = "neural" if int(acting_seat) == int(neural_seat) else "opponent"
    if kind not in {"draw", "pass"}:
        return None
    return {
        "kind": f"{actor}_{kind}",
        "actor": actor,
        "action_kind": kind,
        "unit_reward": float(
            unit_event_reward(kind, by_learner=actor == "neural")
        ),
    }


def terminal_components_for_neural(final_state, neural_seat):
    """Decompose one finished game into its two unit terminal components.

    This is the serialized-state twin of
    ``training.rl.reward_model.terminal_reward_components``: it reads the same
    facts from ``DominoEngine.to_dict()`` instead of a live engine, and returns
    the same ``R_E``/``R_B`` decomposition with the observations it was derived
    from. Exactly one component is non-zero, and neither carries a weight.
    """
    from training.rl.reward_model import (  # Imported lazily for worker safety.
        KNOWN_TERMINAL_WIN_REASONS,
        blocked_reward_magnitude,
    )
    from middleware.domino_engine import WIN_REASON_EMPTY_HAND
    from middleware.rulesets import resolve_ruleset

    neural_seat = int(neural_seat)
    winner = final_state["winner"]
    if winner is None:
        raise ValueError("Finished game has no winner.")
    winner = int(winner)
    win_reason = final_state["win_reason"]
    if win_reason not in KNOWN_TERMINAL_WIN_REASONS:
        raise ValueError(
            f"Unknown terminal win reason {win_reason!r}; expected one of "
            f"{', '.join(sorted(KNOWN_TERMINAL_WIN_REASONS))}."
        )
    neural_won = winner == neural_seat
    sign = 1.0 if neural_won else -1.0
    hands = final_state["hands"]
    pips = [
        sum(int(tile[0]) + int(tile[1]) for tile in hand) for hand in hands
    ]
    if win_reason == WIN_REASON_EMPTY_HAND:
        return {
            "win_reason": win_reason,
            "learner_won": bool(neural_won),
            "empty_hand_component": sign,
            "blocked_component": 0.0,
            "winner_final_pips": int(pips[winner]),
            "loser_final_pips": int(pips[1 - winner]),
            "pip_margin": None,
            "blocked_magnitude": None,
        }
    if len(hands) != 2:
        raise ValueError(
            "Blocked terminal reward is defined for the canonical two-player "
            f"training game, got {len(hands)} hands."
        )
    winner_pips = pips[winner]
    loser_pips = pips[1 - winner]
    pip_margin = loser_pips - winner_pips
    magnitude = blocked_reward_magnitude(
        pip_margin,
        resolve_ruleset(final_state["ruleset_name"]).max_pip,
    )
    return {
        "win_reason": win_reason,
        "learner_won": bool(neural_won),
        "empty_hand_component": 0.0,
        "blocked_component": sign * magnitude,
        "winner_final_pips": int(winner_pips),
        "loser_final_pips": int(loser_pips),
        "pip_margin": int(pip_margin),
        "blocked_magnitude": float(magnitude),
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
