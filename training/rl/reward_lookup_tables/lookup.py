"""Load and evaluate the fixed hand-size reward lookup artifacts.

The tables store four *unit semantic* components under two distance clocks, one
per term of the reward model in ``training.rl.reward_model``: ``empty_hand``
carries ``R_E`` (signed, or zero for a blocked ending), ``blocked`` carries
``R_B = +/-m(Delta_p)``, and ``pass``/``draw`` carry signed event counts. None
of them contains a weight, a discount factor, or ``reward_eta``: the normalized
scales, ``gamma_f``/``gamma_i`` and the terminal/immediate mixture stay runtime
training configuration and are applied only during evaluation.
"""

from __future__ import annotations

from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from middleware.rulesets import resolve_ruleset
from training.rl.reward_distance import resolve_reward_distance_mode


LOOKUP_FORMAT = "domino_fixed_signed_reward_lookup"
# Version 3 replaced the old ``final``/``pips`` pair with the two terminal
# components of the redesigned reward. The shapes are identical, so the version
# is what keeps a version 2 artifact from being read as if its ``final`` column
# were the new terminal utility.
LOOKUP_FORMAT_VERSION = 3
COMPONENTS = ("empty_hand", "blocked", "pass", "draw")
CLOCKS = ("turn", "decision")
EXPECTED_COMPONENT_SEMANTICS = {
    "empty_hand": "signed_empty_hand_terminal_indicator",
    "blocked": "signed_blocked_terminal_margin_utility",
    "pass": "signed_event_count",
    "draw": "signed_event_count",
}
# The component whose stored cell keys define the table's shared cell set.
REFERENCE_COMPONENT = "empty_hand"
ARTIFACT_ROOT = Path(__file__).resolve().parent


def _artifact_paths(ruleset_name):
    stem = f"{ruleset_name}_fixed_signed_reward_lookup"
    return ARTIFACT_ROOT / f"{stem}.json.gz", ARTIFACT_ROOT / f"{stem}_manifest.json"


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cell_key(agent_hand_size, opponent_hand_size):
    agent_hand_size = int(agent_hand_size)
    opponent_hand_size = int(opponent_hand_size)
    if agent_hand_size < 1 or opponent_hand_size < 1:
        raise ValueError(
            "Reward lookup hand sizes must both be positive, got "
            f"({agent_hand_size}, {opponent_hand_size})."
        )
    return agent_hand_size, opponent_hand_size, (
        f"{agent_hand_size},{opponent_hand_size}"
    )


def evaluate_histogram(histogram, gamma):
    """Evaluate one dense exponent-indexed histogram at ``gamma``."""
    gamma = float(gamma)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("Reward lookup gamma must be between zero and one.")
    value = 0.0
    power = 1.0
    for coefficient in histogram:
        value += float(coefficient) * power
        power *= gamma
    return value


# The raw weight pairs and the normalized scales the resolved schema derives
# from them. Only the scales are read here; the raw weights are validated too so
# a schema assembled by hand cannot present scales that contradict them.
_WEIGHT_PAIRS = (
    ("terminal_empty_hand_weight", "terminal_blocked_weight"),
    ("immediate_draw_weight", "immediate_pass_weight"),
)
_SCALE_KEYS = ("empty_hand_scale", "blocked_scale", "draw_scale", "pass_scale")


def _validate_reward_scales(schema):
    """Check the resolved reward weights this lookup is evaluated against.

    The redesigned reward has no symmetric win/loss magnitudes left to check:
    the sign lives in the stored unit components and the schema only carries
    non-negative weights. What must hold is that neither pair was zeroed out,
    which would delete a whole half of the objective, and that the normalized
    scales are the finite ``[0, 1]`` values the pair normalization produces.
    """
    for first_key, second_key in _WEIGHT_PAIRS:
        weights = []
        for key in (first_key, second_key):
            weight = float(schema[key])
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    f"The reward lookup requires a finite non-negative "
                    f"{key}, got {weight!r}."
                )
            weights.append(weight)
        if max(weights) <= 0.0:
            raise ValueError(
                f"{first_key} and {second_key} cannot both be zero."
            )
    for key in _SCALE_KEYS:
        scale = float(schema[key])
        if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
            raise ValueError(
                f"The reward lookup requires a normalized {key} in [0, 1], "
                f"got {scale!r}."
            )


class RewardLookupTable:
    """One validated immutable lookup for a compact domino ruleset."""

    def __init__(self, payload, *, artifact_digest):
        self.ruleset_name = resolve_ruleset(payload["ruleset_name"]).name
        self.artifact_sha256 = str(artifact_digest)
        self.tables = payload["tables"]
        self._keys = frozenset(self.tables[REFERENCE_COMPONENT]["turn"])

    @classmethod
    def load(cls, ruleset_name):
        """Load one packaged lookup and validate its manifest and schema."""
        ruleset_name = resolve_ruleset(ruleset_name).name
        lookup_path, manifest_path = _artifact_paths(ruleset_name)
        if not lookup_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"Packaged reward lookup is missing for {ruleset_name!r}."
            )
        digest = _file_sha256(lookup_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("output_file") != lookup_path.name:
            raise ValueError("Reward lookup manifest names a different artifact.")
        if manifest.get("output_sha256") != digest:
            raise ValueError("Reward lookup SHA-256 differs from its manifest.")
        if int(manifest.get("output_bytes", -1)) != lookup_path.stat().st_size:
            raise ValueError("Reward lookup size differs from its manifest.")
        with gzip.open(lookup_path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        cls._validate_payload(payload, ruleset_name)
        return cls(payload, artifact_digest=digest)

    @staticmethod
    def _validate_payload(payload, ruleset_name):
        if payload.get("format") != LOOKUP_FORMAT:
            raise ValueError("Unknown reward lookup format.")
        stored_version = int(payload.get("format_version", -1))
        if stored_version != LOOKUP_FORMAT_VERSION:
            raise ValueError(
                f"Reward lookup for {ruleset_name!r} is format version "
                f"{stored_version}, not {LOOKUP_FORMAT_VERSION}. Version 2 "
                "stores the superseded final/pips terminal pair, which cannot "
                "express the empty-hand/blocked decomposition this reward "
                "trains on. Rebuild the artifact with "
                "analysis/reward_lookup_table/build_fixed_signed_reward_lookup"
                ".py; see training/rl/reward_lookup_tables/README.md."
            )
        if payload.get("ruleset_name") != ruleset_name:
            raise ValueError("Reward lookup ruleset does not match its filename.")
        if payload.get("component_semantics") != EXPECTED_COMPONENT_SEMANTICS:
            raise ValueError("Reward lookup component semantics are incompatible.")
        tables = payload.get("tables")
        if not isinstance(tables, dict):
            raise ValueError("Reward lookup has no tables object.")
        try:
            reference = set(tables[REFERENCE_COMPONENT]["turn"])
        except (KeyError, TypeError) as exc:
            raise ValueError("Reward lookup tables are incomplete.") from exc
        for component in COMPONENTS:
            for clock in CLOCKS:
                table = tables.get(component, {}).get(clock)
                if not isinstance(table, dict) or set(table) != reference:
                    raise ValueError(
                        f"Reward lookup {component}/{clock} cell keys differ."
                    )
                for cell, histogram in table.items():
                    if not isinstance(histogram, list):
                        raise ValueError(
                            f"Reward lookup histogram {component}/{clock}/{cell} "
                            "is not a list."
                        )
                    values = np.asarray(histogram, dtype=np.float64)
                    if not np.all(np.isfinite(values)):
                        raise ValueError("Reward lookup contains NaN or infinity.")
        for required_anchor in ("2,4", "5,1", "6,2"):
            if required_anchor not in reference:
                raise ValueError(
                    f"Reward lookup is missing fallback anchor {required_anchor}."
                )

    def resolve_cell(self, agent_hand_size, opponent_hand_size):
        """Return the stored cell used by the documented ad hoc policy.

        ``None`` is the exact structural one-tile case: the decision wins
        immediately by emptying the hand, so ``empty_hand`` is ``+1`` and every
        other component is zero.
        """
        agent_size, opponent_size, key = _cell_key(
            agent_hand_size,
            opponent_hand_size,
        )
        while True:
            if key in self._keys:
                return key
            if agent_size == 1:
                return None
            if agent_size == 2 and opponent_size > 4:
                opponent_size = 4
            elif opponent_size == 1 and agent_size > 5:
                agent_size = 5
            elif opponent_size == 2 and agent_size > 6:
                agent_size = 6
            else:
                agent_size -= 1
                opponent_size -= 1
            key = f"{agent_size},{opponent_size}"

    def component_value(
        self,
        component,
        clock,
        agent_hand_size,
        opponent_hand_size,
        gamma,
    ):
        """Evaluate one component after resolving its hand-size cell."""
        if component not in COMPONENTS or clock not in CLOCKS:
            raise ValueError(f"Unknown reward lookup component/clock: {component}/{clock}")
        cell = self.resolve_cell(agent_hand_size, opponent_hand_size)
        if cell is None:
            return 1.0 if component == "empty_hand" else 0.0
        return evaluate_histogram(self.tables[component][clock][cell], gamma)

    def baseline_values(self, samples, schema):
        """Return the configured expected reward for every decision sample."""
        _validate_reward_scales(schema)
        gamma_f = float(schema["gamma_f"])
        gamma_i = float(schema["gamma_i"])
        reward_eta = float(schema["reward_eta"])
        if not 0.0 <= reward_eta <= 1.0:
            raise ValueError("reward_eta must be between zero and one.")
        local_clock, terminal_clock = resolve_reward_distance_mode(
            schema["reward_distance_mode"]
        )
        empty_hand_scale = float(schema["empty_hand_scale"])
        blocked_scale = float(schema["blocked_scale"])
        pass_scale = float(schema["pass_scale"])
        draw_scale = float(schema["draw_scale"])
        values = []
        for sample in samples:
            agent_size = getattr(sample, "agent_hand_size", None)
            opponent_size = getattr(sample, "opponent_hand_size", None)
            if agent_size is None or opponent_size is None:
                raise ValueError(
                    "The lookup-table baseline needs hand sizes on every "
                    "training decision."
                )
            terminal = (
                empty_hand_scale
                * self.component_value(
                    "empty_hand",
                    terminal_clock,
                    agent_size,
                    opponent_size,
                    gamma_f,
                )
                + blocked_scale
                * self.component_value(
                    "blocked",
                    terminal_clock,
                    agent_size,
                    opponent_size,
                    gamma_f,
                )
            )
            local = (
                pass_scale
                * self.component_value(
                    "pass", local_clock, agent_size, opponent_size, gamma_i
                )
                + draw_scale
                * self.component_value(
                    "draw", local_clock, agent_size, opponent_size, gamma_i
                )
            )
            values.append((1.0 - reward_eta) * terminal + reward_eta * local)
        result = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(result)):
            raise ValueError("Reward lookup baseline produced NaN or infinity.")
        return result


@lru_cache(maxsize=4)
def load_reward_lookup(ruleset_name):
    """Return one process-local cached and validated lookup."""
    return RewardLookupTable.load(resolve_ruleset(ruleset_name).name)


def artifact_sha256(ruleset_name):
    """Return the durable identity of one packaged lookup artifact."""
    return load_reward_lookup(ruleset_name).artifact_sha256
