"""Load and evaluate the fixed hand-size reward lookup artifacts.

The tables store four unit components under two distance clocks. Final, pass,
and draw are signed; pips is the nonnegative number of terminal pips remaining
in the learner's hand. Reward magnitudes, discount factors, and ``reward_eta``
remain runtime training configuration and are applied only during evaluation.
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
LOOKUP_FORMAT_VERSION = 2
COMPONENTS = ("final", "pips", "pass", "draw")
CLOCKS = ("turn", "decision")
EXPECTED_COMPONENT_SEMANTICS = {
    "final": "signed_terminal_outcome",
    "pips": "nonnegative_terminal_remaining_pip_count",
    "pass": "signed_event_count",
    "draw": "signed_event_count",
}
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


def _validate_symmetric_rewards(schema):
    pairs = (
        ("terminal_win", "terminal_loss"),
        ("opponent_pass", "learner_pass"),
        ("opponent_draw", "learner_draw"),
    )
    for positive_key, negative_key in pairs:
        positive = float(schema[positive_key])
        negative = float(schema[negative_key])
        if positive <= 0.0 or not math.isclose(
            negative,
            -positive,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "The reward lookup requires symmetric rewards, but "
                f"{positive_key}={positive!r} and {negative_key}={negative!r}."
            )
    pip_penalty = float(schema["final_pip_penalty"])
    if not math.isfinite(pip_penalty) or pip_penalty < 0.0:
        raise ValueError("final_pip_penalty must be finite and nonnegative.")


class RewardLookupTable:
    """One validated immutable lookup for a compact domino ruleset."""

    def __init__(self, payload, *, artifact_digest):
        self.ruleset_name = resolve_ruleset(payload["ruleset_name"]).name
        self.artifact_sha256 = str(artifact_digest)
        self.tables = payload["tables"]
        self._keys = frozenset(self.tables["final"]["turn"])

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
        if int(payload.get("format_version", -1)) != LOOKUP_FORMAT_VERSION:
            raise ValueError("Unsupported reward lookup format version.")
        if payload.get("ruleset_name") != ruleset_name:
            raise ValueError("Reward lookup ruleset does not match its filename.")
        if payload.get("component_semantics") != EXPECTED_COMPONENT_SEMANTICS:
            raise ValueError("Reward lookup component semantics are incompatible.")
        tables = payload.get("tables")
        if not isinstance(tables, dict):
            raise ValueError("Reward lookup has no tables object.")
        try:
            reference = set(tables["final"]["turn"])
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
                    if component == "pips" and np.any(values < 0.0):
                        raise ValueError("Reward lookup pips cannot be negative.")
        for required_anchor in ("2,4", "5,1", "6,2"):
            if required_anchor not in reference:
                raise ValueError(
                    f"Reward lookup is missing fallback anchor {required_anchor}."
                )

    def resolve_cell(self, agent_hand_size, opponent_hand_size):
        """Return the stored cell used by the documented ad hoc policy.

        ``None`` is the exact structural one-tile case: the decision wins
        immediately, giving final ``[1]`` and zero for every other component.
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
            return 1.0 if component == "final" else 0.0
        return evaluate_histogram(self.tables[component][clock][cell], gamma)

    def baseline_values(self, samples, schema):
        """Return the configured expected reward for every decision sample."""
        _validate_symmetric_rewards(schema)
        gamma_f = float(schema["gamma_f"])
        gamma_i = float(schema["gamma_i"])
        reward_eta = float(schema["reward_eta"])
        if not 0.0 <= reward_eta <= 1.0:
            raise ValueError("reward_eta must be between zero and one.")
        local_clock, terminal_clock = resolve_reward_distance_mode(
            schema["reward_distance_mode"]
        )
        final_scale = float(schema["terminal_win"])
        pip_scale = float(schema["final_pip_penalty"])
        pass_scale = float(schema["opponent_pass"])
        draw_scale = float(schema["opponent_draw"])
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
                final_scale
                * self.component_value(
                    "final", terminal_clock, agent_size, opponent_size, gamma_f
                )
                - pip_scale
                * self.component_value(
                    "pips", terminal_clock, agent_size, opponent_size, gamma_f
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
