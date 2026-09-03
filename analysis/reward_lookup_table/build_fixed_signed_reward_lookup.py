"""Build runtime-ready unit-component reward histograms from sampled lookups."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import gzip
import io
import json
import math
import os
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from reward_lookup_common import (
    LOOKUP_FORMAT,
    LOOKUP_FORMAT_VERSION,
    atomic_write_json,
    file_sha256,
    parse_ruleset_selection,
)


FIXED_FORMAT = "domino_fixed_signed_reward_lookup"
# Version 3 stores the two terminal components of the redesigned reward,
# ``R_E`` and ``R_B``, in place of the old signed outcome and pip count.
FIXED_FORMAT_VERSION = 3
ELIGIBILITY_FRACTION = 0.005
VALIDATION_GAMMAS = (0.0, 0.5, 0.9, 0.95, 1.0)
COMPONENTS = ("empty_hand", "blocked", "pass", "draw")
# ``blocked`` carries ``+/-m(Delta_p)``, which is not an integer, so its bins
# and every invariant over them are compared with a tolerance rather than
# exactly. The other three components stay signed integer counts.
REAL_VALUED_COMPONENTS = ("blocked",)
TERMINAL_COMPONENTS = ("empty_hand", "blocked")
CLOCKS = ("turn", "decision")
DEFAULT_DERIVED_ROOT = SCRIPT_DIR / "derived"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "fixed"

_EVENT_CONTRACT = {
    "opponent_pass": ("pass", "opponent", 1),
    "neural_pass": ("pass", "neural", -1),
    "opponent_draw": ("draw", "opponent", 1),
    "neural_draw": ("draw", "neural", -1),
}


def eligibility_threshold(ruleset_decisions):
    """Return the inclusive 0.5% cell threshold."""
    decisions = int(ruleset_decisions)
    if decisions < 1:
        raise ValueError("Ruleset decision count must be positive")
    return int(math.ceil(ELIGIBILITY_FRACTION * decisions))


def evaluate_histogram(histogram, gamma):
    """Evaluate coefficients whose indices are the actual gamma exponents."""
    gamma = float(gamma)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between zero and one")
    value = 0.0
    power = 1.0
    for coefficient in histogram:
        value += float(coefficient) * power
        power *= gamma
    return value


def _canonical_cell_key(value):
    if not isinstance(value, str):
        raise ValueError(f"Cell key must be a string, got {type(value).__name__}")
    pieces = value.split(",")
    if len(pieces) != 2:
        raise ValueError(f"Invalid cell key {value!r}")
    try:
        agent_size, opponent_size = (int(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError(f"Invalid cell key {value!r}") from exc
    if agent_size < 1 or opponent_size < 1:
        raise ValueError(f"Cell sizes must be positive in {value!r}")
    canonical = f"{agent_size},{opponent_size}"
    if value != canonical:
        raise ValueError(f"Noncanonical cell key {value!r}; expected {canonical!r}")
    return agent_size, opponent_size


def _cell_sort_key(value):
    return _canonical_cell_key(value)


class _StreamingJsonCursor:
    """Small incremental JSON cursor used to decode one large cell at a time."""

    def __init__(self, stream, chunk_size=1024 * 1024):
        self.stream = stream
        self.chunk_size = int(chunk_size)
        self.buffer = ""
        self.index = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def _fill(self):
        if self.eof:
            return False
        chunk = self.stream.read(self.chunk_size)
        if chunk:
            self.buffer += chunk
            return True
        self.eof = True
        return False

    def _compact(self):
        if self.index:
            self.buffer = self.buffer[self.index:]
            self.index = 0

    def _skip_whitespace(self):
        while True:
            while (
                self.index < len(self.buffer)
                and self.buffer[self.index].isspace()
            ):
                self.index += 1
            if self.index < len(self.buffer) or not self._fill():
                return

    def peek(self):
        """Return the next non-whitespace character without consuming it."""
        self._skip_whitespace()
        if self.index >= len(self.buffer):
            raise ValueError("Unexpected end of JSON input")
        return self.buffer[self.index]

    def expect(self, expected):
        """Consume one required punctuation character."""
        actual = self.peek()
        if actual != expected:
            raise ValueError(f"Expected {expected!r} in JSON input, got {actual!r}")
        self.index += 1

    def read_value(self):
        """Decode one JSON value, extending the buffer until it is complete."""
        self._skip_whitespace()
        self._compact()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.index)
            except json.JSONDecodeError as exc:
                if not self._fill():
                    raise ValueError(f"Invalid or truncated JSON input: {exc}") from exc
            else:
                self.index = end
                return value

    def require_end(self):
        """Reject non-whitespace data after the top-level object."""
        while True:
            self._skip_whitespace()
            if self.index < len(self.buffer):
                raise ValueError("Unexpected data after top-level JSON object")
            self._compact()
            if not self._fill():
                return


def _iter_cells(cursor):
    cursor.expect("{")
    seen = set()
    if cursor.peek() == "}":
        cursor.expect("}")
        return
    while True:
        key = cursor.read_value()
        if not isinstance(key, str):
            raise ValueError("Lookup cell key must be a JSON string")
        _canonical_cell_key(key)
        if key in seen:
            raise ValueError(f"Duplicate lookup cell {key!r}")
        seen.add(key)
        cursor.expect(":")
        yield key, cursor.read_value()
        separator = cursor.peek()
        if separator == ",":
            cursor.expect(",")
            continue
        if separator == "}":
            cursor.expect("}")
            return
        raise ValueError(f"Unexpected cell separator {separator!r}")


def iter_derived_lookup(path):
    """Yield the header once, then cells from one large compressed JSON file."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        cursor = _StreamingJsonCursor(stream)
        cursor.expect("{")
        header = {}
        while True:
            if cursor.peek() == "}":
                raise ValueError(f"Derived lookup {path} has no cells object")
            key = cursor.read_value()
            if not isinstance(key, str):
                raise ValueError("Top-level lookup key must be a JSON string")
            if key in header:
                raise ValueError(f"Duplicate top-level lookup field {key!r}")
            cursor.expect(":")
            if key == "cells":
                yield "header", None, header
                yield from (
                    ("cell", cell_key, cell)
                    for cell_key, cell in _iter_cells(cursor)
                )
                cursor.expect("}")
                cursor.require_end()
                return
            header[key] = cursor.read_value()
            if cursor.peek() != ",":
                raise ValueError("The cells object must be the final lookup field")
            cursor.expect(",")


def _empty_histograms():
    return {
        component: {clock: Counter() for clock in CLOCKS}
        for component in COMPONENTS
    }


def _empty_direct_totals():
    return {
        component: {
            clock: {gamma: 0.0 for gamma in VALIDATION_GAMMAS}
            for clock in CLOCKS
        }
        for component in COMPONENTS
    }


@dataclass
class CellAccumulator:
    """Integer unit-component bins and independent validation totals."""

    track_direct: bool = False
    sample_count: int = 0
    terminal_contributions: int = 0
    wins: int = 0
    losses: int = 0
    histograms: dict = field(default_factory=_empty_histograms)
    direct_totals: dict | None = None
    max_exponents: dict = field(
        default_factory=lambda: {clock: 0 for clock in CLOCKS}
    )
    blocked_endings: dict = field(
        default_factory=lambda: {clock: 0 for clock in CLOCKS}
    )

    def __post_init__(self):
        if self.track_direct:
            self.direct_totals = _empty_direct_totals()

    def add(self, component, clock, exponent, sign):
        exponent = int(exponent)
        sign = int(sign)
        if exponent < 0:
            raise ValueError(f"Negative {clock} exponent {exponent}")
        if sign not in (-1, 1):
            raise ValueError(f"Reward sign must be -1 or +1, got {sign}")
        self._accumulate(component, clock, exponent, sign)

    def add_value(self, component, clock, exponent, value):
        """Add one real-valued unit contribution such as ``+/-m(Delta_p)``.

        ``R_B`` saturates a pip margin through a nonlinear map, so a blocked
        contribution is a finite float in ``[-1, 1]`` rather than a sign.
        """
        exponent = int(exponent)
        value = float(value)
        if exponent < 0:
            raise ValueError(f"Negative {clock} exponent {exponent}")
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError(
                f"Unit component value must be finite in [-1, 1], got {value!r}"
            )
        self._accumulate(component, clock, exponent, value)

    def add_blocked_ending(self, clock, sign):
        """Count one blocked ending so terminal totals stay auditable.

        ``blocked`` stores ``+/-m(Delta_p)``, whose magnitude is not 1, so the
        signed *count* of blocked endings cannot be recovered from the
        histogram itself. It is tracked here instead of being inferred.
        """
        self.blocked_endings[clock] += int(sign)

    def _accumulate(self, component, clock, exponent, value):
        self.histograms[component][clock][exponent] += value
        self.max_exponents[clock] = max(self.max_exponents[clock], exponent)
        if self.direct_totals is not None:
            for gamma in VALIDATION_GAMMAS:
                self.direct_totals[component][clock][gamma] += (
                    value * gamma ** exponent
                )


def _validated_exponent(value, label):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{label} cannot be negative, got {value}")
    return int(value)


def _terminal_contribution(sample):
    """Return the two unit terminal components and their clock exponents.

    Exactly one of ``R_E`` and ``R_B`` is non-zero, because an ending is either
    an empty hand or a block. Both carry the sign of the decision's own result,
    which is what makes the stored histograms usable under any later choice of
    ``a_E`` and ``a_B``.
    """
    result = sample.get("result")
    if result == "win":
        sign = 1
    elif result == "loss":
        sign = -1
    else:
        raise ValueError(f"Unknown terminal result {result!r}")
    terminal = sample.get("terminal")
    if not isinstance(terminal, dict):
        raise ValueError("Every sample must contain one terminal object")
    if bool(terminal.get("learner_won")) != (sign > 0):
        raise ValueError("Terminal learner_won disagrees with the sample result")
    components = {}
    for component in TERMINAL_COMPONENTS:
        value = float(terminal.get(f"{component}_component"))
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError(
                f"Terminal {component} component must be finite in [-1, 1], "
                f"got {value!r}"
            )
        if value and value * sign <= 0.0:
            raise ValueError(
                f"Terminal {component} component sign disagrees with result"
            )
        components[component] = value
    non_zero = [name for name, value in components.items() if value]
    if len(non_zero) != 1:
        raise ValueError(
            "Exactly one terminal component is non-zero per ending, got "
            f"{non_zero or 'none'}"
        )
    exponents = {
        clock: _validated_exponent(
            terminal.get(f"{clock}_distance"),
            f"terminal {clock} distance",
        )
        for clock in CLOCKS
    }
    decision_turn = _validated_exponent(
        sample.get("decision_turn"),
        "decision turn",
    )
    terminal_turn = _validated_exponent(terminal.get("terminal_turn"), "terminal turn")
    if terminal_turn - decision_turn - 1 != terminal["turn_distance"]:
        raise ValueError("Stored terminal turn exponent is inconsistent")
    return sign, components, exponents


def _local_contributions(sample):
    events = sample.get("future_local_events")
    if not isinstance(events, list):
        raise ValueError("future_local_events must be a list")
    decision_turn = _validated_exponent(sample.get("decision_turn"), "decision turn")
    contributions = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Future local event must be an object")
        kind = event.get("kind")
        try:
            component, actor, sign = _EVENT_CONTRACT[kind]
        except KeyError as exc:
            raise ValueError(f"Unknown local reward event {kind!r}") from exc
        if event.get("actor") != actor:
            raise ValueError(f"Event {kind!r} has inconsistent actor")
        unit_reward = float(event.get("unit_reward"))
        if not math.isfinite(unit_reward) or unit_reward * sign <= 0.0:
            raise ValueError(f"Event {kind!r} has inconsistent reward sign")
        turn_exponent = _validated_exponent(
            event.get("turn_distance"),
            f"{kind} turn distance",
        )
        event_turn = _validated_exponent(event.get("event_turn"), f"{kind} turn")
        if event_turn - decision_turn - 1 != turn_exponent:
            raise ValueError(f"Stored turn exponent is inconsistent for {kind!r}")
        decision_exponent = _validated_exponent(
            event.get("decision_distance"),
            f"{kind} decision distance",
        )
        contributions.append(
            (component, sign, turn_exponent, decision_exponent)
        )
    return contributions


def accumulate_sample(accumulator, sample):
    """Add one decision's unit terminal pair and local-event suffix."""
    if not isinstance(sample, dict):
        raise ValueError("Lookup sample must be an object")
    terminal_sign, terminal_components, terminal_exponents = (
        _terminal_contribution(sample)
    )
    local_contributions = _local_contributions(sample)
    accumulator.sample_count += 1
    accumulator.terminal_contributions += 1
    if terminal_sign > 0:
        accumulator.wins += 1
    else:
        accumulator.losses += 1
    for clock, exponent in terminal_exponents.items():
        for component, value in terminal_components.items():
            accumulator.add_value(component, clock, exponent, value)
        if terminal_components["blocked"]:
            accumulator.add_blocked_ending(clock, terminal_sign)
    for component, sign, turn_exponent, decision_exponent in local_contributions:
        accumulator.add(component, "turn", turn_exponent, sign)
        accumulator.add(component, "decision", decision_exponent, sign)


def validate_sample(sample):
    """Validate one omitted cell sample without retaining histogram bins."""
    if not isinstance(sample, dict):
        raise ValueError("Lookup sample must be an object")
    _terminal_contribution(sample)
    _local_contributions(sample)


def dense_normalized_histogram(counter, denominator):
    """Convert integer unit bins to a trailing-zero-trimmed dense list."""
    denominator = int(denominator)
    if denominator < 1:
        raise ValueError("Histogram denominator must be positive")
    nonzero = [index for index, value in counter.items() if value]
    if not nonzero:
        return []
    maximum = max(nonzero)
    values = [float(counter[index]) / denominator for index in range(maximum + 1)]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Normalized histogram contains NaN or infinity")
    return values


def _blocked_ending_balance(accumulator, clock):
    """Return the signed number of blocked endings recorded for one clock."""
    return int(accumulator.blocked_endings[clock])


def _validate_cell(accumulator, expected_count, cell_key):
    if accumulator.sample_count != expected_count:
        raise ValueError(
            f"Cell {cell_key} parsed {accumulator.sample_count} samples; "
            f"manifest declares {expected_count}"
        )
    if accumulator.terminal_contributions != expected_count:
        raise ValueError(f"Cell {cell_key} does not have one terminal per sample")
    # Every ending contributes to exactly one of the two terminal components,
    # so their two signed totals together must reproduce the cell's win/loss
    # balance. Neither one does so alone any more: an empty-hand ending leaves
    # ``blocked`` at zero and a block leaves ``empty_hand`` at zero.
    expected_terminal_sum = accumulator.wins - accumulator.losses
    for clock in CLOCKS:
        empty_hand_sum = sum(
            accumulator.histograms["empty_hand"][clock].values()
        )
        if empty_hand_sum != int(empty_hand_sum):
            raise ValueError(
                f"Cell {cell_key} empty_hand {clock} sum is not an integer"
            )
        blocked_sum = sum(accumulator.histograms["blocked"][clock].values())
        signed_endings = int(empty_hand_sum) + _blocked_ending_balance(
            accumulator, clock
        )
        if signed_endings != expected_terminal_sum:
            raise ValueError(
                f"Cell {cell_key} terminal {clock} endings are inconsistent"
            )
        # Every blocked contribution has magnitude ``m(Delta_p) <= 1``, so the
        # signed total can never exceed one unit per sample in the cell.
        if abs(blocked_sum) > accumulator.sample_count + 1e-9:
            raise ValueError(
                f"Cell {cell_key} blocked {clock} sum is out of range"
            )
    for component in COMPONENTS:
        turn_sum = sum(accumulator.histograms[component]["turn"].values())
        decision_sum = sum(
            accumulator.histograms[component]["decision"].values()
        )
        consistent = (
            math.isclose(turn_sum, decision_sum, rel_tol=1e-12, abs_tol=1e-12)
            if component in REAL_VALUED_COMPONENTS
            else turn_sum == decision_sum
        )
        if not consistent:
            raise ValueError(
                f"Cell {cell_key} changes unit {component} count by clock"
            )


def _normalized_cell(accumulator, cell_key):
    _validate_cell(accumulator, accumulator.sample_count, cell_key)
    normalized = {
        component: {
            clock: dense_normalized_histogram(
                accumulator.histograms[component][clock],
                accumulator.sample_count,
            )
            for clock in CLOCKS
        }
        for component in COMPONENTS
    }
    for clock in CLOCKS:
        for component in TERMINAL_COMPONENTS:
            component_sum = sum(normalized[component][clock])
            if not -1.0 - 1e-12 <= component_sum <= 1.0 + 1e-12:
                raise ValueError(
                    f"Cell {cell_key} {component} histogram is outside [-1, 1]"
                )
    for component in COMPONENTS:
        if not math.isclose(
            sum(normalized[component]["turn"]),
            sum(normalized[component]["decision"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Cell {cell_key} violates gamma=1 {component} clock invariance"
            )
    if accumulator.direct_totals is not None:
        _validate_direct_returns(accumulator, normalized, cell_key)
    return normalized


def _validate_direct_returns(accumulator, normalized, cell_key):
    for component in COMPONENTS:
        for clock in CLOCKS:
            histogram = normalized[component][clock]
            for gamma in VALIDATION_GAMMAS:
                direct = (
                    accumulator.direct_totals[component][clock][gamma]
                    / accumulator.sample_count
                )
                evaluated = evaluate_histogram(histogram, gamma)
                if not math.isclose(
                    direct,
                    evaluated,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"Cell {cell_key} direct {component}/{clock} return "
                        f"differs at gamma={gamma}"
                    )


def _validate_source_manifest(path, ruleset_name):
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format") != LOOKUP_FORMAT:
        raise ValueError(f"Unexpected derived manifest format in {path}")
    if int(manifest.get("format_version", -1)) != LOOKUP_FORMAT_VERSION:
        raise ValueError(f"Unsupported derived manifest version in {path}")
    if manifest.get("ruleset_name") != ruleset_name:
        raise ValueError(f"Ruleset mismatch in {path}")
    decisions = int(manifest.get("summary", {}).get("decisions", 0))
    counts = manifest.get("cell_sample_counts")
    if not isinstance(counts, dict) or not counts:
        raise ValueError(f"Missing cell counts in {path}")
    normalized_counts = {}
    for key, value in counts.items():
        _canonical_cell_key(key)
        count = int(value)
        if count < 1 or value != count:
            raise ValueError(f"Invalid sample count for cell {key!r}")
        normalized_counts[key] = count
    if sum(normalized_counts.values()) != decisions:
        raise ValueError(f"Cell counts do not sum to decisions in {path}")
    return manifest, decisions, normalized_counts


def _validate_lookup_header(header, source_manifest, ruleset_name):
    expected = {
        "format": LOOKUP_FORMAT,
        "format_version": LOOKUP_FORMAT_VERSION,
        "ruleset_name": ruleset_name,
        "key_fields": ["neural_hand_size", "opponent_hand_size"],
        "action_is_part_of_key": False,
    }
    for key, value in expected.items():
        if header.get(key) != value:
            raise ValueError(f"Derived lookup header field {key!r} is inconsistent")
    if header.get("summary") != source_manifest.get("summary"):
        raise ValueError("Derived lookup and manifest summaries differ")


def _validation_cell_keys(eligible_keys):
    ordered = sorted(eligible_keys, key=_cell_sort_key)
    if len(ordered) <= 5:
        return set(ordered)
    indices = (0, len(ordered) // 4, len(ordered) // 2, 3 * len(ordered) // 4, -1)
    return {ordered[index] for index in indices}


def _process_cell(cell_key, cell, expected_count, track_direct, retain_histograms):
    if not isinstance(cell, dict):
        raise ValueError(f"Cell {cell_key} must be an object")
    agent_size, opponent_size = _canonical_cell_key(cell_key)
    if cell.get("neural_hand_size") != agent_size:
        raise ValueError(f"Cell {cell_key} has wrong neural hand size")
    if cell.get("opponent_hand_size") != opponent_size:
        raise ValueError(f"Cell {cell_key} has wrong opponent hand size")
    if cell.get("sample_count") != expected_count:
        raise ValueError(f"Cell {cell_key} sample count differs from manifest")
    samples = cell.get("samples")
    if not isinstance(samples, list) or len(samples) != expected_count:
        raise ValueError(f"Cell {cell_key} sample list differs from manifest")
    if not retain_histograms:
        for sample in samples:
            validate_sample(sample)
        return None
    accumulator = CellAccumulator(track_direct=track_direct)
    for sample in samples:
        accumulate_sample(accumulator, sample)
    _validate_cell(accumulator, expected_count, cell_key)
    return accumulator


def _runtime_tables(accumulators, eligible_keys):
    tables = {
        component: {clock: {} for clock in CLOCKS}
        for component in COMPONENTS
    }
    for cell_key in sorted(eligible_keys, key=_cell_sort_key):
        normalized = _normalized_cell(accumulators[cell_key], cell_key)
        for component in COMPONENTS:
            for clock in CLOCKS:
                tables[component][clock][cell_key] = normalized[component][clock]
    expected_keys = set(eligible_keys)
    for component in COMPONENTS:
        for clock in CLOCKS:
            if set(tables[component][clock]) != expected_keys:
                raise AssertionError("Fixed lookup tables disagree on cell keys")
    return tables


def _write_deterministic_gzip_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as raw_stream:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw_stream,
                mtime=0,
            ) as compressed:
                with io.TextIOWrapper(
                    compressed,
                    encoding="utf-8",
                    write_through=True,
                ) as text_stream:
                    json.dump(
                        value,
                        text_stream,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _build_runtime_payload(ruleset_name, tables):
    return {
        "format": FIXED_FORMAT,
        "format_version": FIXED_FORMAT_VERSION,
        "ruleset_name": ruleset_name,
        "cell_key_order": ["agent_hand_size", "opponent_hand_size"],
        "histogram_index": "discount_exponent",
        "normalization": "unit_component_sum_per_cell_sample",
        "component_semantics": {
            "empty_hand": "signed_empty_hand_terminal_indicator",
            "blocked": "signed_blocked_terminal_margin_utility",
            "pass": "signed_event_count",
            "draw": "signed_event_count",
        },
        "eligibility_fraction": ELIGIBILITY_FRACTION,
        "tables": tables,
    }


def build_ruleset(derived_root, output_root, ruleset_name, force=False):
    """Build and validate one ruleset's eight fixed logical tables."""
    sample_path = derived_root / f"{ruleset_name}_reward_lookup_samples.json.gz"
    source_manifest_path = (
        derived_root / f"{ruleset_name}_reward_lookup_manifest.json"
    )
    destination = output_root / f"{ruleset_name}_fixed_signed_reward_lookup.json.gz"
    build_manifest_path = (
        output_root / f"{ruleset_name}_fixed_signed_reward_lookup_manifest.json"
    )
    for required in (sample_path, source_manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing derived lookup input: {required}")
    if not force and (destination.exists() or build_manifest_path.exists()):
        raise FileExistsError(
            f"Fixed lookup already exists for {ruleset_name}; pass --force to replace it"
        )

    source_manifest, decisions, counts = _validate_source_manifest(
        source_manifest_path,
        ruleset_name,
    )
    if source_manifest.get("output_file") != sample_path.name:
        raise ValueError("Derived manifest names a different sample file")
    if int(source_manifest.get("output_bytes", -1)) != sample_path.stat().st_size:
        raise ValueError("Derived sample size differs from its manifest")
    if source_manifest.get("output_sha256") != file_sha256(sample_path):
        raise ValueError("Derived sample checksum differs from its manifest")
    threshold = eligibility_threshold(decisions)
    eligible_keys = {
        key for key, count in counts.items() if count >= threshold
    }
    validation_keys = _validation_cell_keys(eligible_keys)
    accumulators = {}
    parsed_counts = {}
    header_seen = False
    for kind, cell_key, value in iter_derived_lookup(sample_path):
        if kind == "header":
            _validate_lookup_header(value, source_manifest, ruleset_name)
            header_seen = True
            continue
        if not header_seen:
            raise ValueError("Derived cells appeared before their header")
        if cell_key not in counts:
            raise ValueError(f"Derived lookup contains unknown cell {cell_key}")
        expected_count = counts[cell_key]
        retain_histograms = cell_key in eligible_keys
        accumulator = _process_cell(
            cell_key,
            value,
            expected_count,
            track_direct=cell_key in validation_keys,
            retain_histograms=retain_histograms,
        )
        parsed_counts[cell_key] = expected_count
        if retain_histograms:
            accumulators[cell_key] = accumulator

    if not header_seen:
        raise ValueError("Derived lookup header was not parsed")
    if parsed_counts != counts:
        missing = sorted(set(counts) - set(parsed_counts), key=_cell_sort_key)
        extra = sorted(set(parsed_counts) - set(counts), key=_cell_sort_key)
        raise ValueError(f"Derived cell reconciliation failed: missing={missing}, extra={extra}")
    if sum(parsed_counts.values()) != decisions:
        raise ValueError("Parsed sample total differs from ruleset decisions")
    if set(accumulators) != eligible_keys:
        raise ValueError("Constructed eligible cell set differs from the 0.5% rule")

    tables = _runtime_tables(accumulators, eligible_keys)
    payload = _build_runtime_payload(ruleset_name, tables)
    _write_deterministic_gzip_json(destination, payload)
    max_turn = max(
        accumulator.max_exponents["turn"] for accumulator in accumulators.values()
    )
    max_decision = max(
        accumulator.max_exponents["decision"]
        for accumulator in accumulators.values()
    )
    build_manifest = {
        "format": FIXED_FORMAT,
        "format_version": FIXED_FORMAT_VERSION,
        "ruleset_name": ruleset_name,
        "source_file": sample_path.name,
        "source_file_sha256": file_sha256(sample_path),
        "source_manifest": source_manifest_path.name,
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "ruleset_decisions": decisions,
        "eligibility_fraction": ELIGIBILITY_FRACTION,
        "eligibility_threshold": threshold,
        "eligible_cells": len(eligible_keys),
        "maximum_turn_exponent": max_turn,
        "maximum_decision_exponent": max_decision,
        "output_file": destination.name,
        "output_bytes": destination.stat().st_size,
        "output_sha256": file_sha256(destination),
        "direct_return_validation_gammas": list(VALIDATION_GAMMAS),
        "direct_return_validation_cells": sorted(
            validation_keys,
            key=_cell_sort_key,
        ),
    }
    atomic_write_json(build_manifest_path, build_manifest)
    print(
        f"{ruleset_name}: {decisions:,} decisions | threshold {threshold:,} | "
        f"{len(eligible_keys)} cells | max exponent turn/decision "
        f"{max_turn}/{max_decision} | {destination} | "
        f"sha256 {build_manifest['output_sha256']}"
    )
    return build_manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rulesets", default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    args.derived_root = args.derived_root.resolve()
    args.output_root = args.output_root.resolve()
    available = tuple(
        path.name.removesuffix("_reward_lookup_manifest.json")
        for path in args.derived_root.glob("*_reward_lookup_manifest.json")
    )
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
        build_ruleset(
            args.derived_root,
            args.output_root,
            ruleset_name,
            force=args.force,
        )


if __name__ == "__main__":
    main()
