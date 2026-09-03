#!/usr/bin/env python3
"""Print selected fixed reward-lookup cells in a compact human-readable form."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOOKUP_ROOT = SCRIPT_DIR / "ad_hoc"
RULESETS = ("double-three", "double-four", "double-five", "double-six")
COMPONENTS = ("empty_hand", "blocked", "pass", "draw")
CLOCKS = ("turn", "decision")


def canonical_cell(value: str) -> str:
    """Return ``hand_size,opponent_size`` after validating one CLI cell."""
    text = value.strip().removeprefix("(").removesuffix(")")
    pieces = [piece.strip() for piece in text.split(",")]
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError(
            f"invalid cell {value!r}; expected AGENT,OPPONENT (for example 1,1)"
        )
    try:
        agent_size, opponent_size = (int(piece) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid cell {value!r}; hand sizes must be integers"
        ) from exc
    if agent_size < 0 or opponent_size < 0:
        raise argparse.ArgumentTypeError(
            f"invalid cell {value!r}; hand sizes cannot be negative"
        )
    return f"{agent_size},{opponent_size}"


def format_histogram(values: list[float], precision: int) -> str:
    """Format one dense histogram without hiding its exponent positions."""
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def load_lookup(root: Path, ruleset: str) -> dict:
    path = root / f"{ruleset}_fixed_signed_reward_lookup.json.gz"
    if not path.is_file():
        raise FileNotFoundError(f"fixed lookup not found: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("ruleset_name") != ruleset:
        raise ValueError(
            f"lookup {path} declares ruleset {payload.get('ruleset_name')!r}"
        )
    return payload


def print_cell(payload: dict, cell: str, precision: int) -> None:
    """Print all four components and both clocks for one cell."""
    tables = payload["tables"]
    available = cell in tables[COMPONENTS[0]][CLOCKS[0]]
    print(f"cell ({cell})")
    if not available:
        print("  MISSING: this cell is not present in the fixed lookup")
        return

    rows = []
    for component in COMPONENTS:
        rows.append(
            (
                component,
                format_histogram(tables[component]["turn"][cell], precision),
                format_histogram(tables[component]["decision"][cell], precision),
            )
        )
    component_width = max(len("component"), *(len(row[0]) for row in rows))
    turn_width = max(len("turn histogram"), *(len(row[1]) for row in rows))
    decision_width = max(len("decision histogram"), *(len(row[2]) for row in rows))
    print(
        f"  {'component':<{component_width}}  "
        f"{'turn histogram':<{turn_width}}  "
        f"{'decision histogram':<{decision_width}}"
    )
    print(f"  {'-' * component_width}  {'-' * turn_width}  {'-' * decision_width}")
    for component, turn, decision in rows:
        print(
            f"  {component:<{component_width}}  "
            f"{turn:<{turn_width}}  {decision:<{decision_width}}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print final, pips, pass, and draw histograms for one or more fixed "
            "reward-lookup cells."
        )
    )
    parser.add_argument(
        "cells",
        nargs="+",
        type=canonical_cell,
        metavar="AGENT,OPPONENT",
        help="cell key; pass several keys separated by spaces",
    )
    parser.add_argument(
        "--rulesets",
        nargs="+",
        choices=RULESETS,
        default=RULESETS,
        help="rulesets to inspect (default: all four)",
    )
    parser.add_argument(
        "--lookup-root",
        type=Path,
        default=DEFAULT_LOOKUP_ROOT,
        help=f"directory containing fixed lookup files (default: {DEFAULT_LOOKUP_ROOT})",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        choices=range(0, 10),
        metavar="DIGITS",
        help="decimal places in histogram coefficients (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for ruleset_index, ruleset in enumerate(args.rulesets):
        if ruleset_index:
            print()
        print(ruleset)
        payload = load_lookup(args.lookup_root, ruleset)
        for cell_index, cell in enumerate(args.cells):
            if cell_index:
                print()
            print_cell(payload, cell, args.precision)


if __name__ == "__main__":
    main()
