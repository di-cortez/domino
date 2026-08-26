"""Aggregate completed exact-solver JSONL files into exact theoretical win rates."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from itertools import combinations
from math import comb
from pathlib import Path

from solver import ExactVsRandomSolver, RULESETS, fraction_to_decimal


def output_name(ruleset: str, seat: int, information_mode: str) -> str:
    stem = ruleset.replace("-", "_")
    if information_mode == "partial":
        return f"{stem}_player{seat}.jsonl"
    return f"{stem}_cheater_player{seat}.jsonl" 


def load_rows(
    path: Path,
    ruleset: str,
    seat: int,
    information_mode: str,
) -> dict[tuple[int, ...], dict]:
    rows: dict[tuple[int, ...], dict] = {}
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSON on {path}:{line_number}") from exc
            if row.get("ruleset") != ruleset:
                raise RuntimeError(f"Wrong ruleset in {path}:{line_number}")
            if int(row.get("hero_seat", -1)) != seat:
                raise RuntimeError(f"Wrong hero_seat in {path}:{line_number}")
            row_mode = row.get("information_mode", "partial")
            if row_mode != information_mode:
                raise RuntimeError(
                    f"Wrong information mode in {path}:{line_number}: {row_mode!r}"
                )
            hand = tuple(int(x) for x in row["hand_ids"])
            rows[hand] = row
    return rows


def row_value(row: dict) -> Fraction:
    return Fraction(int(row["value_num"]), int(row["value_den"]))


def exact_mean(values: list[Fraction]) -> Fraction:
    if not values:
        raise ValueError("Cannot average an empty list")
    return sum(values, Fraction(0)) / len(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate exact per-hand optimal-vs-random results."
    )
    parser.add_argument(
        "--ruleset",
        choices=tuple(RULESETS),
        default="double-three",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--information-mode",
        choices=("partial", "perfect"),
        default="partial",
    )
    return parser.parse_args()


def aggregate_results(
    ruleset: str,
    output_dir: Path,
    information_mode: str = "partial",
) -> dict:
    """Aggregate both seat files and return the JSON-compatible summary."""
    output_dir = Path(output_dir).resolve()
    max_pip, hand_size = RULESETS[ruleset]
    tile_count = (max_pip + 1) * (max_pip + 2) // 2
    expected_hands = list(combinations(range(tile_count), hand_size))
    expected_count = comb(tile_count, hand_size)

    solver_for_tiles = ExactVsRandomSolver(ruleset=ruleset, hero_seat=0)

    seat_rows: dict[int, dict[tuple[int, ...], dict]] = {}
    seat_values: dict[int, Fraction] = {}
    complete = True

    for seat in (0, 1):
        path = output_dir / output_name(ruleset, seat, information_mode)
        rows = load_rows(path, ruleset, seat, information_mode)
        seat_rows[seat] = rows
        missing = [hand for hand in expected_hands if hand not in rows]
        print(f"player{seat}: {len(rows)}/{expected_count} hands complete")
        if missing:
            complete = False
            print(f"  missing {len(missing)} hands; first missing: {missing[0]}")
        else:
            value = exact_mean([row_value(rows[hand]) for hand in expected_hands])
            seat_values[seat] = value
            print(f"  exact P(win) = {value} = {float(value):.12f}")

    summary = {
        "ruleset": ruleset,
        "information_mode": information_mode,
        "tile_count": tile_count,
        "hand_size": hand_size,
        "number_of_initial_hands_per_seat": expected_count,
        "probability_of_each_unordered_initial_hand": {
            "num": 1,
            "den": expected_count,
        },
        "tile_id_map": [list(tile) for tile in solver_for_tiles.tiles],
        "complete": complete,
        "seats": {},
    }

    for seat in (0, 1):
        entry = {
            "completed_hands": len(seat_rows[seat]),
            "expected_hands": expected_count,
        }
        if seat in seat_values:
            value = seat_values[seat]
            entry["win_probability"] = {
                "num": value.numerator,
                "den": value.denominator,
                "decimal": fraction_to_decimal(value),
            }
        summary["seats"][f"player{seat}"] = entry

    if complete:
        equal_seat_value = (seat_values[0] + seat_values[1]) / 2
        summary["equal_probability_player0_player1"] = {
            "num": equal_seat_value.numerator,
            "den": equal_seat_value.denominator,
            "decimal": fraction_to_decimal(equal_seat_value),
        }
        print(
            "equal 50/50 seat assignment: "
            f"{equal_seat_value} = {float(equal_seat_value):.12f}"
        )

    stem = ruleset.replace("-", "_")
    summary_name = (
        f"{stem}_summary.json"
        if information_mode == "partial"
        else f"{stem}_cheater_summary.json"
    )
    summary_path = output_dir / summary_name
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Summary written to {summary_path}")
    return summary


def main() -> None:
    args = parse_args()
    aggregate_results(args.ruleset, args.output_dir, args.information_mode)


if __name__ == "__main__":
    main()
