"""Sequential batch driver for the engine-backed exact solvers.

Typical use from this directory:

    python run.py

The command solves the partial-information player in seats 0 and 1, followed
by the perfect-information cheater in seats 0 and 1, then writes both
aggregates. Each pass has its own tqdm progress bar. Results are appended one
completed initial hand at a time, so an interrupted run resumes safely.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from itertools import combinations
from math import comb
from pathlib import Path

from tqdm.auto import tqdm

from solver import (
    CheaterVsRandomSolver,
    ExactVsRandomSolver,
    RULESETS,
    fraction_to_decimal,
)


INFORMATION_MODES = ("partial", "perfect")


def output_name(ruleset: str, seat: int, information_mode: str) -> str:
    stem = ruleset.replace("-", "_")
    if information_mode == "partial":
        return f"{stem}_player{seat}.jsonl"
    return f"{stem}_cheater_player{seat}.jsonl"


def _load_and_repair_jsonl(
    path: Path,
    ruleset: str,
    seat: int,
    information_mode: str,
) -> dict[tuple[int, ...], dict]:
    """Load completed rows and truncate a partial final line after hard interruption."""

    completed: dict[tuple[int, ...], dict] = {}
    if not path.exists():
        return completed

    good_end = 0
    with path.open("rb") as f:
        while True:
            line_start = f.tell()
            raw = f.readline()
            if not raw:
                good_end = f.tell()
                break
            if not raw.strip():
                good_end = f.tell()
                continue
            try:
                row = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Only an incomplete tail is auto-repaired.  Corruption in the
                # middle is not silently ignored.
                remainder = f.read()
                if remainder.strip():
                    raise RuntimeError(f"Malformed JSONL before EOF in {path}")
                print(f"Repairing incomplete final line in {path.name}")
                good_end = line_start
                break

            if row.get("ruleset") != ruleset or int(row.get("hero_seat", -1)) != seat:
                raise RuntimeError(
                    f"Existing row in {path} belongs to a different ruleset/seat"
                )
            row_mode = row.get("information_mode", "partial")
            if row_mode != information_mode:
                raise RuntimeError(
                    f"Existing row in {path} uses information mode {row_mode!r}, "
                    f"expected {information_mode!r}"
                )
            hand = tuple(int(x) for x in row["hand_ids"])
            completed[hand] = row
            good_end = f.tell()

    current_size = path.stat().st_size
    if good_end < current_size:
        with path.open("r+b") as f:
            f.truncate(good_end)
    return completed


def _append_durable(path: Path, row: dict) -> None:
    encoded = (json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
    with path.open("ab") as f:
        f.write(encoded)
        f.flush()
        os.fsync(f.fileno())


def _computed_row(
    ruleset,
    seat,
    hand_index,
    hand_ids,
    result,
    information_mode,
):
    value = result.value
    return {
        "ruleset": ruleset,
        "hero_seat": seat,
        "information_mode": information_mode,
        "hand_index": hand_index,
        "hand_ids": list(hand_ids),
        "value_num": value.numerator,
        "value_den": value.denominator,
        "value_decimal": fraction_to_decimal(value),
        "initial_worlds": result.initial_world_count,
        "initial_groups": [
            {
                "current_player": current_player,
                "worlds": worlds,
                "value_num": group_value.numerator,
                "value_den": group_value.denominator,
            }
            for current_player, worlds, group_value
            in result.initial_observation_groups
        ],
        "value_calls": result.stats.value_calls,
        "cache_hits": result.stats.cache_hits,
        "cache_entries": result.stats.cache_entries,
        "max_belief_worlds": result.stats.max_belief_worlds,
        "stock_order_mode": "unordered_exact_draw_chance_v1",
        "result_origin": "computed",
    }


def _copied_row(
    source,
    seat,
    expected_worlds,
    information_mode,
    origin,
    *,
    swap_seats,
):
    source_worlds = int(source["initial_worlds"])
    if source_worlds % expected_worlds:
        raise ValueError("Copy source has incompatible world counts.")
    old_order_factor = source_worlds // expected_worlds
    groups = []
    for group in source["initial_groups"]:
        worlds = int(group["worlds"])
        if worlds % old_order_factor:
            raise ValueError("Copy source group cannot be normalized.")
        groups.append({
            **group,
            "current_player": (
                1 - int(group["current_player"])
                if swap_seats
                else int(group["current_player"])
            ),
            "worlds": worlds // old_order_factor,
        })
    return {
        **source,
        "hero_seat": seat,
        "information_mode": information_mode,
        "initial_worlds": expected_worlds,
        "initial_groups": groups,
        "value_calls": 0,
        "cache_hits": 0,
        "cache_entries": 0,
        "max_belief_worlds": 0,
        "stock_order_mode": "unordered_exact_draw_chance_v1",
        "result_origin": origin,
    }


def _seat_symmetry_row(source, seat, expected_worlds, information_mode):
    return {
        **_copied_row(
            source,
            seat,
            expected_worlds,
            information_mode,
            "copied_by_forced_double_seat_symmetry",
            swap_seats=True,
        ),
        "copied_from_hero_seat": 0,
    }


def _perfect_information_extreme_row(source, seat, expected_worlds):
    numerator = int(source["value_num"])
    denominator = int(source["value_den"])
    if numerator not in (0, denominator):
        raise ValueError("Only exact zero/one results can be reused by the cheater.")
    return {
        **_copied_row(
            source,
            seat,
            expected_worlds,
            "perfect",
            "copied_from_partial_information_extreme",
            swap_seats=False,
        ),
        "copied_from_information_mode": "partial",
    }


def solve_seat(
    ruleset: str,
    seat: int,
    output_dir: Path,
    limit: int | None,
    information_mode: str,
    symmetry_source: dict[tuple[int, ...], dict] | None = None,
    partial_information_source: dict[tuple[int, ...], dict] | None = None,
) -> dict[tuple[int, ...], dict]:
    if information_mode not in INFORMATION_MODES:
        raise ValueError(f"Unknown information mode: {information_mode!r}")
    max_pip, hand_size = RULESETS[ruleset]
    tile_count = (max_pip + 1) * (max_pip + 2) // 2
    total_hands = comb(tile_count, hand_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / output_name(ruleset, seat, information_mode)
    completed = _load_and_repair_jsonl(
        path,
        ruleset,
        seat,
        information_mode,
    )

    target_count = total_hands
    if limit is not None:
        target_count = min(total_hands, len(completed) + limit)
    solved_now = 0
    copied_now = 0
    geometry = ExactVsRandomSolver(ruleset=ruleset, hero_seat=seat)
    expected_worlds = comb(tile_count - hand_size, hand_size)
    with tqdm(
        total=target_count,
        initial=min(len(completed), target_count),
        desc=f"{ruleset} {information_mode} seat {seat}",
        unit="hand",
        dynamic_ncols=True,
    ) as progress:
        for hand_index, hand_ids in enumerate(combinations(range(tile_count), hand_size)):
            hand_ids = tuple(hand_ids)
            if hand_ids in completed:
                continue
            if limit is not None and solved_now >= limit:
                break

            source = None if symmetry_source is None else symmetry_source.get(hand_ids)
            if (
                source is not None
                and geometry.initial_hand_has_guaranteed_double_opener(hand_ids)
            ):
                row = _seat_symmetry_row(
                    source,
                    seat,
                    expected_worlds,
                    information_mode,
                )
                copied_now += 1
                cache_entries = 0
            elif information_mode == "perfect":
                partial_source = (
                    None
                    if partial_information_source is None
                    else partial_information_source.get(hand_ids)
                )
                if partial_source is not None and int(partial_source["value_num"]) in (
                    0,
                    int(partial_source["value_den"]),
                ):
                    row = _perfect_information_extreme_row(
                        partial_source,
                        seat,
                        expected_worlds,
                    )
                    copied_now += 1
                    cache_entries = 0
                else:
                    solver = CheaterVsRandomSolver(
                        ruleset=ruleset,
                        hero_seat=seat,
                    )
                    result = solver.solve_initial_hand(hand_ids)
                    row = _computed_row(
                        ruleset,
                        seat,
                        hand_index,
                        hand_ids,
                        result,
                        information_mode,
                    )
                    cache_entries = result.stats.cache_entries
                    del solver
                    gc.collect()
            else:
                # One solver per hand bounds memory and preserves durable resume.
                solver = ExactVsRandomSolver(ruleset=ruleset, hero_seat=seat)
                result = solver.solve_initial_hand(hand_ids)
                row = _computed_row(
                    ruleset,
                    seat,
                    hand_index,
                    hand_ids,
                    result,
                    information_mode,
                )
                cache_entries = result.stats.cache_entries
                del solver
                gc.collect()

            _append_durable(path, row)
            completed[hand_ids] = row
            solved_now += 1
            progress.update(1)
            progress.set_postfix(
                win=f"{float(row['value_decimal']):.6f}",
                cache=cache_entries,
                copied=copied_now,
            )
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve every initial hand exactly against the repo's uniform RandomAgent."
    )
    parser.add_argument(
        "--ruleset",
        choices=tuple(RULESETS),
        default="double-four",
        help="Default: double-four. Other rulesets are useful for validation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the resumable JSONL files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of new hands per seat, useful for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    output_dir = args.output_dir.resolve()
    partial_rows = {}
    partial_rows[0] = solve_seat(
        args.ruleset,
        0,
        output_dir,
        args.limit,
        "partial",
    )
    partial_rows[1] = solve_seat(
        args.ruleset,
        1,
        output_dir,
        args.limit,
        "partial",
        symmetry_source=partial_rows[0],
    )

    perfect_rows = {}
    perfect_rows[0] = solve_seat(
        args.ruleset,
        0,
        output_dir,
        args.limit,
        "perfect",
        partial_information_source=partial_rows[0],
    )
    perfect_rows[1] = solve_seat(
        args.ruleset,
        1,
        output_dir,
        args.limit,
        "perfect",
        symmetry_source=perfect_rows[0],
        partial_information_source=partial_rows[1],
    )

    # Imported here to keep the resumable batch primitives independently usable.
    from aggregate import aggregate_results

    aggregate_results(args.ruleset, output_dir, "partial")
    aggregate_results(args.ruleset, output_dir, "perfect")


if __name__ == "__main__":
    main()
