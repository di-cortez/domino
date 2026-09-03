#!/usr/bin/env python3
"""Apply small, explicit, validated edits to ad hoc fixed reward lookups."""

from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
AD_HOC_ROOT = SCRIPT_DIR / "ad_hoc"
RULESETS = ("double-three", "double-four", "double-five", "double-six")
COMPONENTS = ("empty_hand", "blocked", "pass", "draw")
CLOCKS = ("turn", "decision")
MAX_CELLS_PER_OPERATION = 8


def canonical_cell(value: str) -> str:
    """Validate one explicit CLI cell and return its canonical JSON key."""
    text = value.strip().removeprefix("(").removesuffix(")")
    pieces = [piece.strip() for piece in text.split(",")]
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError(
            f"invalid cell {value!r}; expected AGENT,OPPONENT"
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


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lookup_paths(ruleset: str) -> tuple[Path, Path]:
    stem = f"{ruleset}_fixed_signed_reward_lookup"
    return AD_HOC_ROOT / f"{stem}.json.gz", AD_HOC_ROOT / f"{stem}_manifest.json"


def load_inputs(ruleset: str) -> tuple[dict, dict, Path, Path]:
    lookup_path, manifest_path = lookup_paths(ruleset)
    for path in (lookup_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"required ad hoc artifact not found: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("ruleset_name") != ruleset:
        raise ValueError(f"manifest ruleset differs from requested {ruleset}")
    if manifest.get("output_file") != lookup_path.name:
        raise ValueError("manifest names a different runtime lookup")
    if int(manifest.get("output_bytes", -1)) != lookup_path.stat().st_size:
        raise ValueError("runtime lookup size differs from its manifest")
    if manifest.get("output_sha256") != file_sha256(lookup_path):
        raise ValueError("runtime lookup checksum differs from its manifest")
    with gzip.open(lookup_path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("ruleset_name") != ruleset:
        raise ValueError(f"runtime lookup ruleset differs from requested {ruleset}")
    validate_table_keys(payload)
    return payload, manifest, lookup_path, manifest_path


def validate_table_keys(payload: dict) -> set[str]:
    """Require all component/clock tables to contain exactly the same cells."""
    tables = payload.get("tables")
    if not isinstance(tables, dict):
        raise ValueError("runtime lookup has no tables object")
    reference = set(tables[COMPONENTS[0]][CLOCKS[0]])
    for component in COMPONENTS:
        for clock in CLOCKS:
            keys = set(tables[component][clock])
            if keys != reference:
                raise ValueError(
                    f"table key mismatch in {component}/{clock}: "
                    f"missing={sorted(reference - keys)}, extra={sorted(keys - reference)}"
                )
    return reference


def atomic_write_gzip_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    original_mode = path.stat().st_mode
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
                    compressed, encoding="utf-8", write_through=True
                ) as text_stream:
                    json.dump(
                        payload,
                        text_stream,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
        os.chmod(temporary, original_mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    original_mode = path.stat().st_mode
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, original_mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def remove_cells(ruleset: str, requested_cells: list[str]) -> None:
    if len(requested_cells) > MAX_CELLS_PER_OPERATION:
        raise ValueError(
            f"refusing {len(requested_cells)} cells; the per-operation limit is "
            f"{MAX_CELLS_PER_OPERATION}"
        )
    if len(set(requested_cells)) != len(requested_cells):
        raise ValueError("the explicit cell list contains duplicates")

    payload, manifest, lookup_path, manifest_path = load_inputs(ruleset)
    current_keys = validate_table_keys(payload)
    missing = [cell for cell in requested_cells if cell not in current_keys]
    if missing:
        raise ValueError(f"cells are not present in {ruleset}: {missing}")

    for component in COMPONENTS:
        for clock in CLOCKS:
            table = payload["tables"][component][clock]
            for cell in requested_cells:
                del table[cell]
    remaining_keys = validate_table_keys(payload)

    atomic_write_gzip_json(lookup_path, payload)
    modifications = manifest.setdefault("ad_hoc_modifications", {})
    removed = set(modifications.get("removed_cells", []))
    removed.update(requested_cells)
    modifications["removed_cells"] = sorted(
        removed, key=lambda cell: tuple(int(value) for value in cell.split(","))
    )
    modifications["runtime_cells"] = len(remaining_keys)
    manifest["output_bytes"] = lookup_path.stat().st_size
    manifest["output_sha256"] = file_sha256(lookup_path)
    atomic_write_manifest(manifest_path, manifest)

    print(
        f"{ruleset}: removed {', '.join(requested_cells)}; "
        f"{len(remaining_keys)} runtime cells remain; "
        f"sha256 {manifest['output_sha256']}"
    )


def copy_cell(ruleset: str, source_cell: str, target_cell: str) -> None:
    """Copy all eight histograms from one explicit cell to one absent cell."""
    if source_cell == target_cell:
        raise ValueError("source and target cells must differ")

    payload, manifest, lookup_path, manifest_path = load_inputs(ruleset)
    current_keys = validate_table_keys(payload)
    if source_cell not in current_keys:
        raise ValueError(f"source cell is not present in {ruleset}: {source_cell}")
    if target_cell in current_keys:
        raise ValueError(f"target cell is already present in {ruleset}: {target_cell}")
    modifications = manifest.setdefault("ad_hoc_modifications", {})
    copied = modifications.setdefault("copied_cells", [])
    if any(item.get("target") == target_cell for item in copied):
        raise ValueError(f"manifest already records target cell {target_cell}")

    for component in COMPONENTS:
        for clock in CLOCKS:
            table = payload["tables"][component][clock]
            table[target_cell] = list(table[source_cell])
    resulting_keys = validate_table_keys(payload)
    if len(resulting_keys) != len(current_keys) + 1:
        raise AssertionError("cell copy did not add exactly one runtime cell")

    atomic_write_gzip_json(lookup_path, payload)
    copied.append({"source": source_cell, "target": target_cell})
    modifications["runtime_cells"] = len(resulting_keys)
    manifest["output_bytes"] = lookup_path.stat().st_size
    manifest["output_sha256"] = file_sha256(lookup_path)
    atomic_write_manifest(manifest_path, manifest)

    print(
        f"{ruleset}: copied {source_cell} to {target_cell}; "
        f"{len(resulting_keys)} runtime cells now present; "
        f"sha256 {manifest['output_sha256']}"
    )


def truncates_to_zero(value: float, precision: int) -> bool:
    """Return whether decimal truncation, not rounding, yields exact zero."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"histogram contains non-finite coefficient {value!r}")
    return math.trunc(numeric * (10**precision)) == 0


def trim_zero_tails(ruleset: str, precision: int) -> None:
    """Remove only trailing coefficients that truncate to zero."""
    payload, manifest, lookup_path, manifest_path = load_inputs(ruleset)
    keys_before = validate_table_keys(payload)
    modifications = manifest.setdefault("ad_hoc_modifications", {})
    trimming = modifications.setdefault(
        "zero_tail_trimming",
        {
            "decimal_places": precision,
            "removed_coefficients": {"turn": 0, "decision": 0, "total": 0},
        },
    )
    if trimming.get("decimal_places") != precision:
        raise ValueError(
            "ad hoc lookup was already tail-trimmed at a different precision"
        )
    removed_by_clock = {clock: 0 for clock in CLOCKS}
    for component in COMPONENTS:
        for clock in CLOCKS:
            for histogram in payload["tables"][component][clock].values():
                while histogram and truncates_to_zero(histogram[-1], precision):
                    histogram.pop()
                    removed_by_clock[clock] += 1
    if validate_table_keys(payload) != keys_before:
        raise AssertionError("histogram trimming changed the runtime cell set")

    atomic_write_gzip_json(lookup_path, payload)
    cumulative = trimming["removed_coefficients"]
    for clock in CLOCKS:
        cumulative[clock] = int(cumulative.get(clock, 0)) + removed_by_clock[clock]
    cumulative["total"] = cumulative["turn"] + cumulative["decision"]
    modifications["runtime_cells"] = len(keys_before)
    manifest["output_bytes"] = lookup_path.stat().st_size
    manifest["output_sha256"] = file_sha256(lookup_path)
    atomic_write_manifest(manifest_path, manifest)

    print(
        f"{ruleset}: removed {removed_by_clock['turn']} trailing turn and "
        f"{removed_by_clock['decision']} trailing decision coefficients that "
        f"truncate to zero at {precision} decimal places; "
        f"{len(keys_before)} runtime cells remain; "
        f"sha256 {manifest['output_sha256']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    remove = subparsers.add_parser(
        "remove", help="remove one or a small explicit list of cells"
    )
    remove.add_argument("--ruleset", required=True, choices=RULESETS)
    remove.add_argument(
        "cells",
        nargs="+",
        type=canonical_cell,
        metavar="AGENT,OPPONENT",
    )
    copy = subparsers.add_parser(
        "copy", help="copy one explicit source cell to one absent target cell"
    )
    copy.add_argument("--ruleset", required=True, choices=RULESETS)
    copy.add_argument("--source", required=True, type=canonical_cell)
    copy.add_argument("--target", required=True, type=canonical_cell)
    trim = subparsers.add_parser(
        "trim-zero-tails",
        help="trim zero-at-precision coefficients from histogram tails",
    )
    trim.add_argument("--ruleset", required=True, choices=RULESETS)
    trim.add_argument(
        "--precision",
        type=int,
        default=3,
        choices=range(0, 10),
        metavar="DIGITS",
        help="decimal truncation precision (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.operation == "remove":
        remove_cells(args.ruleset, args.cells)
    elif args.operation == "copy":
        copy_cell(args.ruleset, args.source, args.target)
    elif args.operation == "trim-zero-tails":
        trim_zero_tails(args.ruleset, args.precision)


if __name__ == "__main__":
    main()
