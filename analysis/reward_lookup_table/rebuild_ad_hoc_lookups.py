#!/usr/bin/env python3
"""Rebuild the reviewed ad hoc lookups reproducibly from ``fixed/``."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import edit_ad_hoc_cells as editor


SCRIPT_DIR = Path(__file__).resolve().parent
RULESETS = ("double-three", "double-four", "double-five", "double-six")
REMOVED_CELLS = {
    "double-three": ("1,1",),
    "double-four": ("1,1", "1,2"),
    "double-five": (),
    "double-six": ("1,1",),
}
COPIED_CELLS = {
    "double-three": (("4,1", "5,1"), ("5,2", "6,2")),
    "double-four": (("5,2", "6,2"),),
    "double-five": (),
    "double-six": (),
}
TAIL_TRUNCATION_PRECISION = 3


def _artifact_paths(root: Path, ruleset: str) -> tuple[Path, Path]:
    stem = f"{ruleset}_fixed_signed_reward_lookup"
    return root / f"{stem}.json.gz", root / f"{stem}_manifest.json"


def rebuild_ruleset(fixed_root: Path, output_root: Path, ruleset: str) -> None:
    """Reset one ad hoc lookup, then replay all reviewed explicit edits."""
    source_paths = _artifact_paths(fixed_root, ruleset)
    destination_paths = _artifact_paths(output_root, ruleset)
    for source in source_paths:
        if not source.is_file():
            raise FileNotFoundError(f"required fixed artifact not found: {source}")
    output_root.mkdir(parents=True, exist_ok=True)
    for source, destination in zip(source_paths, destination_paths, strict=True):
        shutil.copyfile(source, destination)

    if REMOVED_CELLS[ruleset]:
        editor.remove_cells(ruleset, list(REMOVED_CELLS[ruleset]))
    editor.trim_zero_tails(ruleset, TAIL_TRUNCATION_PRECISION)
    for source_cell, target_cell in COPIED_CELLS[ruleset]:
        editor.copy_cell(ruleset, source_cell, target_cell)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixed-root",
        type=Path,
        default=SCRIPT_DIR / "fixed",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR / "ad_hoc",
    )
    parser.add_argument(
        "--rulesets",
        nargs="+",
        choices=RULESETS,
        default=RULESETS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixed_root = args.fixed_root.resolve()
    output_root = args.output_root.resolve()
    if fixed_root == output_root:
        raise ValueError("fixed and ad hoc roots must be different directories")
    editor.AD_HOC_ROOT = output_root
    for ruleset in args.rulesets:
        rebuild_ruleset(fixed_root, output_root, ruleset)


if __name__ == "__main__":
    main()
