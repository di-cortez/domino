#!/usr/bin/env python3
"""Generate count- and response-guided lookup-reduction heatmaps.

The source coordinates are ``(agent_hand_size, opponent_hand_size)``. Every
arrow reduces the agent hand by one, so the three possible destinations are:

* up-left: ``(n - 1, m - 1)``;
* up: ``(n - 1, m)``;
* up-right: ``(n - 1, m + 1)``.

Cells covered by an explicit boundary rule or by the fixed lookup's 0.5%
eligibility threshold retain their original sample count. Other cells receive
an arrow. One PDF chooses the destination with the largest original sample
count. The other follows the most frequent complete immediate opponent
response in the raw histories. Missing evidence falls back to up-left.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RULESETS = ("double-three", "double-four", "double-five", "double-six")
DISPLAY_NAMES = {
    "double-three": "Double-three",
    "double-four": "Double-four",
    "double-five": "Double-five",
    "double-six": "Double-six",
}
ELIGIBILITY_FRACTION = 0.005

UP_LEFT = "up_left"
UP = "up"
UP_RIGHT = "up_right"
DIRECTION_ORDER = (UP_LEFT, UP, UP_RIGHT)
DIRECTION_SYMBOLS = {UP_LEFT: "↖", UP: "↑", UP_RIGHT: "↗"}
DIRECTION_COLORS = {
    UP_LEFT: "#1D4ED8",
    UP: "#7E22CE",
    UP_RIGHT: "#047857",
}
DIRECTION_LABELS = {
    UP_LEFT: "↖  (n-1, m-1): oponente joga",
    UP: "↑  (n-1, m): mão do oponente não muda",
    UP_RIGHT: "↗  (n-1, m+1): mão do oponente cresce",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--derived-dir",
        type=Path,
        default=SCRIPT_DIR / "derived",
        help="directory containing the original count manifests",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=SCRIPT_DIR / "raw",
        help="directory containing complete raw histories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="PDF destination directory (default: script directory)",
    )
    return parser.parse_args()


def format_integer(value: int) -> str:
    """Format an integer using the grouping convention of the figures."""
    return f"{int(value):,}".replace(",", ".")


def canonical_counts(manifest: dict[str, Any]) -> dict[tuple[int, int], int]:
    """Parse and validate one original cell-count mapping."""
    raw_counts = manifest.get("cell_sample_counts")
    if not isinstance(raw_counts, dict) or not raw_counts:
        raise ValueError("derived manifest has no cell_sample_counts")
    counts: dict[tuple[int, int], int] = {}
    for key, raw_count in raw_counts.items():
        pieces = key.split(",")
        if len(pieces) != 2:
            raise ValueError(f"invalid cell key {key!r}")
        cell = (int(pieces[0]), int(pieces[1]))
        count = int(raw_count)
        if min(cell) < 1 or count < 1 or count != raw_count:
            raise ValueError(f"invalid cell count {key!r}: {raw_count!r}")
        counts[cell] = count
    expected = int(manifest["summary"]["decisions"])
    if sum(counts.values()) != expected:
        raise ValueError("cell counts do not sum to summary.decisions")
    return counts


def load_original_counts(
    derived_dir: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[tuple[int, int], int]],
    dict[str, np.ndarray],
]:
    """Load original, pre-fixed/ad-hoc counts and construct dense matrices."""
    manifests = {}
    counts_by_ruleset = {}
    matrices = {}
    for ruleset in RULESETS:
        path = derived_dir / f"{ruleset}_reward_lookup_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("ruleset_name") != ruleset:
            raise ValueError(f"ruleset mismatch in {path}")
        counts = canonical_counts(manifest)
        rows = max(cell[0] for cell in counts)
        columns = max(cell[1] for cell in counts)
        matrix = np.zeros((rows, columns), dtype=np.int64)
        for (agent_size, opponent_size), count in counts.items():
            matrix[agent_size - 1, opponent_size - 1] = count
        manifests[ruleset] = manifest
        counts_by_ruleset[ruleset] = counts
        matrices[ruleset] = matrix
    return manifests, counts_by_ruleset, matrices


def load_raw_manifest(raw_dir: Path, ruleset: str) -> dict[str, Any]:
    """Load one complete raw-history manifest."""
    path = raw_dir / ruleset / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"raw history is not complete: {path}")
    configuration = manifest.get("configuration", {})
    if configuration.get("ruleset_name") != ruleset:
        raise ValueError(f"raw ruleset mismatch in {path}")
    return manifest


def response_direction(
    turns: list[dict[str, Any]],
    decision_position: int,
    opponent_seat: int,
    opponent_size_before: int,
) -> str | None:
    """Classify the complete opponent response immediately after a decision."""
    opponent_turns = []
    for turn in turns[decision_position + 1 :]:
        if int(turn["acting_seat"]) != opponent_seat:
            break
        opponent_turns.append(turn)
        if turn["post"]["game_over"]:
            break
    if not opponent_turns:
        return None

    opponent_size_after = int(opponent_turns[-1]["post"]["hand_sizes"][opponent_seat])
    change = opponent_size_after - opponent_size_before
    if change < -1:
        raise ValueError(f"opponent hand unexpectedly shrank by {-change} tiles")
    if change == -1:
        return UP_LEFT
    if change == 0:
        return UP
    return UP_RIGHT


def collect_response_counts(
    raw_dir: Path,
    expected_counts: dict[str, dict[tuple[int, int], int]],
) -> dict[str, dict[tuple[int, int], Counter[str]]]:
    """Stream raw games and count the immediate response destination per cell."""
    results = {}
    for ruleset in RULESETS:
        print(f"Scanning immediate opponent responses: {ruleset}...", flush=True)
        manifest = load_raw_manifest(raw_dir, ruleset)
        decision_counts: Counter[tuple[int, int]] = Counter()
        response_counts: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
        missing_responses: Counter[tuple[int, int]] = Counter()
        for chunk in manifest["chunks"]:
            path = raw_dir / ruleset / chunk["file"]
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    neural_seat = int(record["neural_seat"])
                    opponent_seat = 1 - neural_seat
                    turns = record["turns"]
                    for position, turn in enumerate(turns):
                        if turn["neural_decision_index"] is None:
                            continue
                        hand_sizes = turn["pre"]["hand_sizes"]
                        cell = (
                            int(hand_sizes[neural_seat]),
                            int(hand_sizes[opponent_seat]),
                        )
                        decision_counts[cell] += 1
                        direction = response_direction(
                            turns,
                            position,
                            opponent_seat,
                            cell[1],
                        )
                        if direction is None:
                            missing_responses[cell] += 1
                        else:
                            response_counts[cell][direction] += 1

        if dict(decision_counts) != expected_counts[ruleset]:
            raise ValueError(f"raw decision counts differ in {ruleset}")
        unexpected_missing = {
            cell: count for cell, count in missing_responses.items() if cell[0] > 1
        }
        if unexpected_missing:
            raise ValueError(
                f"nonterminal decisions lack an opponent response: {unexpected_missing}"
            )
        results[ruleset] = response_counts
        classified = sum(sum(counter.values()) for counter in response_counts.values())
        print(
            f"  {format_integer(classified)} responses classified; "
            f"{format_integer(sum(missing_responses.values()))} terminal one-tile "
            "decisions intentionally have no response."
        )
    return results


def destination(cell: tuple[int, int], direction: str) -> tuple[int, int]:
    """Return the coordinate represented by one arrow direction."""
    agent_size, opponent_size = cell
    offsets = {
        UP_LEFT: (-1, -1),
        UP: (-1, 0),
        UP_RIGHT: (-1, 1),
    }
    agent_offset, opponent_offset = offsets[direction]
    return agent_size + agent_offset, opponent_size + opponent_offset


def valid_destination(
    cell: tuple[int, int], direction: str, matrix: np.ndarray
) -> bool:
    """Return whether one arrow remains inside the original heatmap."""
    agent_size, opponent_size = destination(cell, direction)
    return 1 <= agent_size <= matrix.shape[0] and 1 <= opponent_size <= matrix.shape[1]


def choose_direction(scores: dict[str, int], valid: set[str]) -> str:
    """Choose the largest score with deterministic conservative tie-breaking."""
    best_direction = UP_LEFT
    best_score = -1
    for direction in DIRECTION_ORDER:
        if direction not in valid:
            continue
        score = int(scores.get(direction, 0))
        if score > best_score:
            best_direction = direction
            best_score = score
    if best_score <= 0:
        return UP_LEFT
    return best_direction


def count_guided_direction(cell: tuple[int, int], matrix: np.ndarray) -> str:
    """Choose the candidate destination with the largest original count."""
    scores = {}
    valid = set()
    for direction in DIRECTION_ORDER:
        if not valid_destination(cell, direction, matrix):
            continue
        valid.add(direction)
        agent_size, opponent_size = destination(cell, direction)
        scores[direction] = int(matrix[agent_size - 1, opponent_size - 1])
    return choose_direction(scores, valid)


def response_guided_direction(
    cell: tuple[int, int],
    matrix: np.ndarray,
    response_counts: dict[tuple[int, int], Counter[str]],
) -> str:
    """Choose the most frequent immediate response destination."""
    valid = {
        direction
        for direction in DIRECTION_ORDER
        if valid_destination(cell, direction, matrix)
    }
    return choose_direction(dict(response_counts.get(cell, {})), valid)


def eligibility_threshold(manifest: dict[str, Any]) -> int:
    """Return the fixed lookup's original 0.5% support threshold."""
    return math.ceil(
        ELIGIBILITY_FRACTION * int(manifest["summary"]["decisions"])
    )


def preserve_count(agent_size: int, opponent_size: int, count: int, threshold: int) -> bool:
    """Keep structural/boundary cells and statistically eligible observations."""
    return agent_size <= 2 or opponent_size <= 2 or count >= threshold


def build_directions(
    method: str,
    matrix: np.ndarray,
    threshold: int,
    response_counts: dict[tuple[int, int], Counter[str]],
) -> dict[tuple[int, int], str]:
    """Build one complete map of arrows for unsupported cells."""
    directions = {}
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            cell = (row + 1, column + 1)
            count = int(matrix[row, column])
            if preserve_count(*cell, count, threshold):
                continue
            if method == "destination_counts":
                direction = count_guided_direction(cell, matrix)
            elif method == "immediate_response":
                direction = response_guided_direction(
                    cell, matrix, response_counts
                )
            else:
                raise ValueError(f"unknown reduction method {method!r}")
            directions[cell] = direction
    return directions


def text_color(count: int, normalization: LogNorm) -> str:
    """Choose a readable count-label color for one heatmap cell."""
    if count == 0:
        return "#94A3B8"
    return "white" if normalization(count) > 0.69 else "#111827"


def draw_page(
    pdf: PdfPages,
    ruleset: str,
    manifest: dict[str, Any],
    matrix: np.ndarray,
    directions: dict[tuple[int, int], str],
    global_max: int,
    method_description: str,
) -> None:
    """Render one ruleset page into an open multi-page PDF."""
    rows, columns = matrix.shape
    figure_width = max(9.4, 0.88 * columns + 3.8)
    figure_height = max(7.2, 0.72 * rows + 3.6)
    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height), constrained_layout=True
    )
    masked = np.ma.masked_where(matrix == 0, matrix)
    color_map = plt.get_cmap("YlOrRd").copy()
    color_map.set_bad("#F1F5F9")
    normalization = LogNorm(vmin=1, vmax=global_max)
    image = axis.imshow(masked, cmap=color_map, norm=normalization, aspect="equal")

    axis.set_xticks(np.arange(columns), labels=np.arange(1, columns + 1))
    axis.set_yticks(np.arange(rows), labels=np.arange(1, rows + 1))
    axis.set_xlabel("Peças do oponente (m)", fontsize=11, labelpad=10)
    axis.set_ylabel("Peças do agente neural (n)", fontsize=11, labelpad=10)
    axis.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    axis.set_xticks(np.arange(-0.5, columns, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_visible(False)

    for row in range(rows):
        for column in range(columns):
            cell = (row + 1, column + 1)
            count = int(matrix[row, column])
            direction = directions.get(cell)
            if direction is None:
                axis.text(
                    column,
                    row,
                    format_integer(count),
                    ha="center",
                    va="center",
                    fontsize=7.2 if max(rows, columns) >= 10 else 8.5,
                    color=text_color(count, normalization),
                    fontweight="semibold" if count >= 10_000 else "normal",
                )
                continue
            arrow = axis.text(
                column,
                row,
                DIRECTION_SYMBOLS[direction],
                ha="center",
                va="center",
                fontsize=21 if max(rows, columns) >= 10 else 24,
                color=DIRECTION_COLORS[direction],
                fontweight="bold",
            )
            arrow.set_path_effects(
                [path_effects.Stroke(linewidth=2.7, foreground="white"), path_effects.Normal()]
            )

    threshold = eligibility_threshold(manifest)
    summary = manifest["summary"]
    axis.set_title(
        f"{DISPLAY_NAMES[ruleset]} - mapa de redução de células\n"
        f"{method_description}\n"
        f"{format_integer(int(summary['decisions']))} decisões; "
        f"limiar de suporte = {format_integer(threshold)}",
        fontsize=13.5,
        fontweight="bold",
        pad=17,
    )
    color_bar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.035)
    color_bar.set_label("Observações originais (escala log)", labelpad=11)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=f"${DIRECTION_SYMBOLS[direction]}$",
            color="none",
            markerfacecolor=DIRECTION_COLORS[direction],
            markeredgecolor=DIRECTION_COLORS[direction],
            markersize=13,
            label=DIRECTION_LABELS[direction],
        )
        for direction in DIRECTION_ORDER
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#FDE68A",
            markeredgecolor="#64748B",
            markersize=10,
            label="Número: regra explícita ou suporte suficiente",
        )
    )
    axis.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    pdf.savefig(figure, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def write_pdf(
    path: Path,
    method: str,
    manifests: dict[str, dict[str, Any]],
    matrices: dict[str, np.ndarray],
    response_counts: dict[str, dict[tuple[int, int], Counter[str]]],
    global_max: int,
) -> None:
    """Write one four-page reduction PDF."""
    descriptions = {
        "destination_counts": "Seta escolhida pelo destino com mais observações originais",
        "immediate_response": "Seta escolhida pela resposta imediata mais frequente do oponente",
    }
    metadata = {
        "Title": descriptions[method],
        "Subject": "Hand-size lookup reduction maps",
        "Creator": "generate_reduction_heatmaps.py",
        "CreationDate": None,
        "ModDate": None,
    }
    with PdfPages(path, metadata=metadata) as pdf:
        for ruleset in RULESETS:
            threshold = eligibility_threshold(manifests[ruleset])
            directions = build_directions(
                method,
                matrices[ruleset],
                threshold,
                response_counts[ruleset],
            )
            draw_page(
                pdf,
                ruleset,
                manifests[ruleset],
                matrices[ruleset],
                directions,
                global_max,
                descriptions[method],
            )
            counts = Counter(directions.values())
            print(
                f"  {ruleset}: {len(directions)} arrows "
                f"(↖ {counts[UP_LEFT]}, ↑ {counts[UP]}, ↗ {counts[UP_RIGHT]})"
            )


def main() -> None:
    args = parse_args()
    derived_dir = args.derived_dir.resolve()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests, original_counts, matrices = load_original_counts(derived_dir)
    response_counts = collect_response_counts(raw_dir, original_counts)
    global_max = max(int(matrix.max()) for matrix in matrices.values())
    outputs = {
        "destination_counts": output_dir / "heatmaps_reduction_by_destination_counts.pdf",
        "immediate_response": output_dir / "heatmaps_reduction_by_immediate_response.pdf",
    }
    for method, path in outputs.items():
        print(f"Writing {path.name}...")
        write_pdf(
            path,
            method,
            manifests,
            matrices,
            response_counts,
            global_max,
        )
    print("Generated PDFs:")
    for path in outputs.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
