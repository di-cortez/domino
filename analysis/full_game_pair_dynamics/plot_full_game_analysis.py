"""Plot comparative state dynamics and timing for every full-game pairing."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "full_game_pair_analysis.json.gz"
MATCHUP_LABELS = {
    "random_vs_random": "Random × random",
    "random_vs_heuristic": "Random × heuristic",
    "random_vs_neural": "Random × neural",
    "heuristic_vs_heuristic": "Heuristic × heuristic",
    "heuristic_vs_neural": "Heuristic × neural",
    "neural_vs_neural": "Neural × neural",
}
MATCHUP_COLORS = {
    "random_vs_random": "#8c8c8c",
    "random_vs_heuristic": "#4c78a8",
    "random_vs_neural": "#72b7b2",
    "heuristic_vs_heuristic": "#f58518",
    "heuristic_vs_neural": "#e45756",
    "neural_vs_neural": "#7b2cbf",
}
AGENT_LABELS = {
    "random": "Random",
    "heuristic": "Heuristic",
    "neural": "Neural",
}
AGENT_COLORS = {
    "random": "#8c8c8c",
    "heuristic": "#f58518",
    "neural": "#7b2cbf",
}
LOCATION_TITLES = {
    "player0_hand": "Player 0 hand",
    "player1_hand": "Player 1 hand",
    "table": "Table",
    "stock": "Stock",
}
LOCATION_COLORS = {
    "player0_hand": "#2f6db0",
    "player1_hand": "#d46a1f",
    "table": "#2b9348",
    "stock": "#7b2cbf",
}
TIMING_COMPONENTS = (
    "state",
    "legal_actions",
    "agent_decision",
    "engine_transition",
)
TIMING_LABELS = {
    "state": "Build state",
    "legal_actions": "Legal actions",
    "agent_decision": "Agent decision",
    "engine_transition": "Engine transition",
    "overhead": "Turn overhead",
}
TIMING_COLORS = {
    "state": "#4c78a8",
    "legal_actions": "#72b7b2",
    "agent_decision": "#e45756",
    "engine_transition": "#f2cf5b",
    "overhead": "#b8b8b8",
}
DECISION_CLASS_LABELS = {
    "forced_draw": "Forced draw",
    "forced_pass": "Forced pass",
    "forced_tile": "Forced tile",
    "voluntary_choice": "Voluntary choice",
}
PIP_COLORS = plt.get_cmap("tab10")(np.linspace(0, 1, 7))


def load_report(path):
    """Load one compact comparative analysis report, including gzip output."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def save_figure(fig, output_dir, name, dpi):
    """Save one white-background PNG and release its memory."""
    path = output_dir / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def histogram_arrays(summary):
    """Expand a compact consecutive-value histogram into arrays."""
    histogram = summary["histogram"]
    first = int(histogram["first_value"])
    values = np.arange(first, first + len(histogram["counts"]))
    return values, np.asarray(histogram["counts"], dtype=float)


def ordered_matchups(data):
    """Return report matchups in their stable generator order."""
    return [
        (key, data["matchups"][key]) for key in data["matchup_order"]
    ]


def finite_turn_rows(matchup, timing_component=None):
    """Return turn rows that contain decision timing rather than final states."""
    if timing_component is None:
        return matchup["turns"]
    return [
        row for row in matchup["turns"]
        if timing_component in row["timing_us"]
    ]


def plot_mean_sizes(data, output_dir, dpi):
    """Compare mean hand, table, and stock sizes in all six pairings."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 15), sharex=True, sharey=True)
    for axis, (key, matchup) in zip(axes.flat, ordered_matchups(data)):
        turns = matchup["turns"]
        x_values = np.asarray([row["turn"] for row in turns])
        for location in LOCATION_TITLES:
            means = [row["sizes"][location]["mean"] for row in turns]
            axis.plot(
                x_values,
                means,
                linewidth=1.9,
                color=LOCATION_COLORS[location],
                label=LOCATION_TITLES[location],
            )
        axis.set_title(MATCHUP_LABELS[key])
        axis.grid(alpha=0.20)
    for axis in axes[-1]:
        axis.set_xlabel("Completed engine turns")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean domino count")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        frameon=False,
    )
    fig.suptitle("Mean location sizes through each matchup", y=0.995, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_figure(fig, output_dir, "01_mean_location_sizes.png", dpi)


def plot_duration_and_survival(data, output_dir, dpi):
    """Compare duration distributions and the fraction reaching each turn."""
    fig, (duration_axis, survival_axis) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True
    )
    for key, matchup in ordered_matchups(data):
        summary = matchup["game_summary"]["duration_in_engine_turns"]
        turns, counts = histogram_arrays(summary)
        percentages = 100.0 * counts / counts.sum()
        duration_axis.plot(
            turns,
            percentages,
            linewidth=1.8,
            color=MATCHUP_COLORS[key],
            label=f"{MATCHUP_LABELS[key]} (mean {summary['mean']:.1f})",
        )
        rows = matchup["turns"]
        survival_axis.plot(
            [row["turn"] for row in rows],
            [row["games_observed_percent"] for row in rows],
            linewidth=2,
            color=MATCHUP_COLORS[key],
            label=MATCHUP_LABELS[key],
        )
    duration_axis.set_title("Game-duration distribution")
    duration_axis.set_ylabel("Games ending at turn (%)")
    duration_axis.legend(ncols=2, frameon=False, fontsize=9)
    duration_axis.grid(alpha=0.20)
    survival_axis.set_title("Survival of the game cohort")
    survival_axis.set_xlabel("Completed engine turns")
    survival_axis.set_ylabel("Games reaching turn (%)")
    survival_axis.set_ylim(0, 103)
    survival_axis.grid(alpha=0.20)
    fig.tight_layout()
    return save_figure(fig, output_dir, "02_duration_and_survival.png", dpi)


def plot_opening_tiles(data, output_dir, dpi):
    """Show how starting-policy differences alter the opening domino."""
    rows = ordered_matchups(data)
    matrix = np.asarray([
        matchup["game_summary"]["opening_table_tile"]["percent"]
        for _, matchup in rows
    ])
    labels = data["schema"]["tile_order"]
    fig, axis = plt.subplots(figsize=(15, 5.5))
    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="YlGn",
        vmin=0,
    )
    axis.set_xticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    axis.set_yticks(np.arange(len(rows)))
    axis.set_yticklabels([MATCHUP_LABELS[key] for key, _ in rows])
    axis.set_xlabel("First domino placed")
    axis.set_title("Opening-tile frequency by matchup")
    colorbar = fig.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("Games (%)")
    fig.tight_layout()
    return save_figure(fig, output_dir, "03_opening_tile_heatmap.png", dpi)


def size_distribution_matrix(turns, location):
    """Return size-by-turn percentages for one physical location."""
    max_size = max(row["sizes"][location]["max"] for row in turns)
    matrix = np.zeros((max_size + 1, len(turns)), dtype=float)
    for column, row in enumerate(turns):
        summary = row["sizes"][location]
        sizes, counts = histogram_arrays(summary)
        matrix[sizes, column] = 100.0 * counts / summary["count"]
    return matrix


def plot_size_distributions(data, output_dir, dpi):
    """Retain the complete hand/table/stock distributions for every pair."""
    rows = ordered_matchups(data)
    locations = tuple(LOCATION_TITLES)
    fig, axes = plt.subplots(
        len(rows), len(locations), figsize=(22, 25), squeeze=False
    )
    image = None
    for row_index, (key, matchup) in enumerate(rows):
        for column, location in enumerate(locations):
            axis = axes[row_index, column]
            matrix = size_distribution_matrix(matchup["turns"], location)
            image = axis.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                vmin=0,
                vmax=100,
            )
            if row_index == 0:
                axis.set_title(LOCATION_TITLES[location])
            if column == 0:
                axis.set_ylabel(f"{MATCHUP_LABELS[key]}\nDomino count")
            if row_index == len(rows) - 1:
                axis.set_xlabel("Completed turns")
    fig.subplots_adjust(top=0.955, right=0.91, hspace=0.25, wspace=0.20)
    colorbar_axis = fig.add_axes((0.93, 0.12, 0.012, 0.76))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("States at that turn (%)")
    fig.suptitle("Location-size distributions by matchup and turn", fontsize=17)
    return save_figure(fig, output_dir, "04_size_distribution_heatmaps.png", dpi)


def tile_presence_matrix(turns, location):
    """Return domino-presence percentages by tile and absolute turn."""
    return np.asarray([
        row["composition"][location]["tile_presence_percent"]
        for row in turns
    ]).T


def plot_tile_presence(data, output_dir, dpi):
    """Compare all individual dominoes in all locations and matchups."""
    rows = ordered_matchups(data)
    locations = tuple(LOCATION_TITLES)
    matrices = {
        (key, location): tile_presence_matrix(matchup["turns"], location)
        for key, matchup in rows
        for location in locations
    }
    maximum = max(float(matrix.max()) for matrix in matrices.values())
    tile_labels = data["schema"]["tile_order"]
    fig, axes = plt.subplots(
        len(rows), len(locations), figsize=(23, 27), squeeze=False
    )
    image = None
    for row_index, (key, _matchup) in enumerate(rows):
        for column, location in enumerate(locations):
            axis = axes[row_index, column]
            image = axis.imshow(
                matrices[(key, location)],
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap="magma",
                vmin=0,
                vmax=maximum,
            )
            if row_index == 0:
                axis.set_title(LOCATION_TITLES[location])
            if column == 0:
                axis.set_yticks(np.arange(len(tile_labels)))
                axis.set_yticklabels(tile_labels, fontsize=6)
                axis.set_ylabel(f"{MATCHUP_LABELS[key]}\nDomino")
            else:
                axis.set_yticks([])
            if row_index == len(rows) - 1:
                axis.set_xlabel("Completed turns")
    fig.subplots_adjust(top=0.96, right=0.91, hspace=0.18, wspace=0.08)
    colorbar_axis = fig.add_axes((0.93, 0.12, 0.012, 0.76))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Tile present in observed states (%)")
    fig.suptitle(
        "Individual-domino location frequency by matchup and turn",
        fontsize=17,
    )
    return save_figure(fig, output_dir, "05_tile_presence_heatmaps.png", dpi)


def plot_pip_composition(data, output_dir, dpi):
    """Compare endpoint-pip composition in every location and matchup."""
    rows = ordered_matchups(data)
    locations = tuple(LOCATION_TITLES)
    fig, axes = plt.subplots(
        len(rows), len(locations), figsize=(23, 23), sharey=True, squeeze=False
    )
    for row_index, (key, matchup) in enumerate(rows):
        turns = matchup["turns"]
        x_values = np.asarray([row["turn"] for row in turns])
        for column, location in enumerate(locations):
            axis = axes[row_index, column]
            values = np.asarray([
                row["composition"][location]["pip_composition"][
                    "endpoint_share_percent"
                ]
                for row in turns
            ])
            for pip, color in zip(range(7), PIP_COLORS):
                axis.plot(x_values, values[:, pip], color=color, linewidth=1.1)
            axis.grid(alpha=0.16)
            if row_index == 0:
                axis.set_title(LOCATION_TITLES[location])
            if column == 0:
                axis.set_ylabel(f"{MATCHUP_LABELS[key]}\nEndpoint share (%)")
            if row_index == len(rows) - 1:
                axis.set_xlabel("Completed turns")
    handles = [
        plt.Line2D([0], [0], color=color, linewidth=2, label=str(pip))
        for pip, color in zip(range(7), PIP_COLORS)
    ]
    fig.legend(
        handles=handles,
        title="Pip",
        ncols=7,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.982),
        frameon=False,
    )
    fig.suptitle("Pip composition by matchup, location, and turn", y=0.998, fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return save_figure(fig, output_dir, "06_pip_composition.png", dpi)


def plot_game_timing(data, output_dir, dpi):
    """Compare complete simulation latency and sequential throughput."""
    rows = ordered_matchups(data)
    labels = [MATCHUP_LABELS[key] for key, _ in rows]
    medians_ms = []
    means_ms = []
    p95_ms = []
    throughput = []
    for _key, matchup in rows:
        timing = matchup["timing_summary_us"]["per_game"]["simulation"]
        medians_ms.append(timing["median"] / 1_000)
        means_ms.append(timing["mean"] / 1_000)
        p95_ms.append(timing["p95"] / 1_000)
        throughput.append(1_000_000 / timing["mean"])
    positions = np.arange(len(rows))
    fig, (latency_axis, rate_axis) = plt.subplots(1, 2, figsize=(15, 6.5))
    width = 0.24
    latency_axis.bar(
        positions - width,
        medians_ms,
        width,
        color="#4c78a8",
        label="Median",
    )
    latency_axis.bar(
        positions,
        means_ms,
        width,
        color="#f58518",
        label="Mean",
    )
    latency_axis.bar(
        positions + width,
        p95_ms,
        width,
        color="#e45756",
        label="P95",
    )
    latency_axis.set_xticks(positions)
    latency_axis.set_xticklabels(labels, rotation=35, ha="right")
    latency_axis.set_ylabel("Simulation wall time per game (ms)")
    latency_axis.set_title("Single-process game latency")
    latency_axis.legend(frameon=False)
    latency_axis.grid(axis="y", alpha=0.20)
    bars = rate_axis.barh(
        positions,
        throughput,
        color=[MATCHUP_COLORS[key] for key, _ in rows],
    )
    rate_axis.set_yticks(positions)
    rate_axis.set_yticklabels(labels)
    rate_axis.invert_yaxis()
    rate_axis.set_xlabel("Sequential simulations per second")
    rate_axis.set_title("Throughput implied by mean simulation time")
    rate_axis.grid(axis="x", alpha=0.20)
    for bar, value in zip(bars, throughput):
        rate_axis.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {value:.1f}",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    return save_figure(fig, output_dir, "07_game_simulation_timing.png", dpi)


def plot_timing_components(data, output_dir, dpi):
    """Decompose mean complete-turn time by matchup."""
    rows = ordered_matchups(data)
    positions = np.arange(len(rows))
    labels = [MATCHUP_LABELS[key] for key, _ in rows]
    bottom = np.zeros(len(rows))
    fig, axis = plt.subplots(figsize=(13, 7))
    for component in TIMING_COMPONENTS:
        values = np.asarray([
            matchup["timing_summary_us"]["per_move"][component]["mean"]
            for _, matchup in rows
        ])
        axis.bar(
            positions,
            values,
            bottom=bottom,
            color=TIMING_COLORS[component],
            label=TIMING_LABELS[component],
        )
        bottom += values
    turn_means = np.asarray([
        matchup["timing_summary_us"]["per_move"]["turn_wall"]["mean"]
        for _, matchup in rows
    ])
    overhead = np.maximum(0, turn_means - bottom)
    axis.bar(
        positions,
        overhead,
        bottom=bottom,
        color=TIMING_COLORS["overhead"],
        label=TIMING_LABELS["overhead"],
    )
    axis.scatter(
        positions,
        turn_means,
        marker="_",
        s=500,
        linewidths=2,
        color="black",
        label="Measured turn wall",
        zorder=5,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_ylabel("Mean wall time per move (µs)")
    axis.set_title("Where one simulated turn spends time")
    axis.legend(ncols=3, frameon=False)
    axis.grid(axis="y", alpha=0.20)
    fig.tight_layout()
    return save_figure(fig, output_dir, "08_move_timing_components.png", dpi)


def plot_turn_time_by_matchup(data, output_dir, dpi):
    """Show how complete-turn latency changes through each game."""
    fig, (mean_axis, p95_axis) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True
    )
    for key, matchup in ordered_matchups(data):
        rows = finite_turn_rows(matchup, "turn_wall")
        turns = [row["turn"] for row in rows]
        means = [row["timing_us"]["turn_wall"]["mean"] for row in rows]
        p95 = [row["timing_us"]["turn_wall"]["p95"] for row in rows]
        mean_axis.plot(
            turns,
            means,
            linewidth=2,
            color=MATCHUP_COLORS[key],
            label=MATCHUP_LABELS[key],
        )
        p95_axis.plot(
            turns,
            p95,
            linewidth=1.8,
            color=MATCHUP_COLORS[key],
        )
    mean_axis.set_title("Mean complete-turn wall time")
    mean_axis.set_ylabel("Microseconds")
    mean_axis.legend(ncols=2, frameon=False)
    mean_axis.grid(alpha=0.20)
    p95_axis.set_title("P95 complete-turn wall time")
    p95_axis.set_ylabel("Microseconds")
    p95_axis.set_xlabel("Completed engine turns")
    p95_axis.grid(alpha=0.20)
    fig.tight_layout()
    return save_figure(fig, output_dir, "09_turn_time_by_matchup.png", dpi)


def plot_agent_decision_distributions(data, output_dir, dpi):
    """Compare policy-only decision latency for random, heuristic, and neural."""
    timing = data["global_timing_summary_us"]["per_move_by_agent"]
    agents = [agent for agent in data["agent_order"] if agent in timing]
    positions = np.arange(len(agents))
    width = 0.24
    fig, axis = plt.subplots(figsize=(10, 6.5))
    for offset, statistic, label, color in (
        (-width, "median", "Median", "#4c78a8"),
        (0, "mean", "Mean", "#f58518"),
        (width, "p95", "P95", "#e45756"),
    ):
        values = [timing[agent]["agent_decision"][statistic] for agent in agents]
        bars = axis.bar(positions + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    axis.set_xticks(positions)
    axis.set_xticklabels([AGENT_LABELS[agent] for agent in agents])
    axis.set_ylabel("Agent.choose_move wall time (µs)")
    axis.set_title("Decision latency by player type across every matchup")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.20)
    fig.tight_layout()
    return save_figure(fig, output_dir, "10_agent_decision_latency.png", dpi)


def plot_agent_time_by_turn(data, output_dir, dpi):
    """Compare type-specific decision latency as game state changes."""
    rows = data["global_timing_summary_us"]["per_move_by_agent_and_turn"]
    fig, (mean_axis, p95_axis) = plt.subplots(
        2, 1, figsize=(13, 9), sharex=True
    )
    for agent in data["agent_order"]:
        agent_rows = [
            row for row in rows
            if agent in row["agents"]
            and "agent_decision" in row["agents"][agent]
        ]
        turns = [row["turn"] for row in agent_rows]
        means = [
            row["agents"][agent]["agent_decision"]["mean"]
            for row in agent_rows
        ]
        p95 = [
            row["agents"][agent]["agent_decision"]["p95"]
            for row in agent_rows
        ]
        mean_axis.plot(
            turns,
            means,
            color=AGENT_COLORS[agent],
            linewidth=2.2,
            label=AGENT_LABELS[agent],
        )
        p95_axis.plot(
            turns,
            p95,
            color=AGENT_COLORS[agent],
            linewidth=2,
        )
    mean_axis.set_title("Mean policy decision time by absolute turn")
    mean_axis.set_ylabel("Microseconds")
    mean_axis.legend(frameon=False, ncols=3)
    mean_axis.grid(alpha=0.20)
    p95_axis.set_title("P95 policy decision time by absolute turn")
    p95_axis.set_ylabel("Microseconds")
    p95_axis.set_xlabel("Completed engine turns")
    p95_axis.grid(alpha=0.20)
    fig.tight_layout()
    return save_figure(fig, output_dir, "11_agent_decision_time_by_turn.png", dpi)


def plot_decision_classes(data, output_dir, dpi):
    """Separate rule-forced actions from actual multi-option choices."""
    by_agent = data["global_timing_summary_us"][
        "agent_decision_by_agent_and_class"
    ]
    agents = [agent for agent in data["agent_order"] if agent in by_agent]
    categories = list(data["schema"]["decision_class_order"])
    positions = np.arange(len(categories))
    width = 0.24
    fig, (share_axis, timing_axis) = plt.subplots(2, 1, figsize=(13, 10))
    for agent_index, agent in enumerate(agents):
        summaries = by_agent[agent]
        total = sum(
            summaries[category]["count"]
            for category in categories
            if category in summaries
        )
        shares = [
            100.0 * summaries[category]["count"] / total
            if category in summaries else 0
            for category in categories
        ]
        means = np.asarray([
            summaries[category]["mean"] if category in summaries else 0
            for category in categories
        ])
        offset = (agent_index - (len(agents) - 1) / 2) * width
        share_axis.bar(
            positions + offset,
            shares,
            width,
            color=AGENT_COLORS[agent],
            label=AGENT_LABELS[agent],
        )
        timing_bars = timing_axis.bar(
            positions + offset,
            np.maximum(means, 0.5) - 0.5,
            width,
            bottom=0.5,
            color=AGENT_COLORS[agent],
            label=AGENT_LABELS[agent],
        )
        for bar, value in zip(timing_bars, means):
            timing_axis.text(
                bar.get_x() + bar.get_width() / 2,
                max(value, 0.5) * 1.08,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    labels = [DECISION_CLASS_LABELS[category] for category in categories]
    share_axis.set_xticks(positions)
    share_axis.set_xticklabels(labels)
    share_axis.set_ylabel("Share of that agent's turns (%)")
    share_axis.set_title("Decision-type frequency")
    share_axis.legend(frameon=False, ncols=3)
    share_axis.grid(axis="y", alpha=0.20)
    timing_axis.set_xticks(positions)
    timing_axis.set_xticklabels(labels)
    timing_axis.set_yscale("log")
    timing_axis.set_ylim(bottom=0.5)
    timing_axis.set_ylabel("Mean Agent.choose_move wall time (µs, log scale)")
    timing_axis.set_title("Decision latency within each decision type")
    timing_axis.grid(axis="y", alpha=0.20)
    fig.tight_layout()
    return save_figure(fig, output_dir, "12_decision_class_timing.png", dpi)


def plot_turn_timing_heatmaps(data, output_dir, dpi):
    """Provide a compact matchup-by-turn view of mean and P95 latency."""
    rows = ordered_matchups(data)
    max_turn = max(
        max(row["turn"] for row in finite_turn_rows(matchup, "turn_wall"))
        for _, matchup in rows
    )
    mean_matrix = np.full((len(rows), max_turn + 1), np.nan)
    p95_matrix = np.full((len(rows), max_turn + 1), np.nan)
    for row_index, (_key, matchup) in enumerate(rows):
        for turn in finite_turn_rows(matchup, "turn_wall"):
            column = int(turn["turn"])
            summary = turn["timing_us"]["turn_wall"]
            mean_matrix[row_index, column] = summary["mean"]
            p95_matrix[row_index, column] = summary["p95"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    for axis, matrix, title in zip(
        axes,
        (mean_matrix, p95_matrix),
        ("Mean complete-turn time", "P95 complete-turn time"),
    ):
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
        )
        axis.set_yticks(np.arange(len(rows)))
        axis.set_yticklabels([MATCHUP_LABELS[key] for key, _ in rows])
        axis.set_title(title)
        colorbar = fig.colorbar(image, ax=axis, pad=0.015)
        colorbar.set_label("Microseconds")
    axes[-1].set_xlabel("Completed engine turns")
    fig.tight_layout()
    return save_figure(fig, output_dir, "13_turn_timing_heatmaps.png", dpi)


def _reward_agents(matchup):
    """Return stable agent names with per-game raw-reward summaries."""
    summaries = matchup["raw_reward_summary"]["per_game_by_agent"]
    return [agent for agent in AGENT_LABELS if agent in summaries]


def plot_raw_reward_components(data, output_dir, dpi):
    """Compare mean raw event and terminal components by matchup/agent."""
    rows = ordered_matchups(data)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    components = (
        ("event_sum", "Local draw/pass"),
        ("empty_hand_component", "Empty-hand terminal"),
        ("blocked_component", "Blocked terminal"),
    )
    widths = 0.22
    for axis, (key, matchup) in zip(axes.flat, rows):
        agents = _reward_agents(matchup)
        x = np.arange(len(agents), dtype=float)
        summaries = matchup["raw_reward_summary"]["per_game_by_agent"]
        for index, (component, label) in enumerate(components):
            means = [summaries[agent][component]["mean"] for agent in agents]
            offset = (index - 1) * widths
            axis.bar(x + offset, means, width=widths, label=label)
        axis.axhline(0.0, linewidth=0.8, color="black")
        axis.set_xticks(x)
        axis.set_xticklabels([AGENT_LABELS[agent] for agent in agents])
        axis.set_title(MATCHUP_LABELS[key])
        axis.set_ylabel("Mean raw reward / game")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Undiscounted raw reward components", y=1.03)
    return save_figure(fig, output_dir, "14_raw_reward_components.png", dpi)


def plot_raw_total_reward(data, output_dir, dpi):
    """Show mean and standard deviation of total raw reward by matchup/agent."""
    rows = ordered_matchups(data)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (key, matchup) in zip(axes.flat, rows):
        agents = _reward_agents(matchup)
        summaries = matchup["raw_reward_summary"]["per_game_by_agent"]
        means = np.asarray([summaries[agent]["total"]["mean"] for agent in agents])
        stds = np.asarray([summaries[agent]["total"]["stddev"] for agent in agents])
        x = np.arange(len(agents))
        colors = [AGENT_COLORS[agent] for agent in agents]
        axis.bar(x, means, yerr=stds, capsize=3, color=colors)
        axis.axhline(0.0, linewidth=0.8, color="black")
        axis.set_xticks(x)
        axis.set_xticklabels([AGENT_LABELS[agent] for agent in agents])
        axis.set_title(MATCHUP_LABELS[key])
        axis.set_ylabel("Total raw reward / game")
    fig.suptitle("Total undiscounted raw reward (mean ± SD)")
    return save_figure(fig, output_dir, "15_raw_total_reward.png", dpi)


def plot_event_reward_by_turn(data, output_dir, dpi):
    """Plot immediate raw event reward as a function of absolute engine turn."""
    rows = ordered_matchups(data)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (key, matchup) in zip(axes.flat, rows):
        turns = matchup["raw_reward_summary"]["by_turn"]
        x = np.asarray([row["turn"] for row in turns])
        acting = np.asarray([row["mean_acting_player_event_reward"] for row in turns])
        opponent = np.asarray([row["mean_opponent_event_reward"] for row in turns])
        axis.plot(x, acting, label="Acting player")
        axis.plot(x, opponent, label="Opponent")
        axis.axhline(0.0, linewidth=0.8, color="black")
        axis.set_title(MATCHUP_LABELS[key])
        axis.set_xlabel("Engine turn")
        axis.set_ylabel("Mean immediate raw reward")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Immediate draw/pass reward along games", y=1.03)
    return save_figure(fig, output_dir, "16_event_reward_by_turn.png", dpi)


def plot_cumulative_raw_reward_by_turn(data, output_dir, dpi):
    """Plot mean cumulative raw reward, including terminal terms at game end."""
    rows = ordered_matchups(data)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (key, matchup) in zip(axes.flat, rows):
        turns = matchup["raw_reward_summary"]["by_turn"]
        x = np.asarray([row["turn"] for row in turns])
        for agent in _reward_agents(matchup):
            y = np.asarray([
                row["mean_cumulative_raw_reward_by_agent"].get(agent, np.nan)
                for row in turns
            ])
            axis.plot(x, y, label=AGENT_LABELS[agent], color=AGENT_COLORS[agent])
        axis.axhline(0.0, linewidth=0.8, color="black")
        axis.set_title(MATCHUP_LABELS[key])
        axis.set_xlabel("Engine turn")
        axis.set_ylabel("Mean cumulative raw reward")
    handles = [
        plt.Line2D([0], [0], color=AGENT_COLORS[a], label=AGENT_LABELS[a])
        for a in AGENT_LABELS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Cumulative raw reward along games", y=1.03)
    return save_figure(fig, output_dir, "17_cumulative_raw_reward_by_turn.png", dpi)


def plot_draw_pass_rates(data, output_dir, dpi):
    """Aggregate draw/pass event rates for each agent across all six matchups."""
    totals = {
        agent: {"turns": 0, "draws": 0, "passes": 0}
        for agent in AGENT_LABELS
    }
    for _key, matchup in ordered_matchups(data):
        event = matchup["raw_reward_summary"]["event_actions_by_agent"]
        for agent, row in event.items():
            totals[agent]["turns"] += int(row["turns"])
            totals[agent]["draws"] += int(row["draws"])
            totals[agent]["passes"] += int(row["passes"])
    agents = list(AGENT_LABELS)
    x = np.arange(len(agents), dtype=float)
    width = 0.25
    draw = [100.0 * totals[a]["draws"] / totals[a]["turns"] for a in agents]
    passed = [100.0 * totals[a]["passes"] / totals[a]["turns"] for a in agents]
    nonzero = [draw[i] + passed[i] for i in range(len(agents))]
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - width, draw, width=width, label="Draw")
    axis.bar(x, passed, width=width, label="Pass")
    axis.bar(x + width, nonzero, width=width, label="Draw or pass")
    axis.set_xticks(x)
    axis.set_xticklabels([AGENT_LABELS[a] for a in agents])
    axis.set_ylabel("Percent of acting turns")
    axis.set_title("Raw-reward event frequency by agent")
    axis.legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, output_dir, "18_draw_pass_rates.png", dpi)


def plot_winner_loser_raw_reward(data, output_dir, dpi):
    """Compare complete raw reward of winners and losers in each matchup."""
    rows = ordered_matchups(data)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (key, matchup) in zip(axes.flat, rows):
        outcomes = matchup["raw_reward_summary"]["per_game_by_outcome"]
        labels = ["Winner", "Loser"]
        stats = [outcomes["winner"]["total"], outcomes["loser"]["total"]]
        means = np.asarray([row["mean"] for row in stats])
        lower = means - np.asarray([row["p25"] for row in stats])
        upper = np.asarray([row["p75"] for row in stats]) - means
        axis.bar(np.arange(2), means, yerr=np.vstack([lower, upper]), capsize=3)
        axis.axhline(0.0, linewidth=0.8, color="black")
        axis.set_xticks(np.arange(2))
        axis.set_xticklabels(labels)
        axis.set_title(MATCHUP_LABELS[key])
        axis.set_ylabel("Total raw reward / game")
    fig.suptitle("Winner vs loser total raw reward (mean, IQR error bars)")
    return save_figure(fig, output_dir, "19_winner_loser_raw_reward.png", dpi)


def plot_relative_progress_reward(data, output_dir, dpi):
    """Normalize game length and compare cumulative reward trajectories."""
    rows = ordered_matchups(data)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (key, matchup) in zip(axes.flat, rows):
        bins = matchup["raw_reward_summary"]["by_relative_progress"]
        x = np.asarray([
            0.5 * (row["fraction_start"] + row["fraction_end"])
            for row in bins
        ])
        for agent in _reward_agents(matchup):
            y = np.asarray([
                row["mean_cumulative_raw_reward_by_agent"].get(agent, np.nan)
                for row in bins
            ])
            axis.plot(x, y, marker="o", markersize=3,
                      label=AGENT_LABELS[agent], color=AGENT_COLORS[agent])
        axis.axhline(0.0, linewidth=0.8, color="black")
        axis.set_xlim(0.0, 1.0)
        axis.set_title(MATCHUP_LABELS[key])
        axis.set_xlabel("Fraction of game completed")
        axis.set_ylabel("Mean cumulative raw reward")
    handles = [
        plt.Line2D([0], [0], color=AGENT_COLORS[a], label=AGENT_LABELS[a])
        for a in AGENT_LABELS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Cumulative raw reward by normalized game progress", y=1.03)
    return save_figure(fig, output_dir, "20_relative_progress_raw_reward.png", dpi)



def generate_plots(data, output_dir, dpi):
    """Generate all comparative state and timing figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return [
        plot_mean_sizes(data, output_dir, dpi),
        plot_duration_and_survival(data, output_dir, dpi),
        plot_opening_tiles(data, output_dir, dpi),
        plot_size_distributions(data, output_dir, dpi),
        plot_tile_presence(data, output_dir, dpi),
        plot_pip_composition(data, output_dir, dpi),
        plot_game_timing(data, output_dir, dpi),
        plot_timing_components(data, output_dir, dpi),
        plot_turn_time_by_matchup(data, output_dir, dpi),
        plot_agent_decision_distributions(data, output_dir, dpi),
        plot_agent_time_by_turn(data, output_dir, dpi),
        plot_decision_classes(data, output_dir, dpi),
        plot_turn_timing_heatmaps(data, output_dir, dpi),
        plot_raw_reward_components(data, output_dir, dpi),
        plot_raw_total_reward(data, output_dir, dpi),
        plot_event_reward_by_turn(data, output_dir, dpi),
        plot_cumulative_raw_reward_by_turn(data, output_dir, dpi),
        plot_draw_pass_rates(data, output_dir, dpi),
        plot_winner_loser_raw_reward(data, output_dir, dpi),
        plot_relative_progress_reward(data, output_dir, dpi),
    ]


def parse_args():
    """Parse report, output directory, and raster-resolution controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main():
    """Load one report and publish every figure."""
    args = parse_args()
    data = load_report(args.input)
    paths = generate_plots(data, args.output_dir, args.dpi)
    print(f"Generated {len(paths)} figures from {args.input}")


if __name__ == "__main__":
    main()
