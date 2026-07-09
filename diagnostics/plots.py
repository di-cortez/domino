"""
Metrics and plot generation for diagnostics.

This module is intentionally independent from the domino engine. It receives a
list of game records and produces aggregate statistics plus PNG files.
"""

import math

import numpy as np


def wilson_interval(successes, total, z=1.96):
    """Return a Wilson score confidence interval for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize(games, agent_name, opponent_name, seed):
    """Aggregate per-game rows into rates, confidence intervals, and means."""
    game_count = len(games)
    counts = {result: sum(1 for game in games if game["result"] == result)
              for result in ("win", "draw", "loss")}

    summary = {
        "agent": agent_name,
        "opponent": opponent_name,
        "game_count": game_count,
        "seed": seed,
        "counts": counts,
        "rates": {result: count / game_count for result, count in counts.items()},
        "win_ci95": wilson_interval(counts["win"], game_count),
        "draw_ci95": wilson_interval(counts["draw"], game_count),
        "mean_turns": float(np.mean([game["turns"] for game in games])),
        "std_turns": float(np.std([game["turns"] for game in games])),
        "mean_agent_remaining_pips": float(np.mean([game["agent_remaining_pips"] for game in games])),
        "mean_opponent_remaining_pips": float(
            np.mean([game["opponent_remaining_pips"] for game in games])
        ),
        "by_position": {},
    }

    for position in (0, 1):
        group = [game for game in games if game["agent_position"] == position]
        wins = sum(1 for game in group if game["result"] == "win")
        summary["by_position"][str(position)] = {
            "games": len(group),
            "wins": wins,
            "win_rate": wins / len(group) if group else 0.0,
            "ci95": wilson_interval(wins, len(group)),
        }
    return summary


SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

COLOR = {"win": "#2a78d6", "draw": "#1baf7a", "loss": "#eda100"}
LABEL = {"win": "Win", "draw": "Draw", "loss": "Loss"}


def _prepare_axis(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _new_figure(width=8.0, height=4.5):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE, dpi=150)
    return fig, ax


def _save_figure(fig, path):
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_cumulative_rates(games, path, subtitle):
    """Plot cumulative win/draw/loss rates over the evaluation run."""
    fig, ax = _new_figure()
    _prepare_axis(ax, f"Cumulative result rates - {subtitle}")

    n = len(games)
    x = np.arange(1, n + 1)
    for result in ("win", "draw", "loss"):
        cumulative = np.cumsum([game["result"] == result for game in games]) / x
        ax.plot(x, 100 * cumulative, color=COLOR[result], linewidth=2, label=LABEL[result])
        ax.annotate(
            f"{LABEL[result]} {100 * cumulative[-1]:.1f}%",
            xy=(n, 100 * cumulative[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=SECONDARY_INK,
            fontsize=9,
        )

    ax.set_xlim(1, n * 1.18)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Games")
    ax.set_ylabel("Cumulative rate (%)")
    ax.legend(frameon=False, labelcolor=SECONDARY_INK, fontsize=9, loc="upper right")
    _save_figure(fig, path)


def plot_distribution(summary, path, subtitle):
    """Plot final result counts as horizontal bars."""
    fig, ax = _new_figure(height=3.2)
    _prepare_axis(ax, f"Result distribution - {subtitle}")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)

    results = ["loss", "draw", "win"]
    values = [summary["counts"][result] for result in results]
    colors = [COLOR[result] for result in results]
    ax.barh([LABEL[result] for result in results], values, color=colors, height=0.55)

    game_count = summary["game_count"]
    for i, (result, value) in enumerate(zip(results, values)):
        ax.annotate(
            f"{value} ({100 * value / game_count:.1f}%)",
            xy=(value, i),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=SECONDARY_INK,
            fontsize=9,
        )

    ax.set_xlim(0, max(values) * 1.22 if max(values) else 1)
    ax.set_xlabel("Games")
    ax.tick_params(axis="y", labelcolor=INK)
    _save_figure(fig, path)


def plot_by_position(summary, path, subtitle):
    """Plot win rate by starting position with 95% confidence intervals."""
    fig, ax = _new_figure(width=7.5, height=4.2)
    _prepare_axis(ax, f"Win rate by starting position - {subtitle}")

    positions = ["0", "1"]
    rates = [100 * summary["by_position"][position]["win_rate"] for position in positions]
    lower_errors = [
        100 * (
            summary["by_position"][position]["win_rate"]
            - summary["by_position"][position]["ci95"][0]
        )
        for position in positions
    ]
    upper_errors = [
        100 * (
            summary["by_position"][position]["ci95"][1]
            - summary["by_position"][position]["win_rate"]
        )
        for position in positions
    ]

    ax.bar(
        ["Player 0", "Player 1"],
        rates,
        color=COLOR["win"],
        width=0.5,
        yerr=[lower_errors, upper_errors],
        ecolor=SECONDARY_INK,
        capsize=4,
    )

    for i, position in enumerate(positions):
        info = summary["by_position"][position]
        ax.annotate(
            f"{100 * info['win_rate']:.1f}%  (n={info['games']})",
            xy=(i, rates[i] + upper_errors[i]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color=SECONDARY_INK,
            fontsize=9,
        )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Win rate (%)")
    ax.tick_params(axis="x", labelcolor=INK)
    _save_figure(fig, path)


def plot_game_lengths(games, path, subtitle):
    """Plot a histogram of game lengths measured in turns."""
    fig, ax = _new_figure(height=3.8)
    _prepare_axis(ax, f"Game length distribution - {subtitle}")

    turns = [game["turns"] for game in games]
    ax.hist(
        turns,
        bins=min(30, max(5, len(set(turns)))),
        color=COLOR["win"],
        edgecolor=SURFACE,
        linewidth=1,
    )
    mean_turns = float(np.mean(turns))
    ax.axvline(mean_turns, color=SECONDARY_INK, linewidth=1, linestyle="--")
    ax.annotate(
        f"mean {mean_turns:.1f}",
        xy=(mean_turns, ax.get_ylim()[1]),
        xytext=(6, -12),
        textcoords="offset points",
        color=SECONDARY_INK,
        fontsize=9,
    )

    ax.set_xlabel("Turns per game")
    ax.set_ylabel("Games")
    _save_figure(fig, path)


def generate_plots(games, summary, folder):
    """Generate all diagnostic PNGs in the target folder."""
    import matplotlib

    matplotlib.use("Agg")

    subtitle = f"{summary['agent']} vs {summary['opponent']} ({summary['game_count']} games)"
    plot_cumulative_rates(games, folder / "cumulative_rates.png", subtitle)
    plot_distribution(summary, folder / "result_distribution.png", subtitle)
    plot_by_position(summary, folder / "wins_by_position.png", subtitle)
    plot_game_lengths(games, folder / "game_lengths.png", subtitle)



def plot_all_pairs_table(summaries, agents, path):
    """Render the ordered all-pairs result matrix as a PNG table.

    Each cell is read as "row agent evaluated against column opponent" and shows
    win/draw/loss counts plus the row agent's win rate.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_by_pair = {(summary["agent"], summary["opponent"]): summary for summary in summaries}
    cell_text = []
    for agent in agents:
        row = []
        for opponent in agents:
            summary = summary_by_pair[(agent, opponent)]
            counts = summary["counts"]
            row.append(
                f"W/D/L: {counts['win']}/{counts['draw']}/{counts['loss']}\n"
                f"Win rate: {100 * summary['rates']['win']:.1f}%"
            )
        cell_text.append(row)

    width = max(9.0, 2.25 * (len(agents) + 1))
    height = max(4.0, 0.95 * (len(agents) + 2))
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE, dpi=160)
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    ax.set_title(
        "All-pairs diagnostics matrix",
        color=INK,
        fontsize=14,
        loc="left",
        pad=18,
    )

    table = ax.table(
        cellText=cell_text,
        rowLabels=agents,
        colLabels=agents,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(AXIS)
        cell.set_linewidth(0.8)
        if row == 0 or col == -1:
            cell.set_facecolor("#efeee8")
            cell.set_text_props(color=INK, weight="bold")
        else:
            cell.set_facecolor(SURFACE)
            cell.set_text_props(color=SECONDARY_INK)

    ax.text(
        0.0,
        -0.08,
        "Rows are evaluated agents; columns are opponents. W/D/L counts are from the row agent's perspective.",
        transform=ax.transAxes,
        color=SECONDARY_INK,
        fontsize=9,
        va="top",
    )
    _save_figure(fig, path)
