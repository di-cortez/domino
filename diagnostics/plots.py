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


def _choice_plot_rows(choice_info):
    """Return labels and counts for a choice-opportunity histogram."""
    histogram = choice_info.get("choice_histogram", {})
    rows = [
        ("Draw", choice_info.get("forced_draws", 0)),
        ("Pass", choice_info.get("forced_passes", 0)),
    ]

    for option_count, count in sorted(histogram.items(), key=lambda item: int(item[0])):
        label = f"{option_count} tile" if int(option_count) == 1 else f"{option_count} tiles"
        rows.append((label, count))

    return [(label, count) for label, count in rows if count > 0]


def plot_choice_opportunities(summary, path, subtitle):
    """Plot draw/pass/option-count frequencies from choice-opportunity stats."""
    choice_info = summary.get("choice_opportunities", {})
    rows = _choice_plot_rows(choice_info)
    if not rows:
        return

    fig, ax = _new_figure(width=8.0, height=4.2)
    _prepare_axis(ax, f"Choice opportunities - {subtitle}")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)

    labels = [label for label, _count in rows]
    values = [count for _label, count in rows]
    colors = []
    for label in labels:
        if label == "Draw":
            colors.append("#a9a6a0")
        elif label == "Pass":
            colors.append("#c6c2b8")
        elif label.startswith("1 "):
            colors.append("#eda100")
        else:
            colors.append(COLOR["win"])

    ax.barh(labels, values, color=colors, height=0.62)

    total = sum(values)
    for index, value in enumerate(values):
        ax.annotate(
            f"{value} ({100 * value / total:.1f}%)",
            xy=(value, index),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=SECONDARY_INK,
            fontsize=9,
        )

    ax.set_xlim(0, max(values) * 1.24 if max(values) else 1)
    ax.set_xlabel("Turns")
    ax.tick_params(axis="y", labelcolor=INK)
    _save_figure(fig, path)


def plot_first_stock_draw_turns(summary, path, subtitle):
    """Plot the turn where the first stock draw happened in each game."""
    first_draw = summary.get("first_stock_draw", summary)
    histogram = first_draw.get("turn_histogram", {})

    fig, ax = _new_figure(width=8.0, height=4.0)
    _prepare_axis(ax, f"First stock draw turn - {subtitle}")

    if not histogram:
        ax.grid(visible=False)
        ax.text(
            0.5,
            0.5,
            "No stock draws recorded",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=SECONDARY_INK,
            fontsize=11,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        _save_figure(fig, path)
        return

    turns = [int(turn) for turn in sorted(histogram, key=int)]
    counts = [histogram[str(turn)] for turn in turns]
    ax.bar(turns, counts, color=COLOR["draw"], edgecolor=SURFACE, linewidth=1, width=0.82)

    mean_turn = first_draw.get("mean_turn")
    if mean_turn is not None:
        ax.axvline(mean_turn, color=SECONDARY_INK, linewidth=1, linestyle="--")
        ax.annotate(
            f"mean {mean_turn:.1f}",
            xy=(mean_turn, ax.get_ylim()[1]),
            xytext=(6, -12),
            textcoords="offset points",
            color=SECONDARY_INK,
            fontsize=9,
        )

    ax.set_xlim(min(turns) - 0.8, max(turns) + 0.8)
    ax.set_xlabel("First stock draw turn")
    ax.set_ylabel("Games")
    _save_figure(fig, path)


def plot_first_stock_draw_final_state_counts(summary, path, subtitle):
    """Plot raw hidden-hand upper bounds at the first stock draw."""
    expansion_info = summary.get("first_stock_draw_expansion", summary)
    histogram = expansion_info.get("final_state_count_histogram", {})

    fig, ax = _new_figure(width=8.0, height=4.0)
    _prepare_axis(ax, f"First draw raw hand upper bound - {subtitle}")

    if not histogram:
        ax.grid(visible=False)
        ax.text(
            0.5,
            0.5,
            "No first-draw expansion count recorded",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=SECONDARY_INK,
            fontsize=11,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        _save_figure(fig, path)
        return

    values = np.array([int(value) for value in sorted(histogram, key=int)])
    weights = np.array([histogram[str(value)] for value in values])

    if len(values) == 1:
        ax.bar(values, weights, color=COLOR["draw"], edgecolor=SURFACE, linewidth=1, width=8)
        ax.set_xlim(values[0] - 12, values[0] + 12)
    else:
        bins = min(30, max(5, len(values)))
        ax.hist(
            values,
            bins=bins,
            weights=weights,
            color=COLOR["draw"],
            edgecolor=SURFACE,
            linewidth=1,
        )

    mean_count = expansion_info.get("mean_final_state_count")
    if mean_count is not None:
        ax.axvline(mean_count, color=SECONDARY_INK, linewidth=1, linestyle="--")
        ax.annotate(
            f"mean {mean_count:.1f}",
            xy=(mean_count, ax.get_ylim()[1]),
            xytext=(6, -12),
            textcoords="offset points",
            color=SECONDARY_INK,
            fontsize=9,
        )

    ax.set_xlabel("comb(|U|, h) immediately after first stock draw")
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
    plot_choice_opportunities(summary, folder / "choice_opportunities.png", subtitle)
    plot_first_stock_draw_turns(summary, folder / "first_stock_draw_turns.png", subtitle)
    plot_first_stock_draw_final_state_counts(
        summary,
        folder / "first_stock_draw_final_state_counts.png",
        subtitle,
    )


def plot_all_pairs_table(summaries, agents, path):
    """Render a triangular all-pairs win-rate matrix as a clean PNG table."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary_by_pair = {(summary["agent"], summary["opponent"]): summary for summary in summaries}
    cell_text = []
    cell_rates = []
    for agent in agents:
        row = []
        rate_row = []
        for opponent in agents:
            summary = summary_by_pair.get((agent, opponent))
            if summary is None:
                row.append("")
                rate_row.append(None)
                continue
            win_rate = 100 * summary["rates"]["win"]
            row.append(f"{win_rate:.1f}")
            rate_row.append(win_rate)
        cell_text.append(row)
        cell_rates.append(rate_row)

    width = max(9.0, 2.15 * (len(agents) + 1))
    height = max(3.2, 0.58 * (len(agents) + 2))
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE, dpi=160)
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    ax.set_title(
        "All-pairs win-rate matrix (%)",
        color=INK,
        fontsize=14,
        loc="left",
        pad=18,
    )

    table = ax.table(
        cellText=cell_text,
        rowLabels=agents,
        colLabels=agents,
        bbox=[0, 0.24, 1, 0.58],
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(AXIS)
        cell.set_linewidth(0.8)
        if row == 0 or col == -1:
            cell.set_facecolor("#efeee8")
            cell.set_text_props(color=INK, weight="bold")
        elif not cell.get_text().get_text():
            cell.set_facecolor("#f4f3ee")
            cell.set_text_props(color="#c9c6bd")
        else:
            rate = cell_rates[row - 1][col]
            if rate >= 60:
                cell.set_facecolor("#e6f1fb")
            elif rate <= 40:
                cell.set_facecolor("#f9ead9")
            else:
                cell.set_facecolor(SURFACE)
            cell.set_text_props(color=INK, weight="bold")

    ax.text(
        0.0,
        0.06,
        "Rows are evaluated agents; columns are opponents. Blank cells are skipped reverse matchups.",
        transform=ax.transAxes,
        color=SECONDARY_INK,
        fontsize=9,
        va="top",
    )
    _save_figure(fig, path)


def plot_aggregate_choice_opportunities(choice_info, path):
    """Plot the aggregate choice-opportunity histogram for all evaluated pairs."""
    summary = {
        "choice_opportunities": choice_info,
        "agent": "all",
        "opponent": "pairs",
        "game_count": choice_info.get("matchups", 0),
    }
    plot_choice_opportunities(summary, path, "all evaluated pairs")


def plot_aggregate_first_stock_draws(first_draw_info, path):
    """Plot the aggregate first-stock-draw histogram for all evaluated pairs."""
    plot_first_stock_draw_turns(first_draw_info, path, "all evaluated pairs")


def plot_aggregate_first_stock_draw_final_state_counts(expansion_info, path):
    """Plot aggregate first-stock-draw raw hand upper bounds."""
    plot_first_stock_draw_final_state_counts(expansion_info, path, "all evaluated pairs")
