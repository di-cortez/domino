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


WIN_RATE_COLOR_BANDS = (
    (None, 30, "<30%", "#8f1d1d"),
    (30, 35, "30–35%", "#b73b3b"),
    (35, 40, "35–40%", "#d96868"),
    (40, 45, "40–45%", "#ee9b9b"),
    (45, 50, "45–50%", "#f8d6d6"),
    (50, 55, "50–55%", "#f7fbff"),
    (55, 60, "55–60%", "#d8eaf7"),
    (60, 65, "60–65%", "#a9cfea"),
    (65, 70, "65–70%", "#6da7d5"),
    (70, None, "≥70%", "#2468a2"),
)


def worst_case_margin_of_error(sample_size):
    """Return the 95% worst-case proportion margin ``sqrt(0.9604 / n)``."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    return math.sqrt(0.9604 / int(sample_size))


def win_rate_color_band(win_rate_pct):
    """Return the label, fill, and readable text color for a win percentage."""
    rate = float(win_rate_pct)
    if not 0 <= rate <= 100:
        raise ValueError("win_rate_pct must be between 0 and 100")
    for lower, upper, label, fill in WIN_RATE_COLOR_BANDS:
        if (lower is None or rate >= lower) and (upper is None or rate < upper):
            text_color = "#ffffff" if rate < 35 or rate >= 70 else INK
            return label, fill, text_color
    raise RuntimeError(f"no color band configured for win rate {rate}")


def _format_diagnostic_duration(seconds):
    """Format elapsed seconds compactly for the aggregate report header."""
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def _network_header_text(label, metadata):
    """Format one network architecture/checkpoint description."""
    architecture = "→".join(str(value) for value in metadata["architecture"])
    parameters = f"{int(metadata['total_parameters']):,} params"
    if metadata.get("checkpoint_name"):
        source = metadata["checkpoint_name"]
        if metadata.get("checkpoint_sha256"):
            source += f" (sha256 {metadata['checkpoint_sha256'][:12]}...)"
    else:
        source = metadata.get("initialization", "no checkpoint")
    value_head = "value head on" if metadata.get("value_head") else "value head off"
    return f"{label}: {architecture}; {parameters}; {value_head}; {source}"


def _value_head_header_text(label, summary):
    """Format aggregate value predictions for an evaluated checkpoint."""
    values = summary.get("value_head_predictions")
    if not values or not values.get("finite_count"):
        return None
    text = (
        f"{label} V(s) over {int(values['sample_count']):,} real decisions; "
        "mean/std/min/max "
        f"{values['mean']:+.3f}/{values['std']:.3f}/"
        f"{values['min']:+.3f}/{values['max']:+.3f}"
    )
    if values.get("nonfinite_count"):
        text += f"; non-finite {int(values['nonfinite_count']):,}"
    return text


def diagnostic_table_header_lines(summaries, agents, report_metadata=None):
    """Build descriptive lines shown above the one-row diagnostic result."""
    metadata = report_metadata or {}
    game_count = int(
        metadata.get(
            "game_count_per_matchup",
            summaries[0]["game_count"] if summaries else 0,
        )
    )
    matchup_count = int(metadata.get("evaluated_matchups", len(summaries)))
    total_games = game_count * matchup_count
    margin = (
        100 * worst_case_margin_of_error(game_count)
        if game_count
        else None
    )
    margin_text = f"±{margin:.2g} percentage points" if margin is not None else "unknown"
    lines = [
        (
            f"Scope: {matchup_count} agents vs random | {game_count:,} games per "
            f"matchup | {total_games:,} games total"
        ),
        (
            "95% worst-case margin of error for each win rate: "
            f"{margin_text} (normal approximation, p=50%)"
        ),
    ]

    if "duration_s" in metadata:
        duration = _format_diagnostic_duration(metadata["duration_s"])
        seed = metadata.get("seed")
        seed_text = metadata.get("effective_seed") if seed is None else seed
        workers = metadata.get("selected_workers_by_matchup", {})
        worker_text = ", ".join(
            f"{key.removesuffix('_vs_random')}={value}"
            for key, value in workers.items()
        )
        line = f"Evaluation elapsed: {duration} | seed: {seed_text}"
        if worker_text:
            line += f" | workers: {worker_text}"
        lines.append(line)

    networks = metadata.get("network_metadata", {})
    if "rl" in networks:
        lines.append(_network_header_text("RL", networks["rl"]))
        rl_summary = next(
            (
                summary
                for summary in summaries
                if summary.get("agent") == "rl"
                and summary.get("opponent") == "random"
            ),
            {},
        )
        value_line = _value_head_header_text("RL value head", rl_summary)
        if value_line is not None:
            lines.append(value_line)
    if "neural" in networks:
        lines.append(_network_header_text("Neural", networks["neural"]))
    return lines


def plot_all_pairs_table(summaries, agents, path, report_metadata=None):
    """Render one row of agent win rates against random as PNG or PDF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    summary_by_agent = {
        summary["agent"]: summary
        for summary in summaries
        if summary.get("opponent") == "random"
    }
    cell_text = []
    cell_rates = []
    for agent in agents:
        summary = summary_by_agent.get(agent)
        if summary is None:
            cell_text.append("")
            cell_rates.append(None)
            continue
        win_rate = 100 * summary["rates"]["win"]
        cell_text.append(f"{win_rate:.1f}%")
        cell_rates.append(win_rate)

    width = max(12.0, 2.15 * (len(agents) + 1))
    fig, ax = plt.subplots(figsize=(width, 6.6), facecolor=SURFACE, dpi=160)
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    ax.set_title(
        "Agent win rates against the random baseline",
        color=INK,
        fontsize=15,
        loc="left",
        pad=18,
    )

    header_lines = diagnostic_table_header_lines(
        summaries,
        agents,
        report_metadata=report_metadata,
    )
    for index, line in enumerate(header_lines):
        ax.text(
            0.0,
            0.94 - 0.065 * index,
            line,
            transform=ax.transAxes,
            color=INK if index < 2 else SECONDARY_INK,
            fontsize=9.2,
            va="top",
        )

    display_names = [agent.replace("_", " ") for agent in agents]
    table = ax.table(
        cellText=[cell_text],
        rowLabels=["Win rate"],
        colLabels=display_names,
        bbox=[0, 0.32, 1, 0.20],
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(AXIS)
        cell.set_linewidth(0.8)
        if row == 0 or col == -1:
            cell.set_facecolor("#efeee8")
            cell.set_text_props(color=INK, weight="bold")
            continue
        rate = cell_rates[col]
        if rate is None:
            cell.set_facecolor("#f4f3ee")
            cell.set_text_props(color="#c9c6bd")
            continue
        _label, fill, text_color = win_rate_color_band(rate)
        cell.set_facecolor(fill)
        cell.set_text_props(color=text_color, weight="bold")

    ax.text(
        0.0,
        0.245,
        (
            "Each evaluated agent alternates between player 0 and player 1. "
            "Percentages are wins by the column agent against random."
        ),
        transform=ax.transAxes,
        color=SECONDARY_INK,
        fontsize=9,
        va="top",
    )
    legend_handles = [
        Patch(facecolor=fill, edgecolor=AXIS, label=label)
        for _lower, _upper, label, fill in WIN_RATE_COLOR_BANDS
    ]
    ax.legend(
        handles=legend_handles,
        title="Win-rate color bands",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=False,
        fontsize=8,
        title_fontsize=9,
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


SWEEP_TABLE_BASE_COLUMNS = (
    ("critic", "Critic"),
    ("learning_rate", "LR"),
    ("gamma", "Gamma"),
    ("value_coef", "ValueCoef"),
)


def plot_sweep_comparison_table(
    rows,
    path,
    games_per_iteration_values=(40, 80, 160),
):
    """Render grouped runs with one win-rate column per games/iteration value.

    Mirrors ``plot_all_pairs_table``'s look (same palette, same
    ``ax.table``-based rendering). Models that differ only in
    games-per-iteration share one row, reducing the visual table height while
    preserving their separate percentages.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = list(SWEEP_TABLE_BASE_COLUMNS) + [
        (f"win_rate_pct_gpi_{value}", str(value))
        for value in games_per_iteration_values
    ]
    field_names = [field for field, _ in columns]
    headers = [label for _, label in columns]
    win_rate_columns = {
        field_names.index(f"win_rate_pct_gpi_{value}")
        for value in games_per_iteration_values
    }

    def format_cell(row, field):
        value = row.get(field, "")
        if field.startswith("win_rate_pct_gpi_"):
            return "" if value == "" or value is None else f"{float(value):.1f}%"
        return str(value)

    cell_text = [
        [format_cell(row, field) for field in field_names]
        for row in rows
    ]

    # Proportional column widths from the longest cell (header or value) in
    # each column, so long run names don't get clipped by equal-width cells.
    col_chars = [
        max(
            [len(headers[i])] + [len(text_row[i]) for text_row in cell_text]
        )
        for i in range(len(headers))
    ]
    total_chars = sum(col_chars)
    col_widths = [chars / total_chars for chars in col_chars]

    width = max(13.0, 0.16 * total_chars)
    height = max(2.6, 0.42 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE, dpi=160)
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    ax.set_title(
        "RL sweep vs random: win rate (%) by games per iteration",
        color=INK,
        fontsize=14,
        loc="left",
        pad=14,
    )
    if not rows:
        ax.text(
            0.5,
            0.44,
            "No sweep runs found.",
            color=SECONDARY_INK,
            fontsize=11,
            ha="center",
            va="center",
        )
        _save_figure(fig, path)
        return

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        colWidths=col_widths,
        cellLoc="center",
        bbox=[0, 0, 1, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(AXIS)
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#efeee8")
            cell.set_text_props(color=INK, weight="bold")
            continue
        if col in win_rate_columns:
            try:
                rate = float(cell_text[row - 1][col].rstrip("%"))
            except ValueError:
                rate = None
            if rate is not None and rate >= 60:
                cell.set_facecolor("#e6f1fb")
            elif rate is not None and rate <= 40:
                cell.set_facecolor("#f9ead9")
            else:
                cell.set_facecolor(SURFACE)
            cell.set_text_props(color=INK, weight="bold")
        else:
            cell.set_facecolor(SURFACE)
            cell.set_text_props(color=SECONDARY_INK)

    _save_figure(fig, path)
