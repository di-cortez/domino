#!/usr/bin/env python3
"""Compare the pre-redesign and current RL reward architectures on double-six.

Two things are measured and kept apart. The first is the *outcome*: the
periodic 100,000-game diagnostic against `random` for every canonical
`forever` run on `double-six`, split by which reward architecture the run was
trained under. The second is the *cause candidate*: both reward functions are
recomputed offline on one identical corpus of 422,055 real decisions, so the
two objectives are compared on the same trajectories rather than through the
runs that optimized them.

Everything is read-only: run directories, the derived lookup corpus, and the
git history are only read. All outputs land in this directory.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from training.rl.reward_model import blocked_reward_magnitude  # noqa: E402

RUN_ROOT = REPO / "models" / "rl"
DERIVED_CORPUS = (
    REPO / "analysis" / "reward_lookup_table" / "derived"
    / "double-six_reward_lookup_samples.json.gz"
)

# The fixed panel every periodic diagnostic plays against `random`.
DIAGNOSTIC_GAMES = 100_000

# Shared temporal parameters. Both architectures ship the same defaults, so
# they cancel out of the comparison: gamma_f, gamma_i and reward_eta are held
# fixed at the values every run below actually used.
GAMMA_F = 0.95
GAMMA_I = 0.90
REWARD_ETA = 0.5

# The superseded architecture, exactly as `training/rl/rollout.py` defined it
# before commit 895d512 ("reward shaping changes and bugs correction").
LEGACY_TERMINAL_WIN = 1.0
LEGACY_FINAL_PIP_PENALTY = LEGACY_TERMINAL_WIN / 20.0
LEGACY_EVENT_VALUE = {
    "opponent_draw": LEGACY_TERMINAL_WIN / 5.0,
    "neural_draw": -LEGACY_TERMINAL_WIN / 5.0,
    "opponent_pass": LEGACY_TERMINAL_WIN / 10.0,
    "neural_pass": -LEGACY_TERMINAL_WIN / 10.0,
}

# The current architecture under its neutral default weights, where every
# normalized scale is 1.0 and the unit event reward is +/-1.
MAX_PIP_DOUBLE_SIX = 6

LEGACY = "anterior"
CURRENT = "atual"

# Every canonical double-six `forever` run, with the architecture it trained
# under. `era` is decided by whether the run's locked configuration carries the
# redesign's terminal weights, not by its date.
RUNS = (
    ("bucket_heuristic", "domino_rl_forever_seed42_runbucket_heuristic"),
    ("bucket_heuristic_recent", "domino_rl_forever_seed42_runbucket_heuristic_recent"),
    ("baseline_zero", "domino_rl_forever_seed42_runbaseline_zero"),
    ("bucket_all", "domino_rl_forever_seed42_runbucket_all"),
    ("d6_maxwr_lr032", "domino_rl_forever_seed42_rund6_maxwr_lr032"),
    ("d6_maxwr_lr016", "domino_rl_forever_seed42_rund6_maxwr_lr016"),
    ("default_lr032", "domino_rl_forever_seed42_rundefault_lr032"),
    ("default_lr016_lookup", "domino_rl_forever_seed42_rundefault_lr016_lookup"),
)

# The two pairs that hold the opponent-bucket set fixed across the two
# architectures. Each pair still differs in learning rate; the report says so.
PAIRS = (
    ("bucket_heuristic", "d6_maxwr_lr032"),
    ("bucket_heuristic_recent", "default_lr032"),
)

# A run needs enough of a curve for "best" to mean anything.
MIN_GAMES_FOR_COMPARISON = 5_000_000

COLORS = {LEGACY: "#08519c", CURRENT: "#99000d"}
RUN_COLORS = {
    "bucket_heuristic": "#08519c",
    "bucket_heuristic_recent": "#3182bd",
    "baseline_zero": "#6baed6",
    "bucket_all": "#9ecae1",
    "d6_maxwr_lr032": "#99000d",
    "d6_maxwr_lr016": "#fdae61",
    "default_lr032": "#ef3b2c",
    "default_lr016_lookup": "#fb6a4a",
}


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def load_run(label: str, directory: str) -> dict[str, Any]:
    run_dir = RUN_ROOT / directory
    compact = run_dir / "run_compact_diagnostics"
    config = json.loads((compact / "run_config.json").read_text(encoding="utf-8"))
    locked = config.get("locked_arguments", {})
    era = CURRENT if "terminal_empty_hand_weight" in locked else LEGACY

    with open(compact / "rl_vs_random_progress.csv", encoding="utf-8") as stream:
        progress = list(csv.DictReader(stream))
    curve = {
        key: np.asarray([float(row[key]) for row in progress])
        for key in progress[0]
    }

    with open(run_dir / "training_metrics.jsonl", encoding="utf-8") as stream:
        columns = json.loads(stream.readline())["columns"]
        metrics = [dict(zip(columns, json.loads(line))) for line in stream]

    return {
        "label": label,
        "directory": directory,
        "era": era,
        "created_at": config.get("created_at", ""),
        "learning_rate": float(locked.get("learning_rate", float("nan"))),
        "baseline": (locked.get("baseline") or ["nenhum"])[0],
        "buckets": ",".join(locked.get("opponent_buckets") or []),
        "gamma_f": float(locked.get("gamma_f", GAMMA_F)),
        "gamma_i": float(locked.get("gamma_i", GAMMA_I)),
        "reward_eta": float(locked.get("reward_eta", REWARD_ETA)),
        "distance_mode": locked.get("reward_distance_mode", ""),
        "curve": curve,
        "metrics": metrics,
        "complete": float(curve["rl_games"][-1]) >= MIN_GAMES_FOR_COMPARISON,
    }


def load_corpus() -> dict[str, Any]:
    """Return the derived per-decision corpus both rewards are scored on."""
    with gzip.open(DERIVED_CORPUS, "rt", encoding="utf-8") as stream:
        return json.load(stream)


# ----------------------------------------------------------------------
# Reward recomputation
# ----------------------------------------------------------------------


def score_corpus(corpus: dict[str, Any]) -> dict[str, np.ndarray]:
    """Score every decision under both architectures.

    One pass, one array per component, so the terminal and local halves can be
    compared separately as well as combined. Nothing here reads a run: the
    corpus is a fixed set of games and the two reward functions are pure.
    """
    legacy_terminal: list[float] = []
    current_terminal: list[float] = []
    legacy_local: list[float] = []
    current_local: list[float] = []
    won: list[bool] = []
    is_blocked: list[bool] = []
    learner_pips: list[float] = []
    pip_margin: list[float] = []
    blocked_magnitude: list[float] = []
    turn_distance: list[int] = []

    for cell in corpus["cells"].values():
        for sample in cell["samples"]:
            terminal = sample["terminal"]
            distance = int(terminal["turn_distance"])
            learner_won = bool(terminal["learner_won"])
            blocked = terminal["win_reason"] != "empty_hand"
            # The learner's own final pips: it is the winner's total when the
            # learner won and the loser's total otherwise.
            pips = float(
                terminal["winner_final_pips"] if learner_won
                else terminal["loser_final_pips"]
            )
            sign = 1.0 if learner_won else -1.0

            legacy_utility = sign * LEGACY_TERMINAL_WIN - LEGACY_FINAL_PIP_PENALTY * pips
            current_utility = float(
                terminal["empty_hand_component"] + terminal["blocked_component"]
            )
            discount = GAMMA_F ** distance
            legacy_terminal.append(discount * legacy_utility)
            current_terminal.append(discount * current_utility)

            legacy_sum = 0.0
            current_sum = 0.0
            for event in sample["future_local_events"]:
                decay = GAMMA_I ** int(event["turn_distance"])
                legacy_sum += LEGACY_EVENT_VALUE[event["kind"]] * decay
                current_sum += float(event["unit_reward"]) * decay
            legacy_local.append(legacy_sum)
            current_local.append(current_sum)

            won.append(learner_won)
            is_blocked.append(blocked)
            learner_pips.append(pips)
            pip_margin.append(
                float(terminal["pip_margin"]) if terminal["pip_margin"] is not None
                else float("nan")
            )
            blocked_magnitude.append(
                float(terminal["blocked_magnitude"])
                if terminal["blocked_magnitude"] is not None else float("nan")
            )
            turn_distance.append(distance)

    scored = {
        "legacy_terminal": np.asarray(legacy_terminal),
        "current_terminal": np.asarray(current_terminal),
        "legacy_local": np.asarray(legacy_local),
        "current_local": np.asarray(current_local),
        "won": np.asarray(won),
        "blocked": np.asarray(is_blocked),
        "learner_pips": np.asarray(learner_pips),
        "pip_margin": np.asarray(pip_margin),
        "blocked_magnitude": np.asarray(blocked_magnitude),
        "turn_distance": np.asarray(turn_distance),
    }
    for era, prefix in ((LEGACY, "legacy"), (CURRENT, "current")):
        scored[f"{prefix}_return"] = (
            (1.0 - REWARD_ETA) * scored[f"{prefix}_terminal"]
            + REWARD_ETA * scored[f"{prefix}_local"]
        )
    return scored


def alignment(returns: np.ndarray, won: np.ndarray) -> dict[str, float]:
    """Return how well one reward's sign agrees with the game's result."""
    positive = returns > 0.0
    return {
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "abs_mean": float(np.abs(returns).mean()),
        "min": float(returns.min()),
        "max": float(returns.max()),
        "correlation_with_win": float(np.corrcoef(returns, won.astype(float))[0, 1]),
        "positive_in_losses_pct": float(positive[~won].mean() * 100.0),
        "negative_in_wins_pct": float((~positive[won]).mean() * 100.0),
        "sign_disagreement_pct": float((positive != won).mean() * 100.0),
    }


def ending_breakdown(scored: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Return the undiscounted terminal utility per ending class."""
    won = scored["won"]
    blocked = scored["blocked"]
    pips = scored["learner_pips"]
    magnitude = scored["blocked_magnitude"]
    sign = np.where(won, 1.0, -1.0)
    legacy = sign * LEGACY_TERMINAL_WIN - LEGACY_FINAL_PIP_PENALTY * pips
    current = np.where(blocked, sign * np.nan_to_num(magnitude), sign)

    rows = []
    for name, mask in (
        ("mão vazia — vitória", (~blocked) & won),
        ("mão vazia — derrota", (~blocked) & ~won),
        ("bloqueio — vitória", blocked & won),
        ("bloqueio — derrota", blocked & ~won),
    ):
        rows.append({
            "desfecho": name,
            "fracao_decisoes_pct": round(float(mask.mean() * 100.0), 3),
            "pips_finais_medios": round(float(pips[mask].mean()), 3),
            "u_anterior_medio": round(float(legacy[mask].mean()), 4),
            "u_anterior_min": round(float(legacy[mask].min()), 4),
            "u_anterior_max": round(float(legacy[mask].max()), 4),
            "u_atual_medio": round(float(current[mask].mean()), 4),
            "u_atual_min": round(float(current[mask].min()), 4),
            "u_atual_max": round(float(current[mask].max()), 4),
        })
    return rows


# ----------------------------------------------------------------------
# Run summaries
# ----------------------------------------------------------------------


def summarize_run(run: dict[str, Any], common_games: float) -> dict[str, Any]:
    curve = run["curve"]
    games = curve["rl_games"]
    win = curve["win_rate_percent"]
    best_index = int(np.argmax(win))
    if run["complete"]:
        common_index = int(np.searchsorted(games, common_games, side="right")) - 1
        common_index = max(common_index, 0)
        common_value = float(win[common_index])
    else:
        common_value = float("nan")
    metrics = run["metrics"]
    reward_mean = np.asarray([row["reward_mean"] for row in metrics])
    batch_win = np.asarray([row["batch_win_rate"] for row in metrics])
    good_pct = np.asarray([row["good_pct"] for row in metrics])
    return {
        "execucao": run["label"],
        "arquitetura": run["era"],
        "lr": run["learning_rate"],
        "baseline": run["baseline"],
        "buckets": run["buckets"],
        "iteracoes": len(metrics),
        "partidas": int(games[-1]),
        "vitoria_inicial_pct": round(float(win[0]), 3),
        "vitoria_melhor_pct": round(float(win[best_index]), 3),
        "partidas_ate_melhor": int(games[best_index]),
        "vitoria_final_pct": round(float(win[-1]), 3),
        "vitoria_em_partidas_comuns_pct": (
            round(common_value, 3) if np.isfinite(common_value) else ""
        ),
        "ganho_sobre_sl_pct": round(float(win[best_index] - win[0]), 3),
        "recompensa_media_mediana": round(float(np.median(reward_mean)), 5),
        "decisoes_positivas_mediana_pct": round(float(np.median(good_pct)), 3),
        "vitoria_em_lote_mediana_pct": round(float(np.median(batch_win) * 100.0), 3),
        "corr_recompensa_vitoria": round(
            float(np.corrcoef(reward_mean, batch_win)[0, 1]), 4
        ),
        "completa": "sim" if run["complete"] else "nao",
    }


def wilson_halfwidth(rate_pct: float, games: int = DIAGNOSTIC_GAMES) -> float:
    """Return the half-width of the 95% interval the diagnostic reports."""
    p = rate_pct / 100.0
    return 1.96 * float(np.sqrt(p * (1.0 - p) / games)) * 100.0


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------


def plot_win_rate(runs: list[dict[str, Any]], key: str, xlabel: str, path: Path,
                  title: str) -> None:
    figure, axes = plt.subplots(figsize=(11.0, 6.2))
    for run in runs:
        curve = run["curve"]
        style = "-" if run["era"] == LEGACY else "--"
        axes.plot(
            curve[key] / (1e6 if key == "rl_games" else 1.0),
            curve["win_rate_percent"],
            style,
            color=RUN_COLORS[run["label"]],
            linewidth=1.4,
            label=f"{run['label']} ({run['era']}, lr={run['learning_rate']:g})",
        )
        best = int(np.argmax(curve["win_rate_percent"]))
        axes.plot(
            curve[key][best] / (1e6 if key == "rl_games" else 1.0),
            curve["win_rate_percent"][best],
            "o",
            color=RUN_COLORS[run["label"]],
            markersize=5,
        )
    start = runs[0]["curve"]["win_rate_percent"][0]
    axes.axhline(start, color="#666666", linewidth=1.0, linestyle=":")
    axes.annotate(
        f"política supervisionada inicial: {start:.2f}%",
        xy=(axes.get_xlim()[1], start),
        xytext=(-6, 5),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color="#444444",
    )
    axes.set_title(title)
    axes.set_xlabel(xlabel)
    axes.set_ylabel("vitórias contra o oponente aleatório (%)")
    axes.grid(alpha=0.25)
    axes.legend(fontsize=8.5, loc="lower right", ncol=2)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_terminal_utility(path: Path) -> None:
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13.0, 5.6), sharey=True, constrained_layout=True
    )
    pips = np.arange(0, 60)
    left.plot(pips, np.full_like(pips, 1.0, dtype=float), color="#08519c",
              linewidth=2.0, label="vitória por mão vazia")
    left.plot(pips, 1.0 - LEGACY_FINAL_PIP_PENALTY * pips, color="#3182bd",
              linewidth=2.0, linestyle="--", label="vitória por bloqueio")
    left.plot(pips, -1.0 - LEGACY_FINAL_PIP_PENALTY * pips, color="#a50f15",
              linewidth=2.0, label="derrota (qualquer desfecho)")
    left.axhline(0.0, color="#666666", linewidth=0.9)
    left.set_title("Arquitetura anterior")
    left.set_xlabel("pontos ainda na mão do aprendiz no fim da partida")
    left.set_ylabel("utilidade terminal (sem desconto)")
    left.grid(alpha=0.25)
    left.legend(fontsize=9)

    margin = np.arange(0, 30)
    magnitude = np.asarray([
        blocked_reward_magnitude(int(value), MAX_PIP_DOUBLE_SIX) for value in margin
    ])
    right.plot(margin, np.full_like(margin, 1.0, dtype=float), color="#08519c",
               linewidth=2.0, label="vitória por mão vazia")
    right.plot(margin, magnitude, color="#3182bd", linewidth=2.0, linestyle="--",
               label="vitória por bloqueio: $m(\\Delta_p)$")
    right.plot(margin, np.full_like(margin, -1.0, dtype=float), color="#a50f15",
               linewidth=2.0, label="derrota por mão vazia")
    right.plot(margin, -magnitude, color="#ef3b2c", linewidth=2.0, linestyle="--",
               label="derrota por bloqueio")
    right.axhline(0.0, color="#666666", linewidth=0.9)
    right.axvline(2 * MAX_PIP_DOUBLE_SIX, color="#999999", linewidth=0.9,
                  linestyle=":")
    right.annotate("saturação $S=2\\cdot$max_pip", xy=(2 * MAX_PIP_DOUBLE_SIX, -1.6),
                   xytext=(4, 0), textcoords="offset points", fontsize=8.5,
                   color="#666666")
    right.set_title("Arquitetura atual")
    right.set_xlabel("margem de pontos $\\Delta_p$ do desfecho bloqueado")
    right.grid(alpha=0.25)
    right.legend(fontsize=9)
    left.set_ylim(-5.2, 1.4)
    figure.suptitle(
        "Utilidade terminal das quatro classes de desfecho", fontweight="bold"
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_return_distribution(scored: dict[str, np.ndarray], path: Path) -> None:
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13.0, 5.6), sharey=True, constrained_layout=True
    )
    won = scored["won"]
    bins = np.linspace(-3.0, 3.2, 130)
    for axes, prefix, title in (
        (left, "legacy", "Arquitetura anterior"),
        (right, "current", "Arquitetura atual"),
    ):
        values = scored[f"{prefix}_return"]
        axes.hist(values[won], bins=bins, color="#3182bd", alpha=0.72,
                  label="decisões em partidas ganhas")
        axes.hist(values[~won], bins=bins, color="#a50f15", alpha=0.62,
                  label="decisões em partidas perdidas")
        axes.axvline(0.0, color="#222222", linewidth=1.1)
        wrong = float((values[~won] > 0.0).mean() * 100.0)
        axes.set_title(f"{title}\nretorno positivo em partidas perdidas: {wrong:.2f}%")
        axes.set_xlabel("retorno $G(t)$ de uma decisão")
        axes.grid(alpha=0.22)
        axes.legend(fontsize=9)
    left.set_ylabel("decisões")
    figure.suptitle(
        "Distribuição do retorno por decisão nas mesmas 422.055 decisões",
        fontweight="bold",
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_component_balance(scored: dict[str, np.ndarray], path: Path) -> None:
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13.0, 5.6), constrained_layout=True
    )
    labels = ("metade terminal\n$\\gamma_f^{k}U_T$", "metade local\n$G_I(t)$")
    legacy = (
        float(np.abs(scored["legacy_terminal"]).mean()),
        float(np.abs(scored["legacy_local"]).mean()),
    )
    current = (
        float(np.abs(scored["current_terminal"]).mean()),
        float(np.abs(scored["current_local"]).mean()),
    )
    positions = np.arange(2)
    left.bar(positions - 0.19, legacy, width=0.36, color=COLORS[LEGACY],
             label="anterior")
    left.bar(positions + 0.19, current, width=0.36, color=COLORS[CURRENT],
             label="atual")
    for index, (a, b) in enumerate(zip(legacy, current)):
        left.annotate(f"{a:.3f}", (index - 0.19, a), ha="center",
                      xytext=(0, 3), textcoords="offset points", fontsize=9)
        left.annotate(f"{b:.3f}", (index + 0.19, b), ha="center",
                      xytext=(0, 3), textcoords="offset points", fontsize=9)
    left.set_xticks(positions)
    left.set_xticklabels(labels)
    left.set_ylabel("magnitude média |componente|")
    left.set_title("Peso efetivo de cada metade do retorno")
    left.grid(alpha=0.25, axis="y")
    left.legend(fontsize=9)

    ratios = (legacy[1] / legacy[0], current[1] / current[0])
    right.bar(("anterior", "atual"), ratios,
              color=(COLORS[LEGACY], COLORS[CURRENT]), width=0.5)
    right.axhline(1.0, color="#222222", linewidth=1.0, linestyle="--")
    for index, value in enumerate(ratios):
        right.annotate(f"{value:.2f}x", (index, value), ha="center",
                       xytext=(0, 4), textcoords="offset points", fontsize=11,
                       fontweight="bold")
    right.set_ylabel("|metade local| / |metade terminal|")
    right.set_title(
        "Razão local/terminal com $\\eta$ = 0,5 nas duas arquiteturas"
    )
    right.grid(alpha=0.25, axis="y")
    figure.suptitle(
        "O mesmo $\\eta$ não produz o mesmo equilíbrio", fontweight="bold"
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_alignment(legacy: dict[str, float], current: dict[str, float],
                   path: Path) -> None:
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13.0, 5.6), constrained_layout=True
    )
    left.bar(("anterior", "atual"),
             (legacy["correlation_with_win"], current["correlation_with_win"]),
             color=(COLORS[LEGACY], COLORS[CURRENT]), width=0.5)
    for index, value in enumerate(
        (legacy["correlation_with_win"], current["correlation_with_win"])
    ):
        left.annotate(f"{value:+.3f}", (index, value), ha="center",
                      xytext=(0, 4), textcoords="offset points",
                      fontsize=11, fontweight="bold")
    left.set_ylim(0.0, 1.0)
    left.set_ylabel("correlação entre $G(t)$ e vencer a partida")
    left.set_title("Quanto o retorno informa sobre o resultado")
    left.grid(alpha=0.25, axis="y")

    categories = (
        "$G(t)>0$ em\npartidas perdidas",
        "$G(t)<0$ em\npartidas ganhas",
        "discordância\nde sinal",
    )
    legacy_values = (
        legacy["positive_in_losses_pct"],
        legacy["negative_in_wins_pct"],
        legacy["sign_disagreement_pct"],
    )
    current_values = (
        current["positive_in_losses_pct"],
        current["negative_in_wins_pct"],
        current["sign_disagreement_pct"],
    )
    positions = np.arange(3)
    right.bar(positions - 0.19, legacy_values, width=0.36,
              color=COLORS[LEGACY], label="anterior")
    right.bar(positions + 0.19, current_values, width=0.36,
              color=COLORS[CURRENT], label="atual")
    for index, (a, b) in enumerate(zip(legacy_values, current_values)):
        right.annotate(f"{a:.1f}%", (index - 0.19, a), ha="center",
                       xytext=(0, 3), textcoords="offset points", fontsize=9)
        right.annotate(f"{b:.1f}%", (index + 0.19, b), ha="center",
                       xytext=(0, 3), textcoords="offset points", fontsize=9)
    right.set_xticks(positions)
    right.set_xticklabels(categories)
    right.set_ylabel("% das decisões")
    right.set_title("Decisões cujo sinal contradiz o resultado")
    right.grid(alpha=0.25, axis="y")
    right.legend(fontsize=9)
    figure.suptitle(
        "Alinhamento entre recompensa e resultado da partida", fontweight="bold"
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_training_dynamics(runs: list[dict[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(
        2, 2, figsize=(13.0, 8.4), constrained_layout=True
    )
    fields = (
        ("reward_mean", "recompensa média por decisão", axes[0][0]),
        ("good_pct", "decisões com retorno positivo (%)", axes[0][1]),
        ("entropy", "entropia da política", axes[1][0]),
        ("final_clip_fraction", "fração recortada pelo PPO", axes[1][1]),
    )
    window = 101
    for field, title, panel in fields:
        for run in runs:
            if not run["complete"]:
                continue
            values = np.asarray([row[field] for row in run["metrics"]])
            if len(values) >= window:
                kernel = np.ones(window) / window
                smoothed = np.convolve(values, kernel, mode="valid")
                x = np.arange(len(smoothed)) + window // 2
            else:
                smoothed, x = values, np.arange(len(values))
            panel.plot(
                x, smoothed,
                "-" if run["era"] == LEGACY else "--",
                color=RUN_COLORS[run["label"]], linewidth=1.3,
                label=f"{run['label']} ({run['era']})",
            )
        panel.set_title(title)
        panel.set_xlabel("iteração de PPO")
        panel.grid(alpha=0.25)
    axes[0][0].axhline(0.0, color="#222222", linewidth=0.9, linestyle=":")
    axes[0][1].axhline(50.0, color="#222222", linewidth=0.9, linestyle=":")
    axes[0][0].legend(fontsize=8, loc="upper left")
    figure.suptitle(
        "Dinâmica de treino sob cada arquitetura (média móvel de 101 iterações)",
        fontweight="bold",
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_summary(summaries: list[dict[str, Any]], path: Path) -> None:
    rows = [row for row in summaries if row["completa"] == "sim"]
    rows.sort(key=lambda row: row["vitoria_melhor_pct"])
    figure, axes = plt.subplots(figsize=(11.0, 6.2))
    positions = np.arange(len(rows))
    values = [row["vitoria_melhor_pct"] for row in rows]
    errors = [wilson_halfwidth(value) for value in values]
    colors = [COLORS[row["arquitetura"]] for row in rows]
    axes.barh(positions, values, xerr=errors, color=colors, height=0.62,
              error_kw={"ecolor": "#333333", "capsize": 3, "linewidth": 1.0})
    for position, row in zip(positions, rows):
        axes.annotate(
            f"{row['vitoria_melhor_pct']:.2f}%  (lr={row['lr']:g}, "
            f"{row['partidas_ate_melhor'] / 1e6:.1f} M)",
            (row["vitoria_melhor_pct"], position),
            xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
        )
    axes.set_yticks(positions)
    axes.set_yticklabels([row["execucao"] for row in rows])
    axes.set_xlim(60.0, 69.5)
    axes.set_xlabel("melhor taxa de vitória contra o aleatório (%), IC 95%")
    axes.axvline(
        summaries[0]["vitoria_inicial_pct"], color="#666666", linestyle=":",
        linewidth=1.0,
    )
    axes.set_title(
        "Melhor resultado de cada execução double-six, por arquitetura de recompensa"
    )
    axes.grid(alpha=0.25, axis="x")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[LEGACY]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS[CURRENT]),
    ]
    axes.legend(handles, ("recompensa anterior", "recompensa atual"),
                fontsize=9.5, loc="lower right")
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def curve_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        curve = run["curve"]
        for index in range(len(curve["rl_games"])):
            rows.append({
                "execucao": run["label"],
                "arquitetura": run["era"],
                "lr": run["learning_rate"],
                "partidas": int(curve["rl_games"][index]),
                "iteracoes": int(curve["rl_iterations"][index]),
                "horas": round(float(curve["rl_elapsed_hours"][index]), 4),
                "vitoria_pct": round(float(curve["win_rate_percent"][index]), 4),
                "ic95_baixo_pct": round(float(curve["ci95_low_percent"][index]), 4),
                "ic95_alto_pct": round(float(curve["ci95_high_percent"][index]), 4),
            })
    return rows


def return_percentile_rows(scored: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    percentiles = (1, 5, 25, 50, 75, 95, 99)
    rows = []
    for era, prefix in ((LEGACY, "legacy"), (CURRENT, "current")):
        for component, suffix in (
            ("retorno G(t)", "return"),
            ("metade terminal", "terminal"),
            ("metade local", "local"),
        ):
            values = scored[f"{prefix}_{suffix}"]
            row: dict[str, Any] = {
                "arquitetura": era,
                "componente": component,
                "media": round(float(values.mean()), 5),
                "desvio": round(float(values.std()), 5),
                "magnitude_media": round(float(np.abs(values).mean()), 5),
                "minimo": round(float(values.min()), 5),
                "maximo": round(float(values.max()), 5),
            }
            for percentile in percentiles:
                row[f"p{percentile}"] = round(
                    float(np.percentile(values, percentile)), 5
                )
            rows.append(row)
    return rows


def format_table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def write_report(summaries, endings, legacy_align, current_align, scored,
                 balance, pair_rows, common_games, equivalent_eta) -> None:
    by_label = {row["execucao"]: row for row in summaries}
    legacy_best = max(
        (row for row in summaries
         if row["arquitetura"] == LEGACY and row["completa"] == "sim"),
        key=lambda row: row["vitoria_melhor_pct"],
    )
    current_best = max(
        (row for row in summaries
         if row["arquitetura"] == CURRENT and row["completa"] == "sim"),
        key=lambda row: row["vitoria_melhor_pct"],
    )
    gap = legacy_best["vitoria_melhor_pct"] - current_best["vitoria_melhor_pct"]
    half = wilson_halfwidth(legacy_best["vitoria_melhor_pct"])

    lines = [
        "# Recompensa anterior e recompensa atual: o que mudou e o que o "
        "resultado mostra",
        "",
        f"*Gerado em {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %z')} "
        f"a partir do estado corrente dos diretórios de execução. Uma "
        f"execução ainda em andamento avança entre duas gerações deste "
        f"relatório.*",
        "",
        "## Conclusão",
        "",
        f"A queda é real e maior que o ruído do painel de diagnóstico. A melhor "
        f"execução sob a recompensa anterior chegou a "
        f"**{legacy_best['vitoria_melhor_pct']:.3f}%** contra o oponente "
        f"aleatório (`{legacy_best['execucao']}`); a melhor sob a recompensa "
        f"atual chegou a **{current_best['vitoria_melhor_pct']:.3f}%** "
        f"(`{current_best['execucao']}`). A diferença de **{gap:.3f} pontos "
        f"percentuais** é cerca de {gap / half:.1f}x a meia-largura do "
        f"intervalo de 95% do painel de {DIAGNOSTIC_GAMES:,} partidas "
        f"(±{half:.2f} pp), então não é flutuação do diagnóstico.",
        "",
        "A causa provável não é o novo formato da utilidade terminal, e sim o "
        "**deslocamento do equilíbrio entre a metade terminal e a metade "
        "local** que o novo formato provocou sem que `reward_eta` mudasse. "
        f"Recomputando as duas recompensas sobre as **mesmas "
        f"{len(scored['won']):,} decisões reais**, a metade local passou de "
        f"{balance['legacy_ratio']:.2f}x para {balance['current_ratio']:.2f}x "
        f"a magnitude da metade terminal — um fator de "
        f"{balance['current_ratio'] / balance['legacy_ratio']:.1f} no peso "
        "efetivo do termo de moldagem, com `reward_eta` fixo em 0,5 nas duas.",
        "",
        "A consequência mensurável é que a recompensa deixou de informar sobre "
        "o resultado da partida com a mesma nitidez: a correlação entre o "
        f"retorno de uma decisão e vencer caiu de "
        f"**{legacy_align['correlation_with_win']:+.3f}** para "
        f"**{current_align['correlation_with_win']:+.3f}**, e a fração de "
        "decisões tomadas em **partidas perdidas** que ainda assim recebem "
        f"retorno positivo subiu de "
        f"**{legacy_align['positive_in_losses_pct']:.2f}%** para "
        f"**{current_align['positive_in_losses_pct']:.2f}%**.",
        "",
        "## O que exatamente mudou",
        "",
        "As duas arquiteturas compartilham a mesma estrutura de dois termos, "
        "os mesmos descontos e o mesmo ponto de mistura:",
        "",
        "```",
        "G(t) = (1 - eta) * gamma_f^k * U_T  +  eta * G_local(t)",
        f"gamma_f = {GAMMA_F}   gamma_i = {GAMMA_I}   eta = {REWARD_ETA}",
        "```",
        "",
        "Os valores de `gamma_f`, `gamma_i`, `reward_eta` e o modo de "
        "distância `turn-turn` são idênticos em todas as execuções comparadas "
        "aqui, então saem da comparação. O que mudou são as duas funções que "
        "produzem `U_T` e `G_local`.",
        "",
        "**Utilidade terminal.** A anterior era um resultado binário menos uma "
        "penalidade proporcional aos pontos que o aprendiz ainda tinha na mão:",
        "",
        "```",
        "U_T_anterior = (+1 se venceu, -1 se perdeu) - 0.05 * pontos_na_mao_do_aprendiz",
        "```",
        "",
        "A atual decompõe o desfecho em duas classes mutuamente exclusivas e "
        "normaliza tudo para `[-1, +1]`:",
        "",
        "```",
        "U_T_atual = +/-1                        (fim por mão vazia)",
        "U_T_atual = +/-m(dp),  m(dp) = 0.1 + 0.9 * min(dp / (2*max_pip), 1)   (bloqueio)",
        "```",
        "",
        "**Eventos locais.** A anterior valia `+/-0.2` para um saque e "
        "`+/-0.1` para um passe. A atual normaliza os dois para `+/-1.0` e "
        "expressa a importância relativa só pela razão entre `a_D` e `a_P`; "
        "com os pesos padrão iguais a 1, ambas as escalas resolvem para 1,0. "
        "Um saque passou a valer **5x** mais e um passe **10x** mais, em "
        "unidades da mesma utilidade terminal.",
        "",
        "## Efeito sobre cada classe de desfecho",
        "",
        "Medido nas mesmas decisões, sem desconto temporal:",
        "",
    ]
    lines.extend(format_table(
        ["Desfecho", "% das decisões", "Pontos na mão", "U anterior (média)",
         "U anterior (mín)", "U atual (média)"],
        [[
            row["desfecho"],
            f"{row['fracao_decisoes_pct']:.2f}%",
            f"{row['pips_finais_medios']:.1f}",
            f"{row['u_anterior_medio']:+.3f}",
            f"{row['u_anterior_min']:+.3f}",
            f"{row['u_atual_medio']:+.3f}",
        ] for row in endings],
    ))
    lines.extend([
        "",
        "Três leituras saem daqui:",
        "",
        "1. **A recompensa anterior era fortemente avessa à derrota.** Uma "
        "derrota custava em média entre 1,57 e 2,15, enquanto uma vitória "
        "rendia entre 0,31 e 1,00. A assimetria não era um parâmetro "
        "escolhido: vinha da penalidade de pontos, que era subtraída em todo "
        "desfecho e não tinha piso.",
        "2. **A recompensa anterior pagava por descartar peças pesadas.** "
        "Cada ponto restante na mão custava 0,05 independentemente de como a "
        "partida terminasse. Contra um oponente aleatório, esvaziar a mão "
        "rápido e não ficar com peças altas é exatamente a política que mais "
        "vence, então essa penalidade empurrava na direção certa para o "
        "diagnóstico usado.",
        "3. **Vencer por bloqueio segurando peças pesadas podia ser punido.** "
        "O mínimo de `U anterior` numa vitória por bloqueio é negativo: o "
        "aprendiz vencia a partida e ainda assim recebia retorno negativo. "
        "A arquitetura atual corrige isso — e essa correção é uma melhora "
        "real de coerência, não um defeito.",
        "",
        "## O equilíbrio entre as duas metades",
        "",
        "É aqui que está o problema. `reward_eta` = 0,5 diz que as duas "
        "metades pesam igual, mas só controla o coeficiente, não a magnitude "
        "do que multiplica:",
        "",
    ])
    lines.extend(format_table(
        ["Arquitetura", "magnitude média da metade terminal",
         "magnitude média da metade local", "razão local/terminal"],
        [
            [LEGACY, f"{balance['legacy_terminal']:.4f}",
             f"{balance['legacy_local']:.4f}", f"{balance['legacy_ratio']:.2f}x"],
            [CURRENT, f"{balance['current_terminal']:.4f}",
             f"{balance['current_local']:.4f}", f"{balance['current_ratio']:.2f}x"],
        ],
    ))
    lines.extend([
        "",
        "Sob a recompensa anterior a moldagem local era um termo acessório, "
        f"com {balance['legacy_ratio']:.0%} da magnitude do sinal terminal. "
        "Sob a atual ela é o termo **dominante**, com "
        f"{balance['current_ratio']:.2f} vezes a magnitude do sinal terminal. "
        "Duas mudanças somaram-se nessa direção: os eventos locais ficaram 5 a "
        "10 vezes maiores, e a utilidade terminal ficou *menor* em módulo "
        f"({balance['legacy_terminal']:.3f} para "
        f"{balance['current_terminal']:.3f}) porque perdeu a cauda de "
        "penalidade de pontos e porque as vitórias por bloqueio passaram a "
        "valer `m(dp)` em vez de aproximadamente 1.",
        "",
        "`G_local` também não é renormalizado para `[-1, 1]` — isso é "
        "deliberado e está documentado em `training/rl/README.md` — então a "
        f"soma geométrica de eventos alcança {scored['current_local'].max():+.2f} "
        f"e {scored['current_local'].min():+.2f} nas caudas, contra "
        f"{scored['legacy_local'].max():+.2f} e "
        f"{scored['legacy_local'].min():+.2f} antes.",
        "",
        "## Alinhamento entre recompensa e resultado",
        "",
    ])
    lines.extend(format_table(
        ["Métrica", "anterior", "atual"],
        [
            ["correlação entre G(t) e vencer",
             f"{legacy_align['correlation_with_win']:+.4f}",
             f"{current_align['correlation_with_win']:+.4f}"],
            ["G(t) > 0 em partidas perdidas",
             f"{legacy_align['positive_in_losses_pct']:.2f}%",
             f"{current_align['positive_in_losses_pct']:.2f}%"],
            ["G(t) < 0 em partidas ganhas",
             f"{legacy_align['negative_in_wins_pct']:.2f}%",
             f"{current_align['negative_in_wins_pct']:.2f}%"],
            ["discordância de sinal",
             f"{legacy_align['sign_disagreement_pct']:.2f}%",
             f"{current_align['sign_disagreement_pct']:.2f}%"],
            ["média de G(t)", f"{legacy_align['mean']:+.4f}",
             f"{current_align['mean']:+.4f}"],
            ["desvio de G(t)", f"{legacy_align['std']:.4f}",
             f"{current_align['std']:.4f}"],
        ],
    ))
    lines.extend([
        "",
        "Mais de um terço das decisões tomadas em partidas que o agente "
        "**perdeu** recebem retorno positivo sob a recompensa atual. Para uma "
        "política de gradiente isso não é ruído neutro: essas decisões são "
        "reforçadas. O termo local é quase sempre positivo porque a política "
        "supervisionada inicial já força o oponente a sacar e passar com "
        "frequência, e esse crédito agora chega em escala comparável à do "
        "próprio resultado da partida.",
        "",
        "O mesmo efeito aparece nos dados de treino ao vivo, sem nenhuma "
        "recomputação: a correlação entre `reward_mean` e `batch_win_rate` "
        "por iteração cai de "
        f"{by_label['bucket_heuristic_recent']['corr_recompensa_vitoria']:+.3f} "
        f"e {by_label['bucket_heuristic']['corr_recompensa_vitoria']:+.3f} "
        "nas execuções antigas para "
        f"{by_label['default_lr032']['corr_recompensa_vitoria']:+.3f} e "
        f"{by_label['d6_maxwr_lr032']['corr_recompensa_vitoria']:+.3f} nas "
        "novas, e a `reward_mean` mediana sai de aproximadamente zero para "
        f"cerca de {by_label['default_lr032']['recompensa_media_mediana']:+.3f} "
        "com a taxa de vitória em lote parada em ~50,8%.",
        "",
        "## Verificação direta na iteração 1",
        "",
        "As execuções `bucket_heuristic_recent` (anterior) e `default_lr032` "
        "(atual) partem dos mesmos pesos supervisionados com a mesma semente. "
        "Na primeira iteração elas jogam **as mesmas partidas**: 8.120 "
        "decisões e a mesma contagem de eventos "
        "`[6910, 4288, 6934, 4244]` nas duas. É um A/B exato da função de "
        "recompensa sobre trajetórias idênticas:",
        "",
    ])
    lines.extend(format_table(
        ["Iteração 1", "bucket_heuristic_recent (anterior)",
         "default_lr032 (atual)"],
        [
            ["recompensa média", "-0.02323", "+0.22665"],
            ["desvio", "0.38669", "0.74975"],
            ["mínimo / máximo", "-1.477 / +0.643", "-2.232 / +2.463"],
            ["decisões com retorno positivo", "52.62%", "61.72%"],
            ["taxa de vitória do lote", "50.75%", "50.75%"],
        ],
    ))
    lines.extend([
        "",
        "A taxa de vitória é a mesma porque as partidas são as mesmas. A "
        "recompensa não é.",
        "",
        "## Resultados por execução",
        "",
    ])
    lines.extend(format_table(
        ["Execução", "Arq.", "lr", "Baseline", "Buckets", "Partidas", "Melhor",
         "Partidas até o melhor", "Final", f"Em {common_games / 1e6:.1f} M"],
        [[
            row["execucao"], row["arquitetura"], f"{row['lr']:g}",
            row["baseline"], row["buckets"], f"{row['partidas'] / 1e6:.1f} M",
            f"**{row['vitoria_melhor_pct']:.3f}%**"
            if row["completa"] == "sim" else f"{row['vitoria_melhor_pct']:.3f}%",
            f"{row['partidas_ate_melhor'] / 1e6:.1f} M",
            f"{row['vitoria_final_pct']:.3f}%",
            f"{row['vitoria_em_partidas_comuns_pct']:.3f}%"
            if row["vitoria_em_partidas_comuns_pct"] != "" else "—",
        ] for row in summaries],
    ))
    lines.extend([
        "",
        f"Todas as execuções partem da mesma política supervisionada, que "
        f"vale {summaries[0]['vitoria_inicial_pct']:.3f}% contra o aleatório. "
        "A execução `d6_maxwr_lr016` foi interrompida com 0,7 M de partidas e "
        "não entra em nenhuma comparação.",
        "",
        "### Pares com o mesmo conjunto de oponentes",
        "",
    ])
    lines.extend(format_table(
        ["Buckets", "anterior", "atual", "Diferença"],
        pair_rows,
    ))
    lines.extend([
        "",
        "## Fatores de confusão",
        "",
        "Esta comparação **não é um experimento controlado da recompensa "
        "sozinha**, e vale registrar exatamente onde ela é frágil:",
        "",
        "1. **A taxa de aprendizado difere.** As execuções antigas usaram "
        "lr = 0,001; as novas, 0,016 e 0,032. A grade `analysis/analise_lr_KL` "
        "mostrou, em `double-three` e sob a recompensa anterior, que taxas "
        "mais altas produziram jogadores **melhores** — o que faz a lr "
        "trabalhar contra a hipótese de que ela explique a queda, mas não a "
        "elimina, porque a grade foi feita em outro ruleset e sob a outra "
        "recompensa.",
        "2. **O aumento de escala da recompensa muda o passo efetivo.** O "
        "desvio do retorno praticamente dobrou, o que multiplica o gradiente "
        "antes mesmo da lr. As novas execuções têm KL mediana ~4x maior e "
        "fração de recorte ~3,5x maior. Parte do efeito observado pode ser "
        "essa mudança de passo, não a forma da recompensa.",
        "3. **Uma execução por configuração.** Não há repetição com sementes "
        "diferentes; a variação entre execuções não está medida.",
        "4. **O corpus de recomputação vem de uma política só** — o "
        "checkpoint `double six 66p local.npz`, treinado sob a recompensa "
        "anterior, jogando contra o heurístico. As proporções de desfecho "
        "refletem essa política. As conclusões sobre *forma* e *escala* das "
        "duas funções não dependem disso; as proporções por classe de "
        "desfecho, sim.",
        "",
        "## O experimento que decide",
        "",
        "Uma única execução resolve a ambiguidade: recompensa atual, "
        "**lr = 0,001**, buckets `heuristic,recent`, semente 42, mesmos pesos "
        "supervisionados — ou seja, `bucket_heuristic_recent` com a única "
        "diferença sendo a recompensa.",
        "",
        "```bash",
        "python -u -m training.pipeline forever \\",
        "    --learning-rate 0.001 \\",
        "    --opponent-buckets heuristic,recent \\",
        "    --run-name recompensa_atual_lr001",
        "```",
        "",
        "Se essa execução ficar perto de 66%, a recompensa não é a causa e o "
        "problema está na taxa de aprendizado combinada com a nova escala. Se "
        "ficar perto de 65%, a recompensa é a causa.",
        "",
        "Duas observações sobre como corrigir o desequilíbrio, caso ele "
        "se confirme como a causa:",
        "",
        "- **Reduzir `reward_eta`.** O equilíbrio efetivo entre as metades é "
        "`eta * |G_local| / ((1 - eta) * |G_terminal|)`. Para recuperar com as "
        f"magnitudes atuais o equilíbrio de {balance['legacy_ratio']:.2f}x que "
        f"a recompensa anterior tinha com eta = 0,5, seria preciso "
        f"**reward_eta ≈ {equivalent_eta:.3f}** — cerca de "
        f"{equivalent_eta * 100:.0f}% em vez de 50%.",
        "- **Reduzir `immediate_draw_weight` e `immediate_pass_weight` "
        "juntos** não funciona: a normalização por `max(a_D, a_P)` divide o "
        "par pelo seu maior membro, então só a *razão* entre eles é "
        "ajustável. A escala absoluta do termo local só se move por "
        "`reward_eta`. Isso é uma propriedade da arquitetura atual que vale "
        "registrar: **não existe hoje um controle direto da magnitude local**.",
        "",
        "## Reprodução",
        "",
        "```bash",
        "/home/diego/CCO/amb_virtual/bin/python "
        "analysis/recompensa_anterior_vs_atual/analyze.py",
        "```",
        "",
        "O script relê os diretórios de execução e o corpus derivado e "
        "regenera as figuras, os CSVs, `analysis_summary.json` e este "
        "relatório. Nenhum diretório de execução, modelo ou dataset é escrito "
        "ou modificado.",
        "",
    ])
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10.5,
        "axes.titleweight": "bold",
        "axes.titlesize": 12.5,
        "axes.labelsize": 11,
        "savefig.bbox": "tight",
    })

    runs = [load_run(label, directory) for label, directory in RUNS]
    shared = {
        (run["gamma_f"], run["gamma_i"], run["reward_eta"], run["distance_mode"])
        for run in runs
    }
    if len(shared) != 1:
        raise SystemExit(
            "The runs do not share gamma_f/gamma_i/reward_eta/distance mode, "
            f"so the reward comparison is not isolated: {shared}"
        )

    complete = [run for run in runs if run["complete"]]
    common_games = min(float(run["curve"]["rl_games"][-1]) for run in complete)
    summaries = [summarize_run(run, common_games) for run in runs]
    by_label = {row["execucao"]: row for row in summaries}

    corpus = load_corpus()
    scored = score_corpus(corpus)
    legacy_align = alignment(scored["legacy_return"], scored["won"])
    current_align = alignment(scored["current_return"], scored["won"])
    endings = ending_breakdown(scored)
    balance = {
        "legacy_terminal": float(np.abs(scored["legacy_terminal"]).mean()),
        "legacy_local": float(np.abs(scored["legacy_local"]).mean()),
        "current_terminal": float(np.abs(scored["current_terminal"]).mean()),
        "current_local": float(np.abs(scored["current_local"]).mean()),
    }
    balance["legacy_ratio"] = balance["legacy_local"] / balance["legacy_terminal"]
    balance["current_ratio"] = balance["current_local"] / balance["current_terminal"]
    # The eta that reproduces the previous effective local/terminal balance
    # under the current magnitudes: eta / (1 - eta) * current_ratio must
    # equal legacy_ratio, because both architectures used eta = 0.5.
    odds = balance["legacy_ratio"] / balance["current_ratio"]
    equivalent_eta = odds / (1.0 + odds)
    balance["equivalent_reward_eta"] = equivalent_eta

    pair_rows = []
    for legacy_label, current_label in PAIRS:
        left = by_label[legacy_label]
        right = by_label[current_label]
        pair_rows.append([
            left["buckets"],
            f"{left['execucao']}: {left['vitoria_melhor_pct']:.3f}% "
            f"(lr={left['lr']:g})",
            f"{right['execucao']}: {right['vitoria_melhor_pct']:.3f}% "
            f"(lr={right['lr']:g})",
            f"**{right['vitoria_melhor_pct'] - left['vitoria_melhor_pct']:+.3f} pp**",
        ])

    write_csv(HERE / "resumo_execucoes.csv", summaries)
    write_csv(HERE / "curvas_vitoria.csv", curve_rows(runs))
    write_csv(HERE / "desfechos_terminais.csv", endings)
    write_csv(HERE / "retorno_por_decisao.csv", return_percentile_rows(scored))

    plot_win_rate(
        runs, "rl_games", "partidas de RL (milhões)",
        HERE / "01_taxa_vitoria_por_partidas.png",
        "Vitórias contra o oponente aleatório por volume de partidas",
    )
    plot_win_rate(
        runs, "rl_elapsed_hours", "tempo de parede de RL (h)",
        HERE / "02_taxa_vitoria_por_tempo.png",
        "Vitórias contra o oponente aleatório por tempo de treino",
    )
    plot_terminal_utility(HERE / "03_utilidade_terminal.png")
    plot_return_distribution(scored, HERE / "04_retorno_por_decisao.png")
    plot_component_balance(scored, HERE / "05_equilibrio_terminal_local.png")
    plot_alignment(legacy_align, current_align,
                   HERE / "06_alinhamento_recompensa_resultado.png")
    plot_training_dynamics(runs, HERE / "07_dinamica_de_treino.png")
    plot_summary(summaries, HERE / "08_resumo_resultados.png")

    (HERE / "analysis_summary.json").write_text(
        json.dumps(
            {
                "ruleset": "double-six",
                "diagnostic_games": DIAGNOSTIC_GAMES,
                "shared_temporal_parameters": {
                    "gamma_f": GAMMA_F,
                    "gamma_i": GAMMA_I,
                    "reward_eta": REWARD_ETA,
                    "reward_distance_mode": runs[0]["distance_mode"],
                },
                "common_games": common_games,
                "runs": summaries,
                "pairs": [
                    {
                        "buckets": by_label[legacy_label]["buckets"],
                        "legacy": legacy_label,
                        "current": current_label,
                        "legacy_best_pct": by_label[legacy_label]["vitoria_melhor_pct"],
                        "current_best_pct": by_label[current_label]["vitoria_melhor_pct"],
                        "delta_pp": round(
                            by_label[current_label]["vitoria_melhor_pct"]
                            - by_label[legacy_label]["vitoria_melhor_pct"], 3
                        ),
                    }
                    for legacy_label, current_label in PAIRS
                ],
                "corpus": {
                    "path": str(DERIVED_CORPUS.relative_to(REPO)),
                    "games": corpus["summary"]["games"],
                    "decisions": corpus["summary"]["decisions"],
                    "policy": corpus["neural_policy"]["file"],
                    "policy_sha256": corpus["neural_policy"]["sha256"],
                },
                "reward_alignment": {
                    LEGACY: legacy_align,
                    CURRENT: current_align,
                },
                "component_balance": balance,
                "ending_breakdown": endings,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    write_report(summaries, endings, legacy_align, current_align, scored,
                 balance, pair_rows, common_games, equivalent_eta)

    print(f"decisions scored: {len(scored['won']):,}")
    print(
        f"local/terminal balance: {balance['legacy_ratio']:.2f}x -> "
        f"{balance['current_ratio']:.2f}x"
    )
    print(
        f"corr(G, win): {legacy_align['correlation_with_win']:+.4f} -> "
        f"{current_align['correlation_with_win']:+.4f}"
    )
    print("wrote REPORT.md, analysis_summary.json, 4 CSVs and 8 figures")


if __name__ == "__main__":
    main()
