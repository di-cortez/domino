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
DIAGNOSTIC_OPPONENT = "random"

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

# Small counts read as words in the report's prose.
NUMBER_WORDS = {2: "duas", 3: "três", 4: "quatro", 5: "cinco", 6: "seis"}

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
    ("default_lookup", "domino_rl_forever_seed42_rundefault_lookup"),
    # Trained on three other machines and shared as the four-file compact
    # bundle, so they carry a win-rate curve but no per-iteration metrics.
    ("rick_heuristic",
     "modelos_rick/20260904-154_rick_notebook-novo_s42_bucket_heuristic_batch_mean"),
    ("rick_random_desktop",
     "modelos_rick/20260904-153_rick_desktop_s42_bucket_random_batch_mean"),
    ("rick_random_notebook",
     "modelos_rick/20260904-152_rick_notebook-antigo_s42_bucket_random_batch_mean"),
    # The two runs that act on this report's own prediction: same current
    # reward, same everything else, but eta lowered to 0.115 -- the value the
    # corpus said would restore the previous local/terminal balance. They are
    # younger than the rest, which is what the matched horizon below is for.
    ("d6_random_eta0115", "domino_rl_forever_seed42_rund6_random_eta0115"),
    ("rick_random_eta0115",
     "modelos_rick/20260904-155_domino_rl_forever_seed42_rund6_random_eta0115"),
)

# The two runs that hold everything fixed except eta. The first pair is the
# controlled one: both ran on the same GPU from the same supervised binary.
ETA_PAIRS = (
    ("rick_random_desktop", "rick_random_eta0115"),
    ("rick_random_notebook", "d6_random_eta0115"),
)

# Pairs that hold the opponent-bucket set fixed across the two
# architectures. The third one also holds the learning rate fixed, which is
# what makes it the controlled comparison the earlier revisions of this report
# could only ask for.
PAIRS = (
    ("bucket_heuristic", "d6_maxwr_lr032"),
    ("bucket_heuristic_recent", "default_lr032"),
    ("bucket_heuristic", "rick_heuristic"),
    ("bucket_heuristic_recent", "default_lookup"),
)

# The one pair whose two runs differ in the reward architecture and in nothing
# else that the training loop reads except the advantage baseline.
CONTROLLED_PAIR = ("bucket_heuristic_recent", "default_lookup")

# Iterations averaged before any per-iteration series is read as a trend.
SMOOTHING_WINDOW = 101

# A run needs enough of a curve for "best" to mean anything. The threshold
# admits the eta runs, which are younger than the rest; the comparison that
# includes them is made at the matched horizon, not at each run's own end.
MIN_GAMES_FOR_COMPARISON = 2_500_000

COLORS = {LEGACY: "#08519c", CURRENT: "#99000d"}
# The current reward with eta moved back down: neither of the two
# architectures the report set out to compare, so neither colour.
CORRECTED_COLOR = "#238b45"
RUN_COLORS = {
    "bucket_heuristic": "#08519c",
    "bucket_heuristic_recent": "#3182bd",
    "baseline_zero": "#6baed6",
    "bucket_all": "#9ecae1",
    "d6_maxwr_lr032": "#99000d",
    "d6_maxwr_lr016": "#969696",
    "default_lr032": "#ef3b2c",
    "default_lr016_lookup": "#fb6a4a",
    "default_lookup": "#cb181d",
    "rick_heuristic": "#d94801",
    "rick_random_desktop": "#e6550d",
    "rick_random_notebook": "#fd8d3c",
    "d6_random_eta0115": "#006d2c",
    "rick_random_eta0115": "#41ab5d",
}

# Runs trained elsewhere and received as the four-file bundle. They are read
# the same way as the local ones; the set only exists so the report can say
# which numbers came from another machine and which of its columns are blank.
EXTERNAL_RUNS = frozenset({
    "rick_heuristic", "rick_random_desktop", "rick_random_notebook",
    "rick_random_eta0115",
})

# Drawn thicker: the run that isolates the reward from the learning rate, and
# the two that isolate eta from everything else.
EMPHASIZED_RUNS = frozenset({
    "default_lookup", "d6_random_eta0115", "rick_random_eta0115",
})


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def gzip_text(path: Path, *, encoding: str):
    """Open a gzipped file as text, matching the signature of ``open``."""
    return gzip.open(path, "rt", encoding=encoding)


def load_run(label: str, directory: str) -> dict[str, Any]:
    run_dir = RUN_ROOT / directory
    # A run trained here keeps the shareable files in a subdirectory; a run
    # received as the four-file bundle is that subdirectory. Both layouts hold
    # the same names, so the only difference is where to start looking.
    compact = run_dir / "run_compact_diagnostics"
    if not compact.is_dir():
        compact = run_dir
    config = json.loads((compact / "run_config.json").read_text(encoding="utf-8"))
    locked = config.get("locked_arguments", {})
    era = CURRENT if "terminal_empty_hand_weight" in locked else LEGACY

    with open(compact / "rl_vs_random_progress.csv", encoding="utf-8") as stream:
        progress = list(csv.DictReader(stream))
    curve = {
        key: np.asarray([float(row[key]) for row in progress])
        for key in progress[0]
    }

    # The per-iteration log is the one file the bundle leaves out, because it
    # is three orders of magnitude larger than the rest. Without it a run still
    # answers "how well did it play"; it cannot answer "how did it train".
    metrics: list[dict[str, Any]] = []
    for name, opener in (("training_metrics.jsonl", open),
                         ("training_metrics.jsonl.gz", gzip_text)):
        # A run trained here has the plain file; a bundle that chose to carry
        # the log compresses it, because gzip takes it from 6 MB to under 2.
        path = run_dir / name if name.endswith("jsonl") else compact / name
        if not path.is_file():
            continue
        with opener(path, encoding="utf-8") as stream:
            columns = json.loads(stream.readline())["columns"]
            # A log copied while its run was still going can end mid-line. The
            # rows before it are complete and usable, so the tail is dropped
            # rather than failing the whole analysis.
            metrics = [
                dict(zip(columns, json.loads(line)))
                for line in stream if line.endswith("\n")
            ]
        break

    return {
        "label": label,
        "directory": directory,
        "era": era,
        "created_at": config.get("created_at", ""),
        "learning_rate": float(locked.get("learning_rate", float("nan"))),
        # A null ``--baseline`` is not "no baseline": with advantage
        # normalization on, which every run here uses, it resolves to
        # batch-mean. Naming it that keeps the comparison table honest.
        "baseline": (locked.get("baseline") or ["batch-mean"])[0],
        "buckets": ",".join(locked.get("opponent_buckets") or []),
        "gamma_f": float(locked.get("gamma_f", GAMMA_F)),
        "gamma_i": float(locked.get("gamma_i", GAMMA_I)),
        "reward_eta": float(locked.get("reward_eta", REWARD_ETA)),
        "distance_mode": locked.get("reward_distance_mode", ""),
        "machine": config.get("machine", {}).get("gpu_name", ""),
        "sl_weights": config.get("supervised_weights_sha256", "")[:8],
        "curve": curve,
        "metrics": metrics,
        "external": label in EXTERNAL_RUNS,
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
        # The best the run had reached by the shared horizon. The single
        # reading at the horizon is one 100,000-game panel and carries the
        # full +/-0.3 pp of its own noise; the running maximum up to the same
        # point is the same statistic every run's headline number uses,
        # measured on a budget they all reached.
        common_best = float(np.max(win[:common_index + 1]))
        common_best_games = int(games[int(np.argmax(win[:common_index + 1]))])
    else:
        common_value = float("nan")
        common_best = float("nan")
        common_best_games = 0
    return {
        "execucao": run["label"],
        "arquitetura": run["era"],
        "origem": "externa" if run["external"] else "local",
        "gpu": run["machine"],
        "pesos_sl": run["sl_weights"],
        "lr": run["learning_rate"],
        "baseline": run["baseline"],
        "eta": run["reward_eta"],
        "buckets": run["buckets"],
        # A run whose opponent pool contains the diagnostic opponent is
        # measured on the distribution it trained on. Its number is real, but
        # it does not belong in a ranking against runs that never saw it.
        "treina_no_avaliador": (
            "sim" if DIAGNOSTIC_OPPONENT in run["buckets"].split(",") else "nao"
        ),
        "iteracoes": int(curve["rl_iterations"][-1]),
        "partidas": int(games[-1]),
        "vitoria_inicial_pct": round(float(win[0]), 3),
        "vitoria_melhor_pct": round(float(win[best_index]), 3),
        "partidas_ate_melhor": int(games[best_index]),
        "vitoria_final_pct": round(float(win[-1]), 3),
        "vitoria_em_partidas_comuns_pct": (
            round(common_value, 3) if np.isfinite(common_value) else ""
        ),
        "melhor_em_partidas_comuns_pct": (
            round(common_best, 3) if np.isfinite(common_best) else ""
        ),
        "partidas_ate_melhor_comum": common_best_games,
        "ganho_sobre_sl_pct": round(float(win[best_index] - win[0]), 3),
        **training_columns(run["metrics"]),
        "completa": "sim" if run["complete"] else "nao",
    }


# Every summary column that is read out of the per-iteration log, blank for a
# run that arrived without one. Listing them here keeps the CSV rectangular:
# an external run gets the same header as a local one, with empty cells where
# the answer genuinely is not available rather than a zero that looks like one.
TRAINING_COLUMNS = (
    "recompensa_media_mediana", "decisoes_positivas_mediana_pct",
    "vitoria_em_lote_mediana_pct", "corr_recompensa_vitoria",
    "entropia_inicial", "entropia_final", "entropia_minima",
    "iteracao_da_minima", "entropia_maxima_apos_minima",
    "kl_mediana", "recorte_mediano",
    "terminal_abs_mediana", "local_abs_mediana", "razao_local_terminal",
)


def training_columns(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the per-iteration log, or return blanks when there is none."""
    if not metrics:
        return {name: "" for name in TRAINING_COLUMNS}
    reward_mean = np.asarray([row["reward_mean"] for row in metrics])
    batch_win = np.asarray([row["batch_win_rate"] for row in metrics])
    good_pct = np.asarray([row["good_pct"] for row in metrics])
    entropy = np.asarray([row["entropy"] for row in metrics])
    # The per-iteration entropy is noisy; the trajectory only reads as a
    # trajectory after the same 101-iteration smoothing the figure uses.
    smoothed_entropy = (
        np.convolve(entropy, np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW,
                    mode="valid")
        if len(entropy) >= SMOOTHING_WINDOW else entropy
    )
    clip = np.asarray([row["final_clip_fraction"] for row in metrics])
    approx_kl = np.asarray([row["final_approx_kl"] for row in metrics])
    # Metrics schema v8 records both halves of every decision's return, so a
    # run under the current architecture reports its own terminal/local balance
    # without any offline recomputation. A v7 run predates the columns and
    # leaves them blank rather than pretending to a value.
    mixture = mixture_series(metrics)
    return {
        "recompensa_media_mediana": round(float(np.median(reward_mean)), 5),
        "decisoes_positivas_mediana_pct": round(float(np.median(good_pct)), 3),
        "vitoria_em_lote_mediana_pct": round(float(np.median(batch_win) * 100.0), 3),
        "corr_recompensa_vitoria": round(
            float(np.corrcoef(reward_mean, batch_win)[0, 1]), 4
        ),
        "entropia_inicial": round(float(entropy[0]), 4),
        "entropia_final": round(float(smoothed_entropy[-1]), 4),
        "entropia_minima": round(float(smoothed_entropy.min()), 4),
        "iteracao_da_minima": int(smoothed_entropy.argmin()),
        "entropia_maxima_apos_minima": round(
            float(smoothed_entropy[int(smoothed_entropy.argmin()):].max()), 4
        ),
        "kl_mediana": round(float(np.median(approx_kl)), 5),
        "recorte_mediano": round(float(np.median(clip)), 4),
        "terminal_abs_mediana": (
            round(float(np.median(mixture[0])), 5) if mixture else ""
        ),
        "local_abs_mediana": (
            round(float(np.median(mixture[1])), 5) if mixture else ""
        ),
        "razao_local_terminal": (
            round(float(np.median(mixture[1] / mixture[0])), 4) if mixture else ""
        ),
    }


def mixture_series(metrics: list[dict[str, Any]]):
    """Return the two logged return halves, or ``None`` for a pre-v8 run.

    ``terminal_abs_mean`` and ``local_abs_mean`` are ``E|(1-eta) G_T|`` and
    ``E|eta G_I|`` as the rollout actually credited them, so their ratio is the
    empirical mixture the run trained under rather than the nominal one that
    ``reward_eta`` names.
    """
    if not metrics or metrics[0].get("terminal_abs_mean") is None:
        return None
    terminal = np.asarray([row["terminal_abs_mean"] for row in metrics], dtype=float)
    local = np.asarray([row["local_abs_mean"] for row in metrics], dtype=float)
    return terminal, local


def trajectory_comparison(legacy: dict[str, Any], current: dict[str, Any],
                          checkpoints: tuple[float, ...] = (
                              2e6, 5e6, 10e6, 15e6, 18e6)) -> list[list[str]]:
    """Compare two runs at equal game counts rather than at their own peaks.

    A peak-to-peak comparison can be won by whichever run was left going for
    longer. Reading both curves at the same milestones removes that, and shows
    whether the gap is a plateau or a delay.
    """
    rows = []
    for target in checkpoints:
        cells = []
        for run in (legacy, current):
            games = run["curve"]["rl_games"]
            if games[-1] < target:
                cells.append(None)
                continue
            index = int(np.searchsorted(games, target, side="right")) - 1
            cells.append(float(run["curve"]["win_rate_percent"][index]))
        if cells[0] is None or cells[1] is None:
            continue
        rows.append([
            f"{target / 1e6:.0f} M",
            f"{cells[0]:.3f}%",
            f"{cells[1]:.3f}%",
            f"{cells[1] - cells[0]:+.3f} pp",
        ])
    return rows


def comma(value: float) -> str:
    """Format a number the way the surrounding Portuguese prose reads it."""
    return f"{value:g}".replace(".", ",")


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
    starred = False
    for run in runs:
        curve = run["curve"]
        # Solid: previous reward. Dashed: current reward. Dash-dot: current
        # reward, but trained against the opponent this axis measures, so the
        # curve is not on the same footing as the others.
        if DIAGNOSTIC_OPPONENT in run["buckets"].split(","):
            style, mark = "-.", " *"
            starred = True
        else:
            style, mark = ("-", "") if run["era"] == LEGACY else ("--", "")
        axes.plot(
            curve[key] / (1e6 if key == "rl_games" else 1.0),
            curve["win_rate_percent"],
            style,
            color=RUN_COLORS[run["label"]],
            linewidth=2.4 if run["label"] in EMPHASIZED_RUNS else 1.4,
            label=(
                f"{run['label']}{mark} ({run['era']}, "
                f"lr={run['learning_rate']:g}"
                # eta is only worth the width where it is not the default.
                + (f", $\\eta$={run['reward_eta']:g}"
                   if run["reward_eta"] != REWARD_ETA else "")
                + ")"
            ),
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
    if starred:
        axes.annotate(
            "*  treinou contra o próprio oponente do diagnóstico",
            xy=(0.015, 0.975), xycoords="axes fraction", va="top", fontsize=9,
            color="#444444",
        )
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
    window = SMOOTHING_WINDOW
    for field, title, panel in fields:
        for run in runs:
            if not run["complete"] or not run["metrics"]:
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
                label=(
                    f"{run['label']} ({run['era']}"
                    + (f", $\\eta$={run['reward_eta']:g}"
                       if run["reward_eta"] != REWARD_ETA else "")
                    + ")"
                ),
            )
        panel.set_title(title)
        panel.set_xlabel("iteração de PPO")
        panel.grid(alpha=0.25)
    axes[0][0].axhline(0.0, color="#222222", linewidth=0.9, linestyle=":")
    axes[0][1].axhline(50.0, color="#222222", linewidth=0.9, linestyle=":")
    # Lower right: the upper left of that panel is where the shortest runs
    # end, and a legend there hides them completely.
    axes[0][0].legend(fontsize=8, loc="lower right")
    figure.suptitle(
        "Dinâmica de treino sob cada arquitetura (média móvel de 101 iterações)",
        fontweight="bold",
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_live_mixture(runs: list[dict[str, Any]], balance: dict[str, float],
                      path: Path) -> None:
    """Plot the terminal/local balance every current-era run logged itself.

    The offline recomputation on the fixed corpus and the live columns are two
    independent measurements of the same quantity: one scores a frozen set of
    422,055 decisions, the other reads what each run credited during its own
    rollouts. Drawing them together is the check that the corpus estimate is
    not an artifact of the policy that produced the corpus.
    """
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13.0, 5.4), constrained_layout=True
    )
    window = SMOOTHING_WINDOW
    plotted = 0
    for run in runs:
        series = mixture_series(run["metrics"])
        if series is None or not run["complete"]:
            continue
        plotted += 1
        terminal, local = series
        kernel = np.ones(window) / window
        x = np.arange(len(terminal) - window + 1) + window // 2
        smooth_terminal = np.convolve(terminal, kernel, mode="valid")
        smooth_local = np.convolve(local, kernel, mode="valid")
        color = RUN_COLORS[run["label"]]
        width = 2.4 if run["label"] in EMPHASIZED_RUNS else 1.3
        left.plot(x, smooth_local, "-", color=color, linewidth=width,
                  label=f"{run['label']}: $|\\eta G_I|$")
        left.plot(x, smooth_terminal, ":", color=color, linewidth=width)
        right.plot(x, smooth_local / smooth_terminal, "-", color=color,
                   linewidth=width,
                   label=f"{run['label']} ($\\eta$={run['reward_eta']:g})")
    left.set_title(
        "Magnitude de cada metade\n"
        "(linha cheia: local; pontilhada: terminal)"
    )
    left.set_xlabel("iteração de PPO")
    left.set_ylabel("magnitude média por decisão")
    left.set_ylim(bottom=0.0)
    left.grid(alpha=0.25)
    # The empty band between the two families of curves: the eta runs sit far
    # below the rest, so the middle of the panel is the only clear space.
    left.legend(fontsize=7.5, loc="center", bbox_to_anchor=(0.66, 0.42),
                framealpha=0.92)

    right.axhline(balance["current_ratio"], color="#99000d", linewidth=1.2,
                  linestyle="--")
    right.annotate(
        f"corpus recomputado, atual: {balance['current_ratio']:.2f}x",
        xy=(0.02, balance["current_ratio"]), xycoords=("axes fraction", "data"),
        xytext=(0, -6), textcoords="offset points", va="top", fontsize=8.5,
        color="#99000d",
    )
    right.axhline(balance["legacy_ratio"], color="#08519c", linewidth=1.2,
                  linestyle="--")
    right.annotate(
        f"corpus recomputado, anterior: {balance['legacy_ratio']:.2f}x",
        xy=(0.98, balance["legacy_ratio"]), xycoords=("axes fraction", "data"),
        xytext=(0, 5), textcoords="offset points", ha="right", fontsize=8.5,
        color="#08519c",
    )
    right.axhline(1.0, color="#222222", linewidth=0.9, linestyle=":")
    right.set_ylim(0.0, 3.0)
    right.set_title("Razão local/terminal medida pela própria execução")
    right.set_xlabel("iteração de PPO")
    right.set_ylabel("$|\\eta G_I| \\, / \\, |(1-\\eta) \\gamma_f^k U_T|$")
    right.grid(alpha=0.25)
    right.legend(fontsize=8.5, loc="lower right")
    figure.suptitle(
        f"Equilíbrio efetivo registrado ao vivo pelas {plotted} execuções da "
        "arquitetura atual (média móvel de 101 iterações)",
        fontweight="bold",
    )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def summary_color(row: dict[str, Any]) -> str:
    """Bar colour for a run: its architecture, unless it moved eta."""
    if row["eta"] != REWARD_ETA:
        return CORRECTED_COLOR
    return COLORS[row["arquitetura"]]


def plot_summary(summaries: list[dict[str, Any]], path: Path,
                 common_games: float) -> None:
    """Rank every completed run twice: at the shared budget, and at its own end.

    The pale bar is each run's best over its whole life, which is the number
    the earlier revisions of this report ranked on. It rewards a run for
    having been left training longer, so once runs of very different ages sit
    on the same axis it stops being a comparison. The solid bar is the best
    each run had reached by the horizon they all crossed, which is the same
    statistic measured on the same budget. Where the two orders disagree, the
    solid one is the one that answers the question.
    """
    rows = [row for row in summaries if row["completa"] == "sim"]
    rows.sort(key=lambda row: row["melhor_em_partidas_comuns_pct"])
    figure, axes = plt.subplots(figsize=(11.5, 6.8))
    positions = np.arange(len(rows), dtype=float)
    matched = [row["melhor_em_partidas_comuns_pct"] for row in rows]
    lifetime = [row["vitoria_melhor_pct"] for row in rows]
    errors = [wilson_halfwidth(value) for value in matched]
    colors = [summary_color(row) for row in rows]
    axes.barh(positions + 0.20, lifetime, color=colors, height=0.32, alpha=0.30)
    bars = axes.barh(positions - 0.20, matched, xerr=errors, color=colors,
                     height=0.32,
                     error_kw={"ecolor": "#333333", "capsize": 3,
                               "linewidth": 1.0})
    for bar, row in zip(bars, rows):
        if row["treina_no_avaliador"] == "sim":
            bar.set_hatch("//")
            bar.set_edgecolor("white")
    for position, row, error in zip(positions, rows, errors):
        # Anchor past the error bar so the label never sits on top of it.
        axes.annotate(
            f"{row['melhor_em_partidas_comuns_pct']:.2f}%  (lr={row['lr']:g}, "
            f"\u03b7={row['eta']:g})",
            (row["melhor_em_partidas_comuns_pct"] + error, position - 0.20),
            xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5,
        )
        axes.annotate(
            f"{row['vitoria_melhor_pct']:.2f}%  "
            f"({row['partidas'] / 1e6:.1f} M partidas)",
            (row["vitoria_melhor_pct"], position + 0.20),
            xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5,
            color="#666666",
        )
    axes.set_yticks(positions)
    axes.set_yticklabels([
        row["execucao"] + (" *" if row["treina_no_avaliador"] == "sim" else "")
        for row in rows
    ])
    axes.set_ylim(-0.75, len(rows) - 0.25)
    # Room on the right for both value labels and the legend, which would
    # otherwise land on the longest of them.
    axes.set_xlim(60.0, 70.8)
    axes.set_xlabel("melhor taxa de vitória contra o aleatório (%), IC 95%")
    axes.axvline(
        summaries[0]["vitoria_inicial_pct"], color="#666666", linestyle=":",
        linewidth=1.0,
    )
    axes.set_title(
        "Melhor resultado de cada execução double-six, no orçamento comum "
        f"de {common_games / 1e6:.1f} M partidas"
    )
    axes.grid(alpha=0.25, axis="x")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[LEGACY]),
        plt.Rectangle((0, 0), 1, 1, color=COLORS[CURRENT]),
        plt.Rectangle((0, 0), 1, 1, color=CORRECTED_COLOR),
        plt.Rectangle((0, 0), 1, 1, color="#999999", alpha=0.30),
    ]
    axes.legend(
        handles,
        ("recompensa anterior", "recompensa atual ($\\eta = 0{,}5$)",
         "recompensa atual ($\\eta = 0{,}115$)",
         "barra clara: execução inteira"),
        fontsize=9, loc="lower right",
    )
    if any(row["treina_no_avaliador"] == "sim" for row in rows):
        # Below the axes: every bar starts at the left spine, so there is no
        # clear space inside the plot for it.
        figure.text(
            0.5, -0.02, "*  treinou contra o próprio oponente do diagnóstico",
            ha="center", fontsize=9, color="#444444",
        )
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
                 balance, pair_rows, common_games, equivalent_eta,
                 trajectory_rows, eta_pair_rows) -> None:
    by_label = {row["execucao"]: row for row in summaries}
    controlled_legacy = by_label[CONTROLLED_PAIR[0]]
    controlled_current = by_label[CONTROLLED_PAIR[1]]
    controlled_gap = (
        controlled_current["vitoria_melhor_pct"]
        - controlled_legacy["vitoria_melhor_pct"]
    )
    controlled_half = wilson_halfwidth(controlled_legacy["vitoria_melhor_pct"])
    # Only completed runs: an interrupted run's median is taken over a few
    # hundred iterations and does not describe a training trajectory.
    measured_mixture = [
        row for row in summaries
        if row["razao_local_terminal"] != "" and row["completa"] == "sim"
    ]
    ratios_measured = [row["razao_local_terminal"] for row in measured_mixture]
    # Runs left at the reference eta measure the imbalance as shipped; runs
    # that lowered eta measure what lowering it did. Averaging the two
    # together would report a range that no run ever occupied.
    reference_mixture = [
        row for row in measured_mixture if row["eta"] == REWARD_ETA
    ]
    corrected_mixture = [
        row for row in measured_mixture if row["eta"] != REWARD_ETA
    ]
    reference_ratios = [row["razao_local_terminal"] for row in reference_mixture]
    corrected_ratios = [row["razao_local_terminal"] for row in corrected_mixture]

    def correlations(rows) -> list[float]:
        """The per-iteration reward/win correlation, where a run logged one."""
        return [
            row["corr_recompensa_vitoria"] for row in rows
            if row["corr_recompensa_vitoria"] != ""
        ]

    legacy_corr = correlations(
        [row for row in summaries if row["arquitetura"] == LEGACY]
    )
    reference_corr = correlations(reference_mixture)
    corrected_corr = correlations(corrected_mixture)
    comparable = [
        row for row in summaries
        if row["completa"] == "sim" and row["treina_no_avaliador"] == "nao"
    ]
    held_out = [
        row for row in summaries
        if row["completa"] == "sim" and row["treina_no_avaliador"] == "sim"
    ]
    external = [row for row in summaries if row["origem"] == "externa"]
    # The bundle carries the win-rate curve; whether it also carried the
    # per-iteration log is what decides which columns a received run can fill.
    external_bare = [
        row for row in external if row["razao_local_terminal"] == ""
    ]
    external_full = [
        row for row in external if row["razao_local_terminal"] != ""
    ]
    # Among the runs that trained on the evaluator, the ones left at the
    # reference eta are the reproducibility check; the ones that moved it are
    # the eta experiment. Only the first group may be read as a repeat.
    reference_random = [row for row in held_out if row["eta"] == REWARD_ETA]
    corrected_random = [row for row in held_out if row["eta"] != REWARD_ETA]
    # One row per distinct supervised binary, keeping the first run that used
    # it: the point is how far apart the starting policies are, not how many
    # runs each one seeded.
    sl_rows = list({row["pesos_sl"]: row for row in reversed(summaries)}.values())
    sl_rows.sort(key=lambda row: row["vitoria_inicial_pct"])
    sl_hashes = {row["pesos_sl"] for row in summaries}
    starts = [row["vitoria_inicial_pct"] for row in summaries]
    sl_spread = max(starts) - min(starts)
    rick_heuristic = by_label["rick_heuristic"]
    held_out_pcts = " e ".join(
        f"{row['vitoria_melhor_pct']:.3f}%" for row in
        sorted(held_out, key=lambda row: row["vitoria_melhor_pct"])
    )
    reference_random_gpus = " e ".join(
        sorted({row["gpu"] for row in reference_random})
    )
    reference_random_spread = (
        max(row["vitoria_melhor_pct"] for row in reference_random)
        - min(row["vitoria_melhor_pct"] for row in reference_random)
    )
    ranked_legacy = sorted(
        (row for row in comparable if row["arquitetura"] == LEGACY),
        key=lambda row: -row["vitoria_melhor_pct"],
    )
    ranked_current = sorted(
        (row for row in comparable if row["arquitetura"] == CURRENT),
        key=lambda row: -row["vitoria_melhor_pct"],
    )
    worst_legacy_low = ranked_legacy[-1]["vitoria_melhor_pct"] - wilson_halfwidth(
        ranked_legacy[-1]["vitoria_melhor_pct"]
    )
    best_current_high = ranked_current[0]["vitoria_melhor_pct"] + wilson_halfwidth(
        ranked_current[0]["vitoria_melhor_pct"]
    )
    # The same two rankings read at the budget every run reached, which is
    # the only ordering that stays meaningful once runs of very different ages
    # share the table.
    matched_legacy = sorted(
        (row for row in comparable if row["arquitetura"] == LEGACY),
        key=lambda row: -row["melhor_em_partidas_comuns_pct"],
    )
    matched_current = sorted(
        (row for row in comparable if row["arquitetura"] == CURRENT),
        key=lambda row: -row["melhor_em_partidas_comuns_pct"],
    )
    matched_gap = (
        matched_legacy[-1]["melhor_em_partidas_comuns_pct"]
        - matched_current[0]["melhor_em_partidas_comuns_pct"]
    )
    legacy_best = ranked_legacy[0]
    current_best = ranked_current[0]
    gap = legacy_best["vitoria_melhor_pct"] - current_best["vitoria_melhor_pct"]
    half = wilson_halfwidth(legacy_best["vitoria_melhor_pct"])

    eta_gains = sorted(
        by_label[corrected]["melhor_em_partidas_comuns_pct"]
        - by_label[reference]["melhor_em_partidas_comuns_pct"]
        for reference, corrected in ETA_PAIRS
    )
    eta_gain, eta_gain_other = eta_gains[0], eta_gains[-1]

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
        f"A queda é real, maior que o ruído do painel de diagnóstico, e **a "
        f"recompensa é a causa**. A execução `{controlled_current['execucao']}` "
        f"fecha a lacuna que as versões anteriores deste relatório não "
        f"conseguiam fechar: ela usa a recompensa atual com "
        f"**lr = {controlled_current['lr']:g}**, os mesmos buckets "
        f"`{controlled_current['buckets']}`, a mesma semente e os mesmos pesos "
        f"supervisionados de `{controlled_legacy['execucao']}`. Chegou a "
        f"**{controlled_current['vitoria_melhor_pct']:.3f}%** contra o "
        f"oponente aleatório, contra "
        f"**{controlled_legacy['vitoria_melhor_pct']:.3f}%** da execução "
        f"equivalente sob a recompensa anterior: "
        f"**{controlled_gap:+.3f} pontos percentuais**, cerca de "
        f"{abs(controlled_gap) / controlled_half:.1f}x a meia-largura do "
        f"intervalo de 95% do painel de {DIAGNOSTIC_GAMES:,} partidas "
        f"(±{controlled_half:.2f} pp).",
        "",
        "Igualar a taxa de aprendizado **recuperou parte** da diferença, e só "
        "parte. O par que isola a lr limpo é "
        f"`{by_label['default_lr016_lookup']['execucao']}` / "
        f"`{controlled_current['execucao']}`, que compartilham recompensa, "
        f"buckets e baseline `{controlled_current['baseline']}` e diferem só "
        f"na lr ({by_label['default_lr016_lookup']['lr']:g} contra "
        f"{controlled_current['lr']:g}): "
        f"{by_label['default_lr016_lookup']['vitoria_melhor_pct']:.3f}% para "
        f"{controlled_current['vitoria_melhor_pct']:.3f}%, "
        f"{controlled_current['vitoria_melhor_pct'] - by_label['default_lr016_lookup']['vitoria_melhor_pct']:+.3f} pp. "
        f"Contra `{by_label['default_lr032']['execucao']}` a diferença é "
        f"{controlled_current['vitoria_melhor_pct'] - by_label['default_lr032']['vitoria_melhor_pct']:+.3f} pp, "
        "mas ali o baseline também muda. Em qualquer das duas leituras, os "
        f"{abs(controlled_gap):.3f} pp que separam a recompensa atual da "
        "anterior não têm mais a lr como explicação possível.",
        "",
        "A causa provável dentro da recompensa não é o novo formato da "
        "utilidade terminal, e sim o **deslocamento do equilíbrio entre a "
        "metade terminal e a metade local** que o novo formato provocou sem "
        "que `reward_eta` mudasse. Recomputando as duas recompensas sobre as "
        f"**mesmas {len(scored['won']):,} decisões reais**, a metade local "
        f"passou de {balance['legacy_ratio']:.2f}x para "
        f"{balance['current_ratio']:.2f}x a magnitude da metade terminal — um "
        f"fator de {balance['current_ratio'] / balance['legacy_ratio']:.1f} no "
        "peso efetivo do termo de moldagem, com `reward_eta` fixo em 0,5 nas "
        f"duas. As próprias execuções da arquitetura atual confirmam o número "
        f"ao vivo: a razão mediana registrada por "
        f"`{controlled_current['execucao']}` é "
        f"{controlled_current['razao_local_terminal']:.2f}x.",
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
        "O veredito não depende desta máquina. Uma execução treinada em "
        "outro computador, com outro binário supervisionado e lr = 0.002, sob "
        "a recompensa atual e o bucket `heuristic`, chegou a "
        f"{by_label['rick_heuristic']['vitoria_melhor_pct']:.3f}% contra "
        f"{by_label['bucket_heuristic']['vitoria_melhor_pct']:.3f}% da "
        "execução equivalente sob a recompensa anterior — "
        f"{by_label['rick_heuristic']['vitoria_melhor_pct'] - by_label['bucket_heuristic']['vitoria_melhor_pct']:+.3f} pp, "
        "a mesma direção e praticamente o mesmo tamanho.",
        "",
        "Uma segunda conclusão sai do mesmo experimento, e é uma **correção** "
        "do que as versões anteriores deste relatório sugeriam: a anomalia de "
        "entropia e a pressão sobre a região de confiança do PPO **eram a taxa "
        f"de aprendizado, não a recompensa**. Com lr = "
        f"{controlled_current['lr']:g}, `{controlled_current['execucao']}` "
        "volta a ter entropia monotonicamente decrescente, KL mediana de "
        f"{controlled_current['kl_mediana']:.5f} e fração de recorte de "
        f"{controlled_current['recorte_mediano']:.4f} — os mesmos valores das "
        "execuções antigas. Os dois efeitos que antes apareciam juntos agora "
        "estão separados: **a lr explica a dinâmica, a recompensa explica o "
        "resultado**.",
        "",
        "**E a correção proposta foi testada.** As versões anteriores deste "
        f"relatório terminavam prevendo que `reward_eta ≈ {equivalent_eta:.3f}` "
        "devolveria à metade local o peso que ela tinha antes. Duas execuções "
        f"com `reward_eta = {corrected_random[0]['eta']:g}` — uma aqui, outra "
        "na máquina do orientador — confirmam a previsão em três medidas "
        "independentes: "
        f"registraram ao vivo a razão local/terminal em "
        f"{min(corrected_ratios):.3f}x, contra os "
        f"{balance['legacy_ratio']:.2f}x da recompensa anterior; a correlação "
        "por iteração entre recompensa e vitória subiu de "
        f"{min(reference_corr):+.3f}–{max(reference_corr):+.3f} para "
        f"{min(corrected_corr):+.3f}–{max(corrected_corr):+.3f}, de volta à "
        f"faixa antiga ({min(legacy_corr):+.3f}–{max(legacy_corr):+.3f}); e "
        f"ficaram {eta_gain:+.3f} pp e {eta_gain_other:+.3f} pp acima das "
        "execuções gêmeas que mantiveram eta = 0,5, medidas no mesmo "
        f"orçamento de {common_games / 1e6:.1f} M de partidas. O detalhe que "
        "impede de declarar o problema resolvido é que essas "
        f"{NUMBER_WORDS.get(len(held_out), len(held_out))} execuções treinam "
        "com o bucket `random`, o mesmo oponente do diagnóstico: o contraste "
        "entre elas é limpo, o nível absoluto não é comparável com o bloco da "
        "recompensa anterior. A seção "
        "[A correção de eta, executada](#a-correção-de-eta-executada) trata "
        "disso, e o experimento que falta está no fim dela.",
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
        "As execuções da arquitetura atual medem esse mesmo desequilíbrio "
        "**sozinhas**, sem nenhuma recomputação offline: a versão 8 do "
        "esquema de métricas registra `terminal_abs_mean` e `local_abs_mean`, "
        "que são exatamente `E|(1-eta) G_T|` e `E|eta G_I|` como o rollout os "
        "creditou. A figura `09_mistura_ao_vivo.png` traz as duas séries.",
        "",
    ])
    lines.extend(format_table(
        ["Execução", "lr", "eta", "magnitude terminal", "magnitude local",
         "razão local/terminal"],
        [[
            row["execucao"], f"{row['lr']:g}", f"{row['eta']:g}",
            f"{row['terminal_abs_mediana']:.4f}",
            f"{row['local_abs_mediana']:.4f}",
            f"{row['razao_local_terminal']:.2f}x",
        ] for row in measured_mixture],
    ))
    lines.extend([
        "",
        f"As {NUMBER_WORDS.get(len(reference_mixture), len(reference_mixture))} "
        f"execuções que mantiveram eta = {comma(REWARD_ETA)} concordam entre si "
        f"dentro de {min(reference_ratios):.2f}x–{max(reference_ratios):.2f}x "
        f"e concordam com os {balance['current_ratio']:.2f}x recomputados "
        "sobre o corpus fixo, que vem de outra política e de outro conjunto "
        "de partidas. São duas medições independentes do mesmo número, o que "
        "descarta a possibilidade de o desequilíbrio ser um artefato da "
        "política que gerou o corpus. A razão também **não depende da taxa "
        "de aprendizado**: é uma propriedade da função de recompensa — as "
        "quatro cobrem lr de 0.001 a 0.032 e medem o mesmo valor.",
        "",
        f"As {NUMBER_WORDS.get(len(corrected_mixture), len(corrected_mixture))} "
        f"últimas linhas são a verificação da correção proposta adiante: com "
        f"eta = {comma(corrected_mixture[0]['eta'])} a mesma coluna registra "
        f"{min(corrected_ratios):.3f}x e {max(corrected_ratios):.3f}x, contra "
        f"os {balance['legacy_ratio']:.2f}x que a recompensa anterior tinha. "
        "A conta que produziu esse eta foi feita sobre o corpus recomputado; "
        "quem a confirma aqui é o próprio rollout, em outra política e em "
        "outro conjunto de partidas. **O controle funciona como previsto.**",
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
        f"{by_label['default_lr032']['corr_recompensa_vitoria']:+.3f}, "
        f"{by_label['d6_maxwr_lr032']['corr_recompensa_vitoria']:+.3f} e "
        f"{controlled_current['corr_recompensa_vitoria']:+.3f} nas novas — "
        f"incluindo `{controlled_current['execucao']}`, que usa a mesma lr das "
        "antigas —, e a `reward_mean` mediana sai de aproximadamente zero para "
        f"cerca de {by_label['default_lr032']['recompensa_media_mediana']:+.3f} "
        "com a taxa de vitória em lote parada em ~51%.",
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
        f"`{controlled_current['execucao']}` abre com exatamente os mesmos "
        "números da coluna da direita, porque na primeira iteração a política "
        "ainda é a supervisionada e a taxa de aprendizado não teve como "
        "importar. A divergência entre as duas execuções da arquitetura atual "
        "começa na iteração 2.",
        "",
        "## O que a dinâmica de treino mostra",
        "",
        "A figura `07_dinamica_de_treino.png` traz um efeito que as versões "
        "anteriores deste relatório não conseguiam atribuir, e que a execução "
        f"`{controlled_current['execucao']}` agora resolve: **a entropia da "
        "política deixava de cair de forma monótona**. Sob a recompensa "
        "anterior ela desce ao longo de todo o treino e termina no seu mínimo "
        f"({by_label['bucket_heuristic']['entropia_inicial']:.3f} para "
        f"{by_label['bucket_heuristic']['entropia_final']:.3f} em "
        "`bucket_heuristic`). Sob a atual **com lr alta** ela cai rápido, "
        f"atinge o mínimo de {by_label['d6_maxwr_lr032']['entropia_minima']:.3f} "
        f"já na iteração {by_label['d6_maxwr_lr032']['iteracao_da_minima']:,} e "
        "depois **volta a subir**, até "
        f"{by_label['d6_maxwr_lr032']['entropia_maxima_apos_minima']:.3f}.",
        "",
        "A pergunta em aberto era se isso vinha da recompensa ou da lr, já que "
        "as duas mudaram juntas. Vinha da lr. Com a recompensa atual e "
        f"lr = {controlled_current['lr']:g}, `{controlled_current['execucao']}` "
        "se comporta como as execuções antigas em todos os três indicadores:",
        "",
    ])
    lines.extend(format_table(
        ["Indicador", f"{controlled_legacy['execucao']} (anterior, lr "
         f"{controlled_legacy['lr']:g})",
         f"{controlled_current['execucao']} (atual, lr "
         f"{controlled_current['lr']:g})",
         f"default_lr032 (atual, lr {by_label['default_lr032']['lr']:g})"],
        [
            ["entropia final",
             f"{controlled_legacy['entropia_final']:.3f}",
             f"{controlled_current['entropia_final']:.3f}",
             f"{by_label['default_lr032']['entropia_final']:.3f}"],
            ["iteração da entropia mínima",
             f"{controlled_legacy['iteracao_da_minima']:,} de "
             f"{controlled_legacy['iteracoes']:,}",
             f"{controlled_current['iteracao_da_minima']:,} de "
             f"{controlled_current['iteracoes']:,}",
             f"{by_label['default_lr032']['iteracao_da_minima']:,} de "
             f"{by_label['default_lr032']['iteracoes']:,}"],
            ["KL mediana",
             f"{controlled_legacy['kl_mediana']:.5f}",
             f"{controlled_current['kl_mediana']:.5f}",
             f"{by_label['default_lr032']['kl_mediana']:.5f}"],
            ["fração recortada mediana",
             f"{controlled_legacy['recorte_mediano']:.4f}",
             f"{controlled_current['recorte_mediano']:.4f}",
             f"{by_label['default_lr032']['recorte_mediano']:.4f}"],
        ],
    ))
    lines.extend([
        "",
        "A entropia volta a ser monótona: o mínimo cai na iteração "
        f"{controlled_current['iteracao_da_minima']:,} de "
        f"{controlled_current['iteracoes']:,}, ou seja, no fim do treino, e "
        "não há repique. A KL e o recorte voltam aos valores das execuções "
        "antigas. O aumento de escala do retorno, que praticamente dobrou o "
        f"desvio ({legacy_align['std']:.3f} para {current_align['std']:.3f}), "
        "não foi suficiente por si só para pressionar a região de confiança "
        "quando a lr é baixa.",
        "",
        "O que **não** volta ao normal com lr baixa é o sinal em si: a "
        "correlação por iteração entre `reward_mean` e `batch_win_rate` fica "
        f"em {controlled_current['corr_recompensa_vitoria']:+.3f} em "
        f"`{controlled_current['execucao']}`, contra "
        f"{controlled_legacy['corr_recompensa_vitoria']:+.3f} em "
        f"`{controlled_legacy['execucao']}`, e a `reward_mean` mediana "
        f"permanece em {controlled_current['recompensa_media_mediana']:+.3f} "
        "com a taxa de vitória em lote em "
        f"{controlled_current['vitoria_em_lote_mediana_pct']:.2f}%. Esses são "
        "os indicadores que dependem da forma da recompensa, e eles não se "
        "movem com a lr.",
        "",
        "## Resultados por execução",
        "",
    ])
    lines.extend(format_table(
        ["Execução", "Arq.", "lr", "eta", "Baseline", "Buckets", "Partidas",
         "Melhor", "Partidas até o melhor", "Final",
         f"Melhor em {common_games / 1e6:.1f} M"],
        [[
            row["execucao"], row["arquitetura"], f"{row['lr']:g}",
            f"{row['eta']:g}",
            row["baseline"], row["buckets"], f"{row['partidas'] / 1e6:.1f} M",
            f"{row['vitoria_melhor_pct']:.3f}%",
            f"{row['partidas_ate_melhor'] / 1e6:.1f} M",
            f"{row['vitoria_final_pct']:.3f}%",
            f"**{row['melhor_em_partidas_comuns_pct']:.3f}%**"
            if row["melhor_em_partidas_comuns_pct"] != "" else "—",
        ] for row in summaries],
    ))
    lines.extend([
        "",
        f"Todas as execuções partem da mesma política supervisionada, que "
        f"vale {summaries[0]['vitoria_inicial_pct']:.3f}% contra o aleatório. "
        "A execução `d6_maxwr_lr016` foi interrompida com 0,7 M de partidas e "
        "não entra em nenhuma comparação.",
        "",
        "A coluna em negrito é a que compara. As execuções desta tabela têm "
        f"idades muito diferentes — de "
        f"{min(row['partidas'] for row in summaries if row['completa'] == 'sim') / 1e6:.1f} M "
        f"a {max(row['partidas'] for row in summaries) / 1e6:.1f} M de partidas "
        "— e o melhor de uma vida inteira premia quem foi deixado treinando "
        "por mais tempo, o que não é a variável em estudo. A última coluna é "
        "o melhor que cada execução já tinha alcançado com "
        f"{common_games / 1e6:.1f} M de partidas, o horizonte que **todas** "
        "atravessaram; é a mesma estatística, medida no mesmo orçamento. A "
        "figura `08_resumo_resultados.png` traz as duas leituras lado a lado: "
        "a barra cheia no horizonte comum, a barra clara na execução inteira.",
        "",
        f"O melhor resultado absoluto continua sendo o de "
        f"`{legacy_best['execucao']}` sob a recompensa anterior, "
        f"{legacy_best['vitoria_melhor_pct']:.3f}%, contra "
        f"{current_best['vitoria_melhor_pct']:.3f}% de "
        f"`{current_best['execucao']}` sob a atual ({gap:.3f} pp, "
        f"{gap / half:.1f}x a meia-largura do intervalo). Esses dois "
        "compartilham o bucket `heuristic` e o baseline, mas foram treinados "
        "em máquinas diferentes e com lr diferente; o par controlado abaixo é "
        "o que decide.",
        "",
        "A separação é completa, e a figura `08_resumo_resultados.png` mostra "
        f"isso de uma vez: as "
        f"{NUMBER_WORDS.get(len(ranked_legacy), len(ranked_legacy))} "
        "execuções completas sob a recompensa anterior ficam **todas** acima "
        f"das {NUMBER_WORDS.get(len(ranked_current), len(ranked_current))} "
        "sob a atual. A pior das antigas "
        f"(`{ranked_legacy[-1]['execucao']}`, "
        f"{ranked_legacy[-1]['vitoria_melhor_pct']:.3f}%) ainda supera a melhor "
        f"das novas (`{ranked_current[0]['execucao']}`, "
        f"{ranked_current[0]['vitoria_melhor_pct']:.3f}%), e dessa vez os "
        "intervalos de 95% nem se tocam "
        f"({worst_legacy_low:.3f}% contra {best_current_high:.3f}%). A "
        "ordenação também não é uma escala de taxa de aprendizado: as "
        f"{NUMBER_WORDS.get(len(ranked_current), len(ranked_current))} "
        f"execuções novas cobrem lr de {min(row['lr'] for row in ranked_current):g} "
        f"a {max(row['lr'] for row in ranked_current):g}, as duas que mais se "
        "aproximam do bloco antigo usam as duas lr mais baixas, e ainda assim "
        "param antes.",
        "",
        "A separação também não é um efeito do tempo de treino: ela sobrevive "
        f"ao corte em {common_games / 1e6:.1f} M de partidas. Ali a pior das "
        f"antigas (`{matched_legacy[-1]['execucao']}`, "
        f"{matched_legacy[-1]['melhor_em_partidas_comuns_pct']:.3f}%) ainda "
        f"supera a melhor das novas (`{matched_current[0]['execucao']}`, "
        f"{matched_current[0]['melhor_em_partidas_comuns_pct']:.3f}%) por "
        f"{matched_gap:.3f} pp, e os dois blocos continuam sem se "
        "interpenetrar.",
        "",
        "Essa contagem exclui as "
        f"{NUMBER_WORDS.get(len(held_out), len(held_out))} execuções que "
        "treinaram com o bucket `random`, discutidas na seção seguinte: elas "
        "treinam contra o mesmo oponente que o diagnóstico mede, então o "
        "número delas não é comparável com o das demais.",
        "",
        "### Pares com o mesmo conjunto de oponentes",
        "",
        "A última linha é a comparação controlada: as duas execuções "
        "compartilham buckets, taxa de aprendizado, semente e pesos "
        "supervisionados, e diferem na recompensa. A penúltima é a mesma "
        "comparação feita em outra máquina, com outra lr e outro binário "
        "supervisionado.",
        "",
    ])
    lines.extend(format_table(
        ["Buckets", "anterior", "atual", "Mesma lr", "Diferença"],
        pair_rows,
    ))
    lines.extend([
        "",
        "## As execuções recebidas de fora",
        "",
        f"{NUMBER_WORDS.get(len(external), len(external)).capitalize()} das "
        "execuções da tabela não foram treinadas nesta máquina. Elas chegaram "
        "como o pacote de quatro arquivos que trocamos para comparar modelos "
        "sem transferir o modelo inteiro: `run_config.json`, "
        "`periodic_diagnostics.jsonl`, `rl_vs_random_progress.csv` e o PNG do "
        "progresso. Esse pacote basta para tudo que este relatório mede por "
        "execução — a curva de vitória, o pico, a trajetória e os intervalos "
        "— porque a variável de desfecho sai inteira do CSV de progresso.",
        "",
        "O que os quatro arquivos não trazem é `training_metrics.jsonl`, o "
        "registro por iteração, que fica um nível acima deles na raiz do "
        f"diretório da execução. "
        f"{NUMBER_WORDS.get(len(external_bare), len(external_bare)).capitalize()} "
        "das recebidas vieram sem ele — "
        + ", ".join(f"`{row['execucao']}`" for row in external_bare)
        + " — e por isso têm as colunas de entropia, KL, recorte e mistura "
        "viva vazias em `resumo_execucoes.csv` e não aparecem nas figuras "
        "`07_dinamica_de_treino.png` e `09_mistura_ao_vivo.png`. "
        + ", ".join(f"`{row['execucao']}`" for row in external_full)
        + " veio com ele e entra em tudo, o que mostra que o arquivo é a "
        "única peça que faltava: uma execução recebida com os cinco arquivos "
        "é indistinguível de uma treinada aqui, para efeitos desta análise.",
        "",
        "### Os pesos supervisionados não são um confundidor",
        "",
        f"As {NUMBER_WORDS[len(sl_hashes)]} máquinas geraram cada uma o seu "
        "próprio `domino_sl_standard_seed42.npz`, e os quatro arquivos têm "
        f"sha256 diferentes ({', '.join(sorted(sl_hashes))}). Mesma semente, "
        "binários distintos: a ordem de acumulação em ponto flutuante muda "
        "com o número de trabalhadores e com a GPU. Isso poderia ser um "
        "confundidor sério, e o diagnóstico periódico resolve a dúvida sem "
        "custo, porque a primeira linha de cada curva mede exatamente essas "
        "políticas supervisionadas nas mesmas 100.000 partidas fixas:",
        "",
    ])
    lines.extend(format_table(
        ["Pesos supervisionados", "Máquina", "Execução que os usa",
         "Vitória da política inicial"],
        [[f"`{row['pesos_sl']}`", row["gpu"], f"`{row['execucao']}`",
          f"{row['vitoria_inicial_pct']:.3f}%"] for row in sl_rows],
    ))
    lines.extend([
        "",
        f"Os {NUMBER_WORDS[len(sl_hashes)]} pontos de partida caem dentro de "
        f"{sl_spread:.3f} pp uns dos outros, contra uma meia-largura de "
        f"±{wilson_halfwidth(62.6):.2f} pp no próprio diagnóstico. São o mesmo "
        "jogador para efeitos de medida. A diferença entre as arquiteturas de "
        "recompensa não pode ser atribuída ao ponto de partida.",
        "",
        "### A replicação",
        "",
        f"`{rick_heuristic['execucao']}` repete, em outra máquina e com lr "
        f"{rick_heuristic['lr']:g}, a mesma condição de "
        f"`{by_label['bucket_heuristic']['execucao']}`: bucket `heuristic`, "
        "semente 42, baseline batch-mean, double-six. O resultado é "
        f"{rick_heuristic['vitoria_melhor_pct']:.3f}% contra "
        f"{by_label['bucket_heuristic']['vitoria_melhor_pct']:.3f}%, uma "
        f"queda de {rick_heuristic['vitoria_melhor_pct'] - by_label['bucket_heuristic']['vitoria_melhor_pct']:+.3f} pp. "
        "É a mesma direção e praticamente o mesmo tamanho da queda do par "
        f"controlado ({controlled_gap:+.3f} pp), obtida com outro hardware, "
        "outro binário supervisionado e outra taxa de aprendizado. O efeito "
        "não é uma peculiaridade desta máquina.",
        "",
        "### As execuções contra o bucket `random`",
        "",
        f"{NUMBER_WORDS.get(len(held_out), len(held_out)).capitalize()} "
        "execuções da tabela — recebidas ou locais — treinam com o bucket "
        "`random`, isto é, contra o mesmo oponente que o diagnóstico usa para "
        f"medir. Elas chegam a {held_out_pcts} — acima de qualquer execução "
        "sob a recompensa atual, e acima até da pior execução sob a anterior. "
        "O número é real, mas mede outra coisa: treinar e avaliar contra a "
        "mesma política é otimizar diretamente a métrica, e o valor deixa de "
        "indicar força de jogo geral. Por isso elas estão marcadas com `*` na "
        "figura `08_resumo_resultados.png` e ficam fora do ordenamento entre "
        "arquiteturas.",
        "",
        "Elas não são inúteis: um viés que todas compartilham não atrapalha a "
        "comparação **entre** elas. É exatamente isso que sustenta o teste de "
        "eta da seção seguinte, em que os dois lados treinam contra o mesmo "
        "oponente e diferem só em eta.",
        "",
        f"E {NUMBER_WORDS.get(len(reference_random), len(reference_random))} "
        f"delas medem reprodutibilidade. Rodaram a mesma configuração, com "
        f"eta = {comma(REWARD_ETA)}, em GPUs diferentes "
        f"({reference_random_gpus}) "
        f"e terminaram a {reference_random_spread:.3f} pp uma da outra, "
        "dentro do ruído do diagnóstico. Duas máquinas independentes, o mesmo "
        "resultado.",
        "",
        "## Fatores de confusão",
        "",
        f"O par controlado "
        f"`{controlled_legacy['execucao']}` / "
        f"`{controlled_current['execucao']}` fecha os dois confundidores "
        "principais que as versões anteriores deste relatório listavam. "
        "Restam três, e vale registrar exatamente onde a comparação ainda é "
        "frágil:",
        "",
        "1. **O baseline de vantagem difere dentro do par controlado.** "
        f"`{controlled_legacy['execucao']}` usou o baseline padrão "
        f"(`{controlled_legacy['baseline']}`) e "
        f"`{controlled_current['execucao']}` usou "
        f"`{controlled_current['baseline']}`. É o único parâmetro de treino "
        "que ainda separa os dois. O tamanho desse efeito está medido em "
        "outra execução: `baseline_zero`, que é `bucket_heuristic_recent` com "
        "o baseline trocado por zero sob a mesma recompensa, chegou a "
        f"{by_label['baseline_zero']['vitoria_melhor_pct']:.3f}% contra "
        f"{controlled_legacy['vitoria_melhor_pct']:.3f}% — "
        f"{by_label['baseline_zero']['vitoria_melhor_pct'] - controlled_legacy['vitoria_melhor_pct']:+.3f} pp, "
        f"cerca de {abs(by_label['baseline_zero']['vitoria_melhor_pct'] - controlled_legacy['vitoria_melhor_pct']) / abs(controlled_gap):.0%} "
        "da lacuna que precisa ser explicada. É pequeno, mas não é zero, e um "
        "baseline diferente do avaliado ali não está medido.",
        "2. **Nenhuma repetição com semente diferente.** Toda execução aqui "
        "usa semente 42. As execuções recebidas atenuam isso em parte — a "
        "condição `heuristic` sob a recompensa atual foi reproduzida em outra "
        f"máquina, as duas execuções `random` com eta = {comma(REWARD_ETA)} "
        f"reproduziram uma à outra dentro de {reference_random_spread:.3f} pp, "
        f"e as duas com eta = {comma(corrected_random[0]['eta'])} concordaram "
        "dentro de "
        f"{abs(corrected_random[0]['melhor_em_partidas_comuns_pct'] - corrected_random[1]['melhor_em_partidas_comuns_pct']):.3f} pp "
        "no horizonte comum — mas variar a máquina não é o mesmo que variar a "
        "semente, e a dispersão entre sementes continua sem medida.",
        "3. **O corpus de recomputação vem de uma política só** — o "
        "checkpoint `double six 66p local.npz`, treinado sob a recompensa "
        "anterior, jogando contra o heurístico. As proporções de desfecho "
        "refletem essa política. As conclusões sobre *forma* e *escala* das "
        "duas funções não dependem disso; as proporções por classe de "
        "desfecho, sim.",
        "",
        "## O experimento que decidiu",
        "",
        "As versões anteriores deste relatório terminavam pedindo uma "
        "execução: recompensa atual, **lr = 0,001**, buckets "
        "`heuristic,recent`, semente 42, mesmos pesos supervisionados. Essa "
        f"execução existe e é `{controlled_current['execucao']}`, com "
        f"{controlled_current['partidas'] / 1e6:.1f} M de partidas e "
        f"{controlled_current['iteracoes']:,} iterações.",
        "",
        "O critério declarado na ocasião era: *se ficar perto de 66%, a "
        "recompensa não é a causa; se ficar perto de 65%, a recompensa é a "
        f"causa*. O melhor resultado foi "
        f"**{controlled_current['vitoria_melhor_pct']:.3f}%**, atingido com "
        f"{controlled_current['partidas_ate_melhor'] / 1e6:.1f} M de partidas, "
        f"e a curva termina em {controlled_current['vitoria_final_pct']:.3f}%. "
        "**A recompensa é a causa.**",
        "",
        "A curva fica abaixo da equivalente anterior em todo o percurso, e "
        "não apenas no pico, o que descarta a leitura de que seria só uma "
        "questão de mais partidas:",
        "",
    ])
    lines.extend(format_table(
        ["Partidas de RL",
         f"{controlled_legacy['execucao']} (anterior)",
         f"{controlled_current['execucao']} (atual)", "Diferença"],
        trajectory_rows,
    ))
    lines.extend([
        "",
        "Duas observações sobre como corrigir o desequilíbrio:",
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
        "## A correção de eta, executada",
        "",
        f"Essa previsão foi testada. "
        f"{NUMBER_WORDS.get(len(corrected_random), len(corrected_random)).capitalize()} "
        f"execuções repetiram a recompensa atual com "
        f"`reward_eta = {corrected_random[0]['eta']:g}`, o valor mais próximo "
        f"de {equivalent_eta:.3f} que foi de fato lançado, mantendo tudo o "
        "mais: mesma semente, mesma lr, mesmo baseline, mesmos buckets, mesma "
        "arquitetura de recompensa. Uma rodou nesta máquina, a outra na do "
        "orientador. São as execuções mais novas da tabela, e é por isso que "
        "a comparação abaixo é lida no horizonte comum de "
        f"{common_games / 1e6:.1f} M de partidas.",
        "",
        "**Primeiro, o alvo foi acertado.** As duas registraram ao vivo a "
        f"razão local/terminal em {min(corrected_ratios):.3f}x e "
        f"{max(corrected_ratios):.3f}x, contra os "
        f"{balance['legacy_ratio']:.2f}x da recompensa anterior e os "
        f"{min(reference_ratios):.2f}x–{max(reference_ratios):.2f}x das "
        f"execuções que ficaram em eta = {comma(REWARD_ETA)}. O valor foi "
        "calculado sobre o "
        "corpus recomputado e confirmado pelo rollout de duas execuções "
        "independentes: `reward_eta` é, de fato, o controle da magnitude "
        "relativa, e a conta que o dimensionou estava certa.",
        "",
        "**Segundo, o sinal voltou a informar sobre o resultado.** A "
        "correlação por iteração entre `reward_mean` e `batch_win_rate` mede "
        "o quanto a recompensa que o treino persegue tem a ver com ganhar a "
        f"partida. Ela vale {min(legacy_corr):+.3f} a {max(legacy_corr):+.3f} "
        "nas execuções da recompensa anterior e cai para "
        f"{min(reference_corr):+.3f} a {max(reference_corr):+.3f} nas da "
        f"atual com eta = {comma(REWARD_ETA)}. Com eta = "
        f"{comma(corrected_mixture[0]['eta'])} ela volta a "
        f"{min(corrected_corr):+.3f} e {max(corrected_corr):+.3f} — dentro "
        "da faixa antiga. Este número é lido dentro do próprio treino, sobre "
        "as partidas que a execução jogou, e **não passa pelo diagnóstico "
        "contra o aleatório**: a ressalva do bucket `random`, discutida "
        "adiante, não o afeta.",
        "",
        "**Terceiro, o resultado subiu.** Cada execução corrigida tem, entre "
        f"as que ficaram em eta = {comma(REWARD_ETA)}, uma contraparte que "
        "compartilha o "
        "bucket `random` e o resto da configuração:",
        "",
    ])
    lines.extend(format_table(
        [f"eta = {comma(REWARD_ETA)}",
         f"eta = {comma(corrected_random[0]['eta'])}",
         "Mesma máquina e mesmos pesos SL", "Diferença"],
        eta_pair_rows,
    ))
    lines.extend([
        "",
        "A primeira linha é a comparação controlada do eta: as duas execuções "
        "rodaram na mesma GPU, a partir do mesmo binário supervisionado, com "
        "a mesma semente, a mesma lr, o mesmo baseline e os mesmos buckets. "
        "**O único parâmetro diferente é `reward_eta`.** A segunda linha "
        "repete o contraste em outro par de máquinas.",
        "",
        "**Quarto, e é aqui que a leitura precisa de cuidado:** as "
        f"{NUMBER_WORDS.get(len(held_out), len(held_out))} execuções desse "
        "bloco treinam com o bucket `random`, o mesmo "
        "oponente que o diagnóstico mede. Isso infla o nível de todas elas e "
        "as mantém fora do ordenamento contra as execuções da recompensa "
        "anterior. O que **não** é inflado é a diferença *dentro* do bloco: "
        "os dois lados carregam o mesmo viés, então o ganho do eta baixo é "
        "medido limpo. A conclusão que se sustenta é que **`reward_eta` move "
        "o resultado na direção prevista**; não que a recompensa atual com "
        "eta corrigido já alcance a anterior, o que estas execuções não têm "
        "como mostrar.",
        "",
        "### O próximo experimento",
        "",
        f"Repetir `{controlled_current['execucao']}` trocando apenas "
        f"`reward_eta` de {REWARD_ETA:g} para "
        f"{corrected_random[0]['eta']:g} — a mesma correção já validada, "
        "agora com buckets que **não** contêm o oponente do diagnóstico, "
        "para que o número volte a ser comparável com o bloco da recompensa "
        "anterior:",
        "",
        "```bash",
        "python -u -m training.pipeline forever \\",
        "    --learning-rate 0.001 \\",
        "    --opponent-buckets heuristic,recent \\",
        f"    --reward-eta {corrected_random[0]['eta']:g} \\",
        "    --baseline lookup-table \\",
        "    --run-name recompensa_atual_eta_equivalente",
        "```",
        "",
        f"O par a bater é `{controlled_legacy['execucao']}`, "
        f"{controlled_legacy['melhor_em_partidas_comuns_pct']:.3f}% no "
        f"horizonte comum e {controlled_legacy['vitoria_melhor_pct']:.3f}% no "
        f"total, contra os {controlled_current['melhor_em_partidas_comuns_pct']:.3f}% "
        f"e {controlled_current['vitoria_melhor_pct']:.3f}% de "
        f"`{controlled_current['execucao']}`. Se essa execução fechar a "
        f"lacuna de {abs(controlled_gap):.3f} pp, o desequilíbrio entre as "
        "metades era a causa inteira e `reward_eta` a corrige. Se fechar "
        "apenas parte dela, o restante está na *forma* da utilidade terminal "
        "— provavelmente na perda da penalidade de pontos, que empurrava na "
        "direção que o diagnóstico contra o aleatório premia.",
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
    # eta used to belong in this invariant: while every run set it to 0.5 it
    # cancelled out of the comparison, which is what let the report attribute
    # the gap to the reward alone. The eta runs deliberately break that, so
    # eta moves out of the invariant and into a reported column, and the rest
    # of the mixing rule -- the discounts and how distance is counted -- is
    # what still has to be identical everywhere.
    shared = {
        (run["gamma_f"], run["gamma_i"], run["distance_mode"]) for run in runs
    }
    if len(shared) != 1:
        raise SystemExit(
            "The runs do not share gamma_f/gamma_i/distance mode, so the "
            f"reward comparison is not isolated: {shared}"
        )
    etas = {run["reward_eta"] for run in runs}
    if etas - {REWARD_ETA} and not (etas & {REWARD_ETA}):
        raise SystemExit(
            f"No run left at the reference eta = {REWARD_ETA}, so there is "
            "nothing for the eta runs to be compared against."
        )

    complete = [run for run in runs if run["complete"]]
    # The horizon every completed run reaches. Reading each run at its own end
    # would compare a 2.7 M-game run against a 28 M-game one; reading them all
    # here compares them on the budget they all actually spent.
    common_games = min(float(run["curve"]["rl_games"][-1]) for run in complete)
    summaries = [summarize_run(run, common_games) for run in runs]
    by_label = {row["execucao"]: row for row in summaries}
    by_directory = {run["label"]: run for run in runs}

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
        same_lr = left["lr"] == right["lr"]
        pair_rows.append([
            left["buckets"],
            f"{left['execucao']}: {left['vitoria_melhor_pct']:.3f}% "
            f"(lr={left['lr']:g})",
            f"{right['execucao']}: {right['vitoria_melhor_pct']:.3f}% "
            f"(lr={right['lr']:g})",
            "**sim**" if same_lr else "não",
            f"**{right['vitoria_melhor_pct'] - left['vitoria_melhor_pct']:+.3f} pp**",
        ])

    trajectory_rows = trajectory_comparison(
        by_directory[CONTROLLED_PAIR[0]], by_directory[CONTROLLED_PAIR[1]]
    )

    # The eta contrast is read at the shared horizon, because the corrected
    # runs are the youngest on the table and their own peaks would be
    # compared against three to nine times as much training.
    eta_pair_rows = []
    for reference_label, corrected_label in ETA_PAIRS:
        left = by_label[reference_label]
        right = by_label[corrected_label]
        same_machine = left["gpu"] == right["gpu"] and left["pesos_sl"] == right["pesos_sl"]
        eta_pair_rows.append([
            f"`{left['execucao']}`: "
            f"{left['melhor_em_partidas_comuns_pct']:.3f}%",
            f"`{right['execucao']}`: "
            f"{right['melhor_em_partidas_comuns_pct']:.3f}%",
            "**sim**" if same_machine else "não",
            f"**{right['melhor_em_partidas_comuns_pct'] - left['melhor_em_partidas_comuns_pct']:+.3f} pp**",
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
    plot_summary(summaries, HERE / "08_resumo_resultados.png",
                 common_games)
    plot_live_mixture(runs, balance, HERE / "09_mistura_ao_vivo.png")

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
                "controlled_pair": {
                    "legacy": CONTROLLED_PAIR[0],
                    "current": CONTROLLED_PAIR[1],
                    "shared": [
                        "ruleset", "seed", "supervised_weights",
                        "opponent_buckets", "learning_rate", "gamma_f",
                        "gamma_i", "reward_eta", "reward_distance_mode",
                    ],
                    "differs": ["reward_architecture", "advantage_baseline"],
                    "delta_pp": round(
                        by_label[CONTROLLED_PAIR[1]]["vitoria_melhor_pct"]
                        - by_label[CONTROLLED_PAIR[0]]["vitoria_melhor_pct"], 3
                    ),
                },
                "pairs": [
                    {
                        "buckets": by_label[legacy_label]["buckets"],
                        "legacy": legacy_label,
                        "current": current_label,
                        "same_learning_rate": (
                            by_label[legacy_label]["lr"]
                            == by_label[current_label]["lr"]
                        ),
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
                 balance, pair_rows, common_games, equivalent_eta,
                 trajectory_rows, eta_pair_rows)

    print(f"decisions scored: {len(scored['won']):,}")
    print(
        f"local/terminal balance: {balance['legacy_ratio']:.2f}x -> "
        f"{balance['current_ratio']:.2f}x"
    )
    print(
        f"corr(G, win): {legacy_align['correlation_with_win']:+.4f} -> "
        f"{current_align['correlation_with_win']:+.4f}"
    )
    print("wrote REPORT.md, analysis_summary.json, 4 CSVs and 9 figures")


if __name__ == "__main__":
    main()
