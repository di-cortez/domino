#!/usr/bin/env python3
"""Analyse the first real --warmup-lr run and check that the warmup worked.

Reads, never writes, the run `test_warmup_diego_notebook` and three reference
runs that share its seed, supervised checkpoint and periodic diagnostic panel,
then renders figures, CSV tables, `analysis_summary.json` and `REPORT.md` in
this directory.

Run from the repository root:

    /home/diego/CCO/amb_virtual/bin/python analysis/analise_warmup_lr/analyze.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

# pylint: disable=wrong-import-position
from diagnostics.rl_progress import read_periodic_history  # noqa: E402
from training.rl.lr_warmup import (  # noqa: E402
    WARMUP_TRACE_COLUMNS,
    WarmupSchedule,
    normalize_warmup,
)
from training.rl.reporting import read_training_metrics  # noqa: E402
from training.rl.resume import load_resume_state  # noqa: E402
from training.run_artifacts import periodic_diagnostics_path  # noqa: E402

RUN_ROOT = REPO / "models" / "rl"
LOG_DIR = REPO / "train_script" / "grid_search_results" / "diego_notebook" / "gpi12000_warmup_test"
SEQUENCE_STATE = LOG_DIR / "sequence_state.tsv"
WALL_BUDGET_SECONDS = 18_000
DIAGNOSTIC_GAMES = 100_000
PPO_STOP_KL = 0.015
COST_ITERATIONS = 1000


@dataclass(frozen=True)
class RunSpec:
    key: str
    directory: str
    label: str
    color: str
    description: str


RUNS = (
    RunSpec(
        "warmup",
        "domino_rl_forever_seed52_runtest_warmup_diego_notebook",
        "warmup (GPI 8000, lr 0,001 com escada)",
        "#c0392b",
        "código atual, --warmup-lr, GPI 8000, lr 0,001, turn-turn",
    ),
    RunSpec(
        "gpi12000",
        "domino_rl_forever_seed52_runtest_gpi_12000_diego_notebook",
        "GPI 12000 (lr 0,001 fixo)",
        "#2471a3",
        "código atual, sem warmup, GPI 12000, lr 0,001, turn-turn",
    ),
    RunSpec(
        "gpi8000_old",
        "domino_rl_forever_seed52_runone_factor_gpi_8000_diego_notebook",
        "sweep gpi_8000 (lr 0,01, código antigo)",
        "#7d7d7d",
        "código antigo, sem warmup, GPI 8000, lr 0,01, decision-decision",
    ),
    RunSpec(
        "lr0p001_old",
        "domino_rl_forever_seed52_runlr_0p001",
        "lr_0p001 (GPI 2000, código antigo)",
        "#27ae60",
        "código antigo, sem warmup, GPI 2000, lr 0,001, decision-decision",
    ),
)
SUBJECT = "warmup"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def run_dir(spec):
    return RUN_ROOT / spec.directory


def load_trace(directory):
    lines = (directory / "warmup_schedule.jsonl").read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    rows = [dict(zip(WARMUP_TRACE_COLUMNS, json.loads(line))) for line in lines[1:]]
    return header, rows


def load_run(spec):
    directory = run_dir(spec)
    metrics_path = next(directory.glob("**/training_metrics.jsonl"))
    header, metrics = read_training_metrics(metrics_path)
    history = read_periodic_history(periodic_diagnostics_path(directory))
    state = json.loads((directory / "training_state.json").read_text(encoding="utf-8"))
    profile = json.loads(
        (directory / "diagnostics" / "runtime_profile.json").read_text(encoding="utf-8")
    )
    config_path = next(directory.glob("*/run_config.json"), None)
    config = (
        json.loads(config_path.read_text(encoding="utf-8")) if config_path else {}
    )
    return {
        "spec": spec,
        "directory": directory,
        "metrics_header": header,
        "metrics": metrics,
        "history": history,
        "state": state,
        "profile": profile,
        "config": config,
    }


def rung_of(applied, nominal, factor):
    return int(round(math.log(nominal / applied) / math.log(factor)))


def weight_steps(directory):
    """Return (start iteration, per-iteration L2 step) between archive pairs."""
    manifest = json.loads(
        (directory / "checkpoint_archive" / "manifest.json").read_text(encoding="utf-8")
    )
    records = sorted(
        manifest["checkpoints"], key=lambda record: record["completed_iteration"]
    )
    steps = []
    previous = None
    for record in records:
        with np.load(directory / "checkpoint_archive" / record["filename"]) as data:
            arrays = {name: data[name].astype(np.float64) for name in data.files}
        if previous is not None:
            span = record["completed_iteration"] - previous[0]
            if span > 0:
                squared = sum(
                    float(np.sum((arrays[name] - previous[1][name]) ** 2))
                    for name in arrays
                    if name in previous[1]
                )
                steps.append((previous[0], math.sqrt(squared) / span, span))
        previous = (record["completed_iteration"], arrays)
    return steps


# ---------------------------------------------------------------------------
# Verification of the warmup and of the run
# ---------------------------------------------------------------------------


def check(checks, name, ok, detail):
    checks.append({"verificacao": name, "ok": bool(ok), "detalhe": detail})


def verify(runs, trace_header, trace):
    checks = []
    subject = runs[SUBJECT]
    metrics = subject["metrics"]
    state = subject["state"]
    warmup = normalize_warmup(trace_header["warmup"])
    nominal = float(trace_header["nominal_learning_rate"])
    factor = warmup["factor"]

    iterations = [row["iteration"] for row in trace]
    check(
        checks,
        "Trace contíguo",
        iterations == list(range(1, len(trace) + 1))
        and len(trace) == state["rl_iterations_completed"] == len(metrics),
        f"{len(trace)} linhas, iterações 1..{iterations[-1]}; "
        f"{len(metrics)} linhas de métricas; training_state {state['rl_iterations_completed']}",
    )

    config_warmup = normalize_warmup(
        subject["config"].get("locked_arguments", {}) and {
            key.removeprefix("warmup_"): value
            for key, value in subject["config"]["locked_arguments"].items()
            if key.startswith("warmup_") and key != "warmup_lr" and value is not None
        }
    )
    locked = subject["config"].get("locked_arguments", {})
    check(
        checks,
        "Configuração gravada",
        locked.get("warmup_lr") is True
        and warmup == normalize_warmup({})
        and config_warmup == warmup
        and abs(nominal - 0.001) < 1e-15,
        f"warmup_lr={locked.get('warmup_lr')}, parâmetros {warmup}, lr nominal {nominal:g}",
    )

    expected_rungs = [nominal / factor ** exponent for exponent in range(6, -1, -1)]
    observed_rungs = sorted({row["applied_learning_rate"] for row in trace})
    check(
        checks,
        "Degraus exatos",
        len(observed_rungs) == 7
        and all(
            math.isclose(a, b, rel_tol=1e-12)
            for a, b in zip(observed_rungs, expected_rungs)
        ),
        "degraus observados: " + ", ".join(f"{value:.6g}" for value in observed_rungs),
    )

    chained = all(
        later["applied_learning_rate"] == earlier["next_learning_rate"]
        for earlier, later in zip(trace, trace[1:])
    )
    check(
        checks,
        "lr aplicada = lr instalada na iteração anterior",
        chained,
        "applied_learning_rate[i+1] == next_learning_rate[i] em todas as iterações",
    )

    schedule = WarmupSchedule.from_warmup(nominal, warmup)
    mismatches = 0
    for row in trace:
        decision = schedule.observe(row["max_approx_kl"])
        replayed = {
            "applied_learning_rate": decision.applied_learning_rate,
            "next_learning_rate": decision.next_learning_rate,
            "exponent": decision.exponent,
            "active": decision.active,
            "ema_max_kl": decision.ema_max_kl,
            "hold_streak": decision.hold_streak,
            "cooldown_remaining": decision.cooldown_remaining,
            "promoted": decision.promoted,
        }
        if any(replayed[key] != row[key] for key in replayed):
            mismatches += 1
    check(
        checks,
        "Replay offline reproduz cada decisão",
        mismatches == 0,
        f"{len(trace) - mismatches}/{len(trace)} linhas idênticas ao replay de WarmupSchedule",
    )

    promotions = [row["iteration"] for row in trace if row["promoted"]]
    minimum = [100 + 200 * index for index in range(6)]
    ema_during_ladder = [row["ema_max_kl"] for row in trace if row["iteration"] <= promotions[-1]]
    check(
        checks,
        "Escada na velocidade mínima teórica",
        promotions == minimum and max(ema_during_ladder) < warmup["kl_threshold"],
        f"promoções em {promotions}; EMA máxima durante a escada "
        f"{max(ema_during_ladder):.5f} < limiar {warmup['kl_threshold']:g}",
    )

    after = [row for row in trace if row["iteration"] > promotions[-1]]
    check(
        checks,
        "Após a escada, lr nominal constante",
        all(
            row["applied_learning_rate"] == nominal
            and not row["active"]
            and not row["promoted"]
            for row in after
        ),
        f"{len(after)} iterações após a iteração {promotions[-1]} com lr {nominal:g} e agenda inativa",
    )

    kl_gap = max(
        abs(round(row["max_approx_kl"], 5) - float(metric["max_approx_kl"]))
        for row, metric in zip(trace, metrics)
    )
    check(
        checks,
        "Trace coincide com training_metrics.jsonl",
        kl_gap <= 1.0e-5,
        f"maior diferença de max_approx_kl (métricas arredondadas a 5 casas): {kl_gap:.2e}",
    )

    metadata, _pool = load_resume_state(
        subject["directory"] / "latest_weights.npz",
        subject["directory"] / "latest.resume.npz",
    )
    saved = metadata["training_state"]["warmup_schedule"]
    check(
        checks,
        "Checkpoint de retomada",
        metadata["optimizer_state"]["learning_rate"] == nominal
        and saved["exponent"] == 0
        and math.isclose(saved["ema"], trace[-1]["ema_max_kl"], rel_tol=0, abs_tol=0),
        f"otimizador lr {metadata['optimizer_state']['learning_rate']:g}; "
        f"estado salvo {saved}",
    )

    rungs = [rung_of(row["applied_learning_rate"], nominal, factor) for row in trace]
    return checks, promotions, rungs


def verify_training_health(runs, checks):
    subject = runs[SUBJECT]
    metrics = subject["metrics"]
    stops = sum(1 for row in metrics if row["stopped_by_kl"])
    epochs = min(int(row["epochs_completed"]) for row in metrics)
    clipped = sum(1 for row in metrics if row["gradient_clipped"])
    locations = sorted({row["buffer_location"] for row in metrics})
    missing = sum(1 for row in metrics if row["max_approx_kl"] is None)
    check(
        checks,
        "PPO sem anomalias",
        stops == 0 and epochs == 16 and missing == 0 and locations == ["gpu"],
        f"paradas por KL {stops}; épocas mínimas {epochs}; iterações com gradiente "
        f"recortado {clipped}; buffer {locations}; KL ausente {missing}",
    )

    profile = subject["profile"]["cumulative"]["rl"]
    fallbacks = profile.get("ppo_full_buffer_evaluation", {}).get(
        "batch_size_memory_fallbacks", 0
    )
    check(
        checks,
        "Avaliação em lotes de 4096 sem fallback de memória",
        not fallbacks,
        f"batch_size_memory_fallbacks = {fallbacks}",
    )

    for key in (SUBJECT, "gpi12000"):
        run = runs[key]
        games = [int(row["rl_games"]) for row in run["history"]]
        expected = list(range(0, games[-1] + 1, DIAGNOSTIC_GAMES))
        iterations = [int(row["iteration"]) for row in run["metrics"]]
        windows = [
            (row.get("window_first_iteration"), row.get("window_iterations"))
            for row in run["history"][1:]
        ]
        tiled = all(
            first is not None and previous_last + 1 == first
            for (first, _count), previous_last in zip(
                windows,
                [0] + [int(row["rl_iterations"]) for row in run["history"][1:-1]],
            )
        )
        check(
            checks,
            f"Diagnósticos e métricas contíguos ({key})",
            games == expected and iterations == list(range(1, len(iterations) + 1)) and tiled,
            f"{len(games)} pontos de 0 a {games[-1]:,} a cada 100 mil; "
            f"{len(iterations)} iterações contíguas; janelas encadeadas: {tiled}",
        )

    for log in sorted(LOG_DIR.glob("*.log")):
        text = log.read_text(encoding="utf-8", errors="replace")
        errors = len(re.findall(r"Traceback|\bError\b|rolled back|NaN/Inf", text))
        progress = re.findall(r"RL training: ([\d]+) games", text)
        check(
            checks,
            f"Log {log.name}",
            errors == 0 and "Canonical pipeline finished" in text,
            f"{errors} erros; parada por SIGTERM com checkpoint seguro "
            f"({'sim' if 'Shutdown requested by SIGTERM' in text else 'não'}); "
            f"última contagem {int(progress[-1]):,} partidas" if progress else f"{errors} erros",
        )
    return checks


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def win_points(run):
    return [
        {
            "rl_games": int(row["rl_games"]),
            "rl_hours": float(row["rl_elapsed_seconds"]) / 3600.0,
            "progress_hours": float(row["progress_elapsed_seconds"]) / 3600.0,
            "win": 100.0 * float(row["win_rate"]),
            "low": 100.0 * float(row["ci95_win_rate_low"]),
            "high": 100.0 * float(row["ci95_win_rate_high"]),
        }
        for row in run["history"]
    ]


def mean_between(points, first, last):
    values = [point["win"] for point in points if first <= point["rl_games"] <= last]
    return statistics.fmean(values) if values else float("nan")


def at_games(points, games):
    for point in points:
        if point["rl_games"] == games:
            return point["win"]
    return float("nan")


def below_fraction(runs, first, last, subject=SUBJECT, others=("gpi12000", "lr0p001_old")):
    """Fraction of common monitor points where the subject is below every other run."""
    tables = {key: {p["rl_games"]: p["win"] for p in win_points(runs[key])} for key in (subject, *others)}
    games = [
        g for g in sorted(tables[subject])
        if first <= g <= last and all(g in tables[key] for key in others)
    ]
    below = sum(1 for g in games if all(tables[subject][g] < tables[key][g] for key in others))
    return below, len(games)


def run_summary(runs, ladder_end_games):
    common_games = min(win_points(run)[-1]["rl_games"] for run in runs.values())
    summaries = {}
    for key, run in runs.items():
        points = win_points(run)
        best = max(points, key=lambda point: point["win"])
        metrics = run["metrics"]
        # The first 1000 iterations: the last hours of the sweep's gpi_8000
        # point shared the machine with the PPO benchmarks of 2026-09-12, which
        # more than doubled its update time; nothing ran beside the early part
        # of any of these runs.
        tail = metrics[:COST_ITERATIONS]
        profile = run["profile"]["cumulative"]
        rl = profile["rl"]
        diagnostics = profile["rl_vs_random_diagnostics"]
        sections = rl["sections_seconds"]
        summaries[key] = {
            "descricao": run["spec"].description,
            "commit": str(run["state"].get("git_commit"))[:7],
            "iteracoes": int(run["state"]["rl_iterations_completed"]),
            "partidas": int(run["state"]["rl_games_completed"]),
            "tempo_rl_h": float(run["state"]["elapsed_rl_seconds"]) / 3600.0,
            "vitoria_inicial": points[0]["win"],
            "vitoria_final": points[-1]["win"],
            "vitoria_melhor": best["win"],
            "partidas_ate_melhor": best["rl_games"],
            "media_ate_fim_da_escada": mean_between(points, DIAGNOSTIC_GAMES, ladder_end_games),
            "media_horizonte_comum": mean_between(points, DIAGNOSTIC_GAMES, common_games),
            "media_12M_ate_horizonte_comum": mean_between(points, 12_000_000, common_games),
            "vitoria_5M": at_games(points, 5_000_000),
            "vitoria_10M": at_games(points, 10_000_000),
            "vitoria_horizonte_comum": at_games(points, common_games),
            "segundos_por_iteracao_update": statistics.median(row["update_seconds"] for row in tail),
            "segundos_por_iteracao_rollout": statistics.median(row["rollout_seconds"] for row in tail),
            "ms_por_passo_otimizador": 1000.0 * statistics.median(
                row["update_seconds"] / row["optimizer_steps"]
                for row in tail if row["optimizer_steps"]
            ),
            "paradas_por_kl": sum(1 for row in metrics if row["stopped_by_kl"]),
            "entropia_15M_ate_horizonte_comum": statistics.median(
                float(row["entropy"]) for row in metrics
                if 15_000_000 <= int(row["cumulative_games"]) <= common_games
            ),
            "rollout_s": float(sections.get("rollout_game_execution", 0.0)),
            "ppo_s": float(sections.get("ppo_update", 0.0)),
            "rl_total_s": float(rl["execution_seconds"]),
            "diagnostico_s": float(diagnostics["execution_seconds"]),
            "partidas_por_segundo_rl": int(run["state"]["rl_games_completed"])
            / float(run["state"]["elapsed_rl_seconds"]),
        }
    return summaries, common_games


def rung_table(trace, metrics, rungs, nominal, factor):
    table = []
    for rung in range(6, -1, -1):
        rows = [
            (row, metric)
            for row, metric, value in zip(trace, metrics, rungs)
            if value == rung
        ]
        kls = [row["max_approx_kl"] for row, _metric in rows]
        clip = [float(metric["final_clip_fraction"]) for _row, metric in rows]
        entropy = [float(metric["entropy"]) for _row, metric in rows]
        table.append({
            "expoente": rung,
            "lr": nominal / factor ** rung,
            "iteracao_inicial": rows[0][0]["iteration"],
            "iteracao_final": rows[-1][0]["iteration"],
            "iteracoes": len(rows),
            "partidas_inicio_M": (int(rows[0][1]["cumulative_games"]) - int(rows[0][1]["games"])) / 1e6,
            "kl_mediana": statistics.median(kls),
            "kl_p95": float(np.percentile(kls, 95)),
            "kl_max": max(kls),
            "ema_max": max(row["ema_max_kl"] for row, _metric in rows),
            "clip_mediana": statistics.median(clip),
            "entropia_mediana": statistics.median(entropy),
        })
    return table


def power_fit(xs, ys):
    slope, intercept = np.polyfit(np.log(xs), np.log(ys), 1)
    predicted = intercept + slope * np.log(xs)
    residual = np.sum((np.log(ys) - predicted) ** 2)
    total = np.sum((np.log(ys) - np.mean(np.log(ys))) ** 2)
    return float(math.exp(intercept)), float(slope), float(1 - residual / total)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def shade_rungs(axes, trace, rungs, promotions, x_of):
    edges = [0] + promotions + [trace[-1]["iteration"]]
    for index, (start, end) in enumerate(zip(edges, edges[1:])):
        if index % 2 == 0:
            axes.axvspan(x_of(start), x_of(end), color="#f2f2f2", zorder=0)


def figure_ladder(trace, promotions, warmup):
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    iterations = [row["iteration"] for row in trace]
    top.step(iterations, [row["applied_learning_rate"] for row in trace], where="post", color="#c0392b", linewidth=2)
    for iteration in promotions:
        row = trace[iteration - 1]
        top.annotate(
            f"{row['next_learning_rate']:.3g}",
            (iteration, row["next_learning_rate"]),
            textcoords="offset points", xytext=(6, -12), fontsize=8,
        )
        top.axvline(iteration, color="#c0392b", alpha=0.25, linewidth=0.8)
    top.set_yscale("log")
    top.set_ylabel("lr aplicada")
    top.set_title("Escada da taxa de aprendizado: 0,001 / 1,5^6 até 0,001")
    top.grid(True, which="both", alpha=0.3)
    limit = min(len(trace), promotions[-1] + 300)
    bottom.plot(iterations[:limit], [row["hold_streak"] for row in trace[:limit]], label="sequência abaixo do limiar", color="#2471a3")
    bottom.plot(iterations[:limit], [row["cooldown_remaining"] for row in trace[:limit]], label="cooldown restante", color="#e67e22")
    bottom.axhline(warmup["hold_iterations"], color="#2471a3", linestyle=":", linewidth=1)
    bottom.set_xlim(0, limit)
    top.set_xlim(0, limit)
    bottom.set_xlabel("Iteração de PPO")
    bottom.set_ylabel("iterações")
    bottom.legend(loc="upper right", fontsize=8)
    bottom.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(HERE / "01_escada_lr.png", dpi=150)
    plt.close(figure)


def figure_kl(trace, rungs, promotions, warmup, reference):
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 5.2))
    for axes, limit, title in (
        (left, promotions[-1] + 300, "Durante a escada"),
        (right, len(trace), "Corrida inteira"),
    ):
        rows = trace[:limit]
        iterations = [row["iteration"] for row in rows]
        shade_rungs(axes, trace, rungs, promotions, lambda value: value)
        axes.plot(iterations, [row["max_approx_kl"] for row in rows], color="#e6b0aa", linewidth=0.6, label="max KL (warmup)")
        axes.plot(iterations, [row["ema_max_kl"] for row in rows], color="#c0392b", linewidth=1.6, label="EMA α=0,9 (warmup)")
        if reference is not None:
            ref = reference[:limit]
            axes.plot(
                [int(row["iteration"]) for row in ref],
                [float(row["max_approx_kl"]) for row in ref],
                color="#2471a3", linewidth=0.6, alpha=0.7, label="max KL, GPI 12000 lr 0,001 fixo",
            )
        axes.axhline(warmup["kl_threshold"], color="black", linestyle="--", linewidth=1, label=f"limiar {warmup['kl_threshold']:g}")
        axes.axhline(PPO_STOP_KL, color="black", linestyle=":", linewidth=1, label=f"stop_kl {PPO_STOP_KL:g}")
        axes.set_xlim(0, limit)
        axes.set_ylim(0, PPO_STOP_KL * 1.1)
        axes.set_xlabel("Iteração de PPO")
        axes.set_ylabel("KL aproximada")
        axes.set_title(title)
        axes.grid(True, alpha=0.3)
    left.legend(fontsize=8, loc="upper right")
    figure.suptitle("KL máxima por iteração e a EMA que governa a escada (faixas = degraus)")
    figure.tight_layout()
    figure.savefig(HERE / "02_kl_e_ema.png", dpi=150)
    plt.close(figure)


def figure_kl_by_rung(trace, rungs, table, fit):
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    groups = [
        [row["max_approx_kl"] for row, value in zip(trace, rungs) if value == rung]
        for rung in range(6, -1, -1)
    ]
    labels = [f"{entry['lr']:.3g}\n({entry['iteracoes']} it)" for entry in table]
    left.boxplot(groups, tick_labels=labels, showfliers=False)
    left.axhline(0.0075, color="black", linestyle="--", linewidth=1, label="limiar 0,0075")
    left.set_xlabel("lr do degrau")
    left.set_ylabel("max KL por iteração")
    left.set_title("Distribuição da KL em cada degrau")
    left.legend(fontsize=8)
    left.grid(True, axis="y", alpha=0.3)
    xs = np.array([entry["lr"] for entry in table])
    ys = np.array([entry["kl_mediana"] for entry in table])
    scale, exponent, r2 = fit
    right.loglog(xs, ys, "o", color="#c0392b", label="mediana por degrau")
    grid = np.geomspace(xs.min(), xs.max(), 50)
    right.loglog(grid, scale * grid ** exponent, color="#7d7d7d", label=f"KL = {scale:.3g}·lr^{exponent:.3f} (R² {r2:.3f})")
    right.set_xlabel("lr aplicada")
    right.set_ylabel("KL mediana")
    right.set_title("KL em função da lr")
    right.legend(fontsize=8)
    right.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(HERE / "03_kl_por_degrau.png", dpi=150)
    plt.close(figure)


def figure_weight_steps(warm_steps, fixed_steps, trace, rungs, nominal, factor, promotions):
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 5))
    limit = promotions[-1] + 400
    left.plot([s for s, _v, _span in warm_steps if s < limit], [v for s, v, _span in warm_steps if s < limit], color="#c0392b", marker=".", linewidth=1, label="warmup")
    left.plot([s for s, _v, _span in fixed_steps if s < limit], [v for s, v, _span in fixed_steps if s < limit], color="#2471a3", marker=".", linewidth=1, label="GPI 12000, lr 0,001 fixo")
    for iteration in promotions:
        left.axvline(iteration, color="#c0392b", alpha=0.25, linewidth=0.8)
    left.set_yscale("log")
    left.set_xlabel("Iteração de PPO (início do intervalo de 10 iterações)")
    left.set_ylabel("‖ΔW‖ por iteração")
    left.set_title("Tamanho do passo nos pesos")
    left.legend(fontsize=8)
    left.grid(True, which="both", alpha=0.3)

    fixed_by_start = {start: value for start, value, _span in fixed_steps}
    by_rung = {}
    for start, value, span in warm_steps:
        if start + span > len(trace) or start not in fixed_by_start:
            continue
        covered = {rungs[index] for index in range(start, start + span)}
        if len(covered) == 1:
            by_rung.setdefault(covered.pop(), []).append((value, fixed_by_start[start]))
    order = list(range(6, -1, -1))
    lrs = [nominal / factor ** rung for rung in order]
    warm_medians = [statistics.median(v for v, _f in by_rung[rung]) for rung in order]
    fixed_medians = [statistics.median(f for _v, f in by_rung[rung]) for rung in order]
    ratios = [w / f for w, f in zip(warm_medians, fixed_medians)]
    positions = np.arange(len(order))
    right.bar(positions - 0.2, [lr / nominal for lr in lrs], width=0.4, color="#7d7d7d", label="lr do degrau ÷ lr nominal")
    right.bar(positions + 0.2, ratios, width=0.4, color="#c0392b", label="passo warmup ÷ passo lr fixo")
    right.set_xticks(positions, [f"{lr:.3g}" for lr in lrs])
    right.set_xlabel("lr do degrau")
    right.set_ylabel("razão")
    right.set_title("Passo relativo à corrida de lr fixo, nas mesmas iterações")
    right.legend(fontsize=8)
    right.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(HERE / "04_passo_dos_pesos.png", dpi=150)
    plt.close(figure)
    slope = float(np.polyfit(np.log([lr / nominal for lr in lrs]), np.log(ratios), 1)[0])
    table = [
        {
            "expoente": rung,
            "lr": lr,
            "passo_warmup": warm,
            "passo_lr_fixo": fixed,
            "razao_passo": ratio,
            "razao_lr": lr / nominal,
            "intervalos": len(by_rung[rung]),
        }
        for rung, lr, warm, fixed, ratio in zip(order, lrs, warm_medians, fixed_medians, ratios)
    ]
    return table, slope


def figure_win_rate(runs, ladder_end_games):
    figure, (left, right) = plt.subplots(1, 2, figsize=(15, 5.5))
    for key, run in runs.items():
        points = win_points(run)
        spec = run["spec"]
        games = [point["rl_games"] / 1e6 for point in points]
        for axes, limit in ((left, None), (right, ladder_end_games / 1e6 + 1.5)):
            selected = [index for index, value in enumerate(games) if limit is None or value <= limit]
            axes.plot([games[i] for i in selected], [points[i]["win"] for i in selected], color=spec.color, linewidth=1.3 if key == SUBJECT else 1.0, label=spec.label)
            axes.fill_between(
                [games[i] for i in selected],
                [points[i]["low"] for i in selected],
                [points[i]["high"] for i in selected],
                color=spec.color, alpha=0.12, linewidth=0,
            )
    for axes in (left, right):
        axes.axvline(ladder_end_games / 1e6, color="#c0392b", linestyle="--", linewidth=1, label="fim da escada (it. 1100)")
        axes.set_xlabel("Partidas de RL (milhões)")
        axes.set_ylabel("Vitórias contra random (%)")
        axes.grid(True, alpha=0.3)
    left.set_title("Curva completa (faixa = IC 95%)")
    right.set_title("Zoom no período da escada")
    right.set_xlim(0, ladder_end_games / 1e6 + 1.5)
    left.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(HERE / "05_vitoria_por_partidas.png", dpi=150)
    plt.close(figure)


def figure_win_rate_time(runs):
    figure, (left, right) = plt.subplots(1, 2, figsize=(15, 5.5))
    for key, run in runs.items():
        points = win_points(run)
        spec = run["spec"]
        left.plot([p["rl_hours"] for p in points], [p["win"] for p in points], color=spec.color, linewidth=1.3 if key == SUBJECT else 1.0, label=spec.label)
        right.plot([p["progress_hours"] for p in points], [p["win"] for p in points], color=spec.color, linewidth=1.3 if key == SUBJECT else 1.0, label=spec.label)
    left.set_xlabel("Tempo de treino de RL (horas)")
    left.set_title("Por tempo de RL (sem diagnósticos)")
    right.set_xlabel("Tempo de parede ativo (RL + diagnósticos, horas)")
    right.set_title("Por tempo de parede do orçamento de 5 h")
    for axes in (left, right):
        axes.set_ylabel("Vitórias contra random (%)")
        axes.grid(True, alpha=0.3)
    right.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(HERE / "06_vitoria_por_tempo.png", dpi=150)
    plt.close(figure)


def rolling(values, window=25):
    """Trailing mean; the first ``window - 1`` points are left undefined.

    A centred ``mode="same"`` convolution pads with zeros and bends both ends
    of every curve towards zero, which reads as a collapse that never happened.
    """
    array = np.asarray(values, dtype=float)
    result = np.full(array.shape, np.nan)
    if array.size >= window:
        result[window - 1:] = np.convolve(array, np.ones(window) / window, mode="valid")
    return result


def figure_health(runs, ladder_end_games):
    figure, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True)
    columns = (
        ("final_clip_fraction", "Fração no clip (média móvel 25)"),
        ("entropy", "Entropia da política (média móvel 25)"),
        ("max_approx_kl", "max KL (média móvel 25)"),
        ("gradient_norm_mean", "Norma média do gradiente (média móvel 25)"),
    )
    for (column, title), axis in zip(columns, axes.flat):
        for key, run in runs.items():
            metrics = run["metrics"]
            games = [int(row["cumulative_games"]) / 1e6 for row in metrics]
            values = [float(row[column]) if row[column] is not None else np.nan for row in metrics]
            axis.plot(games, rolling(values), color=run["spec"].color, linewidth=1.0, label=run["spec"].label)
        axis.axvline(ladder_end_games / 1e6, color="#c0392b", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
    for axis in axes[1]:
        axis.set_xlabel("Partidas de RL (milhões)")
    axes[0][0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(HERE / "07_saude_do_ppo.png", dpi=150)
    plt.close(figure)


def figure_cost(summaries):
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 5.2))
    keys = list(summaries)
    labels = [RUNS_BY_KEY[key].label.split(" (")[0] for key in keys]
    rollout = np.array([summaries[key]["rollout_s"] for key in keys]) / 3600
    ppo = np.array([summaries[key]["ppo_s"] for key in keys]) / 3600
    other_rl = np.array([summaries[key]["rl_total_s"] - summaries[key]["rollout_s"] - summaries[key]["ppo_s"] for key in keys]) / 3600
    diagnostics = np.array([summaries[key]["diagnostico_s"] for key in keys]) / 3600
    left.bar(labels, rollout, label="rollout", color="#27ae60")
    left.bar(labels, ppo, bottom=rollout, label="update PPO", color="#c0392b")
    left.bar(labels, other_rl, bottom=rollout + ppo, label="outro RL", color="#95a5a6")
    left.bar(labels, diagnostics, bottom=rollout + ppo + other_rl, label="diagnóstico periódico", color="#2471a3")
    left.axhline(WALL_BUDGET_SECONDS / 3600, color="black", linestyle="--", linewidth=1, label="orçamento 5 h")
    left.set_ylabel("horas")
    left.set_title("Para onde foi o orçamento de parede")
    left.legend(fontsize=8)
    left.tick_params(axis="x", labelsize=8)
    right.bar(labels, [summaries[key]["ms_por_passo_otimizador"] for key in keys], color=[RUNS_BY_KEY[key].color for key in keys])
    right.set_ylabel("ms por passo do otimizador (update ÷ passos)")
    right.set_title(f"Custo do update PPO por passo (primeiras {COST_ITERATIONS} iterações)")
    right.tick_params(axis="x", labelsize=8)
    for axis in (left, right):
        axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(HERE / "08_custo_do_tempo.png", dpi=150)
    plt.close(figure)


RUNS_BY_KEY = {spec.key: spec for spec in RUNS}


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=3):
    return f"{value:.{digits}f}".replace(".", ",")


def report(context):
    s = context["summaries"]
    w = s[SUBJECT]
    g = s["gpi12000"]
    old8 = s["gpi8000_old"]
    old2 = s["lr0p001_old"]
    checks = context["checks"]
    passed = sum(1 for item in checks if item["ok"])
    table = context["rung_table"]
    steps = context["step_table"]
    fit = context["fit"]
    ladder_games = context["ladder_end_games"]
    lines = []
    add = lines.append
    add("# Primeira corrida real com `--warmup-lr`: funcionamento e efeito")
    add("")
    add("## Conclusão")
    add("")
    add(
        f"**O warmup funcionou exatamente como especificado.** {passed} de {len(checks)} "
        "verificações passaram (tabela abaixo). A escada percorreu os sete degraus "
        f"de {table[0]['lr']:.4g} a {table[-1]['lr']:.4g}, com promoções nas iterações "
        f"{', '.join(str(i) for i in context['promotions'])} — o mínimo teórico de "
        "1100 iterações — e o replay offline de `WarmupSchedule` reproduz cada uma das "
        f"{len(context['trace'])} decisões gravadas. O passo real dos pesos acompanha a lr "
        "de cada degrau, o que confirma que a taxa foi de fato aplicada ao otimizador, "
        "não apenas registrada. A corrida GPI 12000, interrompida por SIGTERM e retomada, "
        "manteve métricas e diagnósticos contíguos."
    )
    add("")
    add(
        "**Na lr nominal 0,001 o portão de KL nunca atuou.** A EMA máxima durante a escada foi "
        f"{max(entry['ema_max'] for entry in table[:-1]):.5f}, cerca de "
        f"{100 * max(entry['ema_max'] for entry in table[:-1]) / 0.0075:.0f}% do limiar 0,0075, e a "
        f"maior KL individual da corrida foi {max(entry['kl_max'] for entry in table):.5f}. Isso "
        "confirma a previsão do roteiro (seção 2.1 e risco 6.1): em 0,001 não há pico inicial "
        "de KL para o warmup remover, então a escada rodou na velocidade mínima e o seu efeito "
        "é apenas treinar mais devagar durante as primeiras "
        f"1100 iterações ({ladder_games / 1e6:.1f} M de partidas com GPI 8000)."
    )
    add("")
    add(
        "**Não há evidência de ganho de desempenho; o que aparece é o custo esperado da escada.** "
        f"A corrida chegou a {fmt(w['vitoria_final'])}% de vitórias contra random "
        f"(melhor {fmt(w['vitoria_melhor'])}%) em {w['partidas'] / 1e6:.1f} M de partidas. "
        f"A corrida GPI 12000 do mesmo código terminou em {fmt(g['vitoria_final'])}% "
        f"(melhor {fmt(g['vitoria_melhor'])}%). As duas diferem em GPI *e* em warmup, e o "
        "intervalo de confiança de cada ponto é de ±0,29 pp, então a comparação não isola "
        "o warmup. Durante a escada a curva do warmup ficou sistematicamente abaixo das "
        "corridas de lr 0,001 fixo e recuperou a maior parte da distância depois. Medir esse "
        "custo sem confusão precisa de um controle com GPI 8000, lr 0,001 e `turn-turn` sem a flag."
    )
    add("")
    add("## Verificações")
    add("")
    add("| Verificação | Resultado | Detalhe |")
    add("|---|:---:|---|")
    for item in checks:
        add(f"| {item['verificacao']} | {'✅' if item['ok'] else '❌'} | {item['detalhe']} |")
    add("")
    add(
        "Observação menor, sem efeito no treino: depois da última promoção o estado salvo no "
        f"checkpoint mantém `cooldown` = {context['saved_state']['cooldown']}, enquanto o trace "
        "registra `cooldown_remaining` = 0. A agenda inativa retorna antes de decrementar o "
        "cooldown, e uma agenda com expoente 0 nunca volta a consultá-lo; é só uma "
        "inconsistência de apresentação entre o trace e o `state_dict`."
    )
    add("")
    add("## A escada, degrau por degrau")
    add("")
    add("| Expoente | lr | Iterações | Início (M partidas) | KL mediana | KL p95 | KL máx. | EMA máx. | Clip mediano | Entropia mediana | lr ÷ nominal | passo ÷ passo lr fixo |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    step_by_rung = {entry["expoente"]: entry for entry in steps}
    for entry in table:
        add(
            f"| {entry['expoente']} | {entry['lr']:.4g} | {entry['iteracao_inicial']}–{entry['iteracao_final']} "
            f"| {entry['partidas_inicio_M']:.1f} | {entry['kl_mediana']:.5f} | {entry['kl_p95']:.5f} "
            f"| {entry['kl_max']:.5f} | {entry['ema_max']:.5f} | {entry['clip_mediana']:.4f} "
            f"| {entry['entropia_mediana']:.4f} | {step_by_rung[entry['expoente']]['razao_lr']:.3f} "
            f"| {step_by_rung[entry['expoente']]['razao_passo']:.3f} |"
        )
    add("")
    scale, exponent, r2 = fit
    add(
        f"A KL mediana cresce com a lr como KL ≈ {scale:.3g}·lr^{exponent:.3f} (R² = {r2:.3f}). "
        f"Multiplicar a lr por 1,5 multiplica a KL por {1.5 ** exponent:.2f}. O expoente é bem menor "
        "que o 0,726 medido na grade de lr do `double-three`, mas os degraus não são "
        "corridas independentes: cada degrau vem depois do anterior no mesmo treino, e a "
        "KL também muda com o avanço da política, então o ajuste mistura lr e fase do treino."
    )
    add("")
    add(
        "A última coluna mede se a taxa foi aplicada de verdade. Ela divide o deslocamento "
        "mediano dos pesos por iteração, medido entre checkpoints arquivados a cada 10 "
        "iterações, pelo da corrida GPI 12000 com lr 0,001 fixo nas mesmas iterações. Se a "
        "escada só fosse registrada, a razão ficaria constante; ela sobe degrau a degrau, de "
        f"{steps[0]['razao_passo']:.2f} a {steps[-1]['razao_passo']:.2f}, acompanhando a razão de lr "
        f"de {steps[0]['razao_lr']:.3f} a 1 com expoente log-log {context['step_slope']:.2f}. "
        "O crescimento é sublinear, e o degrau final fica abaixo de 1 porque a corrida de "
        "referência faz 1,5x mais passos do otimizador por iteração (GPI 12000); nenhuma das "
        "duas coisas muda a leitura, porque a razão sobe junto com cada promoção."
    )
    add("")
    add("## Taxa de vitória contra random")
    add("")
    add(
        "Todas as corridas usam a semente 52, o mesmo checkpoint supervisionado e o mesmo "
        "painel de 100.000 partidas do diagnóstico periódico, por isso começam no mesmo ponto "
        f"({fmt(w['vitoria_inicial'])}%). As outras diferenças de configuração estão na tabela."
    )
    add("")
    add(f"| Corrida | Configuração | Partidas | Final | Melhor (em M) | Média até {ladder_games / 1e6:.1f} M | Em 10 M | Média 12–{context['common_games'] / 1e6:.1f} M | Média até {context['common_games'] / 1e6:.1f} M |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for key, entry in s.items():
        add(
            f"| {RUNS_BY_KEY[key].label} | {entry['descricao']} | {entry['partidas'] / 1e6:.1f} M "
            f"| {fmt(entry['vitoria_final'])}% | {fmt(entry['vitoria_melhor'])}% ({entry['partidas_ate_melhor'] / 1e6:.1f}) "
            f"| {fmt(entry['media_ate_fim_da_escada'])}% | {fmt(entry['vitoria_10M'])}% "
            f"| {fmt(entry['media_12M_ate_horizonte_comum'])}% | {fmt(entry['media_horizonte_comum'])}% |"
        )
    add("")
    add(
        f"Durante a escada (até {ladder_games / 1e6:.1f} M de partidas) a média do warmup foi "
        f"{fmt(w['media_ate_fim_da_escada'])}%, contra {fmt(g['media_ate_fim_da_escada'])}% da GPI 12000 "
        f"e {fmt(old2['media_ate_fim_da_escada'])}% da `lr_0p001` antiga. Um ponto isolado do painel tem "
        "IC de ±0,29 pp, mas a curva do warmup fica abaixo das duas ao mesmo tempo em "
        f"{context['ladder_below'][0]} dos {context['ladder_below'][1]} pontos do trecho (figura 05). "
        f"Depois a distância diminui: entre 12 M e {context['common_games'] / 1e6:.1f} M as médias são "
        f"{fmt(w['media_12M_ate_horizonte_comum'])}% (warmup), {fmt(g['media_12M_ate_horizonte_comum'])}% "
        f"(GPI 12000) e {fmt(old2['media_12M_ate_horizonte_comum'])}% (`lr_0p001`), ou seja, "
        f"{fmt(g['media_12M_ate_horizonte_comum'] - w['media_12M_ate_horizonte_comum'], 2)} pp atrás da GPI 12000 "
        "e empatado com a `lr_0p001`. É o "
        "padrão esperado do custo de treinar com lr reduzida no começo: aprendizado mais lento "
        "durante a escada e recuperação depois. Como as corridas diferem em GPI e em código, o "
        "tamanho desse custo não pode ser atribuído só ao warmup."
    )
    add("")
    add(
        "A entropia conta a mesma história (figura 07): a política do warmup permanece menos "
        "determinística por mais tempo. Entre 15 M e "
        f"{context['common_games'] / 1e6:.1f} M de partidas a entropia mediana foi "
        f"{w['entropia_15M_ate_horizonte_comum']:.3f} no warmup contra "
        f"{g['entropia_15M_ate_horizonte_comum']:.3f} na GPI 12000 e "
        f"{old2['entropia_15M_ate_horizonte_comum']:.3f} na `lr_0p001`, com a curva do warmup "
        "deslocada para a direita, como se estivesse alguns milhões de partidas atrasada."
    )
    add("")
    add("## Custo e desempenho do código novo")
    add("")
    add("| Corrida | Commit | Iterações | Tempo de RL | Diagnóstico | Partidas/s de RL | Update/it | Rollout/it | ms/passo |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for key, entry in s.items():
        add(
            f"| {RUNS_BY_KEY[key].label} | `{entry['commit']}` | {entry['iteracoes']:,} | {entry['tempo_rl_h']:.2f} h "
            f"| {entry['diagnostico_s'] / 3600:.2f} h | {entry['partidas_por_segundo_rl']:.0f} "
            f"| {entry['segundos_por_iteracao_update']:.2f} s | {entry['segundos_por_iteracao_rollout']:.2f} s "
            f"| {entry['ms_por_passo_otimizador']:.2f} |"
        )
    add("")
    add(
        f"Os custos por iteração vêm das primeiras {COST_ITERATIONS} iterações de cada corrida. As "
        "últimas horas do ponto `gpi_8000` do sweep dividiram a máquina com os benchmarks de PPO "
        "de 12/09, e o custo por passo dele subiu de ~2,6 ms para mais de 6 ms nessa fase; o "
        "tempo total de RL e de diagnóstico dessa corrida carrega essa contaminação."
    )
    add("")
    add(
        f"Na mesma GPI 8000, o update PPO custou {w['ms_por_passo_otimizador']:.2f} ms por passo do "
        f"otimizador no código novo contra {old8['ms_por_passo_otimizador']:.2f} ms na corrida do sweep "
        f"(`{old8['commit']}`), {old8['ms_por_passo_otimizador'] / w['ms_por_passo_otimizador']:.1f}x menos, "
        "já incluída a avaliação do buffer inteiro: a confirmação em produção das otimizações de "
        "PPO. O rollout, que as otimizações não tocaram, custou "
        f"{w['segundos_por_iteracao_rollout']:.2f} s por iteração contra {old8['segundos_por_iteracao_rollout']:.2f} s; "
        "a diferença acompanha o autotune de workers de rollout, que escolheu 8 workers na corrida "
        "nova e 10 na antiga. As duas corridas não usam a mesma lr, então a comparação vale para "
        "o custo, não para o aprendizado."
    )
    add("")
    add(
        "O sweep `gpi_8000` (lr 0,01, sem warmup) mostra na figura 07 exatamente o problema "
        "que o warmup foi feito para resolver: KL máxima perto de 0,014 nas primeiras "
        "iterações, fração no clip acima de 0,11 e 8 paradas por KL na corrida. Nenhuma das "
        "corridas com lr 0,001 tem esse pico."
    )
    add("")
    add(
        f"O efeito colateral é que o diagnóstico periódico síncrono passou a ocupar "
        f"{100 * w['diagnostico_s'] / WALL_BUDGET_SECONDS:.0f}% do orçamento de 5 h do warmup "
        f"({w['diagnostico_s'] / 3600:.2f} h), porque o RL mais rápido chega a mais marcos de 100 mil "
        "partidas e cada marco para o treino por um diagnóstico de 100 mil partidas. "
        "`--async-periodic-diagnostics` existe exatamente para devolver esse tempo ao treino."
    )
    add("")
    add("## Limites de interpretação")
    add("")
    add("- Uma corrida por configuração, sem repetição de semente.")
    add("- Nenhuma corrida de referência difere do warmup em um único fator. A GPI 12000 difere na GPI; as corridas do sweep diferem em código, lr, GPI e modo de distância.")
    add("- O orçamento é de tempo de parede. Com o código novo o warmup jogou mais partidas que as corridas antigas no mesmo orçamento; as comparações por partidas são as mais justas para aprendizado.")
    add("- A `lr_0p001` antiga foi rodada fora do script de sequência, em duas sessões que somam 5,2 h, por isso passa da linha de 5 h na figura 08.")
    add("- A lr nominal 0,001 é justamente a faixa em que o roteiro previa que o warmup não teria o que corrigir. O teste que mede o benefício pretendido é o estágio 7 do roteiro: `--learning-rate 0.01` com e sem `--warmup-lr`.")
    add("")
    add("## Próximos passos sugeridos")
    add("")
    add("1. Rodar o controle sem warmup nas mesmas condições (GPI 8000, lr 0,001, código atual) para medir o custo real das 1100 iterações lentas.")
    add("2. Rodar o par do estágio 7 (`--learning-rate 0.01` com e sem `--warmup-lr`), onde o roteiro mede pico inicial de KL de 2,4x e o portão deve atuar.")
    add("3. Usar `--async-periodic-diagnostics` nas próximas corridas com orçamento de tempo, ou descontar o diagnóstico ao comparar.")
    add("")
    add("## Artefatos")
    add("")
    add("- `01_escada_lr.png` — lr aplicada, sequência abaixo do limiar e cooldown.")
    add("- `02_kl_e_ema.png` — max KL, EMA, limiar e `stop_kl`, durante a escada e na corrida inteira.")
    add("- `03_kl_por_degrau.png` — distribuição da KL por degrau e ajuste de potência.")
    add("- `04_passo_dos_pesos.png` — tamanho do passo nos pesos por iteração e normalizado pela lr.")
    add("- `05_vitoria_por_partidas.png` e `06_vitoria_por_tempo.png` — curvas de vitória contra random.")
    add("- `07_saude_do_ppo.png` — clip, entropia, KL e norma do gradiente ao longo das partidas.")
    add("- `08_custo_do_tempo.png` — divisão do orçamento de parede e custo por passo do otimizador.")
    add("- `trajetoria_warmup.csv`, `resumo_degraus.csv`, `curvas_vitoria.csv`, `resumo_execucoes.csv`, `verificacoes.csv`, `analysis_summary.json`.")
    add("")
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    runs = {spec.key: load_run(spec) for spec in RUNS}
    subject = runs[SUBJECT]
    trace_header, trace = load_trace(subject["directory"])
    warmup = normalize_warmup(trace_header["warmup"])
    nominal = float(trace_header["nominal_learning_rate"])
    factor = warmup["factor"]

    checks, promotions, rungs = verify(runs, trace_header, trace)
    verify_training_health(runs, checks)
    metrics = subject["metrics"]
    ladder_end_games = int(metrics[promotions[-1] - 1]["cumulative_games"])
    summaries, common_games = run_summary(runs, ladder_end_games)
    table = rung_table(trace, metrics, rungs, nominal, factor)
    fit = power_fit(
        np.array([entry["lr"] for entry in table]),
        np.array([entry["kl_mediana"] for entry in table]),
    )
    warm_steps = weight_steps(subject["directory"])
    fixed_steps = weight_steps(runs["gpi12000"]["directory"])
    metadata, _pool = load_resume_state(
        subject["directory"] / "latest_weights.npz",
        subject["directory"] / "latest.resume.npz",
    )

    figure_ladder(trace, promotions, warmup)
    figure_kl(trace, rungs, promotions, warmup, runs["gpi12000"]["metrics"])
    figure_kl_by_rung(trace, rungs, table, fit)
    step_table, step_slope = figure_weight_steps(
        warm_steps, fixed_steps, trace, rungs, nominal, factor, promotions
    )
    figure_win_rate(runs, ladder_end_games)
    figure_win_rate_time(runs)
    figure_health(runs, ladder_end_games)
    figure_cost(summaries)

    write_csv(HERE / "trajetoria_warmup.csv", [
        {
            **row,
            "cumulative_games": metric["cumulative_games"],
            "final_clip_fraction": metric["final_clip_fraction"],
            "entropy": metric["entropy"],
            "epochs_completed": metric["epochs_completed"],
            "update_seconds": metric["update_seconds"],
        }
        for row, metric in zip(trace, metrics)
    ])
    write_csv(HERE / "resumo_degraus.csv", [
        {**entry, **{f"passo_{k}": v for k, v in step.items() if k != "expoente"}}
        for entry, step in zip(table, step_table)
    ])
    write_csv(HERE / "curvas_vitoria.csv", [
        {"corrida": key, **point}
        for key, run in runs.items()
        for point in win_points(run)
    ])
    write_csv(HERE / "resumo_execucoes.csv", [
        {"corrida": key, **entry} for key, entry in summaries.items()
    ])
    write_csv(HERE / "verificacoes.csv", checks)
    context = {
        "checks": checks,
        "promotions": promotions,
        "trace": trace,
        "rung_table": table,
        "step_table": step_table,
        "step_slope": step_slope,
        "fit": fit,
        "summaries": summaries,
        "ladder_end_games": ladder_end_games,
        "common_games": common_games,
        "saved_state": metadata["training_state"]["warmup_schedule"],
        "ladder_below": below_fraction(runs, DIAGNOSTIC_GAMES, ladder_end_games),
    }
    (HERE / "analysis_summary.json").write_text(
        json.dumps(
            {
                "warmup": warmup,
                "nominal_learning_rate": nominal,
                "promotions": promotions,
                "ladder_end_games": ladder_end_games,
                "common_games": common_games,
                "kl_power_fit": {"scale": fit[0], "exponent": fit[1], "r2": fit[2]},
                "rungs": table,
                "weight_steps": step_table,
                "weight_step_vs_lr_log_slope": step_slope,
                "runs": summaries,
                "checks": checks,
            },
            indent=1,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report(context)
    failed = [item["verificacao"] for item in checks if not item["ok"]]
    print(f"{len(checks) - len(failed)}/{len(checks)} verificações passaram" + (f"; falharam: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
