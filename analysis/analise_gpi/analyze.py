#!/usr/bin/env python3
"""Compare the GPI ladder 1000, 2000, 4000, 8000 and 12000 on the Diego notebook.

Reads, never writes, six runs under `models/rl/`: the four GPI points of the
one-factor sweep, the GPI 12000 test and the GPI 8000 control that ran on the
same code and defaults as that test. Writes `dados_pagina.json` (everything the
page draws), CSV tables and `analysis_summary.json` into this directory.

Run from the repository root:

    /home/diego/CCO/amb_virtual/bin/python analysis/analise_gpi/analyze.py
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

# pylint: disable=wrong-import-position
from diagnostics.rl_progress import read_periodic_history  # noqa: E402
from training.rl.reporting import read_training_metrics  # noqa: E402
from training.run_artifacts import periodic_diagnostics_path  # noqa: E402

RUN_ROOT = REPO / "models" / "rl"
SWEEP = RUN_ROOT / "run_one_factor_tests_diego_notebook"

# The professor's reference figures: a 5-point trailing rolling mean, the mean
# of the raw diagnostics in [16M, 17M] games and the maximum up to 17M games.
ROLLING_POINTS = 5
LATE_WINDOW = (16_000_000, 17_000_000)
COMMON_GAMES = 17_000_000
# Reading ruler of the one-factor sweep: 90th percentile of the differences
# inside its 22 replicate pairs (references/resumo_expandido/analises_agente_atual).
RULER_PP = 0.22
# Point-to-point oscillation is measured after the initial climb.
OSCILLATION_FROM_GAMES = 5_000_000


@dataclass(frozen=True)
class RunSpec:
    key: str
    gpi: int
    config: str  # "antiga" (one-factor sweep defaults) or "nova" (commit 36a8160 on)
    directory: Path


RUNS = (
    RunSpec("gpi1000", 1000, "antiga", SWEEP / "domino_rl_forever_seed52_runone_factor_gpi_1000_diego_notebook"),
    RunSpec("gpi2000", 2000, "antiga", SWEEP / "domino_rl_forever_seed52_runone_factor_control_diego_notebook"),
    RunSpec("gpi4000", 4000, "antiga", SWEEP / "domino_rl_forever_seed52_runone_factor_gpi_4000_diego_notebook"),
    RunSpec("gpi8000", 8000, "antiga", RUN_ROOT / "domino_rl_forever_seed52_runone_factor_gpi_8000_diego_notebook"),
    RunSpec("gpi8000_nova", 8000, "nova", RUN_ROOT / "domino_rl_forever_seed52_runtest_control_diego_notebook"),
    RunSpec("gpi12000_nova", 12000, "nova", RUN_ROOT / "domino_rl_forever_seed52_runtest_gpi_12000_diego_notebook"),
)
LADDER = ("gpi1000", "gpi2000", "gpi4000", "gpi8000")
PAIR = ("gpi8000_nova", "gpi12000_nova")

# Runs outside the GPI ladder, read only for the additivity check of the bridge.
BRIDGE_CHECK = {
    "lr_0p001": RUN_ROOT / "domino_rl_forever_seed52_runlr_0p001",
    "turn_turn": SWEEP / "domino_rl_forever_seed52_runone_factor_distance_turn_turn_diego_notebook",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_config(directory):
    bundle = next(directory.glob("2026*/run_config.json"))
    return json.loads(bundle.read_text(encoding="utf-8"))


def load_curve(directory):
    history = read_periodic_history(periodic_diagnostics_path(directory))
    games = np.array([record["rl_games"] for record in history], dtype=float)
    win_rate = np.array([record["win_rate"] * 100.0 for record in history])
    hours = np.array([record["rl_elapsed_seconds"] for record in history]) / 3600.0
    if np.any(np.diff(games) <= 0):
        raise ValueError(f"non-increasing diagnostic milestones in {directory}")
    return games, win_rate, hours


def trailing_mean(values, points=ROLLING_POINTS):
    out = np.empty_like(values)
    for index in range(len(values)):
        out[index] = values[max(0, index - points + 1): index + 1].mean()
    return out


def window_mean(games, win_rate, low, high):
    mask = (games >= low) & (games <= high)
    return float(win_rate[mask].mean()), int(mask.sum())


def area_mean(games, win_rate, horizon):
    mask = games <= horizon
    return float(np.trapezoid(win_rate[mask], games[mask]) / horizon)


def mechanism(directory, horizon=COMMON_GAMES):
    """PPO statistics over the iterations that end at or before `horizon`."""
    _, rows = read_training_metrics(directory / "training_metrics.jsonl")
    rows = [row for row in rows if row["cumulative_games"] <= horizon]

    def column(name):
        return np.array([np.nan if row[name] is None else row[name] for row in rows], dtype=float)

    games = column("games")
    full = games == games.max()
    kl = column("max_approx_kl")[full]
    total_games = games.sum()
    return {
        "updates": len(rows),
        "shortened_updates": int((~full).sum()),
        "mean_games_per_update": float(total_games / len(rows)),
        "optimizer_steps_per_million_games": float(column("optimizer_steps").sum() / (total_games / 1e6)),
        "optimizer_steps_per_update": float(np.median(column("optimizer_steps")[full])),
        "decisions_per_game": float(column("decisions").sum() / total_games),
        "kl_median": float(np.nanmedian(kl)),
        "kl_p90": float(np.nanpercentile(kl, 90)),
        "epochs_mean": float(np.nanmean(column("epochs_completed")[full])),
        "stopped_by_kl_percent": float(100.0 * np.nanmean(column("stopped_by_kl")[full])),
        "clip_fraction_median": float(np.nanmedian(column("final_clip_fraction")[full])),
        "entropy_first_20": float(np.nanmean(column("entropy")[:20])),
        "entropy_last_20": float(np.nanmean(column("entropy")[-20:])),
        "update_share_of_iteration_time": float(column("update_seconds").sum() / column("iteration_seconds").sum()),
    }


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def r_squared(y, fitted):
    return float(1.0 - ((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def fit_log_linear(gpis, values):
    x = np.log2(gpis)
    slope, intercept = np.polyfit(x, values, 1)
    return {
        "slope_pp_per_doubling": float(slope),
        "intercept": float(intercept),
        "r2": r_squared(values, intercept + slope * x),
    }


def fit_quadratic(gpis, values):
    x = np.log2(gpis)
    coefficients = np.polyfit(x, values, 2)
    vertex = -coefficients[1] / (2.0 * coefficients[0])
    return {
        "coefficients": [float(value) for value in coefficients],
        "vertex_gpi": float(2.0 ** vertex),
        "vertex_value": float(np.polyval(coefficients, vertex)),
        "r2": r_squared(values, np.polyval(coefficients, x)),
    }


def eval_log_linear(fit, gpi):
    return fit["intercept"] + fit["slope_pp_per_doubling"] * float(np.log2(gpi))


def eval_quadratic(fit, gpi):
    return float(np.polyval(fit["coefficients"], np.log2(gpi)))


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyse():
    runs = {}
    for spec in RUNS:
        config = load_config(spec.directory)
        state = json.loads((spec.directory / "training_state.json").read_text(encoding="utf-8"))
        rl_config = config["rl_config"]
        if rl_config["games_per_iteration"] != spec.gpi:
            raise ValueError(f"{spec.key}: GPI {rl_config['games_per_iteration']} != {spec.gpi}")
        games, win_rate, hours = load_curve(spec.directory)
        late, late_points = window_mean(games, win_rate, *LATE_WINDOW)
        settled = games >= OSCILLATION_FROM_GAMES
        runs[spec.key] = {
            "key": spec.key,
            "gpi": spec.gpi,
            "config": spec.config,
            "run_name": config["run_name"],
            "directory": str(spec.directory.relative_to(REPO)),
            "git_commit": config["git_commit"][:7],
            "learning_rate": rl_config["learning_rate"],
            "reward_distance_mode": rl_config["reward_distance_mode"],
            "games_total": int(state["rl_games_completed"]),
            "iterations_total": int(state["rl_iterations_completed"]),
            "rl_hours": float(state["elapsed_rl_seconds"] / 3600.0),
            "million_games_per_rl_hour": float(games[-1] / 1e6 / hours[-1]),
            "start_win_rate": float(win_rate[0]),
            "late_16_17": late,
            "late_16_17_points": late_points,
            "max_to_17": float(win_rate[games <= COMMON_GAMES].max()),
            "max_to_17_games": float(games[games <= COMMON_GAMES][np.argmax(win_rate[games <= COMMON_GAMES])]),
            "auc_to_17": area_mean(games, win_rate, COMMON_GAMES),
            "final_win_rate": float(win_rate[-1]),
            "oscillation_pp": float(np.diff(win_rate[settled]).std() / np.sqrt(2.0)),
            "mechanism": mechanism(spec.directory),
            "curve": {
                "games_millions": [round(value / 1e6, 1) for value in games],
                "win_rate": [round(float(value), 3) for value in win_rate],
                "rolling": [round(float(value), 4) for value in trailing_mean(win_rate)],
            },
            "_arrays": (games, win_rate),
        }

    reference = {
        key: {field: runs[key][field] for field in ("learning_rate", "reward_distance_mode")}
        for key in runs
    }
    ladder_gpis = np.array([runs[key]["gpi"] for key in LADDER], dtype=float)
    ladder_late = np.array([runs[key]["late_16_17"] for key in LADDER])
    ladder_max = np.array([runs[key]["max_to_17"] for key in LADDER])
    linear = fit_log_linear(ladder_gpis, ladder_late)
    quadratic = fit_quadratic(ladder_gpis, ladder_max)

    # Direct pair: same code, same defaults, only the GPI differs.
    (g8, w8), (g12, w12) = runs[PAIR[0]]["_arrays"], runs[PAIR[1]]["_arrays"]
    common_end = min(g8[-1], g12[-1])
    count = int((g8 <= common_end).sum())
    if not np.array_equal(g8[:count], g12[:count]):
        raise ValueError("the direct pair does not share its diagnostic milestones")
    end_window = (common_end - 1_000_000, common_end)
    pair = {
        "common_end_games": float(common_end),
        "late_16_17_delta": runs[PAIR[1]]["late_16_17"] - runs[PAIR[0]]["late_16_17"],
        "max_to_17_delta": runs[PAIR[1]]["max_to_17"] - runs[PAIR[0]]["max_to_17"],
        "auc_to_17_delta": runs[PAIR[1]]["auc_to_17"] - runs[PAIR[0]]["auc_to_17"],
        "end_window": [float(value) for value in end_window],
        "end_window_8000": window_mean(g8, w8, *end_window)[0],
        "end_window_12000": window_mean(g12, w12, *end_window)[0],
        "auc_to_end_8000": area_mean(g8, w8, common_end),
        "auc_to_end_12000": area_mean(g12, w12, common_end),
        "games_in_budget_ratio": runs[PAIR[1]]["games_total"] / runs[PAIR[0]]["games_total"],
        "difference_curve": {
            "games_millions": [round(value / 1e6, 1) for value in g8[:count]],
            "raw": [round(float(value), 3) for value in (w12[:count] - w8[:count])],
            "rolling": [round(float(value), 4) for value in trailing_mean(w12[:count] - w8[:count])],
        },
    }
    pair["end_window_delta"] = pair["end_window_12000"] - pair["end_window_8000"]
    pair["auc_to_end_delta"] = pair["auc_to_end_12000"] - pair["auc_to_end_8000"]
    settled = g8[:count] >= OSCILLATION_FROM_GAMES
    diff_raw = w12[:count] - w8[:count]
    diff_rolling = trailing_mean(diff_raw)
    pair["difference_mean_after_5M"] = float(diff_raw[settled].mean())
    pair["difference_points_after_5M"] = int(settled.sum())
    pair["difference_points_outside_ruler"] = int((np.abs(diff_raw[settled]) > RULER_PP).sum())
    pair["difference_share_inside_ruler"] = float(np.mean(np.abs(diff_raw[settled]) <= RULER_PP))
    pair["rolling_points_outside_ruler"] = int((np.abs(diff_rolling[g8[:count] >= 1_000_000]) > RULER_PP).sum())
    pair["rolling_abs_max_after_1M"] = float(np.abs(diff_rolling[g8[:count] >= 1_000_000]).max())

    # Bridge: 12000 placed on the old ladder through its GPI 8000 twin.
    bridged_late = runs["gpi8000"]["late_16_17"] + pair["late_16_17_delta"]
    bridged_max = runs["gpi8000"]["max_to_17"] + pair["max_to_17_delta"]
    five_gpis = np.append(ladder_gpis, 12000.0)
    bridge = {
        "late_16_17": bridged_late,
        "max_to_17": bridged_max,
        "linear_prediction_12000": eval_log_linear(linear, 12000),
        "quadratic_prediction_12000": eval_quadratic(quadratic, 12000),
        "five_point_linear_late": fit_log_linear(five_gpis, np.append(ladder_late, bridged_late)),
        "five_point_quadratic_max": fit_quadratic(five_gpis, np.append(ladder_max, bridged_max)),
    }

    # Additivity check: do the learning-rate and distance gains measured at
    # GPI 2000 stack onto GPI 8000 and predict its twin on the new defaults?
    control = runs["gpi2000"]
    deltas = {}
    for key, directory in BRIDGE_CHECK.items():
        games, win_rate, _ = load_curve(directory)
        deltas[key] = window_mean(games, win_rate, *LATE_WINDOW)[0] - control["late_16_17"]
    predicted_twin = runs["gpi8000"]["late_16_17"] + sum(deltas.values())
    additivity = {
        "delta_lr_0p001_at_gpi2000": deltas["lr_0p001"],
        "delta_turn_turn_at_gpi2000": deltas["turn_turn"],
        "predicted_gpi8000_nova": predicted_twin,
        "observed_gpi8000_nova": runs["gpi8000_nova"]["late_16_17"],
        "error": predicted_twin - runs["gpi8000_nova"]["late_16_17"],
    }

    ladder_gain = {
        key: runs[key]["late_16_17"] - runs["gpi8000"]["late_16_17"] for key in LADDER
    }

    # How often the smoothed ladder curves sit in GPI order, and when each
    # neighbouring pair last swapped, over the common 0-17M horizon.
    smoothed = np.vstack([trailing_mean(runs[key]["_arrays"][1][runs[key]["_arrays"][0] <= COMMON_GAMES]) for key in LADDER])
    milestones = runs[LADDER[0]]["_arrays"][0][runs[LADDER[0]]["_arrays"][0] <= COMMON_GAMES]
    after = milestones >= OSCILLATION_FROM_GAMES
    ladder_order = {
        "fully_ordered_share_after_5M": float(np.all(np.diff(smoothed[:, after], axis=0) > 0, axis=0).mean()),
        "last_swap_games": {},
    }
    for lower, upper in zip(LADDER, LADDER[1:]):
        i, j = LADDER.index(lower), LADDER.index(upper)
        swapped = milestones[smoothed[i] >= smoothed[j]]
        swapped = swapped[swapped >= 1_000_000]
        ladder_order["last_swap_games"][f"{lower}_{upper}"] = float(swapped.max()) if swapped.size else None
    for run in runs.values():
        del run["_arrays"]
    return {
        "settings": {
            "rolling_points": ROLLING_POINTS,
            "late_window": list(LATE_WINDOW),
            "common_games": COMMON_GAMES,
            "ruler_pp": RULER_PP,
            "oscillation_from_games": OSCILLATION_FROM_GAMES,
            "diagnostic_games": 100_000,
            "seed": 52,
            "ruleset": "double-six",
            "budget_hours": 5,
        },
        "reference_configs": reference,
        "runs": runs,
        "ladder": list(LADDER),
        "pair": list(PAIR),
        "fits": {"late_log_linear": linear, "max_quadratic": quadratic},
        "ladder_gain_vs_8000": ladder_gain,
        "ladder_order": ladder_order,
        "direct_pair": pair,
        "bridge": bridge,
        "additivity": additivity,
    }


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def write_outputs(result):
    (HERE / "dados_pagina.json").write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    summary = {key: value for key, value in result.items() if key != "runs"}
    summary["direct_pair"] = {
        key: value for key, value in result["direct_pair"].items() if key != "difference_curve"
    }
    summary["runs"] = {
        key: {field: value for field, value in run.items() if field != "curve"}
        for key, run in result["runs"].items()
    }
    (HERE / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (HERE / "curvas_vitoria.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["corrida", "gpi", "config", "partidas", "vitoria_pct", "media_movel_5_pct"])
        for run in result["runs"].values():
            curve = run["curve"]
            for games, win, rolling in zip(curve["games_millions"], curve["win_rate"], curve["rolling"]):
                writer.writerow([run["key"], run["gpi"], run["config"], int(round(games * 1e6)), win, rolling])

    fields = [
        "key", "gpi", "config", "git_commit", "learning_rate", "reward_distance_mode",
        "games_total", "iterations_total", "rl_hours", "million_games_per_rl_hour",
        "late_16_17", "max_to_17", "max_to_17_games", "auc_to_17", "final_win_rate", "oscillation_pp",
    ]
    mechanism_fields = list(next(iter(result["runs"].values()))["mechanism"])
    with (HERE / "resumo_gpi.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields + mechanism_fields)
        for run in result["runs"].values():
            writer.writerow([run[field] for field in fields] + [run["mechanism"][field] for field in mechanism_fields])


def main():
    result = analyse()
    write_outputs(result)
    fits = result["fits"]
    pair = result["direct_pair"]
    print(f"log-linear: {fits['late_log_linear']['slope_pp_per_doubling']:+.3f} pp/doubling, "
          f"R2={fits['late_log_linear']['r2']:.3f}")
    print(f"quadratic max vertex: {fits['max_quadratic']['vertex_gpi']:,.0f}")
    print(f"12000 - 8000 (new defaults): late {pair['late_16_17_delta']:+.3f} pp, "
          f"end window {pair['end_window_delta']:+.3f} pp, AUC {pair['auc_to_end_delta']:+.3f} pp")
    print(f"bridged 12000: late {result['bridge']['late_16_17']:.3f}, max {result['bridge']['max_to_17']:.3f}")
    print(f"additivity error: {result['additivity']['error']:+.3f} pp")


if __name__ == "__main__":
    main()
