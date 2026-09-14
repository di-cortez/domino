#!/usr/bin/env python3
"""Build the standalone HTML page from `analyze.py`'s `dados_pagina.json`.

Every number in the prose is a `{{token}}` filled from the analysis, and the
charts read the same JSON inlined into the page, so the published page needs
nothing from this repository at view time. Run `analyze.py` first.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def decimal(value, places):
    """Brazilian formatting: 1.234,567."""
    text = f"{abs(value):,.{places}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return ("−" if value < 0 else "") + text


def signed(value, places):
    return ("+" if value > 0 else "") + decimal(value, places) if round(value, places) else decimal(0.0, places)


def tokens(data):
    runs = data["runs"]
    fits = data["fits"]
    pair = data["direct_pair"]
    bridge = data["bridge"]
    additivity = data["additivity"]
    ladder = [runs[key] for key in data["ladder"]]
    old = [runs[key]["mechanism"] for key in data["ladder"]]
    new8, new12 = runs["gpi8000_nova"], runs["gpi12000_nova"]
    swaps = data["ladder_order"]["last_swap_games"]
    return {
        "slope": signed(fits["late_log_linear"]["slope_pp_per_doubling"], 3),
        "r2": decimal(fits["late_log_linear"]["r2"], 3),
        "vertex": decimal(fits["max_quadratic"]["vertex_gpi"], 0),
        "ruler": decimal(data["settings"]["ruler_pp"], 2),
        "gain_1k_8k": signed(-data["ladder_gain_vs_8000"]["gpi1000"], 3),
        "late_1k": decimal(ladder[0]["late_16_17"], 3),
        "late_2k": decimal(ladder[1]["late_16_17"], 3),
        "late_4k": decimal(ladder[2]["late_16_17"], 3),
        "late_8k": decimal(ladder[3]["late_16_17"], 3),
        "start": decimal(ladder[0]["start_win_rate"], 2),
        "raw_gap": signed(new12["late_16_17"] - runs["gpi8000"]["late_16_17"], 3),
        "pair_late": signed(pair["late_16_17_delta"], 3),
        "pair_end": signed(pair["end_window_delta"], 3),
        "pair_auc_end": signed(pair["auc_to_end_delta"], 3),
        "pair_diff_mean": signed(pair["difference_mean_after_5M"], 3),
        "pair_points": str(pair["difference_points_after_5M"]),
        "pair_inside_n": str(pair["difference_points_after_5M"] - pair["difference_points_outside_ruler"]),
        "roll_out": str(pair["rolling_points_outside_ruler"]),
        "roll_max": decimal(pair["rolling_abs_max_after_1M"], 3),
        "order_share": decimal(100 * data["ladder_order"]["fully_ordered_share_after_5M"], 0),
        "swap_12": decimal(swaps["gpi1000_gpi2000"] / 1e6, 1),
        "swap_24": decimal(swaps["gpi2000_gpi4000"] / 1e6, 1),
        "swap_48": decimal(swaps["gpi4000_gpi8000"] / 1e6, 1),
        "bridge_late": decimal(bridge["late_16_17"], 3),
        "bridge_max": decimal(bridge["max_to_17"], 3),
        "linear_pred": decimal(bridge["linear_prediction_12000"], 3),
        "quad_pred": decimal(bridge["quadratic_prediction_12000"], 3),
        "bridge_vs_linear": decimal(abs(bridge["late_16_17"] - bridge["linear_prediction_12000"]), 3),
        "slope5": signed(bridge["five_point_linear_late"]["slope_pp_per_doubling"], 3),
        "r2_5": decimal(bridge["five_point_linear_late"]["r2"], 3),
        "vertex5": decimal(bridge["five_point_quadratic_max"]["vertex_gpi"], 0),
        "osc_1k": decimal(ladder[0]["oscillation_pp"], 3),
        "osc_8k": decimal(ladder[3]["oscillation_pp"], 3),
        "osc_8k_nova": decimal(new8["oscillation_pp"], 3),
        "osc_12k": decimal(new12["oscillation_pp"], 3),
        "steps_lo": decimal(min(m["optimizer_steps_per_million_games"] for m in old) / 1000, 0),
        "steps_hi": decimal(max(m["optimizer_steps_per_million_games"] for m in old) / 1000, 0),
        "kl_lo": decimal(min(m["kl_median"] for m in old), 4),
        "kl_hi": decimal(max(m["kl_median"] for m in old), 4),
        "upd_ratio": decimal(old[0]["updates"] / old[3]["updates"], 1),
        "kl_ratio": decimal(old[3]["kl_median"] / new8["mechanism"]["kl_median"], 1),
        "kl_new": decimal(new8["mechanism"]["kl_median"], 4),
        "thr_lo": decimal(min(run["million_games_per_rl_hour"] for run in ladder), 1),
        "thr_hi": decimal(max(run["million_games_per_rl_hour"] for run in ladder), 1),
        "thr_ratio": decimal(100 * (new12["million_games_per_rl_hour"] / new8["million_games_per_rl_hour"] - 1), 0),
        "games_12k": decimal(new12["games_total"] / 1e6, 1),
        "games_8k": decimal(new8["games_total"] / 1e6, 1),
        "add_lr": signed(additivity["delta_lr_0p001_at_gpi2000"], 3),
        "add_tt": signed(additivity["delta_turn_turn_at_gpi2000"], 3),
        "add_pred": decimal(additivity["predicted_gpi8000_nova"], 3),
        "add_obs": decimal(additivity["observed_gpi8000_nova"], 3),
        "add_err": signed(additivity["error"], 3),
    }


def main():
    data = json.loads((HERE / "dados_pagina.json").read_text(encoding="utf-8"))
    page = (HERE / "pagina_modelo.html").read_text(encoding="utf-8")
    values = tokens(data)
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", page)) - set(values))
    if missing:
        raise KeyError(f"tokens without a value: {missing}")
    page = re.sub(r"\{\{(\w+)\}\}", lambda match: values[match.group(1)], page)
    page = page.replace("/*__DADOS__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    (HERE / "pagina.html").write_text(page, encoding="utf-8")
    print(f"wrote {HERE / 'pagina.html'} ({len(page) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
