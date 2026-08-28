#!/usr/bin/env python3
"""Aggregate RL ruleset timing profiles and render comparison artifacts."""

from __future__ import annotations

import csv
from collections import defaultdict
from hashlib import sha256
import json
from math import comb
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw_30_iterations"
BENCHMARK_MANIFEST = HERE / "benchmark_manifest_30_iterations.json"
LEARNING_CURVE_RAW = HERE / "learning_curve_raw.jsonl"
RULESETS = ("double-three", "double-four", "double-five", "double-six")
SHORT = {
    "double-three": "D3",
    "double-four": "D4",
    "double-five": "D5",
    "double-six": "D6",
}
COLORS = {
    "double-three": "#2ca02c",
    "double-four": "#1f77b4",
    "double-five": "#ff7f0e",
    "double-six": "#d62728",
}
GEOMETRY = {
    "double-three": {"max_pip": 3, "tiles": 10, "hand": 4, "stock": 2, "input": 69, "hidden1": 96, "hidden2": 48, "output": 20},
    "double-four": {"max_pip": 4, "tiles": 15, "hand": 5, "stock": 5, "input": 97, "hidden1": 128, "hidden2": 64, "output": 30},
    "double-five": {"max_pip": 5, "tiles": 21, "hand": 6, "stock": 9, "input": 130, "hidden1": 192, "hidden2": 96, "output": 42},
    "double-six": {"max_pip": 6, "tiles": 28, "hand": 7, "stock": 14, "input": 168, "hidden1": 256, "hidden2": 128, "output": 56},
}
REPETITIONS = (1, 2)
TOP_LEVEL_PRIMARY = (
    "rollout_game_execution",
    "ppo_update",
    "ppo_buffer_assembly_and_advantage_normalization",
    "reward_statistics",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_header_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        header = json.loads(next(stream))
        columns = header["columns"]
        rows = [dict(zip(columns, json.loads(line), strict=True)) for line in stream]
    return header, rows


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parameter_count(geometry: dict[str, int]) -> int:
    widths = (
        geometry["input"],
        geometry["hidden1"],
        geometry["hidden2"],
        geometry["output"],
    )
    return sum(right * left + right for left, right in zip(widths, widths[1:]))


def manifest_run_times() -> dict[tuple[str, int], float]:
    manifest = load_json(BENCHMARK_MANIFEST)
    return {
        (row["ruleset"], int(row["repetition"])): float(row["subprocess_wall_seconds"])
        for row in manifest["runs"]
    }


def load_runs() -> list[dict[str, Any]]:
    external_times = manifest_run_times()
    runs = []
    for ruleset in RULESETS:
        geometry = GEOMETRY[ruleset]
        for repetition in REPETITIONS:
            run_dir = RAW / ruleset / f"rep{repetition}"
            summary = load_json(run_dir / "summary.json")
            profile = summary["runtime_profile_delta"]
            metrics_path = Path(summary["metrics_output_path"])
            _header, metrics = load_header_rows(metrics_path)
            sections = profile["sections_seconds"]
            ppo_sections = profile["ppo_sections_seconds"]
            worker = profile["rollout_worker"]
            learner = worker["learner_policy"]
            learner_sections = learner["sections_seconds"]
            profiled_games = int(worker["profiled_games"])
            execution_seconds = float(profile["execution_seconds"])
            ppo_seconds = float(sections["ppo_update"])
            rollout_seconds = float(sections["rollout_game_execution"])
            fixed_and_other = execution_seconds - sum(
                float(sections.get(name, 0.0)) for name in TOP_LEVEL_PRIMARY
            )
            runs.append(
                {
                    "ruleset": ruleset,
                    "ruleset_short": SHORT[ruleset],
                    "repetition": repetition,
                    "tiles": geometry["tiles"],
                    "hand_size": geometry["hand"],
                    "initial_stock": geometry["stock"],
                    "input_size": geometry["input"],
                    "hidden1": geometry["hidden1"],
                    "hidden2": geometry["hidden2"],
                    "output_size": geometry["output"],
                    "parameter_count": parameter_count(geometry),
                    "initial_hidden_hand_upper_bound": comb(
                        geometry["tiles"] - geometry["hand"],
                        geometry["hand"],
                    ),
                    "training_games": int(summary["completed_training_games"]),
                    "iterations": int(profile["iterations"]),
                    "decisions": int(profile["decisions"]),
                    "decisions_per_game": float(summary["decisions_per_game"]),
                    "optimizer_steps": int(profile["optimizer_steps"]),
                    "optimizer_steps_per_iteration": float(profile["optimizer_steps"] / profile["iterations"]),
                    "full_buffer_evaluations": int(profile["ppo_full_buffer_evaluation"]["calls"]),
                    "full_buffer_evaluations_per_iteration": float(profile["ppo_full_buffer_evaluation"]["calls"] / profile["iterations"]),
                    "external_wall_seconds": external_times[(ruleset, repetition)],
                    "execution_seconds": execution_seconds,
                    "games_per_second": float(profile["games"] / execution_seconds),
                    "seconds_per_iteration": float(execution_seconds / profile["iterations"]),
                    "rollout_seconds": rollout_seconds,
                    "ppo_seconds": ppo_seconds,
                    "buffer_and_reward_seconds": float(
                        sections["ppo_buffer_assembly_and_advantage_normalization"]
                        + sections["reward_statistics"]
                    ),
                    "checkpoint_serialization_seconds": float(
                        sections.get("checkpoint_serialization", 0.0)
                    ),
                    "fixed_and_other_seconds": fixed_and_other,
                    "rollout_share_percent": 100.0 * rollout_seconds / execution_seconds,
                    "ppo_share_percent": 100.0 * ppo_seconds / execution_seconds,
                    "ppo_optimizer_seconds": float(ppo_sections["optimizer_steps"]),
                    "ppo_full_evaluation_seconds": float(ppo_sections["full_buffer_evaluation"]),
                    "ppo_other_seconds": float(
                        ppo_seconds
                        - ppo_sections["optimizer_steps"]
                        - ppo_sections["full_buffer_evaluation"]
                    ),
                    "ppo_full_evaluation_share_percent": float(
                        100.0 * ppo_sections["full_buffer_evaluation"] / ppo_seconds
                    ),
                    "worker_cpu_seconds": float(worker["worker_cpu_seconds"]),
                    "profiled_games": profiled_games,
                    "profiled_game_cpu_seconds": float(worker["profiled_game_cpu_seconds"]),
                    "profiled_cpu_ms_per_game": float(
                        1000.0 * worker["profiled_game_cpu_seconds"] / profiled_games
                    ),
                    "learner_policy_calls": int(learner["calls"]),
                    "learner_policy_us_per_call": float(
                        1_000_000.0 * learner["total_seconds"] / learner["calls"]
                    ),
                    "learner_exact_model_us_per_call": float(
                        1_000_000.0
                        * learner_sections["exact_opponent_model_update"]
                        / learner["calls"]
                    ),
                    "learner_forward_us_per_call": float(
                        1_000_000.0
                        * learner_sections["network_forward_and_host_transfer"]
                        / learner["calls"]
                    ),
                    "learner_encoding_us_per_call": float(
                        1_000_000.0
                        * learner_sections["state_encoding_and_backend_transfer"]
                        / learner["calls"]
                    ),
                    "learner_selection_us_per_call": float(
                        1_000_000.0
                        * (
                            learner_sections["legal_mask_and_action_selection"]
                            + learner_sections["action_filtering_and_forced_choice"]
                        )
                        / learner["calls"]
                    ),
                    "learner_exact_model_share_percent": float(
                        100.0
                        * learner_sections["exact_opponent_model_update"]
                        / learner["total_seconds"]
                    ),
                    "mean_final_kl": float(mean(float(row["final_approx_kl"]) for row in metrics)),
                    "max_final_kl": float(max(float(row["final_approx_kl"]) for row in metrics)),
                    "mean_clip_fraction": float(mean(float(row["final_clip_fraction"]) for row in metrics)),
                    "mean_epochs": float(mean(float(row["epochs_completed"]) for row in metrics)),
                    "kl_early_stops": int(sum(bool(row["stopped_by_kl"]) for row in metrics)),
                    "gradient_clipped_iterations": int(sum(bool(row["gradient_clipped"]) for row in metrics)),
                    "weights_sha256": file_sha256(
                        run_dir
                        / "iteration_weights"
                        / f"iteration_{int(profile['iterations']):06d}.npz"
                    ),
                    "profile": profile,
                    "metrics": metrics,
                }
            )
    return runs


def validate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {(ruleset, repetition) for ruleset in RULESETS for repetition in REPETITIONS}
    actual = {(row["ruleset"], row["repetition"]) for row in runs}
    if actual != expected:
        raise RuntimeError(f"Incomplete benchmark grid: {sorted(expected - actual)}")
    determinism = {}
    for ruleset in RULESETS:
        selected = [row for row in runs if row["ruleset"] == ruleset]
        hashes = {row["weights_sha256"] for row in selected}
        retained_hash_maps = []
        for repetition in REPETITIONS:
            summary = load_json(RAW / ruleset / f"rep{repetition}" / "summary.json")
            retained_hash_maps.append(
                summary["retained_iteration_weights"]["sha256_by_iteration"]
            )
        all_retained_identical = all(
            value == retained_hash_maps[0] for value in retained_hash_maps[1:]
        )
        initial_path = (
            RAW / ruleset / "rep1" / "iteration_weights" / "iteration_000000.npz"
        )
        archive_initial_path = (
            RAW / ruleset / "rep1" / "checkpoint_archive" / "checkpoint_iter000000.npz"
        )
        with np.load(initial_path, allow_pickle=False) as initial, np.load(
            archive_initial_path,
            allow_pickle=False,
        ) as archived:
            initial_matches_archive = all(
                name in initial and np.array_equal(initial[name], archived[name])
                for name in archived.files
            )
        structural = {
            (
                row["training_games"],
                row["decisions"],
                row["optimizer_steps"],
                row["mean_epochs"],
                row["kl_early_stops"],
            )
            for row in selected
        }
        determinism[ruleset] = {
            "weight_hashes": sorted(hashes),
            "byte_identical_weights": len(hashes) == 1,
            "byte_identical_all_31_policies": all_retained_identical,
            "initial_policy_matches_training_archive": initial_matches_archive,
            "identical_work_counts": len(structural) == 1,
        }
        if (
            len(hashes) != 1
            or not all_retained_identical
            or not initial_matches_archive
            or len(structural) != 1
        ):
            raise RuntimeError(f"Fixed-seed benchmark was not reproducible for {ruleset}")
    return {"determinism": determinism}


def plain_run_row(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key not in {"profile", "metrics"}}


def aggregate_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric_fields = [
        key
        for key, value in plain_run_row(runs[0]).items()
        if isinstance(value, (int, float)) and key != "repetition"
    ]
    aggregates = []
    for ruleset in RULESETS:
        selected = [row for row in runs if row["ruleset"] == ruleset]
        row: dict[str, Any] = {
            "ruleset": ruleset,
            "ruleset_short": SHORT[ruleset],
            "repetitions": len(selected),
        }
        for field in numeric_fields:
            values = [float(item[field]) for item in selected]
            row[f"median_{field}"] = median(values)
            if field in {"external_wall_seconds", "execution_seconds", "rollout_seconds", "ppo_seconds"}:
                row[f"min_{field}"] = min(values)
                row[f"max_{field}"] = max(values)
                row[f"cv_{field}_percent"] = 100.0 * pstdev(values) / mean(values)
        aggregates.append(row)
    return aggregates


def load_learning_curves(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join fixed-panel win rates to cumulative RL work and wall time."""
    raw_rows = [
        json.loads(line)
        for line in LEARNING_CURVE_RAW.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["ruleset"], int(row["iteration"]))].append(row)

    curves = []
    for ruleset in RULESETS:
        selected_runs = [row for row in runs if row["ruleset"] == ruleset]
        for iteration in range(31):
            diagnostics = grouped.get((ruleset, iteration), [])
            if not diagnostics:
                raise RuntimeError(
                    f"Missing learning-curve diagnostic for {ruleset} iteration {iteration}"
                )
            if iteration == 0:
                elapsed_seconds = 0.0
                cumulative_decisions = 0.0
                cumulative_optimizer_steps = 0.0
            else:
                metrics = [run["metrics"][iteration - 1] for run in selected_runs]
                elapsed_seconds = median(
                    float(row["elapsed_training_seconds"]) for row in metrics
                )
                cumulative_decisions = median(
                    float(row["cumulative_normal_decisions"])
                    + float(row["cumulative_restart_decisions"])
                    for row in metrics
                )
                cumulative_optimizer_steps = median(
                    sum(
                        int(metric["optimizer_steps"])
                        for metric in run["metrics"][:iteration]
                    )
                    for run in selected_runs
                )
            curves.append({
                "ruleset": ruleset,
                "ruleset_short": SHORT[ruleset],
                "iteration": iteration,
                "training_games": int(diagnostics[0]["training_games"]),
                "training_elapsed_seconds": elapsed_seconds,
                "cumulative_decisions": cumulative_decisions,
                "cumulative_optimizer_steps": cumulative_optimizer_steps,
                "evaluated_repetitions": len(diagnostics),
                "diagnostic_games": int(diagnostics[0]["diagnostic_games"]),
                "wins": mean(float(row["wins"]) for row in diagnostics),
                "losses": mean(float(row["losses"]) for row in diagnostics),
                "win_rate_percent": 100.0 * mean(
                    float(row["win_rate"]) for row in diagnostics
                ),
                "ci95_low_percent": 100.0 * mean(
                    float(row["ci95_low"]) for row in diagnostics
                ),
                "ci95_high_percent": 100.0 * mean(
                    float(row["ci95_high"]) for row in diagnostics
                ),
                "diagnostic_seconds": sum(
                    float(row["diagnostic_seconds"]) for row in diagnostics
                ),
            })
    return curves


def _curve_slope(points: list[dict[str, Any]], lower: int, upper: int) -> float:
    selected = [row for row in points if lower <= row["iteration"] <= upper]
    return float(np.polyfit(
        [row["iteration"] for row in selected],
        [row["win_rate_percent"] for row in selected],
        1,
    )[0])


def summarize_learning_curves(
    curves: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    timing_lookup = aggregate_lookup(aggregates)
    summaries = []
    for ruleset in RULESETS:
        points = sorted(
            (row for row in curves if row["ruleset"] == ruleset),
            key=lambda row: row["iteration"],
        )
        baseline = points[0]["win_rate_percent"]
        final = points[-1]["win_rate_percent"]
        gain = final - baseline

        def first_gain(delta):
            for point in points:
                if point["win_rate_percent"] >= baseline + delta:
                    return point
            return None

        plus_one = first_gain(1.0)
        plus_two = first_gain(2.0)
        early_slope = _curve_slope(points, 0, 10)
        middle_slope = _curve_slope(points, 10, 20)
        late_slope = _curve_slope(points, 20, 30)
        final_decisions = points[-1]["cumulative_decisions"]
        final_steps = points[-1]["cumulative_optimizer_steps"]
        final_seconds = points[-1]["training_elapsed_seconds"]
        checkpoint_capture_seconds = timing_lookup[ruleset][
            "median_checkpoint_serialization_seconds"
        ]
        # A normal 30-iteration invocation needs its final checkpoint, but this
        # experiment wrote all 30 to recover the curve. Remove 29/30 of that
        # measured serialization only for the adjusted efficiency estimate.
        adjusted_seconds = (
            final_seconds - checkpoint_capture_seconds * 29.0 / 30.0
        )
        summaries.append({
            "ruleset": ruleset,
            "ruleset_short": SHORT[ruleset],
            "baseline_win_rate_percent": baseline,
            "final_win_rate_percent": final,
            "gain_percentage_points": gain,
            "best_win_rate_percent": max(row["win_rate_percent"] for row in points),
            "best_iteration": max(points, key=lambda row: row["win_rate_percent"])["iteration"],
            "gain_first_10_iterations_pp": points[10]["win_rate_percent"] - baseline,
            "gain_middle_10_iterations_pp": points[20]["win_rate_percent"] - points[10]["win_rate_percent"],
            "gain_last_10_iterations_pp": final - points[20]["win_rate_percent"],
            "early_slope_pp_per_iteration": early_slope,
            "middle_slope_pp_per_iteration": middle_slope,
            "late_slope_pp_per_iteration": late_slope,
            "late_to_early_slope_ratio": (
                late_slope / early_slope if early_slope else None
            ),
            "gain_fraction_at_iteration_10": (
                (points[10]["win_rate_percent"] - baseline) / gain if gain else None
            ),
            "gain_fraction_at_iteration_20": (
                (points[20]["win_rate_percent"] - baseline) / gain if gain else None
            ),
            "iteration_to_plus_1pp": None if plus_one is None else plus_one["iteration"],
            "seconds_to_plus_1pp": None if plus_one is None else plus_one["training_elapsed_seconds"],
            "iteration_to_plus_2pp": None if plus_two is None else plus_two["iteration"],
            "seconds_to_plus_2pp": None if plus_two is None else plus_two["training_elapsed_seconds"],
            "training_elapsed_seconds": final_seconds,
            "experimental_checkpoint_capture_seconds": checkpoint_capture_seconds,
            "adjusted_training_seconds_without_extra_curve_checkpoints": adjusted_seconds,
            "total_decisions": final_decisions,
            "total_optimizer_steps": final_steps,
            "gain_pp_per_training_minute": gain / (final_seconds / 60.0),
            "adjusted_gain_pp_per_training_minute": gain / (adjusted_seconds / 60.0),
            "gain_pp_per_100k_decisions": gain * 100_000.0 / final_decisions,
            "gain_pp_per_1000_optimizer_steps": gain * 1_000.0 / final_steps,
            "diagnostic_games_per_point": points[0]["diagnostic_games"],
            "diagnostic_seconds_all_points": sum(
                row["diagnostic_seconds"] for row in points
            ),
        })
    return summaries


def profile_table(
    runs: list[dict[str, Any]],
    tree_name: str,
    subtree_name: str | None = None,
) -> list[dict[str, Any]]:
    records = []
    for run in runs:
        tree = run["profile"][tree_name]
        if subtree_name is not None:
            tree = tree[subtree_name]
        for section, seconds in tree.items():
            if not isinstance(seconds, (int, float)):
                continue
            records.append(
                {
                    "ruleset": run["ruleset"],
                    "repetition": run["repetition"],
                    "section": section,
                    "seconds": seconds,
                }
            )
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(axis: plt.Axes, xlabel: str, ylabel: str) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def aggregate_lookup(aggregates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["ruleset"]: row for row in aggregates}


def figure_total_time(runs: list[dict[str, Any]], aggregates: list[dict[str, Any]]) -> None:
    lookup = aggregate_lookup(aggregates)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    x = np.arange(len(RULESETS))
    medians = [lookup[rs]["median_execution_seconds"] for rs in RULESETS]
    throughputs = [lookup[rs]["median_games_per_second"] for rs in RULESETS]
    axes[0].bar(x, medians, color=[COLORS[rs] for rs in RULESETS], alpha=0.82)
    axes[1].bar(x, throughputs, color=[COLORS[rs] for rs in RULESETS], alpha=0.82)
    for index, ruleset in enumerate(RULESETS):
        selected = [row for row in runs if row["ruleset"] == ruleset]
        axes[0].scatter(
            np.full(len(selected), index),
            [row["execution_seconds"] for row in selected],
            color="#222222",
            s=22,
            zorder=3,
        )
        axes[1].scatter(
            np.full(len(selected), index),
            [row["games_per_second"] for row in selected],
            color="#222222",
            s=22,
            zorder=3,
        )
    for axis in axes:
        axis.set_xticks(x, [SHORT[rs] for rs in RULESETS])
    game_count = int(aggregates[0]["median_training_games"])
    style_axis(axes[0], "Ruleset", f"Profiled seconds for {game_count:,} games")
    style_axis(axes[1], "Ruleset", "Training games per second")
    axes[0].set_title("End-to-end RL invocation time")
    axes[1].set_title("RL throughput")
    figure.savefig(HERE / "01_total_time_and_throughput.png", dpi=180)
    plt.close(figure)


def figure_top_level_breakdown(aggregates: list[dict[str, Any]]) -> None:
    lookup = aggregate_lookup(aggregates)
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    x = np.arange(len(RULESETS))
    categories = (
        ("Rollout", "median_rollout_seconds", "#4c78a8"),
        ("PPO update", "median_ppo_seconds", "#f28e2b"),
        ("Buffer + reward summaries", "median_buffer_and_reward_seconds", "#59a14f"),
        ("All other session overhead", "median_fixed_and_other_seconds", "#bab0ac"),
    )
    bottom = np.zeros(len(RULESETS))
    for label, field, color in categories:
        values = np.asarray([lookup[rs][field] for rs in RULESETS])
        axis.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    axis.set_xticks(x, [SHORT[rs] for rs in RULESETS])
    game_count = int(aggregates[0]["median_training_games"])
    style_axis(axis, "Ruleset", f"Seconds per {game_count:,} training games")
    axis.set_title("Where RL wall time goes")
    axis.legend(frameon=False)
    figure.savefig(HERE / "02_top_level_time_breakdown.png", dpi=180)
    plt.close(figure)


def figure_scaling(aggregates: list[dict[str, Any]]) -> None:
    lookup = aggregate_lookup(aggregates)
    d6 = lookup["double-six"]
    fields = (
        ("Tile count", "median_tiles"),
        ("Network parameters", "median_parameter_count"),
        ("Decisions / game", "median_decisions_per_game"),
        ("Rollout time", "median_rollout_seconds"),
        ("PPO time", "median_ppo_seconds"),
        ("Total RL time", "median_execution_seconds"),
    )
    figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    x = np.arange(len(RULESETS))
    markers = ("o", "s", "^", "D", "v", "P")
    for (label, field), marker in zip(fields, markers, strict=True):
        values = [100.0 * lookup[rs][field] / d6[field] for rs in RULESETS]
        axis.plot(x, values, marker=marker, linewidth=2, label=label)
    axis.set_xticks(x, [SHORT[rs] for rs in RULESETS])
    style_axis(axis, "Ruleset", "Relative to double-six (%)")
    axis.set_title("Measured scaling follows decision density, not parameter count")
    axis.legend(ncols=2, frameon=False)
    figure.savefig(HERE / "03_normalized_scaling.png", dpi=180)
    plt.close(figure)


def median_worker_sections(runs: list[dict[str, Any]], ruleset: str) -> dict[str, float]:
    selected = [row for row in runs if row["ruleset"] == ruleset]
    names = set().union(*(row["profile"]["rollout_worker"]["sections_seconds"] for row in selected))
    return {
        name: median(
            1000.0
            * row["profile"]["rollout_worker"]["sections_seconds"].get(name, 0.0)
            / row["profiled_games"]
            for row in selected
        )
        for name in names
    }


def figure_rollout_breakdown(runs: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    x = np.arange(len(RULESETS))
    groups = (
        ("Learner decisions", ("learner_agent_decisions",), "#4c78a8"),
        ("Opponent decisions", ("opponent_agent_decisions",), "#f28e2b"),
        ("State + legal actions", ("state_and_legal_action_generation",), "#59a14f"),
        ("Engine", ("engine_initialization", "engine_state_transition", "terminal_reward_and_trajectory_finalization"), "#e15759"),
        ("Setup, RNG, reward, payload, other", ("agent_setup", "per_game_rng_setup", "reward_shaping", "result_payload_construction", "forced_learner_action_selection", "unaccounted"), "#bab0ac"),
    )
    bottom = np.zeros(len(RULESETS))
    medians = {ruleset: median_worker_sections(runs, ruleset) for ruleset in RULESETS}
    for label, names, color in groups:
        values = np.asarray(
            [sum(medians[ruleset].get(name, 0.0) for name in names) for ruleset in RULESETS]
        )
        axis.bar(x, values, bottom=bottom, color=color, label=label)
        bottom += values
    axis.set_xticks(x, [SHORT[rs] for rs in RULESETS])
    style_axis(axis, "Ruleset", "Sampled worker CPU milliseconds per game")
    axis.set_title("Deep-profiled rollout CPU cost (1 in every 32 games)")
    axis.legend(frameon=False, fontsize=9)
    figure.savefig(HERE / "04_rollout_cpu_breakdown.png", dpi=180)
    plt.close(figure)


def figure_learner_policy(aggregates: list[dict[str, Any]]) -> None:
    lookup = aggregate_lookup(aggregates)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    x = np.arange(len(RULESETS))
    width = 0.19
    fields = (
        ("Exact opponent update", "median_learner_exact_model_us_per_call", "#e15759"),
        ("Network forward", "median_learner_forward_us_per_call", "#4c78a8"),
        ("State encoding", "median_learner_encoding_us_per_call", "#59a14f"),
        ("Mask + action selection", "median_learner_selection_us_per_call", "#f28e2b"),
    )
    for offset, (label, field, color) in enumerate(fields):
        axes[0].bar(
            x + (offset - 1.5) * width,
            [lookup[rs][field] for rs in RULESETS],
            width=width,
            color=color,
            label=label,
        )
    shares = [lookup[rs]["median_learner_exact_model_share_percent"] for rs in RULESETS]
    axes[1].bar(x, shares, color=[COLORS[rs] for rs in RULESETS], alpha=0.82)
    for axis in axes:
        axis.set_xticks(x, [SHORT[rs] for rs in RULESETS])
    style_axis(axes[0], "Ruleset", "Microseconds per trainable learner decision")
    style_axis(axes[1], "Ruleset", "Exact-model share of learner policy time (%)")
    axes[0].set_title("Learner policy microprofile")
    axes[1].set_title("The double-four exact-model anomaly")
    axes[0].legend(fontsize=8, frameon=False)
    figure.savefig(HERE / "05_learner_policy_microprofile.png", dpi=180)
    plt.close(figure)


def figure_ppo_breakdown(aggregates: list[dict[str, Any]]) -> None:
    lookup = aggregate_lookup(aggregates)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    x = np.arange(len(RULESETS))
    bottom = np.zeros(len(RULESETS))
    categories = (
        ("Optimizer steps", "median_ppo_optimizer_seconds", "#4c78a8"),
        ("Full-buffer evaluation", "median_ppo_full_evaluation_seconds", "#f28e2b"),
        ("Partition, materialization, storage, control", "median_ppo_other_seconds", "#bab0ac"),
    )
    for label, field, color in categories:
        values = np.asarray([lookup[rs][field] / lookup[rs]["median_iterations"] for rs in RULESETS])
        axes[0].bar(x, values, bottom=bottom, color=color, label=label)
        bottom += values
    steps = [lookup[rs]["median_optimizer_steps_per_iteration"] for rs in RULESETS]
    evaluations = [lookup[rs]["median_full_buffer_evaluations_per_iteration"] for rs in RULESETS]
    axes[1].plot(x, steps, marker="o", linewidth=2.2, label="Optimizer steps / iteration")
    axes[1].plot(x, evaluations, marker="s", linewidth=2.2, label="Full-buffer evaluations / iteration")
    for axis in axes:
        axis.set_xticks(x, [SHORT[rs] for rs in RULESETS])
    style_axis(axes[0], "Ruleset", "PPO seconds per iteration")
    style_axis(axes[1], "Ruleset", "Calls per iteration")
    axes[0].set_title("PPO work breakdown")
    axes[1].set_title("PPO call counts")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].legend(fontsize=8, frameon=False)
    figure.savefig(HERE / "06_ppo_breakdown_and_calls.png", dpi=180)
    plt.close(figure)


def figure_iteration_overlay(runs: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ruleset in RULESETS:
        selected = [row for row in runs if row["ruleset"] == ruleset]
        iteration_count = len(selected[0]["metrics"])
        x = np.arange(1, iteration_count + 1)
        rollout = np.median(
            [[float(metric["rollout_seconds"]) for metric in row["metrics"]] for row in selected],
            axis=0,
        )
        update = np.median(
            [[float(metric["update_seconds"]) for metric in row["metrics"]] for row in selected],
            axis=0,
        )
        axes[0].plot(x, rollout, color=COLORS[ruleset], linewidth=2, marker="o", label=SHORT[ruleset])
        axes[1].plot(x, update, color=COLORS[ruleset], linewidth=2, marker="o", label=SHORT[ruleset])
    style_axis(axes[0], "RL iteration", "Rollout wall seconds")
    style_axis(axes[1], "RL iteration", "PPO update seconds")
    axes[0].set_title("Per-iteration rollout timing")
    axes[1].set_title("Per-iteration PPO timing")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    figure.savefig(HERE / "07_iteration_timing_overlay.png", dpi=180)
    plt.close(figure)


def figure_time_per_decision(aggregates: list[dict[str, Any]]) -> None:
    lookup = aggregate_lookup(aggregates)
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    x = np.arange(len(RULESETS))
    rollout = [
        1_000_000.0 * lookup[rs]["median_rollout_seconds"] / lookup[rs]["median_decisions"]
        for rs in RULESETS
    ]
    ppo = [
        1_000_000.0 * lookup[rs]["median_ppo_seconds"] / lookup[rs]["median_decisions"]
        for rs in RULESETS
    ]
    width = 0.36
    axis.bar(x - width / 2, rollout, width=width, label="Rollout wall time / trainable decision", color="#4c78a8")
    axis.bar(x + width / 2, ppo, width=width, label="PPO wall time / trainable decision", color="#f28e2b")
    axis.set_xticks(x, [SHORT[rs] for rs in RULESETS])
    style_axis(axis, "Ruleset", "Amortized microseconds per trainable decision")
    axis.set_title("Cost after normalizing away different decision counts")
    axis.legend(frameon=False)
    figure.savefig(HERE / "08_amortized_time_per_decision.png", dpi=180)
    plt.close(figure)


def figure_learning_curve_iteration(curves: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for ruleset in RULESETS:
        points = [row for row in curves if row["ruleset"] == ruleset]
        x = np.asarray([row["iteration"] for row in points])
        rate = np.asarray([row["win_rate_percent"] for row in points])
        low = np.asarray([row["ci95_low_percent"] for row in points])
        high = np.asarray([row["ci95_high_percent"] for row in points])
        axis.fill_between(x, low, high, color=COLORS[ruleset], alpha=0.08)
        axis.plot(
            x,
            rate,
            color=COLORS[ruleset],
            linewidth=2.2,
            marker="o",
            markersize=3.5,
            label=f"{SHORT[ruleset]} ({rate[0]:.2f}% to {rate[-1]:.2f}%)",
        )
    axis.axhline(50.0, color="#555555", linewidth=1, linestyle="--")
    style_axis(axis, "Completed RL iterations", "Deterministic RL wins vs random (%)")
    axis.set_title("Learning curves from the exact random-policy baseline")
    axis.legend(frameon=False, ncols=2)
    figure.savefig(HERE / "09_learning_curve_by_iteration.png", dpi=180)
    plt.close(figure)


def figure_learning_curve_time(curves: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for ruleset in RULESETS:
        points = [row for row in curves if row["ruleset"] == ruleset]
        baseline = points[0]["win_rate_percent"]
        x = np.asarray([row["training_elapsed_seconds"] for row in points])
        gain = np.asarray([row["win_rate_percent"] - baseline for row in points])
        axis.plot(
            x,
            gain,
            color=COLORS[ruleset],
            linewidth=2.4,
            marker="o",
            markersize=3.5,
            label=SHORT[ruleset],
        )
    style_axis(axis, "RL training wall seconds", "Gain over initial checkpoint (percentage points)")
    axis.set_title("Practical learning speed after accounting for ruleset runtime")
    axis.legend(frameon=False)
    figure.savefig(HERE / "10_learning_curve_by_wall_time.png", dpi=180)
    plt.close(figure)


def figure_learning_efficiency(summaries: list[dict[str, Any]]) -> None:
    lookup = {row["ruleset"]: row for row in summaries}
    figure, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    x = np.arange(len(RULESETS))
    colors = [COLORS[ruleset] for ruleset in RULESETS]
    axes[0].bar(
        x,
        [lookup[rs]["adjusted_gain_pp_per_training_minute"] for rs in RULESETS],
        color=colors,
        alpha=0.84,
    )
    axes[1].bar(
        x,
        [lookup[rs]["gain_pp_per_100k_decisions"] for rs in RULESETS],
        color=colors,
        alpha=0.84,
    )
    width = 0.36
    axes[2].bar(
        x - width / 2,
        [lookup[rs]["seconds_to_plus_1pp"] for rs in RULESETS],
        width=width,
        color="#4c78a8",
        label="Reach +1 pp",
    )
    axes[2].bar(
        x + width / 2,
        [lookup[rs]["seconds_to_plus_2pp"] for rs in RULESETS],
        width=width,
        color="#f28e2b",
        label="Reach +2 pp",
    )
    titles = (
        "Gain per training minute (capture-adjusted)",
        "Gain per 100k decisions",
        "Wall time to relative improvement",
    )
    ylabels = (
        "Percentage points per minute",
        "Percentage points per 100k decisions",
        "RL training seconds",
    )
    for axis, title, ylabel in zip(axes, titles, ylabels, strict=True):
        axis.set_xticks(x, [SHORT[rs] for rs in RULESETS])
        style_axis(axis, "Ruleset", ylabel)
        axis.set_title(title)
    axes[2].legend(frameon=False, fontsize=8)
    figure.savefig(HERE / "11_learning_efficiency_and_thresholds.png", dpi=180)
    plt.close(figure)


def figure_learning_phases(summaries: list[dict[str, Any]]) -> None:
    lookup = {row["ruleset"]: row for row in summaries}
    figure, axis = plt.subplots(figsize=(12, 7), constrained_layout=True)
    x = np.arange(len(RULESETS))
    width = 0.25
    phases = (
        ("Iterations 0-10", "gain_first_10_iterations_pp", "#4c78a8"),
        ("Iterations 10-20", "gain_middle_10_iterations_pp", "#f28e2b"),
        ("Iterations 20-30", "gain_last_10_iterations_pp", "#59a14f"),
    )
    for index, (label, field, color) in enumerate(phases):
        axis.bar(
            x + (index - 1) * width,
            [lookup[rs][field] for rs in RULESETS],
            width=width,
            color=color,
            label=label,
        )
    axis.set_xticks(x, [SHORT[rs] for rs in RULESETS])
    style_axis(axis, "Ruleset", "Win-rate gain during phase (percentage points)")
    axis.set_title("How much of the improvement arrives in each ten-iteration phase")
    axis.legend(frameon=False)
    figure.savefig(HERE / "12_learning_gain_by_phase.png", dpi=180)
    plt.close(figure)


def render_report(
    runs: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    validation: dict[str, Any],
    learning_summaries: list[dict[str, Any]],
) -> None:
    lookup = aggregate_lookup(aggregates)
    d3 = lookup["double-three"]
    d4 = lookup["double-four"]
    d6 = lookup["double-six"]
    speedup_d3 = d6["median_execution_seconds"] / d3["median_execution_seconds"]
    parameter_ratio = d6["median_parameter_count"] / d3["median_parameter_count"]
    decision_ratio = d6["median_decisions_per_game"] / d3["median_decisions_per_game"]
    d4_exact_ratio = (
        d4["median_learner_exact_model_us_per_call"]
        / d6["median_learner_exact_model_us_per_call"]
    )
    decision_values = np.asarray([lookup[rs]["median_decisions_per_game"] for rs in RULESETS])
    time_values = np.asarray([lookup[rs]["median_execution_seconds"] for rs in RULESETS])
    correlation = float(np.corrcoef(decision_values, time_values)[0, 1])
    total_iterations = sum(int(row["iterations"]) for row in runs)
    total_early_stops = sum(int(row["kl_early_stops"]) for row in runs)
    total_clipped = sum(int(row["gradient_clipped_iterations"]) for row in runs)
    learning_lookup = {row["ruleset"]: row for row in learning_summaries}
    benchmark = load_json(BENCHMARK_MANIFEST)["configuration"]
    repetitions = int(benchmark["repetitions"])
    games_per_run = int(benchmark["training_games_per_run"])
    iterations_per_run = int(benchmark["iterations_per_run"])
    profiled_games = int(d3["median_profiled_games"])

    lines = [
        "# RL ruleset timing report",
        "",
        "## Conclusion",
        "",
        (
            "The compact rulesets are faster, and their end-to-end time is almost perfectly explained by the number "
            f"of trainable decisions per game (Pearson r = {correlation:.3f} across the four rulesets). Double-three "
            f"is {speedup_d3:.2f}x faster than double-six. It is not {parameter_ratio:.2f}x faster like the parameter "
            f"count might suggest because it still produces 1/{decision_ratio:.2f} as many trainable decisions and pays "
            "the same 16-epoch PPO orchestration and GPU-launch structure."
        ),
        "",
        (
            "One non-monotonic hotspot is real: a double-four learner decision spends about "
            f"{d4['median_learner_exact_model_us_per_call']:.1f} us updating the exact opponent model, "
            f"{d4_exact_ratio:.2f}x the double-six cost per call. This makes double-four rollout scale worse than its "
            "decision count predicts."
        ),
        "",
        (
            "The learning curves do not show compact rulesets learning faster per fixed iteration. After 30 "
            f"iterations, double-three gained {learning_lookup['double-three']['gain_percentage_points']:.2f} "
            "win-rate points versus random while double-six gained "
            f"{learning_lookup['double-six']['gain_percentage_points']:.2f}. Double-three is nevertheless the "
            "most efficient per wall minute and per trainable decision because each iteration is much cheaper."
        ),
        "",
        "## Benchmark controls",
        "",
        f"- 4 rulesets x {repetitions} subprocess-isolated repetitions; {games_per_run:,} games and {iterations_per_run} iterations per repetition.",
        "- Fixed seed 20,260,828, GPI 2,000, 10 CPU rollout workers, GPU policy updates, and 16 PPO epochs.",
        "- Default `heuristic,recent` opponent buckets; no decision restarts, diagnostics, dataset generation, or supervised training.",
        "- Random ruleset-default networks were initialized from the fixed seed; all 31 policies from iteration zero through 30 were byte-identical between repetitions.",
            f"- Worker deep profiling sampled exactly {profiled_games:,}/{games_per_run:,} games per run (one in 32); the normal hot path remains uninstrumented for the other games.",
            "- Every retained checkpoint was evaluated deterministically against the same fixed panel of 10,000 random-opponent games.",
            "- Retaining every policy required an experimental numbered checkpoint after every iteration; capture-adjusted efficiency removes 29 of those 30 serializations while retaining the normal final save.",
            "",
            "## Interpretation limits",
            "",
            "These are controlled RL smoke benchmarks, not forecasts of final playing strength. They isolate the cost of "
            "the training loop by excluding dataset generation, supervised learning, and diagnostics. A mature run has "
            "a larger recent-opponent pool and a more developed policy, so absolute throughput can move somewhat; the "
            "ruleset geometry, PPO work per collected decision, and exact-model representation effect measured here remain "
            "the relevant mechanisms. Two repetitions and alternating execution order keep the comparison stable, but "
            "a four-point correlation should be read as a strong engineering clue rather than a statistical law.",
            "",
            "## Measured totals",
        "",
        "| Ruleset | Parameters | Decisions/game | Median time | Games/s | Rollout | PPO | PPO share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ruleset in RULESETS:
        row = lookup[ruleset]
        lines.append(
            f"| {ruleset} | {row['median_parameter_count']:,.0f} | "
            f"{row['median_decisions_per_game']:.3f} | {row['median_execution_seconds']:.2f} s | "
            f"{row['median_games_per_second']:.1f} | {row['median_rollout_seconds']:.2f} s | "
            f"{row['median_ppo_seconds']:.2f} s | {row['median_ppo_share_percent']:.1f}% |"
        )
    lines.extend([
        "",
        "## Early learning curves versus random",
        "",
        "| Ruleset | Initial | Iteration 30 | Gain | Gain 0-10 / 10-20 / 20-30 | adjusted pp/min | pp/100k decisions | Time to +1 pp | Time to +2 pp |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for ruleset in RULESETS:
        row = learning_lookup[ruleset]
        lines.append(
            f"| {ruleset} | {row['baseline_win_rate_percent']:.2f}% | "
            f"{row['final_win_rate_percent']:.2f}% | {row['gain_percentage_points']:+.2f} pp | "
            f"{row['gain_first_10_iterations_pp']:+.2f} / "
            f"{row['gain_middle_10_iterations_pp']:+.2f} / "
            f"{row['gain_last_10_iterations_pp']:+.2f} | "
            f"{row['adjusted_gain_pp_per_training_minute']:.2f} | "
            f"{row['gain_pp_per_100k_decisions']:.2f} | "
            f"{row['seconds_to_plus_1pp']:.1f} s | "
            f"{row['seconds_to_plus_2pp']:.1f} s |"
        )
    lines.extend([
        "",
        "The first hope is not supported in iteration units: double-six improves most rapidly per iteration, "
        "and reaches +1 point in only five iterations. A fixed 2,000-game iteration is not equal work across "
        "rulesets: double-six collected about 236,890 trainable decisions over the experiment, versus 91,671 "
        "for double-three, and therefore performed many more optimizer steps.",
        "",
        "The compact ruleset does recover an efficiency advantage after normalizing the work. Double-three gains "
        "3.17 points per capture-adjusted training minute and 2.61 points per 100,000 decisions. The other rulesets "
        "produce only 1.93-2.12 points per minute and 1.77-1.97 points per 100,000 decisions. It is the fastest to reach a "
        "+2-point relative improvement (34.5 seconds), although double-six narrowly reaches +1 point first "
        "(21.7 versus 23.6 seconds).",
        "",
        "Saving all 31 policies added an artificial 1.93, 2.82, 5.14, and 8.47 seconds to D3 through D6, respectively. "
        "The adjusted per-minute comparison removes the 29 intermediate saves a normal 30-iteration invocation would "
        "not make. Threshold times above are the directly observed conservative values, including those saves; this "
        "choice cannot create the double-three efficiency advantage.",
        "",
        "There is an early flattening hint, not an asymptote measurement. Double-three gains +1.13 points in "
        "iterations 10-20 but only +0.57 in iterations 20-30; its fitted late slope falls to 0.038 points per "
        "iteration. Double-five also flattens in the last third. Double-four and double-six still have clearer "
        "positive late slopes. These 30 iterations are enough to reject an orders-of-magnitude learning-speed "
        "advantage, but not enough to estimate final ceilings or prove convergence.",
    ])
    lines.extend(
        [
            "",
            "## What explains the scaling",
            "",
            (
                f"1. **Decision density dominates.** Double-three has {d3['median_decisions_per_game']:.3f} trainable "
                f"decisions/game versus {d6['median_decisions_per_game']:.3f} for double-six. PPO time falls from "
                f"{d6['median_ppo_seconds']:.2f} s to {d3['median_ppo_seconds']:.2f} s, almost exactly with this ratio."
            ),
            (
                "2. **PPO is the majority cost in every ruleset.** It consumes 60-71% of invocation time. The optimizer "
                "steps scale with buffer decisions, while the full buffer is evaluated once after every epoch. That "
                "evaluation alone consumes about 36% of PPO time in all four rulesets."
            ),
            (
                "3. **The GPU does not scale with parameter count alone.** These networks are all small; fixed kernel "
                "launches, mask validation, host transfers, minibatch materialization, and 16 epoch-level evaluations "
                "remain. Double-three has only 14.8% of double-six parameters but needs 39.6% of its PPO time."
            ),
            (
                "4. **Generic orchestration is small.** Excluding the explicitly measured experimental per-iteration "
                "checkpoint capture, match planning, metrics, archive work, buffer preflight, and final writes together "
                "are a small fraction. There is no sign that a large double-six-only operation is accidentally running "
                "unchanged in every compact ruleset."
            ),
            "",
            "## The double-four exact-model anomaly",
            "",
            "The fixed representation threshold is `SWITCH_TO_MU_MAX_HANDS = 500` in `middleware/opponent_model.py:35`. "
            "At the end of a non-terminal turn, `_maybe_switch_to_mu()` converts when the raw hidden-hand upper bound "
            "is at most 500 (`middleware/opponent_model.py:1314-1345`). Initial upper bounds are:",
            "",
            "| Ruleset | Initial hidden-hand upper bound | Immediate relationship to threshold | Exact-model us/learner call | Exact-model policy share |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for ruleset in RULESETS:
        row = lookup[ruleset]
        relation = "at/below 500" if row["median_initial_hidden_hand_upper_bound"] <= 500 else "above 500"
        lines.append(
            f"| {ruleset} | {row['median_initial_hidden_hand_upper_bound']:,.0f} | {relation} | "
            f"{row['median_learner_exact_model_us_per_call']:.1f} | "
            f"{row['median_learner_exact_model_share_percent']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "Double-three converts to only 15 hidden hands and is cheap. Double-four can convert early to as many as "
            "252 hands and then repeatedly filters/sums that dictionary. Double-five and double-six initially exceed the "
            "threshold, so they remain in the slot representation until the state becomes smaller. This explains why "
            "per-call exact inference peaks at double-four rather than increasing monotonically with ruleset size.",
            "",
            "This is the clearest candidate for a future optimization experiment: keep double-four in the slot "
            "representation longer (or choose the representation from measured operation cost rather than hand count), "
            "then verify probability traces and fixed-seed RL weights remain identical. No production change was made here.",
            "",
            "## PPO observations",
            "",
            f"- All {total_iterations} benchmark iterations completed 16 epochs; KL early stops: {total_early_stops}; gradient-clipped iterations: {total_clipped}.",
            "- `training/rl/ppo.py:847-923` performs minibatch updates and then a full-buffer evaluation after every epoch so KL stopping and final metrics use the complete buffer.",
            "- Removing or reducing that evaluation would save substantial time but would weaken or redesign PPO control; it is not dead work under the current algorithm.",
            "- Reducing PPO epochs would also accelerate every ruleset, but it changes the optimization budget rather than fixing a ruleset-specific inefficiency.",
            "",
            "## Recommended next measurements before changing code",
            "",
            "1. A controlled double-four model-only benchmark comparing the current threshold 500 with a threshold below 252, while asserting exact probabilities at every public action.",
            "2. A short fixed-seed RL A/B for double-four after that change, requiring byte-identical trajectories/rewards and comparing rollout time only.",
            "3. Separately investigate whether PPO full-buffer evaluation can share forward results with the last epoch without changing KL semantics.",
            "4. Do not optimize generic session/reporting code first; it is too small to matter.",
            "",
            "## Artifacts",
            "",
            "- `raw_30_iterations/`: all eight isolated runs, 248 retained policy checkpoints, metrics, summaries, and logs.",
            "- `curve_diagnostics/` and `learning_curve_raw.jsonl`: the 124 fixed-panel RL-vs-random evaluations.",
            "- `run_summary.csv`: one row per repetition.",
            "- `ruleset_summary.csv`: medians, ranges, and coefficients of variation.",
            "- `learning_curve.csv` and `learning_curve_summary.csv`: point-level and ruleset-level learning results.",
            "- `top_level_sections.csv`, `ppo_sections.csv`, and `rollout_sections.csv`: long-form profiles.",
            "- `analysis_summary.json`: machine-readable conclusions and validation.",
            "- Figures `01` through `12`: timing, scaling, rollout, PPO, learning curves, and efficiency comparisons.",
            "",
        ]
    )
    (HERE / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.size": 10.5,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "savefig.bbox": "tight",
        }
    )
    runs = load_runs()
    validation = validate(runs)
    aggregates = aggregate_runs(runs)
    learning_curves = load_learning_curves(runs)
    learning_summaries = summarize_learning_curves(learning_curves, aggregates)
    run_rows = [plain_run_row(run) for run in runs]
    write_csv(HERE / "run_summary.csv", run_rows)
    write_csv(HERE / "ruleset_summary.csv", aggregates)
    write_csv(HERE / "top_level_sections.csv", profile_table(runs, "sections_seconds"))
    write_csv(HERE / "ppo_sections.csv", profile_table(runs, "ppo_sections_seconds"))
    write_csv(
        HERE / "rollout_sections.csv",
        profile_table(runs, "rollout_worker", "sections_seconds"),
    )
    write_csv(HERE / "learning_curve.csv", learning_curves)
    write_csv(HERE / "learning_curve_summary.csv", learning_summaries)
    payload = {
        "benchmark": load_json(BENCHMARK_MANIFEST)["configuration"],
        "validation": validation,
        "ruleset_summary": aggregates,
        "learning_curve_summary": learning_summaries,
    }
    (HERE / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figure_total_time(runs, aggregates)
    figure_top_level_breakdown(aggregates)
    figure_scaling(aggregates)
    figure_rollout_breakdown(runs)
    figure_learner_policy(aggregates)
    figure_ppo_breakdown(aggregates)
    figure_iteration_overlay(runs)
    figure_time_per_decision(aggregates)
    figure_learning_curve_iteration(learning_curves)
    figure_learning_curve_time(learning_curves)
    figure_learning_efficiency(learning_summaries)
    figure_learning_phases(learning_summaries)
    render_report(runs, aggregates, validation, learning_summaries)

    print("Ruleset timing analysis complete")
    for row in aggregates:
        print(
            f"{row['ruleset']}: {row['median_execution_seconds']:.2f}s, "
            f"{row['median_games_per_second']:.1f} games/s, "
            f"{row['median_decisions_per_game']:.3f} decisions/game"
        )


if __name__ == "__main__":
    main()
