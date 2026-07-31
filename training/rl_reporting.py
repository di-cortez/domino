"""Reporting and cumulative runtime profiling for RL self-play.

This module intentionally has no dependency on :mod:`training.self_play` so
logging, metrics persistence, and profile aggregation remain reusable without
reintroducing the rollout/orchestrator import cycle.
"""

import json
import os
from pathlib import Path
import secrets
import time

import numpy as np

from training.rl_rollout import REWARD_ZERO_EPSILON


TRAINING_METRICS_FORMAT = "domino_rl_training_metrics"
TRAINING_METRICS_VERSION = 2
TRAINING_METRIC_COLUMNS = (
    "iteration",
    "total_iterations",
    "games",
    "cumulative_games",
    "decisions",
    "wins_in_batch",
    "batch_win_rate",
    "moving_average_win_rate",
    "reward_mean",
    "reward_std",
    "reward_min",
    "reward_max",
    "good_pct",
    "neutral_pct",
    "bad_pct",
    "entropy",
    "gradient_norm_max",
    "applied_gradient_norm",
    "gradient_clipped",
    "value_loss",
    "moving_average_value_loss",
    "requested_minibatches",
    "effective_minibatches",
    "minibatch_sizes",
    "epochs_completed",
    "stopped_by_kl",
    "optimizer_steps",
    "final_approx_kl",
    "max_approx_kl",
    "final_clip_fraction",
    "final_policy_loss",
    "gradient_norm_mean",
    "buffer_location",
    "buffer_bytes",
    "selected_workers",
    "pool_size",
    "rollout_seconds",
    "update_seconds",
    "iteration_seconds",
    "checkpoint_written",
    "checkpoint_path",
    "elapsed_training_seconds",
)


class RLRuntimeProfile:
    """Accumulate one self-play invocation's hierarchical runtime profile."""

    def __init__(self):
        self.started = time.perf_counter()
        self.sections = {}
        self.ppo_sections = {}
        self.rollout_worker = {}
        self.ppo_optimizer_step = {}
        self.ppo_full_buffer_evaluation = {}

    def add(self, section, seconds):
        """Add elapsed seconds to one top-level runtime section."""
        self.sections[section] = self.sections.get(section, 0.0) + float(seconds)

    @staticmethod
    def merge_numeric_tree(target, source):
        """Recursively add numeric counters while preserving nested schemas."""
        for key, value in source.items():
            if isinstance(value, dict):
                RLRuntimeProfile.merge_numeric_tree(
                    target.setdefault(key, {}),
                    value,
                )
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                target[key] = target.get(key, 0) + value

    def merge_rollout_worker(self, source):
        """Merge sampled worker-side rollout timings and counters."""
        self.merge_numeric_tree(self.rollout_worker, source)

    def merge_ppo_metrics(self, metrics):
        """Merge PPO phase totals and its two detailed nested profiles."""
        for name, seconds in metrics["runtime_timing_seconds"].items():
            if name != "total":
                self.ppo_sections[name] = (
                    self.ppo_sections.get(name, 0.0) + float(seconds)
                )
        detail = metrics.get("runtime_profile_detail", {})
        self.merge_numeric_tree(
            self.ppo_optimizer_step,
            detail.get("optimizer_step", {}),
        )
        self.merge_numeric_tree(
            self.ppo_full_buffer_evaluation,
            detail.get("full_buffer_evaluation", {}),
        )

    def accounted_seconds(self):
        """Return the sum already assigned to top-level runtime sections."""
        return sum(self.sections.values())

    def finish(self, *, games, iterations, decisions, optimizer_steps):
        """Close accounting and return the stable public profile schema."""
        total_seconds = time.perf_counter() - self.started
        self.sections["unaccounted"] = max(
            0.0,
            total_seconds - self.accounted_seconds(),
        )
        return {
            "execution_count": 1,
            "games": int(games),
            "iterations": int(iterations),
            "decisions": int(decisions),
            "optimizer_steps": int(optimizer_steps),
            "execution_seconds": float(total_seconds),
            "sections_seconds": {
                name: float(seconds)
                for name, seconds in self.sections.items()
            },
            "ppo_sections_seconds": {
                name: float(seconds)
                for name, seconds in self.ppo_sections.items()
            },
            "rollout_worker": self.rollout_worker,
            "ppo_optimizer_step": self.ppo_optimizer_step,
            "ppo_full_buffer_evaluation": self.ppo_full_buffer_evaluation,
        }


def _reward_signal_summary(samples, xp=None):
    """Return compact diagnostics for finalized decision rewards.

    ``reward_std`` disambiguates a falling value loss from a merely
    low-variance batch: since a value head that has not learned anything
    predicts close to the batch mean, its loss is approximately
    ``0.5 * reward_std ** 2`` -- logging the standard deviation next to the
    loss makes that identity checkable instead of hidden behind a noisy
    scalar.

    ``xp`` should be the training run's resolved array backend (``train()``
    passes ``network.xp``); it defaults to NumPy for direct callers, which is
    fine here since this is small-scale summary math, not the training path.
    """
    if xp is None:
        xp = np
    rewards = xp.asarray([sample.policy_reward for sample in samples], dtype=float)
    local_rewards = xp.asarray([sample.local_reward for sample in samples], dtype=float)
    total = rewards.size

    good = xp.sum(rewards > REWARD_ZERO_EPSILON)
    neutral = xp.sum(xp.abs(rewards) <= REWARD_ZERO_EPSILON)
    bad = xp.sum(rewards < -REWARD_ZERO_EPSILON)

    return {
        "reward_mean": float(xp.mean(rewards)),
        "reward_std": float(xp.std(rewards)),
        "reward_min": float(xp.min(rewards)),
        "reward_max": float(xp.max(rewards)),
        "local_mean": float(xp.mean(local_rewards)),
        "good_pct": float(100.0 * good / total),
        "neutral_pct": float(100.0 * neutral / total),
        "bad_pct": float(100.0 * bad / total),
    }


def _gradient_log_text(metrics):
    """Return a compact gradient-norm string for the iteration log."""
    suffix = " clipped" if metrics.get("grad_clipped") else ""
    return f"{metrics['grad_norm']:.2f}{suffix}"


def _new_parallel_summary(requested_workers):
    """Return mutable aggregate metadata for all RL worker-pool phases."""
    return {
        "requested_workers": requested_workers,
        "initial_workers": None,
        "final_workers": None,
        "peak_worker_rss_mb": 0.0,
        "peak_total_children_rss_mb": 0.0,
        "min_available_memory_mb": None,
        "fallback_count": 0,
        "fallback_history": [],
        "attempted_worker_counts": [],
        "safety_capped": False,
        "memory_monitoring_available": True,
        "workers_cpu_only": True,
        "rollout_batches": 0,
    }


def _merge_parallel_summary(summary, run_info, *, phase, iteration):
    """Accumulate one rollout/evaluation pool run into the public summary."""
    if summary["initial_workers"] is None:
        summary["initial_workers"] = run_info.initial_workers
    summary["final_workers"] = run_info.final_workers
    summary["peak_worker_rss_mb"] = max(
        summary["peak_worker_rss_mb"],
        run_info.peak_worker_rss_mb,
    )
    summary["peak_total_children_rss_mb"] = max(
        summary["peak_total_children_rss_mb"],
        run_info.peak_total_children_rss_mb,
    )
    available = run_info.min_available_memory_mb
    if available is not None:
        current = summary["min_available_memory_mb"]
        summary["min_available_memory_mb"] = (
            available if current is None else min(current, available)
        )
    summary["fallback_count"] += run_info.fallback_count
    for item in run_info.fallback_history:
        tagged = dict(item)
        tagged["rl_phase"] = phase
        tagged["iteration"] = int(iteration)
        summary["fallback_history"].append(tagged)
    summary["attempted_worker_counts"].extend(run_info.attempted_worker_counts)
    summary["safety_capped"] = summary["safety_capped"] or run_info.safety_capped
    summary["memory_monitoring_available"] = (
        summary["memory_monitoring_available"]
        and run_info.memory_monitoring_available
    )
    summary[f"{phase}_batches"] += 1


def _print_ppo_window(rows):
    """Print the requested ten-iteration PPO aggregate without minibatch chatter."""
    rows = list(rows)
    if not rows:
        return
    count = len(rows)
    effective = [row["effective_minibatches"] for row in rows]
    epochs = [row["epochs_completed"] for row in rows]
    buffer_bytes = [row["buffer_bytes"] for row in rows]
    print(
        f"  PPO/{count}: GPI {rows[-1]['games']} | decisions "
        f"{sum(row['decisions'] for row in rows)} total/"
        f"{np.mean([row['decisions'] for row in rows]):.1f} avg | "
        f"minibatches requested {np.mean([row['requested_minibatches'] for row in rows]):.1f} avg, "
        f"effective {np.mean(effective):.1f}/{min(effective)}/{max(effective)} avg/min/max"
    )
    print(
        f"  PPO/{count}: optimizer steps {sum(row['optimizer_steps'] for row in rows)} total/"
        f"{np.mean([row['optimizer_steps'] for row in rows]):.1f} avg | epochs "
        f"{np.mean(epochs):.1f}/{min(epochs)}/{max(epochs)} avg/min/max | "
        f"KL stops {sum(row['stopped_by_kl'] for row in rows)}/{count} | final KL "
        f"{np.mean([row['final_approx_kl'] for row in rows]):.5f} avg/"
        f"{max(row['final_approx_kl'] for row in rows):.5f} max"
    )
    print(
        f"  PPO/{count}: clip fraction {np.mean([row['final_clip_fraction'] for row in rows]):.3f} | "
        f"policy loss {np.mean([row['final_policy_loss'] for row in rows]):+.4f} | "
        f"entropy {np.mean([row['final_entropy'] for row in rows]):.4f} | grad norm "
        f"{np.mean([row['gradient_norm_mean'] for row in rows]):.3f} avg/"
        f"{max(row['gradient_norm_max'] for row in rows):.3f} max"
    )
    value_rows = [row for row in rows if row.get("final_value_loss") is not None]
    if value_rows:
        explained = [
            row["final_explained_variance"]
            for row in value_rows
            if row.get("final_explained_variance") is not None
        ]
        explained_text = (
            f"{np.mean(explained):+.3f} avg"
            if explained else "undefined"
        )
        print(
            f"  PPO/{count}: value loss "
            f"{np.mean([row['final_value_loss'] for row in value_rows]):.4f} | "
            f"value clip fraction "
            f"{np.mean([row['final_value_clip_fraction'] for row in value_rows]):.3f} | "
            f"explained variance {explained_text}"
        )
    gpu_count = sum(row["buffer_location"] == "gpu" for row in rows)
    print(
        f"  PPO/{count}: buffer GPU {gpu_count}, RAM {count - gpu_count} | bytes "
        f"{np.mean(buffer_bytes):.0f} avg/{max(buffer_bytes)} max | PPO update "
        f"{sum(row['ppo_seconds'] for row in rows):.2f}s total/"
        f"{np.mean([row['ppo_seconds'] for row in rows]):.3f}s avg | rollout "
        f"{sum(row['rollout_seconds'] for row in rows):.2f}s total/"
        f"{np.mean([row['rollout_seconds'] for row in rows]):.3f}s avg"
    )


def _metrics_header(metadata):
    return {
        "format": TRAINING_METRICS_FORMAT,
        "version": TRAINING_METRICS_VERSION,
        "columns": list(TRAINING_METRIC_COLUMNS),
        "metadata": dict(metadata or {}),
    }


def _metric_values(row):
    """Project one verbose in-memory row onto the compact stable v2 schema."""
    values = {
        "iteration": row["iteration"],
        "total_iterations": row["total_iterations"],
        "games": row["games"],
        "cumulative_games": row["cumulative_games"],
        "decisions": row["decisions"],
        "wins_in_batch": row["wins_in_batch"],
        "batch_win_rate": row["batch_win_rate"],
        "moving_average_win_rate": row["moving_average_win_rate"],
        "reward_mean": row["reward_mean"],
        "reward_std": row["reward_std"],
        "reward_min": row["reward_min"],
        "reward_max": row["reward_max"],
        "good_pct": row["good_pct"],
        "neutral_pct": row["neutral_pct"],
        "bad_pct": row["bad_pct"],
        "entropy": (
            row["final_entropy"]
            if row.get("final_entropy") is not None
            else row.get("entropy")
        ),
        "gradient_norm_max": (
            row["gradient_norm_max"]
            if row.get("gradient_norm_max") is not None
            else row.get("grad_norm")
        ),
        "applied_gradient_norm": row.get("applied_grad_norm"),
        "gradient_clipped": row["grad_clipped"],
        "value_loss": row["value_loss"],
        "moving_average_value_loss": row["moving_average_value_loss"],
        "requested_minibatches": row["requested_minibatches"],
        "effective_minibatches": row["effective_minibatches"],
        "minibatch_sizes": row["minibatch_sizes"],
        "epochs_completed": row["epochs_completed"],
        "stopped_by_kl": row["stopped_by_kl"],
        "optimizer_steps": row["optimizer_steps"],
        "final_approx_kl": row["final_approx_kl"],
        "max_approx_kl": row["max_approx_kl"],
        "final_clip_fraction": row["final_clip_fraction"],
        "final_policy_loss": row["final_policy_loss"],
        "gradient_norm_mean": row["gradient_norm_mean"],
        "buffer_location": row["buffer_location"],
        "buffer_bytes": row["buffer_bytes"],
        "selected_workers": row["selected_workers"],
        "pool_size": row["pool_size"],
        "rollout_seconds": row["rollout_seconds"],
        "update_seconds": row["update_duration_s"],
        "iteration_seconds": row["iteration_duration_s"],
        "checkpoint_written": row["checkpoint_written"],
        "checkpoint_path": row["checkpoint_path"],
        "elapsed_training_seconds": row["elapsed_training_s"],
    }
    return [values[column] for column in TRAINING_METRIC_COLUMNS]


def read_training_metrics(path):
    """Read a v2 metrics header and return decoded per-iteration dictionaries."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Training metrics file is empty: {path}.")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Training metrics header is invalid: {path}.") from exc
    if (
        not isinstance(header, dict)
        or header.get("format") != TRAINING_METRICS_FORMAT
        or header.get("version") != TRAINING_METRICS_VERSION
        or header.get("columns") != list(TRAINING_METRIC_COLUMNS)
    ):
        raise ValueError(
            f"Unsupported training metrics format in {path}; "
            f"expected v{TRAINING_METRICS_VERSION}."
        )
    rows = []
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        try:
            values = json.loads(line)
        except json.JSONDecodeError:
            if line_number == len(lines):
                break
            raise ValueError(
                f"Training metrics row {line_number} is invalid."
            )
        if not isinstance(values, list) or len(values) != len(
            TRAINING_METRIC_COLUMNS
        ):
            raise ValueError(
                f"Training metrics row {line_number} has the wrong column count."
            )
        rows.append(dict(zip(TRAINING_METRIC_COLUMNS, values)))
    return header, rows


def _prepare_metrics_file(path, start_iteration, metadata=None):
    """Create a v2 trace or truncate it to the exact resumed iteration."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _metrics_header(metadata)
    retained = []
    if start_iteration and path.is_file():
        existing_header, existing_rows = read_training_metrics(path)
        existing_hash = existing_header.get("metadata", {}).get(
            "run_configuration_sha256"
        )
        requested_hash = header.get("metadata", {}).get(
            "run_configuration_sha256"
        )
        if existing_hash != requested_hash:
            raise ValueError(
                "Training metrics configuration hash does not match the "
                "resumed run."
            )
        header = existing_header
        for row in existing_rows:
            if int(row["iteration"]) <= int(start_iteration):
                retained.append(row)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
    )
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(header, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
            for row in retained:
                values = [row[column] for column in TRAINING_METRIC_COLUMNS]
                stream.write(
                    json.dumps(values, separators=(",", ":"), allow_nan=False)
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _write_metrics_row(stream, row):
    """Append and durably flush one self-play metrics row."""
    stream.write(
        json.dumps(
            _metric_values(row),
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    stream.flush()
    os.fsync(stream.fileno())
