"""Controlled, resumable RL sweep over games per iteration.

Every point starts from the same supervised checkpoint and trains on exactly
the same number of games. Games per iteration, PPO minibatches, and update
count deliberately change, while opponent-pool refresh stays fixed by
cumulative training games. There is no replay or cross-iteration reuse.

Examples:
  python train_script/run_rl_games_per_iteration_sweep.py --preset standard --dry-run
  python train_script/run_rl_games_per_iteration_sweep.py --preset quick --critic-mode off --run-id gpi_quick_off
  python train_script/run_rl_games_per_iteration_sweep.py --run-id gpi_quick_off --resume
  python train_script/run_rl_games_per_iteration_sweep.py --run-id gpi_quick_off --report-only
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from diagnostics.pairwise import run_pairwise
from diagnostics.parallel_runner import MAX_DIAGNOSTIC_WORKERS, MAX_PARALLEL_WORKERS
from diagnostics.rl_gpi_sweep_report import build_report, file_sha256
from training import self_play
from utils.resource_limits import gpu_memory_info, system_memory_info


SCHEMA_VERSION = "2.0"
DEFAULT_GPI_VALUES = (40, 80, 160, 320, 640, 960, 1280)
DEFAULT_TOTAL_TRAINING_GAMES = 384_000
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_DIAGNOSTIC_GAMES = 10_000
DEFAULT_CHECKPOINT_COUNT = 10
DEFAULT_SL_WEIGHTS_PATH = ROOT / "models" / "domino_sl_weights.npz"
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_GAMMA = 1.0
DEFAULT_VALUE_COEF = 0.5
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_CRITIC_MODE = "off"
DEFAULT_TRAINING_OPPONENT = "self_play"
DEFAULT_DEVICE = "auto"
DEFAULT_RL_WORKERS = "auto"
DEFAULT_DIAGNOSTIC_WORKERS = 1
DEFAULT_POOL_REFRESH_GAMES = self_play.DEFAULT_POOL_REFRESH_GAMES
DEFAULT_MAX_POOL_SIZE = 50
DEFAULT_REWARD_SCHEMA = "default"
DEFAULT_CLIP_GRAD_NORM = 5.0

PRESETS = {
    "quick": {
        "total_training_games": 38_400,
        "seeds": (42,),
        "diagnostic_games": 1_000,
        "checkpoint_count": 10,
    },
    "standard": {
        "total_training_games": DEFAULT_TOTAL_TRAINING_GAMES,
        "seeds": DEFAULT_SEEDS,
        "diagnostic_games": DEFAULT_DIAGNOSTIC_GAMES,
        "checkpoint_count": DEFAULT_CHECKPOINT_COUNT,
    },
    "thorough": {
        "total_training_games": 1_152_000,
        "seeds": (42, 43, 44, 45, 46),
        "diagnostic_games": 20_000,
        "checkpoint_count": 10,
    },
}

PLAN_FIELDS = (
    "execution_order",
    "run_key",
    "seed",
    "critic",
    "games_per_iteration",
    "iterations",
    "total_training_games",
    "status",
)
_ITERATION_PATTERN = re.compile(r"_iter(\d{6})\.npz$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(json_ready(value), indent=2, ensure_ascii=False) + "\n")


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    fields = list(fields or dict.fromkeys(key for row in rows for key in row) or ("status",))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({field: row.get(field) for field in fields} for row in rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_sl_weights_once(path: Path | str) -> dict[str, np.ndarray]:
    """Load and validate the shared SL checkpoint into reusable host arrays."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing supervised checkpoint: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            required = ("W1", "b1", "W2", "b2", "W3", "b3")
            missing = [name for name in required if name not in data]
            if missing:
                raise ValueError(f"SL checkpoint {path} is missing {missing}.")
            weights = {name: np.asarray(data[name]).copy() for name in required}
    except (OSError, ValueError) as exc:
        raise ValueError(f"SL checkpoint {path} is not a readable compatible NPZ: {exc}") from exc
    if weights["W1"].shape[1] != 168 or weights["W3"].shape[0] != 56:
        raise ValueError(
            f"SL checkpoint {path} has input/output {weights['W1'].shape[1]}/"
            f"{weights['W3'].shape[0]}; expected 168/56."
        )
    return weights


def make_run_key(critic: str | bool, gpi: int, total_games: int, seed: int) -> str:
    critic_label = "on" if critic is True or critic == "on" else "off"
    return (
        f"critic_{critic_label}_gpi{int(gpi):04d}_"
        f"games{int(total_games):07d}_seed{int(seed)}"
    )


def critic_values(mode: str) -> tuple[str, ...]:
    return ("off", "on") if mode == "both" else (mode,)


def validate_configuration(config: dict[str, Any]) -> None:
    """Fail before the first game when the experiment is not exact/safe."""
    total = int(config["total_training_games"])
    gpis = tuple(int(value) for value in config["gpi_values"])
    seeds = tuple(int(value) for value in config["seeds"])
    checkpoint_count = int(config["checkpoint_count"])
    if total <= 0:
        raise ValueError("total_training_games must be positive.")
    if not gpis or any(value <= 0 for value in gpis):
        raise ValueError("games_per_iteration values must be positive and non-empty.")
    if len(set(gpis)) != len(gpis):
        raise ValueError("games_per_iteration values must be unique.")
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")
    invalid = [value for value in gpis if total % value]
    if invalid:
        raise ValueError(
            f"total_training_games={total} is not exactly divisible by GPI values "
            f"{invalid}; no training was started."
        )
    if int(config["diagnostic_games"]) <= 0:
        raise ValueError("diagnostic_games must be positive.")
    if int(config["pool_refresh_games"]) <= 0:
        raise ValueError("pool_refresh_games must be positive.")
    if checkpoint_count < 1:
        raise ValueError("checkpoint_count must be at least one.")
    bad_checkpoints = []
    for gpi in gpis:
        iterations = total // gpi
        if iterations < checkpoint_count or iterations % checkpoint_count:
            bad_checkpoints.append((gpi, iterations))
    if bad_checkpoints:
        raise ValueError(
            "Every iteration count must be >= and divisible by checkpoint_count="
            f"{checkpoint_count}; invalid (GPI, iterations): {bad_checkpoints}."
        )
    workers = config["rl_workers"]
    if workers != "auto" and not 1 <= int(workers) <= MAX_PARALLEL_WORKERS:
        raise ValueError(f"rl_workers must be auto or 1..{MAX_PARALLEL_WORKERS}.")
    diagnostic_workers = int(config["diagnostic_workers"])
    if not 1 <= diagnostic_workers <= MAX_DIAGNOSTIC_WORKERS:
        raise ValueError(f"diagnostic_workers must be 1..{MAX_DIAGNOSTIC_WORKERS}.")
    if config["critic_mode"] not in {"off", "on", "both"}:
        raise ValueError("critic_mode must be off, on, or both.")


def build_run_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a deterministic order that varies the GPI permutation by seed."""
    validate_configuration(config)
    plan: list[dict[str, Any]] = []
    total = int(config["total_training_games"])
    canonical_gpis = tuple(int(value) for value in config["gpi_values"])
    execution_order = 1
    for critic in critic_values(config["critic_mode"]):
        for seed in config["seeds"]:
            ordered_gpis = list(canonical_gpis)
            random.Random(int(seed)).shuffle(ordered_gpis)
            for gpi in ordered_gpis:
                plan.append({
                    "execution_order": execution_order,
                    "run_key": make_run_key(critic, gpi, total, int(seed)),
                    "seed": int(seed),
                    "critic": critic,
                    "games_per_iteration": int(gpi),
                    "iterations": total // int(gpi),
                    "total_training_games": total,
                    "status": "planned",
                })
                execution_order += 1
    return plan


def stable_diagnostic_seed(training_seed: int, opponent: str) -> int:
    """Derive a stable seed independent of GPI and Python's hash randomization."""
    digest = hashlib.sha256(f"{int(training_seed)}:{opponent}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _configuration_differences(saved: dict[str, Any], requested: dict[str, Any]) -> list[str]:
    differences = []
    for key in sorted(set(saved) | set(requested)):
        if saved.get(key) != requested.get(key):
            differences.append(f"{key}: saved={saved.get(key)!r}, requested={requested.get(key)!r}")
    return differences


def truncate_metrics_file(path: Path | str, start_iteration: int) -> list[dict[str, Any]]:
    """Keep only metrics at/before the validated resume checkpoint."""
    path = Path(path)
    retained: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                item = json.loads(line)
                if int(item["iteration"]) <= int(start_iteration):
                    retained.append(item)
    except FileNotFoundError:
        return []
    content = "".join(json.dumps(item, sort_keys=True) + "\n" for item in retained)
    atomic_write_text(path, content)
    return retained


def _iteration_from_path(path: Path) -> int | None:
    match = _ITERATION_PATTERN.search(path.name)
    return int(match.group(1)) if match else None


def find_latest_resume_pair(base_path: Path | str, target_iterations: int) -> tuple[Path, Path, dict[str, Any]] | None:
    """Return the newest valid numbered weights/state pair at or below target."""
    base_path = Path(base_path)
    stem = base_path.stem
    candidates = []
    for weights_path in base_path.parent.glob(f"{stem}_iter*.npz"):
        if ".resume." in weights_path.name:
            continue
        iteration = _iteration_from_path(weights_path)
        if iteration is None or iteration > int(target_iterations):
            continue
        state_path = self_play.resume_state_path(weights_path)
        if not state_path.is_file():
            continue
        try:
            metadata, _pool = self_play.load_resume_state(weights_path, state_path)
        except (OSError, ValueError):
            continue
        if int(metadata.get("completed_iteration", -1)) == iteration:
            candidates.append((iteration, weights_path, state_path, metadata))
    if not candidates:
        return None
    _iteration, weights, state, metadata = max(candidates, key=lambda item: item[0])
    return weights, state, metadata


def _git_environment() -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            return subprocess.run(
                arguments,
                cwd=ROOT,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("git", "status", "--porcelain")
    return {"git_commit": run("git", "rev-parse", "HEAD"), "git_dirty": bool(status)}


def collect_environment(config: dict[str, Any]) -> dict[str, Any]:
    memory = system_memory_info()
    gpu = gpu_memory_info()
    if config.get("csv_only"):
        openpyxl_version = None
    else:
        try:
            import openpyxl
            openpyxl_version = openpyxl.__version__
        except ImportError:
            openpyxl_version = None
    gpu_name = None
    try:
        import cupy
        properties = cupy.cuda.runtime.getDeviceProperties(0)
        gpu_name = properties.get("name")
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
    except Exception:
        pass
    return {
        "command": sys.argv,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "openpyxl": openpyxl_version,
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "ram_total_bytes": memory.total if memory else None,
        "ram_available_bytes_at_start": memory.available if memory else None,
        "device_requested": config["device"],
        "gpu_name": gpu_name,
        "gpu_total_bytes": gpu.total if gpu else None,
        "gpu_available_bytes_at_start": gpu.available if gpu else None,
        **_git_environment(),
    }


class ExperimentJournal:
    def __init__(self, experiment_dir: Path, logger: logging.Logger):
        self.experiment_dir = experiment_dir
        self.logger = logger
        self.events_path = experiment_dir / "events.jsonl"

    def event(self, event: str, *, run_key: str = "experiment", stage: str = "orchestration", **fields: Any) -> None:
        record = {"timestamp": utc_now(), "event": event, "run_key": run_key, "stage": stage, **json_ready(fields)}
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.events_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.logger.info("[%s][%s] %s", run_key, stage, fields.get("message", event))
        run_dir = self.experiment_dir / "runs" / run_key
        if run_key != "experiment" and run_dir.exists():
            with open(run_dir / "run.log", "a", encoding="utf-8") as stream:
                stream.write(f"{record['timestamp']} {stage} {event} {fields.get('message', '')}\n")


def _setup_logging(experiment_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"rl_gpi_sweep.{experiment_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(experiment_dir / "sweep.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _write_plan(experiment_dir: Path, manifest: dict[str, Any]) -> None:
    atomic_write_csv(experiment_dir / "run_plan.csv", manifest["plan"], PLAN_FIELDS)
    atomic_write_json(experiment_dir / "experiment_manifest.json", manifest)


def _set_status(experiment_dir: Path, manifest: dict[str, Any], point: dict[str, Any], status: str, failure: dict[str, Any] | None = None) -> None:
    point["status"] = status
    point["updated_at"] = utc_now()
    if failure is not None:
        point["failure"] = failure
    elif status != "failed":
        point.pop("failure", None)
    _write_plan(experiment_dir, manifest)


def _metrics_to_csv(metrics_path: Path, csv_path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with open(metrics_path, "r", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    except FileNotFoundError:
        pass
    atomic_write_csv(csv_path, rows)
    return rows


def _metrics_elapsed_s(metrics: list[dict[str, Any]]) -> float:
    """Sum callback elapsed clocks across original and resumed train segments."""
    total = 0.0
    previous = 0.0
    for item in metrics:
        current = float(item.get("elapsed_training_s", 0.0))
        if current < previous:
            total += previous
        previous = current
    return total + previous


def _valid_final_model(
    run_dir: Path,
    iterations: int,
    configuration_fingerprint: str | None = None,
) -> dict[str, Any] | None:
    metadata = None
    try:
        with open(run_dir / "final_model.json", "r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        model = Path(metadata["model_path"])
        state = Path(metadata["resume_state_path"])
        if int(metadata["final_iteration"]) != int(iterations):
            return None
        if (
            configuration_fingerprint is not None
            and metadata.get("configuration_fingerprint") != configuration_fingerprint
        ):
            return None
        if not model.is_file() or not state.is_file():
            return None
        resume_metadata, _pool = self_play.load_resume_state(model, state)
        if int(resume_metadata["completed_iteration"]) != int(iterations):
            return None
        if metadata["model_sha256"] != file_sha256(model):
            return None
        return metadata
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _diagnostic_is_valid(directory: Path, *, games: int, seed: int, opponent: str, model_hash: str) -> bool:
    try:
        with open(directory / "diagnostic_config.json", "r", encoding="utf-8") as stream:
            config = json.load(stream)
        with open(directory / "summary.json", "r", encoding="utf-8") as stream:
            summary = json.load(stream)
        with open(directory / "games.csv", "r", encoding="utf-8", newline="") as stream:
            row_count = sum(1 for _ in csv.reader(stream)) - 1
    except (OSError, json.JSONDecodeError, csv.Error):
        return False
    return (
        config.get("opponent") == opponent
        and int(config.get("game_count", -1)) == int(games)
        and int(config.get("seed", -1)) == int(seed)
        and config.get("model_sha256") == model_hash
        and int(summary.get("game_count", -1)) == int(games)
        and row_count == int(games)
    )


def _run_diagnostic(
    run_dir: Path,
    opponent: str,
    *,
    model_path: Path,
    model_hash: str,
    game_count: int,
    seed: int,
    workers: int,
    generate_plots: bool,
    quiet: bool,
    journal: ExperimentJournal,
    run_key: str,
    resume: bool,
) -> dict[str, Any]:
    output_dir = run_dir / "diagnostics" / f"vs_{opponent}"
    if resume and _diagnostic_is_valid(
        output_dir, games=game_count, seed=seed, opponent=opponent, model_hash=model_hash
    ):
        journal.event("diagnostic_reused", run_key=run_key, stage=f"diagnostic_{opponent}")
        with open(output_dir / "summary.json", "r", encoding="utf-8") as stream:
            return json.load(stream)
    journal.event("diagnostic_started", run_key=run_key, stage=f"diagnostic_{opponent}", games=game_count, seed=seed)
    result = run_pairwise(
        "rl",
        opponent,
        game_count=game_count,
        weights=model_path,
        seed=seed,
        output_dir=output_dir,
        workers=workers,
        generate_plots=generate_plots,
        print_console_summary=not quiet,
        print_memory_summary=False,
    )
    atomic_write_json(output_dir / "diagnostic_config.json", {
        "schema_version": SCHEMA_VERSION,
        "opponent": opponent,
        "game_count": game_count,
        "seed": seed,
        "model_path": str(model_path.resolve()),
        "model_sha256": model_hash,
        "workers_requested": workers,
        "generated_at": utc_now(),
    })
    journal.event("diagnostic_completed", run_key=run_key, stage=f"diagnostic_{opponent}", duration_s=result["duration_s"])
    return result["summary"]


def _release_backend_cache() -> None:
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _run_point(
    point: dict[str, Any],
    *,
    config: dict[str, Any],
    experiment_dir: Path,
    model_dir: Path,
    manifest: dict[str, Any],
    sl_weights_data: dict[str, np.ndarray],
    sl_hash: str,
    journal: ExperimentJournal,
    resume: bool,
) -> None:
    run_key = point["run_key"]
    run_dir = experiment_dir / "runs" / run_key
    run_dir.mkdir(parents=True, exist_ok=True)
    point_model_dir = model_dir / run_key
    point_model_dir.mkdir(parents=True, exist_ok=True)
    base_model_path = point_model_dir / "domino_rl.npz"
    metrics_path = run_dir / "training_metrics.jsonl"
    metrics_csv_path = run_dir / "training_metrics.csv"
    use_critic = point["critic"] == "on"
    ppo_enabled = not use_critic
    normalize_advantages = (
        ppo_enabled
        if config["normalize_advantages"] is None
        else bool(config["normalize_advantages"])
    )
    run_config = {
        "schema_version": SCHEMA_VERSION,
        **{key: value for key, value in config.items() if key not in {"seeds", "gpi_values"}},
        "run_key": run_key,
        "execution_order": point["execution_order"],
        "seed": point["seed"],
        "critic": point["critic"],
        "use_value_head": use_critic,
        "rl_training_algorithm": (
            self_play.PPO_TRAINING_ALGORITHM
            if ppo_enabled else self_play.LEGACY_TRAINING_ALGORITHM
        ),
        "ppo_enabled": ppo_enabled,
        "normalize_advantages": normalize_advantages,
        "games_per_iteration": point["games_per_iteration"],
        "iterations": point["iterations"],
        "total_training_games": point["total_training_games"],
        "checkpoint_interval": point["iterations"] // config["checkpoint_count"],
        "sl_weights_presented": config["sl_weights_path"],
        "sl_weights_resolved": str(Path(config["sl_weights_path"]).resolve()),
        "sl_sha256": sl_hash,
        "git_commit": manifest["environment"].get("git_commit"),
        "model_base_path": str(base_model_path.resolve()),
    }
    fingerprint_source = {key: value for key, value in run_config.items() if key not in {"configuration_fingerprint"}}
    run_config["configuration_fingerprint"] = canonical_sha256(fingerprint_source)
    existing_config = None
    try:
        with open(run_dir / "run_config.json", "r", encoding="utf-8") as stream:
            existing_config = json.load(stream)
    except FileNotFoundError:
        pass
    if existing_config and existing_config.get("configuration_fingerprint") != run_config["configuration_fingerprint"]:
        differences = _configuration_differences(existing_config, run_config)
        raise ValueError("Run configuration changed; resume refused: " + "; ".join(differences))
    atomic_write_json(run_dir / "run_config.json", run_config)
    point_started = time.perf_counter()
    final_model = (
        _valid_final_model(
            run_dir,
            point["iterations"],
            run_config["configuration_fingerprint"],
        )
        if resume
        else None
    )

    if final_model is None:
        _set_status(experiment_dir, manifest, point, "training")
        journal.event("run_started", run_key=run_key, stage="training")
        resume_pair = find_latest_resume_pair(base_model_path, point["iterations"]) if resume else None
        resume_kwargs: dict[str, Any] = {}
        prior_training_wall_s = 0.0
        if resume_pair is not None:
            resume_weights, resume_state, resume_metadata = resume_pair
            start_iteration = int(resume_metadata["completed_iteration"])
            selected_device, _fallback_reason = self_play.choose_safe_rl_device(
                config["device"]
            )
            expected_resume_configuration = self_play._resume_configuration(
                total_training_games=point["total_training_games"],
                selected_gpi=point["games_per_iteration"],
                selected_workers=int(
                    resume_metadata["configuration"]["selected_workers"]
                ),
                rl_training_algorithm=run_config["rl_training_algorithm"],
                training_opponent=config["training_opponent"],
                learning_rate=config["learning_rate"],
                entropy_coef=config["entropy_coef"],
                pool_refresh_games=config["pool_refresh_games"],
                max_pool_size=config["max_pool_size"],
                use_value_head=use_critic,
                value_coef=config["value_coef"],
                gamma=config["gamma"],
                reward_schema=config["reward_schema"],
                clip_grad_norm=config["clip_grad_norm"],
                normalize_advantages=normalize_advantages,
                effective_seed=point["seed"],
                device=selected_device,
                sl_weights_sha256=sl_hash,
                ppo_clip_epsilon=self_play.DEFAULT_CLIP_EPSILON,
                ppo_target_kl=self_play.DEFAULT_TARGET_KL,
                ppo_stop_kl=self_play.DEFAULT_STOP_KL,
                ppo_max_epochs=self_play.DEFAULT_MAX_EPOCHS,
                ppo_min_minibatches=self_play.DEFAULT_MIN_MINIBATCHES,
                ppo_max_minibatches=self_play.DEFAULT_MAX_MINIBATCHES,
                ppo_games_per_minibatch_scale=(
                    self_play.DEFAULT_GAMES_PER_MINIBATCH_SCALE
                ),
                ppo_min_decisions_per_minibatch=(
                    self_play.DEFAULT_MIN_DECISIONS_PER_MINIBATCH
                ),
                prefer_gpu_buffer=True,
                gpu_buffer_safety_fraction=(
                    self_play.DEFAULT_GPU_BUFFER_SAFETY_FRACTION
                ),
            )
            self_play._validate_resume_configuration(
                resume_metadata,
                expected_resume_configuration,
            )
            retained_metrics = truncate_metrics_file(metrics_path, start_iteration)
            prior_training_wall_s = _metrics_elapsed_s(retained_metrics)
            if start_iteration == point["iterations"]:
                final_model = {
                    "schema_version": SCHEMA_VERSION,
                    "model_path": str(resume_weights.resolve()),
                    "model_sha256": file_sha256(resume_weights),
                    "model_size_bytes": resume_weights.stat().st_size,
                    "final_iteration": start_iteration,
                    "resume_state_path": str(resume_state.resolve()),
                    "configuration_fingerprint": run_config["configuration_fingerprint"],
                }
                atomic_write_json(run_dir / "final_model.json", final_model)
            else:
                resume_kwargs = {
                    "start_iteration": start_iteration,
                    "resume_weights_path": str(resume_weights),
                    "resume_state_file": str(resume_state),
                }
        else:
            truncate_metrics_file(metrics_path, 0)

        if final_model is None:
            metrics_stream = open(metrics_path, "a", encoding="utf-8")

            def metrics_callback(item: dict[str, Any]) -> None:
                metrics_stream.write(json.dumps(item, sort_keys=True) + "\n")
                metrics_stream.flush()
                os.fsync(metrics_stream.fileno())
                if item.get("checkpoint_written"):
                    journal.event(
                        "checkpoint_saved",
                        run_key=run_key,
                        stage="training",
                        iteration=item["iteration"],
                        checkpoint_path=item["checkpoint_path"],
                    )

            def status_callback(message: str) -> None:
                journal.event("autotune_status", run_key=run_key, stage="training", message=message)

            training_started = time.perf_counter()
            try:
                summary = self_play.train(
                    iterations=point["iterations"],
                    gpi=point["games_per_iteration"],
                    training_opponent=config["training_opponent"],
                    learning_rate=config["learning_rate"],
                    entropy_coef=config["entropy_coef"],
                    checkpoint_interval=run_config["checkpoint_interval"],
                    pool_refresh_games=config["pool_refresh_games"],
                    max_pool_size=config["max_pool_size"],
                    sl_weights_path=config["sl_weights_path"],
                    sl_weights_data=sl_weights_data,
                    rl_weights_path=str(base_model_path),
                    fresh_from_sl=not bool(resume_kwargs),
                    quiet=config["quiet_training"],
                    use_value_head=use_critic,
                    value_coef=config["value_coef"],
                    gamma=config["gamma"],
                    reward_schema=config["reward_schema"],
                    clip_grad_norm=config["clip_grad_norm"],
                    normalize_advantages=normalize_advantages,
                    seed=point["seed"],
                    device=config["device"],
                    workers=config["rl_workers"],
                    adaptive_tuning_path=str(run_dir / "adaptive_tuning.json"),
                    numbered_checkpoints=True,
                    metrics_callback=metrics_callback,
                    status_callback=status_callback,
                    ppo_enabled=ppo_enabled,
                    **resume_kwargs,
                )
            finally:
                metrics_stream.close()
            training_wall = time.perf_counter() - training_started
            metrics = _metrics_to_csv(metrics_path, metrics_csv_path)
            if int(summary["completed_training_games"]) != point["total_training_games"]:
                raise RuntimeError("Training summary violated the exact game budget.")
            if int(summary["effective_seed"]) != point["seed"] or bool(summary["use_value_head"]) != use_critic:
                raise RuntimeError("Training summary identity does not match the run plan.")
            summary["training_wall_s"] = prior_training_wall_s + training_wall
            summary["total_decision_samples"] = sum(int(item.get("decision_sample_count", 0)) for item in metrics)
            summary["decisions_per_game"] = summary["total_decision_samples"] / point["total_training_games"]
            summary["total_rollout_duration_s"] = sum(float(item.get("rollout_duration_s", 0.0)) for item in metrics)
            summary["total_update_duration_s"] = sum(float(item.get("update_duration_s", 0.0)) for item in metrics)
            summary["clipped_iteration_count"] = sum(bool(item.get("grad_clipped")) for item in metrics)
            summary["clipped_iteration_rate"] = summary["clipped_iteration_count"] / len(metrics) if metrics else 0.0
            atomic_write_json(run_dir / "training_summary.json", summary)
            final_path = Path(summary["rl_weights_path"]).resolve()
            final_state = self_play.resume_state_path(final_path).resolve()
            if _iteration_from_path(final_path) != point["iterations"]:
                raise RuntimeError(f"Final model {final_path} is not iteration {point['iterations']}.")
            self_play.load_resume_state(final_path, final_state)
            final_model = {
                "schema_version": SCHEMA_VERSION,
                "model_path": str(final_path),
                "model_sha256": file_sha256(final_path),
                "model_size_bytes": final_path.stat().st_size,
                "final_iteration": point["iterations"],
                "resume_state_path": str(final_state),
                "configuration_fingerprint": run_config["configuration_fingerprint"],
            }
            atomic_write_json(run_dir / "final_model.json", final_model)
            journal.event("training_completed", run_key=run_key, stage="training", training_wall_s=training_wall)
    else:
        journal.event("training_reused", run_key=run_key, stage="training")

    _set_status(experiment_dir, manifest, point, "trained")
    _set_status(experiment_dir, manifest, point, "diagnosing")
    model_path = Path(final_model["model_path"])
    heuristic_seed = stable_diagnostic_seed(point["seed"], "heuristic")
    random_seed = stable_diagnostic_seed(point["seed"], "random")
    heuristic_summary = _run_diagnostic(
        run_dir,
        "heuristic",
        model_path=model_path,
        model_hash=final_model["model_sha256"],
        game_count=config["diagnostic_games"],
        seed=heuristic_seed,
        workers=config["diagnostic_workers"],
        generate_plots=not config["diag_no_plots"],
        quiet=config["quiet_training"],
        journal=journal,
        run_key=run_key,
        resume=resume,
    )
    random_summary = _run_diagnostic(
        run_dir,
        "random",
        model_path=model_path,
        model_hash=final_model["model_sha256"],
        game_count=config["diagnostic_games"],
        seed=random_seed,
        workers=config["diagnostic_workers"],
        generate_plots=not config["diag_no_plots"],
        quiet=config["quiet_training"],
        journal=journal,
        run_key=run_key,
        resume=resume,
    )
    summary_path = run_dir / "training_summary.json"
    try:
        with open(summary_path, "r", encoding="utf-8") as stream:
            training_summary = json.load(stream)
    except FileNotFoundError:
        training_summary = {
            "iterations": point["iterations"],
            "games_per_iteration": point["games_per_iteration"],
            "total_training_games": point["total_training_games"],
            "effective_seed": point["seed"],
            "use_value_head": use_critic,
            "device": None,
            "rl_weights_path": str(model_path),
            "training_wall_s": _metrics_elapsed_s(
                _metrics_to_csv(metrics_path, metrics_csv_path)
            ),
        }
    training_summary["diagnostic_heuristic_s"] = heuristic_summary.get("duration_s")
    training_summary["diagnostic_random_s"] = random_summary.get("duration_s")
    training_summary["point_total_wall_s"] = time.perf_counter() - point_started
    atomic_write_json(summary_path, training_summary)
    _set_status(experiment_dir, manifest, point, "complete")
    journal.event("run_completed", run_key=run_key, stage="complete", point_total_wall_s=training_summary["point_total_wall_s"])


def _print_plan(config: dict[str, Any], plan: list[dict[str, Any]], paths: dict[str, Path]) -> None:
    print("RL games-per-iteration sweep plan")
    print("=" * 86)
    print("Important: training games, SL checkpoint, hyperparameters and diagnostics are controlled;")
    print("GPI and update count deliberately change; pool refresh is fixed by training games.")
    print(f"Canonical GPI values: {list(config['gpi_values'])}")
    print("order | critic | seed | GPI | iterations | training games | run key")
    for point in plan:
        print(
            f"{point['execution_order']:5d} | {point['critic']:6s} | {point['seed']:4d} | "
            f"{point['games_per_iteration']:4d} | {point['iterations']:10d} | "
            f"{point['total_training_games']:14d} | {point['run_key']}"
        )
    models = len(plan)
    print("-" * 86)
    print(f"Models: {models}")
    print(f"Aggregate training games: {models * config['total_training_games']:,}")
    print(f"Final diagnostic games: {models * 2 * config['diagnostic_games']:,}")
    print(f"Results: {paths['results_dir']}")
    print(f"Models: {paths['model_dir']}")
    print(f"Report: {paths['report_output_dir']}")
    print("External points run sequentially in the order shown above.")


def resolve_configuration(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Path]]:
    preset = PRESETS[args.preset]
    total = args.total_training_games if args.total_training_games is not None else preset["total_training_games"]
    seeds = tuple(args.seeds) if args.seeds is not None else tuple(preset["seeds"])
    diagnostic_games = args.diagnostic_games if args.diagnostic_games is not None else preset["diagnostic_games"]
    checkpoint_count = args.checkpoint_count if args.checkpoint_count is not None else preset["checkpoint_count"]
    run_id = args.run_id
    if run_id is None and args.results_dir is not None:
        run_id = Path(args.results_dir).resolve().name
    if run_id is None:
        if args.resume or args.report_only:
            raise ValueError("--resume and --report-only require --run-id or an unambiguous --results-dir.")
        run_id = datetime.now().strftime("gpi_%Y%m%d_%H%M%S")
    results_dir = Path(args.results_dir).resolve() if args.results_dir else (ROOT / "diagnostics" / "results" / "rl_gpi_sweep" / run_id).resolve()
    model_dir = Path(args.model_dir).resolve() if args.model_dir else (ROOT / "models" / "rl_gpi_sweep" / run_id).resolve()
    report_output_dir = Path(args.report_output_dir).resolve() if args.report_output_dir else (results_dir / "report").resolve()
    config = {
        "preset": args.preset,
        "run_id": run_id,
        "total_training_games": int(total),
        "gpi_values": tuple(args.games_per_iteration_values or DEFAULT_GPI_VALUES),
        "seeds": seeds,
        "critic_mode": args.critic_mode,
        "sl_weights_path": str(Path(args.sl_weights_path).resolve()),
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "value_coef": args.value_coef,
        "entropy_coef": args.entropy_coef,
        "training_opponent": args.training_opponent,
        "reward_schema": args.reward_schema,
        "clip_grad_norm": args.clip_grad_norm,
        "normalize_advantages": args.normalize_advantages,
        "pool_refresh_games": args.pool_refresh_games,
        "max_pool_size": args.max_pool_size,
        "checkpoint_count": int(checkpoint_count),
        "diagnostic_games": int(diagnostic_games),
        "device": args.device,
        "rl_workers": args.rl_workers,
        "diagnostic_workers": args.diagnostic_workers,
        "diag_no_plots": bool(args.diag_no_plots),
        "csv_only": bool(args.csv_only),
        "quiet_training": bool(args.quiet_training),
        "results_dir": str(results_dir),
        "model_dir": str(model_dir),
        "report_output_dir": str(report_output_dir),
    }
    validate_configuration(config)
    return config, {"results_dir": results_dir, "model_dir": model_dir, "report_output_dir": report_output_dir}


def run_experiment(args: argparse.Namespace) -> dict[str, Any] | None:
    config, paths = resolve_configuration(args)
    if args.report_only:
        if not (paths["results_dir"] / "experiment_manifest.json").is_file():
            raise FileNotFoundError(f"No experiment manifest in {paths['results_dir']}.")
        return build_report(
            paths["results_dir"],
            paths["report_output_dir"],
            csv_only=config["csv_only"],
            generate_plots=not config["diag_no_plots"],
        )
    sl_weights_data = load_sl_weights_once(config["sl_weights_path"])
    sl_hash = file_sha256(config["sl_weights_path"])
    if not config["csv_only"]:
        try:
            import openpyxl  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "openpyxl is required before training. Install with: "
                "python -m pip install -r train_script/requirements_gpi_sweep.txt; "
                "or use --csv-only."
            ) from exc
    plan = build_run_plan(config)
    _print_plan(config, plan, paths)
    if args.dry_run:
        return {"configuration": config, "plan": plan, "paths": {key: str(value) for key, value in paths.items()}}

    experiment_dir = paths["results_dir"]
    manifest_path = experiment_dir / "experiment_manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume: missing {manifest_path}.")
        with open(manifest_path, "r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        saved_config = manifest.get("configuration", {})
        differences = _configuration_differences(saved_config, json_ready(config))
        if differences:
            raise ValueError("Experiment configuration changed; resume refused:\n  " + "\n  ".join(differences))
        plan = manifest["plan"]
    else:
        if experiment_dir.exists() or paths["model_dir"].exists():
            raise FileExistsError(
                "A new experiment refuses to overwrite an existing results/model directory: "
                f"{experiment_dir} or {paths['model_dir']}. Use --resume or a new --run-id."
            )
        experiment_dir.mkdir(parents=True)
        paths["model_dir"].mkdir(parents=True)
        environment = collect_environment(config)
        models = len(plan)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": config["run_id"],
            "started_at": utc_now(),
            "finished_at": None,
            "duration_s": None,
            "status": "running",
            "configuration": json_ready(config),
            "configuration_fingerprint": canonical_sha256(config),
            "canonical_gpi_values": list(config["gpi_values"]),
            "plan": plan,
            "environment": environment,
            "sl_checkpoint": {
                "presented_path": args.sl_weights_path,
                "resolved_path": str(Path(config["sl_weights_path"]).resolve()),
                "sha256": sl_hash,
            },
            "budget": {
                "model_count": models,
                "training_games_per_model": config["total_training_games"],
                "aggregate_training_games": models * config["total_training_games"],
                "final_diagnostic_games_per_model": 2 * config["diagnostic_games"],
            },
            "methodology_note": "Training games, initial SL checkpoint, pool-refresh frequency, hyperparameters and diagnostics are controlled. GPI and update count deliberately change.",
        }
        _write_plan(experiment_dir, manifest)

    logger = _setup_logging(experiment_dir)
    journal = ExperimentJournal(experiment_dir, logger)
    started = time.perf_counter()
    previous_duration_s = float(manifest.get("duration_s") or 0.0) if args.resume else 0.0
    journal.event("experiment_started", plan_size=len(plan))
    try:
        for point in sorted(plan, key=lambda item: int(item["execution_order"])):
            if args.resume and point.get("status") == "complete":
                run_dir = experiment_dir / "runs" / point["run_key"]
                current_run_config = None
                try:
                    with open(run_dir / "run_config.json", "r", encoding="utf-8") as stream:
                        current_run_config = json.load(stream)
                except (OSError, json.JSONDecodeError):
                    pass
                final = _valid_final_model(
                    run_dir,
                    point["iterations"],
                    (
                        current_run_config.get("configuration_fingerprint")
                        if current_run_config
                        else None
                    ),
                )
                if final is not None:
                    heuristic_seed = stable_diagnostic_seed(point["seed"], "heuristic")
                    random_seed = stable_diagnostic_seed(point["seed"], "random")
                    if _diagnostic_is_valid(run_dir / "diagnostics" / "vs_heuristic", games=config["diagnostic_games"], seed=heuristic_seed, opponent="heuristic", model_hash=final["model_sha256"]) and _diagnostic_is_valid(run_dir / "diagnostics" / "vs_random", games=config["diagnostic_games"], seed=random_seed, opponent="random", model_hash=final["model_sha256"]):
                        journal.event("run_reused", run_key=point["run_key"], stage="complete")
                        continue
            try:
                _run_point(
                    point,
                    config=config,
                    experiment_dir=experiment_dir,
                    model_dir=paths["model_dir"],
                    manifest=manifest,
                    sl_weights_data=sl_weights_data,
                    sl_hash=sl_hash,
                    journal=journal,
                    resume=args.resume,
                )
            except Exception as exc:
                failure = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "stage": point.get("status", "unknown"),
                    "timestamp": utc_now(),
                    "traceback": traceback.format_exc(),
                }
                _set_status(experiment_dir, manifest, point, "failed", failure)
                journal.event("run_failed", run_key=point["run_key"], stage=failure["stage"], message=f"{failure['type']}: {failure['message']}")
                if not args.continue_on_error:
                    raise
            finally:
                _release_backend_cache()
        manifest["status"] = "complete" if all(point["status"] == "complete" for point in plan) else "incomplete"
        manifest["finished_at"] = utc_now()
        manifest["duration_s"] = previous_duration_s + time.perf_counter() - started
        _write_plan(experiment_dir, manifest)
        report = build_report(
            experiment_dir,
            paths["report_output_dir"],
            csv_only=config["csv_only"],
            generate_plots=not config["diag_no_plots"],
        )
        journal.event("report_written", output_dir=report["output_dir"])
        journal.event("experiment_completed", status=manifest["status"], duration_s=manifest["duration_s"])
        return report
    except BaseException:
        manifest["status"] = "failed"
        manifest["finished_at"] = utc_now()
        manifest["duration_s"] = previous_duration_s + time.perf_counter() - started
        _write_plan(experiment_dir, manifest)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    examples = """examples:
  %(prog)s --preset standard --dry-run
  %(prog)s --preset quick --critic-mode off --device cpu --run-id gpi_quick_off
  %(prog)s --preset standard --critic-mode both --run-id gpi_standard_both
  %(prog)s --preset standard --critic-mode off --run-id gpi_standard_off --resume
  %(prog)s --run-id gpi_standard_off --report-only
  %(prog)s --preset quick --csv-only --run-id gpi_csv_only
"""
    parser = argparse.ArgumentParser(
        description="Controlled RL sweep with a fixed total training-game budget per GPI.",
        epilog=examples,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", choices=tuple(PRESETS), default="standard")
    parser.add_argument("--total-training-games", type=int, default=None)
    parser.add_argument("--games-per-iteration-values", type=int, nargs="+", default=None, metavar="N")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, metavar="N")
    parser.add_argument("--critic-mode", choices=("off", "on", "both"), default=DEFAULT_CRITIC_MODE)
    parser.add_argument("--sl-weights-path", type=Path, default=DEFAULT_SL_WEIGHTS_PATH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--value-coef", type=float, default=DEFAULT_VALUE_COEF, help="Applied only when critic is on.")
    parser.add_argument("--entropy-coef", type=float, default=DEFAULT_ENTROPY_COEF)
    parser.add_argument("--training-opponent", choices=("self_play", "heuristic"), default=DEFAULT_TRAINING_OPPONENT)
    parser.add_argument("--reward-schema", choices=tuple(self_play.REWARD_SCHEMAS), default=DEFAULT_REWARD_SCHEMA)
    parser.add_argument("--clip-grad-norm", type=float, default=DEFAULT_CLIP_GRAD_NORM)
    normalization = parser.add_mutually_exclusive_group()
    normalization.add_argument("--normalize-advantages", action="store_true", dest="normalize_advantages")
    normalization.add_argument("--no-normalize-advantages", action="store_false", dest="normalize_advantages")
    parser.set_defaults(normalize_advantages=None)
    parser.add_argument(
        "--pool-refresh-games",
        type=int,
        default=DEFAULT_POOL_REFRESH_GAMES,
        help="Training games between opponent-pool snapshots.",
    )
    parser.add_argument("--max-pool-size", type=int, default=DEFAULT_MAX_POOL_SIZE)
    parser.add_argument("--checkpoint-count", type=int, default=None)
    parser.add_argument("--diagnostic-games", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default=DEFAULT_DEVICE)
    parser.add_argument("--rl-workers", type=self_play.parse_rl_worker_count, default=DEFAULT_RL_WORKERS, metavar="N|auto")
    parser.add_argument("--diagnostic-workers", type=int, default=DEFAULT_DIAGNOSTIC_WORKERS)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--report-output-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Rebuild reports only; never trains or diagnoses.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the exact plan without creating outputs or training.")
    parser.add_argument("--diag-no-plots", action="store_true")
    parser.add_argument("--csv-only", action="store_true", help="Do not import openpyxl or create XLSX output.")
    parser.add_argument("--quiet-training", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and args.report_only:
        parser.error("--resume and --report-only are mutually exclusive")
    if args.dry_run and (args.resume or args.report_only):
        parser.error("--dry-run cannot be combined with --resume or --report-only")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_experiment(args)
    if args.dry_run:
        print("Dry run complete: no directories, models, training, or diagnostics were created.")
    elif result is not None:
        print(f"Sweep report: {result.get('output_dir')}")


if __name__ == "__main__":
    main()
