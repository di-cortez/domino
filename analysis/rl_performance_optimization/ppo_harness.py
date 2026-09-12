"""Fixed-workload PPO harness for the RL performance optimization stages.

The script never adds its own repository to ``sys.path``. Run it with
``PYTHONPATH`` pointing at the checkout under test, so one file measures both a
reference worktree and the candidate tree:

    PYTHONPATH=/path/to/checkout python ppo_harness.py run ...

Subcommands:

    capture  play real learner games once and freeze their decisions
    run      apply fixed PPO updates to frozen weights and record everything
    compare  diff two ``run`` outputs: exact equality first, then errors
    timing   alternate checkouts in fresh processes and summarize timings

Every ``run`` restores the captured policy before each measured iteration, so
each update sees on-policy ``old_log_probs`` exactly as training does, while a
different iteration index still changes the minibatch shuffles. Outputs are
written only where ``--output`` points; nothing here reads or writes a run
directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


# Each variant exercises one code path the stages touch. ``production`` mirrors
# the measured forever runs (168-256-128-56, entropy 0, no critic, 16 epochs).
VARIANTS = {
    "production": {
        "learning_rate": 0.0008,
        "entropy_coef": 0.0,
        "baseline": "batch-mean",
        "max_epochs": 16,
        "dropout_rate": 0.0,
    },
    # A rate that spikes KL on the first epochs: KL stops and norm clipping.
    "hot_lr": {
        "learning_rate": 0.03,
        "entropy_coef": 0.0,
        "baseline": "batch-mean",
        "max_epochs": 16,
        "dropout_rate": 0.0,
    },
    "entropy_shared_critic": {
        "learning_rate": 0.003,
        "entropy_coef": 0.01,
        "baseline": "value-head",
        "max_epochs": 8,
        "dropout_rate": 0.0,
    },
    "own_critic_dropout": {
        "learning_rate": 0.003,
        "entropy_coef": 0.0,
        "baseline": "value-head-own-nn",
        "max_epochs": 8,
        "dropout_rate": 0.1,
    },
    "no_up_entropy_dropout": {
        "learning_rate": 0.003,
        "entropy_coef": 0.02,
        "baseline": "value-head-no-up",
        "max_epochs": 4,
        "dropout_rate": 0.1,
    },
}

# Machine- and time-dependent fields, compared separately from the payload.
VOLATILE_METRIC_KEYS = ("runtime_timing_seconds", "runtime_profile_detail")
VOLATILE_PREFLIGHT_KEYS = ("reported_free_vram_bytes", "usable_free_vram_bytes")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unserializable {type(value).__name__}")


def _host(array):
    return np.asarray(array.get() if hasattr(array, "get") else array)


def _environment(device):
    info = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "device": device,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "CUPY_ACCELERATORS",
            )
        },
    }
    checkout = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
    if checkout:
        try:
            info["checkout_commit"] = subprocess.run(
                ["git", "-C", checkout, "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            info["checkout_dirty"] = bool(subprocess.run(
                ["git", "-C", checkout, "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip())
        except (OSError, subprocess.CalledProcessError):
            pass
    if device == "gpu":
        import cupy  # pylint: disable=import-outside-toplevel

        info["cupy"] = cupy.__version__
        info["cuda_runtime"] = int(cupy.cuda.runtime.runtimeGetVersion())
        info["cuda_driver"] = int(cupy.cuda.runtime.driverGetVersion())
        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props.get("name", b"")
        info["gpu"] = name.decode() if isinstance(name, bytes) else str(name)
    return info


def cmd_capture(args):
    """Play real learner-vs-opponent games and freeze their decisions."""
    # pylint: disable=import-outside-toplevel
    from agents.rl_nn import PolicyNetwork
    from training.rl import rollout

    network = PolicyNetwork.load_from_sl(
        args.weights,
        learning_rate=0.001,
        device="cpu",
    )
    collect = {
        "random": rollout._collect_steps_vs_random,  # pylint: disable=protected-access
        "heuristic": rollout._collect_steps_vs_heuristic,  # pylint: disable=protected-access
    }[args.opponent]
    schema = dict(rollout.DEFAULT_REWARD_SCHEMA)
    random.seed(args.seed)
    np.random.seed(args.seed)
    samples = []
    started = time.perf_counter()
    for _ in range(args.games):
        game_samples = collect(
            network,
            schema,
            schema["gamma_f"],
            ruleset_name=args.ruleset,
        )[0]
        samples.extend(game_samples)
    arrays = {
        "states": np.hstack([_host(s.x).astype(np.float32) for s in samples]),
        "legal_masks": np.hstack([_host(s.legal_mask) > 0 for s in samples]),
        "actions": np.asarray([s.action_index for s in samples], dtype=np.int64),
        "old_log_probs": np.asarray([s.old_log_prob for s in samples], dtype=np.float32),
        "policy_rewards": np.asarray([s.policy_reward for s in samples], dtype=np.float32),
        "raw_rewards": np.asarray([s.raw_reward for s in samples], dtype=np.float32),
        "local_rewards": np.asarray([s.local_reward for s in samples], dtype=np.float32),
        "terminal_rewards": np.asarray([s.terminal_reward for s in samples], dtype=np.float32),
    }
    meta = {
        "weights": str(Path(args.weights).resolve()),
        "weights_sha256": hashlib.sha256(Path(args.weights).read_bytes()).hexdigest(),
        "ruleset": args.ruleset,
        "opponent": args.opponent,
        "games": args.games,
        "seed": args.seed,
        "decisions": len(samples),
        "capture_seconds": time.perf_counter() - started,
        "environment": _environment("cpu"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial.npz")
    np.savez(temporary, meta=np.asarray(json.dumps(meta)), **arrays)
    os.replace(temporary, output)
    print(json.dumps(meta, indent=2))


def _load_samples(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    meta = json.loads(str(arrays.pop("meta")))
    count = arrays["actions"].size
    samples = [
        SimpleNamespace(
            x=arrays["states"][:, index:index + 1],
            legal_mask=arrays["legal_masks"][:, index:index + 1],
            action_index=int(arrays["actions"][index]),
            old_log_prob=float(arrays["old_log_probs"][index]),
            policy_reward=float(arrays["policy_rewards"][index]),
            raw_reward=float(arrays["raw_rewards"][index]),
            local_reward=float(arrays["local_rewards"][index]),
            terminal_reward=float(arrays["terminal_rewards"][index]),
        )
        for index in range(count)
    ]
    return samples, meta


def _build_network(weights, variant, device):
    # pylint: disable=import-outside-toplevel
    from agents.rl_nn import PolicyNetwork
    from training.rl import baseline as baselines

    spec = baselines.BaselineSpec(kind=variant["baseline"])
    return PolicyNetwork.load_from_sl(
        weights,
        learning_rate=variant["learning_rate"],
        use_value_head=spec.kind in baselines.CRITIC_KINDS,
        critic_updates_trunk=spec.critic_updates_trunk,
        critic_owns_network=spec.critic_owns_network,
        device=device,
        dropout_rate=variant["dropout_rate"],
    ), spec


def _seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    if device == "gpu":
        import cupy  # pylint: disable=import-outside-toplevel

        cupy.random.seed(seed)


def _parameter_names(network):
    return (*network.weight_names, *network.critic_parameter_names)


def _parameters(network):
    return {
        name: _host(network.parameter_array(name)).copy()
        for name in _parameter_names(network)
    }


def _restore(network, initial):
    network.restore_parameters({
        name: network.xp.asarray(value, dtype=network.xp.float32)
        for name, value in initial.items()
    })
    network.optimizer_step_count = 0


def _deterministic_metrics(metrics):
    payload = {
        key: value
        for key, value in metrics.items()
        if key not in VOLATILE_METRIC_KEYS
    }
    preflight = dict(payload.get("buffer_preflight") or {})
    for key in VOLATILE_PREFLIGHT_KEYS:
        preflight.pop(key, None)
    payload["buffer_preflight"] = preflight
    return json.loads(json.dumps(payload, default=_json_default))


class _PeakPoolUsage:
    """Track the CuPy pool's in-use high-water mark through allocations."""

    def __init__(self):
        import cupy  # pylint: disable=import-outside-toplevel

        self.pool = cupy.get_default_memory_pool()
        self.peak = 0
        hook_base = cupy.cuda.memory_hook.MemoryHook
        tracker = self

        class _Hook(hook_base):  # pylint: disable=too-few-public-methods
            name = "PeakPoolUsage"

            def malloc_postprocess(self, **_kwargs):
                tracker.peak = max(tracker.peak, int(tracker.pool.used_bytes()))

        self.hook = _Hook()

    def __enter__(self):
        self.hook.__enter__()
        return self

    def __exit__(self, *exc):
        self.hook.__exit__(*exc)


def cmd_run(args):
    """Apply fixed PPO updates and record weights, metrics, and timings."""
    # pylint: disable=import-outside-toplevel
    from training.rl.ppo import update_from_samples

    variant = dict(VARIANTS[args.variant])
    if args.learning_rate is not None:
        variant["learning_rate"] = float(args.learning_rate)
    if args.max_epochs is not None:
        variant["max_epochs"] = int(args.max_epochs)
    samples, capture_meta = _load_samples(args.buffer)
    _seed_everything(args.seed, args.device)
    network, spec = _build_network(args.weights, variant, args.device)
    initial = _parameters(network)

    def one_update(iteration):
        return update_from_samples(
            network,
            samples,
            base_seed=args.seed,
            iteration=iteration,
            entropy_coef=variant["entropy_coef"],
            value_coef=0.5,
            normalize_advantages=True,
            max_epochs=variant["max_epochs"],
            baseline=spec,
            collect_value_predictions=False,
        )

    for warmup in range(args.warmup_updates):
        # Kernel compilation and pool growth stay outside the measured updates.
        _seed_everything(args.seed + 1_000_000 + warmup, args.device)
        one_update(1_000_000 + warmup)
        _restore(network, initial)

    records = []
    final_arrays = {}
    peak_tracker = None
    if args.measure_gpu_memory and args.device == "gpu":
        peak_tracker = _PeakPoolUsage()
        peak_tracker.__enter__()
    try:
        for iteration in range(1, args.iterations + 1):
            _restore(network, initial)
            _seed_everything(args.seed + iteration, args.device)
            wall_started = time.perf_counter()
            result = one_update(iteration)
            network.synchronize()
            wall_seconds = time.perf_counter() - wall_started
            parameters = _parameters(network)
            for name, value in parameters.items():
                final_arrays[f"iter{iteration:03d}/{name}"] = value
            records.append({
                "iteration": iteration,
                "optimizer_step_count": int(network.optimizer_step_count),
                "parameter_sha256": {
                    name: hashlib.sha256(
                        np.ascontiguousarray(value).tobytes()
                    ).hexdigest()
                    for name, value in parameters.items()
                },
                "parameter_dtypes": {
                    name: str(value.dtype) for name, value in parameters.items()
                },
                "parameter_shapes": {
                    name: list(value.shape) for name, value in parameters.items()
                },
                "metrics": _deterministic_metrics(result.metrics),
                "timing": {
                    "wall_seconds": wall_seconds,
                    "buffer_seconds": result.buffer_seconds,
                    "update_seconds": result.update_seconds,
                    "runtime_timing_seconds": result.metrics.get(
                        "runtime_timing_seconds"
                    ),
                    "runtime_profile_detail": result.metrics.get(
                        "runtime_profile_detail"
                    ),
                },
            })
    finally:
        if peak_tracker is not None:
            peak_tracker.__exit__(None, None, None)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "ppo_harness_run",
        "variant": args.variant,
        "variant_settings": variant,
        "device": args.device,
        "seed": args.seed,
        "iterations": args.iterations,
        "warmup_updates": args.warmup_updates,
        "buffer": str(Path(args.buffer).resolve()),
        "capture": capture_meta,
        "environment": _environment(args.device),
        "peak_gpu_pool_used_bytes": (
            None if peak_tracker is None else int(peak_tracker.peak)
        ),
        "peak_host_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "records": records,
    }
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=1, default=_json_default),
        encoding="utf-8",
    )
    os.replace(temporary, output)
    if args.save_parameters:
        arrays_path = output.with_suffix(".parameters.npz")
        partial = arrays_path.with_name(arrays_path.name + ".partial.npz")
        np.savez(partial, **final_arrays)
        os.replace(partial, arrays_path)
    summary = [
        (
            record["iteration"],
            record["metrics"]["epochs_completed"],
            record["metrics"]["stopped_by_kl"],
            round(record["timing"]["update_seconds"], 4),
        )
        for record in records
    ]
    print(json.dumps({"output": str(output), "iterations": summary}))


def _numeric_leaves(value, prefix=""):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from _numeric_leaves(value[key], f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _numeric_leaves(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def cmd_compare(args):
    """Compare deterministic payloads of two runs; timings are ignored."""
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    report = {
        "reference": args.reference,
        "candidate": args.candidate,
        "variant": reference["variant"],
        "iterations": [],
    }
    if reference["variant"] != candidate["variant"] or len(
        reference["records"]
    ) != len(candidate["records"]):
        raise SystemExit("The two runs used different variants or iteration counts.")
    ref_arrays = cand_arrays = None
    ref_arrays_path = Path(args.reference).with_suffix(".parameters.npz")
    cand_arrays_path = Path(args.candidate).with_suffix(".parameters.npz")
    if ref_arrays_path.is_file() and cand_arrays_path.is_file():
        ref_arrays = np.load(ref_arrays_path)
        cand_arrays = np.load(cand_arrays_path)
    all_exact = True
    for ref, cand in zip(reference["records"], candidate["records"]):
        row = {
            "iteration": ref["iteration"],
            "parameters_bytes_equal": ref["parameter_sha256"] == cand["parameter_sha256"],
            "shapes_equal": ref["parameter_shapes"] == cand["parameter_shapes"],
            "dtypes_equal": ref["parameter_dtypes"] == cand["parameter_dtypes"],
            "optimizer_steps_equal": (
                ref["optimizer_step_count"] == cand["optimizer_step_count"]
            ),
            "metrics_equal": ref["metrics"] == cand["metrics"],
            "epochs": [
                ref["metrics"]["epochs_completed"],
                cand["metrics"]["epochs_completed"],
            ],
            "stopped_by_kl": [
                ref["metrics"]["stopped_by_kl"],
                cand["metrics"]["stopped_by_kl"],
            ],
        }
        ref_leaves = dict(_numeric_leaves(ref["metrics"]))
        cand_leaves = dict(_numeric_leaves(cand["metrics"]))
        differences = {}
        for key in sorted(set(ref_leaves) | set(cand_leaves)):
            left = ref_leaves.get(key, "<missing>")
            right = cand_leaves.get(key, "<missing>")
            if left == right:
                continue
            if (
                isinstance(left, (int, float))
                and isinstance(right, (int, float))
                and not isinstance(left, bool)
                and not isinstance(right, bool)
            ):
                absolute = abs(float(left) - float(right))
                scale = max(abs(float(left)), abs(float(right)), 1e-300)
                differences[key] = {
                    "reference": left,
                    "candidate": right,
                    "abs": absolute,
                    "rel": absolute / scale,
                }
            else:
                differences[key] = {"reference": left, "candidate": right}
        row["metric_differences"] = differences
        if ref_arrays is not None and not row["parameters_bytes_equal"]:
            parameter_errors = {}
            for name in ref["parameter_sha256"]:
                key = f"iter{ref['iteration']:03d}/{name}"
                left = ref_arrays[key].astype(np.float64)
                right = cand_arrays[key].astype(np.float64)
                absolute = float(np.max(np.abs(left - right)))
                scale = float(max(np.max(np.abs(left)), 1e-300))
                parameter_errors[name] = {
                    "max_abs": absolute,
                    "max_abs_relative_to_max_weight": absolute / scale,
                }
            row["parameter_errors"] = parameter_errors
        exact = (
            row["parameters_bytes_equal"]
            and row["shapes_equal"]
            and row["dtypes_equal"]
            and row["optimizer_steps_equal"]
            and row["metrics_equal"]
        )
        row["exact"] = exact
        all_exact = all_exact and exact
        report["iterations"].append(row)
    report["all_exact"] = all_exact
    text = json.dumps(report, indent=1, default=_json_default)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    if args.oneline:
        cells = []
        for row in report["iterations"]:
            relative = max(
                (
                    item["rel"]
                    for item in row["metric_differences"].values()
                    if "rel" in item
                ),
                default=0.0,
            )
            parameter = max(
                (
                    item["max_abs"]
                    for item in row.get("parameter_errors", {}).values()
                ),
                default=0.0,
            )
            state = (
                "exact" if row["exact"]
                else f"metric_rel<={relative:.1e},w_abs<={parameter:.1e}"
            )
            cells.append(
                f"it{row['iteration']}:{state}"
                f"/epochs{row['epochs'][0]}-{row['epochs'][1]}"
            )
        print(f"{args.oneline:40s} all_exact={all_exact} " + " ".join(cells))
        return
    print(json.dumps({
        "variant": report["variant"],
        "all_exact": all_exact,
        "iterations": [
            {
                "iteration": row["iteration"],
                "exact": row["exact"],
                "epochs": row["epochs"],
                "stopped_by_kl": row["stopped_by_kl"],
                "different_metrics": len(row["metric_differences"]),
                "max_metric_rel": max(
                    (
                        item["rel"]
                        for item in row["metric_differences"].values()
                        if "rel" in item
                    ),
                    default=0.0,
                ),
            }
            for row in report["iterations"]
        ],
    }, indent=1))


def _summarize(values):
    values = sorted(values)
    if not values:
        return None
    quartiles = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
    return {
        "count": len(values),
        "median": statistics.median(values),
        "q1": quartiles[0],
        "q3": quartiles[2],
        "min": values[0],
        "max": values[-1],
    }


def _timing_rows(run_payload):
    rows = []
    for record in run_payload["records"]:
        timing = record["timing"]
        ppo = timing["runtime_timing_seconds"] or {}
        detail = timing["runtime_profile_detail"] or {}
        full = detail.get("full_buffer_evaluation", {})
        optimizer = detail.get("optimizer_step", {})
        rows.append({
            "update_seconds": timing["update_seconds"],
            "optimizer_steps_seconds": ppo.get("optimizer_steps"),
            "minibatch_materialization_seconds": ppo.get("minibatch_materialization"),
            "full_buffer_evaluation_seconds": ppo.get("full_buffer_evaluation"),
            "optimizer_calls": optimizer.get("calls"),
            "full_buffer_calls": full.get("calls"),
            "epochs_completed": record["metrics"]["epochs_completed"],
        })
    return rows


def cmd_timing(args):
    """Alternate checkouts in fresh processes and summarize their timings."""
    checkouts = [Path(path).resolve() for path in args.checkout]
    labels = args.label or [path.name for path in checkouts]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    collected = {label: [] for label in labels}
    epochs = {label: [] for label in labels}
    runs = {label: [] for label in labels}
    for repetition in range(args.repetitions):
        order = list(zip(labels, checkouts))
        if repetition % 2:
            order.reverse()
        for label, checkout in order:
            destination = output_dir / f"{label}_rep{repetition:02d}.json"
            environment = dict(os.environ, PYTHONPATH=str(checkout))
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "run",
                "--buffer", args.buffer,
                "--weights", args.weights,
                "--variant", args.variant,
                "--device", args.device,
                "--iterations", str(args.iterations),
                "--warmup-updates", str(args.warmup_updates),
                "--seed", str(args.seed),
                "--output", str(destination),
            ]
            if args.max_epochs is not None:
                command += ["--max-epochs", str(args.max_epochs)]
            if args.learning_rate is not None:
                command += ["--learning-rate", str(args.learning_rate)]
            if args.measure_gpu_memory:
                command.append("--measure-gpu-memory")
            subprocess.run(command, check=True, env=environment, stdout=subprocess.DEVNULL)
            payload = json.loads(destination.read_text(encoding="utf-8"))
            rows = _timing_rows(payload)
            collected[label].extend(rows)
            epochs[label].extend(row["epochs_completed"] for row in rows)
            runs[label].append({
                "file": str(destination),
                "peak_gpu_pool_used_bytes": payload["peak_gpu_pool_used_bytes"],
                "peak_host_rss_kib": payload["peak_host_rss_kib"],
            })
            print(
                f"rep {repetition} {label}: median update "
                f"{statistics.median(r['update_seconds'] for r in rows):.4f}s",
                flush=True,
            )
    summary = {
        "kind": "ppo_harness_timing",
        "variant": args.variant,
        "device": args.device,
        "repetitions": args.repetitions,
        "iterations_per_repetition": args.iterations,
        "checkouts": dict(zip(labels, map(str, checkouts))),
        "runs": runs,
        "epochs_completed": epochs,
        "per_update": {
            label: {
                key: _summarize([
                    row[key] for row in rows if row[key] is not None
                ])
                for key in (
                    "update_seconds",
                    "optimizer_steps_seconds",
                    "minibatch_materialization_seconds",
                    "full_buffer_evaluation_seconds",
                )
            }
            for label, rows in collected.items()
        },
        "optimizer_seconds_per_step": {
            label: _summarize([
                row["optimizer_steps_seconds"] / row["optimizer_calls"]
                for row in rows
                if row["optimizer_calls"]
            ])
            for label, rows in collected.items()
        },
        "full_buffer_seconds_per_call": {
            label: _summarize([
                row["full_buffer_evaluation_seconds"] / row["full_buffer_calls"]
                for row in rows
                if row["full_buffer_calls"]
            ])
            for label, rows in collected.items()
        },
    }
    if len(labels) >= 2:
        base = summary["per_update"][labels[0]]["update_seconds"]["median"]
        summary["median_update_ratio_vs_first"] = {
            label: summary["per_update"][label]["update_seconds"]["median"] / base
            for label in labels
        }
    path = output_dir / "timing_summary.json"
    path.write_text(json.dumps(summary, indent=1, default=_json_default), encoding="utf-8")
    print(json.dumps({
        "summary": str(path),
        "median_update_seconds": {
            label: summary["per_update"][label]["update_seconds"]["median"]
            for label in labels
        },
        "ratio": summary.get("median_update_ratio_vs_first"),
    }, indent=1))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    capture = commands.add_parser("capture", help=cmd_capture.__doc__)
    capture.add_argument("--weights", required=True)
    capture.add_argument("--ruleset", default="double-six")
    capture.add_argument("--opponent", choices=("random", "heuristic"), default="random")
    capture.add_argument("--games", type=int, default=2000)
    capture.add_argument("--seed", type=int, default=52)
    capture.add_argument("--output", required=True)

    def add_run_arguments(sub):
        sub.add_argument("--buffer", required=True)
        sub.add_argument("--weights", required=True)
        sub.add_argument("--variant", choices=tuple(VARIANTS), default="production")
        sub.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
        sub.add_argument("--iterations", type=int, default=5)
        sub.add_argument("--warmup-updates", type=int, default=1)
        sub.add_argument("--seed", type=int, default=52)
        sub.add_argument("--max-epochs", type=int, default=None)
        sub.add_argument("--learning-rate", type=float, default=None)
        sub.add_argument("--measure-gpu-memory", action="store_true")

    run = commands.add_parser("run", help=cmd_run.__doc__)
    add_run_arguments(run)
    run.add_argument("--output", required=True)
    run.add_argument("--save-parameters", action="store_true")

    compare = commands.add_parser("compare", help=cmd_compare.__doc__)
    compare.add_argument("reference")
    compare.add_argument("candidate")
    compare.add_argument("--output", default=None)
    compare.add_argument("--oneline", default=None, metavar="LABEL")

    timing = commands.add_parser("timing", help=cmd_timing.__doc__)
    add_run_arguments(timing)
    timing.add_argument("--checkout", action="append", required=True)
    timing.add_argument("--label", action="append", default=None)
    timing.add_argument("--repetitions", type=int, default=3)
    timing.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    {
        "capture": cmd_capture,
        "run": cmd_run,
        "compare": cmd_compare,
        "timing": cmd_timing,
    }[args.command](args)


if __name__ == "__main__":
    main()
