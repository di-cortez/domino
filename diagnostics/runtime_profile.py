"""Durable cumulative runtime profiles for canonical RL pipeline runs."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import secrets
import socket

from utils.artifacts import atomic_write_json


FORMAT_VERSION = 1
REPORT_FILENAME = "runtime_profile.json"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _empty_totals():
    return {
        "rl": {
            "execution_count": 0,
            "games": 0,
            "iterations": 0,
            "decisions": 0,
            "optimizer_steps": 0,
            "execution_seconds": 0.0,
            "sections_seconds": {},
            "ppo_sections_seconds": {},
            "rollout_worker": {
                "games": 0,
                "profiled_games": 0,
                "profiled_game_cpu_seconds": 0.0,
                "worker_cpu_seconds": 0.0,
                "sections_seconds": {},
                "learner_policy": {},
                "opponent_policy": {},
            },
            "ppo_optimizer_step": {},
            "ppo_full_buffer_evaluation": {},
        },
        "rl_vs_random_diagnostics": {
            "execution_count": 0,
            "reused_execution_count": 0,
            "games": 0,
            "execution_seconds": 0.0,
            "sections_seconds": {},
            "pairwise_sections_seconds": {},
            "game_worker": {
                "games": 0,
                "profiled_games": 0,
                "profiled_game_cpu_seconds": 0.0,
                "worker_cpu_seconds": 0.0,
                "sections_seconds": {},
                "evaluated_agent_policy": {},
                "opponent_policy": {},
            },
        },
    }


def _add_numeric(target, source):
    """Recursively add numeric leaves while ignoring descriptive metadata."""
    for key, value in source.items():
        if isinstance(value, dict):
            child = target.setdefault(key, {})
            if isinstance(child, dict):
                _add_numeric(child, value)
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value


def _section_shares(sections, denominator):
    denominator = float(denominator)
    return {
        key: (100.0 * float(value) / denominator if denominator > 0.0 else 0.0)
        for key, value in sorted(sections.items())
    }


def _derived(totals):
    rl = totals["rl"]
    diagnostic = totals["rl_vs_random_diagnostics"]
    rl_seconds = float(rl["execution_seconds"])
    diagnostic_seconds = float(diagnostic["execution_seconds"])
    ppo_seconds = float(rl["sections_seconds"].get("ppo_update", 0.0))
    pairwise_seconds = float(
        diagnostic["sections_seconds"].get("pairwise_evaluation", 0.0)
    )
    rollout_worker = rl.get("rollout_worker", {})
    rollout_learner = rollout_worker.get("learner_policy", {})
    rollout_opponent = rollout_worker.get("opponent_policy", {})
    optimizer_detail = rl.get("ppo_optimizer_step", {})
    full_buffer_detail = rl.get("ppo_full_buffer_evaluation", {})
    game_worker = diagnostic.get("game_worker", {})
    diagnostic_agent = game_worker.get("evaluated_agent_policy", {})
    diagnostic_opponent = game_worker.get("opponent_policy", {})
    return {
        "rl": {
            "games_per_second": (
                float(rl["games"]) / rl_seconds if rl_seconds > 0.0 else 0.0
            ),
            "seconds_per_iteration": (
                rl_seconds / int(rl["iterations"])
                if int(rl["iterations"]) > 0 else 0.0
            ),
            "section_share_percent_of_execution": _section_shares(
                rl["sections_seconds"], rl_seconds
            ),
            "ppo_section_share_percent_of_ppo_update": _section_shares(
                rl["ppo_sections_seconds"], ppo_seconds
            ),
            "rollout_deep_profile_coverage_percent": (
                100.0 * float(rollout_worker.get("profiled_games", 0))
                / float(rollout_worker.get("games", 0))
                if float(rollout_worker.get("games", 0)) > 0.0 else 0.0
            ),
            "rollout_worker_cpu_parallelism": (
                float(rollout_worker.get("worker_cpu_seconds", 0.0))
                / float(rl["sections_seconds"].get("rollout_game_execution", 0.0))
                if float(rl["sections_seconds"].get("rollout_game_execution", 0.0)) > 0.0
                else 0.0
            ),
            "rollout_worker_section_share_percent_of_profiled_game_cpu": _section_shares(
                rollout_worker.get("sections_seconds", {}),
                rollout_worker.get("profiled_game_cpu_seconds", 0.0),
            ),
            "rollout_learner_policy_share_percent": _section_shares(
                rollout_learner.get("sections_seconds", {}),
                rollout_learner.get("total_seconds", 0.0),
            ),
            "rollout_opponent_policy_share_percent": _section_shares(
                rollout_opponent.get("sections_seconds", {}),
                rollout_opponent.get("total_seconds", 0.0),
            ),
            "ppo_optimizer_step_section_share_percent": _section_shares(
                optimizer_detail.get("sections_seconds", {}),
                optimizer_detail.get("execution_seconds", 0.0),
            ),
            "ppo_full_buffer_section_share_percent": _section_shares(
                full_buffer_detail.get("sections_seconds", {}),
                full_buffer_detail.get("execution_seconds", 0.0),
            ),
        },
        "rl_vs_random_diagnostics": {
            "games_per_second": (
                float(diagnostic["games"]) / diagnostic_seconds
                if diagnostic_seconds > 0.0 else 0.0
            ),
            "section_share_percent_of_execution": _section_shares(
                diagnostic["sections_seconds"], diagnostic_seconds
            ),
            "pairwise_section_share_percent_of_pairwise": _section_shares(
                diagnostic["pairwise_sections_seconds"], pairwise_seconds
            ),
            "game_worker_deep_profile_coverage_percent": (
                100.0 * float(game_worker.get("profiled_games", 0))
                / float(game_worker.get("games", 0))
                if float(game_worker.get("games", 0)) > 0.0 else 0.0
            ),
            "game_worker_cpu_parallelism": (
                float(game_worker.get("worker_cpu_seconds", 0.0))
                / float(diagnostic["pairwise_sections_seconds"].get(
                    "new_game_execution", 0.0
                ))
                if float(diagnostic["pairwise_sections_seconds"].get(
                    "new_game_execution", 0.0
                )) > 0.0 else 0.0
            ),
            "game_worker_section_share_percent_of_profiled_game_cpu": _section_shares(
                game_worker.get("sections_seconds", {}),
                game_worker.get("profiled_game_cpu_seconds", 0.0),
            ),
            "evaluated_agent_policy_share_percent": _section_shares(
                diagnostic_agent.get("sections_seconds", {}),
                diagnostic_agent.get("total_seconds", 0.0),
            ),
            "opponent_policy_share_percent": _section_shares(
                diagnostic_opponent.get("sections_seconds", {}),
                diagnostic_opponent.get("total_seconds", 0.0),
            ),
        },
    }


class RuntimeProfileRecorder:
    """Append one process session and atomically maintain cumulative totals."""

    def __init__(self, run_dir, *, pipeline_level, seed, start_rl_games):
        self.run_dir = Path(run_dir).resolve()
        self.path = self.run_dir / "diagnostics" / REPORT_FILENAME
        now = _now()
        if self.path.is_file():
            import json

            report = json.loads(self.path.read_text(encoding="utf-8"))
            if int(report.get("format_version", -1)) != FORMAT_VERSION:
                raise ValueError(
                    f"Unsupported runtime-profile format in {self.path}."
                )
            identity = report.get("run", {})
            if (
                identity.get("pipeline_level") != str(pipeline_level)
                or int(identity.get("seed", -1)) != int(seed)
            ):
                raise ValueError(
                    f"Runtime-profile identity does not match this run: {self.path}."
                )
            for session in report.get("sessions", []):
                # Profiles written before deterministic deep sampling timed
                # every game. Mark those historic values explicitly so they
                # remain compatible with sampled sessions in cumulative sums.
                rollout_worker = session.get("rl", {}).get("rollout_worker", {})
                if rollout_worker.get("sections_seconds") and "profiled_games" not in rollout_worker:
                    rollout_worker["profiled_games"] = int(
                        rollout_worker.get("games", 0)
                    )
                    rollout_worker["profiled_game_cpu_seconds"] = float(
                        rollout_worker.get("worker_cpu_seconds", 0.0)
                    )
                game_worker = session.get(
                    "rl_vs_random_diagnostics", {}
                ).get("game_worker", {})
                if game_worker.get("sections_seconds") and "profiled_games" not in game_worker:
                    game_worker["profiled_games"] = int(
                        game_worker.get("games", 0)
                    )
                    game_worker["profiled_game_cpu_seconds"] = float(
                        game_worker.get("worker_cpu_seconds", 0.0)
                    )
                if session.get("status") == "running":
                    session["status"] = "abandoned_before_next_resume"
                    session["finished_at"] = now
        else:
            report = {
                "format_version": FORMAT_VERSION,
                "kind": "canonical_rl_runtime_profile",
                "created_at": now,
                "updated_at": now,
                "run": {
                    "pipeline_level": str(pipeline_level),
                    "seed": int(seed),
                    "run_dir": str(self.run_dir),
                },
                "coverage": {
                    "fine_grained_profile_started_at_rl_games": int(start_rl_games),
                    "unprofiled_rl_games_before_first_profile": int(start_rl_games),
                    "profiled_rl_games": 0,
                    "note": (
                        "Fine-grained totals include only executions made after "
                        "this profiler was introduced; earlier RL games are not "
                        "retrospectively estimated."
                    ),
                },
                "cumulative": _empty_totals(),
                "derived": _derived(_empty_totals()),
                "sessions": [],
            }
        self.report = report
        self.session = {
            "session_id": (
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}"
                f"-{os.getpid()}-{secrets.token_hex(3)}"
            ),
            "status": "running",
            "started_at": now,
            "finished_at": None,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "start_rl_games": int(start_rl_games),
            "end_rl_games": int(start_rl_games),
            **_empty_totals(),
        }
        self.report.setdefault("sessions", []).append(self.session)
        self._save()

    def _recompute(self):
        totals = _empty_totals()
        for session in self.report.get("sessions", []):
            for category in totals:
                value = session.get(category)
                if isinstance(value, dict):
                    _add_numeric(totals[category], value)
        self.report["cumulative"] = totals
        self.report["coverage"]["profiled_rl_games"] = int(
            totals["rl"]["games"]
        )
        self.report["derived"] = _derived(totals)
        self.report["updated_at"] = _now()

    def _save(self):
        self._recompute()
        atomic_write_json(self.path, self.report)

    def record_rl(self, profile, *, end_rl_games):
        _add_numeric(self.session["rl"], profile)
        self.session["end_rl_games"] = int(end_rl_games)
        self._save()

    def record_diagnostic(self, profile, *, end_rl_games):
        _add_numeric(self.session["rl_vs_random_diagnostics"], profile)
        self.session["end_rl_games"] = int(end_rl_games)
        self._save()

    def finish(self, *, status, end_rl_games, error=None):
        if self.session.get("status") != "running":
            return
        self.session["status"] = str(status)
        self.session["finished_at"] = _now()
        self.session["end_rl_games"] = int(end_rl_games)
        if error:
            self.session["error"] = str(error)
        self._save()
