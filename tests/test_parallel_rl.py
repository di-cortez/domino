"""Correctness and safety tests for parallel reinforcement-learning rollouts."""

import os
import json
import sys
import tempfile
import unittest
from dataclasses import dataclass, fields
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.rl_nn import PolicyNetwork
from diagnostics.parallel_runner import (
    MAX_PARALLEL_WORKERS,
    DiagnosticMemoryPressure,
    ParallelSafetyConfig,
)
from training.rl.parallel import RLRolloutRunner, worker_count
from training.rl.matchmaking import UniformRotationState, build_match_plan
from training.rl.config import (
    RLExecutionOptions,
    RLResourceOptions,
    RLTrainingOptions,
)
from training.rl.resume import (
    NUMBERED_CHECKPOINT_WEIGHT_RETENTION,
    _load_initial_network,
    load_resume_state,
    numbered_checkpoint_path,
    resume_state_path,
)
from training.rl.pool import (
    CHAMPION_VS_HEURISTIC_BUCKET,
    K_RECENT,
    MEDIUM_TERM_INTERVAL_ITERATIONS,
)
# The pool now keys champion state by bucket. Every existing champion test
# targets the fixed-heuristic bucket, so it gets one short local name rather
# than the constant repeated inside dozens of assertions.
HEURISTIC_CHAMPION = CHAMPION_VS_HEURISTIC_BUCKET


def _champion_state(metadata):
    """Return the heuristic champion's durable block of exported pool state."""
    return metadata["opponent_pool_state"]["champion_state_by_bucket"][
        HEURISTIC_CHAMPION
    ]
from training.rl.reporting import read_training_metrics
from training.rl.rollout import REWARD_SCHEMAS
from training.rl.cli import parse_args as parse_rl_args
from training.rl.training_loop import train
from train_script.run_pipeline import parse_args as parse_pipeline_args
from utils.resource_limits import MemorySafetyError


@dataclass(frozen=True)
class _ArchivedIdentity:
    """Archive metadata a test needs for archive-backed band selection."""

    checkpoint_id: str
    opponent_id: str
    completed_iteration: int
    completed_rl_games: int


def _fill_delayed_band(pool, network):
    """Age one pool past the recent band so ``medium_term`` has real members."""
    archived = []
    baseline = pool.initial_snapshot_record
    archived.append((baseline.checkpoint_id, baseline.opponent_id, 0, 0))

    def _load_weights(_checkpoint_id):
        return {
            name: np.asarray(getattr(network, name)).copy()
            for name in network.weight_names
        }

    last = K_RECENT + MEDIUM_TERM_INTERVAL_ITERATIONS
    for iteration in range(1, last + 1):
        record = pool.consider_updated_policy(
            network,
            iteration=iteration,
            completed_games=iteration * 100,
            has_samples=True,
        )
        if record is not None and not iteration % MEDIUM_TERM_INTERVAL_ITERATIONS:
            archived.append((
                record.checkpoint_id,
                record.opponent_id,
                iteration,
                iteration * 100,
            ))
    pool.reconcile_archive_backed_buckets(
        [_ArchivedIdentity(*value) for value in archived],
        completed_iteration=last,
        load_weights=_load_weights,
    )


def _rollout_fingerprint(results):
    """Return a comparison-safe representation of arrays and scalar metadata."""
    rows = []
    for result in results:
        samples = []
        for sample in result["samples"]:
            samples.append((
                np.asarray(sample.x).tobytes(),
                sample.action_index,
                np.asarray(sample.legal_mask).tobytes(),
                sample.policy_reward,
                sample.raw_reward,
                sample.local_reward,
                sample.terminal_reward,
                sample.old_log_prob,
            ))
        rows.append((
            result["game_index"],
            result["game_seed"],
            tuple(samples),
            tuple(sorted(result["event_stats"].items())),
            result["winner"],
            result["learner_position"],
        ))
    return rows


class ParallelRLTests(unittest.TestCase):
    def setUp(self):
        self.safety = ParallelSafetyConfig(
            memory_reserve_mb=0,
            estimated_worker_mb=1,
            max_worker_rss_mb=1024,
        )
        self._temporary_assets = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_assets.cleanup)
        self.sl_weights_path = (
            Path(self._temporary_assets.name) / "supervised.npz"
        )
        network = PolicyNetwork(random_seed=19, device="cpu")
        np.savez(
            self.sl_weights_path,
            **{
                name: np.asarray(getattr(network, name))
                for name in network.weight_names
            },
        )

    def _train(self, **kwargs):
        kwargs.setdefault("sl_weights_path", str(self.sl_weights_path))
        groups = []
        for option_type in (
            RLTrainingOptions,
            RLResourceOptions,
            RLExecutionOptions,
        ):
            names = {field.name for field in fields(option_type)}
            groups.append(option_type(**{
                key: kwargs.pop(key)
                for key in tuple(kwargs)
                if key in names
            }))
        self.assertFalse(kwargs, f"Unknown RL test options: {sorted(kwargs)}")
        return train(*groups)

    def _network(self):
        return PolicyNetwork.load_from_sl(
            self.sl_weights_path,
            device="cpu",
        )

    def _collect(
        self,
        workers,
        game_count=10,
        seed=1234,
        opponent_buckets=("heuristic", "recent"),
        prepare=None,
    ):
        network = self._network()
        runner = RLRolloutRunner(
            network,
            opponent_buckets=opponent_buckets,
            schema=dict(REWARD_SCHEMAS),
            gamma=1.0,
            safety=self.safety,
        )
        try:
            if prepare is not None:
                prepare(runner.opponent_pool, network)
            runner.set_workers(workers)
            runner.sync_current(network)
            plan = build_match_plan(
                opponent_pool=runner.opponent_pool,
                performance_tracker=runner.performance_tracker,
                selected_buckets=opponent_buckets,
                uniform_rotation=UniformRotationState(opponent_buckets),
                difficulty_weight=0.5,
                iteration=1,
                first_absolute_game=0,
                game_count=game_count,
                base_seed=seed,
            )
            return runner.collect_games(plan, seed)
        finally:
            runner.close()

    def _champion_stage(self, workers, *, candidates=3, games=20):
        """Play one reduced champion stage through the real worker pool."""
        from training.rl import champion_evaluation as champion
        from training.rl.champion_evaluation import ChampionGameSpec

        buckets = ("heuristic", "recent", "champion_vs_heuristic")
        network = self._network()
        runner = RLRolloutRunner(
            network,
            opponent_buckets=buckets,
            schema=dict(REWARD_SCHEMAS),
            gamma=1.0,
            safety=self.safety,
        )
        try:
            pool = runner.opponent_pool
            for iteration in range(1, candidates + 1):
                pool.consider_updated_policy(
                    network,
                    iteration=iteration,
                    completed_games=iteration * 10,
                    has_samples=True,
                )
            runner.set_workers(workers)
            runner.sync_current(network)
            candidate_ids = pool.champion_pending_candidate_ids(HEURISTIC_CHAMPION)[:candidates]
            specs = []
            for candidate_id in candidate_ids:
                for game_index in range(games):
                    specs.append(ChampionGameSpec(
                        stage_index=0,
                        candidate_id=candidate_id,
                        bank_slot=pool.bank_slot(candidate_id),
                        game_index=game_index,
                        seed=champion.champion_stage_seed(
                            4242,
                            event_index=0,
                            stage_index=0,
                            game_index=game_index,
                        ),
                        candidate_position=champion.champion_seat_position(
                            game_index
                        ),
                        sequence=len(specs),
                    ))
            results, _run_info = runner.evaluate_champion_games(specs)
            # The rollout profile must survive a racing stage untouched.
            assert runner.last_runtime_profile == {}
            assert runner.last_champion_runtime_profile
            return tuple(specs), results
        finally:
            runner.close()

    def test_champion_evaluation_is_identical_with_one_and_many_workers(self):
        """Rankings must not depend on how the games were scheduled."""
        single_specs, single = self._champion_stage(1)
        multi_specs, multi = self._champion_stage(3)
        self.assertEqual(single_specs, multi_specs)
        self.assertEqual(single, multi)
        self.assertEqual(len(single), len(single_specs))

        # Every returned game is binary and keeps its assigned identity.
        for spec, result in zip(single_specs, single):
            self.assertEqual(result["sequence"], spec.sequence)
            self.assertEqual(result["candidate_id"], spec.candidate_id)
            self.assertEqual(result["game_index"], spec.game_index)
            self.assertEqual(
                result["candidate_position"],
                spec.candidate_position,
            )
            self.assertIn(result["winner"], (0, 1))

    def test_champion_evaluation_shares_one_panel_across_candidates(self):
        """Common random numbers: identical policies must score identically."""
        specs, results = self._champion_stage(2)
        by_candidate = {}
        for spec, result in zip(specs, results):
            by_candidate.setdefault(spec.candidate_id, []).append(
                (spec.game_index, result["winner"], spec.candidate_position)
            )
        outcomes = {tuple(value) for value in by_candidate.values()}
        # The pool snapshots every iteration from the same untrained network,
        # so every candidate holds identical weights and the shared panel must
        # produce byte-identical results. Per-candidate seeds could not.
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(len(by_candidate), 3)

    def test_the_policy_bank_uses_one_shared_segment_for_every_slot(self):
        """A per-slot segment layout exhausts the process descriptor limit.

        A POSIX shared segment costs two descriptors in every process that maps
        it. At the 800 slots a five-bucket selection reserves, one segment per
        slot needed about 1,600 descriptors and raised
        ``OSError: [Errno 24] Too many open files`` from ``shm_open`` on any
        machine with the common 1024 limit -- in the parent and again in every
        worker.
        """
        from training.rl.pool import SharedPolicyBank

        capacity = 800
        network = self._network()
        proc_fds = Path("/proc/self/fd")
        before = len(os.listdir(proc_fds)) if proc_fds.is_dir() else None
        bank = SharedPolicyBank(network, capacity)
        try:
            if before is not None:
                self.assertLess(len(os.listdir(proc_fds)) - before, 8)
            descriptors = (bank.current_descriptor,) + bank.opponent_descriptors
            self.assertEqual(len(descriptors), capacity + 1)
            self.assertEqual(len({value.name for value in descriptors}), 1)
            self.assertEqual(
                [value.offset for value in descriptors],
                [index * bank.policy_bytes for index in range(capacity + 1)],
            )
            self.assertEqual(
                bank.allocated_bytes,
                (capacity + 1) * bank.policy_bytes,
            )

            # Distinct offsets must address distinct storage.
            slots = [bank.allocate_slot() for _ in range(capacity)]
            first, last = slots[0], slots[-1]
            other = self._network()
            other.W1 = np.asarray(other.W1) + 1.0
            bank.write_policy(first, network)
            bank.write_policy(last, other)
            np.testing.assert_array_equal(
                bank.read_policy(first)["W1"], np.asarray(network.W1),
            )
            np.testing.assert_array_equal(
                bank.read_policy(last)["W1"], np.asarray(other.W1),
            )
        finally:
            bank.close()

    def test_worker_parser_enforces_hard_limit(self):
        self.assertEqual(worker_count("auto"), "auto")
        self.assertEqual(worker_count("20"), MAX_PARALLEL_WORKERS)
        with self.assertRaises(ValueError):
            worker_count("21")

        args = parse_pipeline_args([
            "small",
            "--rl-workers",
            "3",
            "--rl-memory-reserve-mb",
            "256",
        ])
        self.assertEqual(args.rl_workers, 3)
        self.assertEqual(args.rl_memory_reserve_mb, 256)
        self.assertTrue(parse_rl_args(["--compact"]).compact)

    def test_rollouts_are_identical_with_one_and_multiple_workers(self):
        one_worker, one_info = self._collect(1)
        two_workers, two_info = self._collect(2)
        self.assertEqual(
            _rollout_fingerprint(one_worker),
            _rollout_fingerprint(two_workers),
        )
        self.assertTrue(one_info.workers_cpu_only)
        self.assertTrue(two_info.workers_cpu_only)
        saved_steps = [sample for result in one_worker for sample in result["samples"]]
        self.assertTrue(saved_steps)
        self.assertTrue(all(np.isfinite(sample.old_log_prob) for sample in saved_steps))
        self.assertTrue(all(sample.old_log_prob <= 0.0 for sample in saved_steps))

    def test_random_bucket_executes_deterministically_without_neural_opponent(self):
        one_worker, _one_info = self._collect(
            1,
            opponent_buckets=("random",),
        )
        two_workers, _two_info = self._collect(
            2,
            opponent_buckets=("random",),
        )
        self.assertEqual(
            _rollout_fingerprint(one_worker),
            _rollout_fingerprint(two_workers),
        )
        self.assertTrue(one_worker)
        self.assertEqual(
            {row["opponent_kind"] for row in one_worker},
            {"random"},
        )
        self.assertEqual(
            {row["bank_slot"] for row in one_worker},
            {None},
        )

    def test_medium_term_rollouts_are_invariant_to_worker_count(self):
        buckets = ("heuristic", "recent", "medium_term")
        one_worker, _one_info = self._collect(
            1,
            game_count=18,
            opponent_buckets=buckets,
            prepare=_fill_delayed_band,
        )
        two_workers, _two_info = self._collect(
            2,
            game_count=18,
            opponent_buckets=buckets,
            prepare=_fill_delayed_band,
        )
        self.assertEqual(
            _rollout_fingerprint(one_worker),
            _rollout_fingerprint(two_workers),
        )
        self.assertEqual(
            {row["bucket_name"] for row in one_worker},
            set(buckets),
        )

    def test_warm_up_rollouts_are_invariant_with_an_empty_delayed_band(self):
        buckets = ("heuristic", "recent", "medium_term")
        one_worker, _one_info = self._collect(
            1,
            game_count=18,
            opponent_buckets=buckets,
        )
        two_workers, _two_info = self._collect(
            2,
            game_count=18,
            opponent_buckets=buckets,
        )
        self.assertEqual(
            _rollout_fingerprint(one_worker),
            _rollout_fingerprint(two_workers),
        )
        self.assertEqual(
            {row["bucket_name"] for row in one_worker},
            {"heuristic", "recent"},
        )

    def test_random_only_training_reports_one_slotless_opponent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = self._train(
                iterations=1,
                gpi=4,
                opponent_buckets=("random",),
                difficulty_weight=0.0,
                checkpoint_interval=100,
                seed=2026,
                device="cpu",
                workers=1,
                safety_config=self.safety,
                rl_weights_path=str(Path(temp_dir) / "random-only.npz"),
                quiet=True,
                ppo_max_epochs=1,
            )
        self.assertEqual(summary["opponent_count"], 1)
        self.assertEqual(summary["unique_neural_opponent_count"], 0)
        self.assertEqual(summary["opponent_bucket_sizes"], {"random": 1})

    def test_warm_up_training_keeps_zero_rows_for_an_empty_bucket(self):
        buckets = ("heuristic", "recent", "historical_uniform")
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = Path(temp_dir) / "warm-up.npz"
            summary = self._train(
                iterations=2,
                gpi=6,
                opponent_buckets=buckets,
                difficulty_weight=0.5,
                checkpoint_interval=100,
                log_interval=1,
                seed=2026,
                device="cpu",
                workers=1,
                safety_config=self.safety,
                rl_weights_path=str(weights_path),
                quiet=False,
                ppo_max_epochs=1,
            )
            metrics_path = weights_path.with_name(
                f"{weights_path.stem}_training_metrics.jsonl"
            )
            header, rows = read_training_metrics(metrics_path)
        self.assertEqual(summary["opponent_bucket_sizes"]["historical_uniform"], 0)
        self.assertEqual(header["bucket_results"]["bucket_order"], list(buckets))
        self.assertEqual(len(rows), 2)
        for row in rows:
            bucket_results = row["bucket_results"]
            self.assertEqual(len(bucket_results), len(buckets))
            self.assertEqual(bucket_results[-1], [0, 0, 0])
            self.assertEqual(sum(result[0] for result in bucket_results), 6)

    def _miniature_racing_policy(self, *, batch=4, survivors=2):
        """Patch the fixed racing policy down to a size a test can play.

        The real policy costs 100,000 games. Only the constants are shrunk:
        the stage loop, the seed panels, the seat balance, the tally, the
        ranking, and the pool commit all run unmodified. Workers re-import the
        module in fresh processes and never read the stage table, so patching
        the parent is enough.
        """
        from training.rl.champion_evaluation import ChampionStage

        stages = (
            ChampionStage(games_per_candidate=2, survivors=batch - 1),
            ChampionStage(games_per_candidate=2, survivors=survivors),
        )
        total_games = batch * 2 + (batch - 1) * 2
        patches = {
            "training.rl.pool.CHAMPION_CANDIDATE_BATCH_SIZE": batch,
            "training.rl.pool.CHAMPION_FINAL_SURVIVORS": survivors,
            "training.rl.champion_evaluation.CHAMPION_CANDIDATE_BATCH_SIZE": (
                batch
            ),
            "training.rl.champion_evaluation.CHAMPION_FINAL_SURVIVORS": (
                survivors
            ),
            "training.rl.champion_evaluation.CHAMPION_FINAL_GAMES": 2,
            "training.rl.champion_evaluation.CHAMPION_RACING_STAGES": stages,
            "training.rl.champion_evaluation.CHAMPION_EVALUATION_GAMES": (
                total_games
            ),
        }
        for target, value in patches.items():
            patcher = mock.patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return len(stages), total_games

    def test_training_runs_the_champion_event_and_admits_its_winners(self):
        """The trigger fires from run_iteration and stays out of RL counters."""
        stage_count, racing_games = self._miniature_racing_policy()
        buckets = ("heuristic", "recent", "champion_vs_heuristic")
        iterations = 5
        gpi = 6
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = Path(temp_dir) / "champion.npz"
            summary = self._train(
                iterations=iterations,
                gpi=gpi,
                opponent_buckets=buckets,
                difficulty_weight=0.5,
                checkpoint_interval=100,
                log_interval=1,
                seed=31337,
                device="cpu",
                workers=1,
                safety_config=self.safety,
                rl_weights_path=str(weights_path),
                quiet=False,
                ppo_max_epochs=1,
            )
            metrics_path = weights_path.with_name(
                f"{weights_path.stem}_training_metrics.jsonl"
            )
            header, rows = read_training_metrics(metrics_path)

        champion_state = summary["opponent_pool_final_state"]["buckets"][
            "champion_vs_heuristic"
        ]["champion_state"]
        # The fourth successful update completes the batch, so exactly one
        # event runs and the fifth update starts the next batch.
        self.assertEqual(champion_state["completed_events"], 1)
        self.assertEqual(champion_state["pending_candidates"], 1)
        self.assertEqual(summary["opponent_bucket_sizes"]["champion_vs_heuristic"], 2)
        self.assertIsNotNone(champion_state["minimum_heuristic_win_rate"])
        self.assertIsNotNone(champion_state["maximum_heuristic_win_rate"])

        # Evaluation games are invisible to every RL counter.
        self.assertEqual(summary["completed_training_games"], iterations * gpi)
        self.assertEqual(header["bucket_results"]["bucket_order"], list(buckets))
        self.assertEqual(len(rows), iterations)
        for row in rows:
            self.assertEqual(
                sum(result[0] for result in row["bucket_results"]),
                gpi,
            )
        champion_column = header["bucket_results"]["bucket_order"].index(
            "champion_vs_heuristic"
        )
        # The bucket is empty while its own candidates are still racing, so the
        # racing games cannot have leaked into the event iteration's row.
        for row in rows[:4]:
            self.assertEqual(row["bucket_results"][champion_column], [0, 0, 0])
        self.assertGreater(rows[4]["bucket_results"][champion_column][0], 0)

        # The racing time and worker runs are accounted for separately.
        sections = summary["runtime_profile_delta"]["sections_seconds"]
        self.assertGreater(sections["champion_evaluation"], 0.0)
        self.assertEqual(
            summary["parallel"]["champion_evaluation_batches"],
            stage_count,
        )
        self.assertEqual(summary["parallel"]["rollout_batches"], iterations)
        racing_policy = header["metadata"]["training"]["champion_evaluation"]
        self.assertEqual(racing_policy["total_games"], racing_games)
        self.assertFalse(racing_policy["counts_toward_gpi"])

    def test_training_with_a_racing_event_is_invariant_to_worker_count(self):
        """A completed event must not make training depend on scheduling.

        Racing games run on the same pool as rollouts and each stage waits for
        every one of its games, so a different worker count changes only the
        order the results arrive in. If any part of the event leaked into the
        parent RNG stream or into the learner state, the two runs would diverge
        from iteration 5 on.
        """
        self._miniature_racing_policy()
        buckets = ("heuristic", "recent", "champion_vs_heuristic")
        digests = []
        with tempfile.TemporaryDirectory() as temp_dir:
            for workers, name in ((1, "one.npz"), (2, "two.npz")):
                path = Path(temp_dir) / name
                summary = self._train(
                    iterations=5,
                    gpi=6,
                    opponent_buckets=buckets,
                    difficulty_weight=0.5,
                    checkpoint_interval=100,
                    seed=606,
                    device="cpu",
                    workers=workers,
                    safety_config=self.safety,
                    rl_weights_path=str(path),
                    quiet=True,
                    ppo_max_epochs=1,
                )
                self.assertEqual(
                    summary["opponent_pool_final_state"]["buckets"][
                        "champion_vs_heuristic"
                    ]["champion_state"]["completed_events"],
                    1,
                )
                with np.load(path) as weights:
                    digests.append(tuple(
                        (key, weights[key].tobytes())
                        for key in sorted(weights.files)
                    ))
        self.assertEqual(digests[0], digests[1])

    def test_split_and_uninterrupted_runs_agree_across_a_racing_event(self):
        """Specification 28: resume across a partial batch and a whole event.

        The split point sits between two events with a partial candidate batch
        pending, which is the only state a real checkpoint can ever hold: the
        batch is completed and consumed inside a single ``run_iteration`` call,
        so no published checkpoint can carry a full one.
        """
        self._miniature_racing_policy()
        buckets = ("heuristic", "recent", "champion_vs_heuristic")
        common = {
            "iterations": 8,
            "gpi": 6,
            "opponent_buckets": buckets,
            "difficulty_weight": 0.5,
            "checkpoint_interval": 3,
            "seed": 8080,
            "device": "cpu",
            "workers": 1,
            "safety_config": self.safety,
            "numbered_checkpoints": True,
            "fresh_from_sl": True,
            "quiet": True,
            "ppo_max_epochs": 1,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full_base = root / "full.npz"
            split_base = root / "split.npz"
            full = self._train(rl_weights_path=str(full_base), **common)
            # Iteration 6 is a checkpoint boundary two updates past the first
            # event, so the saved batch is partial.
            self._train(
                rl_weights_path=str(split_base),
                stop_after_training_games=36,
                **common,
            )
            # The numbered checkpoint is keyed by iteration; 36 games at
            # GPI 6 is iteration 6.
            partial = numbered_checkpoint_path(split_base, 6)
            partial_state, _partial_weights = load_resume_state(
                str(partial),
                str(resume_state_path(partial)),
            )
            resumed = self._train(
                rl_weights_path=str(split_base),
                resume_weights_path=str(partial),
                resume_state_file=str(resume_state_path(partial)),
                **{**common, "fresh_from_sl": False},
            )

            with np.load(full["rl_weights_path"], allow_pickle=False) as left:
                with np.load(
                    resumed["rl_weights_path"],
                    allow_pickle=False,
                ) as right:
                    self.assertEqual(sorted(left.files), sorted(right.files))
                    for name in left.files:
                        np.testing.assert_array_equal(left[name], right[name])

            full_state, _full_weights = load_resume_state(
                full["rl_weights_path"],
                full["resume_state_path"],
            )
            resumed_state, _resumed_weights = load_resume_state(
                resumed["rl_weights_path"],
                resumed["resume_state_path"],
            )

        # The checkpoint the resume started from held a partial batch.
        partial_champion = _champion_state(partial_state)
        self.assertEqual(partial_champion["completed_event_count"], 1)
        self.assertEqual(len(partial_champion["pending_candidate_ids"]), 2)

        full_champion = _champion_state(full_state)
        self.assertEqual(full_champion["completed_event_count"], 2)

        # Candidate IDs, event index, survivors, champion IDs, champion scores,
        # and champion membership all agree. Stage seeds agree by construction
        # once the effective seed and the event index do, because the panel is
        # a pure function of the two.
        self.assertEqual(
            _champion_state(full_state),
            _champion_state(resumed_state),
        )
        self.assertEqual(
            full_state["opponent_pool_state"]["buckets"],
            resumed_state["opponent_pool_state"]["buckets"],
        )
        self.assertEqual(
            full_state["opponent_pool_state"]["lifecycle_counters"],
            resumed_state["opponent_pool_state"]["lifecycle_counters"],
        )
        # Uniform anchors, and therefore the next normal match plan, agree: the
        # plan is a pure function of pool, tracker, anchors, seed, and iteration.
        self.assertEqual(
            full_state["uniform_rotation_state"],
            resumed_state["uniform_rotation_state"],
        )
        self.assertEqual(
            full_state["opponent_performance_state"],
            resumed_state["opponent_performance_state"],
        )
        self.assertEqual(full["effective_seed"], resumed["effective_seed"])

    def test_worker_tuning_cannot_advance_champion_or_rotation_state(self):
        """Specification 29: discarded benchmark work touches neither.

        This drives the real ``benchmark_worker_candidates`` against a real
        throwaway ``RLRolloutRunner``, not a stub, so the isolation is proved
        on the code path production uses.
        """
        import copy

        from training.rl import adaptive_tuning
        from training.rl.adaptive_tuning import benchmark_worker_candidates

        self._miniature_racing_policy()
        buckets = ("heuristic", "recent", "champion_vs_heuristic")
        network = self._network()
        source = RLRolloutRunner(
            network,
            opponent_buckets=buckets,
            schema=dict(REWARD_SCHEMAS),
            gamma=1.0,
            safety=self.safety,
        )
        try:
            pool = source.opponent_pool
            # One short of a complete batch: a benchmark that appended even a
            # single candidate would trigger an event.
            for iteration in range(1, 4):
                pool.consider_updated_policy(
                    network,
                    iteration=iteration,
                    completed_games=iteration * 10,
                    has_samples=True,
                )
            source.uniform_rotation.commit({
                "recent": pool.bucket_members("recent")[0],
            })
            # The training loop reconciles the tracker every iteration, so a
            # realistic export covers every active identity.
            active = tuple(
                record.opponent_id for record in pool.active_opponents()
            )
            source.performance_tracker.ensure(active)
            source.performance_tracker.retain_only(active)
            pool_state = pool.export_state()
            pool_weights = pool.export_weights()
            performance_state = source.performance_tracker.export_state()
            rotation_state = source.uniform_rotation.export_state()
        finally:
            source.close()

        before = copy.deepcopy(pool_state)
        rotation_before = copy.deepcopy(rotation_state)
        captured = []
        real_new_runner = adaptive_tuning._new_runner

        def spy(*args, **kwargs):
            runner = real_new_runner(*args, **kwargs)
            captured.append(runner)
            return runner

        def refuse(_self, _specs):
            raise AssertionError("worker tuning ran a champion evaluation")

        with (
            mock.patch.object(adaptive_tuning, "_new_runner", side_effect=spy),
            mock.patch.object(
                RLRolloutRunner,
                "evaluate_champion_games",
                refuse,
            ),
            mock.patch.object(
                adaptive_tuning,
                "DEFAULT_RL_WORKER_CANDIDATES",
                (1,),
            ),
        ):
            test_games, rows = benchmark_worker_candidates(
                self._network(),
                gpi=4,
                total_training_games=400,
                base_seed=99,
                opponent_buckets=buckets,
                difficulty_weight=0.5,
                schema=dict(REWARD_SCHEMAS),
                gamma=1.0,
                safety=self.safety,
                pool_state=pool_state,
                pool_weights=pool_weights,
                performance_state=performance_state,
                rotation_state=rotation_state,
            )

        self.assertEqual(len(captured), 1)
        benchmark_pool = captured[0].opponent_pool
        # The throwaway pool saw only benchmark games: no candidate appended,
        # no event run.
        self.assertEqual(len(benchmark_pool.champion_pending_candidate_ids(HEURISTIC_CHAMPION)), 3)
        self.assertEqual(benchmark_pool.champion_completed_event_count(HEURISTIC_CHAMPION), 0)
        self.assertEqual(benchmark_pool.bucket_members("champion_vs_heuristic"), ())

        # The parent's serialized state is untouched, so the first real plan
        # after tuning is the plan the run would have built without it.
        self.assertEqual(pool_state, before)
        self.assertEqual(rotation_state, rotation_before)

        # Only the planned rollout games are retained as benchmark work.
        self.assertEqual(rows[0]["actual_games"], test_games)
        self.assertTrue(rows[0]["success"])

    def test_a_run_without_the_champion_bucket_reports_no_racing_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = Path(temp_dir) / "plain.npz"
            summary = self._train(
                iterations=1,
                gpi=4,
                checkpoint_interval=100,
                seed=11,
                device="cpu",
                workers=1,
                safety_config=self.safety,
                rl_weights_path=str(weights_path),
                quiet=True,
                ppo_max_epochs=1,
            )
            header, _rows = read_training_metrics(weights_path.with_name(
                f"{weights_path.stem}_training_metrics.jsonl"
            ))
        self.assertIsNone(header["metadata"]["training"]["champion_evaluation"])
        # The section only exists on iterations that actually raced.
        self.assertNotIn(
            "champion_evaluation",
            summary["runtime_profile_delta"]["sections_seconds"],
        )

    def test_seeded_training_checkpoints_are_bit_identical(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = [Path(temp_dir) / "one.npz", Path(temp_dir) / "two.npz"]
            summaries = []
            for workers, path in zip((1, 2), paths):
                summaries.append(self._train(
                    iterations=3,
                    gpi=6,
                    checkpoint_interval=100,
                    seed=987,
                    device="cpu",
                    workers=workers,
                    safety_config=self.safety,
                    rl_weights_path=str(path),
                    quiet=True,
                ))
            with np.load(paths[0], allow_pickle=False) as one:
                with np.load(paths[1], allow_pickle=False) as two:
                    self.assertEqual(one.files, two.files)
                    for name in one.files:
                        np.testing.assert_array_equal(one[name], two[name])
            self.assertEqual(summaries[0]["effective_seed"], 987)
            self.assertEqual(summaries[1]["effective_seed"], 987)

    def test_numbered_policy_checkpoint_history_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary = self._train(
                iterations=NUMBERED_CHECKPOINT_WEIGHT_RETENTION + 2,
                gpi=1,
                checkpoint_interval=1,
                seed=230723,
                device="cpu",
                workers=1,
                safety_config=self.safety,
                rl_weights_path=str(root / "training.npz"),
                numbered_checkpoints=True,
                fresh_from_sl=True,
                quiet=True,
                ppo_max_epochs=1,
            )
            policy_files = [
                path
                for path in root.glob("training_iter*.npz")
                if ".resume." not in path.name
            ]
            resume_files = list(root.glob("training_iter*.resume.npz"))
            metadata, _pool = load_resume_state(
                summary["rl_weights_path"],
                summary["resume_state_path"],
            )

        self.assertEqual(
            len(policy_files),
            NUMBERED_CHECKPOINT_WEIGHT_RETENTION,
        )
        self.assertEqual(len(resume_files), 1)
        self.assertEqual(
            metadata["completed_training_games"],
            NUMBERED_CHECKPOINT_WEIGHT_RETENTION + 2,
        )

    def test_pool_refresh_follows_gpi(self):
        """One frozen snapshot per iteration, so the cadence is exactly gpi.

        Both runs play the same 12 real games. The smaller gpi runs more
        iterations and therefore stores more snapshots; each pool also holds the
        initial policy appended when the runner is built.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = []
            first = self._train(
                iterations=6,
                gpi=2,
                checkpoint_interval=2,
                seed=321,
                device="cpu",
                workers=1,
                safety_config=self.safety,
                rl_weights_path=str(root / "gpi2.npz"),
                metrics_callback=rows.append,
                quiet=True,
                ppo_max_epochs=1,
            )
            second = self._train(
                iterations=4,
                gpi=3,
                checkpoint_interval=2,
                seed=321,
                device="cpu",
                workers=1,
                safety_config=self.safety,
                rl_weights_path=str(root / "gpi3.npz"),
                quiet=True,
                ppo_max_epochs=1,
            )

        self.assertEqual(first["total_training_games"], 12)
        self.assertEqual(second["total_training_games"], 12)
        self.assertEqual(first["opponent_bucket_sizes"]["recent"], 7)
        self.assertEqual(second["opponent_bucket_sizes"]["recent"], 5)
        self.assertEqual(first["parallel"]["rollout_batches"], 6)
        self.assertNotIn("evaluation_batches", first["parallel"])
        self.assertTrue(all("checkpoint_eval_games" not in row for row in rows))
        self.assertFalse(hasattr(RLRolloutRunner, "evaluate_current_against_heuristic"))

    def test_archive_reuses_the_tenth_admitted_snapshot_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary = self._train(
                iterations=10,
                gpi=1,
                checkpoint_interval=10,
                seed=404,
                device="cpu",
                workers=1,
                safety_config=self.safety,
                rl_weights_path=str(root / "training.npz"),
                numbered_checkpoints=True,
                fresh_from_sl=True,
                quiet=True,
                ppo_max_epochs=1,
            )
            metadata, _weights = load_resume_state(
                summary["rl_weights_path"],
                summary["resume_state_path"],
            )
            archive = json.loads(
                (root / "checkpoint_archive" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        admitted = next(
            value
            for value in metadata["opponent_pool_state"]["opponents"]
            if value["introduced_iteration"] == 10
        )
        archived = next(
            value
            for value in archive["checkpoints"]
            if value["completed_iteration"] == 10
        )
        self.assertEqual(archived["opponent_id"], admitted["opponent_id"])
        self.assertEqual(archived["checkpoint_id"], admitted["checkpoint_id"])

    BAND_ITERATIONS = K_RECENT + MEDIUM_TERM_INTERVAL_ITERATIONS
    BAND_CHECKPOINT_INTERVAL = 105

    def _band_run(self, root, name, **overrides):
        # Each run owns its artifact directory: the checkpoint archive is
        # keyed by absolute iteration, so sharing one would let a finished run
        # collide with a resumed run's descendants.
        directory = root / name
        directory.mkdir(exist_ok=True)
        common = {
            "iterations": self.BAND_ITERATIONS,
            "gpi": 1,
            "opponent_buckets": ("recent", "medium_term"),
            "difficulty_weight": 0.5,
            "checkpoint_interval": self.BAND_CHECKPOINT_INTERVAL,
            "seed": 77,
            "device": "cpu",
            "workers": 1,
            "safety_config": self.safety,
            "numbered_checkpoints": True,
            "fresh_from_sl": True,
            "quiet": True,
            "ppo_max_epochs": 1,
        }
        return self._train(
            rl_weights_path=str(directory / f"{name}.npz"),
            **{**common, **overrides},
        )

    def test_delayed_band_populates_and_pins_during_real_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary = self._band_run(root, "band")
            metadata, weights = load_resume_state(
                summary["rl_weights_path"],
                summary["resume_state_path"],
            )
            archive = json.loads(
                (root / "band" / "checkpoint_archive" / "manifest.json")
                .read_text(encoding="utf-8")
            )
        pool_state = metadata["opponent_pool_state"]
        medium_ids = pool_state["buckets"]["medium_term"]["member_ids"]
        recent_ids = pool_state["buckets"]["recent"]["member_ids"]
        iterations = {
            value["opponent_id"]: value["introduced_iteration"]
            for value in pool_state["opponents"]
        }
        completed = int(metadata["completed_iteration"])
        self.assertTrue(medium_ids)
        self.assertEqual(set(medium_ids) & set(recent_ids), set())
        self.assertTrue(all(
            iterations[value] <= completed - K_RECENT for value in medium_ids
        ))
        self.assertLess(
            max(iterations[value] for value in medium_ids),
            min(iterations[value] for value in recent_ids),
        )
        self.assertEqual(
            len(weights),
            len(set(recent_ids) | set(medium_ids)),
        )
        # Active band members and every milestone still waiting inside recent
        # stay pinned, so nothing the band will need can be thinned.
        pinned = {
            value["opponent_id"]
            for value in archive["checkpoints"]
            if value["pinned"]
        }
        self.assertTrue(set(medium_ids) <= pinned)
        archived_ids = {value["opponent_id"] for value in archive["checkpoints"]}
        self.assertTrue(archived_ids & set(recent_ids) <= pinned)

    def test_split_resume_across_the_band_boundary_is_bit_identical(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stop = self.BAND_ITERATIONS - self.BAND_CHECKPOINT_INTERVAL
            full = self._band_run(root, "full")
            self._band_run(root, "split", stop_after_training_games=stop)
            partial = numbered_checkpoint_path(root / "split" / "split.npz", stop)
            resumed = self._band_run(
                root,
                "split",
                fresh_from_sl=False,
                resume_weights_path=str(partial),
                resume_state_file=str(resume_state_path(partial)),
            )
            with np.load(full["rl_weights_path"], allow_pickle=False) as left:
                with np.load(
                    resumed["rl_weights_path"],
                    allow_pickle=False,
                ) as right:
                    self.assertEqual(sorted(left.files), sorted(right.files))
                    for name in left.files:
                        np.testing.assert_array_equal(left[name], right[name])
            full_state, _full_weights = load_resume_state(
                full["rl_weights_path"],
                full["resume_state_path"],
            )
            resumed_state, _resumed_weights = load_resume_state(
                resumed["rl_weights_path"],
                resumed["resume_state_path"],
            )
        for state in (full_state, resumed_state):
            self.assertTrue(
                state["opponent_pool_state"]["buckets"]["medium_term"][
                    "member_ids"
                ]
            )
        # The refresh straddling the resume is neither skipped nor duplicated.
        self.assertEqual(
            full_state["opponent_pool_state"]["buckets"],
            resumed_state["opponent_pool_state"]["buckets"],
        )
        self.assertEqual(
            full_state["opponent_pool_state"]["lifecycle_counters"],
            resumed_state["opponent_pool_state"]["lifecycle_counters"],
        )

    def test_split_resume_preserves_archive_identity_and_an_empty_band(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full_base = root / "full.npz"
            resumed_base = root / "resumed.npz"
            common = {
                "iterations": 12,
                "gpi": 1,
                "opponent_buckets": ("recent", "medium_term"),
                "difficulty_weight": 0.5,
                "checkpoint_interval": 3,
                "seed": 1010,
                "device": "cpu",
                "workers": 1,
                "safety_config": self.safety,
                "numbered_checkpoints": True,
                "fresh_from_sl": True,
                "quiet": True,
                "ppo_max_epochs": 1,
            }
            full = self._train(
                rl_weights_path=str(full_base),
                **common,
            )
            self._train(
                rl_weights_path=str(resumed_base),
                stop_after_training_games=9,
                **common,
            )
            partial_weights = numbered_checkpoint_path(resumed_base, 9)
            resume_options = {**common, "fresh_from_sl": False}
            resumed = self._train(
                rl_weights_path=str(resumed_base),
                resume_weights_path=str(partial_weights),
                resume_state_file=str(resume_state_path(partial_weights)),
                **resume_options,
            )

            with np.load(full["rl_weights_path"], allow_pickle=False) as left:
                with np.load(resumed["rl_weights_path"], allow_pickle=False) as right:
                    for name in left.files:
                        np.testing.assert_array_equal(left[name], right[name])

            metadata, weights = load_resume_state(
                resumed["rl_weights_path"],
                resumed["resume_state_path"],
            )
            pool_state = metadata["opponent_pool_state"]
            # Twelve iterations are far inside the recent band, so the delayed
            # bucket is genuinely empty while the archive already holds the
            # milestones that will eventually populate it.
            self.assertEqual(pool_state["buckets"]["medium_term"]["member_ids"], [])
            recent_ids = pool_state["buckets"]["recent"]["member_ids"]
            self.assertEqual(len(weights), len(set(recent_ids)))
            iterations = {
                value["opponent_id"]: value["introduced_iteration"]
                for value in pool_state["opponents"]
            }

            archive = json.loads(
                (root / "checkpoint_archive" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                [value["completed_iteration"] for value in archive["checkpoints"]],
                [0, 10],
            )
            self.assertEqual(
                {
                    iterations[value["opponent_id"]]
                    for value in archive["checkpoints"]
                },
                {0, 10},
            )
            milestone = next(
                value for value in archive["checkpoints"]
                if value["completed_iteration"] == 10
            )
            with np.load(
                root / "checkpoint_archive" / milestone["filename"],
                allow_pickle=False,
            ) as archived_weights:
                for name in archived_weights.files:
                    np.testing.assert_array_equal(
                        archived_weights[name],
                        weights[milestone["opponent_id"]][name],
                    )

    def test_fresh_from_sl_ignores_existing_rl_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sl_path = root / "sl.npz"
            rl_path = root / "rl.npz"
            with np.load(self.sl_weights_path, allow_pickle=False) as source:
                np.savez(sl_path, **{name: source[name] for name in source.files})

            existing_rl = PolicyNetwork.load_from_sl(sl_path, device="cpu")
            existing_rl.W1 += 1.0
            existing_rl.save(rl_path)

            fresh = _load_initial_network(
                0.001,
                sl_path,
                rl_path,
                quiet=True,
                device="cpu",
                fresh_from_sl=True,
            )
            continued = _load_initial_network(
                0.001,
                sl_path,
                rl_path,
                quiet=True,
                device="cpu",
                fresh_from_sl=False,
            )

            with self.assertRaisesRegex(ValueError, "predates algorithm metadata"):
                self._train(
                    iterations=1,
                    gpi=2,
                    sl_weights_path=str(sl_path),
                    rl_weights_path=str(rl_path),
                    seed=7,
                    device="cpu",
                    workers=1,
                    safety_config=self.safety,
                    quiet=True,
                )

            with np.load(sl_path, allow_pickle=False) as supervised:
                np.testing.assert_array_equal(fresh.W1, supervised["W1"])
            np.testing.assert_array_equal(continued.W1, existing_rl.W1)
            self.assertFalse(np.array_equal(fresh.W1, continued.W1))

    def test_numbered_checkpoint_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full_base = root / "full.npz"
            resumed_base = root / "resumed.npz"
            common = {
                "gpi": 4,
                "checkpoint_interval": 2,
                "seed": 987,
                "device": "cpu",
                "workers": 1,
                "safety_config": self.safety,
                "quiet": True,
                "numbered_checkpoints": True,
                "ppo_max_epochs": 1,
            }

            full = self._train(
                iterations=4,
                rl_weights_path=str(full_base),
                **common,
            )
            self._train(
                iterations=4,
                stop_after_training_games=8,
                rl_weights_path=str(resumed_base),
                **common,
            )
            partial_weights = numbered_checkpoint_path(resumed_base, 2)
            partial_state = resume_state_path(partial_weights)
            metadata, pool = load_resume_state(partial_weights, partial_state)
            self.assertEqual(metadata["completed_iteration"], 2)
            self.assertEqual(metadata["completed_training_games"], 8)
            self.assertEqual(
                metadata["configuration"]["rl_training_algorithm"],
                "reinforce_v1",
            )
            self.assertEqual(metadata["optimizer_state"]["step_count"], 2)
            self.assertTrue(metadata["adaptive_tuning"]["isolation_verified"])
            self.assertEqual(len(pool), 3)
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                load_resume_state(full["rl_weights_path"], partial_state)

            resumed = self._train(
                iterations=4,
                rl_weights_path=str(resumed_base),
                resume_weights_path=str(partial_weights),
                resume_state_file=str(partial_state),
                gamma=0.97,
                **common,
            )
            self.assertEqual(resumed["gamma"], common.get("gamma", 1.0))
            self.assertEqual(resumed["rl_training_algorithm"], "reinforce_v1")
            with np.load(full["rl_weights_path"], allow_pickle=False) as left:
                with np.load(resumed["rl_weights_path"], allow_pickle=False) as right:
                    self.assertEqual(left.files, right.files)
                    for name in left.files:
                        np.testing.assert_array_equal(left[name], right[name])

            final_weights = numbered_checkpoint_path(resumed_base, 4)
            self.assertEqual(resumed["rl_weights_path"], str(final_weights))
            self.assertFalse(partial_state.exists())
            self.assertTrue(resume_state_path(final_weights).exists())
            self.assertEqual(full["games_per_iteration"], resumed["games_per_iteration"])
            self.assertEqual(full["selected_workers"], resumed["selected_workers"])
            self.assertEqual(full["optimizer_step_count"], resumed["optimizer_step_count"])
            self.assertEqual(
                resumed["ppo_configuration"]["fixed_policy"]["target_kl"],
                0.01,
            )

    def test_uniform_rotation_survives_a_numbered_checkpoint_resume(self):
        """A split run must continue the rotation, not restart every bucket."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full_base = root / "full.npz"
            resumed_base = root / "resumed.npz"
            common = {
                "gpi": 4,
                "checkpoint_interval": 3,
                "seed": 4242,
                "device": "cpu",
                "workers": 1,
                "safety_config": self.safety,
                "quiet": True,
                "numbered_checkpoints": True,
                "ppo_max_epochs": 1,
                "opponent_buckets": ("heuristic", "recent"),
            }

            full = self._train(iterations=6, rl_weights_path=str(full_base), **common)
            self._train(
                iterations=6,
                stop_after_training_games=12,
                rl_weights_path=str(resumed_base),
                **common,
            )
            partial_weights = numbered_checkpoint_path(resumed_base, 3)
            partial_state = resume_state_path(partial_weights)
            metadata, _pool = load_resume_state(partial_weights, partial_state)

            rotation_state = metadata["uniform_rotation_state"]
            anchors = rotation_state["anchors"]
            self.assertEqual(sorted(anchors), ["heuristic", "recent"])
            # Two uniform games split one per bucket, so the single-member
            # heuristic bucket never has a remainder while recent always does.
            self.assertIsNone(anchors["heuristic"])
            self.assertIsNotNone(anchors["recent"])

            resumed = self._train(
                iterations=6,
                rl_weights_path=str(resumed_base),
                resume_weights_path=str(partial_weights),
                resume_state_file=str(partial_state),
                **common,
            )
            # Diverging anchors would change which opponents played which games
            # and therefore the learner update itself.
            with np.load(full["rl_weights_path"], allow_pickle=False) as left:
                with np.load(resumed["rl_weights_path"], allow_pickle=False) as right:
                    self.assertEqual(left.files, right.files)
                    for name in left.files:
                        np.testing.assert_array_equal(left[name], right[name])

            final_state = resume_state_path(numbered_checkpoint_path(resumed_base, 6))
            full_state = resume_state_path(numbered_checkpoint_path(full_base, 6))
            resumed_metadata, _resumed_pool = load_resume_state(
                numbered_checkpoint_path(resumed_base, 6),
                final_state,
            )
            full_metadata, _full_pool = load_resume_state(
                numbered_checkpoint_path(full_base, 6),
                full_state,
            )
            self.assertEqual(
                resumed_metadata["uniform_rotation_state"],
                full_metadata["uniform_rotation_state"],
            )

    def test_champion_state_survives_a_numbered_checkpoint_resume(self):
        """Pending candidates and the event count are durable exact state."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full_base = root / "full.npz"
            resumed_base = root / "resumed.npz"
            common = {
                "gpi": 4,
                "checkpoint_interval": 3,
                "seed": 606,
                "device": "cpu",
                "workers": 1,
                "safety_config": self.safety,
                "quiet": True,
                "numbered_checkpoints": True,
                "ppo_max_epochs": 1,
                "opponent_buckets": (
                    "heuristic",
                    "recent",
                    "champion_vs_heuristic",
                ),
            }
            full = self._train(iterations=6, rl_weights_path=str(full_base), **common)
            self._train(
                iterations=6,
                stop_after_training_games=12,
                rl_weights_path=str(resumed_base),
                **common,
            )
            partial_weights = numbered_checkpoint_path(resumed_base, 3)
            partial_state = resume_state_path(partial_weights)
            metadata, _pool = load_resume_state(partial_weights, partial_state)

            champion = _champion_state(metadata)
            pending = champion["pending_candidate_ids"]
            # Every successful update queues exactly one candidate, and no
            # racing event can have happened at only three iterations.
            self.assertEqual(len(pending), 3)
            self.assertEqual(len(set(pending)), 3)
            self.assertEqual(champion["completed_event_count"], 0)
            self.assertEqual(champion["heuristic_win_rate_by_opponent_id"], {})
            recent = metadata["opponent_pool_state"]["buckets"]["recent"]
            self.assertTrue(set(pending) <= set(recent["member_ids"]))
            self.assertEqual(
                metadata["opponent_pool_state"]["buckets"][
                    "champion_vs_heuristic"
                ]["member_ids"],
                [],
            )

            resumed = self._train(
                iterations=6,
                rl_weights_path=str(resumed_base),
                resume_weights_path=str(partial_weights),
                resume_state_file=str(partial_state),
                **common,
            )
            with np.load(full["rl_weights_path"], allow_pickle=False) as left:
                with np.load(resumed["rl_weights_path"], allow_pickle=False) as right:
                    self.assertEqual(left.files, right.files)
                    for name in left.files:
                        np.testing.assert_array_equal(left[name], right[name])

            resumed_metadata, _resumed_pool = load_resume_state(
                numbered_checkpoint_path(resumed_base, 6),
                resume_state_path(numbered_checkpoint_path(resumed_base, 6)),
            )
            full_metadata, _full_pool = load_resume_state(
                numbered_checkpoint_path(full_base, 6),
                resume_state_path(numbered_checkpoint_path(full_base, 6)),
            )
            self.assertEqual(
                _champion_state(resumed_metadata),
                _champion_state(full_metadata),
            )
            self.assertEqual(
                len(
                    _champion_state(resumed_metadata)[
                        "pending_candidate_ids"
                    ]
                ),
                6,
            )

    def test_a_pre_rotation_resume_state_is_rejected_with_a_reason(self):
        """Version 12 stored no anchor, so it cannot be reinterpreted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base = root / "run.npz"
            self._train(
                iterations=2,
                gpi=4,
                checkpoint_interval=2,
                seed=11,
                device="cpu",
                workers=1,
                safety_config=self.safety,
                quiet=True,
                numbered_checkpoints=True,
                ppo_max_epochs=1,
                rl_weights_path=str(base),
            )
            weights = numbered_checkpoint_path(base, 2)
            state_path = resume_state_path(weights)
            with np.load(state_path, allow_pickle=False) as state:
                payload = {name: state[name] for name in state.files}
            metadata = json.loads(str(payload["metadata_json"].item()))
            metadata["version"] = 12
            metadata.pop("uniform_rotation_state")
            payload["metadata_json"] = np.asarray(json.dumps(metadata))
            np.savez(state_path, **payload)
            with self.assertRaisesRegex(ValueError, "rotation anchor"):
                load_resume_state(weights, state_path)

    def test_four_hidden_layer_run_trains_and_resumes_exactly(self):
        """Carry a deeper architecture through rollouts, PPO, and exact resume."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            supervised = PolicyNetwork(
                random_seed=31,
                device="cpu",
                hidden_sizes=(64, 48, 32, 16),
            )
            sl_path = root / "deep_sl.npz"
            supervised.save(sl_path)

            full_base = root / "full.npz"
            resumed_base = root / "resumed.npz"
            common = {
                "sl_weights_path": str(sl_path),
                "gpi": 4,
                "checkpoint_interval": 2,
                "seed": 4242,
                "device": "cpu",
                "workers": 1,
                "safety_config": self.safety,
                "quiet": True,
                "numbered_checkpoints": True,
            }

            full = self._train(
                iterations=4, rl_weights_path=str(full_base), **common
            )
            self._train(
                iterations=4,
                stop_after_training_games=8,
                rl_weights_path=str(resumed_base),
                **common,
            )
            partial_weights = numbered_checkpoint_path(resumed_base, 2)
            partial_state = resume_state_path(partial_weights)
            _metadata, pool = load_resume_state(partial_weights, partial_state)
            # Opponent snapshots keep all five weight layers of the deeper stack.
            self.assertEqual(
                sorted(next(iter(pool.values()))),
                sorted(supervised.weight_names),
            )

            resumed = self._train(
                iterations=4,
                rl_weights_path=str(resumed_base),
                resume_weights_path=str(partial_weights),
                resume_state_file=str(partial_state),
                **common,
            )
            with np.load(full["rl_weights_path"], allow_pickle=False) as left:
                with np.load(resumed["rl_weights_path"], allow_pickle=False) as right:
                    self.assertEqual(left.files, right.files)
                    self.assertEqual(
                        [name for name in left.files if name.startswith("W")],
                        ["W1", "W2", "W3", "W4", "W5"],
                    )
                    for name in left.files:
                        np.testing.assert_array_equal(left[name], right[name])
            self.assertEqual(
                full["optimizer_step_count"],
                resumed["optimizer_step_count"],
            )

    def test_exact_game_budget_uses_a_partial_final_iteration(self):
        rows = []
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = self._train(
                total_training_games=5,
                gpi=2,
                checkpoint_interval=10,
                seed=123,
                device="cpu",
                workers=1,
                safety_config=self.safety,
                rl_weights_path=str(Path(temp_dir) / "partial.npz"),
                metrics_callback=rows.append,
                quiet=True,
            )
            with np.load(summary["rl_weights_path"], allow_pickle=False) as checkpoint:
                saved_algorithm = str(checkpoint["rl_training_algorithm"])

        self.assertEqual([row["games"] for row in rows], [2, 2, 1])
        self.assertEqual([row["cumulative_games"] for row in rows], [2, 4, 5])
        self.assertEqual(summary["completed_training_games"], 5)
        self.assertEqual(summary["completed_iterations_this_run"], 3)
        self.assertEqual(saved_algorithm, "ppo_v2_decision_minibatches")

    def test_numbered_resume_restores_value_head_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            common = {
                "gpi": 4,
                "checkpoint_interval": 1,
                "use_value_head": True,
                "ppo_max_epochs": 1,
                "seed": 654,
                "device": "cpu",
                "workers": 1,
                "safety_config": self.safety,
                "quiet": True,
                "numbered_checkpoints": True,
            }
            full_base = root / "full_critic.npz"
            resumed_base = root / "resumed_critic.npz"
            full = self._train(
                iterations=3, rl_weights_path=str(full_base), **common
            )
            self._train(
                iterations=3,
                stop_after_training_games=4,
                rl_weights_path=str(resumed_base),
                **common,
            )
            partial_weights = numbered_checkpoint_path(resumed_base, 1)
            resumed = self._train(
                iterations=3,
                rl_weights_path=str(resumed_base),
                resume_weights_path=str(partial_weights),
                resume_state_file=str(resume_state_path(partial_weights)),
                **common,
            )

            with np.load(full["rl_weights_path"], allow_pickle=False) as left:
                with np.load(resumed["rl_weights_path"], allow_pickle=False) as right:
                    self.assertIn("Wv", left.files)
                    self.assertIn("bv", left.files)
                    for name in left.files:
                        np.testing.assert_array_equal(left[name], right[name])

    def test_numbered_resume_restores_ppo_value_head_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            common = {
                "gpi": 4,
                "checkpoint_interval": 1,
                "use_value_head": True,
                "ppo_max_epochs": 2,
                "seed": 655,
                "device": "cpu",
                "workers": 1,
                "safety_config": self.safety,
                "quiet": True,
                "numbered_checkpoints": True,
            }
            full_base = root / "full_ppo_critic.npz"
            resumed_base = root / "resumed_ppo_critic.npz"
            full = self._train(
                iterations=3, rl_weights_path=str(full_base), **common
            )
            self._train(
                iterations=3,
                stop_after_training_games=4,
                rl_weights_path=str(resumed_base),
                **common,
            )
            partial_weights = numbered_checkpoint_path(resumed_base, 1)
            resumed = self._train(
                iterations=3,
                rl_weights_path=str(resumed_base),
                resume_weights_path=str(partial_weights),
                resume_state_file=str(resume_state_path(partial_weights)),
                **common,
            )

            with np.load(full["rl_weights_path"], allow_pickle=False) as left:
                with np.load(resumed["rl_weights_path"], allow_pickle=False) as right:
                    self.assertIn("Wv", left.files)
                    self.assertIn("bv", left.files)
                    for name in left.files:
                        np.testing.assert_array_equal(left[name], right[name])

    def test_autotuning_discards_benchmark_games(self):
        messages = []
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch(
                "training.rl.adaptive_tuning.DEFAULT_RL_WORKER_CANDIDATES",
                (1, 2),
            ):
                summary = self._train(
                    iterations=4,
                    gpi=4,
                    checkpoint_interval=100,
                    seed=44,
                    device="cpu",
                    workers="auto",
                    safety_config=self.safety,
                    status_callback=messages.append,
                    rl_weights_path=str(Path(temp_dir) / "auto.npz"),
                    quiet=True,
                )
        tuning = summary["autotune"]
        self.assertIsNone(tuning["iterations_per_test"])
        self.assertEqual(tuning["games_per_test"], 1)
        self.assertEqual(tuning["reused_iteration_count"], 0)
        self.assertEqual(tuning["reused_game_count"], 0)
        self.assertEqual(tuning["discarded_game_count"], 2)
        self.assertEqual(len(tuning["attempts"]), 2)
        self.assertTrue(all(attempt["success"] for attempt in tuning["attempts"]))
        self.assertEqual(summary["completed_training_games"], 16)
        self.assertTrue(summary["adaptive_tuning"]["isolation_verified"])
        self.assertTrue(any("Selecting worker count" in message for message in messages))

    def test_low_ram_stops_autotuning_before_unsafe_candidate(self):
        constrained_memory = {
            "DOMINO_TEST_AVAILABLE_RAM_MB": "400",
            "DOMINO_TEST_TOTAL_RAM_MB": "4096",
        }
        safety = ParallelSafetyConfig(
            memory_reserve_mb=0,
            estimated_worker_mb=300,
            max_worker_rss_mb=1024,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                mock.patch.dict(os.environ, constrained_memory, clear=False),
                mock.patch(
                    "training.rl.adaptive_tuning.DEFAULT_RL_WORKER_CANDIDATES",
                    (1, 2),
                ),
            ):
                summary = self._train(
                    iterations=4,
                    gpi=4,
                    checkpoint_interval=100,
                    seed=55,
                    device="cpu",
                    workers="auto",
                    safety_config=safety,
                    status_callback=lambda _message: None,
                    rl_weights_path=str(Path(temp_dir) / "limited.npz"),
                    quiet=True,
                )
        self.assertEqual(summary["selected_workers"], 1)
        self.assertEqual(len(summary["autotune"]["attempts"]), 2)
        self.assertTrue(summary["autotune"]["attempts"][0]["success"])
        self.assertFalse(summary["autotune"]["attempts"][1]["success"])
        self.assertEqual(
            summary["autotune"]["attempts"][1]["completed_games"],
            0,
        )

    def test_runtime_memory_pressure_retains_games_and_reduces_workers(self):
        safety = ParallelSafetyConfig(
            memory_reserve_mb=512,
            estimated_worker_mb=1,
            max_worker_rss_mb=1024,
            memory_check_interval_s=0.0,
            poll_interval_s=0.01,
        )
        network = self._network()
        runner = RLRolloutRunner(
            network,
            opponent_buckets=("heuristic", "recent"),
            schema=dict(REWARD_SCHEMAS),
            gamma=1.0,
            safety=safety,
        )
        try:
            runner.set_workers(4)
            original_run_jobs = runner._run_jobs
            pressure_triggered = False

            def pressure_after_first_result(
                jobs,
                worker_function,
                on_result,
                run_info,
            ):
                nonlocal pressure_triggered
                if pressure_triggered:
                    return original_run_jobs(
                        jobs,
                        worker_function,
                        on_result,
                        run_info,
                    )

                def store_then_fail(result):
                    nonlocal pressure_triggered
                    on_result(result)
                    pressure_triggered = True
                    raise DiagnosticMemoryPressure("simulated runtime pressure")

                return original_run_jobs(
                    jobs,
                    worker_function,
                    store_then_fail,
                    run_info,
                )

            with mock.patch.object(
                runner,
                "_run_jobs",
                side_effect=pressure_after_first_result,
            ):
                plan = build_match_plan(
                    opponent_pool=runner.opponent_pool,
                    performance_tracker=runner.performance_tracker,
                    selected_buckets=("heuristic", "recent"),
                    uniform_rotation=UniformRotationState(("heuristic", "recent")),
                    difficulty_weight=0.5,
                    iteration=1,
                    first_absolute_game=0,
                    game_count=24,
                    base_seed=4321,
                )
                recovered, run_info = runner.collect_games(plan, 4321)
        finally:
            runner.close()

        baseline, _baseline_info = self._collect(1, game_count=24, seed=4321)
        self.assertEqual(
            _rollout_fingerprint(recovered),
            _rollout_fingerprint(baseline),
        )
        self.assertGreaterEqual(run_info.fallback_count, 1)
        self.assertLess(run_info.final_workers, 4)
        self.assertGreater(
            run_info.fallback_history[-1]["completed_games"],
            0,
        )

    def test_rl_preflight_handles_low_host_and_gpu_memory(self):
        low_gpu = {
            "DOMINO_TEST_GPU_FREE_MB": "64",
            "DOMINO_TEST_GPU_TOTAL_MB": "8192",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            cpu_fallback_path = Path(temp_dir) / "cpu_fallback.npz"
            with mock.patch.dict(os.environ, low_gpu, clear=False):
                summary = self._train(
                    iterations=1,
                    gpi=2,
                    checkpoint_interval=100,
                    seed=99,
                    device="auto",
                    workers=1,
                    safety_config=self.safety,
                    rl_weights_path=str(cpu_fallback_path),
                    quiet=True,
                )
            self.assertEqual(summary["device"], "cpu")
            self.assertIn("64.0 MiB", summary["device_fallback_reason"])

            low_ram = {
                "DOMINO_TEST_AVAILABLE_RAM_MB": "1",
                "DOMINO_TEST_TOTAL_RAM_MB": "4096",
            }
            rejected_path = Path(temp_dir) / "must_not_exist.npz"
            with mock.patch.dict(os.environ, low_ram, clear=False):
                with self.assertRaises(MemorySafetyError):
                    self._train(
                        iterations=1,
                        gpi=40,
                        checkpoint_interval=100,
                        seed=99,
                        device="cpu",
                        workers=1,
                        safety_config=ParallelSafetyConfig(
                            memory_reserve_mb=0,
                            estimated_worker_mb=1,
                        ),
                        rl_weights_path=str(rejected_path),
                        quiet=True,
                    )
            self.assertFalse(rejected_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
