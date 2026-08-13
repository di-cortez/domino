"""Bounded historical RL checkpoint-archive contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agents.rl_nn import PolicyNetwork
from training.rl.checkpoint_archive import (
    ARCHIVE_INTERVAL_ITERATIONS,
    CheckpointArchive,
)
from training.rl.pool import OpponentRecord


def _network():
    return PolicyNetwork(
        input_size=8,
        hidden1_size=6,
        hidden2_size=4,
        output_size=5,
        random_seed=7,
        device="cpu",
    )


def _record(iteration):
    suffix = f"{int(iteration):010d}"
    return OpponentRecord(
        opponent_id=f"snapshot:{suffix}",
        kind="policy_snapshot",
        checkpoint_id=f"checkpoint:{suffix}",
        introduced_iteration=int(iteration),
        introduced_at_rl_games=int(iteration) * 100,
        origin="training_update",
    )


def test_archive_uses_fixed_cadence_identity_hash_and_atomic_manifest(tmp_path):
    archive = CheckpointArchive(tmp_path)
    network = _network()
    assert archive.consider_snapshot(
        network,
        _record(1),
        iteration=1,
        completed_games=100,
    ) is None
    record = archive.consider_snapshot(
        network,
        _record(ARCHIVE_INTERVAL_ITERATIONS),
        iteration=ARCHIVE_INTERVAL_ITERATIONS,
        completed_games=1000,
    )
    assert record.opponent_id == f"snapshot:{ARCHIVE_INTERVAL_ITERATIONS:010d}"
    assert record.checkpoint_id == f"checkpoint:{ARCHIVE_INTERVAL_ITERATIONS:010d}"
    assert (archive.directory / record.filename).is_file()
    manifest = archive.manifest()
    assert manifest["checkpoint_count"] == 1
    assert manifest["retained_bytes"] == record.file_size
    assert archive.manifest_path.is_file()

    # Repeating the same identity is idempotent, while changing the identity
    # for an already-published iteration is an explicit conflict.
    assert archive.consider_snapshot(
        network,
        _record(ARCHIVE_INTERVAL_ITERATIONS),
        iteration=ARCHIVE_INTERVAL_ITERATIONS,
        completed_games=1000,
    ) == record
    with pytest.raises(ValueError, match="identity/hash conflict"):
        archive.consider_snapshot(
            network,
            _record(999),
            iteration=ARCHIVE_INTERVAL_ITERATIONS,
            completed_games=1000,
        )


def test_archive_thinning_is_bounded_deterministic_and_preserves_ends(tmp_path):
    network = _network()
    probe = CheckpointArchive(tmp_path / "probe")
    first = probe.consider_snapshot(
        network,
        _record(10),
        iteration=10,
        completed_games=1000,
    )
    byte_limit = first.file_size * 4

    archive = CheckpointArchive(tmp_path / "bounded", maximum_bytes=byte_limit)
    for iteration in range(10, 211, 10):
        network.W1 += 0.0001
        archive.consider_snapshot(
            network,
            _record(iteration),
            iteration=iteration,
            completed_games=iteration * 100,
        )
    manifest = archive.manifest()
    retained_iterations = [
        value["completed_iteration"] for value in manifest["checkpoints"]
    ]
    assert manifest["retained_bytes"] <= byte_limit
    assert retained_iterations[0] == 10
    assert retained_iterations[-1] == 210
    assert len(retained_iterations) < 21
    assert all(
        (archive.directory / value["filename"]).is_file()
        for value in manifest["checkpoints"]
    )

    reloaded = CheckpointArchive(
        tmp_path / "bounded",
        maximum_bytes=byte_limit,
    )
    assert reloaded.manifest() == manifest


def test_archive_reconciliation_removes_abandoned_descendants(tmp_path):
    archive = CheckpointArchive(tmp_path)
    network = _network()
    for iteration in (10, 20, 30):
        record = archive.consider_snapshot(
            network,
            _record(iteration),
            iteration=iteration,
            completed_games=iteration * 100,
        )
        assert Path(archive.directory / record.filename).is_file()
    removed = archive.reconcile(20)
    assert [record.completed_iteration for record in removed] == [30]
    assert [
        value["completed_iteration"]
        for value in archive.manifest()["checkpoints"]
    ] == [10, 20]
    assert not (archive.directory / "checkpoint_iter000030.npz").exists()
    assert len(archive.manifest()["abandoned_descendants"]) == 1

    original = np.asarray(network.W1).copy()
    network.W1 += 0.25
    with pytest.raises(ValueError, match="different hash"):
        archive.consider_snapshot(
            network,
            _record(30),
            iteration=30,
            completed_games=3000,
        )
    network.W1[...] = original
    recreated = archive.consider_snapshot(
        network,
        _record(30),
        iteration=30,
        completed_games=3000,
    )
    assert recreated.sha256 == removed[0].sha256
    assert archive.manifest()["abandoned_descendants"] == []
