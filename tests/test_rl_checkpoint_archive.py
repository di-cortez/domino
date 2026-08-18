"""Bounded historical RL checkpoint-archive contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from agents.rl_nn import PolicyNetwork
from training.rl.checkpoint_archive import (
    ARCHIVE_INTERVAL_ITERATIONS,
    CheckpointArchive,
    _file_sha256,
)
from training.rl.pool import (
    OpponentRecord,
    select_medium_term_staging_records,
)


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


@dataclass(frozen=True)
class _metadata:
    """Archive metadata stand-in for pure band selection over planned writes."""

    completed_iteration: int

    @property
    def checkpoint_id(self):
        return f"checkpoint:{self.completed_iteration:010d}"

    @property
    def opponent_id(self):
        return f"snapshot:{self.completed_iteration:010d}"

    @property
    def completed_rl_games(self):
        return self.completed_iteration * 100


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


def test_pinned_medium_term_records_survive_thinning_and_can_be_unpinned(
    tmp_path,
):
    network = _network()
    probe = CheckpointArchive(tmp_path / "probe")
    first = probe.consider_snapshot(
        network,
        _record(10),
        iteration=10,
        completed_games=1000,
    )
    archive = CheckpointArchive(
        tmp_path / "pinned",
        maximum_bytes=first.file_size * 3,
    )
    for iteration in (10, 20, 30, 40):
        archive.consider_snapshot(
            network,
            _record(iteration),
            iteration=iteration,
            completed_games=iteration * 100,
            pinned=iteration == 20,
        )
    assert archive.lookup("checkpoint:0000000020").pinned
    assert archive.manifest()["pinned_checkpoint_count"] == 1

    archive.set_pinned_checkpoint_ids(())
    for iteration in (50, 60):
        archive.consider_snapshot(
            network,
            _record(iteration),
            iteration=iteration,
            completed_games=iteration * 100,
        )
    assert archive.lookup("checkpoint:0000000020") is None
    counters = archive.manifest()["lifecycle_counters"]
    assert counters["pins"] == 1
    assert counters["unpins"] == 1
    assert counters["thinned"] >= 1


def test_archived_weights_load_only_after_full_manifest_validation(tmp_path):
    archive = CheckpointArchive(tmp_path)
    network = _network()
    record = archive.consider_snapshot(
        network,
        _record(10),
        iteration=10,
        completed_games=1000,
    )
    weights = archive.load_weights(record.checkpoint_id)
    assert tuple(weights) == tuple(record.weight_names)
    for name, shape in zip(record.weight_names, record.weight_shapes):
        expected = getattr(network, name)
        assert weights[name].shape == tuple(shape)
        assert np.array_equal(weights[name], np.asarray(expected))
    # The returned arrays are the caller's; mutating one cannot reach the file.
    first = record.weight_names[0]
    weights[first][...] = 0.0
    assert np.array_equal(
        archive.load_weights(record.checkpoint_id)[first],
        np.asarray(getattr(network, first)),
    )
    with pytest.raises(KeyError, match="not retained"):
        archive.load_weights("checkpoint:0000009999")


def test_archived_weight_loading_rejects_corrupted_and_incompatible_files(
    tmp_path,
):
    archive = CheckpointArchive(tmp_path)
    network = _network()
    record = archive.consider_snapshot(
        network,
        _record(10),
        iteration=10,
        completed_games=1000,
    )
    path = archive.directory / record.filename
    original = path.read_bytes()

    path.write_bytes(original + b"padding")
    with pytest.raises(ValueError, match="file size changed"):
        archive.load_weights(record.checkpoint_id)

    corrupted = bytearray(original)
    corrupted[-1] ^= 0xFF
    path.write_bytes(bytes(corrupted))
    with pytest.raises(ValueError, match="hash changed"):
        archive.load_weights(record.checkpoint_id)

    path.unlink()
    with pytest.raises(ValueError, match="file is missing"):
        archive.load_weights(record.checkpoint_id)


@pytest.mark.parametrize("defect", ("extra", "missing", "shape"))
def test_archived_weight_loading_rejects_wrong_array_inventories(
    tmp_path,
    defect,
):
    archive = CheckpointArchive(tmp_path)
    network = _network()
    record = archive.consider_snapshot(
        network,
        _record(10),
        iteration=10,
        completed_games=1000,
    )
    path = archive.directory / record.filename
    weights = {
        name: np.asarray(getattr(network, name))
        for name in record.weight_names
    }
    if defect == "extra":
        weights["unexpected_weight"] = np.zeros((2, 2), dtype=np.float32)
        expected = "weight names do not match"
    elif defect == "missing":
        weights.pop(record.weight_names[-1])
        expected = "weight names do not match"
    else:
        first = record.weight_names[0]
        weights[first] = np.zeros(
            (weights[first].shape[0] + 1, weights[first].shape[1]),
            dtype=weights[first].dtype,
        )
        expected = "expected"
    np.savez_compressed(path, **weights)
    # Re-stamp the manifest so only the array inventory is under test.
    archive.records = [replace(
        record,
        file_size=path.stat().st_size,
        sha256=_file_sha256(path),
    )]
    with pytest.raises(ValueError, match=expected):
        archive.load_weights(record.checkpoint_id)


def test_retained_records_are_a_deterministic_immutable_view(tmp_path):
    archive = CheckpointArchive(tmp_path)
    network = _network()
    for iteration in (30, 10, 20):
        archive.consider_snapshot(
            network,
            _record(iteration),
            iteration=iteration,
            completed_games=iteration * 100,
        )
    retained = archive.retained_records()
    assert isinstance(retained, tuple)
    assert [record.completed_iteration for record in retained] == [10, 20, 30]
    assert retained == archive.retained_records()


def test_projected_pin_capacity_reports_an_oversized_long_horizon_pin_set(
    tmp_path,
):
    network = _network()
    probe = CheckpointArchive(tmp_path / "probe")
    first = probe.consider_snapshot(
        network,
        _record(10),
        iteration=10,
        completed_games=1000,
    )
    archive = CheckpointArchive(
        tmp_path / "bounded",
        maximum_bytes=first.file_size * 4,
    )
    for iteration in (10, 20, 30):
        archive.consider_snapshot(
            network,
            _record(iteration),
            iteration=iteration,
            completed_games=iteration * 100,
        )
    projection = archive.projected_pin_capacity(3)
    assert projection["representative_file_bytes"] == first.file_size
    assert projection["projected_pinned_bytes"] == first.file_size * 3
    assert projection["fits_within_limit"] is True
    assert archive.projected_pin_capacity(400)["fits_within_limit"] is False
    assert CheckpointArchive(
        tmp_path / "empty",
    ).projected_pin_capacity(200)["fits_within_limit"] is None


def test_an_oversized_long_horizon_pin_set_fails_instead_of_thinning_a_member(
    tmp_path,
):
    network = _network()
    probe = CheckpointArchive(tmp_path / "probe")
    first = probe.consider_snapshot(
        network,
        _record(10),
        iteration=10,
        completed_games=1000,
    )
    archive = CheckpointArchive(
        tmp_path / "tight",
        maximum_bytes=int(first.file_size * 2.5),
    )
    with pytest.raises(RuntimeError, match="exceed the"):
        for iteration in (10, 20, 30):
            archive.consider_snapshot(
                network,
                _record(iteration),
                iteration=iteration,
                completed_games=iteration * 100,
                pinned=True,
            )


def test_medium_term_staging_pins_keep_a_milestone_until_its_admission(
    tmp_path,
):
    network = _network()
    probe = CheckpointArchive(tmp_path / "probe")
    first = probe.consider_snapshot(
        network,
        _record(10),
        iteration=10,
        completed_games=1000,
    )
    written = tuple(range(10, 610, 10))
    maximum_bytes = first.file_size * 25

    def _write(archive, staged):
        for iteration in written:
            archive.consider_snapshot(
                network,
                _record(iteration),
                iteration=iteration,
                completed_games=iteration * 100,
            )
            if not staged:
                continue
            archive.set_pinned_checkpoint_ids(
                record.checkpoint_id
                for record in select_medium_term_staging_records(
                    archive.retained_records(),
                    completed_iteration=iteration,
                )
            )
        return {
            record.completed_iteration
            for record in archive.retained_records()
        }

    # Both archives share one byte limit, so only the pin policy differs.
    expected_staging = {
        record.completed_iteration
        for record in select_medium_term_staging_records(
            tuple(_metadata(iteration) for iteration in written),
            completed_iteration=written[-1],
        )
    }
    unpinned = _write(
        CheckpointArchive(tmp_path / "unpinned", maximum_bytes=maximum_bytes),
        staged=False,
    )
    staged = _write(
        CheckpointArchive(tmp_path / "staged", maximum_bytes=maximum_bytes),
        staged=True,
    )
    assert expected_staging - unpinned
    assert expected_staging <= staged
