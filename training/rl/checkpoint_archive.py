"""Bounded multi-resolution archive of historical RL policy snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from utils.myrandom import unique_token


ARCHIVE_FORMAT_VERSION = 1
ARCHIVE_POLICY_VERSION = 1
ARCHIVE_INTERVAL_ITERATIONS = 10
CHECKPOINT_ARCHIVE_MAX_BYTES = 1 * 1024**3
ARCHIVE_DENSE_TIER_WIDTH = 8


def archive_policy_manifest():
    """Return all internal archive policy values for audit and exact resume."""
    return {
        "format_version": ARCHIVE_FORMAT_VERSION,
        "policy_version": ARCHIVE_POLICY_VERSION,
        "interval_iterations": ARCHIVE_INTERVAL_ITERATIONS,
        "maximum_bytes": CHECKPOINT_ARCHIVE_MAX_BYTES,
        "thinning": "exponentially_widening_age_tiers",
        "dense_tier_width": ARCHIVE_DENSE_TIER_WIDTH,
        "tier_widths": "dense_tier_width * 2**tier",
        "tier_strides": "2**tier",
        "always_retain": ["baseline", "newest", "pinned"],
    }


@dataclass(frozen=True)
class ArchiveRecord:
    """One manifest entry for a retained historical policy."""

    checkpoint_id: str
    opponent_id: str
    completed_iteration: int
    completed_rl_games: int
    filename: str
    file_size: int
    sha256: str
    created_at: str
    weight_names: tuple[str, ...]
    weight_shapes: tuple[tuple[int, ...], ...]
    pinned: bool = False


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{unique_token(4)}")
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _host_policy_weights(network):
    result = {}
    for name in network.weight_names:
        value = getattr(network, name)
        if hasattr(value, "get"):
            value = value.get()
        result[name] = np.asarray(value).copy()
    return result


class CheckpointArchive:
    """Persist, reconcile, and progressively thin historical policy files."""

    def __init__(self, artifact_directory, *, maximum_bytes=None):
        self.directory = Path(artifact_directory) / "checkpoint_archive"
        self.manifest_path = self.directory / "manifest.json"
        self.maximum_bytes = (
            CHECKPOINT_ARCHIVE_MAX_BYTES
            if maximum_bytes is None
            else int(maximum_bytes)
        )
        if self.maximum_bytes < 1:
            raise ValueError("Checkpoint archive maximum_bytes must be positive")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records, self.abandoned_descendants = self._load_manifest()

    def _policy(self):
        policy = archive_policy_manifest()
        policy["maximum_bytes"] = self.maximum_bytes
        return policy

    def _load_manifest(self):
        if not self.manifest_path.is_file():
            return [], []
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Checkpoint archive manifest cannot be read: {self.manifest_path}"
            ) from exc
        if value.get("format_version") != ARCHIVE_FORMAT_VERSION:
            raise ValueError("Unsupported checkpoint archive format version")
        if value.get("policy") != self._policy():
            raise ValueError("Checkpoint archive policy changed")
        records = [
            ArchiveRecord(
                **{
                    **record,
                    "weight_names": tuple(record["weight_names"]),
                    "weight_shapes": tuple(
                        tuple(shape) for shape in record["weight_shapes"]
                    ),
                }
            )
            for record in value.get("checkpoints", ())
        ]
        for record in records:
            path = self.directory / record.filename
            if (
                not path.is_file()
                or path.stat().st_size != record.file_size
                or _file_sha256(path) != record.sha256
            ):
                raise ValueError(
                    f"Checkpoint archive entry is incomplete: {path}"
                )
        abandoned = [
            ArchiveRecord(
                **{
                    **record,
                    "weight_names": tuple(record["weight_names"]),
                    "weight_shapes": tuple(
                        tuple(shape) for shape in record["weight_shapes"]
                    ),
                }
            )
            for record in value.get("abandoned_descendants", ())
        ]
        return records, abandoned

    def _manifest(self, records=None):
        records = self.records if records is None else records
        return {
            "format_version": ARCHIVE_FORMAT_VERSION,
            "policy": self._policy(),
            "retained_bytes": sum(record.file_size for record in records),
            "checkpoint_count": len(records),
            "checkpoints": [asdict(record) for record in records],
            "abandoned_descendants": [
                asdict(record) for record in self.abandoned_descendants
            ],
        }

    def manifest(self):
        return self._manifest()

    def reconcile(self, completed_iteration):
        """Remove descendants newer than the selected exact-resume point."""
        completed_iteration = int(completed_iteration)
        retained = [
            record
            for record in self.records
            if record.completed_iteration <= completed_iteration
        ]
        removed = [record for record in self.records if record not in retained]
        if not removed:
            return []
        known = {
            record.checkpoint_id: record
            for record in self.abandoned_descendants
        }
        for record in removed:
            previous = known.get(record.checkpoint_id)
            if previous is not None and previous.sha256 != record.sha256:
                raise ValueError(
                    "Checkpoint archive abandoned identity has conflicting hashes"
                )
            known[record.checkpoint_id] = record
        self.abandoned_descendants = sorted(
            known.values(),
            key=lambda item: item.completed_iteration,
        )
        _atomic_json(self.manifest_path, self._manifest(retained))
        for record in removed:
            (self.directory / record.filename).unlink(missing_ok=True)
        self.records = retained
        return removed

    def _save_weights(self, path, weights):
        temporary = path.with_name(
            f".{path.stem}.tmp-{os.getpid()}-{unique_token(4)}.npz"
        )
        try:
            np.savez_compressed(temporary, **weights)
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def consider_snapshot(self, network, record, *, iteration, completed_games):
        """Archive the admitted identity at the internal fixed cadence."""
        iteration = int(iteration)
        if iteration % ARCHIVE_INTERVAL_ITERATIONS:
            return None
        if record is None or record.checkpoint_id is None:
            raise ValueError("An archived snapshot requires a neural checkpoint ID")
        weights = _host_policy_weights(network)
        filename = f"checkpoint_iter{iteration:06d}.npz"
        path = self.directory / filename
        identity_match = next(
            (
                item for item in self.records
                if item.checkpoint_id == record.checkpoint_id
                or item.opponent_id == record.opponent_id
            ),
            None,
        )
        if identity_match is not None and identity_match.completed_iteration != iteration:
            raise ValueError("Checkpoint archive logical identity was reused")
        if path.exists():
            existing = next(
                (
                    item for item in self.records
                    if item.completed_iteration == iteration
                ),
                None,
            )
            if (
                existing is None
                or existing.checkpoint_id != record.checkpoint_id
                or existing.sha256 != _file_sha256(path)
            ):
                raise ValueError(
                    "Checkpoint archive identity/hash conflict at iteration "
                    f"{iteration}"
                )
            return existing
        self._save_weights(path, weights)
        checkpoint_sha256 = _file_sha256(path)
        abandoned = next(
            (
                item for item in self.abandoned_descendants
                if item.checkpoint_id == record.checkpoint_id
                or item.opponent_id == record.opponent_id
            ),
            None,
        )
        if abandoned is not None:
            if (
                abandoned.completed_iteration != iteration
                or abandoned.sha256 != checkpoint_sha256
            ):
                path.unlink(missing_ok=True)
                raise ValueError(
                    "Checkpoint archive recreated identity has a different hash"
                )
            self.abandoned_descendants.remove(abandoned)
        archived = ArchiveRecord(
            checkpoint_id=record.checkpoint_id,
            opponent_id=record.opponent_id,
            completed_iteration=iteration,
            completed_rl_games=int(completed_games),
            filename=filename,
            file_size=path.stat().st_size,
            sha256=checkpoint_sha256,
            created_at=datetime.now(timezone.utc).isoformat(),
            weight_names=tuple(weights),
            weight_shapes=tuple(tuple(value.shape) for value in weights.values()),
        )
        self.records.append(archived)
        self.records.sort(key=lambda item: item.completed_iteration)
        self._prune()
        return archived

    @staticmethod
    def _tier_for_age(age):
        tier = 0
        remaining = int(age)
        width = ARCHIVE_DENSE_TIER_WIDTH
        while remaining >= width:
            remaining -= width
            tier += 1
            width *= 2
        return tier

    def _selected_for_thinning(self, level):
        if len(self.records) <= 2:
            return list(self.records)
        newest_index = len(self.records) - 1
        retained = []
        for index, record in enumerate(self.records):
            if index in (0, newest_index) or record.pinned:
                retained.append(record)
                continue
            age = newest_index - index
            tier = self._tier_for_age(age)
            stride = 2 ** (tier + int(level))
            if index % stride == 0:
                retained.append(record)
        return retained

    def _prune(self):
        retained_bytes = sum(record.file_size for record in self.records)
        if retained_bytes <= self.maximum_bytes:
            _atomic_json(self.manifest_path, self._manifest())
            return
        pinned_bytes = sum(
            record.file_size
            for index, record in enumerate(self.records)
            if record.pinned or index in (0, len(self.records) - 1)
        )
        if pinned_bytes > self.maximum_bytes:
            raise RuntimeError(
                "Pinned/baseline/newest checkpoint archive entries exceed the "
                "internal byte limit"
            )
        level = 0
        retained = self._selected_for_thinning(level)
        while sum(record.file_size for record in retained) > self.maximum_bytes:
            level += 1
            retained = self._selected_for_thinning(level)
            if level > math.ceil(math.log2(max(2, len(self.records)))) + 2:
                raise RuntimeError("Checkpoint archive cannot satisfy its byte limit")
        removed = [record for record in self.records if record not in retained]
        # Publish the replacement manifest first. A crash before deletion leaves
        # harmless orphans rather than a manifest that references missing files.
        _atomic_json(self.manifest_path, self._manifest(retained))
        for record in removed:
            (self.directory / record.filename).unlink(missing_ok=True)
        self.records = retained

    def lookup(self, checkpoint_id):
        for record in self.records:
            if record.checkpoint_id == checkpoint_id:
                return record
        return None
