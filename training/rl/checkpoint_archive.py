"""Bounded multi-resolution archive of historical RL policy snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from training.rl.pool import MEDIUM_TERM_INTERVAL_ITERATIONS
from utils.myrandom import unique_token


ARCHIVE_FORMAT_VERSION = 2
ARCHIVE_POLICY_VERSION = 2
ARCHIVE_INTERVAL_ITERATIONS = MEDIUM_TERM_INTERVAL_ITERATIONS
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
        (
            self.records,
            self.abandoned_descendants,
            self.lifecycle_counters,
        ) = self._load_manifest()

    def _policy(self):
        policy = archive_policy_manifest()
        policy["maximum_bytes"] = self.maximum_bytes
        return policy

    def _load_manifest(self):
        if not self.manifest_path.is_file():
            return [], [], {
                "writes": 0,
                "pins": 0,
                "unpins": 0,
                "thinned": 0,
            }
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
        counters = value.get("lifecycle_counters", {})
        expected_counters = {"writes", "pins", "unpins", "thinned"}
        if set(counters) != expected_counters:
            raise ValueError(
                "Checkpoint archive lifecycle counters are incomplete"
            )
        return records, abandoned, {
            name: int(counters[name]) for name in expected_counters
        }

    def _manifest(self, records=None):
        records = self.records if records is None else records
        return {
            "format_version": ARCHIVE_FORMAT_VERSION,
            "policy": self._policy(),
            "retained_bytes": sum(record.file_size for record in records),
            "checkpoint_count": len(records),
            "pinned_checkpoint_count": sum(
                bool(record.pinned) for record in records
            ),
            "lifecycle_counters": dict(self.lifecycle_counters),
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
            if record.pinned:
                self.lifecycle_counters["unpins"] += 1
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

    def consider_snapshot(
        self,
        network,
        record,
        *,
        iteration,
        completed_games,
        pinned=False,
    ):
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
            if bool(existing.pinned) != bool(pinned):
                pinned_ids = {
                    item.checkpoint_id
                    for item in self.records
                    if item.pinned
                }
                if pinned:
                    pinned_ids.add(existing.checkpoint_id)
                else:
                    pinned_ids.discard(existing.checkpoint_id)
                self.set_pinned_checkpoint_ids(pinned_ids)
                return self.lookup(existing.checkpoint_id)
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
            pinned=bool(pinned),
        )
        self.records.append(archived)
        self.records.sort(key=lambda item: item.completed_iteration)
        self.lifecycle_counters["writes"] += 1
        if archived.pinned:
            self.lifecycle_counters["pins"] += 1
        self._prune()
        return archived

    def set_pinned_checkpoint_ids(self, checkpoint_ids):
        """Synchronize archive pins with active long-horizon memberships."""
        requested = set(checkpoint_ids)
        retained_ids = {record.checkpoint_id for record in self.records}
        missing = requested - retained_ids
        if missing:
            raise ValueError(
                "Cannot pin checkpoint(s) absent from the archive: "
                + ", ".join(sorted(missing))
            )
        changed = False
        updated = []
        for record in self.records:
            should_pin = record.checkpoint_id in requested
            if record.pinned != should_pin:
                counter = "pins" if should_pin else "unpins"
                self.lifecycle_counters[counter] += 1
                record = replace(record, pinned=should_pin)
                changed = True
            updated.append(record)
        if changed:
            self.records = updated
            self._prune()
        return changed

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
        self.lifecycle_counters["thinned"] += len(removed)
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

    def retained_records(self):
        """Return an immutable deterministically ordered manifest view.

        Band selection reads this instead of the mutable list so a later
        thinning pass cannot change a membership decision already computed.
        """
        return tuple(sorted(
            self.records,
            key=lambda record: (record.completed_iteration, record.checkpoint_id),
        ))

    def load_weights(self, checkpoint_id):
        """Return validated host copies of one retained archived policy.

        The archive is the only path that rehydrates a long-horizon opponent,
        so every recorded property is re-verified before the weights are
        allowed to reach the shared bank. Bank slots stay out of this API: a
        reusable storage address is not part of an archived identity.
        """
        record = self.lookup(checkpoint_id)
        if record is None:
            raise KeyError(
                f"Checkpoint {checkpoint_id!r} is not retained in the archive"
            )
        path = self.directory / record.filename
        if not path.is_file():
            raise ValueError(f"Checkpoint archive file is missing: {path}")
        if path.stat().st_size != record.file_size:
            raise ValueError(f"Checkpoint archive file size changed: {path}")
        if _file_sha256(path) != record.sha256:
            raise ValueError(f"Checkpoint archive file hash changed: {path}")
        weights = {}
        with np.load(path, allow_pickle=False) as stored:
            if sorted(stored.files) != sorted(record.weight_names):
                raise ValueError(
                    "Checkpoint archive weight names do not match the "
                    f"manifest: {path}"
                )
            for name, shape in zip(record.weight_names, record.weight_shapes):
                # Decompressing one member already yields an independent host
                # array, so the caller owns it without a second copy.
                value = stored[name]
                if value.shape != tuple(shape):
                    raise ValueError(
                        f"Checkpoint archive weight {name!r} has shape "
                        f"{value.shape}, expected {tuple(shape)}: {path}"
                    )
                weights[name] = value
        return weights

    def projected_pin_capacity(self, pinned_checkpoint_count):
        """Report whether a future pin set can fit inside the byte limit.

        Thinning can never delete an active member, so an oversized pin set
        must be visible before it becomes an unsatisfiable byte limit.
        """
        pinned_checkpoint_count = int(pinned_checkpoint_count)
        representative_bytes = (
            max(record.file_size for record in self.records)
            if self.records
            else None
        )
        projected_bytes = (
            None
            if representative_bytes is None
            else representative_bytes * pinned_checkpoint_count
        )
        return {
            "pinned_checkpoint_count": pinned_checkpoint_count,
            "representative_file_bytes": representative_bytes,
            "projected_pinned_bytes": projected_bytes,
            "maximum_bytes": self.maximum_bytes,
            "fits_within_limit": (
                None
                if projected_bytes is None
                else projected_bytes <= self.maximum_bytes
            ),
        }
