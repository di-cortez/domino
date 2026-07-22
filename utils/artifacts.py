"""Checksummed atomic writes shared by canonical training artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil

import numpy as np


def file_sha256(path, chunk_size=1024 * 1024):
    """Return the SHA-256 digest of one file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(path, suffix=None):
    path = Path(path)
    extension = path.suffix if suffix is None else suffix
    stem = path.name[:-len(path.suffix)] if path.suffix else path.name
    return path.with_name(
        f".{stem}.tmp-{os.getpid()}-{secrets.token_hex(4)}{extension}"
    )


def atomic_write_json(path, value):
    """Write JSON with flush/fsync and publish it by same-directory rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def atomic_write_text(path, text):
    """Atomically publish UTF-8 text after flushing it to stable storage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def atomic_savez(path, *, compressed=False, **arrays):
    """Atomically publish a NumPy archive without changing its array keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path, suffix=".npz")
    try:
        saver = np.savez_compressed if compressed else np.savez
        saver(temporary, **arrays)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def atomic_copy(source, destination):
    """Copy one file and atomically replace the destination after fsync."""
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(destination)
    try:
        with open(source, "rb") as reader, open(temporary, "wb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
