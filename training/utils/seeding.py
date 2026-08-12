"""Process-independent seed derivation shared by every training stage.

Seeds must describe identity, not scheduling: the same base seed and the same
labels always produce the same value regardless of worker count, process, or
Python hash randomization. Dataset generation, supervised training, RL
rollouts, and periodic diagnostics therefore all derive their labeled streams
here rather than each keeping a private hash.
"""

from __future__ import annotations

import hashlib


def stable_seed(base_seed: int, *parts: object) -> int:
    """Return a process-independent 64-bit seed for a labeled operation."""
    digest = hashlib.sha256()
    digest.update(str(int(base_seed)).encode("ascii"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "little", signed=False)
