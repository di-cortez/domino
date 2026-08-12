"""Encoded-dataset feature contract shared by supervised training and assets.

``ENCODED_FEATURE_VERSION`` is a persisted contract string. It is written into
encoded-dataset metadata by supervised training and compared by the canonical
asset checks, so changing its value invalidates every existing ``.npz`` cache
and canonical supervised artifact. Change it only together with a real change
to the encoded feature layout.
"""

from __future__ import annotations


ENCODED_FEATURE_VERSION = "opponent_suit_presence_float32_v2"
