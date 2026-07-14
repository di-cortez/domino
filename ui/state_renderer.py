"""
Small renderer state object used to keep the board pivot stable.

The engine provides the chain in logical order. For a stable animation, the UI
remembers which tile was chosen as the pivot and tries to find it again in later
snapshots.
"""


class StateRenderer:
    def __init__(self):
        self._pivot_key = None

    def _tile_key(self, info):
        """Return the history-based id assigned by ``visual_chain_from_state``."""
        return info["id"]

    def get_pivot_index(self, visual_chain):
        if not visual_chain:
            return None

        if self._pivot_key is not None:
            for index, info in enumerate(visual_chain):
                if self._tile_key(info) == self._pivot_key:
                    return index

        self._pivot_key = self._tile_key(visual_chain[0])
        return 0
