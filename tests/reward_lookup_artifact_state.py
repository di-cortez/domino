"""Whether the packaged reward lookups match the current artifact format.

The reward redesign replaced the lookup's ``final``/``pips`` terminal pair with
the ``empty_hand``/``blocked`` decomposition, which no version 2 artifact can
express. Rebuilding the four packaged tables needs the fixed policy checkpoints
and the raw neural-versus-heuristic corpus, neither of which is versioned, so
the tests that require a loadable artifact are skipped until the rebuild lands
rather than deleted. Every skip names the rebuild command.
"""

import gzip
import json

from training.rl.reward_lookup_tables.lookup import (
    LOOKUP_FORMAT_VERSION,
    _artifact_paths,
)


PACKAGED_RULESETS = ("double-three", "double-four", "double-five", "double-six")


def packaged_format_version(ruleset_name):
    """Return the format version stored in one packaged lookup artifact."""
    lookup_path, _ = _artifact_paths(ruleset_name)
    with gzip.open(lookup_path, "rt", encoding="utf-8") as stream:
        return int(json.load(stream).get("format_version", -1))


def stale_rulesets():
    """Return the packaged rulesets whose artifact predates the redesign."""
    return {
        ruleset_name
        for ruleset_name in PACKAGED_RULESETS
        if packaged_format_version(ruleset_name) != LOOKUP_FORMAT_VERSION
    }


def stale_artifact_reason():
    """Return the skip reason naming the versions found and the rebuild."""
    stale = stale_rulesets()
    versions = sorted({packaged_format_version(name) for name in stale})
    return (
        f"packaged reward lookups are still format version {versions}, not "
        f"{LOOKUP_FORMAT_VERSION}; rebuild them with "
        "analysis/reward_lookup_table/build_fixed_signed_reward_lookup.py"
    )
