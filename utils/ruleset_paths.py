"""Central default artifact paths for named compact domino rulesets."""

from pathlib import Path

from middleware.rulesets import DEFAULT_RULESET_NAME, resolve_ruleset


def _ruleset_infix(ruleset) -> str:
    resolved = resolve_ruleset(ruleset)
    return "" if resolved.name == DEFAULT_RULESET_NAME else f"_{resolved.name}"


def default_sl_weights_path(ruleset=DEFAULT_RULESET_NAME) -> Path:
    """Return the standalone supervised checkpoint path for one ruleset."""
    return Path("models") / f"domino_sl{_ruleset_infix(ruleset)}_weights.npz"


def default_dataset_path(ruleset=DEFAULT_RULESET_NAME) -> Path:
    """Return the standalone supervised JSONL path for one ruleset."""
    return Path("dataset") / (
        f"supervised_dataset{_ruleset_infix(ruleset)}.jsonl"
    )


def default_encoded_dataset_path(ruleset=DEFAULT_RULESET_NAME) -> Path:
    """Return the standalone encoded-cache path for one ruleset."""
    dataset = default_dataset_path(ruleset)
    return dataset.with_name(f"{dataset.stem}_encoded.npz")


def default_rl_weights_path(ruleset=DEFAULT_RULESET_NAME) -> Path:
    """Return the standalone RL checkpoint path for one ruleset."""
    return Path("models") / f"domino_rl{_ruleset_infix(ruleset)}_weights.npz"
