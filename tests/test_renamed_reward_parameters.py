"""The rename of gamma/alpha/event_reward_decay must not disturb run identity.

``gamma`` became ``gamma_f``, ``alpha`` became ``reward_eta`` and
``event_reward_decay`` became ``gamma_i``. Nothing about the experiment changed,
so two rules hold: a run created before the rename keeps the exact identity it
already had, and every persisted mapping stays readable in both spellings. These
tests exist because ``rl_config`` and ``locked_arguments`` are immutable keys
compared on every resume, where a silent spelling change costs a whole run.
"""

import argparse

import pytest

from training.canonical_run import (
    RENAMED_PARAMETER_KEYS,
    configuration_sha256,
    identity_spelling,
)
from training.rl.cli import (
    add_optional_rl_arguments,
    training_options_from_args,
)
from training.rl.config import RLTrainingOptions
from training.rl.resume import RLTrainingConfiguration
from training.rl.reward_model import (
    DEFAULT_GAMMA_F,
    DEFAULT_GAMMA_I,
    DEFAULT_REWARD_ETA,
)
from training.rl.rollout import DEFAULT_REWARD_SCHEMA
from training.rl.reward_distance import (
    DEFAULT_REWARD_DISTANCE_MODE,
    HISTORICAL_REWARD_DISTANCE_MODE,
)


_BASE_RL_CONFIG = {
    "games_per_iteration": 100,
    "opponent_buckets": ["heuristic", "recent"],
    "difficulty_weight": 0.5,
    "opponent_decision_restarts": False,
    "learning_rate": 0.001,
    "entropy_coef": 0.01,
    "weight_decay": 0.0,
    "dropout_rate": 0.0,
    "log_interval": 1,
    "checkpoint_interval": 1,
    "use_value_head": False,
    "value_coef": 0.5,
    "clip_grad_norm": 1.0,
    "normalize_advantages": True,
    "moving_average_window": 10,
    "requested_device": "auto",
    "requested_workers": "auto",
    "worker_autotune_fraction": 0.01,
    "worker_autotune_minimum_gain": 0.1,
    "worker_memory_reserve_mb": 512,
    "worker_estimated_mb": 256,
    "worker_max_rss_mb": 1024,
    "terminal_empty_hand_weight": 1.0,
    "terminal_blocked_weight": 1.0,
    "immediate_draw_weight": 1.0,
    "immediate_pass_weight": 1.0,
}
_LEGACY_VALUES = {"gamma": 1.0, "alpha": 0.5, "event_reward_decay": 0.90}
_CURRENT_VALUES = {"gamma_f": 1.0, "reward_eta": 0.5, "gamma_i": 0.90}


def _run_config(values):
    return {
        "config_hash_version": 4,
        "pipeline_level": "forever",
        "run_name": "suit_off",
        "seed": 42,
        "target_rl_games": None,
        "ruleset_version": 1,
        "ruleset_name": "double-six",
        "encoder_size": 161,
        "action_count": 56,
        "network_architecture": [161, 256, 128, 56],
        "algorithm": "ppo_v2_decision_minibatches",
        "supervised_weights_sha256": "0" * 64,
        "ppo_config": {"max_epochs": 4},
        "rl_config": {**_BASE_RL_CONFIG, **values},
        "diagnostic_config": {},
        "locked_arguments": {**values, "seed": 42},
    }


def test_rename_map_covers_exactly_the_three_parameters():
    assert RENAMED_PARAMETER_KEYS == {
        "gamma_f": "gamma",
        "reward_eta": "alpha",
        "gamma_i": "event_reward_decay",
    }


def test_identity_spelling_normalizes_only_the_renamed_keys():
    normalized = identity_spelling({**_CURRENT_VALUES, "seed": 42})

    assert normalized == {**_LEGACY_VALUES, "seed": 42}
    # Already-legacy mappings pass through untouched, so normalizing twice is
    # the same as normalizing once.
    assert identity_spelling(normalized) == normalized


def test_run_identity_is_pinned_against_every_future_change():
    """One fixed configuration must keep hashing to one fixed value, forever.

    ``rl_config`` is both hashed and immutable, so a key added to it renames
    every existing run and makes them unresumable. The numerical-stability
    guards were deliberately kept out of it for this reason: they are module
    constants, reported in the informational manifest only. This pin is what
    makes a regression of that decision fail here rather than on a machine
    twenty million games into a run.

    The pin moved once, when the four reward-component weights joined
    ``rl_config``. That was the intent rather than an accident: the reward
    redesign replaced the objective itself, so a run trained on the old
    terminal reward must not be resumable under the new one. Any other change
    to this value is a bug.
    """
    assert configuration_sha256(_run_config(_CURRENT_VALUES)) == (
        "16da36a4915f74bce4a9b6d1785e920ebb47856cc568c39779cb4fad30aacacb"
    )


def test_run_identity_survives_the_rename():
    """A run created before the rename must keep its ``configuration_sha256``."""
    legacy = configuration_sha256(_run_config(_LEGACY_VALUES))
    current = configuration_sha256(_run_config(_CURRENT_VALUES))

    assert legacy == current


def test_a_real_value_change_still_changes_the_identity():
    """Normalizing spelling must not blunt the hash against actual edits."""
    changed = dict(_CURRENT_VALUES, gamma_f=0.99)

    assert configuration_sha256(_run_config(changed)) != configuration_sha256(
        _run_config(_CURRENT_VALUES)
    )


@pytest.mark.parametrize(
    ("values", "label"),
    [(_LEGACY_VALUES, "legacy"), (_CURRENT_VALUES, "current")],
)
def test_run_config_reads_either_spelling(values, label):
    run_config = _run_config(values)
    run_config["configuration_sha256"] = configuration_sha256(run_config)

    configuration = RLTrainingConfiguration.from_run_config(
        run_config,
        total_training_games=1000,
        selected_workers=2,
        device="cpu",
    )

    assert configuration.gamma_f == 1.0, label
    assert configuration.reward_eta == 0.5, label
    assert configuration.gamma_i == 0.90, label
    assert configuration.reward_distance_mode == (
        HISTORICAL_REWARD_DISTANCE_MODE
    ), label


def test_new_run_config_persists_its_explicit_distance_mode():
    run_config = _run_config({
        **_CURRENT_VALUES,
        "reward_distance_mode": "decision-turn",
    })
    run_config["configuration_sha256"] = configuration_sha256(run_config)

    configuration = RLTrainingConfiguration.from_run_config(
        run_config,
        total_training_games=1000,
        selected_workers=2,
        device="cpu",
    )
    restored = RLTrainingConfiguration.from_mapping(configuration.to_dict())

    assert configuration.reward_distance_mode == "decision-turn"
    assert restored.reward_distance_mode == "decision-turn"


def test_checkpoint_metadata_reads_the_legacy_spelling():
    """Checkpoints written before the rename must still resume."""
    run_config = _run_config(_CURRENT_VALUES)
    run_config["configuration_sha256"] = configuration_sha256(run_config)
    current = RLTrainingConfiguration.from_run_config(
        run_config,
        total_training_games=1000,
        selected_workers=2,
        device="cpu",
    )

    legacy_metadata = dict(current.to_dict(), **_LEGACY_VALUES)
    for name in _CURRENT_VALUES:
        legacy_metadata.pop(name)
    legacy_metadata.pop("reward_distance_mode")

    restored = RLTrainingConfiguration.from_mapping(legacy_metadata)

    assert restored == current


def test_cli_exposes_the_renamed_flags():
    parser = argparse.ArgumentParser()
    add_optional_rl_arguments(parser)

    args = parser.parse_args(
        [
            "--gamma-f", "0.9",
            "--reward-eta", "0.25",
            "--gamma-i", "0.5",
            "--reward-distance-mode", "decision-turn",
        ]
    )
    training, _resources, _execution = training_options_from_args(args)

    assert (args.gamma_f, args.reward_eta, args.gamma_i) == (0.9, 0.25, 0.5)
    assert training.gamma_f == 0.9
    assert training.reward_eta == 0.25
    assert training.gamma_i == 0.5
    assert training.reward_distance_mode == "decision-turn"


@pytest.mark.parametrize("flag", ["--gamma", "--alpha", "--event-reward-decay"])
def test_the_previous_flag_spellings_are_gone(flag):
    parser = argparse.ArgumentParser(exit_on_error=False)
    add_optional_rl_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args([flag, "0.5"])


def test_defaults_and_schema_keys_follow_the_new_names():
    training = RLTrainingOptions()

    assert training.gamma_f == DEFAULT_GAMMA_F
    assert training.reward_eta == DEFAULT_REWARD_ETA
    assert training.gamma_i == DEFAULT_GAMMA_I
    assert training.reward_distance_mode == DEFAULT_REWARD_DISTANCE_MODE
    assert DEFAULT_GAMMA_F == 0.95
    assert DEFAULT_REWARD_SCHEMA["reward_eta"] == DEFAULT_REWARD_ETA
    assert DEFAULT_REWARD_SCHEMA["gamma_i"] == DEFAULT_GAMMA_I
    assert DEFAULT_REWARD_SCHEMA["reward_distance_mode"] == (
        DEFAULT_REWARD_DISTANCE_MODE
    )
    assert "alpha" not in DEFAULT_REWARD_SCHEMA
    assert "event_decay" not in DEFAULT_REWARD_SCHEMA
