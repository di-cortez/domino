"""Contracts for the per-run analysis bundle directory and its discovery.

The bundle used to have one fixed name. Now it carries the run's start date, an
experiment ordinal an operator fills in by hand, and the machine that trained
it. Two things must survive that: resolving a run directory from an artifact
path, which is silent and destructive when wrong, and finding a bundle whose
ordinal has been renamed after the fact.
"""

from __future__ import annotations

import json

import pytest

from training.run_artifacts import (
    BUNDLE_DIR_PATTERN,
    ORDINAL_PLACEHOLDER,
    RUN_COMPACT_DIAGNOSTICS_DIRNAME,
    bundle_dir_name,
    bundle_suffix,
    find_bundle_dir,
    is_bundle_dir_name,
    migrate_bundle_to_named_dir,
    run_compact_diagnostics_dir,
    run_dir_from_compact_diagnostic_path,
)
from utils.machine_identity import MACHINE_REGISTRY, derived_slug, machine_slug


# ----------------------------------------------------------------------
# The name
# ----------------------------------------------------------------------


def test_a_new_bundle_carries_the_ordinal_placeholder():
    name = bundle_dir_name(date="20260903", machine_slug="diego_notebook")
    assert name == f"20260903-{ORDINAL_PLACEHOLDER}_diego_notebook_"
    assert BUNDLE_DIR_PATTERN.match(name)


def test_a_filled_ordinal_is_zero_padded_to_three_digits():
    assert bundle_dir_name(
        date="20260903", machine_slug="rick_desktop", ordinal=7
    ) == "20260903-007_rick_desktop_"


def test_both_naming_schemes_are_recognized_as_bundles():
    assert is_bundle_dir_name(RUN_COMPACT_DIAGNOSTICS_DIRNAME)
    assert is_bundle_dir_name("20260903-XXX_diego_notebook_")
    assert is_bundle_dir_name("20260831-138_rick_desktop_")
    assert not is_bundle_dir_name("checkpoints")
    assert not is_bundle_dir_name("diagnostics")


def test_a_malformed_date_is_refused():
    with pytest.raises(ValueError, match="8 digits"):
        bundle_dir_name(date="2026-09-03", machine_slug="diego_notebook")


# ----------------------------------------------------------------------
# The trap: resolving a run directory from an artifact path
# ----------------------------------------------------------------------


def _run_root(tmp_path):
    (tmp_path / "training_state.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_a_named_bundle_resolves_to_its_run_directory(tmp_path):
    """The failure this replaced was silent: it returned the bundle itself."""
    run_dir = _run_root(tmp_path)
    bundle = run_dir / "20260903-XXX_diego_notebook_"
    bundle.mkdir()
    artifact = bundle / "periodic_diagnostics.jsonl"
    assert run_dir_from_compact_diagnostic_path(artifact) == run_dir


def test_a_legacy_bundle_still_resolves_to_its_run_directory(tmp_path):
    run_dir = _run_root(tmp_path)
    bundle = run_dir / RUN_COMPACT_DIAGNOSTICS_DIRNAME
    bundle.mkdir()
    assert run_dir_from_compact_diagnostic_path(
        bundle / "rl_vs_random_progress.csv"
    ) == run_dir


def test_an_artifact_in_the_run_root_still_resolves(tmp_path):
    """The pre-bundle layout, accepted only because the root proves itself."""
    run_dir = _run_root(tmp_path)
    assert run_dir_from_compact_diagnostic_path(
        run_dir / "periodic_diagnostics.jsonl"
    ) == run_dir


def test_an_unrecognized_directory_raises_instead_of_guessing(tmp_path):
    """A wrong run directory silently corrupts every stored checkpoint path."""
    stray = tmp_path / "somewhere_else"
    stray.mkdir()
    with pytest.raises(ValueError, match="Cannot identify the run directory"):
        run_dir_from_compact_diagnostic_path(stray / "periodic_diagnostics.jsonl")


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------


def test_discovery_finds_a_bundle_whose_ordinal_was_renamed_by_hand(tmp_path):
    """The whole point of matching a pattern instead of storing the name."""
    run_dir = _run_root(tmp_path)
    created = run_dir / bundle_dir_name(
        date="20260903", machine_slug="diego_notebook"
    )
    created.mkdir()
    assert find_bundle_dir(run_dir) == created

    renamed = run_dir / "20260903-138_diego_notebook_"
    created.rename(renamed)
    assert find_bundle_dir(run_dir) == renamed
    assert run_compact_diagnostics_dir(run_dir) == renamed


def test_a_migrated_run_prefers_the_named_bundle_over_the_legacy_one(tmp_path):
    """Migration leaves both on purpose, so this must not read as ambiguity."""
    run_dir = _run_root(tmp_path)
    (run_dir / RUN_COMPACT_DIAGNOSTICS_DIRNAME).mkdir()
    named = run_dir / "20260903-138_diego_notebook_"
    named.mkdir()
    assert find_bundle_dir(run_dir) == named


def test_two_named_bundles_are_refused_rather_than_guessed(tmp_path):
    run_dir = _run_root(tmp_path)
    (run_dir / "20260903-138_diego_notebook_").mkdir()
    (run_dir / "20260904-139_diego_notebook_").mkdir()
    with pytest.raises(ValueError, match="more than one analysis bundle"):
        find_bundle_dir(run_dir)


def test_a_run_with_no_bundle_falls_back_to_the_legacy_name(tmp_path):
    run_dir = _run_root(tmp_path)
    assert find_bundle_dir(run_dir) is None
    assert run_compact_diagnostics_dir(run_dir).name == (
        RUN_COMPACT_DIAGNOSTICS_DIRNAME
    )
    assert run_compact_diagnostics_dir(
        run_dir, default_name="20260903-XXX_diego_notebook_"
    ).name == "20260903-XXX_diego_notebook_"


# ----------------------------------------------------------------------
# Migration
# ----------------------------------------------------------------------


def test_migration_copies_and_leaves_the_source_intact(tmp_path):
    run_dir = _run_root(tmp_path)
    legacy = run_dir / RUN_COMPACT_DIAGNOSTICS_DIRNAME
    legacy.mkdir()
    (legacy / "run_config.json").write_text('{"seed": 42}', encoding="utf-8")
    (legacy / "rl_vs_random_progress.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    name = "20260903-138_diego_notebook_"
    migrated = migrate_bundle_to_named_dir(run_dir, name)

    assert len(migrated) == 2
    assert legacy.is_dir()
    assert json.loads((run_dir / name / "run_config.json").read_text())["seed"] == 42
    # Running it again is a no-op rather than a duplicate or an error.
    assert migrate_bundle_to_named_dir(run_dir, name) == ()


# ----------------------------------------------------------------------
# Machine identity
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("gpu", "cpus", "expected"),
    [
        ("NVIDIA GeForce RTX 3050 6GB Laptop GPU", 20, "diego_notebook"),
        ("NVIDIA GeForce GTX 1650", 20, "rick_desktop"),
        ("NVIDIA GeForce GTX 960M", 8, "rick_notebook-antigo"),
        ("NVIDIA GeForce RTX 4050 Laptop GPU", 16, "rick_notebook-novo"),
    ],
)
def test_every_machine_in_the_corpus_is_recognized(gpu, cpus, expected):
    assert machine_slug(
        {"gpu_name": gpu, "logical_cpu_count": cpus}
    ) == expected


def test_the_registry_names_are_all_valid_directory_fragments():
    for _signature, slug in MACHINE_REGISTRY:
        assert BUNDLE_DIR_PATTERN.match(f"20260903-XXX_{slug}_")


def test_an_unknown_machine_warns_but_still_produces_a_usable_name():
    """A naming convention must never abort an eight-hour training run."""
    metadata = {"gpu_name": "NVIDIA GeForce RTX 9090", "logical_cpu_count": 64}
    with pytest.warns(RuntimeWarning, match="MACHINE_REGISTRY"):
        slug = machine_slug(metadata)
    assert slug == "unknown_rtx-9090-64cpu"
    assert BUNDLE_DIR_PATTERN.match(f"20260903-XXX_{slug}_")


def test_a_caller_that_described_no_machine_does_not_warn(recwarn):
    """Absent metadata is not an unrecognized machine."""
    assert machine_slug({}) == derived_slug({})
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_an_override_wins_and_is_sanitized():
    assert machine_slug(
        {"gpu_name": "NVIDIA GeForce GTX 1650", "logical_cpu_count": 20},
        override="lab machine #2",
    # A run of disallowed characters collapses to a single separator.
    ) == "lab-machine-2"


# ----------------------------------------------------------------------
# The tested parameter in the bundle's tail
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "value", "expected"),
    [
        ("--learning-rate", "0.02", "lr_0p02"),
        ("--learning-rate", "0.005", "lr_0p005"),
        ("--baseline", "batch-mean", "baseline_batch_mean"),
        ("--baseline", "lookup-table", "baseline_lookup_table"),
        ("--reward-distance-mode", "turn-decision", "distance_turn_decision"),
        ("--entropy-coef", "0.1", "entropy_0p1"),
        ("--gpi", "4000", "gpi_4000"),
        ("--opponent-buckets", "heuristic", "bucket_heuristic"),
        # An unlisted flag keeps its own name rather than being dropped.
        ("--difficulty-weight", "0.25", "difficulty_weight_0p25"),
    ],
)
def test_the_tail_names_the_flag_and_its_value(flag, value, expected):
    assert bundle_suffix(flag, value) == expected


def test_neighbouring_decimals_stay_distinguishable():
    """`002` and `02` would collide; `0p02` and `0p2` do not."""
    assert bundle_suffix("--learning-rate", "0.02") != bundle_suffix(
        "--learning-rate", "0.2"
    )


def test_a_suffixed_name_is_still_a_discoverable_bundle(tmp_path):
    run_dir = _run_root(tmp_path)
    name = bundle_dir_name(
        date="20260904",
        machine_slug="diego_notebook",
        suffix=bundle_suffix("--learning-rate", "0.02"),
    )
    assert name == "20260904-XXX_diego_notebook_lr_0p02"
    created = run_dir / name
    created.mkdir()
    assert find_bundle_dir(run_dir) == created
    # And the trap still resolves: the run directory, not the bundle.
    assert run_dir_from_compact_diagnostic_path(
        created / "periodic_diagnostics.jsonl"
    ) == run_dir


def test_the_suffix_survives_filling_in_the_ordinal_by_hand(tmp_path):
    run_dir = _run_root(tmp_path)
    created = run_dir / "20260904-XXX_diego_notebook_baseline_zero"
    created.mkdir()
    renamed = run_dir / "20260904-141_diego_notebook_baseline_zero"
    created.rename(renamed)
    assert find_bundle_dir(run_dir) == renamed


def test_a_value_that_looks_like_a_path_cannot_escape_the_run_directory():
    """The tail becomes a directory name, so separators must not survive it."""
    tail = bundle_suffix("--baseline", "../../etc")
    assert "/" not in tail and ".." not in tail
    assert BUNDLE_DIR_PATTERN.match(
        bundle_dir_name(date="20260904", machine_slug="diego_notebook", suffix=tail)
    )
