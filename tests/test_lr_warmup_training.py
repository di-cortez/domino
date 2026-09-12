"""End-to-end tests for the learning-rate warmup inside real RL training."""

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from agents.encoder import DominoEncoder
from agents.rl_nn import PolicyNetwork
from diagnostics.parallel_runner import ParallelSafetyConfig
from training.canonical_run import canonical_run_dir, create_run_config
from training.pipeline import (
    _locked_run_arguments,
    _network_architecture,
    _ppo_config,
    _rl_config,
    _write_forever_active_run,
    parse_args as parse_pipeline_args,
)
from training.rl.cli import (
    WARMUP_SUBPARAMETERS,
    add_optional_rl_arguments,
    parse_args as parse_rl_args,
    training_options_from_args,
)
from training.rl.config import (
    RLExecutionOptions,
    RLResourceOptions,
    RLTrainingOptions,
    resolve_training_options,
)
from training.rl.lr_warmup import WARMUP_TRACE_COLUMNS, warmup_trace_path
from training.rl.resume import (
    RLTrainingConfiguration,
    _validate_resume_configuration,
    resume_state_path,
    run_config_warmup,
)
from training.rl.training_loop import train


RULESET = "double-three"
NOMINAL_LEARNING_RATE = 0.001
# double-three plays about 1.7 learner decisions per game, and PPO needs 256 per
# minibatch before it reports a KL at all. Below roughly 150 games an iteration
# measures no KL, the hold never advances, and a test of the ladder would pass
# without the ladder ever having climbed.
GPI = 200
TOTAL_GAMES = GPI * 10
# A threshold no real KL reaches, so the climbs below are decided by the hold
# and cooldown timing alone. These tests are about the wiring; the threshold's
# calibration against real KL is covered in test_lr_warmup.py.
FAST_WARMUP = {
    "exponent": 3,
    "hold_iterations": 2,
    "cooldown_iterations": 1,
    "kl_threshold": 1.0,
}


def _network():
    encoder = DominoEncoder(RULESET)
    return PolicyNetwork(
        input_size=encoder.vector_size,
        hidden1_size=8,
        hidden2_size=4,
        output_size=encoder.action_size,
        random_seed=7,
        device="cpu",
    )


def _save_supervised(path):
    network = _network()
    np.savez(
        path,
        **{
            name: np.asarray(getattr(network, name))
            for name in network.weight_names
        },
    )


def _train_warmup_run(
    root,
    *,
    warmup_lr=None,
    stop_after=None,
    resume_weights=None,
    resume_state=None,
):
    sl_path = Path(root) / "supervised.npz"
    if not sl_path.exists():
        _save_supervised(sl_path)
    return train(
        RLTrainingOptions(
            ruleset_name=RULESET,
            total_training_games=TOTAL_GAMES,
            gpi=GPI,
            opponent_buckets=("random",),
            baseline=("batch-mean",),
            difficulty_weight=0.0,
            learning_rate=NOMINAL_LEARNING_RATE,
            warmup_lr=warmup_lr,
            seed=1234,
            ppo_max_epochs=2,
        ),
        RLResourceOptions(
            sl_weights_path=sl_path,
            rl_weights_path=Path(root) / "training.npz",
            device="cpu",
            workers=1,
            safety_config=ParallelSafetyConfig(
                memory_reserve_mb=0,
                estimated_worker_mb=1,
                max_worker_rss_mb=1024,
            ),
        ),
        RLExecutionOptions(
            quiet=True,
            checkpoint_interval=1,
            numbered_checkpoints=True,
            fresh_from_sl=resume_weights is None,
            stop_after_training_games=stop_after,
            resume_weights_path=resume_weights,
            resume_state_file=resume_state,
        ),
    )


def _trace(root):
    """Return the warmup trace rows as dicts, keyed by the trace columns."""
    metrics = Path(root) / "training_training_metrics.jsonl"
    path = warmup_trace_path(metrics)
    lines = path.read_text(encoding="utf-8").splitlines()
    return [dict(zip(WARMUP_TRACE_COLUMNS, json.loads(line))) for line in lines[1:]]


def test_warmup_run_starts_on_the_first_rung_and_climbs_to_nominal(tmp_path):
    _train_warmup_run(tmp_path, warmup_lr=FAST_WARMUP)
    rows = _trace(tmp_path)
    assert [row["iteration"] for row in rows] == list(range(1, 11))
    # Every iteration measured a KL; otherwise the hold could not advance.
    assert all(row["max_approx_kl"] is not None for row in rows)
    assert rows[0]["applied_learning_rate"] == pytest.approx(
        NOMINAL_LEARNING_RATE / 1.5 ** 3
    )
    # hold 2, then cooldown 1 + hold 2 twice: climbs at 2, 5 and 8.
    assert [row["iteration"] for row in rows if row["promoted"]] == [2, 5, 8]
    assert [row["exponent"] for row in rows] == [3, 2, 2, 2, 1, 1, 1, 0, 0, 0]
    assert rows[-1]["applied_learning_rate"] == NOMINAL_LEARNING_RATE


def test_trace_labels_each_kl_with_the_rate_that_produced_it(tmp_path):
    _train_warmup_run(tmp_path, warmup_lr=FAST_WARMUP)
    rows = _trace(tmp_path)
    for earlier, later in zip(rows, rows[1:]):
        # The rate installed after one iteration is the rate the next one used.
        assert later["applied_learning_rate"] == pytest.approx(
            earlier["next_learning_rate"]
        )
        if not earlier["promoted"]:
            assert earlier["next_learning_rate"] == pytest.approx(
                earlier["applied_learning_rate"]
            )


def test_warmup_resume_reproduces_promotions_and_weights_exactly(tmp_path):
    full_root = tmp_path / "full"
    split_root = tmp_path / "split"
    full_root.mkdir()
    split_root.mkdir()
    full = _train_warmup_run(full_root, warmup_lr=FAST_WARMUP)
    # Stop after iteration 4, one iteration into the second hold: the resume
    # has to restore streak 1, or the next climb slips from 5 to 6.
    partial = _train_warmup_run(
        split_root, warmup_lr=FAST_WARMUP, stop_after=GPI * 4
    )
    resumed = _train_warmup_run(
        split_root,
        warmup_lr=FAST_WARMUP,
        resume_weights=partial["rl_weights_path"],
        resume_state=resume_state_path(partial["rl_weights_path"]),
    )
    full_rows = _trace(full_root)
    split_rows = _trace(split_root)
    assert [row["iteration"] for row in split_rows] == list(range(1, 11))
    for column in (
        "applied_learning_rate",
        "next_learning_rate",
        "exponent",
        "hold_streak",
        "cooldown_remaining",
        "promoted",
    ):
        assert [row[column] for row in split_rows] == [
            row[column] for row in full_rows
        ], column
    with np.load(full["rl_weights_path"], allow_pickle=False) as left:
        with np.load(resumed["rl_weights_path"], allow_pickle=False) as right:
            assert left.files == right.files
            for name in left.files:
                np.testing.assert_array_equal(left[name], right[name])


def test_run_without_warmup_writes_no_trace(tmp_path):
    _train_warmup_run(tmp_path, warmup_lr=None)
    metrics = tmp_path / "training_training_metrics.jsonl"
    assert metrics.is_file()
    assert not warmup_trace_path(metrics).exists()


def test_warmup_is_rejected_on_the_reinforce_path():
    with pytest.raises(ValueError, match="REINFORCE"):
        resolve_training_options(
            RLTrainingOptions(warmup_lr=True, ppo_max_epochs=1),
            RLResourceOptions(),
            RLExecutionOptions(),
        )


def test_every_generated_subparameter_flag_is_a_declared_option():
    parser_flags = set()
    parser = argparse.ArgumentParser()
    add_optional_rl_arguments(parser)
    for action in parser._actions:
        parser_flags.update(action.option_strings)
    assert {flag for flag, _destination in WARMUP_SUBPARAMETERS} <= parser_flags
    assert "--warmup-lr" in parser_flags


@pytest.mark.parametrize("parse", [parse_rl_args, None])
def test_subparameter_without_the_flag_is_a_command_line_error(parse):
    argv = ["--warmup-hold-iterations", "5"]
    with pytest.raises(SystemExit):
        if parse is None:
            parse_pipeline_args(["forever", "--run-name", "probe", *argv])
        else:
            parse(argv)


def test_warmup_run_names_its_bundle():
    args = parse_pipeline_args(["forever", "--run-name", "probe", "--warmup-lr"])
    assert args.bundle_suffix == "warmup_true"


def _canonical_forever_run(tmp_path, *extra):
    """Create a canonical forever run config from ``extra`` flags."""
    initial = parse_pipeline_args([
        "forever",
        "--artifact-root",
        str(tmp_path),
        "--seed",
        "7",
        "--run-name",
        "warmup_probe",
        *extra,
    ])
    run_dir = canonical_run_dir(
        tmp_path, "forever", initial.seed, run_name=initial.run_name
    )
    config = create_run_config(
        run_dir,
        root=tmp_path,
        pipeline_level="forever",
        seed=initial.seed,
        target_rl_games=None,
        supervised_weights_path="supervised.npz",
        supervised_weights_sha256="f" * 64,
        ppo_config=_ppo_config(initial),
        rl_config=_rl_config(initial),
        diagnostic_config={"periodic_games": 100},
        run_name=initial.run_name,
        locked_arguments=_locked_run_arguments(initial),
        network_architecture=_network_architecture(initial),
        machine={
            "cpu_model": "Test CPU",
            "logical_cpu_count": 4,
            "ram_total_bytes": 8 * 1024**3,
            "gpu_name": None,
            "vram_total_bytes": None,
            "rl_device": "cpu",
        },
    )
    _write_forever_active_run(tmp_path, run_dir, config)
    (run_dir / "training_state.json").write_text("{}", encoding="utf-8")
    return run_dir, config


def test_canonical_warmup_survives_forever_hydration(tmp_path):
    run_dir, config = _canonical_forever_run(
        tmp_path, "--warmup-lr", "--warmup-hold-iterations", "50"
    )
    # Recorded in locked_arguments, never in the rebuilt rl_config.
    assert config["locked_arguments"]["warmup_lr"] is True
    assert config["locked_arguments"]["warmup_hold_iterations"] == 50
    assert not any(key.startswith("warmup") for key in config["rl_config"])
    assert run_config_warmup(config)["hold_iterations"] == 50

    # A bare resume restores the flag and the overridden sub-parameter.
    resumed = parse_pipeline_args([
        "forever", "--artifact-root", str(tmp_path)
    ])
    assert resumed._selected_run_dir == run_dir
    training, _resources, _execution = training_options_from_args(resumed)
    assert training.warmup_lr == {"hold_iterations": 50}


def test_run_created_before_the_flag_resumes_with_warmup_off(tmp_path):
    _run_dir, config = _canonical_forever_run(tmp_path)
    # Simulate a run_config written before --warmup-lr existed: its locked
    # arguments carry none of the warmup keys at all.
    legacy = dict(config)
    legacy["locked_arguments"] = {
        key: value
        for key, value in config["locked_arguments"].items()
        if not key.startswith("warmup")
    }
    assert run_config_warmup(legacy) is None


def test_checkpoint_without_warmup_field_validates_against_off(tmp_path):
    """A checkpoint written before the flag must not read as a new experiment."""
    _run_dir, config = _canonical_forever_run(tmp_path)
    expected = RLTrainingConfiguration.from_run_config(
        config, total_training_games=0, selected_workers=1, device="cpu"
    )
    saved = expected.to_dict()
    saved.pop("warmup_lr")  # exactly what a v16 checkpoint records
    _validate_resume_configuration(
        {"configuration": saved}, expected, emit_status=lambda _message: None
    )
