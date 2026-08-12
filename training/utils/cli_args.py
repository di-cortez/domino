"""Argument validators and cross-stage CLI controls for training entry points.

The supervised and RL policies are the same architecture and the RL run starts
from the supervised checkpoint, so the regularization flags are owned by one
stage-independent definition instead of being restated per parser. The value
validators live here for the same reason: every training parser needs them and
:mod:`training.supervised.training_loop` and :mod:`training.rl.cli` previously
kept separate copies of the same positive-integer check.
"""

from __future__ import annotations

import argparse

from agents.nn import DISABLED_DROPOUT_RATE, DISABLED_WEIGHT_DECAY


DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_DROPOUT_RATE = 0.1


def nonnegative_float(value):
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def decay_factor(value):
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be greater than 0 and less than 1")
    return parsed


def dropout_rate(value):
    parsed = float(value)
    if not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError(
            "value must be at least 0 and less than 1"
        )
    return parsed


def add_regularization_arguments(group):
    """Add the regularization controls shared by supervised and RL training.

    One flag owns both stages for each regularizer: the supervised policy and
    the RL policy are the same architecture and the RL run starts from the
    supervised checkpoint, so a single coefficient keeps the two stages
    comparable. Both are opt-in and disabled by default; each accepts an
    explicit coefficient and falls back to its default when passed bare.
    """
    group.add_argument(
        "--weight-decay",
        nargs="?",
        type=nonnegative_float,
        const=DEFAULT_WEIGHT_DECAY,
        default=DISABLED_WEIGHT_DECAY,
        metavar="COEFFICIENT",
        help=(
            "Enable L2 weight decay on the weight matrices of both the "
            "supervised and the RL network; biases are never decayed. Pass a "
            f"coefficient or omit it for {DEFAULT_WEIGHT_DECAY}."
        ),
    )
    group.add_argument(
        "--dropout",
        nargs="?",
        type=dropout_rate,
        const=DEFAULT_DROPOUT_RATE,
        default=DISABLED_DROPOUT_RATE,
        metavar="RATE",
        help=(
            "Enable inverted dropout on every hidden layer of the supervised "
            "and RL networks during training updates only. Pass a rate or "
            f"omit it for {DEFAULT_DROPOUT_RATE}."
        ),
    )
    return group
