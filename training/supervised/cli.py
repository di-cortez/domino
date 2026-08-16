"""Command-line parsing and entry point for supervised training."""

from __future__ import annotations

import argparse

from agents.network_architecture import (
    DEFAULT_HIDDEN_LAYER_COUNT,
    MAX_HIDDEN_LAYER_COUNT,
    default_hidden_size,
    resolve_hidden_sizes,
)
from training.supervised.dataset import DATASET_MEMORY_RESERVE_MB
from training.supervised.runtime import (
    DEFAULT_SUPERVISED_BATCH_SIZE,
    SUPERVISED_BATCH_SIZE_CHOICES,
    SUPERVISED_GPU_MEMORY_RESERVE_MB,
)
from training.supervised.training_loop import (
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_SUPERVISED_LR_DECAY_FACTOR,
    DEFAULT_SUPERVISED_LR_DECAY_PATIENCE,
    train_supervised,
)
from training.utils.cli_args import (
    add_regularization_arguments,
    decay_factor,
    nonnegative_int,
    positive_int,
)
from middleware.rulesets import DEFAULT_RULESET_NAME, RULESET_NAMES


def add_architecture_arguments(group):
    """Add hidden-layer depth and width controls to an argument group."""
    group.add_argument(
        "--hidden-layers",
        type=positive_int,
        choices=range(1, MAX_HIDDEN_LAYER_COUNT + 1),
        default=DEFAULT_HIDDEN_LAYER_COUNT,
        metavar="N",
        help=(
            "Number of hidden policy layers, from 1 to "
            f"{MAX_HIDDEN_LAYER_COUNT} (default: %(default)s)."
        ),
    )
    for position in range(1, MAX_HIDDEN_LAYER_COUNT + 1):
        group.add_argument(
            f"--hidden{position}-size",
            type=positive_int,
            default=argparse.SUPPRESS,
            metavar="N",
            help=(
                f"Neurons in hidden layer {position} "
                "(the first two defaults depend on --ruleset; "
                f"double-six uses {default_hidden_size(position)}). Requires "
                f"--hidden-layers of at least {position}."
            ),
        )
    return group


def resolve_architecture_arguments(args):
    """Resolve hidden-layer arguments in place on an argparse namespace."""
    hidden_sizes = resolve_hidden_sizes(
        args.hidden_layers,
        [
            getattr(args, f"hidden{position}_size", None)
            for position in range(1, MAX_HIDDEN_LAYER_COUNT + 1)
        ],
        maximum=MAX_HIDDEN_LAYER_COUNT,
        ruleset=getattr(args, "ruleset", DEFAULT_RULESET_NAME),
    )
    for position in range(1, MAX_HIDDEN_LAYER_COUNT + 1):
        setattr(
            args,
            f"hidden{position}_size",
            hidden_sizes[position - 1] if position <= len(hidden_sizes) else None,
        )
    return args


def hidden_sizes_from_args(args):
    """Return the resolved hidden widths recorded on a parsed namespace."""
    return tuple(
        getattr(args, f"hidden{position}_size")
        for position in range(1, int(args.hidden_layers) + 1)
    )


def add_optional_training_arguments(parser, *, include_device_alias=False):
    """Add supervised regularization, device, memory, and training controls."""
    group = parser.add_argument_group("supervised-training controls")
    add_regularization_arguments(group)
    group.add_argument(
        "--early-stopping",
        nargs="?",
        type=positive_int,
        const=DEFAULT_EARLY_STOPPING_PATIENCE,
        default=None,
        metavar="PATIENCE",
        help="Stop after this many validation checks without improvement.",
    )
    decay = group.add_mutually_exclusive_group()
    decay.add_argument(
        "--lr-decay",
        nargs="?",
        type=decay_factor,
        const=DEFAULT_SUPERVISED_LR_DECAY_FACTOR,
        default=DEFAULT_SUPERVISED_LR_DECAY_FACTOR,
        metavar="FACTOR",
        help="Plateau LR multiplier; enabled by default.",
    )
    decay.add_argument(
        "--no-lr-decay",
        action="store_const",
        const=None,
        dest="lr_decay",
        help="Disable supervised plateau LR decay.",
    )
    group.add_argument(
        "--lr-decay-patience",
        type=positive_int,
        default=DEFAULT_SUPERVISED_LR_DECAY_PATIENCE,
        help="Failed validation checks required before each LR reduction.",
    )
    group.add_argument(
        "--sl-device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Supervised array backend.",
    )
    if include_device_alias:
        group.add_argument(
            "--device",
            choices=("auto", "cpu", "gpu"),
            dest="sl_device",
            help="Standalone alias for --sl-device.",
        )
    group.add_argument(
        "--sl-batch-size",
        type=positive_int,
        choices=SUPERVISED_BATCH_SIZE_CHOICES,
        default=DEFAULT_SUPERVISED_BATCH_SIZE,
        help="Fixed supervised mini-batch size.",
    )
    add_architecture_arguments(group)
    group.add_argument(
        "--sl-memory-reserve-mb",
        type=nonnegative_int,
        default=DATASET_MEMORY_RESERVE_MB,
        help="Host RAM reserve for supervised data and CPU training.",
    )
    group.add_argument(
        "--sl-gpu-memory-reserve-mb",
        type=nonnegative_int,
        default=SUPERVISED_GPU_MEMORY_RESERVE_MB,
        help="VRAM reserve for supervised GPU training.",
    )
    group.add_argument(
        "--sl-seed",
        type=int,
        default=None,
        help=(
            "Fix the supervised root seed for initialization, epoch shuffles, "
            "and dropout."
        ),
    )
    return parser


def parse_args(argv=None):
    """Parse standalone supervised-training arguments."""
    parser = argparse.ArgumentParser(
        description="Train the supervised-learning domino policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ruleset",
        choices=RULESET_NAMES,
        default=DEFAULT_RULESET_NAME,
        help="Named compact domino ruleset.",
    )
    add_optional_training_arguments(parser, include_device_alias=True)
    args = parser.parse_args(argv)
    try:
        return resolve_architecture_arguments(args)
    except ValueError as exc:
        parser.error(str(exc))
        raise


def main(argv=None):
    """Train the supervised policy from command-line arguments."""
    args = parse_args(argv)
    train_supervised(
        batch_size=args.sl_batch_size,
        hidden_sizes=hidden_sizes_from_args(args),
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        early_stopping_patience=args.early_stopping,
        lr_decay_factor=args.lr_decay,
        lr_decay_patience=args.lr_decay_patience,
        device=args.sl_device,
        memory_reserve_mb=args.sl_memory_reserve_mb,
        gpu_memory_reserve_mb=args.sl_gpu_memory_reserve_mb,
        seed=args.sl_seed,
        ruleset=args.ruleset,
    )


if __name__ == "__main__":
    main()
