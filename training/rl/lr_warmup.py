"""KL-gated learning-rate warmup for the RL trainer.

The rate starts at ``LR_target / factor ** exponent`` and climbs one rung at a
time, with the exponent walking down to zero, so the last rung is the nominal
rate exactly rather than an accumulated product that could drift past it. A rung
is climbed only after ``EMA(max_approx_kl)`` has stayed below the threshold for
``hold_iterations`` consecutive iterations, and every climb is followed by a
``cooldown_iterations`` block in which no further climb may happen.

Three layers live here and are deliberately kept apart:

1. ``WarmupSchedule`` -- the decision, a pure function of the KL series. No
   engine, no network, no I/O, so it can be replayed offline against any
   finished run before it is trusted with GPU time.
2. ``normalize_warmup`` -- the durable spelling. A run records ``None`` (off) or
   one JSON mapping, the same shape ``training.rl.baseline`` uses, so one
   configuration stays directly comparable between a checkpoint and a run
   config.
3. The per-iteration sidecar trace, written only while warmup is on.

See references/roteiros/WARMUP_LR_ROADMAP.md for the measurements behind every
default below.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from training.rl.ppo import PPO_STOP_KL

DEFAULT_WARMUP_EXPONENT = 6
DEFAULT_WARMUP_FACTOR = 1.5
# The alpha convention is ``ema = alpha * ema + (1 - alpha) * x``, so alpha is
# the weight kept on history: 0.9 remembers about ten iterations. The other
# reading, ``alpha * x + (1 - alpha) * ema``, keeps 1.1 iterations and filters
# nothing -- measured at a 1.0x noise reduction.
DEFAULT_WARMUP_EMA_ALPHA = 0.9
DEFAULT_WARMUP_HOLD_ITERATIONS = 100
DEFAULT_WARMUP_COOLDOWN_ITERATIONS = 100
# Half of the only KL limit the trainer actually enforces. PPO_TARGET_KL is
# declarative -- nothing compares against it -- so halving that one would gate
# on a number the trainer never honours, and the ladder simulation shows it
# stalling below the target for every LR_target >= 0.01.
DEFAULT_WARMUP_KL_THRESHOLD = PPO_STOP_KL / 2.0

WARMUP_TRACE_FORMAT = "domino_rl_warmup_schedule"
WARMUP_TRACE_VERSION = 1

# The mapping keys, in the order the durable spelling records them.
WARMUP_PARAMETER_KEYS = (
    "exponent",
    "factor",
    "ema_alpha",
    "kl_threshold",
    "hold_iterations",
    "cooldown_iterations",
)
WARMUP_PARAMETER_DEFAULTS = {
    "exponent": DEFAULT_WARMUP_EXPONENT,
    "factor": DEFAULT_WARMUP_FACTOR,
    "ema_alpha": DEFAULT_WARMUP_EMA_ALPHA,
    "kl_threshold": DEFAULT_WARMUP_KL_THRESHOLD,
    "hold_iterations": DEFAULT_WARMUP_HOLD_ITERATIONS,
    "cooldown_iterations": DEFAULT_WARMUP_COOLDOWN_ITERATIONS,
}

# The argparse destination of the flag and of each sub-parameter. A canonical
# run persists its whole namespace in ``locked_arguments``, so these spellings
# are part of the durable format rather than a CLI detail: they live here, where
# both ``training.rl.cli`` and ``training.rl.resume`` can read them without the
# second importing the first back.
WARMUP_FLAG_DESTINATION = "warmup_lr"
WARMUP_ARGUMENT_DESTINATIONS = {
    "exponent": "warmup_exponent",
    "factor": "warmup_factor",
    "ema_alpha": "warmup_ema_alpha",
    "kl_threshold": "warmup_kl_threshold",
    "hold_iterations": "warmup_hold_iterations",
    "cooldown_iterations": "warmup_cooldown_iterations",
}


def warmup_from_arguments(arguments):
    """Return the warmup a parsed or persisted argument set asks for, or ``None``.

    ``arguments`` is a mapping of argparse destinations -- ``vars(namespace)``
    or a run's ``locked_arguments``. Only the sub-parameters actually set reach
    the mapping; the rest are filled from the defaults by ``normalize_warmup``.
    A run created before the flag existed records none of these keys and so
    reads as off, which is exactly what it was.
    """
    arguments = arguments or {}
    if not arguments.get(WARMUP_FLAG_DESTINATION, False):
        return None
    return {
        key: arguments[destination]
        for key, destination in WARMUP_ARGUMENT_DESTINATIONS.items()
        if arguments.get(destination) is not None
    }


def validate_warmup_parameters(
    *,
    exponent,
    factor,
    ema_alpha,
    kl_threshold,
    hold_iterations,
    cooldown_iterations,
):
    """Return the six warmup parameters validated and typed.

    The one place every bound is checked, so the CLI, the resolved options and
    a restored checkpoint cannot accept different ranges.
    """
    exponent = int(exponent)
    factor = float(factor)
    ema_alpha = float(ema_alpha)
    kl_threshold = float(kl_threshold)
    hold_iterations = int(hold_iterations)
    cooldown_iterations = int(cooldown_iterations)
    if exponent < 0:
        raise ValueError("warmup exponent must be non-negative")
    if not math.isfinite(factor) or factor <= 1.0:
        raise ValueError("warmup factor must be greater than one")
    # 1.0 would freeze the filter on its first sample forever.
    if not 0.0 <= ema_alpha < 1.0:
        raise ValueError("warmup ema_alpha must be at least 0 and below 1")
    if not math.isfinite(kl_threshold) or kl_threshold <= 0.0:
        raise ValueError("warmup kl_threshold must be positive")
    if hold_iterations < 1:
        raise ValueError("warmup hold_iterations must be at least one")
    if cooldown_iterations < 0:
        raise ValueError("warmup cooldown_iterations must be non-negative")
    return {
        "exponent": exponent,
        "factor": factor,
        "ema_alpha": ema_alpha,
        "kl_threshold": kl_threshold,
        "hold_iterations": hold_iterations,
        "cooldown_iterations": cooldown_iterations,
    }


def normalize_warmup(value):
    """Return the durable warmup spelling: ``None`` or one validated mapping.

    ``None`` and ``False`` both mean off, so a run that predates the flag, a
    run that left it unset, and a run that disabled it explicitly all compare
    equal. ``True`` means on with every default. A mapping keeps whatever it
    names and fills the rest from the defaults.
    """
    if value is None or value is False:
        return None
    if value is True:
        return validate_warmup_parameters(**WARMUP_PARAMETER_DEFAULTS)
    if not isinstance(value, dict):
        raise ValueError(
            f"warmup must be None, a boolean, or a mapping, got {value!r}"
        )
    unknown = sorted(set(value) - set(WARMUP_PARAMETER_KEYS))
    if unknown:
        raise ValueError(
            "Unknown warmup parameter(s): " + ", ".join(unknown)
        )
    return validate_warmup_parameters(**{
        **WARMUP_PARAMETER_DEFAULTS,
        **value,
    })


@dataclass
class WarmupDecision:
    """What one finished iteration did to the schedule.

    The two rates are deliberately separate. ``applied_learning_rate`` is what
    the iteration that produced this KL actually trained with, and is the only
    one a trace row may record: labelling a KL with the rate that follows it
    would corrupt exactly the analysis the trace exists for.
    ``next_learning_rate`` is what the caller installs before the next
    iteration. They differ only on a promotion.
    """

    applied_learning_rate: float
    next_learning_rate: float
    exponent: int
    active: bool
    ema_max_kl: float | None
    hold_streak: int
    cooldown_remaining: int
    promoted: bool


@dataclass
class WarmupSchedule:
    """KL-gated ladder from ``nominal / factor ** exponent`` up to ``nominal``.

    The ladder only ever climbs. A reactive demotion would make this a
    closed-loop controller whose stability is a separate question; the flag
    exists to remove the early-training KL spike, and climbing alone does that.
    """

    nominal_learning_rate: float
    factor: float = DEFAULT_WARMUP_FACTOR
    exponent: int = DEFAULT_WARMUP_EXPONENT
    ema_alpha: float = DEFAULT_WARMUP_EMA_ALPHA
    kl_threshold: float = DEFAULT_WARMUP_KL_THRESHOLD
    hold_iterations: int = DEFAULT_WARMUP_HOLD_ITERATIONS
    cooldown_iterations: int = DEFAULT_WARMUP_COOLDOWN_ITERATIONS
    _ema: float | None = None
    _streak: int = 0
    _cooldown: int = 0

    def __post_init__(self):
        if not self.nominal_learning_rate > 0.0:
            raise ValueError("nominal_learning_rate must be positive")
        validated = validate_warmup_parameters(
            exponent=self.exponent,
            factor=self.factor,
            ema_alpha=self.ema_alpha,
            kl_threshold=self.kl_threshold,
            hold_iterations=self.hold_iterations,
            cooldown_iterations=self.cooldown_iterations,
        )
        for name, value in validated.items():
            setattr(self, name, value)

    @classmethod
    def from_warmup(cls, nominal_learning_rate, warmup):
        """Build a fresh schedule from a normalized warmup mapping."""
        warmup = normalize_warmup(warmup)
        if warmup is None:
            raise ValueError("from_warmup needs warmup enabled, got None")
        return cls(float(nominal_learning_rate), **warmup)

    @property
    def learning_rate(self):
        """The rate this iteration must train with."""
        return self.nominal_learning_rate / (self.factor ** self.exponent)

    @property
    def active(self):
        """Whether the ladder still has somewhere to climb."""
        return self.exponent > 0

    def observe(self, max_approx_kl):
        """Feed one finished iteration's ``max_approx_kl`` and decide.

        ``None`` is accepted and leaves the filter untouched: an iteration with
        no PPO update produces no KL, and inventing one would move the EMA on a
        measurement that was never taken. It still counts against the hold,
        because the streak demands consecutive evidence, not merely no bad news.
        """
        applied = self.learning_rate
        if max_approx_kl is not None:
            value = float(max_approx_kl)
            # Seeded with the first observation rather than with zero: a zero
            # seed invents a low KL that was never measured and shortens the
            # first hold by pulling the filter under the threshold from below.
            self._ema = (
                value
                if self._ema is None
                else self.ema_alpha * self._ema
                + (1.0 - self.ema_alpha) * value
            )
        if not self.active:
            # The filter keeps moving after the last climb even though nothing
            # reads it for a decision any more: a trace that froze the EMA at
            # its final-rung value would report a stale number beside every
            # later KL.
            return WarmupDecision(
                applied, applied, 0, False, self._ema, 0, 0, False
            )
        if self._cooldown > 0:
            # The EMA above keeps updating through the cooldown on purpose:
            # freezing it would resume monitoring with a value measured at the
            # previous rate. The cooldown blocks promotion, not measurement.
            self._cooldown -= 1
            self._streak = 0
            return WarmupDecision(
                applied, applied, self.exponent, True,
                self._ema, 0, self._cooldown, False,
            )
        if (
            max_approx_kl is not None
            and self._ema is not None
            and self._ema < self.kl_threshold
        ):
            self._streak += 1
        else:
            self._streak = 0
        if self._streak >= self.hold_iterations:
            self.exponent -= 1
            self._streak = 0
            self._cooldown = self.cooldown_iterations
            return WarmupDecision(
                applied, self.learning_rate, self.exponent, self.active,
                self._ema, 0, self._cooldown, True,
            )
        return WarmupDecision(
            applied, applied, self.exponent, True,
            self._ema, self._streak, 0, False,
        )

    def state_dict(self):
        """Serialize exactly what a resume must restore."""
        return {
            "exponent": int(self.exponent),
            "ema": None if self._ema is None else float(self._ema),
            "streak": int(self._streak),
            "cooldown": int(self._cooldown),
        }

    def load_state_dict(self, state):
        """Restore a mid-hold schedule so a resume is reproducible."""
        state = dict(state or {})
        exponent = int(state.get("exponent", self.exponent))
        if exponent < 0:
            raise ValueError("restored warmup exponent must be non-negative")
        self.exponent = exponent
        ema = state.get("ema")
        self._ema = None if ema is None else float(ema)
        self._streak = int(state.get("streak", 0))
        self._cooldown = int(state.get("cooldown", 0))


def warmup_trace_path(metrics_output_path):
    """Return the sidecar trace path beside one run's training metrics.

    A sidecar rather than new columns in ``training_metrics.jsonl``: that file
    is validated column for column on resume, so adding members there would
    make every run already in progress unresumable, and the warmup telemetry
    is meaningless for a run that never warmed up anyway.
    """
    metrics_output_path = Path(metrics_output_path)
    # Derived from the metrics file's own name, not fixed, because standalone
    # runs name that file ``<weights stem>_training_metrics.jsonl`` precisely so
    # two runs can share a directory. A canonical run's ``training_metrics.jsonl``
    # therefore pairs with ``warmup_schedule.jsonl``, and a standalone
    # ``w_training_metrics.jsonl`` with ``w_warmup_schedule.jsonl``.
    name = metrics_output_path.name
    suffix = "training_metrics.jsonl"
    if name.endswith(suffix):
        return metrics_output_path.with_name(
            name[: -len(suffix)] + "warmup_schedule.jsonl"
        )
    return metrics_output_path.with_name(
        metrics_output_path.stem + "_warmup_schedule.jsonl"
    )


WARMUP_TRACE_COLUMNS = (
    "iteration",
    "applied_learning_rate",
    "next_learning_rate",
    "exponent",
    "active",
    "ema_max_kl",
    "hold_streak",
    "cooldown_remaining",
    "promoted",
    "max_approx_kl",
)


def _trace_header(nominal_learning_rate, warmup):
    return {
        "format": WARMUP_TRACE_FORMAT,
        "version": WARMUP_TRACE_VERSION,
        "columns": list(WARMUP_TRACE_COLUMNS),
        "nominal_learning_rate": float(nominal_learning_rate),
        "warmup": normalize_warmup(warmup),
    }


def prepare_warmup_trace(path, start_iteration, *, nominal_learning_rate, warmup):
    """Create or stream-truncate the sidecar trace to the resumed iteration.

    Mirrors ``training.rl.reporting._prepare_metrics_file``: on resume every row
    past ``start_iteration`` is dropped before appending, so an interrupted
    iteration that wrote a trace row but never reached its checkpoint cannot
    leave a duplicate behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _trace_header(nominal_learning_rate, warmup)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}"
    )
    try:
        with open(temporary, "w", encoding="utf-8") as output:
            output.write(json.dumps(header, separators=(",", ":")) + "\n")
            if start_iteration and path.is_file():
                with open(path, encoding="utf-8") as source:
                    existing = json.loads(source.readline())
                    if (
                        existing.get("format") != WARMUP_TRACE_FORMAT
                        or existing.get("version") != WARMUP_TRACE_VERSION
                    ):
                        raise ValueError(
                            f"Unsupported warmup trace format in {path}."
                        )
                    iteration_index = WARMUP_TRACE_COLUMNS.index("iteration")
                    for line in source:
                        line = line.strip()
                        if not line:
                            continue
                        values = json.loads(line)
                        if int(values[iteration_index]) > int(start_iteration):
                            break
                        output.write(
                            json.dumps(values, separators=(",", ":")) + "\n"
                        )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def warmup_trace_row(iteration, decision, max_approx_kl):
    """Return one positional trace row for one finished iteration."""
    return [
        int(iteration),
        float(decision.applied_learning_rate),
        float(decision.next_learning_rate),
        int(decision.exponent),
        bool(decision.active),
        None if decision.ema_max_kl is None else float(decision.ema_max_kl),
        int(decision.hold_streak),
        int(decision.cooldown_remaining),
        bool(decision.promoted),
        None if max_approx_kl is None else float(max_approx_kl),
    ]


def write_warmup_trace_row(stream, row):
    """Append and durably flush one trace row."""
    stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def replay_metrics(path, nominal_learning_rate=None, **kwargs):
    """Replay a finished run's ``max_approx_kl`` series through the schedule.

    The replay reads KL from a run trained at a *fixed* rate, so from the first
    promotion onward the series no longer matches the rate the schedule chose.
    The bias is always pessimistic -- the series comes from the full nominal
    rate while the ladder would sit on a lower rung with a lower KL -- which
    makes this a lower bound on how fast the ladder climbs.
    """
    with open(path, "r", encoding="utf-8") as stream:
        header = json.loads(stream.readline())
        kl_index = header["columns"].index("max_approx_kl")
        if nominal_learning_rate is None:
            nominal_learning_rate = float(
                header["metadata"]["run_configuration"]["rl_config"][
                    "learning_rate"
                ]
            )
        schedule = WarmupSchedule(nominal_learning_rate, **kwargs)
        events = []
        for iteration, line in enumerate(stream, start=1):
            decision = schedule.observe(json.loads(line)[kl_index])
            if decision.promoted:
                events.append((iteration, decision))
            if not decision.active:
                break
    return nominal_learning_rate, events, schedule


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Replay a finished run's max_approx_kl series through the "
            "KL-gated learning-rate warmup and report when each rung would "
            "have been climbed."
        ),
    )
    parser.add_argument("metrics", help="path to a training_metrics.jsonl")
    parser.add_argument("--nominal-learning-rate", type=float, default=None)
    parser.add_argument(
        "--exponent", type=int, default=DEFAULT_WARMUP_EXPONENT
    )
    parser.add_argument("--factor", type=float, default=DEFAULT_WARMUP_FACTOR)
    parser.add_argument(
        "--ema-alpha", type=float, default=DEFAULT_WARMUP_EMA_ALPHA
    )
    parser.add_argument(
        "--kl-threshold", type=float, default=DEFAULT_WARMUP_KL_THRESHOLD
    )
    parser.add_argument(
        "--hold-iterations",
        type=int,
        default=DEFAULT_WARMUP_HOLD_ITERATIONS,
    )
    parser.add_argument(
        "--cooldown-iterations",
        type=int,
        default=DEFAULT_WARMUP_COOLDOWN_ITERATIONS,
    )
    args = parser.parse_args(argv)
    nominal, events, schedule = replay_metrics(
        args.metrics,
        args.nominal_learning_rate,
        factor=args.factor,
        exponent=args.exponent,
        ema_alpha=args.ema_alpha,
        kl_threshold=args.kl_threshold,
        hold_iterations=args.hold_iterations,
        cooldown_iterations=args.cooldown_iterations,
    )
    print(f"nominal learning rate: {nominal}")
    print(
        f"threshold: {args.kl_threshold} | alpha: {args.ema_alpha}"
        f" | hold: {args.hold_iterations}"
        f" | cooldown: {args.cooldown_iterations}"
    )
    for iteration, decision in events:
        print(
            f"  iteration {iteration:>6}: EMA {decision.ema_max_kl:.6f}"
            f" -> exponent {decision.exponent},"
            f" lr {decision.applied_learning_rate:.6g}"
            f" -> {decision.next_learning_rate:.6g}"
        )
    if schedule.active:
        print(
            f"  ladder stopped at exponent {schedule.exponent},"
            f" lr {schedule.learning_rate:.6g}"
        )
    else:
        print("  reached the nominal rate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
