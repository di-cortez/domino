"""Mergeable scalar distribution summaries for the periodic diagnostic record.

One ``RunningMoments`` describes one scalar population -- the terminal
components ``R_E`` and ``R_B``, the event returns ``G_D`` and ``G_P``, or the
per-decision baseline -- as five numbers that a reader turns back into mean,
standard deviation, minimum and maximum.

Streaming rather than storing the raw values is not an optimization detail, it
is what makes the statistic possible at all: the rollout runs in worker
processes (``training.rl.parallel``), so a population is produced in pieces
that never meet in one address space. Five floats per piece merge in constant
time; a hundred thousand raw floats per worker per iteration would have to
cross the process boundary and be concatenated. ``merge`` is therefore part of
the contract, not a convenience -- the serial and parallel paths must produce
identical summaries.

The variance comes from a sum of squares, which is less numerically stable
than a two-pass computation. With ``float64`` accumulators and the reward
components bounded in roughly ``[-1, 1]`` the error is far below the sampling
noise of the quantities being summarized. If a future reward scale widens that
range by orders of magnitude, replace the internals with Welford's algorithm;
the interface and the merge semantics are already the ones it needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


# Written to the record as ``None`` rather than ``0.0`` when a population is
# empty: zero is a value every one of these statistics can legitimately take,
# so it must not double as "not measured".
EMPTY = None

# The four statistics one population contributes to a periodic record.
STATISTIC_SUFFIXES = ("mean", "max", "min", "std")

# Recorded precision, in significant digits rather than decimal places. Six
# keeps the file readable and diffable while staying far below the sampling
# error of anything summarized here.
#
# Significant digits rather than decimal places because these quantities do not
# share a scale: the reward components live in [-1, 1], while the PPO trust
# region's approximate KL runs from 0.0002 to 0.005. A fixed two decimal places
# would round every KL a run has ever recorded to zero.
RECORDED_SIGNIFICANT_DIGITS = 6


def rounded_statistic(value):
    """Return one recorded float, or ``None`` for an unmeasured statistic."""
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Distribution summaries must be finite.")
    if value == 0.0:
        return 0.0
    exponent = math.floor(math.log10(abs(value)))
    return round(value, RECORDED_SIGNIFICANT_DIGITS - 1 - exponent)


@dataclass
class RunningMoments:
    """Count, sum, sum of squares and extremes for one scalar population."""

    count: int = 0
    total: float = 0.0
    total_squares: float = 0.0
    # Identity elements for ``min``/``max``, so an empty summary merges as a
    # no-op instead of dragging a sentinel zero into the extremes.
    minimum: float = field(default=math.inf)
    maximum: float = field(default=-math.inf)

    def add(self, value):
        """Accumulate one observation."""
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("Distribution observations must be finite.")
        self.count += 1
        self.total += value
        self.total_squares += value * value
        if value < self.minimum:
            self.minimum = value
        if value > self.maximum:
            self.maximum = value

    def merge(self, other):
        """Absorb another partial summary of the same population."""
        if other.count == 0:
            return
        self.count += int(other.count)
        self.total += float(other.total)
        self.total_squares += float(other.total_squares)
        self.minimum = min(self.minimum, float(other.minimum))
        self.maximum = max(self.maximum, float(other.maximum))

    @property
    def mean(self):
        if self.count == 0:
            return EMPTY
        return self.total / self.count

    @property
    def std(self):
        """Population standard deviation, never a negative-variance NaN."""
        if self.count == 0:
            return EMPTY
        variance = self.total_squares / self.count - (self.total / self.count) ** 2
        # Cancellation can push an exactly-zero variance a hair below zero.
        return math.sqrt(max(variance, 0.0))

    def as_dict(self, prefix):
        """Return the four recorded statistics, keyed ``<prefix>_<statistic>``."""
        empty = self.count == 0
        return {
            f"{prefix}_mean": rounded_statistic(self.mean),
            f"{prefix}_max": EMPTY if empty else rounded_statistic(self.maximum),
            f"{prefix}_min": EMPTY if empty else rounded_statistic(self.minimum),
            f"{prefix}_std": rounded_statistic(self.std),
        }

    def to_list(self):
        """Return the transport form used to cross a process boundary.

        The extremes of an empty population are the infinite identity elements
        that make ``merge`` a no-op, and JSON cannot encode those. They travel
        as ``null`` instead, which is also the honest reading of "no value was
        observed", and ``from_list`` restores the identities.
        """
        empty = self.count == 0
        return [
            int(self.count),
            float(self.total),
            float(self.total_squares),
            None if empty else float(self.minimum),
            None if empty else float(self.maximum),
        ]

    @classmethod
    def from_list(cls, values):
        """Rebuild one summary from ``to_list``."""
        count, total, total_squares, minimum, maximum = values
        return cls(
            count=int(count),
            total=float(total),
            total_squares=float(total_squares),
            minimum=math.inf if minimum is None else float(minimum),
            maximum=-math.inf if maximum is None else float(maximum),
        )

    @classmethod
    def from_values(cls, values):
        """Build one summary from an in-memory population."""
        summary = cls()
        for value in values:
            summary.add(value)
        return summary
