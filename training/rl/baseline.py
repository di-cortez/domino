"""Policy-gradient baselines: the term subtracted from each decision return.

A baseline ``b`` is the only value subtracted from the return before the
advantage reaches the policy gradient::

    advantage = return - b

Every choice of ``b`` here is unbiased in expectation, so selecting one is a
pure variance-reduction experiment. The six kinds are:

``zero``
    ``b = 0``. The raw REINFORCE signal, kept as the reference point.
``constant``
    ``b = c`` for one fixed ``c``. Spelled as a bare number on the command
    line: ``--baseline 2``. The number is the constant's value and never an
    index into this list, so ``--baseline 2`` is ``b = 2``, not ``batch-mean``.
``batch-mean``
    ``b = mean(returns)`` over the complete on-policy iteration buffer.
``value-head``
    ``b = V(s)`` from a linear critic reading the policy's last hidden
    activation. The critic's loss is backpropagated into that shared trunk.
``value-head-no-up``
    The same shared critic, but its loss stops at the critic: only ``Wv`` and
    ``bv`` are updated, and the policy trunk never feels it.
``value-head-own-nn``
    ``b = V(s)`` from a critic with its own network, sharing no weights with
    the policy at all.

The last three subtract exactly the same quantity and differ only in how the
critic is wired to the policy, which is the point of having all three: together
they separate the critic's value *as a baseline* from its effect *on the shared
representation*.

======================  ================  =========================
Kind                    Shared trunk      Critic shapes the trunk
======================  ================  =========================
``value-head``          yes               yes
``value-head-no-up``    yes               no
``value-head-own-nn``   no                no (there is no shared trunk)
======================  ================  =========================

Centering belongs to the baseline alone. Advantage *normalization* is a
separate knob that only rescales to unit standard deviation, because a
normalization step that also subtracted the batch mean would silently reimpose
``batch-mean`` on top of whichever baseline was requested and make ``zero`` and
``constant`` unobservable.

WHAT ``batch-mean`` CHANGES, AND WHAT IT DOES NOT

``batch-mean`` is not a new estimator. It is the name for the baseline this
project has always used without calling it one, derived in
``references/explicacoes/ppo_with_out_critic/ppo_sem_critico.tex``: the old
``normalize_advantages`` subtracted the buffer mean and divided by the buffer
standard deviation in one step, so the advantage handed to the clipped
objective was already ``(R - mu_B) / (sigma_B + eps)``. The mean was a baseline
in every mathematical sense; it simply arrived as a side effect of a
variance-reduction step rather than as a choice.

What changed is that the two operations are now factored apart::

    before:  advantage = (R - mu_B) / (sigma_B + eps)     # one inseparable step
    after:   advantage = (R - b)    / (sigma   + eps)     # b chosen, then scaled

So ``--baseline batch-mean`` with normalization on is bit-for-bit the previous
default, and the denominator does not move for three of the four kinds: ``zero``,
``constant`` and ``batch-mean`` all subtract the same value from every decision,
and a standard deviation is invariant under a constant shift, so ``sigma(R - b)``
equals ``sigma(R)`` exactly. Only ``value-head`` changes the scale as well,
because ``V(s)`` differs per decision.

Two behaviors that were previously unreachable now are. Turning normalization
off used to remove the centering with it, leaving the raw return; ``--baseline
batch-mean --no-normalize-advantages`` now gives the centered but unscaled
``R - mu_B``. And a zero-variance iteration used to collapse to all zeros
because the mean was always removed; it now keeps whatever offset the selected
baseline implies, which is still zero for ``batch-mean`` but not for
``constant``.

The critic head and the baseline are independent: ``--value-head`` with a
non-critic baseline still trains ``V(s)`` through the value loss, it just does
not subtract it. That combination is what isolates the cost of training the
critic from the effect of using it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np


ZERO = "zero"
CONSTANT = "constant"
BATCH_MEAN = "batch-mean"
VALUE_HEAD = "value-head"
VALUE_HEAD_OWN_NN = "value-head-own-nn"
VALUE_HEAD_NO_UP = "value-head-no-up"

BASELINE_KINDS = (
    ZERO,
    CONSTANT,
    BATCH_MEAN,
    VALUE_HEAD,
    VALUE_HEAD_OWN_NN,
    VALUE_HEAD_NO_UP,
)

# The three kinds that read a critic's predictions. They differ only in how the
# critic is wired to the policy, never in what is subtracted, so every consumer
# that only needs "does this baseline need V(s)" tests this set.
CRITIC_KINDS = (VALUE_HEAD, VALUE_HEAD_OWN_NN, VALUE_HEAD_NO_UP)

# Kinds that carry a numeric argument on the command line.
_KINDS_WITH_VALUE = (CONSTANT,)

# Accepted spellings for each kind, so ``batch_mean`` and ``batch-mean`` are
# the same request whether they arrive from argparse or from a saved run.
# ``constant`` stays readable even though the command line now spells it as a
# bare number: ``as_tokens`` has always emitted ``["constant", "2.0"]`` into
# ``locked_arguments`` and checkpoints, and dropping the spelling would make
# every run already created with a constant unresumable.
_KIND_ALIASES = {
    "zero": ZERO,
    "none": ZERO,
    "constant": CONSTANT,
    "const": CONSTANT,
    "batch-mean": BATCH_MEAN,
    "batch_mean": BATCH_MEAN,
    "mean": BATCH_MEAN,
    "value-head": VALUE_HEAD,
    "value_head": VALUE_HEAD,
    "critic": VALUE_HEAD,
    "value-head-own-nn": VALUE_HEAD_OWN_NN,
    "value_head_own_nn": VALUE_HEAD_OWN_NN,
    "value-head-no-up": VALUE_HEAD_NO_UP,
    "value_head_no_up": VALUE_HEAD_NO_UP,
}


def canonical_kind(value):
    """Return the canonical spelling for one baseline kind."""
    kind = str(value).strip().lower()
    try:
        return _KIND_ALIASES[kind]
    except KeyError:
        raise ValueError(
            f"Unknown baseline {value!r}; choose one of "
            + ", ".join(BASELINE_KINDS)
            + ", or a bare number for the constant baseline."
        ) from None


def _parsed_constant(token):
    """Return the float one token spells, or ``None`` when it spells a name.

    A bare number on the command line *is* the constant baseline: ``--baseline
    2`` means ``b = 2``. No baseline name is numeric, so trying the number first
    can never shadow one. The number is the constant's value and never an index
    into :data:`BASELINE_KINDS`, which is why ``--baseline 2`` is ``constant(2)``
    rather than the second kind.
    """
    try:
        return float(str(token).strip())
    except ValueError:
        return None


@dataclass(frozen=True)
class BaselineSpec:
    """One fully resolved baseline choice."""

    kind: str
    constant: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "kind", canonical_kind(self.kind))
        constant = float(self.constant)
        if not np.isfinite(constant):
            raise ValueError("A constant baseline must be a finite number.")
        if self.kind != CONSTANT and constant != 0.0:
            raise ValueError(
                f"The {self.kind} baseline takes no value, but "
                f"{constant!r} was supplied."
            )
        object.__setattr__(self, "constant", constant)

    @property
    def requires_value_head(self):
        """Return whether this baseline reads the critic's predictions."""
        return self.kind in CRITIC_KINDS

    @property
    def critic_wiring(self):
        """Return how this baseline's critic is attached to the policy.

        ``None`` for a baseline that needs no critic. The three critic kinds
        subtract exactly the same thing -- one ``V(s)`` per decision -- and
        differ only here, which is what the experiment is measuring:

        ``value-head``
            One linear head over the policy's last hidden activation, and its
            loss is backpropagated into that shared trunk.
        ``value-head-no-up``
            The same shared head, but its loss stops at the head: only ``Wv``
            and ``bv`` are updated and the trunk never feels the critic.
        ``value-head-own-nn``
            A separate network with no weights in common with the policy, so
            there is no shared trunk for the loss to reach.
        """
        return self.kind if self.kind in CRITIC_KINDS else None

    @property
    def critic_updates_trunk(self):
        """Return whether the critic's loss reaches the policy's hidden stack."""
        return self.kind == VALUE_HEAD

    @property
    def critic_owns_network(self):
        """Return whether the critic is a separate network from the policy."""
        return self.kind == VALUE_HEAD_OWN_NN

    @property
    def label(self):
        """Return the short human-readable name used in logs and summaries."""
        if self.kind == CONSTANT:
            return f"{CONSTANT}({self.constant:g})"
        return self.kind

    def as_mapping(self):
        """Return the JSON-safe form persisted in run and checkpoint files."""
        return {"kind": self.kind, "constant": float(self.constant)}

    def as_tokens(self):
        """Return the command-line spelling that reproduces this baseline."""
        if self.kind == CONSTANT:
            return [CONSTANT, repr(float(self.constant))]
        return [self.kind]

    @classmethod
    def from_mapping(cls, value):
        """Rebuild one baseline from its persisted mapping."""
        if value is None:
            return None
        if isinstance(value, BaselineSpec):
            return value
        if isinstance(value, (list, tuple)):
            return from_tokens(value)
        if isinstance(value, str):
            return from_tokens([value])
        return cls(
            kind=value["kind"],
            constant=float(value.get("constant", 0.0)),
        )


def from_tokens(tokens):
    """Build one baseline from ``--baseline`` command-line tokens.

    ``["2"]`` is the constant baseline with ``b = 2``; every other kind is
    named, as in ``["zero"]``, ``["batch-mean"]`` or ``["value-head-no-up"]``.
    The legacy ``["constant", "2"]`` spelling is still accepted so a saved run
    round-trips. Raises ``ValueError`` with a message suitable for
    ``parser.error`` on anything else.
    """
    tokens = [str(token) for token in tokens]
    if not tokens:
        raise ValueError(
            "--baseline needs a kind: "
            + ", ".join(BASELINE_KINDS)
            + ", or a bare number for the constant baseline."
        )
    # The numeric form is tried first because no kind name parses as a float,
    # so this can never shadow a name, and because it is now the documented way
    # to ask for a constant.
    constant = _parsed_constant(tokens[0])
    if constant is not None:
        if len(tokens) > 1:
            raise ValueError(
                f"--baseline {tokens[0]} is the constant baseline and takes "
                f"nothing after it, but {' '.join(tokens[1:])!r} followed it. "
                "A positional pipeline level must come before --baseline, not "
                "after it."
            )
        if not np.isfinite(constant):
            raise ValueError(
                f"--baseline needs a finite number, not {tokens[0]!r}."
            )
        return BaselineSpec(kind=CONSTANT, constant=constant)
    kind = canonical_kind(tokens[0])
    extra = tokens[1:]
    if kind in _KINDS_WITH_VALUE:
        if len(extra) != 1:
            raise ValueError(
                f"--baseline {kind} needs exactly one value, for example "
                f"'--baseline {kind} 2'."
            )
        try:
            constant = float(extra[0])
        except ValueError:
            raise ValueError(
                f"--baseline {kind} needs a number, not {extra[0]!r}."
            ) from None
        if not np.isfinite(constant):
            raise ValueError(
                f"--baseline {kind} needs a finite number, not {extra[0]!r}."
            )
        return BaselineSpec(kind=kind, constant=constant)
    if extra:
        raise ValueError(
            f"--baseline {kind} takes no value, but {' '.join(extra)!r} "
            "followed it. A positional pipeline level must come before "
            "--baseline, not after it."
        )
    return BaselineSpec(kind=kind)


def from_run_config(run_config):
    """Return the baseline one saved canonical run was created with.

    It is read from ``locked_arguments`` rather than ``rl_config`` because
    ``rl_config`` is rebuilt on every invocation and compared as an immutable
    run key, so a new member there would make every run created before
    ``--baseline`` existed unresumable. ``None`` means the run predates the flag
    or left it unset, which resolves to the baseline it already used.
    """
    locked = (run_config or {}).get("locked_arguments") or {}
    return BaselineSpec.from_mapping(locked.get("baseline"))


def resolve(baseline, *, use_value_head, normalize_advantages):
    """Resolve an unset baseline to the choice the run already implied.

    ``None`` reproduces the behavior that existed before ``--baseline``: the
    critic when its head is on, the batch mean when whole-buffer normalization
    is on, and no baseline at all otherwise.
    """
    spec = BaselineSpec.from_mapping(baseline)
    if spec is None:
        if use_value_head:
            spec = BaselineSpec(kind=VALUE_HEAD)
        elif normalize_advantages:
            spec = BaselineSpec(kind=BATCH_MEAN)
        else:
            spec = BaselineSpec(kind=ZERO)
    if spec.requires_value_head and not use_value_head:
        raise ValueError(
            f"--baseline {spec.kind} needs the critic, so pass --value-head "
            "as well, or choose zero, batch-mean, or a bare number."
        )
    return spec


def baseline_values(spec, returns, *, value_predictions=None, xp=np):
    """Return the per-decision baseline array shaped like ``returns``.

    ``xp`` is the array backend, so the same definition serves the NumPy PPO
    buffer and the on-device single-update REINFORCE path.
    """
    if spec.kind == ZERO:
        return xp.zeros_like(returns)
    if spec.kind == CONSTANT:
        return xp.full_like(returns, spec.constant)
    if spec.kind == BATCH_MEAN:
        mean = float(xp.mean(returns, dtype=xp.float64))
        return xp.full_like(returns, mean)
    # Every critic kind subtracts the same thing: one prediction per decision.
    # How that prediction was produced, and whether its loss reached the policy
    # trunk, is the network's business and not visible here.
    if value_predictions is None:
        raise ValueError(
            f"The {spec.kind} baseline needs one critic prediction per "
            "decision."
        )
    return value_predictions


def subtract(spec, returns, *, value_predictions=None, xp=np):
    """Return ``returns`` minus the baseline, as one advantage array."""
    return returns - baseline_values(
        spec,
        returns,
        value_predictions=value_predictions,
        xp=xp,
    )


class _BaselineAction(argparse.Action):
    """Validate ``--baseline KIND [VALUE]`` while the parser can still error.

    The tokens are stored, not the parsed object, because the canonical
    pipeline persists this destination verbatim as JSON in
    ``locked_arguments`` and a resume assigns it straight back onto the
    namespace. A list of strings survives that round trip; the typed baseline
    is rebuilt from it by :func:`resolve`.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        tokens = [values] if isinstance(values, str) else list(values)
        try:
            from_tokens(tokens)
        except ValueError as error:
            parser.error(str(error))
        setattr(namespace, self.dest, [str(token) for token in tokens])


def add_argument(parser):
    """Declare ``--baseline`` on one RL parser."""
    group = parser.add_argument_group("policy-gradient baseline")
    group.add_argument(
        "--baseline",
        nargs="+",
        action=_BaselineAction,
        default=None,
        metavar="KIND | NUMBER",
        help=(
            "Term subtracted from every return before the policy gradient. "
            "A bare number is the constant baseline and the number is its "
            "value: '--baseline 2' subtracts 2, '--baseline -0.5' subtracts "
            "-0.5. It is never an index into the list of kinds, so "
            "'--baseline 2' is the constant 2 and not the second kind. The "
            "named kinds are 'zero', 'batch-mean', 'value-head' (a linear "
            "critic on the policy's last hidden layer, whose loss also trains "
            "that shared trunk), 'value-head-no-up' (the same critic, but its "
            "loss updates only the critic head and never the trunk), and "
            "'value-head-own-nn' (a critic with its own network, sharing no "
            "weights with the policy). The default follows the rest of the "
            "configuration, choosing value-head with --value-head, batch-mean "
            "when advantage normalization is on, and zero otherwise. Every "
            "value-head kind requires --value-head; every other choice may be "
            "combined with it, which keeps training the critic without "
            "subtracting it. Advantage normalization only rescales, so the "
            "baseline alone decides what is subtracted. Put any positional "
            "argument before this flag."
        ),
    )
    return parser


def validate_arguments(parser, args):
    """Reject a baseline the rest of the invocation cannot supply.

    Only the critic requirement is checked here, because it is the one error
    a parser can report. Whether the run normalizes advantages may still be
    unresolved at parse time, so the full resolution stays in
    :func:`resolve`, called from
    ``training.rl.config.resolve_training_options``.
    """
    spec = BaselineSpec.from_mapping(getattr(args, "baseline", None))
    if spec is not None and spec.requires_value_head and not args.value_head:
        parser.error(
            f"--baseline {spec.kind} needs the critic, so pass --value-head "
            "as well, or choose zero, batch-mean, or a bare number."
        )
    return args
