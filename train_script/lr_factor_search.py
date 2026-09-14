"""Adaptive one-factor search of the RL learning rate in powers of two.

The search starts from the project default ``LR`` and expands to both sides,
alternating directions in the fixed order

    LR, 2 LR, LR / 2, 4 LR, LR / 4, 8 LR, LR / 8

with every other setting on the project defaults and ``--warmup-lr`` on in
every run. Only the learning rate moves.

Each finished run is scored by its sustained final level: the mean win rate
against ``random`` over the last fifth of its active wall-clock horizon (the
last hour of a 5-hour Diego-notebook point, the same window the one-factor
sweep analysis reads). A run is *worse* when that score falls more than the
sweep's reading ruler, 0.22 pp, below the best score measured so far; within
the ruler it is a tie. Each direction keeps its own count of worse runs, and a
direction stops exploring after its second one. The search ends when both
directions have stopped or run out of factors, and the best run tested is the
recommended default.

Every warmup run takes part in the decision; the one run without the warmup
does not. A machine that has not measured the bare project default yet runs it
first as a *control run* (``--control-run``), reported beside the search but
never compared, and the search starts right after it at ``LR_default`` with
the warmup. A machine that already has a finished ``LR_default`` warmup run
names it as the *reference run* (``--reference-run``) instead of repeating it:
that run is scored as the search's ``LR_default`` once its recorded settings
are shown to be exactly the ones the search would have launched.

Every run the search launches has its analysis bundle numbered in launch order
from ``DEFAULT_FIRST_RUN_ORDINAL`` (``20260913-101_<machine>_...``, then 102,
...), through ``--run-ordinal``. A skipped candidate and a reference run are
never launched and take no number, so the numbers stay consecutive.

Everything is decided again from the runs themselves on every call, so the
search resumes wherever it was stopped. The shell side lives in
``_lr_factor_search.bash``; this module owns the decision:

    python -m train_script.lr_factor_search plan   --machine-slug diego_notebook
    python -m train_script.lr_factor_search next   --machine-slug ... --results-dir ...
    python -m train_script.lr_factor_search report --machine-slug ... --results-dir ...

``next`` prints one ``kind label value run_name ordinal`` row -- the runner's
``EXPERIMENT_KIND``, its point, and the bundle number -- or exits with
``DONE_EXIT_CODE`` when the search is over.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from decimal import Decimal
import json
from pathlib import Path
import sys

DEFAULT_FACTORS = (2, 4, 8)
# The one-factor sweep's reading ruler on its sustained final level: the 90th
# percentile of the differences between its 22 replicate pairs.
DEFAULT_MARGIN_PP = 0.22
# The last fifth of the horizon. At the Diego notebook's 5-hour budget that is
# the last hour, the sweep's sustained window, and it scales with each
# machine's time coefficient exactly as the budget does.
DEFAULT_WINDOW_FRACTION = 0.2
MIN_WINDOW_POINTS = 3
WORSE_RUNS_TO_STOP = 2
DONE_EXIT_CODE = 3
RUN_NAME_PREFIX = "lr_search"
# The first bundle number a search hands out; earlier numbers in the shared log
# belong to experiments made before it.
DEFAULT_FIRST_RUN_ORDINAL = 101
CONFIG_FILENAME = "search_config.json"
SUMMARY_JSON = "lr_factor_search_summary.json"
SUMMARY_MARKDOWN = "lr_factor_search_summary.md"

DEFAULT = "default"
UP = "up"
DOWN = "down"


def learning_rate_text(value):
    """Spell a rate in plain decimal notation, never as ``1e-05``."""
    return format(Decimal(repr(float(value))), "f")


@dataclass(frozen=True)
class Candidate:
    """One learning rate the search may test, in its fixed position."""

    order: int
    direction: str
    factor: int
    learning_rate: float

    @property
    def label(self):
        if self.direction == DEFAULT:
            return "lr_x1"
        return f"lr_x{self.factor}" if self.direction == UP else f"lr_d{self.factor}"

    @property
    def description(self):
        if self.direction == DEFAULT:
            return "LR_default"
        operator = "x" if self.direction == UP else "/"
        return f"LR_default {operator} {self.factor}"

    def run_name(self, machine_slug):
        token = learning_rate_text(self.learning_rate).replace(".", "p")
        return f"{RUN_NAME_PREFIX}_{token}_{machine_slug}"

    def runner_value(self):
        """The ``combined`` point spelling the sequence runner decodes."""
        return f"lr={learning_rate_text(self.learning_rate)}+warmup=on"


# The control run: the project default without the warmup, as a `one_factor`
# point, whose bundle the runner names `control`.
CONTROL_LABEL = "default"
CONTROL_DESCRIPTION = "LR_default sem warmup"
CONTROL_RUNNER_KIND = "one_factor"
CONTROL_RUNNER_VALUE = "default"
SEARCH_RUNNER_KIND = "combined"

# Locked arguments that name a run rather than configure its training, so a
# reference run may differ from the search's own spelling in them.
IDENTITY_ARGUMENTS = frozenset({"bundle_suffix", "machine_slug", "run_ordinal"})


def control_run_name(machine_slug):
    return f"{RUN_NAME_PREFIX}_{CONTROL_LABEL}_{machine_slug}"


def candidates(default_learning_rate, factors=DEFAULT_FACTORS):
    """Return every candidate in test order: x1, x2, /2, x4, /4, x8, /8."""
    default_learning_rate = float(default_learning_rate)
    if default_learning_rate <= 0.0:
        raise ValueError("the default learning rate must be positive")
    ordered = [Candidate(1, DEFAULT, 1, default_learning_rate)]
    for factor in factors:
        factor = int(factor)
        if factor < 2:
            raise ValueError("every factor must be at least 2")
        for direction in (UP, DOWN):
            value = (
                default_learning_rate * factor
                if direction == UP
                else default_learning_rate / factor
            )
            # Twelve significant digits hide binary representation noise, so
            # 0.001 * 2 is spelled 0.002 in the run name and on the command line.
            ordered.append(Candidate(
                len(ordered) + 1, direction, factor, float(f"{value:.12g}")
            ))
    return tuple(ordered)


# ---------------------------------------------------------------------------
# Scoring one run
# ---------------------------------------------------------------------------


def sustained_final(history, window_fraction=DEFAULT_WINDOW_FRACTION):
    """Score one run's periodic history.

    ``history`` holds rows with ``progress_elapsed_seconds`` and ``win_rate``
    as ``read_periodic_history`` returns them. The time axis is the progress
    clock -- RL training plus the time diagnostics held it -- which is the
    active wall clock the run's budget is spent on.
    """
    if not 0.0 < float(window_fraction) <= 1.0:
        raise ValueError("window_fraction must be in (0, 1]")
    points = sorted(
        (float(row["progress_elapsed_seconds"]) / 3600.0, 100.0 * float(row["win_rate"]))
        for row in history
    )
    if len(points) < 2:
        raise ValueError("a run needs at least two monitor points to be scored")
    horizon = points[-1][0]
    if horizon <= 0.0:
        raise ValueError("a run's monitor points span no time")
    window_start = horizon * (1.0 - float(window_fraction))
    window = [win for hours, win in points if hours >= window_start]
    if len(window) < MIN_WINDOW_POINTS:
        raise ValueError(
            f"only {len(window)} monitor points fall in the final window; "
            f"at least {MIN_WINDOW_POINTS} are needed"
        )
    area = sum(
        (right[0] - left[0]) * (left[1] + right[1]) / 2.0
        for left, right in zip(points, points[1:])
    )
    return {
        "final_sustained_percent": sum(window) / len(window),
        "auc_percent": area / (horizon - points[0][0]),
        "window_points": len(window),
        "horizon_hours": horizon,
        "last_percent": points[-1][1],
        "best_point_percent": max(win for _hours, win in points),
    }


# ---------------------------------------------------------------------------
# Deciding
# ---------------------------------------------------------------------------


@dataclass
class Evaluation:
    """What the search concluded about one candidate."""

    candidate: Candidate
    verdict: str
    score: float | None = None
    best_before: float | None = None
    delta_pp: float | None = None


@dataclass
class SearchState:
    evaluations: list = field(default_factory=list)
    best: Candidate | None = None
    best_score: float | None = None
    worse_counts: dict = field(default_factory=lambda: {UP: 0, DOWN: 0})
    stopped: dict = field(default_factory=lambda: {UP: False, DOWN: False})
    next_candidate: Candidate | None = None

    @property
    def done(self):
        return self.next_candidate is None


def decide(ordered_candidates, scores, margin_pp=DEFAULT_MARGIN_PP):
    """Replay the search over the finished runs and name the next one.

    ``scores`` maps a candidate's ``order`` to its score, for finished runs
    only. Candidates are visited in their fixed order; the first one that is
    neither finished nor skipped is the next run. Worse, tie and best are
    judged against the best score measured *before* that candidate, which is
    what the search knew when the run finished.
    """
    margin_pp = float(margin_pp)
    if margin_pp < 0.0:
        raise ValueError("margin_pp must be non-negative")
    state = SearchState()
    for candidate in ordered_candidates:
        direction = candidate.direction
        if direction != DEFAULT and state.stopped[direction]:
            state.evaluations.append(Evaluation(candidate, "skipped"))
            continue
        if state.next_candidate is not None:
            # Later candidates wait: whether they run depends on the next one.
            state.evaluations.append(Evaluation(candidate, "pending"))
            continue
        if candidate.order not in scores:
            state.next_candidate = candidate
            state.evaluations.append(Evaluation(candidate, "next"))
            continue
        score = float(scores[candidate.order])
        if state.best is None:
            state.best, state.best_score = candidate, score
            state.evaluations.append(Evaluation(candidate, "reference", score))
            continue
        best_before = state.best_score
        delta = score - best_before
        if delta < -margin_pp:
            verdict = "worse"
            state.worse_counts[direction] += 1
            if state.worse_counts[direction] >= WORSE_RUNS_TO_STOP:
                state.stopped[direction] = True
        elif delta > 0.0:
            verdict = "better" if delta > margin_pp else "tie_above"
            state.best, state.best_score = candidate, score
        else:
            verdict = "tie_below"
        state.evaluations.append(
            Evaluation(candidate, verdict, score, best_before, delta)
        )
    return state


def search_ordinals(state, first_ordinal, unlaunched=()):
    """Map each launched or next candidate's ``order`` to its bundle number.

    Numbers follow launch order. Candidates are launched in their fixed order,
    and neither a skipped one nor one in ``unlaunched`` (a reference run made
    elsewhere) ever is, so replaying the decisions reproduces every number
    already handed out -- which a resumed or restarted run relies on.
    """
    ordinals = {}
    ordinal = int(first_ordinal)
    for evaluation in state.evaluations:
        if evaluation.verdict in ("skipped", "pending"):
            continue
        if evaluation.candidate.order in unlaunched:
            continue
        ordinals[evaluation.candidate.order] = ordinal
        ordinal += 1
    return ordinals


# ---------------------------------------------------------------------------
# Reading runs and runner state
# ---------------------------------------------------------------------------


def read_sequence_state(path):
    """Return runner rows by run name; a missing file means nothing ran yet."""
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as stream:
        return {row["run_name"]: row for row in csv.DictReader(stream, delimiter="\t")}


def run_directory(run_root, run_name, seed):
    return Path(run_root) / f"domino_rl_forever_seed{int(seed)}_run{run_name}"


def verify_run_configuration(run_dir, learning_rate, warmup):
    """Refuse to score a run whose recorded settings are not the expected ones."""
    # pylint: disable=import-outside-toplevel
    from training.canonical_run import load_run_config

    config = load_run_config(run_dir)
    locked = config.get("locked_arguments", {})
    recorded = float(config.get("rl_config", {}).get("learning_rate"))
    if abs(recorded - float(learning_rate)) > 1e-15:
        raise ValueError(
            f"{run_dir} trained at learning rate {recorded}, not {learning_rate}."
        )
    if bool(locked.get("warmup_lr")) is not bool(warmup):
        raise ValueError(
            f"{run_dir} ran {'without' if warmup else 'with'} --warmup-lr."
        )


def verify_reference_run(run_dir, candidate):
    """Refuse a reference run unless the search would have launched exactly it.

    The run was made outside the search, so matching the rate and the warmup is
    not enough: every locked training argument has to equal the one the
    search's own ``LR_default`` run would record, identity aside.
    """
    # pylint: disable=import-outside-toplevel,protected-access
    from training.canonical_run import load_run_config
    from training import pipeline

    command = [
        "forever", "--ruleset", "double-six", "--run-name", "reference",
        "--learning-rate", learning_rate_text(candidate.learning_rate), "--warmup-lr",
    ]
    expected = json.loads(json.dumps(
        pipeline._locked_run_arguments(pipeline.parse_args(command))
    ))
    recorded = load_run_config(run_dir).get("locked_arguments", {})
    differences = {
        key: (recorded.get(key), expected.get(key))
        for key in sorted(set(recorded) | set(expected))
        if key not in IDENTITY_ARGUMENTS and recorded.get(key) != expected.get(key)
    }
    if differences:
        raise ValueError(
            f"{run_dir} cannot stand in for {candidate.description}: its "
            f"settings (recorded, expected) differ in {differences}."
        )


def score_run(run_dir, learning_rate, warmup, window_fraction):
    # pylint: disable=import-outside-toplevel
    from diagnostics.rl_progress import read_periodic_history
    from training.run_artifacts import periodic_diagnostics_path

    verify_run_configuration(run_dir, learning_rate, warmup)
    history = read_periodic_history(periodic_diagnostics_path(run_dir))
    return sustained_final(history, window_fraction)


def load_or_lock_config(results_dir, config):
    """Record the search definition once and refuse to continue a different one."""
    path = Path(results_dir) / CONFIG_FILENAME
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != config:
            differences = {
                key: (saved.get(key), config.get(key))
                for key in sorted(set(saved) | set(config))
                if saved.get(key) != config.get(key)
            }
            raise ValueError(
                f"{path} records a different search {differences}. The runs "
                "already made belong to that search; use another results "
                "directory for a new one."
            )
        return saved
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return config


@dataclass
class SearchRuns:
    """The replayed search plus everything measured for its report."""

    state: SearchState
    names: dict
    details: dict
    statuses: dict
    ordinals: dict
    control: dict | None
    reference_run: str | None
    next_run: tuple | None


def evaluate_search(
    *,
    machine_slug,
    results_dir,
    run_root,
    seed,
    default_learning_rate,
    factors=DEFAULT_FACTORS,
    margin_pp=DEFAULT_MARGIN_PP,
    window_fraction=DEFAULT_WINDOW_FRACTION,
    control_run=False,
    reference_run=None,
    reference_results_dir=None,
    first_run_ordinal=DEFAULT_FIRST_RUN_ORDINAL,
):
    """Score every finished run, replay the search and name the next run.

    ``next_run`` is ``(kind, label, value, run_name, ordinal)`` for the
    sequence runner, or ``None`` once the search is over. The control run, when
    enabled, comes first; its score is reported but never compared. A
    ``reference_run`` is read from ``reference_results_dir``'s runner state and
    must have completed: the search never launches it.
    """
    first_run_ordinal = int(first_run_ordinal)
    if first_run_ordinal < 0:
        raise ValueError("first_run_ordinal must be non-negative")
    if reference_run is not None and reference_results_dir is None:
        raise ValueError("a reference run needs the results directory that ran it")
    runner_rows = read_sequence_state(Path(results_dir) / "sequence_state.tsv")

    def measured(rows, name, learning_rate, warmup):
        status = rows.get(name, {}).get("status") or "not_started"
        if status != "completed":
            return status, None
        run_dir = run_directory(run_root, name, seed)
        return status, score_run(run_dir, learning_rate, warmup, window_fraction)

    next_run = None
    control = None
    if control_run:
        name = control_run_name(machine_slug)
        status, scored = measured(runner_rows, name, default_learning_rate, False)
        control = {
            "run_name": name, "ordinal": first_run_ordinal,
            "status": status, "scored": scored,
        }
        if scored is None:
            next_run = (
                CONTROL_RUNNER_KIND, CONTROL_LABEL, CONTROL_RUNNER_VALUE,
                name, first_run_ordinal,
            )

    ordered = candidates(default_learning_rate, factors)
    names = {candidate.order: candidate.run_name(machine_slug) for candidate in ordered}
    scores, details, statuses = {}, {}, {}
    for candidate in ordered:
        rows = runner_rows
        if candidate.order == 1 and reference_run is not None:
            names[1] = reference_run
            rows = read_sequence_state(Path(reference_results_dir) / "sequence_state.tsv")
            status = rows.get(reference_run, {}).get("status") or "not_started"
            if status != "completed":
                raise ValueError(
                    f"The reference run {reference_run} has not completed in "
                    f"{reference_results_dir} (state: {status}); finish it first."
                )
            verify_reference_run(run_directory(run_root, reference_run, seed), candidate)
        status, scored = measured(rows, names[candidate.order], candidate.learning_rate, True)
        statuses[candidate.order] = status
        if scored is not None:
            scores[candidate.order] = scored["final_sustained_percent"]
            details[candidate.order] = scored
    state = decide(ordered, scores, margin_pp)
    ordinals = search_ordinals(
        state,
        first_run_ordinal + (1 if control_run else 0),
        unlaunched={1} if reference_run is not None else (),
    )
    if next_run is None and not state.done:
        candidate = state.next_candidate
        next_run = (
            SEARCH_RUNNER_KIND, candidate.label, candidate.runner_value(),
            names[candidate.order], ordinals[candidate.order],
        )
    return SearchRuns(
        state, names, details, statuses, ordinals, control, reference_run, next_run
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


VERDICT_TEXT = {
    "reference": "referência",
    "better": "melhor (acima da régua)",
    "tie_above": "empate acima (novo melhor)",
    "tie_below": "empate abaixo",
    "worse": "pior",
    "skipped": "não testado (direção parada)",
    "next": "próximo da busca",
    "pending": "aguardando",
}


def summary_payload(runs, config, machine_slug):
    state = runs.state
    rows = []
    for evaluation in state.evaluations:
        candidate = evaluation.candidate
        measured = runs.details.get(candidate.order, {})
        rows.append({
            "order": candidate.order,
            "run_ordinal": runs.ordinals.get(candidate.order),
            "label": candidate.label,
            "description": candidate.description,
            "learning_rate": candidate.learning_rate,
            "run_name": runs.names[candidate.order],
            "reference_run": candidate.order == 1 and runs.reference_run is not None,
            "runner_status": runs.statuses.get(candidate.order),
            "verdict": evaluation.verdict,
            "final_sustained_percent": evaluation.score,
            "best_before_percent": evaluation.best_before,
            "delta_pp": evaluation.delta_pp,
            "auc_percent": measured.get("auc_percent"),
            "horizon_hours": measured.get("horizon_hours"),
            "window_points": measured.get("window_points"),
        })
    control = None
    if runs.control is not None:
        scored = runs.control["scored"] or {}
        control = {
            "run_ordinal": runs.control["ordinal"],
            "description": CONTROL_DESCRIPTION,
            "learning_rate": config["default_learning_rate"],
            "warmup_lr": False,
            "run_name": runs.control["run_name"],
            "runner_status": runs.control["status"],
            "final_sustained_percent": scored.get("final_sustained_percent"),
            "auc_percent": scored.get("auc_percent"),
            "horizon_hours": scored.get("horizon_hours"),
            "window_points": scored.get("window_points"),
        }
    return {
        "config": config,
        "machine_slug": machine_slug,
        "done": runs.next_run is None,
        "next_run": None if runs.next_run is None else runs.next_run[3],
        "next_run_ordinal": None if runs.next_run is None else runs.next_run[4],
        "control_run": control,
        "best": None if state.best is None else {
            "label": state.best.label,
            "learning_rate": state.best.learning_rate,
            "run_name": runs.names[state.best.order],
            "final_sustained_percent": state.best_score,
        },
        "worse_counts": state.worse_counts,
        "stopped": state.stopped,
        "evaluations": rows,
    }


def summary_markdown(payload):
    config = payload["config"]
    lines = [
        f"# Busca de fator único do LR — {payload['machine_slug']}",
        "",
        f"LR_default {learning_rate_text(config['default_learning_rate'])}, fatores "
        f"{', '.join(str(f) for f in config['factors'])}, `--warmup-lr` em todas as runs. "
        f"Métrica: vitória média na última fração {config['window_fraction']:g} do horizonte; "
        f"pior = abaixo do melhor atual por mais de {config['margin_pp']:g} pp; "
        f"direção para após {WORSE_RUNS_TO_STOP} pioras. "
        f"Bundles numerados na ordem de lançamento a partir de {config['first_run_ordinal']}.",
        "",
    ]

    def percent(value):
        return "" if value is None else f"{value:.3f}%"

    def number(value):
        return "" if value is None else str(value)

    row = payload["control_run"]
    if row is not None:
        lines += [
            "## Run de controle (sem warmup, fora da decisão)",
            "",
            "| Nº | Teste | LR | Run | Estado | Final sustentado | AUC |",
            "|---:|---|---:|---|---|---:|---:|",
            f"| {row['run_ordinal']} | {row['description']} "
            f"| {learning_rate_text(row['learning_rate'])} | `{row['run_name']}` "
            f"| {row['runner_status']} | {percent(row['final_sustained_percent'])} "
            f"| {percent(row['auc_percent'])} |",
            "",
            "## Busca (todas com warmup)",
            "",
        ]
    if config.get("reference_run"):
        lines += [
            f"LR_default vem da run existente `{config['reference_run']}`, "
            "que a busca não relança nem numera.",
            "",
        ]
    lines += [
        "| Nº | # | Teste | LR | Run | Estado | Final sustentado | Δ vs melhor anterior | AUC | Decisão |",
        "|---:|---:|---|---:|---|---|---:|---:|---:|---|",
    ]
    for row in payload["evaluations"]:
        delta = "" if row["delta_pp"] is None else f"{row['delta_pp']:+.3f} pp"
        lines.append(
            f"| {number(row['run_ordinal'])} | {row['order']} | {row['description']} "
            f"| {learning_rate_text(row['learning_rate'])} "
            f"| `{row['run_name']}` | {row['runner_status']} "
            f"| {percent(row['final_sustained_percent'])} | {delta} "
            f"| {percent(row['auc_percent'])} | {VERDICT_TEXT[row['verdict']]} |"
        )
    lines += [
        "",
        f"Pioras: acima {payload['worse_counts'][UP]}, abaixo {payload['worse_counts'][DOWN]}. "
        f"Direção acima {'parada' if payload['stopped'][UP] else 'aberta'}, "
        f"abaixo {'parada' if payload['stopped'][DOWN] else 'aberta'}.",
        "",
    ]
    if payload["done"] and payload["best"] is not None:
        best = payload["best"]
        lines.append(
            f"**Busca encerrada.** Melhor LR testado: {learning_rate_text(best['learning_rate'])} "
            f"(`{best['run_name']}`, {best['final_sustained_percent']:.3f}%)."
        )
    elif payload["next_run"] is not None:
        lines.append(
            f"**Próxima run:** `{payload['next_run']}` (Nº {payload['next_run_ordinal']})."
        )
    return "\n".join(lines) + "\n"


def write_summary(results_dir, payload):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    for name, text in (
        (SUMMARY_JSON, json.dumps(payload, indent=2, ensure_ascii=False) + "\n"),
        (SUMMARY_MARKDOWN, summary_markdown(payload)),
    ):
        temporary = results_dir / (name + ".partial")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(results_dir / name)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _project_defaults():
    # pylint: disable=import-outside-toplevel
    from training.pipeline import DEFAULT_SEED
    from training.rl.config import DEFAULT_LEARNING_RATE

    return float(DEFAULT_LEARNING_RATE), int(DEFAULT_SEED)


def _search_config(args):
    default_learning_rate, _seed = _project_defaults()
    return {
        "default_learning_rate": default_learning_rate,
        "factors": list(DEFAULT_FACTORS),
        "margin_pp": float(args.margin_pp),
        "window_fraction": float(args.window_fraction),
        "worse_runs_to_stop": WORSE_RUNS_TO_STOP,
        "warmup_lr": True,
        "ruleset": "double-six",
        "control_run": bool(args.control_run),
        "reference_run": args.reference_run,
        "first_run_ordinal": int(args.first_run_ordinal),
    }


def cmd_plan(args):
    default_learning_rate, _seed = _project_defaults()
    ordinal = int(args.first_run_ordinal)
    rows = []
    if args.control_run:
        rows.append(("controle", CONTROL_DESCRIPTION, default_learning_rate,
                     control_run_name(args.machine_slug), True))
    for candidate in candidates(default_learning_rate):
        reference = candidate.order == 1 and args.reference_run is not None
        name = args.reference_run if reference else candidate.run_name(args.machine_slug)
        rows.append((f"busca {candidate.order}", candidate.description,
                     candidate.learning_rate, name, not reference))
    for position, description, learning_rate, name, launched in rows:
        number = f"{ordinal:>4}" if launched else "   -"
        ordinal += 1 if launched else 0
        suffix = "" if launched else "  (existing run, not launched)"
        print(
            f"{number}  {position:<9} {description:<21} "
            f"lr {learning_rate_text(learning_rate):<10} {name}{suffix}"
        )
    print("Nº is the bundle number if every run is launched; a skipped run takes no number.")
    return 0


def _evaluate(args):
    config = _search_config(args)
    saved = Path(args.results_dir) / CONFIG_FILENAME
    if not args.read_only:
        config = load_or_lock_config(args.results_dir, config)
    elif saved.is_file():
        config = json.loads(saved.read_text(encoding="utf-8"))
    _default, seed = _project_defaults()
    runs = evaluate_search(
        machine_slug=args.machine_slug,
        results_dir=args.results_dir,
        run_root=args.run_root,
        seed=seed,
        default_learning_rate=config["default_learning_rate"],
        factors=tuple(config["factors"]),
        margin_pp=config["margin_pp"],
        window_fraction=config["window_fraction"],
        control_run=config["control_run"],
        reference_run=config["reference_run"],
        reference_results_dir=args.reference_results_dir,
        first_run_ordinal=config["first_run_ordinal"],
    )
    payload = summary_payload(runs, config, args.machine_slug)
    if not args.read_only:
        write_summary(args.results_dir, payload)
    return runs, payload


def cmd_next(args):
    runs, _payload = _evaluate(args)
    if runs.next_run is None:
        return DONE_EXIT_CODE
    print(" ".join(str(part) for part in runs.next_run))
    return 0


def cmd_report(args):
    _runs, payload = _evaluate(args)
    sys.stdout.write(summary_markdown(payload))
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub, *, needs_results):
        sub.add_argument("--machine-slug", required=True)
        sub.add_argument(
            "--control-run",
            action="store_true",
            help="run the default without --warmup-lr first, outside the decision",
        )
        sub.add_argument(
            "--reference-run",
            default=None,
            help="an existing, completed LR_default warmup run to use instead of launching one",
        )
        sub.add_argument(
            "--first-run-ordinal", type=int, default=DEFAULT_FIRST_RUN_ORDINAL,
            help="bundle number of the first run launched (default: %(default)s)",
        )
        if needs_results:
            sub.add_argument("--results-dir", type=Path, required=True)
            sub.add_argument(
                "--reference-results-dir", type=Path, default=None,
                help="the sequence results directory whose state records --reference-run",
            )
            sub.add_argument("--run-root", type=Path, default=Path("models") / "rl")
            sub.add_argument("--margin-pp", type=float, default=DEFAULT_MARGIN_PP)
            sub.add_argument(
                "--window-fraction", type=float, default=DEFAULT_WINDOW_FRACTION
            )
            sub.add_argument(
                "--read-only",
                action="store_true",
                help="decide without recording the search or writing the summary",
            )

    common(commands.add_parser("plan", help="print every candidate in test order"), needs_results=False)
    common(commands.add_parser("next", help="print the next run, or exit 3 when done"), needs_results=True)
    common(commands.add_parser("report", help="print and write the search summary"), needs_results=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    command = {"plan": cmd_plan, "next": cmd_next, "report": cmd_report}[args.command]
    try:
        return command(args)
    except (OSError, ValueError) as error:
        # A run that cannot be scored or a search that does not match its
        # record stops the loop with the reason, not a traceback.
        print(f"lr_factor_search: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
