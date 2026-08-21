"""Audit a completed RL run for champion_vs_learner correctness.

Reads a training metrics stream and, optionally, the run's resume state, and
checks the invariants that a real run can violate but a unit test cannot see:
exact GPI accounting across the bucket matrix, evaluation games staying outside
training counters, both champion buckets filling and evicting independently,
and the two racing targets producing the distinct win-rate signatures they
should.

The pool half of the audit needs the run's resume state. Three different
weights/state naming conventions exist in this repository, so pass the run
directory and let ``training_state.json`` name the pair; a weights file still
works and is resolved by convention.

Usage:
    python -m analysis.check_champion_vs_learner <metrics.jsonl> [run_dir]
    python -m analysis.check_champion_vs_learner <metrics.jsonl> [rl_weights.npz]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from training.rl.reporting import read_training_metrics
from training.rl.resume import load_resume_state, resume_state_path
from training.rl.pool import (
    CHAMPION_BUCKET_NAMES,
    CHAMPION_CANDIDATE_BATCH_SIZE,
    CHAMPION_FINAL_SURVIVORS,
)


def _read(path):
    """Return ``(header, rows)`` with rows as dicts.

    The stream is column-oriented on disk; ``read_training_metrics`` is the
    repository's own reader and owns the column mapping, so this never has to
    know the column order.
    """
    return read_training_metrics(path)


def _check(results, name, condition, detail=""):
    results.append((bool(condition), name, detail))


# Number of checks ``audit_pool`` contributes. Named so a skipped pool half can
# report an honest denominator instead of shrinking it silently.
# ``tests/test_check_champion_vs_learner.py`` pins this to the real count.
POOL_CHECK_COUNT = 9


def _state_beside(weights_path):
    """Return the resume state paired with a weights file, or ``None``.

    Three conventions coexist and only the first follows ``resume_state_path``:

    ``training_iter008427.npz``           + ``.resume.npz``   (numbered)
    ``games_..._latest_..._weights.npz``  + ``..._state.npz`` (canonical)
    ``latest_weights.npz``                + ``latest.resume.npz`` (alias)

    A canonical ``forever`` run publishes the last two and neither the second
    nor the third can be derived by appending ``.resume``, which is why asking
    only ``resume_state_path`` used to skip the pool half of a real run.
    """
    weights_path = Path(weights_path)
    candidates = [resume_state_path(weights_path)]
    name = weights_path.name
    if name.endswith("_weights.npz"):
        candidates.append(
            weights_path.with_name(name[: -len("_weights.npz")] + "_state.npz")
        )
    if name == "latest_weights.npz":
        candidates.append(weights_path.with_name("latest.resume.npz"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def resolve_checkpoint_pair(argument):
    """Return ``(weights, state)`` for a run directory or a weights file.

    For a directory the pair comes from ``training_state.json``, the same
    source ``training.canonical_run`` resumes from, so the audit reads the
    generation the run itself considers current rather than guessing a name.

    Raises ``FileNotFoundError`` with the paths that were tried. The caller
    must not turn that into a silent skip: the pool half is where every
    champion-state invariant lives.
    """
    argument = Path(argument)
    if argument.is_dir():
        state_file = argument / "training_state.json"
        if not state_file.is_file():
            raise FileNotFoundError(
                f"{argument} is not a run directory: no training_state.json"
            )
        published = json.loads(state_file.read_text(encoding="utf-8"))
        try:
            weights = argument / published["latest_weights_path"]
            state = argument / published["latest_resume_state_path"]
        except KeyError as error:
            raise FileNotFoundError(
                f"{state_file} names no current checkpoint ({error})"
            ) from error
        missing = [path for path in (weights, state) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "training_state.json points at files that are gone: "
                + ", ".join(str(path) for path in missing)
            )
        return weights, state
    if not argument.is_file():
        raise FileNotFoundError(f"No such run directory or weights file: {argument}")
    state = _state_beside(argument)
    if state is None:
        raise FileNotFoundError(
            f"No resume state pairs with {argument}. Tried "
            f"{resume_state_path(argument).name}, the canonical _state.npz, and "
            "the latest.resume.npz alias. Passing the run directory instead "
            "reads the pair from training_state.json."
        )
    return argument, state


def audit(metrics_path):
    """Return a list of ``(passed, name, detail)`` checks for one run."""
    header, rows = _read(metrics_path)
    training = header["metadata"]["training"]
    buckets = list(training["opponent_buckets"])
    order = header["bucket_results"]["bucket_order"]
    gpi = int(training["games_per_iteration"])
    results = []

    selected_champions = [
        name for name in CHAMPION_BUCKET_NAMES if name in buckets
    ]
    _check(
        results,
        "both champion buckets selected",
        len(selected_champions) == 2,
        f"selected: {selected_champions}",
    )
    _check(
        results,
        "evaluation manifest records both targets",
        (training.get("champion_evaluation") or {}).get("selected_targets")
        == selected_champions,
    )
    _check(results, "bucket order matches the selection", order == buckets)

    # Every row must account for exactly GPI games. A racing game that leaked
    # into the matrix would break this and nothing else would notice.
    bad_rows = [
        row["iteration"]
        for row in rows
        if sum(value[0] for value in row["bucket_results"]) != gpi
    ]
    _check(
        results,
        f"every row accounts for exactly {gpi} games",
        not bad_rows,
        f"offending iterations: {bad_rows[:5]}",
    )

    # Cumulative training games must equal iterations * GPI: 100,000 evaluation
    # games per event must never reach this counter.
    last = rows[-1]
    _check(
        results,
        "training counter excludes evaluation games",
        last["cumulative_games"] == len(rows) * gpi,
        f"{last['cumulative_games']} vs {len(rows) * gpi}",
    )

    # A champion bucket is empty until its first event, then never empty again.
    for name in selected_champions:
        column = order.index(name)
        games = [row["bucket_results"][column][0] for row in rows]
        first_played = next(
            (index for index, value in enumerate(games) if value > 0),
            None,
        )
        _check(
            results,
            f"{name}: unavailable before its first event",
            first_played is None or all(
                value == 0 for value in games[:first_played]
            ),
        )
        _check(
            results,
            f"{name}: receives games after its first event",
            first_played is not None,
            "no event completed in this run" if first_played is None else
            f"from iteration {rows[first_played]['iteration']}",
        )
        if first_played is not None:
            expected = CHAMPION_CANDIDATE_BATCH_SIZE + 1
            _check(
                results,
                f"{name}: first games arrive after {expected} iterations",
                rows[first_played]["iteration"] >= expected,
                f"at iteration {rows[first_played]['iteration']}",
            )

    return header, rows, results


def audit_pool(state):
    """Check the durable pool state a run finished with."""
    results = []
    pool_state = state["opponent_pool_state"]
    champion = pool_state["champion_state_by_bucket"]
    buckets = pool_state["buckets"]

    for name, block in sorted(champion.items()):
        _check(
            results,
            f"{name}: pending batch is never complete at rest",
            len(block["pending_candidate_ids"]) < CHAMPION_CANDIDATE_BATCH_SIZE,
            f"{len(block['pending_candidate_ids'])} pending",
        )
        members = buckets[name]["member_ids"]
        events = block["completed_event_count"]
        capacity = buckets[name]["capacity"]
        expected = min(events * CHAMPION_FINAL_SURVIVORS, capacity)
        _check(
            results,
            f"{name}: membership matches its event count",
            len(members) == expected,
            f"{len(members)} members after {events} events",
        )
        _check(
            results,
            f"{name}: no duplicate membership",
            len(set(members)) == len(members),
        )

    # Only the fixed-target bucket keeps a durable admission score.
    _check(
        results,
        "heuristic bucket stores its scores",
        "heuristic_win_rate_by_opponent_id"
        in champion.get("champion_vs_heuristic", {}),
    )
    _check(
        results,
        "learner bucket stores no durable score",
        "heuristic_win_rate_by_opponent_id"
        not in champion.get("champion_vs_learner", {}),
    )

    # The two buckets consume the same stream, so their counts stay close, but
    # nothing may require them to be equal.
    counts = [
        block["completed_event_count"] for block in champion.values()
    ]
    _check(
        results,
        "both buckets ran events",
        all(value > 0 for value in counts),
        f"event counts: {dict((k, v['completed_event_count']) for k, v in sorted(champion.items()))}",
    )
    return results


def main(argv):
    if not 2 <= len(argv) <= 3:
        print(__doc__)
        return 2
    header, rows, results = audit(argv[1])
    skipped = 0
    skip_reason = ""
    if len(argv) == 3:
        try:
            weights_path, state_path = resolve_checkpoint_pair(argv[2])
        except FileNotFoundError as error:
            skipped = POOL_CHECK_COUNT
            skip_reason = str(error)
        else:
            # The repository's own reader owns the resume payload format.
            state, _weights = load_resume_state(weights_path, str(state_path))
            results.extend(audit_pool(state))
    else:
        skipped = POOL_CHECK_COUNT
        skip_reason = (
            "No run directory or weights file given, so the champion-state "
            "invariants were not read."
        )

    width = max(len(name) for _passed, name, _detail in results)
    for passed, name, detail in results:
        mark = "PASS" if passed else "FAIL"
        suffix = f"   {detail}" if detail else ""
        print(f"{mark}  {name:<{width}}{suffix}")
    failed = [name for passed, name, _detail in results if not passed]
    print()
    # A skipped pool half must never read like a clean audit: it drops exactly
    # the champion-state checks this script exists for, so it is reported on
    # stdout, kept in the denominator, and made a non-zero exit.
    total = len(results) + skipped
    summary = f"{len(results) - len(failed)}/{total} checks passed"
    if skipped:
        summary += f", {skipped} SKIPPED"
    print(f"{summary} over {len(rows)} iterations")
    if skipped:
        print(f"SKIPPED the pool checks: {skip_reason}")
    return 1 if failed or skipped else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
