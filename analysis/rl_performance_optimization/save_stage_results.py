"""Copy one stage's compact evidence from scratch outputs into ``results/``.

    python save_stage_results.py STAGE SUITE_DIR SMOKE_REF_DIR SMOKE_STAGE_DIR \
        TIMING_DIR [TIMING_DIR ...]

Only summaries are kept: per-case exactness from the correctness suite, the
training-smoke comparisons, and each timing summary with scratch paths
redacted. Raw runs, parameters and fingerprints stay in scratch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _redact(value, prefix):
    if isinstance(value, dict):
        return {key: _redact(item, prefix) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, prefix) for item in value]
    if isinstance(value, str) and prefix and value.startswith(prefix):
        return "<scratch>" + value[len(prefix):]
    return value


def main(argv):
    stage, suite, smoke_reference, smoke_candidate, *timings = argv
    destination = HERE / "results" / stage
    destination.mkdir(parents=True, exist_ok=True)
    scratch = str(Path(suite).resolve().parent.parent)

    cases = []
    for path in sorted(Path(suite).glob("*_comparison.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        cases.append({
            "case": path.name.removesuffix("_comparison.json"),
            "all_exact": report["all_exact"],
            "iterations": [
                {
                    "iteration": row["iteration"],
                    "exact": row["exact"],
                    "epochs": row["epochs"],
                    "stopped_by_kl": row["stopped_by_kl"],
                    "metric_differences": row["metric_differences"],
                    "parameter_errors": row.get("parameter_errors"),
                }
                for row in report["iterations"]
            ],
        })
    (destination / "harness_correctness.json").write_text(
        json.dumps(cases, indent=1), encoding="utf-8"
    )

    smokes = []
    for fingerprint in sorted(Path(smoke_candidate).glob("*/fingerprint.json")):
        name = fingerprint.parent.name
        completed = subprocess.run(
            [
                sys.executable,
                str(HERE / "training_smoke.py"),
                "compare",
                str(Path(smoke_reference) / name / "fingerprint.json"),
                str(fingerprint),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        smokes.append({"smoke": name, **json.loads(completed.stdout)})
    (destination / "training_smoke_comparison.json").write_text(
        json.dumps(smokes, indent=1), encoding="utf-8"
    )

    for timing in timings:
        summary = json.loads(
            (Path(timing) / "timing_summary.json").read_text(encoding="utf-8")
        )
        (destination / f"{Path(timing).name}.json").write_text(
            json.dumps(_redact(summary, scratch), indent=1), encoding="utf-8"
        )
    print(destination)


if __name__ == "__main__":
    main(sys.argv[1:])
