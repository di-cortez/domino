"""Read-only repository identity helpers."""

from pathlib import Path
import subprocess


def current_git_commit(start_path=None):
    """Return the current Git commit, or ``None`` outside a readable checkout."""
    candidates = []
    if start_path is not None:
        candidates.append(Path(start_path))
    candidates.append(Path(__file__).resolve().parents[1])
    for candidate in candidates:
        try:
            return subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=candidate,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
    return None
