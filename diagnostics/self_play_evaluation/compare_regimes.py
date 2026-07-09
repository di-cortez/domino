"""
Compare two RL checkpoints trained with different training opponents.

The script expects one checkpoint trained with ``training_opponent="self_play"``
and another trained with ``training_opponent="heuristic"``. It evaluates both
against the fixed heuristic agent, then evaluates the two checkpoints directly.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.pairwise import evaluate_pair as evaluate, save_csv  # noqa: E402
from diagnostics.plots import generate_plots, summarize  # noqa: E402

SELF_PLAY_WEIGHTS = ROOT / "models" / "domino_rl_self_play_weights.npz"
HEURISTIC_WEIGHTS = ROOT / "models" / "domino_rl_heuristic_weights.npz"
DEFAULT_GAME_COUNT = 1000
DEFAULT_SEED = 7
DEFAULT_OUTPUT_DIR = ROOT / "diagnostics" / "results" / "self_play_vs_heuristic_regimes"


def _evaluate_and_save(agent_name, opponent_name, agent_weights, opponent_weights, game_count, seed, folder):
    """Run an evaluation and write summary.json, games.csv, and PNG plots."""
    folder.mkdir(parents=True, exist_ok=True)
    games = evaluate(
        agent_name,
        opponent_name,
        game_count,
        weights=agent_weights,
        opponent_weights=opponent_weights,
        seed=seed,
    )
    summary = summarize(games, agent_name, opponent_name, seed)
    save_csv(games, folder / "games.csv")
    with open(folder / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    generate_plots(games, summary, folder)
    return summary


def _confidence_intervals_overlap(summary_a, summary_b):
    lo_a, hi_a = summary_a["win_ci95"]
    lo_b, hi_b = summary_b["win_ci95"]
    return not (hi_a < lo_b or hi_b < lo_a)


def main():
    parser = argparse.ArgumentParser(
        description="Compare self-play RL against heuristic-trained RL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--self-play-weights", type=Path, default=SELF_PLAY_WEIGHTS)
    parser.add_argument("--heuristic-weights", type=Path, default=HEURISTIC_WEIGHTS)
    parser.add_argument("-n", "--games", type=int, default=DEFAULT_GAME_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    required = (
        (args.self_play_weights, "self-play checkpoint"),
        (args.heuristic_weights, "heuristic-trained checkpoint"),
    )
    for path, label in required:
        if not path.exists():
            print(f"ERROR: missing {label}: {path}")
            print("Generate it first with training.self_play.train(...).")
            sys.exit(1)

    game_count = args.games
    seed = args.seed
    output_dir = args.output

    print(f"=== 1/3: self-play checkpoint vs heuristic agent (n={game_count}) ===")
    self_play_summary = _evaluate_and_save(
        "rl",
        "heuristic",
        args.self_play_weights,
        None,
        game_count,
        seed,
        output_dir / "self_play_vs_heuristic",
    )
    lo, hi = self_play_summary["win_ci95"]
    print(f"  {self_play_summary['rates']['win']:.1%} wins | 95% CI: [{lo:.1%}, {hi:.1%}]")

    print(f"\n=== 2/3: heuristic-trained checkpoint vs heuristic agent (n={game_count}) ===")
    heuristic_summary = _evaluate_and_save(
        "rl",
        "heuristic",
        args.heuristic_weights,
        None,
        game_count,
        seed,
        output_dir / "heuristic_trained_vs_heuristic",
    )
    lo, hi = heuristic_summary["win_ci95"]
    print(f"  {heuristic_summary['rates']['win']:.1%} wins | 95% CI: [{lo:.1%}, {hi:.1%}]")

    print(f"\n=== 3/3: self-play checkpoint vs heuristic-trained checkpoint (n={game_count}) ===")
    direct_summary = _evaluate_and_save(
        "rl",
        "rl",
        args.self_play_weights,
        args.heuristic_weights,
        game_count,
        seed,
        output_dir / "self_play_vs_heuristic_trained_direct",
    )
    lo, hi = direct_summary["win_ci95"]
    print(f"  self-play checkpoint won {direct_summary['rates']['win']:.1%} | 95% CI: [{lo:.1%}, {hi:.1%}]")

    print("\n=== Conclusion ===")
    if _confidence_intervals_overlap(self_play_summary, heuristic_summary):
        print(
            "The 95% confidence intervals against the heuristic agent overlap. "
            "This run does not show a statistically clear winner between regimes."
        )
    else:
        better = (
            "self-play"
            if self_play_summary["rates"]["win"] > heuristic_summary["rates"]["win"]
            else "heuristic-trained"
        )
        print(f"The confidence intervals do not overlap; {better} has the higher win rate.")

    lo, hi = direct_summary["win_ci95"]
    if lo <= 0.5 <= hi:
        print(f"The direct-match 95% CI [{lo:.1%}, {hi:.1%}] includes 50%, consistent with a tie.")
    else:
        favorite = "self-play" if direct_summary["rates"]["win"] > 0.5 else "heuristic-trained"
        print(f"The direct-match 95% CI [{lo:.1%}, {hi:.1%}] excludes 50%; {favorite} led.")

    print(f"\nArtifacts saved in {output_dir}/")


if __name__ == "__main__":
    main()
