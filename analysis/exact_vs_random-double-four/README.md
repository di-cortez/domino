# Exact optimal vs random — double-four

This directory computes the exact win probability of two optimal controlled
players against the repository's uniform `RandomAgent` policy for every
unordered double-four initial hand. The normal player acts from an exact belief
over hidden opponent hands. The perfect-information "cheater" continuously
observes both hands and the membership of the stock. All probabilities use
`fractions.Fraction`.

The stock is stored as an unordered set. At each draw, the solver branches
uniformly over every remaining tile. This integrates all `5!` possible initial
stock orders exactly without materializing 120 otherwise equivalent worlds.

Game semantics are not reimplemented here. The solver creates initial deals
through `DominoEngine.reset()` and performs legal-action generation, state
transitions, draws, passes, terminal detection, and blocked-game resolution
through `middleware.domino_engine.DominoEngine`.

Run the complete workflow from the repository root:

```bash
.venv/bin/python analysis/exact_vs_random-double-four/run.py
```

The driver processes normal seat 0, normal seat 1, cheater seat 0, and cheater
seat 1, in that order, then creates both aggregate summaries. Each pass
displays one `tqdm` progress bar. Completed hands are appended durably to
separate JSONL files, so rerunning the same command resumes all four passes.

The cheater does not know the future order of the stock. A draw remains a
uniform exact chance event, but after either player draws, the cheater observes
the resulting complete state. A normal result that is exactly zero or one is
reused directly because additional information cannot change that result.

Seat 0 is solved first. Its results are copied to seat 1 whenever at least one
initial hand must contain a double, because swapping player labels then
preserves the compulsory opener and the complete game distribution. This
copies 2,751 of the 3,003 double-four hands. The 252 no-double hero hands are
solved independently for seat 1 because all five doubles can be in the stock,
triggering the asymmetric player-0 fallback opening rule.

Double-four has 3,003 unordered initial hands per seat and 252 opponent-hand
allocations per hand. Stock permutations are chance-integrated as described
above. Exact belief-tree solving can still take a long time. A bounded
validation run is available with `--limit`, although its aggregate remains
marked incomplete:

```bash
.venv/bin/python analysis/exact_vs_random-double-four/run.py --limit 1
```

Outputs:

- `double_four_player0.jsonl`
- `double_four_player1.jsonl`
- `double_four_summary.json`
- `double_four_cheater_player0.jsonl`
- `double_four_cheater_player1.jsonl`
- `double_four_cheater_summary.json`
