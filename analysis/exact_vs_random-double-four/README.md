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
displays one `tqdm` progress bar. Ten fixed worker processes use a dynamic
queue with one complete initial hand per task, so a worker that finishes a
hand immediately receives the next available one. Only the parent process
writes results. Completed hands are appended durably as they arrive, so
rerunning the same command resumes all four passes. At the end of each pass,
the parent atomically rewrites its JSONL in canonical `hand_index` order.

For partial-information hands, each worker keeps four cache layers scoped to
its current hand:

1. `Belief -> Fraction` stores the exact value of a complete information-state
   distribution.
2. `World -> tuple[Action, ...]` stores encoded legal actions. Both public
   legal-action requests and transition execution use the same lookup helper,
   so a transition does not call `DominoEngine.valid_actions()` again after a
   cache hit.
3. `(World, non-draw Action) -> exact transition` stores deterministic play and
   pass successors.
4. `World -> tuple[(drawn tile id, next World), ...]` compactly stores all exact
   DRAW successors. Uniform probabilities are reconstructed when requested,
   rather than retaining redundant `Fraction` objects in the cache.

All World-operation caches are discarded before the worker accepts its next
hand. The perfect-information solver disables them: its existing
`World -> value` cache already makes every legal-action and transition lookup
unique, so these additional caches would have zero hits. Every computed JSONL
row records deterministic hit, miss, and entry counts for the legal-action,
non-draw-transition, and DRAW-transition caches. The DRAW fields are named
`draw_transition_cache_hits`, `draw_transition_cache_misses`, and
`draw_transition_cache_entries`. Rows from runs created before those fields
were introduced remain valid and resumable.

At a hero decision, production recursion trusts the solver invariant that all
hidden Worlds in one observable belief expose the same legal actions and uses
the first World's cached tuple directly. Development validation can set
`ExactVsRandomSolver.VALIDATE_COMMON_HERO_ACTIONS = True` to scan every World
and assert the invariant explicitly; complete double-three validation runs use
this mode before performance changes are accepted.

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

## Results

Both information modes are complete: all 3,003 unordered initial hands were
solved for each seat, and every unordered initial hand is equally likely with
probability 1/3003. Unlike double-three, the two seats do not coincide. The
2,751 hands that force a double opener are label-symmetric, but the 252
no-double hero hands can leave all five doubles in the stock and then trigger
the asymmetric player-0 fallback opening rule, which gives seat 0 a small edge.

| Information mode | Seat | Exact win probability | Decimal |
| --- | --- | --- | --- |
| Partial (belief) | player 0 | 116981237213033 / 164766970368000 | 0.709979900411839 |
| Partial (belief) | player 1 | 116968733940713 / 164766970368000 | 0.709904015831986 |
| Partial (belief) | seat-averaged | 116974985576873 / 164766970368000 | 0.709941958121912 |
| Perfect (cheater) | player 0 | 90952326943451 / 112983065395200 | 0.80500849065576 |
| Perfect (cheater) | player 1 | 90943784800091 / 112983065395200 | 0.804932885136206 |
| Perfect (cheater) | seat-averaged | 12992579410253 / 16140437913600 | 0.804970687895983 |

The seat-averaged row is the value of a game in which the controlled player is
assigned to seat 0 or seat 1 with equal probability; it is the mean of the two
seat values, reported as `equal_probability_player0_player1` in the summaries.

An optimal belief-based player therefore beats the uniform `RandomAgent` in
about 70.99% of double-four games, and about 80.50% when it also observes the
opponent hand and the stock membership. Perfect information is worth about
9.50 percentage points here, noticeably more than the 6.12 points it is worth
in double-three, because the larger stock and hand hide more state. The seat
advantage is small in both modes: about 0.008 percentage points.

These numbers are read from `double_four_summary.json` and
`double_four_cheater_summary.json`, whose `complete` flag is `true` and whose
`completed_hands` equals `expected_hands` for both seats.

Outputs:

- `double_four_player0.jsonl`
- `double_four_player1.jsonl`
- `double_four_summary.json`
- `double_four_cheater_player0.jsonl`
- `double_four_cheater_player1.jsonl`
- `double_four_cheater_summary.json`
