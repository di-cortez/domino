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

Outputs:

- `double_four_player0.jsonl`
- `double_four_player1.jsonl`
- `double_four_summary.json`
- `double_four_cheater_player0.jsonl`
- `double_four_cheater_player1.jsonl`
- `double_four_cheater_summary.json`
