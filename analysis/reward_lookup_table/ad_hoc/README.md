# Ad hoc fixed reward lookups

This directory starts as a byte-for-byte copy of `../fixed/` and is the
editable candidate for structural simplifications and carefully reviewed
manual cell completion. The original fixed artifacts remain unchanged and are
the recovery source.

## Established conclusions

### An agent with one tile wins immediately

For every genuine neural decision with `agent_hand_size == 1`, the chosen play
empties the agent's hand and ends the game immediately. The opponent hand size
does not change this fact. Consequently, both distance clocks have the same
four histograms:

```text
component   turn histogram  decision histogram
empty_hand  [1.000]         [1.000]
blocked     []              []
pass        []              []
draw        []              []
```

The `empty_hand` coefficient is in the exponent-zero bin: `+1 * gamma**0 = +1`
for every gamma. The empty `blocked` histogram records that the ending is not a
block at all. Empty pass and draw lists are likewise identically zero
histograms, not unknown data.

The derived corpus confirms this independently for all four rulesets:

| Ruleset | Samples in cell `(1,1)` | Wins | Distance zero | Future events |
| --- | ---: | ---: | ---: | ---: |
| `double-three` | 1,162 | 1,162 | 1,162 | 0 |
| `double-four` | 3,649 | 3,649 | 3,649 | 0 |
| `double-five` | 926 | 926 | 926 | 0 |
| `double-six` | 2,144 | 2,144 | 2,144 | 0 |

The `double-five` fixed lookup omitted `(1,1)` only because 926 samples are
below its 1,574-sample eligibility threshold. It is not contrary evidence.

Because this result is structural, ad hoc runtime tables deliberately omit all
cells whose agent hand size is one. Any future lookup consumer must apply this
structural result before consulting the table; a missing one-tile cell must not
be interpreted through a generic missing-cell fallback.

### Extrapolating `(2,n)` beyond the observed opponent-hand boundary

For `agent_hand_size == 2`, the largest opponent hand size retained in every
ruleset lookup is four. A future consumer should therefore evaluate every
missing `(2,n)` cell with `n > 4` by reusing the complete `(2,4)` histograms:

```text
lookup(2, n) = lookup(2, 4)  for every n > 4
```

This is a deliberate boundary-clamping approximation, not a structural game
identity. It is preferable to inventing an unsupported trend: `(2,4)` is the
nearest observed cell, is supported in all four rulesets, and already describes
a large material advantage for the two-tile agent. For the terminal component,
the substitution is plausibly conservative because an opponent with more than
four tiles would not normally be expected to be closer to winning than one
with four. Pass and draw dynamics need not be monotonic in opponent hand size,
so their reused histograms remain an explicit approximation to be revisited if
direct data become available.

The ad hoc files are not expanded with duplicate `(2,n)` entries. This rule is
recorded for eventual lookup-consumer behavior, keeping `(2,4)` as the single
stored boundary cell.

### Extrapolating `(n,1)` beyond five agent tiles

For an opponent with one tile, a future consumer should reuse the complete
`(5,1)` histograms whenever the agent has more than five tiles:

```text
lookup(n, 1) = lookup(5, 1)  for every n > 5
```

The `double-three` data originally ended at `(4,1)`. This ad hoc variant adds
`(5,1)` there as an exact copy of all eight `(4,1)` histograms, solely so every
ruleset exposes the same explicit boundary cell. The other three rulesets
already contained an observed `(5,1)` cell.

This is another boundary-clamping approximation, not a structural identity.
The files do not duplicate cells for `n > 5`; an eventual lookup consumer
should clamp those requests to `(5,1)`.

### Extrapolating `(n,2)` beyond six agent tiles

For an opponent with two tiles, a future consumer should reuse the complete
`(6,2)` histograms whenever the agent has more than six tiles:

```text
lookup(n, 2) = lookup(6, 2)  for every n > 6
```

The observed `double-three` and `double-four` data ended at `(5,2)`. This ad
hoc variant adds `(6,2)` to each as an exact copy of all eight `(5,2)`
histograms, making the explicit boundary uniform across rulesets. The
`double-five` and `double-six` lookups already contained observed `(6,2)`
cells.

This is a boundary-clamping approximation, not a structural identity. The
files do not duplicate cells for `n > 6`; an eventual lookup consumer should
clamp those requests to `(6,2)`.

### Current missing-cell rules

The current ad hoc policy uses the following ordered rules. A stored cell is
always used directly. Only a missing cell proceeds through this table:

| Requested cell | Result used | Basis |
| --- | --- | --- |
| `(1,n)` for any opponent size | Immediate empty-hand win; `empty_hand` `[1]`, `blocked`/pass/draw `[]` | Exact structural game rule |
| `(2,n)` for `n > 4` | `(2,4)` | Boundary approximation; observed anchor in every ruleset |
| `(n,1)` for `n > 5` | `(5,1)` | Boundary approximation; synthetic `(5,1)` only in `double-three` |
| `(n,2)` for `n > 6` | `(6,2)` | Boundary approximation; synthetic `(6,2)` in `double-three` and `double-four` |
| Any other missing `(n,m)` | Recursively use `(n-1,m-1)` | General simplified fallback |

Only the first row is a structural identity. The other three rows are explicit
nearest-boundary approximations adopted to avoid inventing an unsupported
trend. The final row completes the policy: after the explicit cases have been
checked, every still-unresolved cell moves diagonally toward smaller hands:

```text
lookup(n, m) = lookup(n - 1, m - 1)
```

Apply this fallback recursively until it reaches a stored cell or one of the
explicit rules above. Both coordinates decrease on every fallback step, so it
cannot cycle and must terminate. The complete set of eight histograms is reused
from the resolved destination cell.

This general fallback is deliberately simple rather than an assertion that
the two game states are identical. The original count-guided and immediate
opponent-response heatmaps were compared before adopting it. The immediate
response was overwhelmingly a tile play, whose hand-size transition is
exactly `(n-1,m-1)`; the few contrary modal cells had only one to five
observations.

## Inspection

Print one or a small list of cells without modifying anything:

```bash
python analysis/reward_lookup_table/show_fixed_cells.py 1,1
python analysis/reward_lookup_table/show_fixed_cells.py 1,1 2,3 \
  --rulesets double-three double-four
```

## Controlled editing

`edit_ad_hoc_cells.py` modifies only files in this directory. Its operations
accept explicit cells rather than batch rules. It can remove one cell or a
small explicit list of at most eight cells from one ruleset:

```bash
python analysis/reward_lookup_table/edit_ad_hoc_cells.py remove \
  --ruleset double-four 1,1 1,2
```

It can also copy all eight histograms from one present source cell to one absent
target cell:

```bash
python analysis/reward_lookup_table/edit_ad_hoc_cells.py copy \
  --ruleset double-three --source 4,1 --target 5,1
```

The editor validates that all eight logical tables have identical key sets,
writes deterministic gzip JSON through an atomic replacement, and refreshes
the output size and SHA-256 in the sidecar manifest. The manifest's original
build fields remain intact; `ad_hoc_modifications` records removed cells and
copied cells, tail trimming, and the current runtime-cell count.

### Removing insignificant histogram tails

Histogram position is semantic: list index `k` is the exponent in `gamma**k`.
An internal coefficient can never be deleted without shifting every later
exponent. The safe shortening operation therefore removes only a consecutive
suffix whose coefficients truncate, rather than round, to zero at three
decimal places:

```bash
python analysis/reward_lookup_table/edit_ad_hoc_cells.py trim-zero-tails \
  --ruleset double-six --precision 3
```

No retained coefficient is rounded, truncated, or replaced. Internal values
whose three-decimal truncation is zero remain untouched as positional
placeholders. The sidecar manifest records the precision and cumulative number
of removed turn-clock and decision-clock coefficients.

> **These inventories describe the superseded version 2 build.** They were
> measured against the old `final`/`pips` components, so every cell and
> coefficient count below has to be re-measured once the corpus is rebuilt
> against the `empty_hand`/`blocked` decomposition. The methodology in this
> document is unchanged; only the numbers are stale.

The three-decimal tail pass produced the following inventory. "Coefficients"
counts stored dense histogram bins across the four components and
across both clocks; removing a one-tile cell and trimming histogram tails are
separate operations.

| Ruleset | Initial cells | After one-tile removal | After tail trim | Initial coefficients | After one-tile removal | After tail trim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `double-three` | 14 | 13 | 13 | 568 | 566 | 560 |
| `double-four` | 22 | 20 | 20 | 1,758 | 1,754 | 1,680 |
| `double-five` | 27 | 27 | 27 | 3,955 | 3,955 | 3,731 |
| `double-six` | 35 | 34 | 34 | 7,583 | 7,581 | 7,057 |

Tail trimming removed 828 coefficients in total:

| Ruleset | Turn coefficients removed | Decision coefficients removed | Total |
| --- | ---: | ---: | ---: |
| `double-three` | 4 | 2 | 6 |
| `double-four` | 41 | 33 | 74 |
| `double-five` | 145 | 79 | 224 |
| `double-six` | 349 | 175 | 524 |

The removed coefficients are small but not mathematically zero. This ad hoc
variant is therefore approximate: after independent clock-tail trimming, the
turn-clock and decision-clock sums at `gamma = 1` can differ slightly. The
original exact coefficients and invariants remain available under `../fixed/`.

After tail trimming, the two explicit `double-three` boundary copies add two
runtime cells and 89 copied coefficients. Its current ad hoc inventory is
therefore 15 cells and 649 coefficients. The `double-four` copy from `(5,2)`
to `(6,2)` adds one cell and 86 coefficients, for a current inventory of 21
cells and 1,766 coefficients. The `double-five` and `double-six` inventories
are unchanged from the tables above.

## Reproducible rebuild

The complete ad hoc set can be reset from `../fixed/` and all reviewed cell
removals, three-decimal tail trims, and boundary copies can be replayed with:

```bash
python analysis/reward_lookup_table/rebuild_ad_hoc_lookups.py
```

The recipe is explicit in that script. It applies every operation to
`empty_hand`, `blocked`, `pass`, and `draw` under both clocks, so a resolved
cell is always reused as one indivisible set of eight histograms.
