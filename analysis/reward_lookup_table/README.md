# Neural-versus-heuristic reward lookup data

This directory contains the fixed policy checkpoints and the two-stage,
deterministic analysis pipeline used to study future rewards after neural
decisions.

The current raw corpus contains 100,000 complete neural-versus-heuristic games
for each compact ruleset. The neural policy alternates seats exactly, and the
heuristic continues to use the normal engine and exact opponent model. Games
never end in a draw.

## Checkpoints

Checkpoint discovery uses network shape rather than filenames:

| Ruleset | Checkpoint | Hidden layers |
| --- | --- | --- |
| `double-three` | `double three 68pp.npz` | 96 x 48 |
| `double-four` | `double four 66p.npz` | 128 x 64 |
| `double-five` | `double five 65pp.npz` | 192 x 96 |
| `double-six` | `double six 66p.npz` | 256 x 128 |

All four are policy-only checkpoints and are evaluated deterministically on
CPU. Their SHA-256 identities and complete structural metadata are recorded in
each raw manifest.

## Raw full histories

Generate or resume the default corpus from the repository root with:

```bash
python analysis/reward_lookup_table/generate_raw_histories.py
```

Outputs are written under `raw/<ruleset>/` as 100 independently checksummed
`part-*.jsonl.gz` chunks plus `manifest.json`. Each chunk contains 1,000 games.
Publishing is atomic, so an interrupted run resumes at the next complete chunk.
The raw directory is intentionally ignored by Git.

The default seed is fixed and each game derives its own NumPy PCG64 stream from
its ruleset and absolute game index. Results therefore do not depend on worker
count or scheduling; focused tests compare one-worker and multi-worker outputs
byte for byte. Ten spawned CPU-only workers are used by default.

Each line is one complete event-sourced game containing:

- both initial hands and the initial stock;
- every turn, acting seat and role, complete legal action set, chosen action,
  drawn tile when applicable, and compact pre/post state;
- identification of every genuine neural decision, excluding forced draws,
  passes, and single-tile choices;
- every raw neural/opponent draw or pass reward event;
- both final hands, final stock, winner, win reason, and the unit terminal
  decomposition into its empty-hand and blocked components.

The initial state plus the ordered actions and drawn tiles reconstruct every
hidden hand and stock state exactly. The final state is retained as an explicit
integrity check. No `gamma_i`, `gamma_f`, or reward decay is applied to the raw
records.

The generated corpus summary is:

| Ruleset | Games | Neural decisions | Neural wins | Neural losses |
| --- | ---: | ---: | ---: | ---: |
| `double-three` | 100,000 | 153,697 | 58,356 | 41,644 |
| `double-four` | 100,000 | 233,262 | 53,861 | 46,139 |
| `double-five` | 100,000 | 314,774 | 53,070 | 46,930 |
| `double-six` | 100,000 | 421,010 | 53,626 | 46,374 |

## Hand-size lookup adapter

The second stage is deliberately separate from raw generation:

```bash
python analysis/reward_lookup_table/build_reward_lookup.py
```

It streams the raw chunks and creates one compressed JSON object per ruleset in
`derived/`. The current generated lookup corpus contains 1,122,743 samples in
232 populated hand-size cells and occupies 51 MiB compressed (approximately
1.21 GB as plain JSON). Its lookup key is exactly:

```text
(neural_hand_size, opponent_hand_size)
```

The JSON spelling of a key is `"neural_hand_size,opponent_hand_size"`. The
chosen action is retained inside each observation for auditing, but it is not a
key field and does not partition the samples.

For every genuine neural decision, a sample retains all later individual
draw/pass unit events plus the unit terminal components. Each future item
includes both distance in engine turns and distance in later genuine neural
decisions. This allows later experiments to apply different initial and final
discount schemes without regenerating games.

## Fixed unit-component runtime lookups

Build the compact fixed tables from the derived samples with:

```bash
python analysis/reward_lookup_table/build_fixed_signed_reward_lookup.py
```

Outputs are written under `fixed/` as one deterministic gzip JSON runtime file
and one build-diagnostic manifest per ruleset. The runtime artifacts are not
integrated with training by this analysis.

Each runtime file contains eight logical tables: empty_hand, blocked, pass, and
draw, each indexed once by engine-turn distance and once by genuine-neural-decision
distance. Histogram index `k` is the actual exponent in `gamma**k`; an event or
terminal state immediately after a sampled decision therefore occupies bin
zero. Forced actions, draws, passes, and opponent actions do not increment the
decision clock.

The four components are the unit terms of the reward model in
`training/rl/reward_model.py`, and exactly one of the two terminal components is
non-zero per ending:

| Component | Meaning | Range |
| --- | --- | --- |
| `empty_hand` | `R_E`, the signed empty-hand outcome | `-1`, `0`, `+1` |
| `blocked` | `R_B = +/-m(Delta_p)`, the signed blocked margin utility | `[-1, 1]` |
| `pass` | signed pass-event count | integer |
| `draw` | signed draw-event count | integer |

An opponent event and a neural win are `+1`; a neural event and a neural loss
are negative. `blocked` is the one real-valued component, because `m(Delta_p)`
saturates a pip margin nonlinearly. Thus the artifact contains no reward weight
(`a_E`, `a_B`, `a_D`, `a_P`), no gamma, no eta, and no sample counts, smoothing,
or interpolation. Runtime evaluation of any component is simply:

```text
sum(coefficient[k] * gamma**k for k in histogram bins)
```

The baseline consumer applies the configured normalized scales and combines the
resulting terminal and local estimates with `reward_eta`. That mixing remains
deliberately outside these lookup artifacts: changing a weight or eta does not
require rebuilding or editing a table.

The packaged runtime artifacts under `training/rl/reward_lookup_tables/` are
still the superseded version 2 tables and are refused by the loader; see that
directory's README for what a rebuild needs.

A cell is retained only when it contains at least
`ceil(0.005 * ruleset_decisions)` starting samples. The generated artifacts
are:

| Ruleset | Threshold | Eligible cells | Maximum turn exponent | Maximum decision exponent |
| --- | ---: | ---: | ---: | ---: |
| `double-three` | 769 | 14 | 13 | 3 |
| `double-four` | 1,167 | 22 | 23 | 6 |
| `double-five` | 1,574 | 27 | 38 | 7 |
| `double-six` | 2,106 | 35 | 57 | 12 |

All eight tables contain exactly the same eligible keys. Empty lists represent
an identically zero histogram. The sidecar manifests retain source
hashes, thresholds, maximum exponents, and output checksums for build auditing;
those diagnostics are deliberately absent from the runtime files.

## Focused validation

```bash
python -m pytest -q analysis/reward_lookup_table/test_reward_lookup.py \
  analysis/reward_lookup_table/test_fixed_signed_reward_lookup.py
```

The tests cover all four checkpoint mappings, worker-invariant raw artifacts,
the hand-size-only key contract, exact replay from raw history to final engine
state, unit reward signs, unit remaining-pip counts, cancellation, multiple
events, clock distances, eligibility boundaries, deterministic output, and
direct equivalence with the production RL return semantics.

## Reduction-direction heatmaps

Generate two four-page maps for reducing unsupported hand-size cells:

```bash
python analysis/reward_lookup_table/generate_reduction_heatmaps.py
```

Cells covered by `(1,n)`, `(2,n)`, `(n,1)`, or `(n,2)`, together with cells
that meet the original 0.5% support threshold, retain their original count.
Every other cell receives one of three arrows. Because rows are agent hand
sizes, the directions mean:

```text
↖  (n-1, m-1)
↑  (n-1, m)
↗  (n-1, m+1)
```

The count-guided PDF chooses the candidate destination with the largest count
in the original, pre-fixed/ad-hoc heatmap. The response-guided PDF streams the
raw histories and follows the most frequent complete opponent response after
the neural decision: the opponent hand shrinks, stays unchanged, or grows.
Multiple draws are capped into the hand-grows direction. Missing evidence and
all-zero candidate counts fall back deterministically to `(n-1,m-1)`.

All arrows reduce the first coordinate, so recursively following them must
terminate. This is why the upper-right destination is `(n-1,m+1)`, rather than
the non-decreasing `(n,m+1)`.
