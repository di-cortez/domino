# Reward lookup tables

This package owns the fixed, versioned hand-size lookup artifacts used by the
`lookup-table` policy-gradient baseline. There is one deterministic gzip JSON
artifact and one checksum manifest for each supported ruleset.

The runtime key is `(learner_hand_size, opponent_hand_size)`. Each stored cell
contains `empty_hand`, `blocked`, `pass`, and `draw` unit histograms under both
engine turn and genuine learner-decision clocks, matching the four terms of the
reward model in `training/rl/reward_model.py`: `empty_hand` holds `R_E`,
`blocked` holds `R_B = +/-m(Delta_p)`, and the two event components hold signed
counts. The loader applies the configured `gamma_f`, `gamma_i`, normalized
reward scales, reward-distance mode, and `reward_eta`; none of those values is
baked into an artifact.

## Status: only `double-six` has been rebuilt

An artifact is refused unless it is format version 3. Version 2 stores the
pre-redesign `final`/`pips` terminal pair, which cannot express the new terminal
utility: `final` merges empty-hand and blocked endings into one signed outcome
and carries no pip margin.

| Ruleset | Format version | `--baseline lookup-table` |
| --- | --- | --- |
| `double-six` | 3 | available |
| `double-three`, `double-four`, `double-five` | 2 | refused until rebuilt |

The `double-six` artifact was rebuilt from a 100,000-game corpus generated on
this repository, not from the corpus the original tables came from. Its
reference policy is a locally available 66.4%-versus-random `double-six`
checkpoint of the same 256x128 architecture the analysis README documents, so
it stands in for `double six 66p.npz` rather than reproducing it: the corpus
produced 422,055 neural decisions against the documented 421,010, and 33
eligible cells against 35. The manifest records the checkpoint's SHA-256, so
which policy generated the table is always recoverable.

Rebuilding a ruleset needs the raw neural-versus-heuristic corpus and one fixed
policy checkpoint per ruleset, as described in
`analysis/reward_lookup_table/README.md`; neither is versioned here. With the
raw corpus in place the source games do not have to be replayed:

```bash
python analysis/reward_lookup_table/build_reward_lookup.py --rulesets RULESET
python analysis/reward_lookup_table/build_fixed_signed_reward_lookup.py --rulesets RULESET
python analysis/reward_lookup_table/rebuild_ad_hoc_lookups.py --rulesets RULESET
```

The last step replays the reviewed ad hoc recipe; copy its `ad_hoc/` output pair
into this directory to package it.

Missing cells follow this ordered policy:

1. a one-tile learner decision is the exact immediate-win result;
2. `(2,n)` with `n > 4` uses `(2,4)`;
3. `(n,1)` with `n > 5` uses `(5,1)`;
4. `(n,2)` with `n > 6` uses `(6,2)`;
5. every other missing `(n,m)` recursively uses `(n-1,m-1)`.

The artifacts are derived from the permanent analysis scripts in
`analysis/reward_lookup_table/`. They are source-controlled runtime inputs,
not generated training outputs. Their SHA-256 identities are saved in new RL
resume configurations so a table cannot change silently during a run.
