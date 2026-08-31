# Reward lookup tables

This package owns the fixed, versioned hand-size lookup artifacts used by the
`lookup-table` policy-gradient baseline. There is one deterministic gzip JSON
artifact and one checksum manifest for each supported ruleset.

The runtime key is `(learner_hand_size, opponent_hand_size)`. Each stored cell
contains `final`, `pips`, `pass`, and `draw` unit histograms under both engine
turn and genuine learner-decision clocks. The loader applies the configured
`gamma_f`, `gamma_i`, reward magnitudes, reward-distance mode, and `reward_eta`;
none of those values is baked into an artifact.

Missing cells follow this ordered policy:

1. a one-tile learner decision is the exact immediate-win result;
2. `(2,n)` with `n > 4` uses `(2,4)`;
3. `(n,1)` with `n > 5` uses `(5,1)`;
4. `(n,2)` with `n > 6` uses `(6,2)`;
5. every other missing `(n,m)` recursively uses `(n-1,m-1)`.

The artifacts were derived from the permanent analysis scripts in
`analysis/reward_lookup_table/`. They are source-controlled runtime inputs,
not generated training outputs. Their SHA-256 identities are saved in new RL
resume configurations so a table cannot change silently during a run.
