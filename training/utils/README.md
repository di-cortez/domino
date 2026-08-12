# Shared training utilities

Cross-stage helpers for dataset generation, supervised training, and RL.
This package imports nothing else from `training/`, so it can be used by
any stage without creating a cycle.

| File | Purpose |
|---|---|
| `seeding.py` | `stable_seed`: process-independent 64-bit seed derivation for labeled operations. |
| `cli_args.py` | Argument validators and the `--weight-decay`/`--dropout` controls shared by supervised and RL parsers. |
| `encoding.py` | `ENCODED_FEATURE_VERSION`, the persisted encoded-dataset feature contract. |

## Seed derivation

`stable_seed(base_seed, *parts)` hashes the base seed and its labels with
SHA-256 and returns the leading 64 bits. It is deliberately independent of
process identity, worker count, and Python hash randomization, so a fixed
seed describes game and update identity rather than scheduling. PPO
minibatch shuffling, rollout worker benchmarking, and the periodic/final
diagnostic streams all derive from it.

## Encoded-feature contract

`ENCODED_FEATURE_VERSION` is written into encoded-dataset metadata by
supervised training and compared by the canonical asset checks in
`training/canonical_assets.py`. Changing its value invalidates every
existing `.npz` cache and canonical supervised artifact, so change it only
together with a real change to the encoded feature layout.

## Shared regularization

Both regularizers are off by default; omitting their flags reproduces the
historical update rules exactly. Each flag takes an optional coefficient and
falls back to its default when passed bare, so `--dropout` and `--dropout 0.1`
are equivalent.

| Flag | Coefficient | Applies to | Behavior |
|---|---:|---|---|
| *(omitted)* | `0.0` | — | No regularization |
| `--weight-decay [COEFFICIENT]` | `0.0001` | supervised **and** RL | L2 decay on the weight matrices; biases are never decayed |
| `--dropout [RATE]` | `0.1` | supervised **and** RL | Inverted dropout on every hidden layer |

Each regularizer is one flag for both stages: the supervised policy and the RL
policy are the same architecture, and the RL run starts from the supervised
checkpoint, so a single coefficient keeps the two stages comparable. The two
flags are independent — requesting one never enables the other.

The two stages differ only in how the decay is applied. Supervised training
folds it into the gradient, matching the historical `--weight-decay` behavior.
The RL update applies it as a decoupled shrink after gradient clipping, so a
clipped step still decays and the logged gradient norms keep describing the
policy and value gradient alone.

Dropout is applied to training forward passes only. Validation, periodic and
final diagnostics, opponent-pool snapshots, rollout play, and the whole-buffer
PPO metrics all evaluate the complete network. The masks are drawn from the
host NumPy generator, whose state is part of the exact RL resume pair, so a
dropout run stays resumable on CPU and GPU alike.

One consequence is worth knowing before enabling dropout with PPO. The rollout
log-probabilities are recorded without dropout while the update evaluates a
thinned network, so the first-epoch importance ratios are no longer exactly
one; expect higher `approx_kl` and clip fractions, and consider raising
the fixed PPO stopping policy if updates stop early.

Runs and canonical assets created before these controls existed record no
regularization field. That absence is read as the disabled value, so existing
supervised checkpoints stay reusable and existing RL runs stay resumable
without a rebuild.

Because both flags change the supervised weights, they are part of the
seed-addressed asset identity. On `big`, `huge`, and `forever`, changing either
against existing reusable assets is refused until the rebuild is explicit:

```bash
python -m training.pipeline big --dropout 0.1 --retrain-supervised
```
