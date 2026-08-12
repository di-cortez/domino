# Supervised training

Trains the supervised domino policy from the generated dataset.
Owned by `training/supervised/`.

| File | Purpose |
|---|---|
| `training_loop.py` | Selects safe host/GPU storage, validates the fixed supervised batch, orchestrates plateau scheduling, and saves `models/domino_sl_weights.npz` plus its loss graph. |
| `cli.py` | Defines standalone and pipeline supervised arguments and the command-line entry point. |
| `dataset.py` | Filters JSONL records and owns memory-safe RAM, compressed NPZ, and mmap encoded-dataset storage. |
| `plotting.py` | Scales and atomically renders the supervised training/validation loss graph. |
| `reporting.py` | Owns detailed user-facing startup, device, checkpoint, artifact, and resource messages. |
| `runtime.py` | Implements supervised batch/workspace safety, GPU residency probes/windows, and supervised memory telemetry. |


Run:

```bash
python -m training.supervised.cli
```

The loop:

- reads `dataset/supervised_dataset.jsonl`;
- filters out forced draw/pass examples;
- filters out single-option tile-play examples;
- scans the JSONL twice and encodes `float32` arrays without retaining decoded records;
- checks cgroup-aware host RAM before every material allocation;
- encodes states and tile-play actions with `DominoEncoder`;
- uses `dataset/supervised_dataset_encoded.npz` when the encoded dataset fits safely in RAM;
- otherwise atomically builds disk-backed `supervised_dataset_X.npy`,
  `supervised_dataset_Y.npy`, and `supervised_dataset_metadata.json` files and
  opens them read-only with `mmap`;
- splits data into training and validation sets;
- selects CPU/GPU independently with `--sl-device {auto,cpu,gpu}` (`--device`
  is a standalone alias);
- uses a fixed mini-batch of 8,192 examples by default;
- keeps the complete dataset in GPU memory when safe, or rotates one reusable
  GPU window through a global per-epoch permutation when it is not;
- derives independent PCG64 streams for initialization, per-epoch shuffling,
  and dropout from one run-level `utils.myrandom.SeedPlan`;
- keeps the best validation checkpoint in memory;
- stops automatically after conservative repeated blocks confirm that
  training loss has saturated;
- saves `models/domino_sl_weights.npz`;
- writes `models/domino_sl_weights.random_manifest.json` with the effective
  root seed and derivation contract;
- writes `models/domino_sl_loss.png`, with one training-loss value per epoch
  and the validation-loss values already computed at validation intervals.

The loss graph uses only metrics collected by the current supervised run; it
does not run extra games or include win-rate data. A custom weights path such
as `models/experiment.npz` produces the sibling
`models/experiment_loss.png`. The PNG is replaced atomically after it is
rendered, so a plotting failure does not destroy the previous graph. Its lower
limit sits slightly below the terminal training loss and its upper limit is
the maximum observed loss, making the learned change visible instead of
spending most of the plot on the unused interval down to zero.

The supervised mini-batch is fixed at 8,192 by default and receives a memory
preflight before the first update. `--sl-batch-size N` can select one of the
former power-of-two candidate sizes from 1,024 through 1,048,576. The selected
size is capped to the number of training examples for small datasets.

GPU mode first probes resident example counts from 2,048 through 1,048,576
without changing weights. It preserves 512 MiB by default for batches,
activations, gradients, CUDA workspace, and fragmentation. `auto` falls back
safely to CPU when that reserve cannot be kept; explicit `gpu` fails before a
training update. Override host and GPU reserves with
`--sl-memory-reserve-mb` and `--sl-gpu-memory-reserve-mb`. The detailed command
reports the selected device, residency mode/capacity, one-time full upload,
fixed batch, and memory high/low watermarks. `train_script/run_pipeline.py` uses
`quiet=True`, so it continues suppressing per-epoch, checkpoint, scheduler, and
memory-detail chatter.

All supervised inputs, targets, weights, activations, gradients, and published
weights are `float32`. Legacy `float64` checkpoints remain loadable and are
cast on input. Supervised training has no resume mode: it starts from fresh
random weights, keeps its best state in memory, and atomically publishes the
weights only after the complete invocation succeeds. If it is interrupted, the
already completed dataset remains reusable and the next supervised invocation
starts from the beginning; no partial supervised checkpoint is selected.

`--sl-seed N` fixes one immutable root seed. Without it, the command obtains a
fresh root seed from operating-system entropy and reports it in the returned
summary and manifest. Initialization uses `SUPERVISED_INITIALIZATION`, every
epoch permutation uses `SUPERVISED_SHUFFLE/<epoch>`, and dropout uses its own
sequential `SUPERVISED_DROPOUT` generator. All draws are made by NumPy PCG64;
GPU runs transfer weights, masks, and permutation indices to CuPy instead of
creating backend-specific random streams. This keeps the random inputs to CPU
and GPU training aligned and leaves module-global NumPy/CuPy state untouched.
Changing this derivation contract invalidates canonical supervised reuse.

CuPy import alone is not treated as proof of a working GPU. At startup,
`agents/nn.py` also asks the CUDA runtime for a visible device; a missing driver,
hidden device, or unusable runtime produces a documented NumPy/CPU fallback
reason. The root README's **Linux GPU setup and verification** section contains
the driver checks, CUDA 12.x/13.x installation commands, a real calculation
test, and troubleshooting steps. `train_script/run_pipeline.py` prints the selected
supervised and RL-parent backends plus free/total RAM and VRAM before dataset
generation starts.

The encoded cache stores the source JSONL SHA-256 rather than its path or
modification time. It is rebuilt automatically when that content hash, the
encoder input/output dimensions, or the feature-version tag changes. Identical
JSONL copies therefore produce identical cache metadata in different paths and
at different times.

## Supervised scheduler and controls

The normal command starts at learning rate `0.005` and treats the requested
epoch count as a maximum. Automatic training-loss stopping is always active.
It compares medians of non-overlapping 25-epoch blocks. A block counts as
saturated when its relative improvement over the previous block is below
`0.001` (0.1%). The run stops only after four consecutive saturated blocks and
never before epoch 100. A genuine improvement resets the counter. These values
are the documented `TP_*` supervised-training constants at the beginning of
`agents/nn.py`; they are implementation policy rather than CLI hyperparameters.

Validation remains every 10 epochs, and validation-based LR decay is also on
by default. The first validation result establishes the global best; after
five consecutive checks without strict improvement, the LR is multiplied by
`0.5` and only the LR-specific failure counter resets. Another five failures
are required for another reduction. Optional validation early stopping has its
own counter; its patience should normally exceed LR patience so a reduced rate
has time to help. Whichever enabled stopping rule triggers first ends the run,
and the summary records `training loss plateau`, `validation loss plateau`, or
`epoch limit`.

Enable any control independently by adding its flag:

```bash
python -m training.supervised.cli --weight-decay
python -m training.supervised.cli --dropout
python -m training.supervised.cli --early-stopping
python -m training.supervised.cli --lr-decay 0.7 --lr-decay-patience 8
python -m training.supervised.cli --no-lr-decay
python -m training.supervised.cli --sl-device cpu --sl-seed 123
```

The supervised controls use these defaults:

| Flag | Behavior | Default |
|---|---|---:|
| `--weight-decay [COEFFICIENT]` | Adds L2 decay to the weight matrices, but not biases | off (`0.0001`) |
| `--dropout [RATE]` | Inverted dropout on every hidden layer | off (`0.1`) |
| `--early-stopping [PATIENCE]` | Stops after this many validation checks without improvement | `5` |
| `--lr-decay [FACTOR]` | Multiplies LR after the configured consecutive failed checks | `0.5` (on) |
| `--lr-decay-patience N` | Consecutive failed validation checks before each reduction | `5` |
| `--no-lr-decay` | Disables plateau scheduling for controlled comparisons | off |
| `--sl-device` / standalone `--device` | `auto`, forced `cpu`, or required `gpu` | `auto` |
| `--sl-batch-size N` | Fixed safe batch; power of two from 1,024 through 1,048,576 | `8,192` |
| `--hidden-layers N` | Number of hidden policy layers, 1 to 8 | `2` |
| `--hidden1-size N` ... `--hidden8-size N` | Width of one hidden layer | `256`, then `128` |
| `--sl-memory-reserve-mb N` | Free host RAM retained | `512` |
| `--sl-gpu-memory-reserve-mb N` | Effective free VRAM retained | `512` |
| `--sl-seed N` | Reproducible initialization and epoch permutations | unset |

Validation is checked every 10 epochs. The options can be combined and can
receive explicit values:

```bash
python -m training.supervised.cli \
  --weight-decay 0.00005 \
  --early-stopping 12 \
  --lr-decay 0.7 --lr-decay-patience 5 \
  --sl-device gpu
```

Reported training and validation losses remain cross-entropy values, allowing
loss curves to be compared with runs that do not enable weight decay.

The canonical pipeline accepts the same supervised controls:

```bash
python -m training.pipeline small \
  --weight-decay --early-stopping 12 --sl-device auto
```

## Hidden-layer depth and width

The policy architecture defaults to `168 -> 256 -> 128 -> 56`. Both the number
of hidden layers and each width come from `agents/network_architecture.py` and
are selected once for supervised training and the canonical pipeline:

| Flag | Meaning | Default |
|---|---|---:|
| `--hidden-layers N` | Hidden layers, from 1 to 8 | `2` |
| `--hidden1-size N` | Width of hidden layer 1 | `256` |
| `--hidden2-size N` | Width of hidden layer 2 | `128` |
| `--hidden3-size N` ... `--hidden8-size N` | Width of hidden layer 3 through 8 | `128` |

An omitted width falls back to the default for that position, which keeps the
historical 256 and 128 for the first two layers and uses 128 for every deeper
layer. Sizing a layer the requested depth does not have is rejected instead of
silently ignored, so `--hidden-layers 2 --hidden3-size 64` is an error.

```bash
# Unchanged default: two layers, 168 -> 256 -> 128 -> 56.
python -m training.pipeline forever --run-name baseline

# Four layers, sized 512, 256, 128 (defaulted), and 64.
python -m training.pipeline forever --hidden-layers 4 \
  --hidden1-size 512 --hidden2-size 256 --hidden4-size 64 \
  --retrain-supervised --run-name deep
```

The RL stage has no separate architecture flags: it adopts whatever depth and
widths the supervised checkpoint stores. Weight arrays stay named `W1..W{L}`
and `b1..b{L}` for `L` layers including the output layer, so a two-layer
checkpoint keeps exactly its historical `W1, b1, W2, b2, W3, b3` keys and every
pre-existing artifact loads unchanged.

Architecture is part of supervised compatibility metadata and the immutable
RL resume identity. A run cannot resume with different dimensions, and changing
depth or width against reusable assets requires `--retrain-supervised`.

Only the command line stops at eight layers, because one `--hidden<n>-size`
option has to exist per layer. Nothing in the networks, checkpoints, resume
state, or metadata is limited to that depth, so a programmatic experiment can
build any `n >= 1`:

```python
from agents.rl_nn import PolicyNetwork

network = PolicyNetwork(hidden_sizes=(512, 384, 256, 192, 128, 96, 64, 48, 32))
```

The learning-curve footer in `rl_vs_random_progress.png` reports the depth and
every width of the run it plots, for example
`hidden 4 layers 512x256x128x64`. That row shrinks its font when a deep
architecture would otherwise collide with the checkpoint block on its left.
