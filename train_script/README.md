# train_script

Automates the full training pipeline described in `training/README.md`:

1. generate supervised examples from heuristic-vs-heuristic games;
2. train the supervised neural policy;
3. refine that policy with self-play reinforcement learning.

## Usage

From the repository root, with the project virtual environment set up as
described in the top-level `README.md`:

```bash
train_script/run_training_pipeline.sh
```

The script activates `.venv` automatically if it exists at the repository
root, then runs the three stages in order, stopping immediately if any stage
fails (`set -euo pipefail`).

Show every available option:

```bash
train_script/run_training_pipeline.sh --help
```

## Examples

Quick smoke test with tiny sizes, to check the pipeline is wired correctly
before committing to a full run:

```bash
train_script/run_training_pipeline.sh --games 200 --sl-epochs 20 --rl-iterations 10
```

Re-run only the self-play RL stage against an already-trained supervised
checkpoint:

```bash
train_script/run_training_pipeline.sh --skip-dataset --skip-sl
```

Use custom file names/locations for a side-by-side experiment:

```bash
train_script/run_training_pipeline.sh \
  --dataset-file dataset/experiment_a.jsonl \
  --sl-weights-file models/experiment_a_sl.npz \
  --rl-weights-file models/experiment_a_rl.npz
```

## Options

Every flag simply forwards to the matching `python -m training.*` module,
which also accepts these same flags directly (see `training/README.md`).

| Flag | Stage | Meaning | Default |
|---|---|---|---|
| `--games` | dataset | Games to simulate | `20000` |
| `--dataset-file` | dataset | Output JSONL path | `dataset/supervised_dataset_teste3.jsonl` |
| `--sl-weights-file` | SL | Output weights path | `models/domino_sl_weights_teste3.npz` |
| `--sl-cache-file` | SL | Encoded dataset cache path | `dataset/supervised_dataset_encoded_teste3.npz` |
| `--sl-epochs` | SL | Training epochs | `800` |
| `--sl-batch-size` | SL | Mini-batch size | `1024` |
| `--sl-learning-rate` | SL | Learning rate | `0.005` |
| `--sl-checkpoint-every` | SL | Epochs between checkpoints | `20` |
| `--sl-checkpoint-dir` | SL | Checkpoint directory | `models/supervised_checkpoints_teste3` |
| `--sl-early-stopping-patience` | SL | Validation checks without improvement before stopping; `0` disables | `5` |
| `--sl-weight-decay` | SL | L2 penalty on the weight matrices; `0` disables | `0.0001` |
| `--sl-lr-decay-factor` | SL | LR multiplier on each validation check without improvement; `1` disables | `0.5` |
| `--rl-weights-file` | RL | Output weights path | `models/domino_rl_weights_teste3.npz` |
| `--rl-iterations` | RL | Training iterations | `800` |
| `--rl-games-per-iteration` | RL | Games played per iteration | `80` |
| `--rl-training-opponent` | RL | `self_play` or `heuristic` | `self_play` |
| `--rl-learning-rate` | RL | Learning rate | `0.001` |
| `--rl-entropy-coef` | RL | Entropy bonus coefficient | `0.01` |
| `--rl-log-interval` | RL | Iterations between log lines | `10` |
| `--rl-checkpoint-interval` | RL | Iterations between checkpoints | `50` |
| `--rl-pool-interval` | RL | Iterations between self-play pool snapshots | `10` |
| `--rl-max-pool-size` | RL | Max frozen snapshots kept in the pool | `50` |
| `--rl-evaluation-games` | RL | Games per checkpoint evaluation | `200` |
| `--rl-value-coef` | RL | Value-loss coefficient | `0.5` |
| `--rl-clip-grad-norm` | RL | Gradient-norm clipping threshold | `5.0` |
| `--rl-gamma` | RL | Terminal-reward discount per remaining decision | `0.99` |
| `--rl-no-normalize-advantages` | RL | Disable per-batch advantage normalization | normalization on |
| `--skip-dataset` | control | Skip dataset generation | off |
| `--skip-sl` | control | Skip supervised training | off |
| `--skip-rl` | control | Skip self-play reinforcement learning | off |

This was verified end-to-end with a tiny run (`--games 5 --sl-epochs 2
--rl-iterations 2`) exercising all three stages and the `--skip-*` flags.
