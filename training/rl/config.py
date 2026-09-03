"""Typed reinforcement-learning inputs and side-effect-free resolution."""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from agents.nn import DISABLED_DROPOUT_RATE, DISABLED_WEIGHT_DECAY
from diagnostics.parallel_runner import MAX_PARALLEL_WORKERS, ParallelSafetyConfig
from training.rl.parallel import DEFAULT_RL_WORKERS
from training.rl.pool import (
    BOOTSTRAP_CAPABLE_BUCKETS,
    CHAMPION_BUCKET_NAMES,
    CHAMPION_REQUIRED_BUCKETS,
    DEFAULT_OPPONENT_BUCKETS,
    canonicalize_bucket_names,
)
from training.rl.ppo import (
    DEFAULT_PPO_MAX_EPOCHS,
    PPO_TRAINING_ALGORITHM,
    REINFORCE_TRAINING_ALGORITHM,
    ppo_is_enabled,
    validate_ppo_max_epochs,
)
from training.rl.baseline import resolve as resolve_baseline
from training.rl.reward_distance import (
    DEFAULT_REWARD_DISTANCE_MODE,
    resolve_reward_distance_mode,
)
from training.rl.reward_model import (
    DEFAULT_GAMMA_F,
    DEFAULT_GAMMA_I,
    DEFAULT_IMMEDIATE_DRAW_WEIGHT,
    DEFAULT_IMMEDIATE_PASS_WEIGHT,
    DEFAULT_REWARD_ETA,
    DEFAULT_TERMINAL_BLOCKED_WEIGHT,
    DEFAULT_TERMINAL_EMPTY_HAND_WEIGHT,
    resolved_reward_scales,
)
from training.rl.rollout import DEFAULT_REWARD_SCHEMA
from middleware.rulesets import DEFAULT_RULESET_NAME, resolve_ruleset
from utils.ruleset_paths import default_rl_weights_path, default_sl_weights_path


# The array backend is resolved once inside train() and always matches the
# PolicyNetwork backend selected for that run.
DEFAULT_DEVICE = "auto"
COMMON_GPI_VALUES = (100, 200, 400, 600, 800, 1000, 2000)
DEFAULT_GPI = 2000
DEFAULT_TOTAL_TRAINING_GAMES = 100_000

SL_WEIGHTS = "models/domino_sl_weights.npz"
RL_WEIGHTS = "models/domino_rl_weights.npz"
DEFAULT_DIFFICULTY_WEIGHT = 0.5
VALUE_COEF = 0.5
DEFAULT_MOVING_AVERAGE_WINDOW = 10

# ``None`` resolves to on for PPO and off for the single-update REINFORCE path.
# Explicit advantage-normalization CLI flags still win.
DEFAULT_NORMALIZE_ADVANTAGES = None

# ``None`` resolves to the baseline the run implied before ``--baseline``
# existed: the critic when its head is on, otherwise the batch mean whenever
# normalization is on and no baseline at all when it is off.
DEFAULT_BASELINE = None


@dataclass(frozen=True)
class RLTrainingOptions:
    """Values that define the learning problem and update behavior."""

    iterations: int | None = None
    ruleset_name: str = DEFAULT_RULESET_NAME
    total_training_games: int | None = None
    gpi: int = DEFAULT_GPI
    opponent_buckets: tuple[str, ...] = DEFAULT_OPPONENT_BUCKETS
    difficulty_weight: float = DEFAULT_DIFFICULTY_WEIGHT
    opponent_decision_restarts: bool = False
    learning_rate: float = 0.001
    entropy_coef: float = 0.01
    weight_decay: float = DISABLED_WEIGHT_DECAY
    dropout_rate: float = DISABLED_DROPOUT_RATE
    use_value_head: bool = False
    value_coef: float = VALUE_COEF
    # False drops the trailing exact-model block from the encoder, shortening
    # the policy input. Checkpoints are not interchangeable across this flag.
    use_opponent_suit_features: bool = True
    # True appends a one-hot of the bucket each seat's adversary was drawn
    # from, lengthening the policy input by the whole bucket registry. Like the
    # flag above it changes the input size, so checkpoints are not
    # interchangeable across it.
    use_opponent_bucket_features: bool = False
    gamma_f: float = DEFAULT_GAMMA_F
    reward_eta: float = DEFAULT_REWARD_ETA
    gamma_i: float = DEFAULT_GAMMA_I
    reward_distance_mode: str = DEFAULT_REWARD_DISTANCE_MODE
    # The four reward-component weights. Only the ratio inside each pair
    # matters, because each pair is normalized by its own larger member, so
    # these are non-negative ratios rather than probabilities.
    terminal_empty_hand_weight: float = DEFAULT_TERMINAL_EMPTY_HAND_WEIGHT
    terminal_blocked_weight: float = DEFAULT_TERMINAL_BLOCKED_WEIGHT
    immediate_draw_weight: float = DEFAULT_IMMEDIATE_DRAW_WEIGHT
    immediate_pass_weight: float = DEFAULT_IMMEDIATE_PASS_WEIGHT
    normalize_advantages: bool | None = DEFAULT_NORMALIZE_ADVANTAGES
    # The term subtracted from every return. ``None`` resolves to the choice
    # the run already implied before ``--baseline`` existed; see
    # ``training/rl/baseline.py``.
    baseline: Any = DEFAULT_BASELINE
    seed: int | None = None
    ppo_max_epochs: int = DEFAULT_PPO_MAX_EPOCHS


@dataclass(frozen=True)
class RLResourceOptions:
    """Model files, compute resources, and worker-tuning controls."""

    sl_weights_path: str | Path | None = None
    rl_weights_path: str | Path | None = None
    device: str = DEFAULT_DEVICE
    workers: int | str = DEFAULT_RL_WORKERS
    safety_config: ParallelSafetyConfig = field(
        default_factory=ParallelSafetyConfig
    )
    retune_workers: bool = False
    adaptive_tuning_training_games: int | None = None

    @property
    def artifact_directory(self):
        """Return the run directory inferred from the policy output path."""
        weights = Path(self.rl_weights_path)
        if weights.parent.name == "checkpoint_states":
            return weights.parent.parent
        return weights.parent

    @property
    def adaptive_tuning_path(self):
        """Derive the worker-tuning artifact path from the RL weights path."""
        return self.artifact_directory / "adaptive_tuning.json"

    @property
    def metrics_output_path(self):
        """Derive the per-iteration metrics path from the RL weights path."""
        weights = Path(self.rl_weights_path)
        if weights.parent.name == "checkpoint_states":
            return self.artifact_directory / "training_metrics.jsonl"
        return weights.with_name(f"{weights.stem}_training_metrics.jsonl")


@dataclass(frozen=True)
class RLExecutionOptions:
    """Invocation boundaries, resume state, reporting, and integration hooks."""

    log_interval: int = 10
    checkpoint_interval: int = 50
    moving_average_window: int = DEFAULT_MOVING_AVERAGE_WINDOW
    quiet: bool = False
    progress_callback: Callable | None = None
    status_callback: Callable | None = None
    metrics_callback: Callable | None = None
    checkpoint_callback: Callable | None = None
    resume_weights_path: str | Path | None = None
    resume_state_file: str | Path | None = None
    numbered_checkpoints: bool = False
    fresh_from_sl: bool = False
    stop_after_training_games: int | None = None
    shutdown_requested: Callable | None = None
    run_configuration: dict[str, Any] | None = None


@dataclass(frozen=True)
class ResolvedTrainingOptions:
    """Validated typed options needed before model loading or checkpoint I/O."""

    training: RLTrainingOptions
    resources: RLResourceOptions
    execution: RLExecutionOptions
    tuning_training_games: int
    algorithm: str
    schema: dict


def resolve_training_options(training, resources, execution):
    """Normalize and validate the three public RL option groups."""
    ruleset_name = resolve_ruleset(training.ruleset_name).name
    gpi = int(training.gpi)
    if gpi < 1:
        raise ValueError("gpi must be positive")
    total_training_games = training.total_training_games
    if training.iterations is not None:
        if training.iterations < 1:
            raise ValueError("iterations must be positive")
        implied_total = int(training.iterations) * gpi
        if (
            total_training_games is not None
            and int(total_training_games) != implied_total
        ):
            raise ValueError(
                "iterations * gpi conflicts with total_training_games"
            )
        total_training_games = implied_total
    else:
        total_training_games = (
            DEFAULT_TOTAL_TRAINING_GAMES
            if total_training_games is None
            else int(total_training_games)
        )
    if total_training_games < 1:
        raise ValueError("total_training_games must be positive")
    tuning_training_games = (
        int(total_training_games)
        if resources.adaptive_tuning_training_games is None
        else int(resources.adaptive_tuning_training_games)
    )
    if tuning_training_games < 1:
        raise ValueError("adaptive_tuning_training_games must be positive")
    if execution.checkpoint_interval < 1 or execution.log_interval < 1:
        raise ValueError("checkpoint_interval and log_interval must be positive")
    if execution.moving_average_window < 1:
        raise ValueError("moving_average_window must be positive")
    opponent_buckets = canonicalize_bucket_names(training.opponent_buckets)
    if not set(opponent_buckets) & set(BOOTSTRAP_CAPABLE_BUCKETS):
        raise ValueError(
            "opponent_buckets must select at least one bucket that is "
            "available from the first iteration ("
            + ", ".join(BOOTSTRAP_CAPABLE_BUCKETS)
            + "); the archive-backed bands cannot bootstrap their own history"
        )
    # A separate and stronger rule than the bootstrap check above: champion
    # candidates are the identities recent already holds, so without recent the
    # 50 pending snapshots would have no active weights left to race. The rule
    # is the same for every champion bucket, so the error names the ones this
    # selection actually asked for rather than one hard-coded bucket.
    missing_champion_requirements = [
        name
        for name in CHAMPION_REQUIRED_BUCKETS
        if name not in opponent_buckets
    ]
    selected_champions = [
        name for name in CHAMPION_BUCKET_NAMES if name in opponent_buckets
    ]
    if selected_champions and missing_champion_requirements:
        raise ValueError(
            ", ".join(selected_champions)
            + (" currently requires " if len(selected_champions) == 1
               else " currently require ")
            + ", ".join(missing_champion_requirements)
            + " so the candidate snapshots remain available for racing"
        )
    difficulty_weight = float(training.difficulty_weight)
    if not 0.0 <= difficulty_weight <= 1.0:
        raise ValueError("difficulty_weight must be between 0 and 1")
    gamma_f = float(training.gamma_f)
    if not 0.0 <= gamma_f <= 1.0:
        raise ValueError("gamma_f must be between 0 and 1")
    reward_eta = float(training.reward_eta)
    if not 0.0 <= reward_eta <= 1.0:
        raise ValueError("reward_eta must be between 0 and 1")
    gamma_i = float(training.gamma_i)
    if not 0.0 <= gamma_i <= 1.0:
        raise ValueError("gamma_i must be between 0 and 1")
    reward_distance_mode = str(training.reward_distance_mode)
    resolve_reward_distance_mode(reward_distance_mode)
    # Validated and normalized once here so every rollout worker receives the
    # derived scales instead of redividing by ``max(a, b)`` per event. The
    # raw weights are kept alongside them so a run records exactly what the
    # experiment asked for.
    reward_scales = resolved_reward_scales(
        terminal_empty_hand_weight=training.terminal_empty_hand_weight,
        terminal_blocked_weight=training.terminal_blocked_weight,
        immediate_draw_weight=training.immediate_draw_weight,
        immediate_pass_weight=training.immediate_pass_weight,
    )
    if float(training.value_coef) < 0:
        raise ValueError("value_coef must be non-negative")
    if float(training.weight_decay) < 0:
        raise ValueError("weight_decay must be non-negative")
    if not 0.0 <= float(training.dropout_rate) < 1.0:
        raise ValueError("dropout_rate must be at least 0 and below 1")
    ppo_max_epochs = validate_ppo_max_epochs(training.ppo_max_epochs)
    algorithm = (
        PPO_TRAINING_ALGORITHM
        if ppo_is_enabled(ppo_max_epochs)
        else REINFORCE_TRAINING_ALGORITHM
    )
    normalize_advantages = (
        ppo_is_enabled(ppo_max_epochs)
        if training.normalize_advantages is None
        else bool(training.normalize_advantages)
    )
    baseline = resolve_baseline(
        training.baseline,
        use_value_head=bool(training.use_value_head),
        normalize_advantages=normalize_advantages,
    )
    workers = resources.workers
    if workers != "auto":
        workers = int(workers)
        if not 1 <= workers <= MAX_PARALLEL_WORKERS:
            raise ValueError(
                f"workers must be 'auto' or between 1 and "
                f"{MAX_PARALLEL_WORKERS}"
            )
    resolved_training = replace(
        training,
        ruleset_name=ruleset_name,
        gpi=gpi,
        total_training_games=total_training_games,
        opponent_buckets=opponent_buckets,
        difficulty_weight=difficulty_weight,
        opponent_decision_restarts=bool(training.opponent_decision_restarts),
        gamma_f=gamma_f,
        reward_eta=reward_eta,
        gamma_i=gamma_i,
        reward_distance_mode=reward_distance_mode,
        terminal_empty_hand_weight=reward_scales[
            "terminal_empty_hand_weight"
        ],
        terminal_blocked_weight=reward_scales["terminal_blocked_weight"],
        immediate_draw_weight=reward_scales["immediate_draw_weight"],
        immediate_pass_weight=reward_scales["immediate_pass_weight"],
        normalize_advantages=normalize_advantages,
        baseline=baseline,
        ppo_max_epochs=ppo_max_epochs,
    )
    resolved_resources = replace(
        resources,
        sl_weights_path=(
            resources.sl_weights_path or default_sl_weights_path(ruleset_name)
        ),
        rl_weights_path=(
            resources.rl_weights_path or default_rl_weights_path(ruleset_name)
        ),
        workers=workers,
        retune_workers=bool(resources.retune_workers),
    )
    return ResolvedTrainingOptions(
        training=resolved_training,
        resources=resolved_resources,
        execution=execution,
        tuning_training_games=tuning_training_games,
        algorithm=algorithm,
        schema={
            **DEFAULT_REWARD_SCHEMA,
            **reward_scales,
            "gamma_i": gamma_i,
            "reward_eta": reward_eta,
            "reward_distance_mode": reward_distance_mode,
        },
    )
