"""Loss-curve paths, scaling, and atomic rendering for supervised training."""

from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np


def supervised_loss_plot_path(weights_file):
    """Return the loss-plot path beside one supervised checkpoint."""
    weights_path = Path(weights_file)
    stem = weights_path.stem
    if stem.endswith("_weights"):
        stem = stem.removesuffix("_weights")
    return weights_path.with_name(f"{stem}_loss.png")


def _supervised_loss_axis_limits(loss_history, validation_points):
    """Frame the visible loss range around the observed supervised curves."""
    training_losses = [
        float(loss) for loss in loss_history if np.isfinite(float(loss))
    ]
    validation_losses = [
        float(loss)
        for _epoch, loss in validation_points
        if np.isfinite(float(loss))
    ]
    if not training_losses:
        raise ValueError("Cannot scale a graph without finite training losses.")

    final_training_loss = training_losses[-1]
    maximum_loss = max(training_losses + validation_losses)
    observed_drop = max(0.0, maximum_loss - final_training_loss)
    scale = max(abs(maximum_loss), abs(final_training_loss), 1e-6)
    lower_padding = max(observed_drop * 0.10, scale * 0.005, 1e-6)
    lower_limit = final_training_loss - lower_padding
    if maximum_loss <= lower_limit:
        lower_limit = maximum_loss - max(scale * 0.01, 1e-6)
    return lower_limit, maximum_loss


def save_supervised_loss_plot(loss_history, epoch_metrics, weights_file):
    """Atomically plot current-run training and validation cross-entropy loss."""
    if not loss_history:
        raise ValueError("Cannot plot an empty supervised loss history.")

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output_path = supervised_loss_plot_path(weights_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.tmp-{os.getpid()}-{time.time_ns()}.png"
    )

    figure = Figure(figsize=(9.0, 5.25), facecolor="#263b34")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.set_facecolor("#263b34")

    epochs = list(range(1, len(loss_history) + 1))
    axis.plot(
        epochs,
        [float(loss) for loss in loss_history],
        color="#f1d36b",
        linewidth=2.2,
        label="Training loss",
    )
    validation_points = [
        (int(metrics["epoch"]) + 1, float(metrics["validation_loss"]))
        for metrics in epoch_metrics
        if metrics.get("validation_loss") is not None
        and np.isfinite(float(metrics["validation_loss"]))
    ]
    if validation_points:
        validation_epochs, validation_losses = zip(*validation_points)
        axis.plot(
            validation_epochs,
            validation_losses,
            color="#d7eee4",
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            label="Validation loss",
        )

    axis.set_title("Supervised Training Loss", color="#f4f0df", pad=12)
    axis.set_xlabel("Epoch", color="#f4f0df")
    axis.set_ylabel("Cross-entropy loss", color="#f4f0df")
    lower_limit, upper_limit = _supervised_loss_axis_limits(
        loss_history,
        validation_points,
    )
    axis.set_ylim(lower_limit, upper_limit)
    axis.grid(color="#81978d", alpha=0.25, linewidth=0.8)
    axis.tick_params(colors="#f4f0df")
    for spine in axis.spines.values():
        spine.set_color("#b7c5bd")
    legend = axis.legend(frameon=False)
    if legend is not None:
        for text in legend.get_texts():
            text.set_color("#f4f0df")
    figure.tight_layout()

    try:
        figure.savefig(
            temporary_path,
            format="png",
            dpi=150,
            facecolor=figure.get_facecolor(),
        )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
        figure.clear()
    return output_path
