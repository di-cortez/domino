"""User-facing status messages for supervised training."""

from __future__ import annotations

from utils.resource_limits import (
    MIB,
    effective_gpu_available_bytes,
    gpu_memory_info,
)
from utils.runtime_status import format_duration, memory_report


def _format_optional_mib(byte_count):
    """Format an optional byte measurement for detailed resource logs."""
    if byte_count is None:
        return "unavailable"
    return f"{byte_count / MIB:.1f} MiB"


class SupervisedTrainingReporter:
    """Keep console presentation out of the supervised training workflow."""

    def __init__(self, quiet=False):
        self.quiet = bool(quiet)

    def startup(self):
        """Report memory before dataset loading and network allocation."""
        if not self.quiet:
            print(f"Supervised training startup memory: {memory_report()}")

    def runtime_configuration(
        self,
        *,
        selected_device,
        fallback_reason,
        data_plan,
        gpu_memory_reserve_mb,
        train_count,
        validation_count,
    ):
        """Report the selected device, residency mode, and data split."""
        if self.quiet:
            return
        print(f"Supervised device: {selected_device}")
        if fallback_reason:
            print(f"Automatic supervised CPU fallback: {fallback_reason}.")
        if selected_device == "gpu":
            gpu_info = gpu_memory_info()
            effective_free = effective_gpu_available_bytes()
            if gpu_info is not None:
                effective_text = (
                    "unknown"
                    if effective_free is None
                    else f"{effective_free / MIB:.1f} MiB"
                )
                print(
                    "GPU VRAM before supervised residency: "
                    f"{gpu_info.available / MIB:.1f} MiB free / "
                    f"{gpu_info.total / MIB:.1f} MiB total; "
                    f"{effective_text} effective free."
                )
            print(
                "GPU dataset residency: "
                f"{data_plan.resident_window_examples:,} examples; "
                f"mode={data_plan.storage_mode}; "
                f"reserve={gpu_memory_reserve_mb} MiB."
            )
            if data_plan.full_upload_seconds is not None:
                print(
                    "One-time full-dataset GPU upload: "
                    f"{data_plan.full_upload_seconds:.3f}s."
                )
        print(f"Split complete: {train_count} train | {validation_count} validation")

    def fresh_weights(self):
        """Report the deliberately fresh supervised initialization."""
        if not self.quiet:
            print("Supervised training starts from fresh random weights.")

    def batch_size(self, *, selected, requested):
        """Report the fixed batch size and any dataset-size cap."""
        if self.quiet:
            return
        print(
            f"Supervised batch size: {selected:,}"
            + (
                f" (requested {requested:,}; capped to training set)."
                if selected != requested
                else "."
            )
        )

    def training_started(self):
        """Mark the beginning of epoch updates."""
        if not self.quiet:
            print("\nStarting supervised training...")

    def new_best(self, *, epoch, validation_loss):
        """Report a newly retained validation checkpoint."""
        if not self.quiet:
            print(
                "  -> New best validation loss "
                f"{validation_loss:.4f} at epoch {epoch}."
            )

    def completion(
        self,
        *,
        weights_file,
        random_manifest_path,
        loss_plot_path,
        best_validation_loss,
        storage_mode,
        epoch_metrics,
        resource_summary,
        elapsed,
    ):
        """Report output artifacts, bounded resources, and total duration."""
        if self.quiet:
            return
        best_text = (
            "unavailable"
            if best_validation_loss == float("inf")
            else f"{best_validation_loss:.4f}"
        )
        print(f"Model saved to {weights_file} (best validation loss: {best_text}).")
        print(f"Random manifest saved to {random_manifest_path}.")
        print(f"Loss graph saved to {loss_plot_path}.")
        if storage_mode == "gpu_windowed":
            rotations = [row["window_rotations"] for row in epoch_metrics]
            print(
                "GPU window rotations: "
                f"{sum(rotations)} total across {len(rotations)} epochs "
                f"({min(rotations)}-{max(rotations)} per epoch)."
            )
        print(
            "Supervised resource bounds: "
            "peak host RSS="
            f"{_format_optional_mib(resource_summary['peak_host_rss_bytes'])}; "
            "minimum available host RAM="
            f"{_format_optional_mib(resource_summary['minimum_available_host_ram_bytes'])}; "
            "peak CuPy pool="
            f"{_format_optional_mib(resource_summary['peak_gpu_pool_used_bytes'])}; "
            "minimum effective free VRAM="
            f"{_format_optional_mib(resource_summary['minimum_effective_free_vram_bytes'])}."
        )
        print(f"Total elapsed time: {format_duration(elapsed)}.")
