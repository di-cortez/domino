"""Small runtime status helpers for memory and elapsed-time logging."""

from __future__ import annotations

from utils.resource_limits import gpu_memory_info, process_rss_bytes, system_memory_info


def format_duration(seconds):
    """Return a compact human-readable duration string."""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, remaining_seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remaining_seconds:.1f}s"

    hours, remaining_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.1f}s"


def _format_bytes(byte_count):
    """Return a compact binary-size string."""
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024


def _process_rss_bytes():
    """Return current process RSS on Linux, or ``None`` when unavailable."""
    return process_rss_bytes()


def _system_memory_bytes():
    """Return ``(used, total)`` system RAM bytes from Linux meminfo."""
    info = system_memory_info()
    return None if info is None else (info.used, info.total)


def _gpu_memory_bytes():
    """Return CuPy/CUDA memory info, or ``None`` when no GPU backend is available."""
    info = gpu_memory_info()
    if info is None:
        return None
    pool_used = 0
    pool_total = 0
    try:
        import cupy

        pool = cupy.get_default_memory_pool()
        pool_used = pool.used_bytes()
        pool_total = pool.total_bytes()
    except Exception:
        pass
    return {
        "used": info.used,
        "free": info.available,
        "total": info.total,
        "pool_used": pool_used,
        "pool_total": pool_total,
    }


def memory_report():
    """Return a one-line RAM/GPU memory report for startup logs."""
    parts = []

    process_rss = _process_rss_bytes()
    if process_rss is not None:
        parts.append(f"process RAM {_format_bytes(process_rss)}")

    system_memory = _system_memory_bytes()
    if system_memory is not None:
        used, total = system_memory
        parts.append(f"system RAM {_format_bytes(used)} / {_format_bytes(total)} used")

    gpu_memory = _gpu_memory_bytes()
    if gpu_memory is None:
        parts.append("GPU memory unavailable")
    else:
        parts.append(
            "GPU memory "
            f"{_format_bytes(gpu_memory['used'])} / {_format_bytes(gpu_memory['total'])} used "
            f"({_format_bytes(gpu_memory['free'])} free, "
            f"CuPy pool {_format_bytes(gpu_memory['pool_used'])} / "
            f"{_format_bytes(gpu_memory['pool_total'])})"
        )

    return " | ".join(parts)


def print_memory_report(label):
    """Print a labelled memory report."""
    print(f"{label}: {memory_report()}")
