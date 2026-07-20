"""Cgroup-aware RAM/VRAM probes and lightweight allocation safeguards.

The diagnostic scheduler and the training entry points use this module instead
of assuming that all host memory is available to the current process.  Linux
cgroup limits are folded into the host figures when present.  The
``DOMINO_TEST_*`` overrides exist solely so low-memory behavior can be tested on
machines that are not actually under pressure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


MIB = 1024 * 1024
GIB = 1024 * MIB


class MemorySafetyError(MemoryError):
    """Raised before an allocation that would consume the configured reserve."""


@dataclass(frozen=True)
class MemoryInfo:
    """Total, used, and available memory in bytes."""

    total: int
    used: int
    available: int
    source: str
    simulated: bool = False


def _positive_env_mb(name: str) -> int | None:
    """Read a non-negative MiB environment override as an integer byte count."""
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number of MiB, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return int(parsed * MIB)


def _read_proc_meminfo() -> tuple[int, int] | None:
    """Return Linux host ``(total, available)`` RAM bytes when readable."""
    try:
        values = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as stream:
            for line in stream:
                key, raw_value = line.split(":", 1)
                values[key] = int(raw_value.strip().split()[0]) * 1024
        total = values["MemTotal"]
        available = values.get("MemAvailable", values.get("MemFree", 0))
        return total, max(0, available)
    except (OSError, KeyError, ValueError):
        return None


def _read_cgroup_memory() -> tuple[int, int] | None:
    """Return cgroup ``(limit, available)`` for v2 or v1, when bounded."""
    candidates = (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    )
    for limit_path, current_path in candidates:
        try:
            raw_limit = limit_path.read_text(encoding="utf-8").strip()
            if raw_limit == "max":
                continue
            limit = int(raw_limit)
            current = int(current_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        # Some cgroup-v1 hosts expose an effectively unlimited huge integer.
        if limit <= 0 or limit >= (1 << 60):
            continue
        return limit, max(0, limit - current)
    return None


def system_memory_info() -> MemoryInfo | None:
    """Return host/cgroup-aware system memory information."""
    simulated_available = _positive_env_mb("DOMINO_TEST_AVAILABLE_RAM_MB")
    simulated_total = _positive_env_mb("DOMINO_TEST_TOTAL_RAM_MB")
    if simulated_available is not None:
        total = simulated_total or max(simulated_available, GIB)
        available = min(simulated_available, total)
        return MemoryInfo(
            total=total,
            used=total - available,
            available=available,
            source="test override",
            simulated=True,
        )

    host = _read_proc_meminfo()
    cgroup = _read_cgroup_memory()
    if host is None and cgroup is None:
        return None
    if host is None:
        total, available = cgroup
        source = "cgroup"
    elif cgroup is None:
        total, available = host
        source = "/proc/meminfo"
    else:
        host_total, host_available = host
        cgroup_total, cgroup_available = cgroup
        total = min(host_total, cgroup_total)
        available = min(host_available, cgroup_available, total)
        source = "/proc/meminfo+cgroup"
    return MemoryInfo(total=total, used=total - available, available=available, source=source)


def available_ram_mb() -> float | None:
    """Return safely available RAM in MiB."""
    info = system_memory_info()
    return None if info is None else info.available / MIB


def process_rss_bytes(pid: int | None = None) -> int | None:
    """Return RSS for ``pid`` (or the current process) on Linux."""
    statm_path = "/proc/self/statm" if pid is None else f"/proc/{int(pid)}/statm"
    try:
        with open(statm_path, "r", encoding="utf-8") as stream:
            pages = int(stream.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        return None


def gpu_memory_info() -> MemoryInfo | None:
    """Return CUDA memory information, supporting deterministic test overrides."""
    simulated_free = _positive_env_mb("DOMINO_TEST_GPU_FREE_MB")
    simulated_total = _positive_env_mb("DOMINO_TEST_GPU_TOTAL_MB")
    if simulated_free is not None:
        total = simulated_total or max(simulated_free, GIB)
        available = min(simulated_free, total)
        return MemoryInfo(
            total=total,
            used=total - available,
            available=available,
            source="test override",
            simulated=True,
        )

    try:
        import cupy

        free_bytes, total_bytes = cupy.cuda.runtime.memGetInfo()
        return MemoryInfo(
            total=int(total_bytes),
            used=int(total_bytes - free_bytes),
            available=int(free_bytes),
            source="CuPy/CUDA",
        )
    except Exception:
        return None


def effective_gpu_available_bytes() -> int | None:
    """Return reusable VRAM after applying the optional CuPy pool ceiling."""
    info = gpu_memory_info()
    pool_limit = _positive_env_mb("DOMINO_VRAM_LIMIT_MB")
    available = None if info is None else info.available
    pool_used = 0
    pool_cached = 0
    if info is not None and not info.simulated:
        try:
            import cupy

            pool = cupy.get_default_memory_pool()
            pool_used = int(pool.used_bytes())
            pool_cached = max(0, int(pool.total_bytes()) - pool_used)
            available += pool_cached
        except Exception:
            pass
    if pool_limit is not None:
        pool_headroom = max(0, pool_limit - pool_used)
        available = (
            pool_headroom
            if available is None
            else min(available, pool_headroom)
        )
    return available


def host_allocation_status(
    required_bytes: int,
    reserve_mb: int,
) -> tuple[bool, dict]:
    """Return a cgroup-aware host-allocation decision and its measurements."""
    info = system_memory_info()
    if info is None:
        return True, {
            "required_bytes": int(required_bytes),
            "reserve_mb": int(reserve_mb),
            "available_bytes": None,
            "safe": True,
            "source": None,
        }
    usable = max(0, info.available - int(reserve_mb * MIB))
    safe = int(required_bytes) <= usable
    return safe, {
        "required_bytes": int(required_bytes),
        "reserve_mb": int(reserve_mb),
        "available_bytes": int(info.available),
        "safe": safe,
        "source": info.source,
    }


def gpu_allocation_status(
    required_bytes: int,
    reserve_mb: int,
) -> tuple[bool, dict]:
    """Return an effective-VRAM allocation decision and its measurements."""
    available = effective_gpu_available_bytes()
    if available is None:
        return False, {
            "required_bytes": int(required_bytes),
            "reserve_mb": int(reserve_mb),
            "available_bytes": None,
            "safe": False,
        }
    usable = max(0, available - int(reserve_mb * MIB))
    safe = int(required_bytes) <= usable
    return safe, {
        "required_bytes": int(required_bytes),
        "reserve_mb": int(reserve_mb),
        "available_bytes": int(available),
        "safe": safe,
    }


def ensure_ram_available(required_bytes: int, reserve_mb: int, context: str) -> None:
    """Fail early if ``required_bytes`` would consume the RAM reserve."""
    info = system_memory_info()
    if info is None:
        return
    usable = max(0, info.available - int(reserve_mb * MIB))
    if required_bytes > usable:
        raise MemorySafetyError(
            f"{context} needs about {required_bytes / MIB:.1f} MiB, but only "
            f"{info.available / MIB:.1f} MiB is available with a "
            f"{reserve_mb} MiB reserve ({info.source})."
        )


def choose_safe_rl_device(
    requested_device: str,
    minimum_free_vram_mb: int = 256,
) -> tuple[str, str | None]:
    """Return a safe RL device, falling back only when ``auto`` selected GPU.

    This is deliberately a preflight decision.  Migrating a partially trained
    CuPy network to CPU after an allocator failure would silently change an
    iteration and is therefore avoided.
    """
    if requested_device not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"Unknown device {requested_device!r}")
    if requested_device == "cpu":
        return "cpu", None

    effective_free = effective_gpu_available_bytes()

    required = int(minimum_free_vram_mb * MIB)
    if effective_free is not None and effective_free < required:
        reason = (
            f"only {effective_free / MIB:.1f} MiB effective VRAM is available; "
            f"the RL safety minimum is {minimum_free_vram_mb} MiB"
        )
        if requested_device == "gpu":
            raise MemorySafetyError(f"Cannot honor device='gpu': {reason}.")
        return "cpu", reason
    return requested_device, None


def choose_safe_supervised_device(
    requested_device: str,
    minimum_free_vram_mb: int = 512,
) -> tuple[str, str | None]:
    """Resolve a supervised CPU/GPU request before any training update."""
    if requested_device not in {"auto", "cpu", "gpu"}:
        raise ValueError(f"Unknown device {requested_device!r}")
    if requested_device == "cpu":
        return "cpu", None

    if os.environ.get("DOMINO_FORCE_CPU") == "1":
        reason = "CPU is forced by DOMINO_FORCE_CPU"
        if requested_device == "gpu":
            raise MemorySafetyError(f"Cannot honor device='gpu': {reason}.")
        return "cpu", reason

    try:
        import cupy

        if int(cupy.cuda.runtime.getDeviceCount()) < 1:
            raise RuntimeError("no CUDA device is visible")
    except Exception as exc:
        reason = f"CuPy/CUDA is unavailable ({type(exc).__name__}: {exc})"
        if requested_device == "gpu":
            raise MemorySafetyError(f"Cannot honor device='gpu': {reason}.")
        return "cpu", reason

    effective_free = effective_gpu_available_bytes()
    required = int(minimum_free_vram_mb * MIB)
    if effective_free is None or effective_free < required:
        available_text = (
            "unknown"
            if effective_free is None
            else f"{effective_free / MIB:.1f} MiB"
        )
        reason = (
            f"only {available_text} effective VRAM is available; the "
            f"supervised safety minimum is {minimum_free_vram_mb} MiB"
        )
        if requested_device == "gpu":
            raise MemorySafetyError(f"Cannot honor device='gpu': {reason}.")
        return "cpu", reason
    return "gpu", None
