# NVIDIA GPU Setup

The neural networks use NVIDIA CUDA through CuPy. A GPU is optional: the
project runs on NumPy/CPU when CuPy or a usable CUDA device is unavailable.

## What uses the GPU

- supervised forward/backward passes can use the GPU;
- the RL parent process can use the GPU for inference and gradient updates;
- dataset workers, RL rollout workers, diagnostics, and the exact opponent
  model intentionally remain CPU-only.

Worker subprocesses are prevented from opening CUDA contexts. This makes one
parent process the owner of GPU memory and avoids multiplying VRAM usage by the
worker count.

## 1. Verify the driver

On Linux, confirm that the machine sees an NVIDIA device and that the loaded
driver responds:

```bash
lspci | grep -i nvidia
nvidia-smi
```

Do not install CuPy until `nvidia-smi` works. The `CUDA Version` displayed by
`nvidia-smi` is the newest CUDA runtime supported by the loaded driver; it does
not prove that a system CUDA Toolkit or `nvcc` is installed.

On Ubuntu or Linux Mint, a typical driver installation is:

```bash
sudo apt update
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
```

Use the distribution's supported driver workflow when it differs from the
commands above.

## 2. Create and activate the project environment

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pygame PyOpenGL PyOpenGL-accelerate \
  matplotlib tqdm openpyxl pytest
which python
python -m pip --version
```

Both final paths should point inside this repository's `.venv`.

## 3. Install exactly one CuPy wheel family

Choose the wheel family matching the CUDA major version supported by the
driver:

```bash
# Driver supports CUDA 13.x
python -m pip install "cupy-cuda13x[ctk]"

# Driver supports CUDA 12.x
python -m pip install "cupy-cuda12x[ctk]"
```

The `[ctk]` extra installs CUDA runtime libraries in the virtual environment,
so a system-wide CUDA Toolkit and `nvcc` are not required. If a matching system
Toolkit is already installed and `nvcc --version` succeeds, the wheel without
`[ctk]` can be used instead.

Never install `cupy`, `cupy-cuda12x`, and `cupy-cuda13x` together. They expose
the same Python modules and conflict. Consult the
[CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) for the
current package matrix.

## 4. Verify CUDA with a real calculation

```bash
python - <<'PY'
import cupy as cp

print("CuPy:", cp.__version__)
print("CUDA runtime:", cp.cuda.runtime.runtimeGetVersion())
print("CUDA driver:", cp.cuda.runtime.driverGetVersion())
print("GPU count:", cp.cuda.runtime.getDeviceCount())
print("GPU:", cp.cuda.runtime.getDeviceProperties(0)["name"])
values = cp.arange(1_000_000, dtype=cp.float32)
print("GPU calculation:", float(cp.sum(values * values).get()))
free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
print(
    f"VRAM: {free_bytes / 1024**2:.1f} MiB free / "
    f"{total_bytes / 1024**2:.1f} MiB total"
)
PY
```

This verifies import, runtime/driver communication, allocation, kernel
execution, synchronization, device-to-host transfer, and memory reporting.

## 5. Verify project device selection

```bash
python - <<'PY'
from agents.nn import GPU_ENABLED, GPU_UNAVAILABLE_REASON

print("GPU enabled:", GPU_ENABLED)
print("Fallback reason:", GPU_UNAVAILABLE_REASON)
PY
```

Then run only a help or intentionally small workload first:

```bash
python -m training.training_loop --help
python -m training.self_play --help
python -m train_script.run_pipeline --help
```

Use `watch -n 1 nvidia-smi` in another terminal during real training to observe
VRAM and utilization.

## Device controls

The device values are:

- `auto`: select GPU only after a real synchronized backend probe succeeds and
  resource checks consider it safe; otherwise use CPU;
- `cpu`: force NumPy even when CuPy is installed;
- `gpu`: require CuPy and a usable CUDA device, failing instead of falling back.

Relevant commands include:

```bash
python -m training.training_loop --sl-device auto
python -m training.training_loop --device gpu
python -m training.self_play --device gpu
python -m train_script.run_pipeline --sl-device gpu --device gpu
```

The standalone supervised command preserves `--device` as an alias; the
pipeline uses `--sl-device` to distinguish supervised and RL selection.

Useful environment controls:

| Variable | Effect |
|---|---|
| `DOMINO_FORCE_CPU=1` | Makes project network code ignore CuPy. |
| `CUDA_VISIBLE_DEVICES` | Controls which NVIDIA devices are visible to CUDA. |
| `DOMINO_VRAM_LIMIT_MB=N` | Caps the process CuPy memory pool for shared-GPU experiments. |

Unset an accidental CPU restriction with:

```bash
unset DOMINO_FORCE_CPU
unset CUDA_VISIBLE_DEVICES
```

Do not globally set `CUDA_VISIBLE_DEVICES` to an empty value if the parent
training process is expected to use the GPU.

## Memory behavior

Supervised training benchmarks retained mini-batch candidates and can keep the
full encoded dataset on GPU or use bounded rotating windows. RL checks host
workspace and effective free VRAM before assembling large batches. Both use
float32 arrays and release disposable forward caches and unused CuPy pool
blocks at defined boundaries.

Controls include:

```bash
python -m training.training_loop \
  --sl-batch-size 1024 \
  --sl-memory-reserve-mb 1024 \
  --sl-gpu-memory-reserve-mb 1024

python -m training.self_play \
  --device gpu \
  --memory-reserve-mb 1024 \
  --gpu-memory-reserve-mb 1024
```

Check each command's `--help` because supervised and RL controls have different
names and ownership. Worker memory limits refer to CPU RAM, not VRAM.

## Troubleshooting

`ModuleNotFoundError: No module named 'cupy'`

- Reactivate `.venv`.
- Check `which python` and `python -m pip --version`.
- Install through `python -m pip`, not `sudo pip` or a bare `pip` tied to a
  different interpreter.

`cudaErrorInsufficientDriver`

- Update the NVIDIA driver or install a CuPy wheel/runtime compatible with the
  current driver.
- Recheck the driver with `nvidia-smi` after rebooting.

Missing `libcudart`, NVRTC, cuBLAS, or cuSPARSE libraries

- Reinstall the matching CuPy wheel with `[ctk]`, or correctly configure
  `CUDA_PATH` and `LD_LIBRARY_PATH` for an existing system Toolkit.

Warnings about multiple CuPy packages

```bash
python -m pip list | grep -i cupy
```

Uninstall all conflicting CuPy variants, then install exactly one matching
wheel family.

GPU is visible but the project selects CPU

- Read `GPU_UNAVAILABLE_REASON` using the verification snippet above.
- Check `DOMINO_FORCE_CPU` and `CUDA_VISIBLE_DEVICES`.
- Check free VRAM and whether another process owns most of the device.
- Retry with `--device gpu` only when a hard failure is preferable to automatic
  CPU fallback.

For driver and Toolkit details, see NVIDIA's
[Linux CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
and
[driver compatibility matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html).
