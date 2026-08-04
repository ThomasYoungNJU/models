# Multi-Platform User Guide

Every model under `ai4s/` must run **training and inference** on all six supported
accelerator families. This guide gives step-by-step setup and run instructions for
each. Follow the section for your hardware, then run the standard train/infer entry
points described in the model's own `README.md`.

## Supported hardware

| # | Vendor | Product line | Software stack | PyTorch backend / plugin | Device string |
|---|--------|--------------|----------------|--------------------------|---------------|
| 1 | NVIDIA | GPU (A100/H100/RTX…) | CUDA + cuDNN | native `torch` (CUDA build) | `cuda` |
| 2 | Huawei 华为 | Ascend 昇腾 (910B…) | CANN | `torch` + `torch_npu` | `npu` |
| 3 | MetaX 沐曦 | GPU (C500…) | MACA | `torch` + `torch_maca` | `cuda` (via MACA) |
| 4 | Hygon 海光 | DCU (Z100/K100…) | DTK (ROCm-compatible) | `torch` (ROCm build) | `cuda` (HIP) |
| 5 | T-Head 平头哥 | PPU-ZW810E | HGGC CUDA-compatible bridge | vendor `torch` build | `cuda` |
| 6 | Moore Threads 摩尔线程 | MTT S-series | MUSA | `torch` + `torch_musa` | `musa` |

> The exact package versions depend on the driver/firmware installed on your node.
> Always match the framework build to the driver version reported by the vendor tool
> (see "Verify the device" in each section).

---

## Common prerequisites

- Python 3.10 (recommended). Use a per-platform virtual environment or conda env.
- Git and the model repository checked out under `ai4s/<model>/`.
- A clean env per platform — do **not** mix vendor PyTorch builds in one env.

```bash
cd ai4s/<model>
python -m venv .venv-<platform>      # e.g. .venv-nvidia
source .venv-<platform>/bin/activate
pip install --upgrade pip
```

Install the model's own Python deps **after** installing the platform framework:

```bash
pip install -r requirements.txt   # model deps; keep torch/vendor plugins un-pinned here
```

---

## 1. NVIDIA (CUDA)

**Verify the device**

```bash
nvidia-smi                 # driver + GPU visible
```

**Install framework**

```bash
# Pick the CUDA tag matching your driver (example: CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Sanity check**

```python
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
```

**Run** — device string `cuda`:

```bash
bash scripts/train.sh --device cuda
bash scripts/infer.sh --device cuda --checkpoint checkpoints/best.pt
```

---

## 2. Huawei Ascend 昇腾 (CANN + torch_npu)

**Verify the device**

```bash
npu-smi info               # shows Ascend chips, health, memory
```

**Install framework**

```bash
# 1) Install the CANN toolkit that matches your firmware (vendor package).
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 2) Install matching torch + torch_npu (versions must be paired).
pip install torch==2.1.0
pip install torch_npu==2.1.0        # version must match the CPU torch above
```

**Sanity check**

```python
import torch, torch_npu
assert torch.npu.is_available()
print(torch.npu.get_device_name(0))
```

**Run** — device string `npu`:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/train.sh --device npu
bash scripts/infer.sh --device npu --checkpoint checkpoints/best.pt
```

> In code, `import torch_npu` and move tensors with `.to("npu")`. Set
> `ASCEND_RT_VISIBLE_DEVICES` to select chips.

---

## 3. MetaX 沐曦 (MACA + torch_maca)

**Verify the device**

```bash
mx-smi                     # MetaX system management interface
```

**Install framework**

```bash
# Install the MACA SDK from MetaX, then source its environment:
source /opt/maca/env.sh
# Install the MetaX-provided torch + torch_maca wheels:
pip install torch torch_maca   # use the wheels shipped in the MACA release
```

**Sanity check**

```python
import torch, torch_maca
assert torch.cuda.is_available()   # MACA exposes the CUDA API surface
print(torch.cuda.get_device_name(0))
```

**Run** — MetaX presents the CUDA API, so use device string `cuda`:

```bash
source /opt/maca/env.sh
bash scripts/train.sh --device cuda
bash scripts/infer.sh --device cuda --checkpoint checkpoints/best.pt
```

> Set `MACA_VISIBLE_DEVICES` to select cards. Existing CUDA model code runs
> unchanged; only the environment and wheels differ.

---

## 4. Hygon 海光 DCU (DTK / ROCm)

**Verify the device**

```bash
rocm-smi                   # or: hy-smi
```

**Install framework**

```bash
# Install the Hygon DTK toolkit, then source it:
source /opt/dtk/env.sh
# Install the ROCm-compatible torch build from Hygon:
pip install torch --index-url <hygon-dcu-wheel-index>
```

**Sanity check**

```python
import torch
assert torch.cuda.is_available()   # ROCm/HIP exposes the CUDA device API
print(torch.cuda.get_device_name(0))
```

**Run** — HIP maps onto the CUDA API, so use device string `cuda`:

```bash
source /opt/dtk/env.sh
bash scripts/train.sh --device cuda
bash scripts/infer.sh --device cuda --checkpoint checkpoints/best.pt
```

> Select cards with `HIP_VISIBLE_DEVICES`.

---

## 5. T-Head 平头哥 PPU-ZW810E (HGGC PyTorch bridge)

The validated T-Head target is PPU-ZW810E. Its vendor framework exposes a
CUDA-compatible PyTorch API and supports both training and inference. It is not
the inference-only HanGuang 800 / HanGuangRT route described in older drafts.

**Verify the device**

```python
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
```

**Install framework**

Use the PPU vendor image or matching HGGC PyTorch wheels supplied for the
installed driver. Do not replace that build with a generic CUDA wheel.

**Run** — the bridge exposes device string `cuda`:

```bash
bash scripts/train.sh --platform thead --device 0
bash scripts/infer.sh --platform thead --device 0
```

> Select cards with `CUDA_VISIBLE_DEVICES`. Confirm the device name reports
> PPU-ZW810E before starting a long job.

---

## 6. Moore Threads 摩尔线程 (MUSA + torch_musa)

**Verify the device**

```bash
mthreads-gmi               # Moore Threads GPU management interface
```

**Install framework**

```bash
# Install the MUSA SDK, then source its environment:
source /usr/local/musa/env.sh
# Install matching torch + torch_musa (versions must be paired):
pip install torch==2.2.0
pip install torch_musa        # use the wheel matching the torch version above
```

**Sanity check**

```python
import torch, torch_musa
assert torch.musa.is_available()
print(torch.musa.get_device_name(0))
```

**Run** — device string `musa`:

```bash
source /usr/local/musa/env.sh
bash scripts/train.sh --device musa
bash scripts/infer.sh --device musa --checkpoint checkpoints/best.pt
```

> In code, `import torch_musa` and move tensors with `.to("musa")`. Select cards
> with `MUSA_VISIBLE_DEVICES`.

---

## Writing portable model code

To pass on all six platforms with one codebase, resolve the device at runtime
instead of hard-coding `cuda`:

```python
import argparse

def resolve_device(name: str):
    import torch
    if name == "npu":
        import torch_npu  # noqa: F401
        return "npu"
    if name == "musa":
        import torch_musa  # noqa: F401
        return "musa"
    if name in ("maca", "dcu", "thead"):
        # MACA and Hygon DCU both expose the CUDA API surface
        return "cuda"
    return name  # "cuda", "cpu", or another supported backend

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda")
args = parser.parse_args()
device = resolve_device(args.device)
```

Guidelines:

- Take `--device` as an argument in `train.sh` / `infer.sh`; never hard-code the backend.
- Import the vendor plugin (`torch_npu` / `torch_musa`) lazily, only when selected.
- Avoid CUDA-only ops/kernels; if unavoidable, provide a fallback path.
- Document the actual T-Head runtime used by the model; the validated PPU target uses the CUDA-compatible bridge.
- Pin the model's own deps in `requirements.txt`, but leave `torch`/vendor plugins
  to be installed per platform (they are not portable across vendors).

---

## Acceptance checklist (per model)

A model is "runs on all platforms" only when every box is checked:

- [ ] NVIDIA: `train.sh` and `infer.sh` complete; loss decreases; outputs match reference.
- [ ] Huawei Ascend: same, with `--device npu`.
- [ ] MetaX: same, with the MACA env + `--device cuda`.
- [ ] Hygon DCU: same, with the DTK env + `--device cuda`.
- [ ] T-Head PPU-ZW810E: `train.sh` and `infer.sh` complete with the HGGC PyTorch bridge; outputs match reference.
- [ ] Moore Threads: same as NVIDIA, with `--device musa`.
- [ ] `requirements.txt` installs cleanly in a fresh env on each platform.
- [ ] The model's `README.md` documents any platform-specific caveats.

> The exact command names (`mx-smi`, wheel index URLs, env script
> paths) come from each vendor's SDK release notes. Confirm them against the SDK
> installed on your node before running.
