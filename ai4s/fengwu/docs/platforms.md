# Platform environments and FlagGems policy

The versions below are reference environments used for acceptance, not package
pins. A vendor framework must match the installed accelerator driver/toolkit.

| Platform | Verified accelerator | Reference Python / PyTorch | Device selection | FlagGems |
|---|---|---|---|---|
| NVIDIA | H100 80 GB | Python 3.12 / PyTorch 2.10 NVIDIA build | `CUDA_VISIBLE_DEVICES` | disabled in `auto` |
| Huawei | Ascend 910C 64 GB | Python 3.11 / PyTorch 2.8 + torch_npu 2.8 | `ASCEND_RT_VISIBLE_DEVICES` | 5.3.0rc2 |
| Hygon | BW1000 64 GB | Python 3.10 / PyTorch 2.10 DTK build | `HIP_VISIBLE_DEVICES` | 5.0.2 |
| MetaX | C550 64 GB | Python 3.12 / PyTorch 2.8 MetaX build | `CUDA_VISIBLE_DEVICES` | 5.0.2 |
| Moore Threads | MTT S5000 80 GB | Python 3.10 / PyTorch 2.5 + torch_musa 2.5 | `MUSA_VISIBLE_DEVICES` | 5.0.2 |
| T-Head | PPU-ZW810E 96 GB | Python 3.12 / PyTorch 2.10 HGGC build | `CUDA_VISIBLE_DEVICES` | 5.0.2 |

The T-Head acceptance target is PPU-ZW810E through its PyTorch CUDA-compatible
bridge. Both training and inference were exercised. It is not the inference-only
HanGuang 800/ONNX path described by older platform documentation.

## Framework setup

1. Install or enter the vendor-provided framework image/environment.
2. Verify the accelerator with the vendor management tool.
3. Verify the matching PyTorch extension (`torch_npu` or `torch_musa` where
   applicable).
4. Install `requirements.txt` without replacing the vendor torch/torchvision.
5. Install the vendor-compatible FlagGems distribution when required.

Hygon environments must load the DTK environment before Python. In the verified
image this was:

```bash
source /opt/dtk-26.04-DCC2602-0317/env.sh
```

When `/opt/dtk` exists, the runtime normalizes `ROCM_PATH=/opt/dtk` before
importing FlagGems so the expected compiler is selected.

MetaX environments may require the vendor Python explicitly, for example
`/opt/conda/bin/python3`. Use `--python` for inference or place that Python first
on `PATH` for training.

Ascend runs default to:

```bash
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

This avoids an address-out-of-range failure observed during activation-
checkpoint recomputation in the accepted 910C FlagGems environment.

## FlagGems exclusions

`FENGWU_FLAGGEMS_UNUSED` appends operators; it does not replace the safe defaults.

### Training

Common exclusions on non-NVIDIA platforms:

```text
conv2d, mean, mean_dim, layer_norm, layer_norm_backward, add_, slice_backward
```

Additional exclusions:

| Platform | Additional operators |
|---|---|
| Hygon | `cat`, `stack` |
| MetaX | `cat`, `stack`, `true_divide_` |
| Moore Threads | `cat`, `stack`, `addmm`, `mm` |
| T-Head | `cat`, `stack` |
| Huawei Ascend | none beyond the common list |

MetaX enables FlagGems after DataLoader and model construction; the other
accepted non-NVIDIA paths enable it before dependency/model imports.

### Inference

Common exclusions on non-NVIDIA platforms:

```text
conv2d, mul, add
```

Additional exclusions:

| Platform | Additional operators |
|---|---|
| Hygon | `cat`, `stack` |
| MetaX | `cat`, `stack`, `true_divide_` |
| Moore Threads | `cat`, `stack`, `addmm`, `mm` |
| T-Head | `cat`, `stack` |
| Huawei Ascend | none beyond the common list |

## Container device caveats

The public `scripts/train.sh` and `scripts/infer.sh` run inside the prepared
environment. They set the appropriate visibility variable before Python starts.

For Moore Threads Docker deployments, the physical device is often selected at
container creation with `MUSA_VISIBLE_DEVICES`. `docker exec -e` cannot grant a
device that the container runtime did not expose. Use the physical card bound
at creation and logical device 0 inside the container.

The legacy `applepen.sh` wrappers support background `docker exec`, but container
names and mount paths are deployment-specific. Set `FENGWU_CONTAINER` and
`FENGWU_CONTAINER_WORKDIR` explicitly in a new environment.

