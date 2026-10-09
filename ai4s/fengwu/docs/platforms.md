# Platform environments and FlagGems policy

The versions below are reference environments used for acceptance, not package
pins. A vendor framework must match the installed accelerator driver/toolkit.

| Platform | Verified accelerator | Reference Python / PyTorch | Device selection | FlagGems |
|---|---|---|---|---|
| NVIDIA | H100 80 GB | Python 3.12 / PyTorch 2.10 NVIDIA build | `CUDA_VISIBLE_DEVICES` | disabled in `auto` |
| Huawei | Ascend 910C, one Chip / 64 GB | Python 3.11 / PyTorch 2.8 + torch_npu 2.8 | `ASCEND_RT_VISIBLE_DEVICES` | 5.3.0rc2 |
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

## FlagGems exclusions and startup verification

Exclusions are **FlagGems implementation function names**, not necessarily ATen
API names (`mean_dim`, not `mean.dim`; `softmax`, not `_softmax`).
Edit `utils/flaggems_runtime.py` in the corresponding training/inference tree.
`FENGWU_FLAGGEMS_UNUSED` appends entries. Unknown environment-supplied names fail
fast; names absent from the installed build are reported explicitly.

| Platform | Training exclusions | Inference exclusions |
|---|---|---|
| Huawei Ascend | `conv2d`, `layer_norm_backward`, `slice_backward` | `conv2d`, `add` |
| Hygon | `conv2d`, `layer_norm_backward`, `cat`, `stack`, `slice_backward` | `cat` |
| MetaX | `conv2d`, `layer_norm_backward`, `slice_backward` | none |
| T-Head | `conv2d`, `layer_norm_backward`, `slice_backward` | none |
| Moore Threads | `conv2d`, `layer_norm_backward`, `slice_backward`, `mm` | `addmm`, `mm` |

These are current project policies, not a claim that all upstream versions fail
for every excluded operator. The five lists were checked against the actual
deployment source and recent run logs on **2026-10-08**, not inferred from older
profiling exclusions. They list explicit project exclusions only, not backend
registration filters. MetaX enables after model/DataLoader construction.
Domestic-platform launchers require FlagGems by default. Import failure, vendor
or device mismatch, or incomplete registration aborts instead of falling back
silently. Explicit `--flaggems off` is still available for native comparisons;
it is **not** an accelerated-project acceptance run.

`[RuntimeEvidence]` records interpreter/package paths, successful dispatch keys,
exclusions and code hashes. Registration checks alone do not prove every model
call reaches a FlagGems kernel; use an instrumented short test for that evidence.
Implicit CPU fallback is forbidden; explicit CPU debug runs require
`--platform cpu --flaggems off`.

### Nominal FP32 acceleration

`FENGWU_FP32_ACCEL=on` sets and reads back native Torch TF32/HF32-style switches;
`off` sets them false; `auto` preserves and reports their initial values. These
are process-local, not shared driver changes. Unsupported controls are reported
as `partial_or_unsupported`, and do not turn FlagGems off.

**These switches do not override FlagGems kernels with internal
`allow_tf32=False`.** A readback of True is not proof that all matrix operations
use TF32/HF32. AMP/BF16 is not enabled by this control.


Domestic-platform launchers default to `FENGWU_FP32_ACCEL=auto`: preserve and
report the native Torch setting without forcing TF32/HF32 on or off. Adequate
FlagGems use, measured performance and numerical acceptance determine success;
enabling TF32/HF32 is not required. This does not change the NVIDIA baseline.

### Timing boundaries

Training host-enqueue spans end at Python return; completed device Event spans
include stream gaps/host launch delays and are not sums of kernel-busy time.
Timing adds no per-stage synchronization. Inference `Inference loop total`
includes model computation, waits and per-step D2H/NumPy conversion, but not
final NetCDF output. Python total ends at the final measurement point; it does
not include SSH/Docker startup or interpreter teardown. Do not add parent and
child timing blocks together.

## Container device caveats

The verified Thead training path applies a process-local launch configuration
to the existing FlagGems `mm_kernel_general`: 16x32x32 tiles, 4 warps, 1 stage.
This avoids pathological compiler/autotuner configurations; it does not exclude
MM or replace it with native Torch. The original kernel and precision controls
remain unchanged. The configuration was checked against 50 observed shape/stride
pairs and one complete short training epoch, not a full long-run acceptance test.
Unsupported tuner structures fail explicitly. Other platforms and Thead
inference are not given this configuration.

The public `scripts/train.sh` and `scripts/infer.sh` run inside the prepared
environment. They set the appropriate visibility variable before Python starts.

For Moore Threads Docker deployments, the physical device is often selected at
container creation with `MUSA_VISIBLE_DEVICES`. `docker exec -e` cannot grant a
device that the container runtime did not expose. Use the physical card bound
at creation and logical device 0 inside the container.

For an Ascend container exposing only one physical Chip, the Docker wrappers
validate the requested physical ID and pass logical `npu:0` to the container
launcher. The argument cannot add devices missing from the container binding.

The legacy `applepen.sh` wrappers support background `docker exec`, but container
names and mount paths are deployment-specific. Set `FENGWU_CONTAINER` and
`FENGWU_CONTAINER_WORKDIR` explicitly in a new environment.

The public wrappers require the whole `ai4s/fengwu` package to be bind-mounted;
they do not copy source into independent container trees. Training and inference
may share a container, or use separate containers, provided each sees this same
package. Site-specific container names and migration helpers are not hardcoded
in the public package. Updated policies do not package private FlagGems kernel
patches: performance still depends on the separately installed vendor/FlagGems
environment. See [validation.md](validation.md) for the scope of recorded tests.

