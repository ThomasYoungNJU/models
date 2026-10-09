# Six-platform validation record

## Historical deployment results (July 2026)

The table records the original single-device unified training results for the
FP32 baseline configuration. It is historical evidence, not a guarantee for the
current exclusion lists or software environment. Loss values are recorded at
the end of each epoch. Numerical agreement alone does not prove FlagGems was
actually used; current launchers verify registration and device/vendor identity.

| Platform | Epoch 1 train / validation | Epoch 2 train / validation | Long run |
|---|---|---|---|
| NVIDIA H100 | `0.994041 / 0.314690` | `0.186197 / 0.026449` | 400 epochs completed |
| Huawei Ascend 910C | `0.994031 / 0.314696` | `0.186192 / 0.026445` | 2-epoch acceptance; production YAML restored to 400 |
| Hygon BW1000 | `0.994030 / 0.314691` | `0.186186 / 0.026445` | 400 epochs completed |
| MetaX C550 | `0.994046 / 0.314689` | `0.186196 / 0.026452` | 400 epochs completed |
| Moore Threads MTT S5000 | `0.994030 / 0.314691` | `0.186186 / 0.026446` | 400 epochs completed |
| T-Head PPU-ZW810E | `0.994035 / 0.314693` | `0.186190 / 0.026443` | 400 epochs completed |

The H100/A40 lineage is the numerical reference. First-epoch differences are at
approximately the `1e-5` scale. Hygon FlagGems and native-vendor-torch checks
also agreed at approximately `1e-6`. The T-Head 400-epoch structured loss file
matched its previous accepted implementation byte for byte.

## Historical inference deployment results

The standard inference target is 40 autoregressive six-hour steps and 40
NetCDF outputs.

| Platform | Recorded status |
|---|---|
| NVIDIA H100 | Native-PyTorch baseline completed; steady accelerator inference about 0.324 s/step in the archived run |
| Huawei Ascend 910C | 40/40 outputs; final output binary-identical before and after unification |
| Hygon BW1000 | 40/40 outputs; recorded total about 124.32 s |
| MetaX C550 | 40/40 outputs; recorded total about 105.94 s |
| Moore Threads MTT S5000 | Runtime, FlagGems, model and input validated; resource-limited check retained two outputs |
| T-Head PPU-ZW810E | 40/40 outputs; recorded total about 136.43 s |

The timing figures are environment-specific and are not cross-vendor benchmark
claims. They are included to make gross regressions visible.

## Current deployment progress (2026-10-08)

Current policies are in [platforms.md](platforms.md) and the two
`utils/flaggems_runtime.py` files. They were checked against five actual domestic
deployments; MetaX uses node 122 rather than the older node 101. NVIDIA remains
the native baseline. Registration verification and prior instrumented model
tests are separate from numerical acceptance.

| Platform | Current training progress | Latest 40-step inference loop total (s) |
|---|---|---:|
| Huawei Ascend 910C, one Chip | Latest 400-epoch log completed; Python total 21875.011418 s | 71.76 |
| Hygon BW1000 | Latest 400-epoch log completed; Python total 25211.732206 s | 80.98 |
| MetaX C550, node 122 | Latest 400-epoch log completed; Python total 28889.713231 s | 104.98 |
| Moore Threads MTT S5000 | Latest log has not yet reached its final summary; retain earlier results as historical | 186.39 |
| T-Head PPU-ZW810E | Latest 400-epoch log completed; Python total 25064.754072 s | 89.22 |

The first three training logs and all five inference logs started on October 6;
the latest T-Head and Moore Threads training logs started on October 8. Training
Python totals include startup, training, validation, checkpoint and final
summary; inference loop totals exclude final NetCDF output. They are different
timing scopes. These are observed environment-specific values, not a promise
that a fresh public-package deployment reproduces privately patched FlagGems
performance. Completion is not an independent numerical revalidation. This
local synchronization did not rerun full models or change installed kernels.

## Acceptance procedure for a new environment

1. Run both launchers with `--dry-run` and confirm platform/device resolution.
2. Import the vendor torch extension and print the visible accelerator.
3. Run one or two training epochs on the baseline NPY set.
4. Compare epoch-1 loss with the table above.
5. Run inference with `--steps 2`, then the normal 40-step case.
6. Confirm output count, dimensions and absence of NaN/Inf values.
7. For inference FlagGems acceptance, use `--flaggems required` and confirm the startup
   log reports `enabled`, including the expected exclusion list.

