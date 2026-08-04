# Six-platform validation record

The table summarizes the accepted single-device unified training results for
the checked-in FP32 baseline configuration. Loss values are recorded at the end
of each epoch.

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

## Inference acceptance

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

## Acceptance procedure for a new environment

1. Run both launchers with `--dry-run` and confirm platform/device resolution.
2. Import the vendor torch extension and print the visible accelerator.
3. Run one or two training epochs on the baseline NPY set.
4. Compare epoch-1 loss with the table above.
5. Run inference with `--steps 2`, then the normal 40-step case.
6. Confirm output count, dimensions and absence of NaN/Inf values.
7. For inference FlagGems acceptance, use `--flaggems required` and confirm the startup
   log reports `enabled`, including the expected exclusion list.

