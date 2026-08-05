# Data and checkpoint preparation

Large assets are deliberately excluded from Git. A source checkout becomes
runnable after the assets for the selected workflow are placed as described
below.

## Project data source for all platforms

Before testing FengWu on any supported accelerator platform, download or copy
the required data from the project H100 data host to the target platform using
the approved internal transfer method. The authoritative source is:

| Item | Value |
|---|---|
| Hostname | `p-jn-sz-cw-h1-su1-gpu02-402-12a-02u-208-118` |
| Internal IP address | `10.6.208.118` |
| Machine ID | `58f05dc7-4bb0-4e9f-8e7f-d8d57e2e0188` |
| Source directory | `/public/FengwuData` |

Do not commit these assets to Git. After transferring them, place or link the
corresponding contents into the training-data and inference-asset locations
described below.

## Inference assets

The default inference asset directory is `FengWu/model`. It must contain:

| File | Purpose |
|---|---|
| `fengwu_v1_converted.pth` | PyTorch state dictionary used by the adapted multi-platform implementation |
| `data_mean.npy` | Per-channel normalization mean |
| `data_std.npy` | Per-channel normalization standard deviation |
| `2023010200_fengwu_inpt.npy` | Two consecutive six-hour input states used by the documented example |

Select another directory with `--model-dir` or `FENGWU_MODEL_PATH`.

The official inference repository provides normalization data, sample data and
ONNX checkpoints: <https://github.com/OpenEarthLab/FengWu>. Its input convention
uses two consecutive six-hour states. Each state contains 69 channels on a
`721 x 1440` latitude/longitude grid; the adapted PyTorch input therefore
contains 138 channels.

The upstream ONNX checkpoint is not interchangeable with
`fengwu_v1_converted.pth`. Obtain the converted PyTorch checkpoint from the
maintainer/project artifact store used for this adaptation, or reproduce the
approved conversion workflow before running this implementation. Do not rename
an ONNX file to `.pth`.

For a different initialization time, prepare an input file with the same
channel order and set `--init-time`. The current program retains the documented
sample filename inside the model directory; a custom asset directory can
contain a replacement file under that name.

## Training data

The checked-in `config/fengwu_single_sample.yaml` expects:

```text
FengWu_train/fengwu_era5/
```

It uses the following variables:

- surface: `u10`, `v10`, `t2m`, `msl`;
- pressure-level: `z`, `q`, `u`, `v`, `t`;
- levels: `50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000` hPa.

The default baseline interval is 2024-01-01 00:00 through 2024-01-02 18:00.
Files are six-hourly NPY fields following the paths consumed by
`datasets/era5_npy_f32.py`. This small dataset is intended for deployment and
numerical-consistency checks. It is not a substitute for the full historical
ERA5 corpus used to train a production forecast model.

`FengWu_train/scripts/convert_71chn_h5_to_fengwu_npy.py` converts a supported
71-channel yearly HDF5 source to this NPY layout. Run `--help` and specify the
input files and output directory explicitly.

ERA5 and operational-analysis data are governed by their providers' access and
licensing terms. Do not commit generated NPY, HDF5 or NetCDF data to this
repository.

## Expected outputs

Training creates a run directory containing logs, a structured loss history,
the resolved YAML and optional checkpoints. These files are ignored by Git.

Inference writes one file per six-hour lead time:

```text
<output-dir>/<YYYYMM>/<YYYYMMDDHH>/
└── Fengwu_V1_GLB_0P25_HOUR_<YYYYMMDDHH>_<lead-hour>.nc
```

A default 40-step run writes 40 files ending at lead hour 240.

