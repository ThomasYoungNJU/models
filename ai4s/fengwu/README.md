# FengWu: unified training and inference on six accelerator platforms

This directory packages FengWu training and 40-step autoregressive inference in
one codebase. The adaptation has been exercised on NVIDIA H100, Huawei Ascend
910C, Hygon BW1000, MetaX C550, Moore Threads MTT S5000 and T-Head
PPU-ZW810E. The non-NVIDIA paths can use FlagGems with platform-specific
operator exclusions; H100 remains the native-PyTorch numerical and performance
baseline.

FengWu predicts global atmospheric states on a 0.25-degree grid. The default
inference performs 40 six-hour steps (10 days) and writes one NetCDF file per
step. The default training configuration uses FP32, activation checkpointing,
`save_best: true` and `save_last: false`.

## About FengWu

"FengWu" is a global medium-range weather forecast AI large model, officially
released in April 2023 by the Shanghai Artificial Intelligence Laboratory in
collaboration with the University of Science and Technology of China, Shanghai
Jiao Tong University, Nanjing University of Information Science and Technology,
and the Institute of Atmospheric Physics of the Chinese Academy of Sciences. The
model is built on multi-modal and multi-task deep learning methods, achieving
for the first time effective forecasts of key atmospheric variables for more
than 10 days at high resolution.

In terms of performance, FengWu's 10-day forecast error is **10.87% lower** than
DeepMind's GraphCast. Its effective forecast horizon reaches **10.75 days**,
surpassing the 8.5-day upper limit of the traditional physics-based ECMWF HRES
model. In March 2024, the upgraded **"FengWu-GHR"** increased the resolution to
0.09°×0.09° (approximately 9 km × 9 km), a more than 7× improvement in
granularity, further extending the effective forecast horizon to **11.25 days**
— once again setting a new world record.

With outstanding forecast accuracy and extremely low computational cost, FengWu
provides critical support for meteorological services in agriculture, forestry,
animal husbandry, fisheries, renewable energy, aviation, and maritime
industries, marking the transition of AI weather forecasting from technical
validation to operational application.

For the model and scientific background, see the
[FengWu paper](https://doi.org/10.1038/s43247-025-02502-y), the
[official inference repository](https://github.com/OpenEarthLab/FengWu), and
the [public training repository](https://github.com/yuchendoudou/FengWu).

## Repository layout

```text
fengwu/
├── README.md                  # this user guide
├── requirements.txt           # portable Python dependencies
├── scripts/
│   ├── train.sh               # standard training entry point
│   ├── infer.sh               # standard inference entry point
│   └── runtime_env.sh         # platform/device selection shared by both
├── docs/
│   ├── platforms.md           # vendor environments and FlagGems policy
│   ├── data-and-checkpoints.md
│   └── validation.md          # six-platform acceptance evidence
├── FengWu_train/              # training implementation and YAML
└── FengWu/                    # inference implementation
```

Weights, ERA5 data, sample NPY input and generated files are intentionally not
stored in Git. Prepare them as described in
[docs/data-and-checkpoints.md](docs/data-and-checkpoints.md).

Before testing FengWu on any supported platform, obtain the required data from
the project H100 data host and transfer it to the target platform:

- hostname: `p-jn-sz-cw-h1-su1-gpu02-402-12a-02u-208-118`;
- internal IP address: `10.6.208.118`;
- machine ID: `58f05dc7-4bb0-4e9f-8e7f-d8d57e2e0188`;
- source directory: `/public/FengwuData`.

## 1. Prepare a platform environment

Use a clean environment or vendor container containing the accelerator driver
toolkit and its matching PyTorch build. Do not install a generic PyPI PyTorch
wheel over a vendor build.

After the platform framework is installed, install the portable dependencies:

```bash
cd ai4s/fengwu
python3 -m pip install -r requirements.txt
```

The environment must also provide a torchvision build matching its PyTorch.
Huawei requires `torch_npu`; Moore Threads requires `torch_musa`. Install
FlagGems from the image/vendor distribution when accelerated operators are
required. Exact reference environments and setup caveats are listed in
[docs/platforms.md](docs/platforms.md).

## 2. Select a platform and device

The standard launchers accept both a platform and a physical device index:

```bash
--platform nvidia|ascend|hcu|metax|mthreads|thead|cpu|auto
--device 0
```

Backend-style values are also accepted:

```bash
--device cuda:2
--device npu:0
--device musa:7
```

The launcher sets the vendor visibility variable and passes logical device 0
to the model. With `--platform auto`, no vendor visibility variable is guessed;
the selected index is passed directly to the runtime.

## 3. Train

The checked-in smoke/baseline YAML expects a small ERA5 NPY directory at
`FengWu_train/fengwu_era5`. After preparing that directory, run:

```bash
bash scripts/train.sh --platform nvidia --device 0
```

Examples for the other backends:

```bash
bash scripts/train.sh --platform ascend --device npu:0
bash scripts/train.sh --platform hcu --device 0
bash scripts/train.sh --platform metax --device 0
bash scripts/train.sh --platform mthreads --device musa:0
bash scripts/train.sh --platform thead --device 0
```

Use another YAML or output directory without editing source code:

```bash
bash scripts/train.sh \
  --platform nvidia --device 0 \
  --config /path/to/train.yaml \
  --outdir /path/to/results \
  --desc my_run
```

Arguments after `--` are forwarded to `train.py`:

```bash
bash scripts/train.sh --platform nvidia --device 0 -- --seed 0
```

Background execution is supported explicitly:

```bash
bash scripts/train.sh --platform nvidia --device 0 --background
tail -f logs/train_YYYYmmdd_HHMMSS.log
```

Training outputs include `iter.log`, `training_loss.jsonl`,
`training_options.yaml` and, when the best validation loss improves,
`checkpoint_best.pth`.

## 4. Inference

Place all four required inference assets in `FengWu/model` or another directory
selected with `--model-dir`. Then run:

```bash
bash scripts/infer.sh \
  --platform nvidia --device 0 \
  --model-dir ./FengWu/model \
  --output-dir ./FengWu/outputs \
  --steps 40 \
  --init-time 2023010200
```

The same command works on the other platforms by changing `--platform` and
`--device`. For example:

```bash
bash scripts/infer.sh --platform ascend --device npu:0 --model-dir ./FengWu/model
bash scripts/infer.sh --platform hcu --device 0 --model-dir ./FengWu/model
bash scripts/infer.sh --platform mthreads --device musa:0 --model-dir ./FengWu/model
```

Run inference in the background with:

```bash
bash scripts/infer.sh --platform nvidia --device 0 \
  --model-dir ./FengWu/model --background
```

For a 40-step run, the expected result is 40 NetCDF files under:

```text
<output-dir>/<YYYYMM>/<YYYYMMDDHH>/
```

## 5. FlagGems policy

The default is `FENGWU_FLAGGEMS=auto`:

- H100/NVIDIA uses native PyTorch and does not enable FlagGems.
- Supported non-NVIDIA platforms enable FlagGems when it can be imported.
- Known incompatible or numerically unsafe operators are excluded per platform.

Override the policy with either the launcher option or environment variable:

```bash
bash scripts/train.sh --platform hcu --device 0 --flaggems off
FENGWU_FLAGGEMS=required bash scripts/infer.sh --platform ascend --device 0
```

`required` is useful for inference acceptance because it fails instead of
silently continuing when FlagGems is unavailable. See
[docs/platforms.md](docs/platforms.md) for the exclusion lists.

## 6. Docker convenience launchers

The `applepen.sh` scripts under `FengWu_train/` and `FengWu/networks/` are
host-side convenience wrappers developed for container deployments. They
default to background mode and accept a physical card number. Because container
names and bind mounts are site-specific, set `FENGWU_CONTAINER` and, when mount
inspection cannot resolve the code path, `FENGWU_CONTAINER_WORKDIR`.

The portable `scripts/train.sh` and `scripts/infer.sh` are the primary public
entry points and should be run inside the prepared environment/container.

## 7. Reproducibility and validation

The six-platform checks use FP32 (`enabled_amp: false`). H100 permits TF32 in
its platform runtime but AMP remains disabled. The first-epoch train/validation
loss is approximately `0.99403 / 0.31469` for the checked-in baseline dataset,
with platform differences at roughly the `1e-5` scale. Detailed recorded
results are in [docs/validation.md](docs/validation.md).

Validate command resolution without running a model:

```bash
bash scripts/train.sh --platform ascend --device 0 --dry-run
bash scripts/infer.sh --platform ascend --device 0 --dry-run
```

## Known boundaries

- The checked-in YAML is a reproducibility/smoke configuration, not a bundled
  full ERA5 training corpus.
- The public upstream ONNX checkpoint cannot be substituted directly for the
  adapted PyTorch `fengwu_v1_converted.pth` file.
- Multi-device distributed execution has not completed the same six-platform
  acceptance matrix as single-device execution.
- Moore Threads device visibility is commonly fixed when a Docker container is
  created; changing `MUSA_VISIBLE_DEVICES` with `docker exec` may not rebind the
  physical card.
- Vendor framework, driver and FlagGems versions must be mutually compatible.

## License and attribution

See the repository-level `LICENSE` and [NOTICE.md](NOTICE.md). Model weights,
datasets and vendor SDKs are not included and retain their providers' terms.

