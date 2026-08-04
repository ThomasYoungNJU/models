import os
import sys
import time


_T_START = time.perf_counter()
_NETWORKS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_NETWORKS_DIR)
_VENDOR_DIR = os.path.join(_PROJECT_ROOT, "vendor")
if os.path.isdir(_VENDOR_DIR) and _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

import numpy as np
import torch
from tqdm import tqdm
import xarray as xr

from utils.accelerator import get_accelerator
from utils.flaggems_runtime import (
    configure_flag_gems,
    format_flag_gems_status,
)


class Timer:
    def __init__(self, name):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.t
        print(f"[Timer] {self.name}: {self.elapsed:.2f}s")


def get_inference_output_nc(
    data,
    init_time,
    out_path="./",
    valid_autoreg_steps=60,
    complevel=5,
    check=True,
):
    """Convert selected channels and write one NetCDF file per forecast step."""
    print("data", data.shape)
    formatted_init_time = np.datetime64(
        init_time[:4]
        + "-"
        + init_time[4:6]
        + "-"
        + init_time[6:8]
        + "T"
        + init_time[8:10]
        + ":00:00"
    )
    time_interval = np.timedelta64(6, "h")

    out_channel_name = [
        "t2m",
        "u10m",
        "v10m",
        "msl",
        "z100",
        "t100",
        "q100",
        "u100",
        "v100",
        "z200",
        "t200",
        "q200",
        "u200",
        "v200",
        "z500",
        "t500",
        "q500",
        "u500",
        "v500",
        "z700",
        "t700",
        "q700",
        "u700",
        "v700",
        "z850",
        "t850",
        "q850",
        "u850",
        "v850",
        "z925",
        "t925",
        "q925",
        "u925",
        "v925",
    ]

    t_uc = 0.0
    t_merge = 0.0
    t_nc = 0.0

    for step in range(valid_autoreg_steps):
        t0 = time.perf_counter()
        current_time = formatted_init_time + time_interval * (step + 1)
        current_time = current_time.astype("datetime64[ns]")
        data_list = []
        for var in out_channel_name:
            if var == "tp":
                var_index = -1
            else:
                var_index = fengwu_channels.index(var)
            out_data = data[step : step + 1, var_index]
            if var == "tp":
                out_data = out_data * 1000
            elif "z" in var:
                out_data = out_data / 9.8
            elif "t" in var:
                out_data = out_data - 273.15
            data_xr = xr.DataArray(
                out_data,
                dims=("time", "lat", "lon"),
                coords={
                    "time": [current_time],
                    "lat": np.arange(90, -90.25, -0.25),
                    "lon": np.arange(0, 360, 0.25),
                },
                name=f"{var}",
            )
            data_list.append(data_xr)
        t1 = time.perf_counter()

        dataset_xr = xr.merge(data_list)
        t2 = time.perf_counter()

        output_dir = os.path.join(
            out_path, init_time[:6], init_time
        )
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            "Fengwu_V1_GLB_0P25_HOUR_{}_{:03d}.nc".format(
                init_time, (step + 1) * 6
            ),
        )
        dataset_xr.to_netcdf(output_file)
        t3 = time.perf_counter()

        t_uc += t1 - t0
        t_merge += t2 - t1
        t_nc += t3 - t2
        print(f"{(step + 1) * 6:03d} done: {output_file}")

    print(f"[Timer] 5a. Unit conv + xarray DataArray: {t_uc:.2f}s")
    print(f"[Timer] 5b. xr.merge: {t_merge:.2f}s")
    print(f"[Timer] 5c. to_netcdf write: {t_nc:.2f}s")


fengwu_channels = [
    "u10m",
    "v10m",
    "t2m",
    "msl",
    "z50",
    "z100",
    "z150",
    "z200",
    "z250",
    "z300",
    "z400",
    "z500",
    "z600",
    "z700",
    "z850",
    "z925",
    "z1000",
    "q50",
    "q100",
    "q150",
    "q200",
    "q250",
    "q300",
    "q400",
    "q500",
    "q600",
    "q700",
    "q850",
    "q925",
    "q1000",
    "u50",
    "u100",
    "u150",
    "u200",
    "u250",
    "u300",
    "u400",
    "u500",
    "u600",
    "u700",
    "u850",
    "u925",
    "u1000",
    "v50",
    "v100",
    "v150",
    "v200",
    "v250",
    "v300",
    "v400",
    "v500",
    "v600",
    "v700",
    "v850",
    "v925",
    "v1000",
    "t50",
    "t100",
    "t150",
    "t200",
    "t250",
    "t300",
    "t400",
    "t500",
    "t600",
    "t700",
    "t850",
    "t925",
    "t1000",
]


model_path = os.environ.get(
    "FENGWU_MODEL_PATH", os.path.join(_PROJECT_ROOT, "model")
)
fengwu_checkpoint = os.path.join(
    model_path, "fengwu_v1_converted.pth"
)
global_means_path = os.path.join(model_path, "data_mean.npy")
global_stds_path = os.path.join(model_path, "data_std.npy")
fengwu_input_path = os.path.join(
    model_path, "2023010200_fengwu_inpt.npy"
)
predict_step = int(os.environ.get("FENGWU_PREDICT_STEPS", "40"))
init_time = os.environ.get("FENGWU_INIT_TIME", "2023010200")
output_root = os.environ.get("FENGWU_OUTPUT_DIR", "./")


device_index = int(os.environ.get("FENGWU_DEVICE", "0"))
runtime = get_accelerator()

with Timer("1. Device setup"):
    runtime.set_device(device_index)
    device = runtime.device(device_index)
    if runtime.platform == "nvidia":
        torch.backends.cudnn.benchmark = True
    print(f"Device: {device}; {runtime.summary()}")
    print(
        "torch={}, torch_npu={}, xarray={}".format(
            torch.__version__,
            runtime.extension_version(),
            xr.__version__,
        )
    )


flag_gems_status = configure_flag_gems(runtime, stage="early")
print(f"[FlagGems] {format_flag_gems_status(flag_gems_status)}")

print(
    "Runtime: platform={}, device_type={}, FlagGems={}, steps={}, output={}".format(
        runtime.platform,
        runtime.device_type,
        flag_gems_status["state"],
        predict_step,
        output_root,
    )
)


def synchronize():
    runtime.synchronize()

with Timer("2a. torch.load checkpoint"):
    ckpt = torch.load(fengwu_checkpoint, map_location="cpu")

with Timer("2b. Model init (LGUnet_all)"):
    from LGUnet_all import LGUnet_all

    fengwu_model = LGUnet_all()

with Timer("2c. load_state_dict"):
    fengwu_model.load_state_dict(
        ckpt["model"]["lgunet_all"], strict=True
    )

with Timer("2d. Model to device"):
    fengwu_model = fengwu_model.to(device)
    synchronize()

with Timer("2e. Model eval"):
    fengwu_model.eval()

if flag_gems_status["state"] == "pending_late_enable":
    flag_gems_status = configure_flag_gems(runtime, stage="late")
    print(f"[FlagGems] {format_flag_gems_status(flag_gems_status)}")


with Timer("3a. Load means.npy + stds.npy (+ to accelerator)"):
    means = np.load(global_means_path)[
        np.newaxis, :, np.newaxis, np.newaxis
    ]
    stds = np.load(global_stds_path)[
        np.newaxis, :, np.newaxis, np.newaxis
    ]
    means_device = torch.from_numpy(means).to(device)
    stds_device = torch.from_numpy(stds).to(device)

with Timer("3b. Load input.npy"):
    fengwu_input_raw = np.load(fengwu_input_path)

with Timer("3c. np->torch + input to accelerator"):
    fengwu_input = torch.tensor(fengwu_input_raw).to(device)
    synchronize()


inference_result = []
t_accelerator_total = 0.0
t_io_total = 0.0

print(f"\nInference loop ({predict_step} steps):")

t_loop_start = time.perf_counter()
with torch.inference_mode():
    for step in range(predict_step):
        t0 = time.perf_counter()
        output = fengwu_model(fengwu_input)
        fengwu_input = torch.cat(
            (fengwu_input[:, 69:], output[:, :69]), dim=1
        )
        pred_device = output[:, :69] * stds_device + means_device
        synchronize()
        t1 = time.perf_counter()

        pred_input_cpu = pred_device.cpu().numpy()
        inference_result.append(pred_input_cpu)
        t2 = time.perf_counter()

        t_accelerator = t1 - t0
        t_io = t2 - t1
        t_accelerator_total += t_accelerator
        t_io_total += t_io
        print(
            "  step {:2d} | accelerator={:.3f}s  io={:.3f}s  "
            "total={:.3f}s".format(
                step + 1,
                t_accelerator,
                t_io,
                t_accelerator + t_io,
            )
        )

t_loop_total = time.perf_counter() - t_loop_start

print(f"\n[Timer] 4. Inference loop total: {t_loop_total:.2f}s")
print(
    "  -- accelerator compute (model + cat + denorm): "
    f"{t_accelerator_total:.2f}s"
)
print(f"  -- IO (accelerator->CPU transfer): {t_io_total:.2f}s")

with Timer("4e. np.concatenate"):
    result = np.concatenate(inference_result, axis=0)


with Timer("5. Post-processing (NetCDF output)"):
    get_inference_output_nc(
        result,
        init_time,
        out_path=output_root,
        valid_autoreg_steps=predict_step,
    )


_T_END = time.perf_counter()
print(f"\n{'=' * 50}")
print(
    f"[Timer] TOTAL (program start to exit): "
    f"{_T_END - _T_START:.2f}s"
)
print(f"{'=' * 50}")
