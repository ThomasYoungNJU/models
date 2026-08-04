#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


CHANNELS = [
    "u10m", "v10m", "t2m", "msl", "q2m",
    "u50", "u100", "u150", "u200", "u250", "u300", "u400", "u500",
    "u600", "u700", "u850", "u925", "u1000",
    "v50", "v100", "v150", "v200", "v250", "v300", "v400", "v500",
    "v600", "v700", "v850", "v925", "v1000",
    "z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500",
    "z600", "z700", "z850", "z925", "z1000",
    "t50", "t100", "t150", "t200", "t250", "t300", "t400", "t500",
    "t600", "t700", "t850", "t925", "t1000",
    "q50", "q100", "q150", "q200", "q250", "q300", "q400", "q500",
    "q600", "q700", "q850", "q925", "q1000", "tp",
]

SINGLE_LEVEL_MAP = {
    "u10m": "u10",
    "v10m": "v10",
    "t2m": "t2m",
    "msl": "msl",
}

MULTI_LEVEL_VARS = {"u", "v", "z", "t", "q"}


@dataclass(frozen=True)
class OutputSpec:
    channel_index: int
    source_name: str
    target_name: str
    level: int | None
    is_single_level: bool


def build_output_specs() -> list[OutputSpec]:
    specs: list[OutputSpec] = []
    for idx, channel in enumerate(CHANNELS):
        if channel in SINGLE_LEVEL_MAP:
            specs.append(
                OutputSpec(
                    channel_index=idx,
                    source_name=channel,
                    target_name=SINGLE_LEVEL_MAP[channel],
                    level=None,
                    is_single_level=True,
                )
            )
            continue

        prefix = "".join(ch for ch in channel if ch.isalpha())
        suffix = channel[len(prefix):]
        if prefix in MULTI_LEVEL_VARS and suffix:
            specs.append(
                OutputSpec(
                    channel_index=idx,
                    source_name=channel,
                    target_name=prefix,
                    level=int(suffix),
                    is_single_level=False,
                )
            )

    return specs


OUTPUT_SPECS = build_output_specs()


def iter_h5_files(src_dir: Path) -> Iterable[Path]:
    yield from sorted(p for p in src_dir.iterdir() if p.suffix == ".h5" and p.stem.isdigit())


def build_timestamp_sequence(year: int, count: int, interval_hours: int) -> list[datetime]:
    start = datetime(year, 1, 1, 0, 0, 0)
    return [start + timedelta(hours=interval_hours * idx) for idx in range(count)]


def validate_timestamps(year: int, timestamps: list[datetime], interval_hours: int) -> None:
    expected_last = datetime(year + 1, 1, 1, 0, 0, 0) - timedelta(hours=interval_hours)
    if timestamps[-1] != expected_last:
        raise ValueError(
            f"{year}.h5 time length mismatch: got {len(timestamps)} steps, "
            f"last timestamp {timestamps[-1]}, expected {expected_last} for {interval_hours}h interval"
        )


def make_output_path(dst_root: Path, ts: datetime, spec: OutputSpec) -> Path:
    day_dir = dst_root / str(ts.year) / ts.strftime("%Y-%m-%d")
    if spec.is_single_level:
        day_dir = dst_root / "single" / str(ts.year) / ts.strftime("%Y-%m-%d")
        filename = f"{ts.strftime('%H:%M:%S')}-{spec.target_name}.npy"
    else:
        filename = f"{ts.strftime('%H:%M:%S')}-{spec.target_name}-{spec.level}.0.npy"
    return day_dir / filename


def save_time_step(
    fields: h5py.Dataset,
    time_index: int,
    ts: datetime,
    dst_root: Path,
    specs: list[OutputSpec],
    overwrite: bool,
    dry_run: bool,
) -> None:
    for spec in specs:
        out_path = make_output_path(dst_root, ts, spec)
        if out_path.exists() and not overwrite:
            continue

        if dry_run:
            print(out_path)
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        array = np.asarray(fields[time_index, spec.channel_index], dtype=np.float32)
        np.save(out_path, array)


def convert_file(
    h5_path: Path,
    dst_root: Path,
    overwrite: bool,
    dry_run: bool,
    interval_hours: int,
    limit_steps: int | None,
) -> None:
    year = int(h5_path.stem)
    with h5py.File(h5_path, "r") as f:
        if "fields" not in f:
            raise KeyError(f"{h5_path} does not contain dataset 'fields'")

        fields = f["fields"]
        if fields.ndim != 4:
            raise ValueError(f"{h5_path} fields ndim must be 4, got {fields.ndim}")
        if fields.shape[1] != len(CHANNELS):
            raise ValueError(
                f"{h5_path} channel count mismatch: got {fields.shape[1]}, expected {len(CHANNELS)}"
            )

        timestamps = build_timestamp_sequence(year, fields.shape[0], interval_hours)
        validate_timestamps(year, timestamps, interval_hours)

        end = fields.shape[0] if limit_steps is None else min(limit_steps, fields.shape[0])
        print(f"converting {h5_path.name}: steps={end}/{fields.shape[0]}")
        for time_index in range(end):
            save_time_step(
                fields=fields,
                time_index=time_index,
                ts=timestamps[time_index],
                dst_root=dst_root,
                specs=OUTPUT_SPECS,
                overwrite=overwrite,
                dry_run=dry_run,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 71-channel yearly HDF5 files into FengWu ERA5 npy layout."
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        required=True,
        help="Directory containing yearly .h5 files.",
    )
    parser.add_argument(
        "--dst-dir",
        type=Path,
        required=True,
        help="Output directory in FengWu npy layout.",
    )
    parser.add_argument(
        "--interval-hours",
        type=int,
        default=6,
        help="Time interval between adjacent frames in the h5 files.",
    )
    parser.add_argument(
        "--limit-steps",
        type=int,
        default=None,
        help="Only convert the first N time steps for validation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing npy files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print output paths without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dst_dir.mkdir(parents=True, exist_ok=True)

    h5_files = list(iter_h5_files(args.src_dir))
    if not h5_files:
        raise FileNotFoundError(f"No yearly .h5 files found in {args.src_dir}")

    for h5_path in h5_files:
        convert_file(
            h5_path=h5_path,
            dst_root=args.dst_dir,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            interval_hours=args.interval_hours,
            limit_steps=args.limit_steps,
        )


if __name__ == "__main__":
    main()
