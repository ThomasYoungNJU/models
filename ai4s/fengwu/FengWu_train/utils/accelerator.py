"""Small runtime abstraction for the accelerator backends used by FengWu.

The vendor PyTorch distributions used by this project expose one of three
device APIs:

* ``torch.cuda``: NVIDIA and CUDA-compatible HCU/MetaX/T-Head bridges
* ``torch.npu``: Huawei Ascend
* ``torch.musa``: Moore Threads

No synchronization is performed by this module.  In particular, event timing
only queries already-completed events so that enabling timing does not change
the original CPU/accelerator overlap.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass
from typing import Any

import torch

_PLATFORM_ALIASES = {
    "auto": "auto",
    "cuda": "generic_cuda",
    "generic_cuda": "generic_cuda",
    "nvidia": "nvidia",
    "hcu": "hcu",
    "haiguang": "hcu",
    "hygon": "hcu",
    "metax": "metax",
    "mx": "metax",
    "musa": "mthreads",
    "moore": "mthreads",
    "mthreads": "mthreads",
    "npu": "ascend",
    "ascend": "ascend",
    "huawei": "ascend",
    "ptg": "thead",
    "pingtouge": "thead",
    "thead": "thead",
    "cpu": "cpu",
}


def _normalize_platform(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in _PLATFORM_ALIASES:
        supported = ", ".join(sorted(_PLATFORM_ALIASES))
        raise ValueError(f"Unsupported FENGWU_PLATFORM={value!r}. Supported values: {supported}")
    return _PLATFORM_ALIASES[normalized]


def _optional_import(module_name: str) -> None:
    """Import a vendor extension when it is installed.

    Vendor extensions normally attach ``npu`` or ``musa`` to the torch module.
    Import failures are ignored during auto detection and surfaced later if the
    corresponding backend is explicitly selected but unavailable.
    """

    try:
        if importlib.util.find_spec(module_name) is not None:
            importlib.import_module(module_name)
    except Exception:
        pass


def _module_available(module: Any) -> bool:
    if module is None:
        return False
    try:
        return bool(module.is_available())
    except Exception:
        return False


def _cuda_platform_hint() -> str:
    tokens = [str(getattr(torch, "__version__", ""))]
    try:
        tokens.append(str(torch.cuda.get_device_name(0)))
    except Exception:
        pass
    text = " ".join(tokens).lower()

    if any(token in text for token in ("metax", "mxgpu", "c550")):
        return "metax"
    if any(token in text for token in ("hygon", "haiguang", "bw1000", "dtk", "das.")):
        return "hcu"
    if any(token in text for token in ("thead", "hggc", "ppu")):
        return "thead"
    if "nvidia" in text:
        return "nvidia"
    if getattr(torch.version, "hip", None):
        return "hcu"
    return "generic_cuda"


@dataclass(frozen=True)
class AcceleratorRuntime:
    platform: str
    device_type: str
    module: Any

    @property
    def available(self) -> bool:
        return self.device_type != "cpu" and _module_available(self.module)

    @property
    def distributed_backend(self) -> str:
        if self.device_type == "npu":
            return "hccl"
        if self.device_type == "musa":
            return "mccl"
        return "nccl"

    def device(self, index: int = 0) -> torch.device:
        if not self.available:
            return torch.device("cpu")
        return torch.device(f"{self.device_type}:{index}")

    def set_device(self, index: int) -> None:
        if self.available and hasattr(self.module, "set_device"):
            self.module.set_device(index)

    def manual_seed_all(self, seed: int) -> None:
        if self.available and hasattr(self.module, "manual_seed_all"):
            self.module.manual_seed_all(seed)

    def memory_reserved(self) -> int:
        if not self.available or not hasattr(self.module, "memory_reserved"):
            return 0
        try:
            return int(self.module.memory_reserved())
        except Exception:
            return 0

    def max_memory_allocated(self) -> int:
        if not self.available or not hasattr(self.module, "max_memory_allocated"):
            return 0
        try:
            return int(self.module.max_memory_allocated())
        except Exception:
            return 0

    def empty_cache(self) -> None:
        if self.available and hasattr(self.module, "empty_cache"):
            self.module.empty_cache()

    def record_event(self) -> Any | None:
        """Record a non-blocking event on the current default stream."""

        if not self.available or not hasattr(self.module, "Event"):
            return None
        try:
            event = self.module.Event(enable_timing=True)
            event.record()
            return event
        except Exception:
            return None

    @staticmethod
    def event_ready(event: Any | None) -> bool:
        if event is None:
            return False
        try:
            return bool(event.query())
        except Exception:
            return False

    @classmethod
    def event_elapsed_seconds(cls, start_event: Any | None, end_event: Any | None) -> float | None:
        if start_event is None or not cls.event_ready(end_event):
            return None
        try:
            return float(start_event.elapsed_time(end_event)) / 1000.0
        except Exception:
            return None

    def ddp_device_ids(self, local_rank: int):
        return [local_rank] if self.available else None

    def summary(self) -> str:
        return (
            f"platform={self.platform}, device_type={self.device_type}, available={self.available}"
        )


def _detect_runtime() -> AcceleratorRuntime:
    override = _normalize_platform(os.environ.get("FENGWU_PLATFORM", "auto"))

    if override in {"ascend", "auto"}:
        _optional_import("torch_npu")
    if override in {"mthreads", "auto"}:
        _optional_import("torch_musa")

    npu = getattr(torch, "npu", None)
    musa = getattr(torch, "musa", None)

    if override == "ascend":
        return AcceleratorRuntime("ascend", "npu", npu)
    if override == "mthreads":
        return AcceleratorRuntime("mthreads", "musa", musa)
    if override == "cpu":
        return AcceleratorRuntime("cpu", "cpu", None)
    if override != "auto":
        return AcceleratorRuntime(override, "cuda", torch.cuda)

    if _module_available(npu):
        return AcceleratorRuntime("ascend", "npu", npu)
    if _module_available(musa):
        return AcceleratorRuntime("mthreads", "musa", musa)
    if _module_available(torch.cuda):
        return AcceleratorRuntime(_cuda_platform_hint(), "cuda", torch.cuda)
    return AcceleratorRuntime("cpu", "cpu", None)


_RUNTIME: AcceleratorRuntime | None = None


def get_accelerator() -> AcceleratorRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = _detect_runtime()
    return _RUNTIME
