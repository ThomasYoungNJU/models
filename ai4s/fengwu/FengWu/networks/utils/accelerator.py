"""Accelerator detection shared by unified FengWu inference.

The vendor PyTorch distributions expose one of three device APIs:

* ``torch.cuda``: NVIDIA and CUDA-compatible HCU/MetaX/T-Head bridges
* ``torch.npu``: Huawei Ascend
* ``torch.musa``: Moore Threads
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


def _optional_import(module_name: str) -> Any | None:
    try:
        if importlib.util.find_spec(module_name) is not None:
            return importlib.import_module(module_name)
    except Exception:
        pass
    return None


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
    if any(token in text for token in ("thead", "hggc", "ppu", "zw810e")):
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
    vendor_extension: Any | None = None

    @property
    def available(self) -> bool:
        return self.device_type != "cpu" and _module_available(self.module)

    def device(self, index: int = 0) -> torch.device:
        if not self.available:
            return torch.device("cpu")
        return torch.device(f"{self.device_type}:{index}")

    def set_device(self, index: int) -> None:
        if self.available and hasattr(self.module, "set_device"):
            self.module.set_device(index)

    def synchronize(self) -> None:
        if self.available and hasattr(self.module, "synchronize"):
            self.module.synchronize()

    def extension_version(self) -> str:
        if self.vendor_extension is None:
            return "not-installed"
        return str(getattr(self.vendor_extension, "__version__", "unknown"))

    def summary(self) -> str:
        return (
            f"platform={self.platform}, device_type={self.device_type}, available={self.available}"
        )


def _detect_runtime() -> AcceleratorRuntime:
    override = _normalize_platform(os.environ.get("FENGWU_PLATFORM", "auto"))
    torch_npu = None
    torch_musa = None

    if override in {"ascend", "auto"}:
        torch_npu = _optional_import("torch_npu")
    if override in {"mthreads", "auto"}:
        torch_musa = _optional_import("torch_musa")

    npu = getattr(torch, "npu", None)
    musa = getattr(torch, "musa", None)

    if override == "ascend":
        return AcceleratorRuntime("ascend", "npu", npu, torch_npu)
    if override == "mthreads":
        return AcceleratorRuntime("mthreads", "musa", musa, torch_musa)
    if override == "cpu":
        return AcceleratorRuntime("cpu", "cpu", None)
    if override != "auto":
        return AcceleratorRuntime(override, "cuda", torch.cuda)

    if _module_available(npu):
        return AcceleratorRuntime("ascend", "npu", npu, torch_npu)
    if _module_available(musa):
        return AcceleratorRuntime("mthreads", "musa", musa, torch_musa)
    if _module_available(torch.cuda):
        return AcceleratorRuntime(_cuda_platform_hint(), "cuda", torch.cuda)
    return AcceleratorRuntime("cpu", "cpu", None)


_RUNTIME: AcceleratorRuntime | None = None


def get_accelerator() -> AcceleratorRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = _detect_runtime()
    return _RUNTIME
