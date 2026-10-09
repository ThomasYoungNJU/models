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
from typing import Any, Dict, Optional

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
        raise ValueError(
            f"Unsupported FENGWU_PLATFORM={value!r}. Supported values: {supported}"
        )
    return _PLATFORM_ALIASES[normalized]


def _optional_import(module_name: str) -> Optional[Any]:
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
    if any(
        token in text
        for token in ("hygon", "haiguang", "bw1000", "dtk", "das.")
    ):
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
    vendor_extension: Optional[Any] = None

    @property
    def available(self) -> bool:
        return self.device_type != "cpu" and _module_available(self.module)

    def device(self, index: int = 0) -> torch.device:
        if not self.available:
            if self.platform == "cpu" and os.environ.get("FENGWU_PLATFORM") == "cpu":
                return torch.device("cpu")
            raise RuntimeError("Accelerator unavailable; refusing implicit CPU fallback")
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
            f"platform={self.platform}, device_type={self.device_type}, "
            f"available={self.available}"
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


_RUNTIME: Optional[AcceleratorRuntime] = None


def get_accelerator() -> AcceleratorRuntime:
    global _RUNTIME
    from utils.runtime_guard import verify_source_manifest
    if _RUNTIME is None:
        verify_source_manifest()
        _RUNTIME = _detect_runtime()
        if not _RUNTIME.available and os.environ.get("FENGWU_PLATFORM") != "cpu":
            raise RuntimeError("No accelerator available; check container mounts/visibility. CPU fallback forbidden.")
        if _RUNTIME.device_type == "cuda" and _RUNTIME.available:
            actual = _cuda_platform_hint()
            if actual != "generic_cuda" and _RUNTIME.platform not in {actual, "generic_cuda"}:
                raise RuntimeError(f"Platform override {_RUNTIME.platform} conflicts with detected {actual}")
    return _RUNTIME


def configure_fp32_acceleration(runtime):
    """Process-local native Torch switches, NOT FlagGems kernel precision control."""
    mode = os.environ.get("FENGWU_FP32_ACCEL", "auto").strip().lower()
    if mode in {"1", "true", "on", "yes"}: desired = True
    elif mode in {"0", "false", "off", "no"}: desired = False
    elif mode == "auto": desired = None
    else: raise ValueError("FENGWU_FP32_ACCEL must be on, off, or auto")
    targets=[]
    if runtime.platform == "ascend":
        targets=[("torch.npu.matmul.allow_hf32", torch.npu.matmul, "allow_hf32"),
                 ("torch.npu.conv.allow_hf32", torch.npu.conv, "allow_hf32")]
    elif runtime.platform == "mthreads":
        _optional_import("torch_musa")
        targets=[("torch.backends.mudnn.allow_tf32", getattr(torch.backends, "mudnn", None), "allow_tf32")]
    elif runtime.platform in {"hcu","metax","thead","nvidia","generic_cuda"}:
        targets=[("torch.backends.cuda.matmul.allow_tf32", torch.backends.cuda.matmul, "allow_tf32"),
                 ("torch.backends.cudnn.allow_tf32", torch.backends.cudnn, "allow_tf32")]
    switches, unavailable = {}, []
    for name, obj, attr in targets:
        try:
            if obj is None or not hasattr(obj, attr): raise AttributeError(name)
            if desired is not None: setattr(obj, attr, desired)
            value=bool(getattr(obj, attr))
            if desired is not None and value != desired: raise RuntimeError(f"readback={value}")
            switches[name]=value
        except Exception as exc:
            unavailable.append(f"{name}: {exc}")
    # Unsupported native control is reported, not allowed to disable FlagGems.
    return {"state": "partial_or_unsupported" if unavailable or not targets else ("inherited" if desired is None else ("enabled" if desired else "disabled")),
            "platform": runtime.platform, "mode": mode, "switches": switches,
            "unavailable": unavailable, "scope": "native Torch only; FlagGems internal precision unchanged"}

def format_fp32_acceleration_status(status):
    return str(status)
