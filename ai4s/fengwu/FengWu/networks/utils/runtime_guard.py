"""Startup-only checks: no tensor allocation, hot-path wrappers or synchronization."""
from __future__ import annotations
import hashlib
import json
import os
import pathlib
import sys
import warnings

VENDORS = {"ascend": "ascend", "hcu": "hygon", "metax": "metax", "thead": "thead", "mthreads": "mthreads", "nvidia": "nvidia"}

def configure_training_launch(runtime, flag_gems):
    """Private process only: avoid pathological Thead general-MM autotuning.

    Keep the original FlagGems MM kernel and precision; change launch tiling only.
    Tested on all 50 shape/stride pairs observed in a complete training epoch.
    """
    if runtime.platform != "thead":
        return None
    previous = getattr(flag_gems, "_fengwu_bounded_general_mm", None)
    if previous is not None:
        return previous
    import importlib
    import triton
    module = importlib.import_module("flag_gems.ops.mm")
    node = module.mm_kernel_general
    for _ in range(12):
        if hasattr(node, "configs"):
            break
        node = getattr(node, "fn", None)
        if node is None:
            raise RuntimeError("Unsupported FlagGems MM tuner: cannot apply verified Thead launch config")
    else:
        raise RuntimeError("Unsupported FlagGems MM tuner nesting")
    old_count = len(node.configs)
    config = triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=1)
    node.configs = [config]
    evidence = {"platform": "thead", "kernel": "mm_kernel_general", "old_config_count": old_count,
                "configuration": str(config), "kernel_source": module.__file__,
                "scope": "process-local launch tiling; original FlagGems kernel and precision unchanged"}
    flag_gems._fengwu_bounded_general_mm = evidence
    print("[LaunchConfig] " + json.dumps(evidence, sort_keys=True), flush=True)
    return evidence

def verify_source_manifest():
    expected = os.environ.get("FENGWU_SOURCE_SHA256")
    if not expected:
        return
    root = pathlib.Path(__file__).resolve().parent.parent
    files = os.environ["FENGWU_SOURCE_FILES"].split(":")
    rows = []
    for relative in files:
        path = pathlib.PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Invalid source-manifest path")
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        rows.append(f"{digest}  {relative}\n")
    actual = hashlib.sha256("".join(rows).encode()).hexdigest()
    if actual != expected:
        raise RuntimeError("Host/container code differs: refusing stale container copy. Synchronize code first.")
    print(f"[SourceManifest] verified={actual}, root={root}", flush=True)

def prepare_vendor(runtime):
    if not runtime.available:
        raise RuntimeError("FlagGems requires an available accelerator; CPU fallback is forbidden")
    expected = VENDORS.get(runtime.platform)
    if expected is None:
        raise RuntimeError(f"Cannot verify FlagGems vendor for {runtime.platform!r}")
    configured = os.environ.get("GEMS_VENDOR")
    if configured and configured != expected:
        raise RuntimeError(f"GEMS_VENDOR={configured!r} conflicts with {runtime.platform} ({expected})")
    os.environ["GEMS_VENDOR"] = expected
    return expected

def validate_vendor(flag_gems, runtime):
    expected = prepare_vendor(runtime)
    actual = str(getattr(flag_gems, "vendor_name", ""))
    device = str(getattr(flag_gems, "device", ""))
    if actual != expected or device != runtime.device_type:
        raise RuntimeError(f"FlagGems mismatch: {actual}/{device}; expected {expected}/{runtime.device_type}. Restart the process.")

def checked_enable(flag_gems, runtime, unused):
    """Record only registrations that returned successfully, not pre-appended lists."""
    validate_vendor(flag_gems, runtime)
    successful, failed = [], []
    parent = flag_gems.registrar
    class CheckedRegister(parent):
        def register_impl(self, key, fn, *args, **kwargs):
            expected_keys = {"cuda": {"CUDA"}, "npu": {"PrivateUse1", "NPU"}, "musa": {"PrivateUse1", "MUSA"}}[runtime.device_type]
            if self.reg_key not in expected_keys:
                failed.append({"key": key, "error": f"wrong dispatch {self.reg_key}"})
                raise RuntimeError(f"FlagGems dispatch {self.reg_key} is not {expected_keys}")
            try:
                result = super().register_impl(key, fn, *args, **kwargs)
            except Exception as exc:
                failed.append({"key": key, "function": fn.__name__, "error": str(exc)})
                raise
            successful.append({"key": key, "function": fn.__name__, "dispatch": self.reg_key})
            return result
    config = getattr(flag_gems, "_FULL_CONFIG", None)
    if config is None:
        raise RuntimeError("FlagGems registration config missing; cannot verify exclusions")
    names = {entry[1].__name__ for entry in config}
    extra = {s.strip() for s in os.environ.get("FENGWU_FLAGGEMS_UNUSED", "").split(",") if s.strip()}
    unknown_extra = sorted(extra - names)
    if unknown_extra:
        raise ValueError(f"Unknown exclusion function(s): {unknown_extra}; use mean_dim/softmax, not ATen aliases")
    unknown = sorted(set(unused) - names)
    if unknown:
        warnings.warn(f"Exclusions absent from this FlagGems build (not effective): {unknown}")
    flag_gems.enable(unused=unused, registrar=CheckedRegister)
    registrar = flag_gems.current_work_registrar
    expected = {entry[0] for entry in registrar.config}
    completed = {entry["key"] for entry in successful}
    missing = sorted(expected - completed)
    if failed or missing or not completed:
        raise RuntimeError(f"Incomplete FlagGems registration: failed={failed}, missing={missing}")
    leaked = sorted({r["function"] for r in successful} & set(unused))
    if leaked:
        raise RuntimeError(f"Requested exclusions still registered: {leaked}")
    metadata = {
        "python": sys.executable, "torch": str(__import__("torch").__version__),
        "flag_gems_source": flag_gems.__file__, "vendor": flag_gems.vendor_name,
        "device_type": runtime.device_type, "dispatch_key": registrar.reg_key,
        "registration_count": len(completed), "registration_failures": failed,
        "effective_unused": sorted(set(unused) & names), "absent_unused": unknown,
        "backend_unused": list(getattr(registrar, "vendor_unused_ops_list", [])),
        "registered_keys": sorted(completed), "source_sha256": source_hashes(),
        "env": {k: v for k, v in os.environ.items() if k.startswith(("FENGWU_", "GEMS_VENDOR", "CUDA_VISIBLE", "HIP_VISIBLE", "MUSA_VISIBLE", "ASCEND_RT_VISIBLE"))},
        "timing_scope": "host call / completed stream Event spans, not kernel-busy sums",
        "fp32_scope": "Torch switches only; FlagGems kernels can explicitly disable TF32/HF32",
    }
    print("[RuntimeEvidence] " + json.dumps(metadata, sort_keys=True), flush=True)
    return {"dispatch_key": registrar.reg_key, "registered_count": len(completed), "registration_failures": [], "effective_unused": metadata["effective_unused"], "absent_unused": unknown, "flag_gems_source": flag_gems.__file__}

def source_hashes():
    root = pathlib.Path(__file__).resolve().parent.parent
    result = {}
    for pattern in ("*.py", "utils/*.py", "models/*.py", "*.sh", "config/*.yaml"):
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result
