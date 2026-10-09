"""Verified policy. Exclusions are FlagGems implementation function names."""
from __future__ import annotations
import importlib
import os
from utils.runtime_guard import prepare_vendor, validate_vendor, checked_enable, configure_training_launch

_BASE_UNUSED = []
# Verified against the five live deployments on 2026-10-08.
_EXTRA_UNUSED = {
    'ascend': ['conv2d', 'layer_norm_backward', 'slice_backward'],
    'hcu': ['conv2d', 'layer_norm_backward', 'cat', 'stack', 'slice_backward'],
    'metax': ['conv2d', 'layer_norm_backward', 'slice_backward'],
    'thead': ['conv2d', 'layer_norm_backward', 'slice_backward'],
    'mthreads': ['conv2d', 'layer_norm_backward', 'slice_backward', 'mm'],
}
_ENABLED = False
_STATUS = {"state": "not_configured", "platform": None, "vendor": None, "unused": []}

def _mode():
    value = os.environ.get("FENGWU_FLAGGEMS", "auto").strip().lower()
    if value in {"0", "false", "off", "no", "disabled"}: return "off"
    if value in {"1", "true", "on", "yes", "required"}: return "on"
    if value == "auto": return "auto"
    raise ValueError("FENGWU_FLAGGEMS must be auto, off, or on/required")

def configure_flag_gems(runtime, stage="early"):
    global _ENABLED, _STATUS
    mode = _mode()
    if _ENABLED:
        if mode != _STATUS["mode"] or runtime.platform != _STATUS["platform"]:
            raise RuntimeError("FlagGems mode/device changed after registration; restart the process")
        if stage == "training":
            _STATUS["launch_tuning"] = configure_training_launch(runtime, importlib.import_module("flag_gems"))
        return dict(_STATUS)
    if mode == "off" or (mode == "auto" and runtime.platform == "nvidia"):
        _STATUS = {"state": "disabled" if mode == "off" else "disabled_auto", "platform": runtime.platform, "vendor": None, "unused": [], "mode": mode}
        return dict(_STATUS)
    prepare_vendor(runtime)
    if runtime.platform == "hcu" and os.path.isdir("/opt/dtk"):
        os.environ["ROCM_PATH"] = "/opt/dtk"
    try:
        flag_gems = importlib.import_module("flag_gems")
    except Exception as exc:
        raise RuntimeError("FlagGems is required but import failed; refusing native fallback") from exc
    validate_vendor(flag_gems, runtime)
    launch_tuning = configure_training_launch(runtime, flag_gems) if stage == "training" else None
    if runtime.platform == "metax" and stage == "early":
        _STATUS = {"state": "pending_late_enable", "platform": runtime.platform, "vendor": flag_gems.vendor_name, "unused": [], "mode": mode}
        return dict(_STATUS)
    extra = [s.strip() for s in os.environ.get("FENGWU_FLAGGEMS_UNUSED", "").split(",") if s.strip()]
    unused = list(dict.fromkeys(_BASE_UNUSED + _EXTRA_UNUSED.get(runtime.platform, []) + extra))
    evidence = checked_enable(flag_gems, runtime, unused)
    _ENABLED = True
    _STATUS = {"state": "enabled", "platform": runtime.platform, "vendor": flag_gems.vendor_name, "unused": unused, "mode": mode, "launch_tuning": launch_tuning, **evidence}
    return dict(_STATUS)

def format_flag_gems_status(status):
    unused = ",".join(status.get("unused", [])) or "-"
    return (f"state={status.get('state')}, platform={status.get('platform')}, vendor={status.get('vendor') or '-'}, unused={unused}, dispatch={status.get('dispatch_key', '-')}, registered={status.get('registered_count', 0)}, absent_unused={status.get('absent_unused', [])}")
