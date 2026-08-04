"""Platform-specific FlagGems policy for unified FengWu inference."""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterable

from utils.accelerator import AcceleratorRuntime

_BASE_UNUSED = ["conv2d", "mul", "add"]

_EXTRA_UNUSED = {
    "hcu": ["cat", "stack"],
    "thead": ["cat", "stack"],
    "metax": ["cat", "stack", "true_divide_"],
    "mthreads": ["cat", "stack", "addmm", "mm"],
    "ascend": [],
}

_ENABLED = False
_STATUS: dict[str, object] = {
    "state": "not_configured",
    "platform": None,
    "vendor": None,
    "unused": [],
}


def _mode() -> str:
    value = os.environ.get("FENGWU_FLAGGEMS", "auto").strip().lower()
    if value in {"0", "false", "off", "no", "disabled"}:
        return "off"
    if value in {"1", "true", "on", "yes", "required"}:
        return "on"
    if value == "auto":
        return "auto"
    raise ValueError("FENGWU_FLAGGEMS must be one of: auto, 0/off, 1/on/required")


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _extra_unused_from_env() -> list[str]:
    value = os.environ.get("FENGWU_FLAGGEMS_UNUSED", "")
    return [item.strip() for item in value.split(",") if item.strip()]


def configure_flag_gems(runtime: AcceleratorRuntime, stage: str = "early") -> dict[str, object]:
    """Enable FlagGems at the safe point for the detected platform.

    MetaX delays enablement until model construction and device transfer have
    completed because its CPU dispatch bridge may otherwise intercept
    import-time CPU tensor creation.
    """

    global _ENABLED, _STATUS

    if _ENABLED:
        return dict(_STATUS)

    mode = _mode()
    if mode == "off":
        _STATUS = {
            "state": "disabled",
            "platform": runtime.platform,
            "vendor": None,
            "unused": [],
        }
        return dict(_STATUS)

    if mode == "auto" and runtime.platform in {
        "nvidia",
        "generic_cuda",
        "cpu",
    }:
        _STATUS = {
            "state": "disabled_auto",
            "platform": runtime.platform,
            "vendor": None,
            "unused": [],
        }
        return dict(_STATUS)

    if runtime.platform == "hcu" and os.path.isdir("/opt/dtk"):
        os.environ["ROCM_PATH"] = "/opt/dtk"

    try:
        flag_gems = importlib.import_module("flag_gems")
    except Exception as exc:
        if mode == "on":
            raise RuntimeError("FENGWU_FLAGGEMS=1 but flag_gems cannot be imported") from exc
        _STATUS = {
            "state": "unavailable",
            "platform": runtime.platform,
            "vendor": None,
            "unused": [],
            "detail": str(exc),
        }
        return dict(_STATUS)

    vendor = str(getattr(flag_gems, "vendor_name", "") or "")
    if runtime.platform == "metax" and stage == "early":
        _STATUS = {
            "state": "pending_late_enable",
            "platform": runtime.platform,
            "vendor": vendor,
            "unused": [],
        }
        return dict(_STATUS)

    unused = _unique(
        _BASE_UNUSED + _EXTRA_UNUSED.get(runtime.platform, []) + _extra_unused_from_env()
    )
    flag_gems.enable(unused=unused)
    _ENABLED = True
    _STATUS = {
        "state": "enabled",
        "platform": runtime.platform,
        "vendor": vendor,
        "unused": unused,
    }
    return dict(_STATUS)


def format_flag_gems_status(status: dict[str, object]) -> str:
    unused = ",".join(status.get("unused", [])) or "-"
    text = (
        f"state={status.get('state')}, platform={status.get('platform')}, "
        f"vendor={status.get('vendor') or '-'}, unused={unused}"
    )
    detail = status.get("detail")
    return f"{text}, detail={detail}" if detail else text
