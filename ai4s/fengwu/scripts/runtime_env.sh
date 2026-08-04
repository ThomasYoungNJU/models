#!/usr/bin/env bash

# Shared device selection for the public FengWu train/inference launchers.
# This file is sourced by train.sh and infer.sh.

normalize_fengwu_platform() {
  local value="${1,,}"
  case "${value}" in
    auto) echo "auto" ;;
    cuda|nvidia) echo "nvidia" ;;
    npu|ascend|huawei) echo "ascend" ;;
    hcu|dcu|hygon|rocm|hip) echo "hcu" ;;
    maca|metax) echo "metax" ;;
    musa|mthreads|moore) echo "mthreads" ;;
    thead|ppu|pingtouge) echo "thead" ;;
    cpu) echo "cpu" ;;
    *)
      echo "ERROR: unsupported platform '${1}'." >&2
      echo "Supported values: auto, nvidia, ascend, hcu, metax, mthreads, thead, cpu." >&2
      return 2
      ;;
  esac
}

configure_fengwu_runtime() {
  local requested_platform="$1"
  local requested_device="$2"
  local device_backend=""
  local device_index="${requested_device}"

  FENGWU_SELECTED_PLATFORM="$(normalize_fengwu_platform "${requested_platform}")" || return

  case "${requested_device,,}" in
    cuda|npu|musa|cpu)
      device_backend="${requested_device,,}"
      device_index="0"
      ;;
    cuda:*|npu:*|musa:*)
      device_backend="${requested_device%%:*}"
      device_backend="${device_backend,,}"
      device_index="${requested_device#*:}"
      ;;
  esac

  if [[ -n "${device_backend}" && "${FENGWU_SELECTED_PLATFORM}" == "auto" ]]; then
    case "${device_backend}" in
      cuda) FENGWU_SELECTED_PLATFORM="nvidia" ;;
      npu) FENGWU_SELECTED_PLATFORM="ascend" ;;
      musa) FENGWU_SELECTED_PLATFORM="mthreads" ;;
      cpu) FENGWU_SELECTED_PLATFORM="cpu" ;;
    esac
  fi

  if [[ ! "${device_index}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: device must be an integer, cuda[:N], npu[:N], musa[:N], or cpu." >&2
    return 2
  fi

  case "${FENGWU_SELECTED_PLATFORM}" in
    nvidia|metax|thead)
      export CUDA_VISIBLE_DEVICES="${device_index}"
      FENGWU_LOGICAL_DEVICE=0
      ;;
    hcu)
      export HIP_VISIBLE_DEVICES="${device_index}"
      FENGWU_LOGICAL_DEVICE=0
      ;;
    ascend)
      export ASCEND_RT_VISIBLE_DEVICES="${device_index}"
      export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
      FENGWU_LOGICAL_DEVICE=0
      ;;
    mthreads)
      export MUSA_VISIBLE_DEVICES="${device_index}"
      FENGWU_LOGICAL_DEVICE=0
      ;;
    auto)
      # With no platform hint, do not guess a vendor visibility variable.
      FENGWU_LOGICAL_DEVICE="${device_index}"
      ;;
    cpu)
      FENGWU_LOGICAL_DEVICE=0
      ;;
  esac

  FENGWU_PHYSICAL_DEVICE="${device_index}"
  export FENGWU_PLATFORM="${FENGWU_SELECTED_PLATFORM}"
  export FENGWU_DEVICE="${FENGWU_LOGICAL_DEVICE}"
}

