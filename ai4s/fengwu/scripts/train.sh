#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "${BASH_SOURCE[0]%/*}" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_ROOT="${MODEL_ROOT}/FengWu_train"
# shellcheck disable=SC1091
source "${MODEL_ROOT}/scripts/runtime_env.sh"

PLATFORM="${FENGWU_PLATFORM:-auto}"
DEVICE="${FENGWU_DEVICE_SPEC:-0}"
CONFIG="${FENGWU_CONFIG:-${TRAIN_ROOT}/config/fengwu_single_sample.yaml}"
OUTDIR="${FENGWU_OUTDIR:-${TRAIN_ROOT}/debug_results}"
DESC="${FENGWU_DESC:-fengwu_single_sample}"
BACKGROUND="${FENGWU_BACKGROUND:-0}"
DRY_RUN="${FENGWU_DRY_RUN:-0}"
LOG_FILE="${FENGWU_LOG:-}"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/train.sh [options] [-- train.py arguments...]

Options:
  --platform NAME     auto|nvidia|ascend|hcu|metax|mthreads|thead|cpu
  --device DEVICE     Physical index, cuda[:N], npu[:N], musa[:N], or cpu
  --config PATH       Training YAML (default: FengWu_train/config/fengwu_single_sample.yaml)
  --outdir PATH       Training output root
  --desc TEXT         Run description used in the result-directory name
  --flaggems MODE     auto|on|off
  --timing MODE       1/on or 0/off
  --background        Run with nohup and return immediately
  --foreground        Run in the foreground (default)
  --log PATH          Background log path
  --dry-run           Print the resolved command without running it
  -h, --help          Show this help

Examples:
  bash scripts/train.sh --platform nvidia --device 0
  bash scripts/train.sh --platform ascend --device npu:2 --background
  bash scripts/train.sh --platform hcu --device 0 -- --seed 0

The vendor PyTorch environment must already be active. For Docker-based setups,
run this command inside the prepared container or use a site-specific wrapper.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform) PLATFORM="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --desc) DESC="$2"; shift 2 ;;
    --flaggems) export FENGWU_FLAGGEMS="$2"; shift 2 ;;
    --timing) export FENGWU_TIMING="$2"; shift 2 ;;
    --background) BACKGROUND=1; shift ;;
    --foreground) BACKGROUND=0; shift ;;
    --log) LOG_FILE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

configure_fengwu_runtime "${PLATFORM}" "${DEVICE}"
export FENGWU_FLAGGEMS="${FENGWU_FLAGGEMS:-auto}"
export FENGWU_TIMING="${FENGWU_TIMING:-1}"
export FENGWU_CONFIG="${CONFIG}"
export FENGWU_OUTDIR="${OUTDIR}"
export FENGWU_DESC="${DESC}"

COMMAND=(bash "${TRAIN_ROOT}/run_train.sh" --cuda "${FENGWU_LOGICAL_DEVICE}" "${EXTRA_ARGS[@]}")

echo "Platform : ${FENGWU_SELECTED_PLATFORM}"
echo "Device   : physical=${FENGWU_PHYSICAL_DEVICE}, logical=${FENGWU_LOGICAL_DEVICE}"
echo "Config   : ${CONFIG}"
echo "Output   : ${OUTDIR}"
echo "FlagGems : ${FENGWU_FLAGGEMS}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Command  :'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: training config not found: ${CONFIG}" >&2
  exit 2
fi

if [[ "${BACKGROUND}" == "1" ]]; then
  mkdir -p "${MODEL_ROOT}/logs"
  LOG_FILE="${LOG_FILE:-${MODEL_ROOT}/logs/train_$(date +%Y%m%d_%H%M%S).log}"
  mkdir -p "$(dirname "${LOG_FILE}")"
  nohup "${COMMAND[@]}" >"${LOG_FILE}" 2>&1 < /dev/null &
  echo "Started background PID=$!"
  echo "Log: ${LOG_FILE}"
else
  exec "${COMMAND[@]}"
fi

