#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "${BASH_SOURCE[0]%/*}" && pwd)"
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFER_ROOT="${MODEL_ROOT}/FengWu"
NETWORKS_ROOT="${INFER_ROOT}/networks"
# shellcheck disable=SC1091
source "${MODEL_ROOT}/scripts/runtime_env.sh"

PLATFORM="${FENGWU_PLATFORM:-auto}"
DEVICE="${FENGWU_DEVICE_SPEC:-0}"
MODEL_DIR="${FENGWU_MODEL_PATH:-${INFER_ROOT}/model}"
OUTPUT_DIR="${FENGWU_OUTPUT_DIR:-${INFER_ROOT}/outputs}"
STEPS="${FENGWU_PREDICT_STEPS:-40}"
INIT_TIME="${FENGWU_INIT_TIME:-2023010200}"
PYTHON_BIN="${FENGWU_PYTHON:-python3}"
BACKGROUND="${FENGWU_BACKGROUND:-0}"
DRY_RUN="${FENGWU_DRY_RUN:-0}"
LOG_FILE="${FENGWU_LOG:-}"

usage() {
  cat <<'EOF'
Usage: bash scripts/infer.sh [options]

Options:
  --platform NAME     auto|nvidia|ascend|hcu|metax|mthreads|thead|cpu
  --device DEVICE     Physical index, cuda[:N], npu[:N], musa[:N], or cpu
  --model-dir PATH    Directory containing checkpoint, mean/std and input NPY
  --output-dir PATH   NetCDF output root
  --steps N           Autoregressive forecast steps (default: 40)
  --init-time YYYYMMDDHH
  --python PATH       Python executable from the vendor environment
  --flaggems MODE     auto|on|off|required
  --background        Run with nohup and return immediately
  --foreground        Run in the foreground (default)
  --log PATH          Background log path
  --dry-run           Print the resolved command without running it
  -h, --help          Show this help

Examples:
  bash scripts/infer.sh --platform nvidia --device cuda:0 --model-dir ./FengWu/model
  bash scripts/infer.sh --platform ascend --device npu:1 --background
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform) PLATFORM="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --init-time) INIT_TIME="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --flaggems) export FENGWU_FLAGGEMS="$2"; shift 2 ;;
    --background) BACKGROUND=1; shift ;;
    --foreground) BACKGROUND=0; shift ;;
    --log) LOG_FILE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "${STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --steps must be a positive integer." >&2
  exit 2
fi
if [[ ! "${INIT_TIME}" =~ ^[0-9]{10}$ ]]; then
  echo "ERROR: --init-time must use YYYYMMDDHH." >&2
  exit 2
fi

configure_fengwu_runtime "${PLATFORM}" "${DEVICE}"
export FENGWU_FLAGGEMS="${FENGWU_FLAGGEMS:-auto}"
export FENGWU_MODEL_PATH="${MODEL_DIR}"
export FENGWU_OUTPUT_DIR="${OUTPUT_DIR}"
export FENGWU_PREDICT_STEPS="${STEPS}"
export FENGWU_INIT_TIME="${INIT_TIME}"

COMMAND=("${PYTHON_BIN}" -u "${NETWORKS_ROOT}/run_fengwu.py")

echo "Platform : ${FENGWU_SELECTED_PLATFORM}"
echo "Device   : physical=${FENGWU_PHYSICAL_DEVICE}, logical=${FENGWU_LOGICAL_DEVICE}"
echo "Model    : ${MODEL_DIR}"
echo "Output   : ${OUTPUT_DIR}"
echo "Steps    : ${STEPS}"
echo "FlagGems : ${FENGWU_FLAGGEMS}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'Command  :'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

REQUIRED_ASSETS=(
  fengwu_v1_converted.pth
  data_mean.npy
  data_std.npy
  2023010200_fengwu_inpt.npy
)
for asset in "${REQUIRED_ASSETS[@]}"; do
  if [[ ! -f "${MODEL_DIR}/${asset}" ]]; then
    echo "ERROR: required inference asset not found: ${MODEL_DIR}/${asset}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"
if [[ "${BACKGROUND}" == "1" ]]; then
  mkdir -p "${MODEL_ROOT}/logs"
  LOG_FILE="${LOG_FILE:-${MODEL_ROOT}/logs/infer_$(date +%Y%m%d_%H%M%S).log}"
  mkdir -p "$(dirname "${LOG_FILE}")"
  nohup "${COMMAND[@]}" >"${LOG_FILE}" 2>&1 < /dev/null &
  echo "Started background PID=$!"
  echo "Log: ${LOG_FILE}"
else
  exec "${COMMAND[@]}"
fi

