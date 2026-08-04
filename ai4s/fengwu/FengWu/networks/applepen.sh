#!/usr/bin/env bash
set -euo pipefail

# Generic Docker convenience wrapper for the model-level scripts/infer.sh.

NETWORKS_DIR="$(cd -- "${BASH_SOURCE[0]%/*}" && pwd)"
MODEL_ROOT="$(cd "${NETWORKS_DIR}/../.." && pwd)"
DEVICE="${1:-${FENGWU_DEVICE_SPEC:-0}}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS=("$@")

usage() {
  cat <<'EOF'
Usage: FENGWU_CONTAINER=<name> ./applepen.sh [physical_device] [infer options]

This optional wrapper defaults to background Docker execution. The portable
entry point is scripts/infer.sh at the model root.

Environment variables:
  FENGWU_CONTAINER              Docker container name (required on the host)
  FENGWU_CONTAINER_WORKDIR      Model root inside the container; inferred from a bind mount
  FENGWU_PLATFORM               Platform passed to scripts/infer.sh (default: auto)
  FENGWU_BACKGROUND=0           Run in the foreground (default: 1)
  FENGWU_LOG=/path/to/log       Override the host-side log
  FENGWU_DRY_RUN=1              Print without executing

Example:
  FENGWU_CONTAINER=my-fengwu FENGWU_PLATFORM=hcu ./applepen.sh 0 --steps 40
EOF
}

if [[ "${DEVICE}" == "-h" || "${DEVICE}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -f /.dockerenv || "${FENGWU_INSIDE_CONTAINER:-0}" == "1" ]]; then
  exec bash "${MODEL_ROOT}/scripts/infer.sh" \
    --platform "${FENGWU_PLATFORM:-auto}" \
    --device "${DEVICE}" --foreground "${EXTRA_ARGS[@]}"
fi

CONTAINER="${FENGWU_CONTAINER:-}"
if [[ -z "${CONTAINER}" ]]; then
  echo "ERROR: FENGWU_CONTAINER is required by the public Docker wrapper." >&2
  exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not available; run scripts/infer.sh inside the environment." >&2
  exit 2
fi
if ! docker inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo "ERROR: Docker container not found: ${CONTAINER}" >&2
  exit 2
fi

CONTAINER_ROOT="${FENGWU_CONTAINER_WORKDIR:-}"
if [[ -z "${CONTAINER_ROOT}" ]]; then
  while IFS='|' read -r source destination; do
    [[ -n "${source}" && -n "${destination}" ]] || continue
    if [[ "${MODEL_ROOT}" == "${source}" ]]; then
      CONTAINER_ROOT="${destination}"
      break
    fi
    if [[ "${MODEL_ROOT}" == "${source}/"* ]]; then
      CONTAINER_ROOT="${destination}${MODEL_ROOT#"${source}"}"
      break
    fi
  done < <(
    docker inspect -f \
      '{{range .Mounts}}{{printf "%s|%s\n" .Source .Destination}}{{end}}' \
      "${CONTAINER}" 2>/dev/null || true
  )
fi
if [[ -z "${CONTAINER_ROOT}" ]]; then
  echo "ERROR: cannot map the host model directory into ${CONTAINER}." >&2
  echo "Set FENGWU_CONTAINER_WORKDIR to the ai4s/fengwu path inside the container." >&2
  exit 2
fi

RUNNING="$(docker inspect -f '{{.State.Running}}' "${CONTAINER}")"
if [[ "${RUNNING}" != "true" ]]; then
  if [[ "${FENGWU_DRY_RUN:-0}" == "1" ]]; then
    echo "NOTE: container is stopped; a real run would start it."
  elif [[ "${FENGWU_START_CONTAINER:-1}" == "1" ]]; then
    docker start "${CONTAINER}" >/dev/null
  else
    echo "ERROR: container is stopped: ${CONTAINER}" >&2
    exit 2
  fi
fi

COMMAND=(
  docker exec -w "${CONTAINER_ROOT}" "${CONTAINER}"
  bash scripts/infer.sh
  --platform "${FENGWU_PLATFORM:-auto}"
  --device "${DEVICE}"
  --foreground
  "${EXTRA_ARGS[@]}"
)

echo "Container : ${CONTAINER}"
echo "Workdir   : ${CONTAINER_ROOT}"
echo "Platform  : ${FENGWU_PLATFORM:-auto}"
echo "Device    : ${DEVICE}"

if [[ "${FENGWU_DRY_RUN:-0}" == "1" ]]; then
  printf 'Command   :'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

if [[ "${FENGWU_BACKGROUND:-1}" == "1" ]]; then
  mkdir -p "${MODEL_ROOT}/logs"
  LOG_FILE="${FENGWU_LOG:-${MODEL_ROOT}/logs/docker_infer_$(date +%Y%m%d_%H%M%S).log}"
  mkdir -p "$(dirname "${LOG_FILE}")"
  nohup "${COMMAND[@]}" >"${LOG_FILE}" 2>&1 < /dev/null &
  echo "Started background PID=$!"
  echo "Log: ${LOG_FILE}"
else
  exec "${COMMAND[@]}"
fi

