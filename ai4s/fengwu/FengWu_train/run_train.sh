#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export FENGWU_TIMING="${FENGWU_TIMING:-1}"
export FENGWU_PLATFORM="${FENGWU_PLATFORM:-auto}"
export FENGWU_FLAGGEMS="${FENGWU_FLAGGEMS:-auto}"

# The verified Ascend 910C FlagGems path requires the expandable allocator;
# without it, activation-checkpoint recomputation can hit an AICore DDR
# address-out-of-range error. Preserve an explicit caller setting.
if [[ "${FENGWU_PLATFORM}" == "ascend" ||
      -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ||
      -d /usr/local/Ascend ]]; then
  export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
fi

CONFIG="${FENGWU_CONFIG:-./config/fengwu_single_sample.yaml}"
OUTDIR="${FENGWU_OUTDIR:-./debug_results}"
DESC="${FENGWU_DESC:-fengwu_single_sample}"

exec python3 -u train.py \
  --world_size "${WORLD_SIZE:-1}" \
  --per_cpus "${FENGWU_PER_CPUS:-4}" \
  --tensor_model_parallel_size "${TENSOR_MODEL_PARALLEL_SIZE:-1}" \
  --pipeline_model_parallel_size "${PIPELINE_MODEL_PARALLEL_SIZE:-1}" \
  --outdir "${OUTDIR}" \
  --desc "${DESC}" \
  -c "${CONFIG}" \
  "$@"
