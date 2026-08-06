#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/tangkun/project/dinov3newimprove/dinov3"
CONFIG_FILE="dinov3/configs/train/newdino/vitb_plus_davc_abm.yaml"
OUTPUT_DIR="${REPO_ROOT}/output/newdino/vitb_plus_davc_abm"
LOG_FILE="${OUTPUT_DIR}/train_g0_3_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

/data/tangkun/anaconda3/envs/dinov3/bin/torchrun \
  --standalone \
  --nproc_per_node=4 \
  dinov3/train/train.py \
  --config-file "${CONFIG_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee -a "${LOG_FILE}"
