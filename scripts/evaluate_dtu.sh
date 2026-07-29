#!/usr/bin/env bash
set -euo pipefail

# Cull and evaluate one reconstructed DTU mesh. This requires the official DTU
# evaluation data in addition to the prepared input scene.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/path/to/DTU}"
OFFICIAL_DTU_ROOT="${OFFICIAL_DTU_ROOT:-/path/to/Official_DTU_Dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/dtu}"
SCENE="${SCENE:-scan24}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCAN_ID="${SCENE#scan}"
MESH_PATH="${OUTPUT_ROOT}/${SCENE}/train/ours_30000/fuse_post.ply"
EVALUATION_DIR="${OUTPUT_ROOT}/${SCENE}/dtu_evaluation"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/eval_dtu/evaluate_single_scene.py \
    --input_mesh "${MESH_PATH}" \
    --scan_id "${SCAN_ID}" \
    --output_dir "${EVALUATION_DIR}" \
    --mask_dir "${DATA_ROOT}" \
    --DTU "${OFFICIAL_DTU_ROOT}"
