#!/usr/bin/env bash
set -euo pipefail

# Render training views and extract the bounded TSDF mesh for one DTU scene.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/path/to/DTU}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/dtu}"
SCENE="${SCENE:-scan24}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" render.py \
    -s "${DATA_ROOT}/${SCENE}" \
    -m "${OUTPUT_ROOT}/${SCENE}" \
    --iteration 30000 \
    --skip_test \
    --depth_ratio 1.0 \
    --num_cluster 1 \
    --voxel_size 0.004 \
    --sdf_trunc 0.016 \
    --depth_trunc 3.0
