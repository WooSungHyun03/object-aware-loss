#!/usr/bin/env bash
set -euo pipefail

# Train one DTU scene with the recorded 2DGS DTU geometry settings and the
# paper's default object-aware schedule.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/path/to/DTU}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/dtu}"
SCENE="${SCENE:-scan24}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PRUNING_ITERATIONS=()
for ((iteration = 2500; iteration < 30000; iteration += 500)); do
    PRUNING_ITERATIONS+=("${iteration}")
done

cd "${REPO_ROOT}"
"${PYTHON_BIN}" train.py \
    -s "${DATA_ROOT}/${SCENE}" \
    -m "${OUTPUT_ROOT}/${SCENE}" \
    --quiet \
    --test_iterations 7000 30000 \
    --save_iterations 7000 30000 \
    --depth_ratio 1.0 \
    --resolution 2 \
    --lambda_dist 1000 \
    --lambda_pa 0.1 \
    --lambda_po 1.0 \
    --objectmark_start_iter 0 \
    --objectmark_end_iter 30000 \
    --objectmark_pruning_threshold 0.5 \
    --objectmark_pruning_iterations "${PRUNING_ITERATIONS[@]}"
