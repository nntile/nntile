#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file examples/run_gpt2_graph_training_demo.sh
# Prepare tiny train.bin and run gpt2_graph_training for a few epochs.
#
# Usage (from repo root, after building gpt2_graph_training):
#   ./examples/run_gpt2_graph_training_demo.sh
#
# Optional environment variables:
#   BUILD_DIR      CMake build directory (default: <repo>/build)
#   DATA_DIR       Where to write train.bin (default: <build>/examples/demo_data/gpt2)
#   EXECUTION_OUT  If set, write static task schedule to this path (--execution-out)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=demo_common.sh
source "${SCRIPT_DIR}/demo_common.sh"

REPO_ROOT="$(demo_repo_root)"
BUILD_DIR="$(demo_resolve_build_dir "${REPO_ROOT}")"
DATA_DIR="${DATA_DIR:-${BUILD_DIR}/examples/demo_data/gpt2}"
TRAIN_BIN="${DATA_DIR}/train.bin"
LOG_FILE="${DATA_DIR}/training.log"
BIN="$(demo_example_bin "${BUILD_DIR}" gpt2_graph_training)"
CONFIG_JSON="${SCRIPT_DIR}/demo_configs/gpt2_tiny_config.json"
TILING_JSON="${SCRIPT_DIR}/demo_configs/gpt2_tiny_tiling.json"
EXECUTION_JSON=""
if [[ -n "${EXECUTION_OUT:-}" ]]; then
    EXECUTION_JSON="${EXECUTION_OUT}"
fi

SEQ_LEN="${SEQ_LEN:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_BATCHES="${NUM_BATCHES:-8}"
EPOCHS="${EPOCHS:-4}"
MAX_BATCHES="${MAX_BATCHES:-32}"
LR="${LR:-0.003}"

demo_require_executable "${BIN}" "gpt2_graph_training"

echo "=== GPT-2 graph training demo ==="
echo "Build dir:  ${BUILD_DIR}"
echo "Data dir:   ${DATA_DIR}"
echo ""

echo "--- Preparing tiny train.bin ---"
python3 "${SCRIPT_DIR}/prepare_tiny_train_bin.py" \
    --output "${TRAIN_BIN}" \
    --seq-len "${SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --num-batches "${NUM_BATCHES}" \
    --vocab-size 256 \
    --seed 42
echo ""

echo "--- Training (tiny config + tiling.json) ---"
mkdir -p "${DATA_DIR}"
TRAIN_ARGS=(
    --train-bin "${TRAIN_BIN}"
    --config "${CONFIG_JSON}"
    --tiling "${TILING_JSON}"
    --seq "${SEQ_LEN}"
    --batch "${BATCH_SIZE}"
    --epochs "${EPOCHS}"
    --max-batches "${MAX_BATCHES}"
    --lr "${LR}"
    --seed 42
)
if [[ -n "${EXECUTION_OUT:-}" ]]; then
    TRAIN_ARGS+=(--execution-out "${EXECUTION_JSON}")
    echo "Execution schedule output: ${EXECUTION_JSON}"
fi
"${BIN}" "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

echo ""
echo "--- Loss summary ---"
demo_summarize_loss "${LOG_FILE}"
