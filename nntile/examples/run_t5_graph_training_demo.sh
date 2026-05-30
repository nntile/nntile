#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file examples/run_t5_graph_training_demo.sh
# Prepare tiny train.bin and run t5_graph_training for a few epochs.
#
# Usage (from repo root, after building t5_graph_training):
#   ./examples/run_t5_graph_training_demo.sh
#
# Optional environment variables:
#   BUILD_DIR  CMake build directory (default: <repo>/build)
#   DATA_DIR   Where to write train.bin (default: <build>/examples/demo_data/t5)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=demo_common.sh
source "${SCRIPT_DIR}/demo_common.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="$(demo_resolve_build_dir "${REPO_ROOT}")"
DATA_DIR="${DATA_DIR:-${BUILD_DIR}/examples/demo_data/t5}"
TRAIN_BIN="${DATA_DIR}/train.bin"
LOG_FILE="${DATA_DIR}/training.log"
BIN="${BUILD_DIR}/examples/t5_graph_training"

ENC_SEQ_LEN="${ENC_SEQ_LEN:-8}"
DEC_SEQ_LEN="${DEC_SEQ_LEN:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_BATCHES="${NUM_BATCHES:-8}"
EPOCHS="${EPOCHS:-4}"
MAX_BATCHES="${MAX_BATCHES:-32}"
LR="${LR:-0.003}"

demo_require_executable "${BIN}" "t5_graph_training"

echo "=== T5 graph training demo ==="
echo "Build dir:  ${BUILD_DIR}"
echo "Data dir:   ${DATA_DIR}"
echo ""

echo "--- Preparing tiny train.bin (seq2seq windows) ---"
python3 "${SCRIPT_DIR}/prepare_tiny_seq2seq_train_bin.py" \
    --output "${TRAIN_BIN}" \
    --enc-seq-len "${ENC_SEQ_LEN}" \
    --dec-seq-len "${DEC_SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --num-batches "${NUM_BATCHES}" \
    --vocab-size 256 \
    --seed 42
echo ""

echo "--- Training (tiny model, repeated epochs on same data) ---"
mkdir -p "${DATA_DIR}"
"${BIN}" \
    --train-bin "${TRAIN_BIN}" \
    --tiny \
    --enc-seq "${ENC_SEQ_LEN}" \
    --dec-seq "${DEC_SEQ_LEN}" \
    --batch "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --max-batches "${MAX_BATCHES}" \
    --lr "${LR}" \
    --seed 42 \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "--- Loss summary ---"
demo_summarize_loss "${LOG_FILE}"
