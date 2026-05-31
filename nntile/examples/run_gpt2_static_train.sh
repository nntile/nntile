#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file examples/run_gpt2_static_train.sh
# End-to-end: generate execution.json, then train with --execution.
#
# Usage (from repo root, after building gpt2_graph_training):
#   ./nntile/examples/run_gpt2_static_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=demo_common.sh
source "${SCRIPT_DIR}/demo_common.sh"

REPO_ROOT="$(demo_repo_root)"
BUILD_DIR="$(demo_resolve_build_dir "${REPO_ROOT}")"
DATA_DIR="${DATA_DIR:-${BUILD_DIR}/examples/demo_data/gpt2_static}"
TRAIN_BIN="${DATA_DIR}/train.bin"
EXECUTION_JSON="${DATA_DIR}/execution.json"
LOG_GEN="${DATA_DIR}/generate.log"
LOG_TRAIN="${DATA_DIR}/train.log"
BIN="$(demo_example_bin "${BUILD_DIR}" gpt2_graph_training)"
CONFIG_JSON="${SCRIPT_DIR}/demo_configs/gpt2_tiny_config.json"
TILING_JSON="${SCRIPT_DIR}/demo_configs/gpt2_tiny_tiling.json"

SEQ_LEN="${SEQ_LEN:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_BATCHES="${NUM_BATCHES:-8}"
EPOCHS="${EPOCHS:-4}"
MAX_BATCHES_TRAIN="${MAX_BATCHES_TRAIN:-32}"
LR="${LR:-0.003}"

demo_require_executable "${BIN}" "gpt2_graph_training"

echo "=== GPT-2 static train (generate schedule, then reload) ==="
mkdir -p "${DATA_DIR}"

if [[ ! -f "${TRAIN_BIN}" ]]; then
    python3 "${SCRIPT_DIR}/prepare_tiny_train_bin.py" \
        --output "${TRAIN_BIN}" \
        --seq-len "${SEQ_LEN}" \
        --batch-size "${BATCH_SIZE}" \
        --num-batches "${NUM_BATCHES}" \
        --vocab-size 256 \
        --seed 42
fi

COMMON_ARGS=(
    --train-bin "${TRAIN_BIN}"
    --config "${CONFIG_JSON}"
    --tiling "${TILING_JSON}"
    --seq "${SEQ_LEN}"
    --batch "${BATCH_SIZE}"
    --lr "${LR}"
    --seed 42
)

echo "--- Step 1: write execution.json (1 batch) ---"
"${BIN}" "${COMMON_ARGS[@]}" \
    --max-batches 1 \
    --epochs 1 \
    --execution-out "${EXECUTION_JSON}" 2>&1 | tee "${LOG_GEN}"

if [[ ! -f "${EXECUTION_JSON}" ]]; then
    echo "error: ${EXECUTION_JSON} was not created" >&2
    exit 1
fi

echo "--- Step 2: train with --execution ---"
"${BIN}" "${COMMON_ARGS[@]}" \
    --execution "${EXECUTION_JSON}" \
    --epochs "${EPOCHS}" \
    --max-batches "${MAX_BATCHES_TRAIN}" 2>&1 | tee "${LOG_TRAIN}"

echo ""
echo "--- Loss summary (training) ---"
demo_summarize_loss "${LOG_TRAIN}"

if ! read -r FIRST_LOSS LAST_LOSS < <(python3 - "${LOG_TRAIN}" <<'PY'
import re
import sys

path = sys.argv[1]
first_batch_loss = None
last_loss = None
batch_loss = re.compile(
    r"^Batch\s+(\d+)\s+.*\bloss=([0-9.eE+-]+)")
with open(path, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = batch_loss.search(line)
        if not m:
            continue
        step = int(m.group(1))
        loss = float(m.group(2))
        last_loss = loss
        if step == 0 and first_batch_loss is None:
            first_batch_loss = loss
if first_batch_loss is None or last_loss is None:
    sys.exit(1)
print(f"{first_batch_loss} {last_loss}")
PY
); then
    echo "error: could not parse batch-0 and final loss from ${LOG_TRAIN}" >&2
    exit 1
fi
python3 - "${FIRST_LOSS}" "${LAST_LOSS}" <<'PY'
import sys

first, last = float(sys.argv[1]), float(sys.argv[2])
if not (last < first):
    raise SystemExit(f"expected loss to decrease: first={first} last={last}")
print(f"Loss decreased: {first} -> {last}")
PY

echo "Done. execution.json: ${EXECUTION_JSON}"
