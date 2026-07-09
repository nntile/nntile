#!/usr/bin/env bash
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/examples/run_gpt2_hf_cuda_vs_nntile.sh
# Train GPT-2 HF from scratch on CUDA and on nntile (same seed), then compare.
#
# Torch cannot use CUDA and device=nntile in one process, so this script runs
# two separate Python invocations and then compares checkpoints.
#
# Usage (from repo root, with torch_nntile installed and a CUDA GPU):
#   ./torch_nntile/examples/run_gpt2_hf_cuda_vs_nntile.sh
#
# Optional environment variables:
#   SEED            Init seed (default: 42)
#   EPOCHS          Epochs per device run (default: 2)
#   OUTPUT_ROOT     Run directory (default: /tmp/gpt2_hf_cuda_vs_nntile)
#   SEQ_LEN         Sequence length (default: 32)
#   BATCH_SIZE      Batch size (default: 4)
#   MAX_SEQUENCES   Cap packed sequences (default: 64)
#   LR              Learning rate (default: 1e-3)
#   PYTHON          Python interpreter (default: python3)
#   EXTRA_NNTILE_ARGS  Extra args for the nntile train invocation
#                      (e.g. --restrict-cuda)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_PY="${SCRIPT_DIR}/train_gpt2_hf_wikitext2.py"
CONFIG_JSON="${SCRIPT_DIR}/gpt2_hf_tiny_config.json"

SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/gpt2_hf_cuda_vs_nntile}"
SEQ_LEN="${SEQ_LEN:-32}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SEQUENCES="${MAX_SEQUENCES:-64}"
LR="${LR:-1e-3}"
PYTHON="${PYTHON:-python3}"
EXTRA_NNTILE_ARGS="${EXTRA_NNTILE_ARGS:-}"

CUDA_DIR="${OUTPUT_ROOT}/cuda"
NNTILE_DIR="${OUTPUT_ROOT}/nntile"
CUDA_CKPT="${CUDA_DIR}/checkpoint.pt"
NNTILE_CKPT="${NNTILE_DIR}/checkpoint.pt"

export LD_LIBRARY_PATH="${REPO_ROOT}/build/nntile:/opt/starpu/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export STARPU_SILENT="${STARPU_SILENT:-1}"
export STARPU_FXT_TRACE="${STARPU_FXT_TRACE:-0}"
export STARPU_WORKERS_NOBIND="${STARPU_WORKERS_NOBIND:-1}"

mkdir -p "${CUDA_DIR}" "${NNTILE_DIR}"

echo "=== GPT-2 HF: CUDA vs nntile (seed=${SEED}) ==="
echo "Config:  ${CONFIG_JSON}"
echo "Output:  ${OUTPUT_ROOT}"
echo ""

if ! "${PYTHON}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA is not available in this Python/torch install." >&2
    echo "Install a CUDA build of PyTorch and re-run on a machine with a GPU." >&2
    exit 1
fi

COMMON_ARGS=(
    --seed "${SEED}"
    --config "${CONFIG_JSON}"
    --epochs "${EPOCHS}"
    --seq-len "${SEQ_LEN}"
    --batch-size "${BATCH_SIZE}"
    --max-sequences "${MAX_SEQUENCES}"
    --lr "${LR}"
    --no-shuffle
)

echo "--- Train device=cuda from scratch ---"
"${PYTHON}" "${TRAIN_PY}" train \
    --device cuda \
    --output-dir "${CUDA_DIR}" \
    "${COMMON_ARGS[@]}"
echo ""

echo "--- Train device=nntile from scratch (same seed) ---"
# shellcheck disable=SC2086
"${PYTHON}" "${TRAIN_PY}" train \
    --device nntile \
    --output-dir "${NNTILE_DIR}" \
    "${COMMON_ARGS[@]}" \
    ${EXTRA_NNTILE_ARGS}
echo ""

echo "--- Compare checkpoints (relative Frobenius norms) ---"
"${PYTHON}" "${TRAIN_PY}" compare \
    --checkpoint-a "${CUDA_CKPT}" \
    --checkpoint-b "${NNTILE_CKPT}"

echo ""
echo "Done."
echo "  CUDA checkpoint:   ${CUDA_CKPT}"
echo "  nntile checkpoint: ${NNTILE_CKPT}"
