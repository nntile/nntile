#!/usr/bin/env bash
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# Run all torch-native (stock HF / CNN / DiT) tiny training smokes.
# Skips specialized torch_nntile.models.* scripts (disabled under
# NNTILE_TORCH_NATIVE_OPS).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXAMPLES="${SCRIPT_DIR}"

export PYTHONPATH="${REPO_ROOT}/torch_nntile${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -d "${REPO_ROOT}/build/nntile" ]]; then
  export LD_LIBRARY_PATH="${REPO_ROOT}/build/nntile${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -d "${REPO_ROOT}/build/torch_nntile" ]]; then
  export LD_LIBRARY_PATH="${REPO_ROOT}/build/torch_nntile${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -d /opt/starpu/lib ]]; then
  export LD_LIBRARY_PATH="/opt/starpu/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export STARPU_SILENT="${STARPU_SILENT:-1}"
export STARPU_FXT_TRACE="${STARPU_FXT_TRACE:-0}"
export STARPU_WORKERS_NOBIND="${STARPU_WORKERS_NOBIND:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

NCPU="${NCPU:-1}"
NCUDA="${NCUDA:-0}"
STEPS="${STEPS:-1}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-nntile}"
BATCH_SIZE="${BATCH_SIZE:-2}"
SEQ_LEN="${SEQ_LEN:-16}"

PYTHON="${PYTHON:-python3}"
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
fi

run_py() {
  echo "==> $*"
  "${PYTHON}" "$@"
}

cd "${REPO_ROOT}"

# HuggingFace stock models
for script in \
  train_gpt_neo_hf.py \
  train_gpt_neox_hf.py \
  train_llama_hf.py \
  train_bert_hf.py \
  train_roberta_hf.py \
  train_t5_hf.py
do
  run_py "${EXAMPLES}/${script}" train \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --steps "${STEPS}" \
    --seq-len "${SEQ_LEN}" \
    --batch-size 1 \
    --ncpu "${NCPU}" \
    --ncuda "${NCUDA}"
done

GPT2_OUT="${TMPDIR:-/tmp}/torch_nntile_gpt2_hf_smoke_$$"
mkdir -p "${GPT2_OUT}"
GPT2_ARGS=(
  "${EXAMPLES}/train_gpt2_hf.py" train
  --device "${DEVICE}"
  --seed "${SEED}"
  --data-seed "${SEED}"
  --epochs 1
  --max-sequences 4
  --batch-size 2
  --seq-len 8
  --output-dir "${GPT2_OUT}"
  --ncpu "${NCPU}"
  --ncuda "${NCUDA}"
  --no-shuffle
)
# train_gpt2_hf does not auto-restrict; match HF/CNN/DiT commons.
if [[ "${DEVICE}" == "nntile" ]]; then
  if [[ "${NCUDA}" != "0" ]]; then
    GPT2_ARGS+=(--restrict-cuda)
  else
    GPT2_ARGS+=(--restrict-cpu)
  fi
fi
run_py "${GPT2_ARGS[@]}"

# CNN torch-native stacks
for script in \
  train_lenet_tiny.py \
  train_resnet_tiny.py \
  train_vgg_tiny.py \
  train_mobilenet_tiny.py \
  train_unet_tiny.py \
  train_unet_modern_tiny.py
do
  run_py "${EXAMPLES}/${script}" train \
    --device "${DEVICE}" \
    --seed "${SEED}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --ncpu "${NCPU}" \
    --ncuda "${NCUDA}"
done

# Diffusers DiT
run_py "${EXAMPLES}/train_dit_hf.py" train \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --ncpu "${NCPU}" \
  --ncuda "${NCUDA}"

echo "All torch-native model smokes finished (device=${DEVICE})."
