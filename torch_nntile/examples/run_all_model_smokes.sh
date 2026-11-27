#!/usr/bin/env bash
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# Run tiny training smokes for each torch_nntile model example (no tiling).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EXAMPLES="${SCRIPT_DIR}"

export PYTHONPATH="${REPO_ROOT}/torch_nntile${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -d "${REPO_ROOT}/build/nntile" ]]; then
  export LD_LIBRARY_PATH="${REPO_ROOT}/build/nntile${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -d /opt/starpu/lib ]]; then
  export LD_LIBRARY_PATH="/opt/starpu/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export STARPU_SILENT="${STARPU_SILENT:-1}"
export STARPU_FXT_TRACE="${STARPU_FXT_TRACE:-0}"
export STARPU_WORKERS_NOBIND="${STARPU_WORKERS_NOBIND:-1}"

NCPU="${NCPU:-1}"
STEPS="${STEPS:-2}"
SEED="${SEED:-0}"

run_py() {
  echo "==> $*"
  python3 "$@"
}

cd "${REPO_ROOT}"

run_py "${EXAMPLES}/train_llama.py" train --steps "${STEPS}" --seed "${SEED}" --ncpu "${NCPU}"
run_py "${EXAMPLES}/train_gpt_neo.py" train --steps "${STEPS}" --seed "${SEED}" --ncpu "${NCPU}"
run_py "${EXAMPLES}/train_gpt_neox.py" train --steps "${STEPS}" --seed "${SEED}" --ncpu "${NCPU}"
run_py "${EXAMPLES}/train_bert.py" train --steps "${STEPS}" --seed "${SEED}" --ncpu "${NCPU}"
run_py "${EXAMPLES}/train_roberta.py" train --steps "${STEPS}" --seed "${SEED}" --ncpu "${NCPU}"
run_py "${EXAMPLES}/train_t5.py" train --steps "${STEPS}" --seed "${SEED}" --ncpu "${NCPU}"
run_py "${EXAMPLES}/train_dit.py" train --steps "${STEPS}" --seed "${SEED}" --ncpu "${NCPU}"

echo "==> train_deep_relu_mnist.py --help"
python3 "${EXAMPLES}/train_deep_relu_mnist.py" --help >/dev/null
echo "train_deep_relu_mnist.py --help ok"

GPT2_OUT="${TMPDIR:-/tmp}/torch_nntile_gpt2_smoke_$$"
mkdir -p "${GPT2_OUT}"
run_py "${EXAMPLES}/train_gpt2.py" train \
  --seed "${SEED}" \
  --epochs 1 \
  --max-sequences 4 \
  --batch-size 2 \
  --seq-len 8 \
  --output-dir "${GPT2_OUT}" \
  --ncpu "${NCPU}" \
  --ncuda 0 \
  --restrict-cpu \
  --no-shuffle

echo "All model smokes finished."
