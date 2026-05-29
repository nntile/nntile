#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file examples/run_roberta_graph_training_demo.sh
# Tiny RoBERTa MLM graph training, then checkpoint load on the same batch.
#
# Usage (from repo root, after building roberta_graph_training):
#   ./examples/run_roberta_graph_training_demo.sh
#
# Optional environment variables:
#   BUILD_DIR  CMake build directory (default: <repo>/build)
#   DATA_DIR   Where to write training.log (default: <build>/examples/demo_data/roberta)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=demo_common.sh
source "${SCRIPT_DIR}/demo_common.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="$(demo_resolve_build_dir "${REPO_ROOT}")"
DATA_DIR="${DATA_DIR:-${BUILD_DIR}/examples/demo_data/roberta}"
LOG_FILE="${DATA_DIR}/training.log"
BIN="${BUILD_DIR}/examples/roberta_graph_training"

demo_require_executable "${BIN}" "roberta_graph_training"

echo "=== RoBERTa graph training demo ==="
echo "Build dir:  ${BUILD_DIR}"
echo "Data dir:   ${DATA_DIR}"
echo ""

mkdir -p "${DATA_DIR}"
"${BIN}" 2>&1 | tee "${LOG_FILE}"

echo ""
echo "--- Loss summary ---"
python3 - "${LOG_FILE}" <<'PY'
import re
import sys

path = sys.argv[1]
losses = []
scratch = best = loaded = None
for line in open(path, encoding="utf-8", errors="replace"):
    m = re.search(r"^Batch \d+\s+loss=([0-9.eE+-]+)", line)
    if m:
        losses.append(float(m.group(1)))
    m = re.search(r"Scratch first loss=([0-9.eE+-]+)", line)
    if m:
        scratch = float(m.group(1))
    m = re.search(r"best training loss=([0-9.eE+-]+)", line)
    if m:
        best = float(m.group(1))
    m = re.search(r"Loaded checkpoint loss=([0-9.eE+-]+)", line)
    if m:
        loaded = float(m.group(1))
if losses:
    print(f"Training steps: {len(losses)}")
    print(f"First batch loss: {losses[0]:.6f}")
    print(f"Last batch loss:  {losses[-1]:.6f}")
if scratch is not None:
    print(f"Scratch first loss:     {scratch:.6f}")
if best is not None:
    print(f"Best training loss:     {best:.6f}")
if loaded is not None:
    print(f"Loaded checkpoint loss: {loaded:.6f}")
if scratch is not None and best is not None and best < scratch:
    print("Loss decreased over training (demo OK).")
elif losses and losses[-1] < losses[0]:
    print("Loss decreased over training (demo OK).")
else:
    print("Warning: loss did not decrease as expected.")
PY
