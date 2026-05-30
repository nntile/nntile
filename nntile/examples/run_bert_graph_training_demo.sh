#!/usr/bin/env bash
# Demo: tiny BERT MLM graph training. Scratch loss should be much higher than
# after training or loading a checkpoint for the next step (not zero).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/build"
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH:-/opt/starpu/lib/pkgconfig}"
export LD_LIBRARY_PATH="/opt/starpu/lib:${LD_LIBRARY_PATH:-}"
cmake -S "${ROOT}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -GNinja
cmake --build "${BUILD}" --target bert_graph_training -j"$(nproc)"
"${BUILD}/examples/bert_graph_training"
