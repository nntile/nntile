#!/usr/bin/env bash
# Demo: build and run tiny BERT MLM graph training (loss should decrease).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${ROOT}/build"
cmake -S "${ROOT}" -B "${BUILD}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD}" --target bert_graph_training -j"$(nproc)"
"${BUILD}/examples/bert_graph_training"
