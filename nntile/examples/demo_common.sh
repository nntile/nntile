#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# Shared helpers for graph training demo scripts in examples/.

set -euo pipefail

demo_script_dir() {
    cd "$(dirname "${BASH_SOURCE[1]}")" && pwd
}

demo_repo_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
    cd "${script_dir}/.." && pwd
}

demo_resolve_build_dir() {
    local root="${1:?}"
    if [[ -n "${BUILD_DIR:-}" ]]; then
        echo "${BUILD_DIR}"
        return
    fi
    if [[ -d "${root}/build/examples" ]]; then
        echo "${root}/build"
        return
    fi
    echo "${root}/build"
}

demo_require_executable() {
    local bin="${1:?}"
    local target="${2:?}"
    if [[ ! -x "${bin}" ]]; then
        echo "Missing executable: ${bin}" >&2
        echo "Build it from the repository root:" >&2
        echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release" >&2
        echo "  cmake --build build --target ${target}" >&2
        exit 1
    fi
}

demo_summarize_loss() {
    local log_file="${1:?}"
    if [[ ! -s "${log_file}" ]]; then
        echo "No training output captured."
        return
    fi
    python3 - "${log_file}" <<'PY'
import re
import sys

path = sys.argv[1]
losses = []
for line in open(path, encoding="utf-8", errors="replace"):
    m = re.search(r"loss=([0-9.eE+-]+)", line)
    if m:
        losses.append(float(m.group(1)))
if not losses:
    print("Could not parse any loss= lines from the log.")
    sys.exit(0)
print(f"Steps logged: {len(losses)}")
print(f"First loss:   {losses[0]:.6f}")
print(f"Last loss:    {losses[-1]:.6f}")
if losses[-1] < losses[0]:
    print("Loss decreased over the run (demo OK).")
else:
    print("Warning: last loss >= first loss; try more epochs or a higher --lr.")
PY
}
