#!/bin/sh
# SPDX-License-Identifier: MIT
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
# @file tools/run_chunked_suite.sh
# Run the test suite with the smallest chunk class (1 MiB) so every tile
# larger than one class is multi-chunk. There is no runtime "is this
# chunked" flag to assert on: on this backend every tile is chunked by
# construction and NNHAUL_CHUNK_TILES=0 is refused. The gate is therefore
# the chunk class plus the tests that assert chunk counts.
#
# Usage: tools/run_chunked_suite.sh <build-dir> [ctest args...]
# @version 2.0.0
set -eu
build=${1:?usage: run_chunked_suite.sh <build-dir> [ctest args...]}
shift || true

# The library path must lead with this build tree: some runtime images
# carry a stale system-wide libnntile that would otherwise win.
LD_LIBRARY_PATH="${build}/nntile:${build}/nnhaul:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH

# The nntile fixtures share one card, so the nntile-side run needs the
# documented caps. The NNHaul engine tests must NOT inherit them: gpu_cap
# and cap_throw assert their own init arguments, which an environment cap
# overrides.
# The engine's own unit tests assert their own pool arithmetic; the requested
# class applies to both groups.
requested_class=${NNHAUL_CHUNK_SIZE_MB:-1}
requested_cuda_cap=${NNHAUL_LIMIT_CUDA_MEM:-2048}
requested_cpu_cap=${NNHAUL_LIMIT_CPU_MEM:-256}
unset NNHAUL_LIMIT_CUDA_MEM NNHAUL_LIMIT_CPU_MEM
# The requested chunk class applies to both groups: the engine tests assert
# their own pool arithmetic but pass at 1 and 2 MiB, and running them at the
# build default would leave the scripted gate unproven.
export NNHAUL_CHUNK_SIZE_MB=${requested_class}
run_group() {
    if [ "$(ctest --test-dir "${build}" -N "$@" 2>/dev/null \
            | grep -c 'Test #')" -eq 0 ]; then
        echo "run_chunked_suite: selection matched no tests" >&2
        exit 1
    fi
    ctest --test-dir "${build}" -j1 --output-on-failure --no-tests=error -LE NotImplemented "$@"
}
run_group -R '^(nnhaul_.*|trace_analyzer|trace_pipeline)$' "$@"

NNHAUL_LIMIT_CUDA_MEM=${requested_cuda_cap}
NNHAUL_LIMIT_CPU_MEM=${requested_cpu_cap}
export NNHAUL_LIMIT_CUDA_MEM NNHAUL_LIMIT_CPU_MEM
run_group \
    -E '^(nnhaul_.*|trace_analyzer|trace_pipeline|tests_kernel_sumprod_fiber|tests_kernel_total_sum_accum)$' \
    "$@"
