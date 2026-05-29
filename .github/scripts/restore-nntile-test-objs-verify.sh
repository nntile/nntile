#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .github/scripts/restore-nntile-test-objs-verify.sh
#
# Verify cached test OBJECT tree exists before link-only build-tests.
#
# @version 1.1.0
set -euo pipefail

sub="${1:?subsystem name required}"
build_dir="${2:-build}"

obj_dir="${build_dir}/nntile/tests/CMakeFiles/nntile_test_objs_${sub}.dir"
if [ ! -d "$obj_dir" ]; then
    echo "missing test object dir: $obj_dir" >&2
    exit 1
fi

count=$(find "$obj_dir" -name '*.o' | wc -l)
if [ "$count" -eq 0 ]; then
    echo "no .o files in $obj_dir" >&2
    exit 1
fi

echo "ok ${sub}: ${count} cached test object file(s)"
