#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .github/scripts/pack-nntile-test-objs.sh
#
# Tar compile-check-tests OBJECT dir for build-tests link-only restore.
#
# @version 1.1.0
set -euo pipefail

sub="${1:?subsystem name required}"
build_dir="${2:-build}"

obj_dir="${build_dir}/nntile/tests/CMakeFiles/nntile_test_objs_${sub}.dir"
out_dir="${build_dir}/nntile_test_objs_cache"
archive="${out_dir}/nntile_test_objs_${sub}.tar.gz"

if [ ! -d "$obj_dir" ]; then
    echo "test object directory not found: $obj_dir" >&2
    exit 1
fi

count=$(find "$obj_dir" -name '*.o' | wc -l)
if [ "$count" -eq 0 ]; then
    echo "no .o files under $obj_dir" >&2
    exit 1
fi

mkdir -p "$out_dir"
rm -f "$archive"
tar -czf "$archive" -C "${build_dir}/nntile/tests/CMakeFiles" \
    "nntile_test_objs_${sub}.dir"
echo "packed ${count} object(s) -> ${archive}"
