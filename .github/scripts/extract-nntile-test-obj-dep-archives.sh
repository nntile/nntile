#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .github/scripts/extract-nntile-test-obj-dep-archives.sh
#
# Extract dependency test OBJECT tarballs (not the primary subsystem).
#
# @version 1.1.0
set -euo pipefail

sub="${1:?subsystem name required}"
build_dir="${2:-build}"

chmod +x .github/scripts/nntile-test-obj-subsystems.sh
chmod +x .github/scripts/restore-nntile-test-objs-verify.sh

mkdir -p "${build_dir}/nntile/tests/CMakeFiles"

while IFS= read -r dep; do
    if [ "$dep" = "$sub" ]; then
        continue
    fi
    archive="${build_dir}/nntile_test_objs_cache/nntile_test_objs_${dep}.tar.gz"
    if [ ! -f "$archive" ]; then
        echo "missing dependency test object archive: $archive" >&2
        exit 1
    fi
    tar -xzf "$archive" -C "${build_dir}/nntile/tests/CMakeFiles"
    .github/scripts/restore-nntile-test-objs-verify.sh "$dep" "$build_dir"
done < <(.github/scripts/nntile-test-obj-subsystems.sh "$sub")
