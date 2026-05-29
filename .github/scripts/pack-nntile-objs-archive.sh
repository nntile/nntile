#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .github/scripts/pack-nntile-objs-archive.sh
#
# Pack compile-check OBJECT files into a static archive for link-only CI.
#
# @version 1.1.0
set -euo pipefail

sub="${1:?subsystem name required}"
build_dir="${2:-build}"

obj_dir="${build_dir}/nntile/src/CMakeFiles/nntile_objs_${sub}.dir"
out_dir="${build_dir}/nntile_objs_cache"
lib="${out_dir}/libnntile_objs_${sub}.a"

if [ ! -d "$obj_dir" ]; then
    echo "object directory not found: $obj_dir" >&2
    exit 1
fi

mkdir -p "$out_dir"
rm -f "$lib"

mapfile -t objs < <(find "$obj_dir" -name '*.o' | sort)
if [ "${#objs[@]}" -eq 0 ]; then
    echo "no .o files under $obj_dir" >&2
    exit 1
fi

ar rcs "$lib" "${objs[@]}"
echo "packed ${#objs[@]} object(s) -> $lib"
