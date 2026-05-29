#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .github/scripts/restore-nntile-objs-verify.sh
#
# Verify prebuilt subsystem archives exist before link-only libnntile build.
#
# @version 1.1.0
set -euo pipefail

build_dir="${1:-build}"
list="${2:-.github/scripts/nntile-lib-obj-subsystems.txt}"

if [ ! -f "$list" ]; then
    echo "subsystem list not found: $list" >&2
    exit 1
fi

missing=0
while IFS= read -r sub || [ -n "$sub" ]; do
  case "$sub" in
    '' | \#*) continue ;;
  esac
  lib="${build_dir}/nntile_objs_cache/libnntile_objs_${sub}.a"
  if [ ! -f "$lib" ]; then
    echo "missing archive: $lib" >&2
    missing=1
    continue
  fi
  count=$(ar t "$lib" | wc -l)
  if [ "$count" -eq 0 ]; then
    echo "empty archive: $lib" >&2
    missing=1
    continue
  fi
  echo "ok ${sub}: ${count} member(s) in archive"
done <"$list"

if [ "$missing" -ne 0 ]; then
    exit 1
fi
