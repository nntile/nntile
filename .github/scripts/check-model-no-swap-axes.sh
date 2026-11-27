#!/usr/bin/env bash
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file .github/scripts/check-model-no-swap-axes.sh
#
# Native C++ torch_nntile models must not call swap_two_axes / ATen
# transpose. That op is HF-layout only and is much slower than cyclic
# model_transpose.
set -euo pipefail

root="${1:-.}"
cd "$root"

models_dir="torch_nntile/csrc/models"
if [ ! -d "$models_dir" ]; then
    echo "skip: $models_dir not found"
    exit 0
fi

bad=/tmp/bad-model-swap-axes.txt
rm -f "$bad"

# Forbidden call sites / includes (not documentation mentions).
git grep -nE \
    'swap_two_axes[[:space:]]*\(|nntile_swap_two_axes\.h|aten::transpose|\.transpose[[:space:]]*\(' \
    -- "$models_dir" >>"$bad" 2>/dev/null || true

# Models must include cyclic API only, not the HF swap / umbrella header.
git grep -nE '#include[[:space:]]*"nntile_(transpose|swap_two_axes)\.h"' \
    -- "$models_dir" >>"$bad" 2>/dev/null || true

if [ -s "$bad" ]; then
    echo "::error::Native C++ models must not use swap_two_axes / ATen"
    echo "transpose (HF-only). Use cyclic model_transpose instead:"
    cat "$bad"
    exit 1
fi

echo "Native C++ models do not use swap_two_axes"
