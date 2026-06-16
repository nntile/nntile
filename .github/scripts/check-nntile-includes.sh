#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .github/scripts/check-nntile-includes.sh
#
# Enforce flat public includes: no nntile/core/ or nntile/graph/ path segments.
#
# @version 1.1.0
set -euo pipefail

root="${1:-.}"
cd "$root"

bad=/tmp/bad-nntile-includes.txt
rm -f "$bad"

git grep -E '#include[[:space:]]*[<"]nntile/(core|graph)/' -- \
    ':(exclude)external' \
    ':(exclude)build' \
    ':(exclude).git' \
    ':(exclude)nntile/' \
    ':(exclude)scripts/' \
    ':(exclude)wrappers/python' \
    ':(exclude)*.md' \
    >>"$bad" 2>/dev/null || true

legacy='#include[[:space:]]*[<"]nntile/(kernel|starpu|tile|tensor)/'
git grep -E "$legacy" -- \
    ':(exclude)external' \
    ':(exclude)build' \
    ':(exclude).git' \
    ':(exclude)nntile/' \
    ':(exclude)scripts/' \
    >>"$bad" 2>/dev/null || true

# Documentation may show legacy include examples.
if [ -s "$bad" ]; then
    grep -v '\.md:' "$bad" >"${bad}.filtered" || true
    mv "${bad}.filtered" "$bad"
fi

if [ -s "$bad" ]; then
    echo "::error::Forbidden include paths (use flat nntile/... layout):"
    cat "$bad"
    exit 1
fi

echo "All checked includes use the flat nntile/ layout"
