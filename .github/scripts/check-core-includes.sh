#!/usr/bin/env bash
# Fail if any #include uses nntile/<core-layer>/ without the core/ segment.
set -euo pipefail

root="${1:-.}"
cd "$root"

pattern='#include[[:space:]]*[<"]nntile/(kernel|starpu|tile|tensor|context|constants|logger|base_types|defs\.h)(/|\.hh|")'
bad=/tmp/bad-includes.txt
rm -f "$bad"

git grep -E "$pattern" -- \
    ':(exclude)external' \
    ':(exclude)build' \
    ':(exclude).git' \
    ':(exclude)scripts/migrate_core_split.py' \
    >"$bad" 2>/dev/null || true

if [ -s "$bad" ]; then
    echo "::error::Forbidden include paths (use nntile/...):"
    cat "$bad"
    exit 1
fi

echo ":: All core includes use the nntile/ prefix"
