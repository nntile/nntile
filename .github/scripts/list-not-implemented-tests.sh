#!/usr/bin/env bash
# List CTest entries labeled NotImplemented (from a configured build tree).
#
# @file .github/scripts/list-not-implemented-tests.sh
set -euo pipefail

build_dir="${1:-build}"
tests_root="${build_dir}/nntile/tests"

if [ ! -d "$tests_root" ]; then
    echo "No ${tests_root}; configure with -DBUILD_TESTS=ON first" >&2
    exit 1
fi

mapfile -t _tests < <(
    grep -rh 'LABELS "NotImplemented' "$tests_root" --include='CTestTestfile.cmake' \
        | grep -oE 'tests_(kernel|starpu|core|graph)_[a-zA-Z0-9_]+' \
        | sort -u
)

echo "NotImplemented tests (${#_tests[@]}), excluded from CI via ctest -LE NotImplemented:"
if [ "${#_tests[@]}" -eq 0 ]; then
    echo "  (none)"
    exit 0
fi
printf '  %s\n' "${_tests[@]}"
