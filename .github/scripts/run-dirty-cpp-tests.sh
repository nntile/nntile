#!/usr/bin/env bash
# Run C++ tests affected by a branch diff (local/PR helper).
# CI uses ci-dirty-cpp-tests.sh instead.
#
# @file .github/scripts/run-dirty-cpp-tests.sh
set -e

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=dirty-cpp-tests-lib.sh
source "${script_dir}/dirty-cpp-tests-lib.sh"

if [ -d build/nntile/tests ]; then
    "${script_dir}/restore-ctest-execute-bits.sh" build/nntile/tests
fi

branch=$1
base_branch=${2:-main}
ctest_label=${3:-}
if [ -z "$branch" ]; then
    branch=$(git branch --show-current)
    echo "no branch specified: assume current branch is $branch"
fi

ctest_label_args=()
case "$ctest_label" in
    core|graph)
        ctest_label_args=(-L "$ctest_label")
        echo ":: CTest label filter: ${ctest_label}"
        ;;
    "")
        ;;
    *)
        echo "Unknown ctest label filter: ${ctest_label}" >&2
        exit 2
        ;;
esac

nntile_dirty_cpp_collect "${base_branch}" "${branch}" || exit 0

if $NNTILE_DIRTY_RUN_ALL; then
    echo ":: Core files changed, running all C++ tests"
    ctest --test-dir build -E wrappers -LE NotImplemented \
        "${ctest_label_args[@]}" --output-on-failure
    exit
fi

if ! nntile_dirty_cpp_filter_label "${ctest_label}"; then
    exit 0
fi

if [ ${#NNTILE_DIRTY_AFFECTED[@]} -eq 0 ]; then
    echo ":: Unknown changes (no pattern matched), running all C++ tests"
    ctest --test-dir build -E wrappers -LE NotImplemented \
        "${ctest_label_args[@]}" --output-on-failure
    exit
fi

regex=$(nntile_dirty_cpp_ctest_regex)
echo ":: Running ${#NNTILE_DIRTY_AFFECTED[@]} affected C++ test pattern(s):"
printf '  - %s\n' "${!NNTILE_DIRTY_AFFECTED[@]}" | sort
echo ":: CTest regex: $regex"

ctest --test-dir build -R "$regex" -E wrappers -LE NotImplemented \
    "${ctest_label_args[@]}" --output-on-failure
