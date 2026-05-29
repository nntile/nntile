#!/usr/bin/env bash
# CI: plan, build, and run dirty C++ tests (see docs/build/dirty-cpp-tests.md).
#
# @file .github/scripts/ci-dirty-cpp-tests.sh
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
cd "${repo_root}"

# shellcheck source=dirty-cpp-tests-lib.sh
source "${script_dir}/dirty-cpp-tests-lib.sh"

cmd=${1:?usage: ci-dirty-cpp-tests.sh plan|build|run}
base_ref=${DIFF_BASE:-${GITHUB_BASE_REF:-graph_api}}

ensure_git_repo() {
    if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "::error::Not a git repository (cwd=$(pwd))" >&2
        exit 1
    fi
}

# Resolve a commit SHA for the diff base (never a branch name like origin/graph_api).
resolve_merge_base() {
    if [ -n "${NNTILE_DIFF_BASE:-}" ]; then
        if git rev-parse "${NNTILE_DIFF_BASE}" >/dev/null 2>&1; then
            git rev-parse "${NNTILE_DIFF_BASE}"
            return 0
        fi
        echo "::warning::NNTILE_DIFF_BASE=${NNTILE_DIFF_BASE} is not valid; refetching"
    fi

    local remote_base="origin/${base_ref}"
    git fetch --no-tags origin \
        "${base_ref}:refs/remotes/${remote_base}" --depth=500 2>/dev/null \
        || git fetch --no-tags origin "${base_ref}" --depth=500 2>/dev/null \
        || true

    if git rev-parse "${remote_base}" >/dev/null 2>&1; then
        git merge-base "${remote_base}" HEAD
        return 0
    fi

    echo "::warning::Cannot resolve origin/${base_ref}; dirty scope = full test suite" >&2
    return 1
}

load_plan() {
    ensure_git_repo
    local mb
    if ! mb=$(resolve_merge_base); then
        nntile_dirty_cpp_reset_plan
        NNTILE_DIRTY_RUN_ALL=true
        eval "$(nntile_dirty_cpp_emit_plan)"
        return 0
    fi
    nntile_dirty_cpp_collect "${mb}" HEAD || true
    eval "$(nntile_dirty_cpp_emit_plan)"
}

case "${cmd}" in
    plan)
        load_plan
        if [ "${NNTILE_DIRTY_SKIP:-0}" = 1 ]; then
            echo ":: No dirty C++ tests for this diff"
            exit 0
        fi
        if [ "${NNTILE_DIRTY_RUN_ALL:-0}" = 1 ]; then
            echo ":: Dirty scope: full C++ test suite"
        else
            echo ":: Dirty CMake targets:${NNTILE_DIRTY_CMAKE_TARGETS}"
            echo ":: Dirty CTest regex:${NNTILE_DIRTY_CTEST_REGEX}"
        fi
        ;;
    build)
        load_plan
        if [ "${NNTILE_DIRTY_SKIP:-0}" = 1 ]; then
            echo ":: Nothing to build"
            exit 0
        fi
        if [ "${NNTILE_DIRTY_RUN_ALL:-0}" = 1 ]; then
            echo ":: Building all test targets"
            cmake --build build -j"$(nproc)"
        else
            read -ra _targets <<< "${NNTILE_DIRTY_CMAKE_TARGETS}"
            echo ":: Building ${_targets[*]}"
            cmake --build build --target "${_targets[@]}" -j"$(nproc)"
        fi
        ;;
    run)
        load_plan
        if [ "${NNTILE_DIRTY_SKIP:-0}" = 1 ]; then
            echo ":: Nothing to run"
            exit 0
        fi
        if [ -d build/nntile/tests ]; then
            "${script_dir}/restore-ctest-execute-bits.sh" build/nntile/tests
        fi
        export LD_LIBRARY_PATH="${PWD}/build/nntile:${PWD}/build/_deps/catch2-build/src:${LD_LIBRARY_PATH:-}"
        if [ -x "${script_dir}/list-not-implemented-tests.sh" ]; then
            "${script_dir}/list-not-implemented-tests.sh" build || true
        fi
        if [ "${NNTILE_DIRTY_RUN_ALL:-0}" = 1 ]; then
            echo ":: Running full C++ test suite (minus NotImplemented)"
            ctest --test-dir build -E wrappers -LE NotImplemented \
                -j"$(nproc)" --output-on-failure
        else
            echo ":: Running dirty CTest regex:${NNTILE_DIRTY_CTEST_REGEX}"
            ctest --test-dir build -R "${NNTILE_DIRTY_CTEST_REGEX}" \
                -E wrappers -LE NotImplemented \
                -j"$(nproc)" --output-on-failure
        fi
        ;;
    *)
        echo "unknown command: ${cmd}" >&2
        exit 2
        ;;
esac
