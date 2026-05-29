#!/usr/bin/env bash
# Make the repository usable for git diff inside a GitHub Actions container job.
# checkout@v6 often leaves a .git gitfile pointing at /home/runner/work/... which
# is not visible inside the ubuntu:24.04 container at /__w/...
#
# @file .github/scripts/ci-ensure-git-worktree.sh
set -euo pipefail

ws="${GITHUB_WORKSPACE:-$PWD}"
cd "$ws"
git config --global --add safe.directory "$ws"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo ":: Git worktree OK at ${ws}"
    exit 0
fi

try_gitdir() {
    local gitdir=$1
    [ -n "$gitdir" ] && [ -d "$gitdir" ] || return 1
    export GIT_DIR="$gitdir"
    export GIT_WORK_TREE="$ws"
    git rev-parse --is-inside-work-tree >/dev/null 2>&1
}

if [ -f .git ]; then
    raw_gitdir=$(sed 's/^gitdir: //' .git | tr -d '[:space:]')
    if try_gitdir "$raw_gitdir"; then
        echo ":: Git worktree OK via GIT_DIR=${raw_gitdir}"
        exit 0
    fi
    for prefix in \
        "/home/runner/work" \
        "/github/workspace"; do
        if [[ "$raw_gitdir" == "${prefix}"* ]]; then
            mapped="/__w${raw_gitdir#"${prefix}"}"
            if try_gitdir "$mapped"; then
                echo ":: Git worktree OK via mapped GIT_DIR=${mapped}"
                exit 0
            fi
        fi
    done
fi

echo ":: Re-initializing git metadata inside container (gitfile not usable)"
rm -rf .git
git init
# checkout credential helper is not inherited after git init; use GITHUB_TOKEN.
if [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "::error::GITHUB_TOKEN is required to re-fetch the repository in container" >&2
    exit 1
fi
git remote add origin \
  "https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git -c protocol.version=2 fetch --no-tags --depth=500 origin "${GITHUB_SHA:?GITHUB_SHA required}"
git checkout -f FETCH_HEAD

if [ -n "${NNTILE_DIFF_BASE:-}" ]; then
    git fetch --no-tags --depth=1 origin "${NNTILE_DIFF_BASE}" || true
fi

echo ":: Git worktree re-initialized at ${ws}"
