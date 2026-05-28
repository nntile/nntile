#!/usr/bin/env bash
# Deepen and checkout the PR head branch (expects workflow to set env vars).
set -euo pipefail

: "${PR_REF:?}"
: "${PR_HEAD_REF:?}"
: "${PR_HEAD_BRANCH:?}"
: "${PR_BASE_REF:?}"
: "${PR_COMMIT_COUNT:?}"

git config --global --add safe.directory "$(pwd)"
git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"
git fetch origin "${PR_REF}:${PR_HEAD_REF}" --depth=$((PR_COMMIT_COUNT + 1))
git checkout "${PR_HEAD_BRANCH}"
git fetch origin "${PR_BASE_REF}:${PR_BASE_REF}" --depth=100
