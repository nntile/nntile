#!/usr/bin/env bash
# Print ccache statistics after a CI build (CCACHE_DIR is set in the workflow).
#
# @file .github/scripts/ccache-print-stats.sh
set -euo pipefail

label=${1:-CI}
if ! command -v ccache >/dev/null 2>&1; then
    echo "ccache not installed; skipping stats for ${label}"
    exit 0
fi

echo "::group::ccache stats (${label})"
echo "CCACHE_DIR=${CCACHE_DIR:-<default>}"
ccache -s
echo "::endgroup::"
