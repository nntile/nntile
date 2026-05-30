#!/usr/bin/env bash
# GitHub Actions upload/download-artifact does not preserve the executable bit.
# CTest then fails with BAD_COMMAND / "permission denied" on a separate runner.

set -euo pipefail

root=${1:-build/nntile/tests}
if [ ! -d "$root" ]; then
    exit 0
fi

restored=0
while IFS= read -r -d '' f; do
    case $(file -b "$f") in
        ELF*)
            chmod +x "$f"
            restored=$((restored + 1))
            ;;
    esac
done < <(find "$root" -type f ! -name '*.cmake' -print0)

echo ":: Restored execute bit on ${restored} CTest binary(ies) under ${root}"
