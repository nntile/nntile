#!/usr/bin/env bash
# List CMake executable targets for one test subsystem (for selective test-build).
set -euo pipefail
sub="${1:?subsystem name required}"
build_dir="${2:-build}"

re="$(.github/scripts/ctest-run-subsystem.sh "$sub")"
mapfile -t targets < <(
    cd "$build_dir"
    ctest -N -R "$re" 2>/dev/null \
        | sed -n 's/^[[:space:]]*Test[[:space:]]*#[0-9]*:[[:space:]]*\([^[:space:]]*\).*/\1/p'
)

if ((${#targets[@]} == 0)); then
    echo "No CTest targets matched subsystem '${sub}' (regex: ${re})" >&2
    exit 1
fi

printf '%s\n' "${targets[@]}"
