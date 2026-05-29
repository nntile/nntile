#!/usr/bin/env bash
# Print -DBUILD_TESTS_* flags: enable tests only for SUBSYSTEM (inclusive layer).
set -euo pipefail
sub="${1:?subsystem name required}"

all=(kernel starpu core tile tensor nn module model io)
flags=()
for s in "${all[@]}"; do
    u=$(echo "$s" | tr '[:lower:]' '[:upper:]')
    flags+=("-DBUILD_TESTS_${u}=OFF")
done

enable() {
    local s=$1
    local u
    u=$(echo "$s" | tr '[:lower:]' '[:upper:]')
    flags+=("-DBUILD_TESTS_${u}=ON")
}

case "$sub" in
    kernel) enable kernel ;;
    starpu) enable kernel; enable starpu ;;
    core) enable kernel; enable starpu; enable core ;;
    tile) enable tile ;;
    tensor) enable tensor ;;
    nn) enable nn ;;
    module) enable module ;;
    model) enable model ;;
    io) enable io ;;
    runtime|optim|dataset)
        echo "No dedicated test tree for subsystem ${sub}" >&2
        exit 1
        ;;
    *)
        echo "unknown subsystem: $sub" >&2
        exit 1
        ;;
esac

printf '%s\n' "${flags[@]}"
