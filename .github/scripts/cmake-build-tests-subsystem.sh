#!/usr/bin/env bash
# Print -DNNTILE_TEST_SUBSYSTEM=... for CMake (see nntile/cmake/NNTileTests.cmake).
set -euo pipefail
sub="${1:?subsystem name required}"

case "$sub" in
    kernel|starpu|core|tile|tensor|nn|module|model|io)
        printf '%s\n' "-DNNTILE_TEST_SUBSYSTEM=${sub}"
        ;;
    runtime|optim|dataset)
        echo "No dedicated test tree for subsystem ${sub}" >&2
        exit 1
        ;;
    *)
        echo "unknown subsystem: $sub" >&2
        exit 1
        ;;
esac
