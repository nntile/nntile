#!/usr/bin/env bash
# Merge libnntile.so (and matching object files) from the build-lib artifact
# into the current job's build/ tree. Intended to run after `cmake` configure
# in the test job so CTest targets exist while library outputs stay prebuilt.
#
# @file .github/scripts/ci-restore-libnntile-artifact.sh

set -euo pipefail

staging=${1:?artifact staging directory (e.g. build-lib-artifact)}
build=${2:-build}

lib="${staging}/nntile/libnntile.so"
if [[ ! -f "${lib}" ]]; then
    echo "::error::Missing ${lib} from build-lib artifact" >&2
    exit 1
fi

mkdir -p "${build}/nntile" "${build}/include/nntile"
cp -a "${lib}" "${build}/nntile/"
if [[ -f "${staging}/include/nntile/defs.h" ]]; then
    cp -a "${staging}/include/nntile/defs.h" "${build}/include/nntile/"
fi

while IFS= read -r -d '' obj; do
    rel=${obj#"${staging}/"}
    dest="${build}/${rel}"
    mkdir -p "$(dirname "${dest}")"
    cp -a "${obj}" "${dest}"
done < <(find "${staging}/nntile" -name '*.o' -print0)

echo ":: Restored libnntile.so and object files from build-lib artifact"
