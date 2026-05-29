#!/usr/bin/env bash
# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file .github/scripts/subsystem-test-pipeline.sh
#
# One configure/compile pass for test objects, one link pass, then ctest.
#
# @version 1.1.0
set -euo pipefail

sub="${1:?subsystem name required}"
build_dir="${2:-build}"

chmod +x .github/scripts/verify-catch2-deps-for-compile-check.sh
chmod +x .github/scripts/cmake-build-tests-subsystem.sh
chmod +x .github/scripts/cmake-test-build-targets.sh
chmod +x .github/scripts/nntile-prebuilt-lib-path.sh
chmod +x .github/scripts/restore-nntile-test-obj-deps.sh
chmod +x .github/scripts/ctest-run-subsystem.sh
chmod +x .github/scripts/ctest-label-exclude-subsystem.sh

.github/scripts/verify-catch2-deps-for-compile-check.sh "$build_dir"

echo "=== compile test objects (${sub} only) ==="
cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=OFF \
    -DNNTILE_COMPILE_CHECK_TESTS_SUBSYSTEM="${sub}" \
    -DNNTILE_FETCHCONTENT_DISCONNECTED=ON \
    -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_PYTHON_WRAPPERS=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
cmake --build "$build_dir" --target "nntile_compile_check_tests_${sub}" \
    -j"$(nproc)"
.github/scripts/restore-nntile-test-objs-verify.sh "$sub" "$build_dir"

echo "=== link test executables (cached objects + prebuilt libnntile) ==="
_lib=$(.github/scripts/nntile-prebuilt-lib-path.sh "$build_dir")
mapfile -t _test_flags < <(.github/scripts/cmake-build-tests-subsystem.sh "$sub")
.github/scripts/restore-nntile-test-obj-deps.sh "$sub" "$build_dir" "$sub"
cmake -S . -B "$build_dir" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=OFF \
    -DNNTILE_PRESET=full -DNNTILE_PREBUILT_LIBRARY="${_lib}" \
    -DNNTILE_LINK_CACHED_TEST_OBJECTS=ON \
    -DBUILD_TESTS=ON -DBUILD_TESTS_PYTORCH=OFF \
    -DNNTILE_FETCHCONTENT_DISCONNECTED=ON \
    -DBUILD_EXAMPLES=OFF -DBUILD_PYTHON_WRAPPERS=OFF \
    "${_test_flags[@]}"
mapfile -t _test_targets < <(.github/scripts/cmake-test-build-targets.sh \
    "$sub" "$build_dir")
echo "=== ninja plan (link test executables only) ==="
ninja -C "$build_dir" -n "${_test_targets[@]}" 2>&1 | tee /tmp/build-tests-ninja-plan.txt
if grep -E 'CXX_COMPILER|CUDA_COMPILER' /tmp/build-tests-ninja-plan.txt \
    | grep -qE 'nntile/src/|nntile/tests/CMakeFiles/nntile_test_objs_.*\.cc\.o:'; then
    echo "unexpected library or test source compile" >&2
    grep -E 'CXX_COMPILER|CUDA_COMPILER' /tmp/build-tests-ninja-plan.txt \
        | grep -E 'nntile/src/|nntile_test_objs_.*\.cc\.o:' >&2 || true
    exit 1
fi
echo "Linking ${#_test_targets[@]} test executable(s)"
cmake --build "$build_dir" --target "${_test_targets[@]}" -j"$(nproc)"

echo "=== ctest ==="
export LD_LIBRARY_PATH="/opt/starpu/lib:${LD_LIBRARY_PATH:-}"
_re=$(.github/scripts/ctest-run-subsystem.sh "$sub")
_le=$(.github/scripts/ctest-label-exclude-subsystem.sh "$sub")
cd "$build_dir" && ctest -R "${_re}" -LE "${_le}" -j"$(nproc)" --output-on-failure
