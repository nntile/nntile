# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file cmake/NNTileKernelScope.cmake
# Transformer-focused kernel scope: the disabled-family cache variable and
# the test-list filter. Included unconditionally by the top-level CMake so
# a tests-only build (BUILD_LIBNNTILE=OFF) can filter its test lists; the
# product build configures it into defs.h and exposes it to the source
# guards.
#
# A tests-only build adopts the installed product's list from
# nntileConfig.cmake after find_package(nntile), so the tests match the
# library they link against rather than this source tree's default.
#
# @version 2.0.0

# Transformer-focused scope: families outside the transformer model and
# training paths are disabled at the public op level, and their tests are
# not registered. Set -DNNTILE_DISABLED_KERNELS= (empty) to re-enable all.
set(NNTILE_DISABLED_KERNELS
    "conv2d_inplace;conv2d_bwd_input_inplace;conv2d_bwd_weight_inplace;isfinite;pow;sqrt;sqrt_inplace"
    CACHE STRING
    "Semicolon-separated kernel families disabled in this build")

# Remove disabled kernel families from a test-list variable. Called by the
# per-subsystem test CMake files after they build their op lists.
function(nntile_filter_disabled_kernel_tests)
    foreach(_var IN LISTS ARGN)
        if(DEFINED ${_var})
            foreach(_k IN LISTS NNTILE_DISABLED_KERNELS)
                list(REMOVE_ITEM ${_var} ${_k})
            endforeach()
            set(${_var} "${${_var}}" PARENT_SCOPE)
        endif()
    endforeach()
endfunction()
