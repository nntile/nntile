# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                          (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/kernel/CMakeLists.txt
# Tests for nntile::kernel functions
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-01-31

# All unit tests without arguments to test executable
set(TESTS
    "bias"
    "subcopy"
    "gelu"
    "gelutanh"
    "dgelu"
    "dgelutanh"
    "drelu"
    "randn"
    "sumnorm"
    "relu"
    "hypot"
    "normalize"
    "prod"
    "maxsumexp"
    "softmax"
    "maximum"
    )

# Setup a target for coverage of kernels
if(BUILD_COVERAGE)
    setup_target_for_coverage_lcov(NAME coverage_kernel
        EXECUTABLE ctest -R tests_kernel_
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
endif()

foreach(test IN LISTS TESTS)
    add_test_set(TARGET_NAME tests_kernel_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile $<$<BOOL:${NNTILE_USE_CUDA}>:CUDA::cudart>
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_kernel_${test}
        COV_GLOBAL coverage_kernel coverage
        )
endforeach()

