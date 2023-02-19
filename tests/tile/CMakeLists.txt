# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                          (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/tile/CMakeLists.txt
# Tests for nntile::tile functions
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @author Aleksandr Katrutsa
# @date 2023-02-10

# All unit tests without arguments to test executable
set(TESTS
    "traits"
    "tile"
    "axpy"
    "bias"
    "clear"
    "copy"
    "copy_intersection"
    "gelu"
    "gelutanh"
    "dgelu"
    "dgelutanh"
    "drelu"
    "gemm"
    "nrm2"
    "normalize"
    "randn"
    "relu"
    "sumnorm"
    "prod"
    "maxsumexp"
    "sum"
    "softmax"
    "sqrt"
    "maximum"
    "addcdiv"
    )

# Add target for local coverage
if(BUILD_COVERAGE)
    setup_target_for_coverage_lcov(NAME coverage_tile
        EXECUTABLE ctest -R tests_tile_
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
endif()

foreach(test IN LISTS TESTS)
    add_test_set(TARGET_NAME tests_tile_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_tile_${test}
        COV_GLOBAL coverage_tile coverage
        )
endforeach()

