# @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
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
# @date 2022-08-09

# All unit tests without arguments to test executable
set(TESTS
    "traits"
    "tile"
    "randn"
    "copy"
    "clear"
    "gemm"
    "bias"
    "gelu"
    "relu"
    "sumnorm"
    "normalize"
    )

foreach(test IN LISTS TESTS)
    add_test_set(TARGET_NAME tests_tile_${test}
        EXEC_NAME ${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_tile_${test}
        COV_GLOBAL coverage
        )
endforeach()

