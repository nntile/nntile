# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/layer/CMakeLists.txt
# Tests for nntile::layer functions
#
# @version 1.1.0

# All unit tests for layers
set(TESTS
    #"linear"
    #"gelu"
    #"gelutanh"
    #"mlp"
    )

# Add target for local coverage
if(BUILD_COVERAGE)
    setup_target_for_coverage_lcov(NAME coverage_layer
        EXECUTABLE ctest -R tests_layer_
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
endif()

foreach(test IN LISTS TESTS)
    add_test_set(TARGET_NAME tests_layer_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile
        MPI_NUMPROC 4
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_layer_${test}
        COV_GLOBAL coverage_layer coverage
        )
endforeach()
