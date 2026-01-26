# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/graph/CMakeLists.txt
# Tests for nntile::graph functions
#
# @version 1.1.0

# All graph tests using Catch2
set(TESTS_CATCH2
    "logical_graph"
    "nn_graph"
    "compiled_graph"
)

# Include compiled graph tests
add_subdirectory(compiled)

# Setup a target for coverage of graph tests
if(BUILD_COVERAGE)
    setup_target_for_coverage_lcov(NAME coverage_graph
        EXECUTABLE ctest -R tests_graph_
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
endif()

foreach(test IN LISTS TESTS_CATCH2)
    add_test_set(TARGET_NAME tests_graph_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_graph_${test}
        COV_GLOBAL coverage_graph coverage
    )
endforeach()
