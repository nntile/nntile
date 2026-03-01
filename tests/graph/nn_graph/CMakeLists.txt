# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                 2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file tests/graph/nn_graph/CMakeLists.txt
# Tests for NNGraph autograd operations
#
# @version 1.1.0

set(TESTS_NN_GRAPH_CATCH2
    "add"
    "add_fiber"
    "grad_mode"
    "gelu"
    "gemm"
    "sum_fiber"
)

foreach(test IN LISTS TESTS_NN_GRAPH_CATCH2)
    add_test_set(TARGET_NAME tests_graph_nn_graph_${test}
        EXEC_NAME test_nn_graph_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_graph_nn_graph_${test}
        COV_GLOBAL coverage_graph coverage
    )
endforeach()
