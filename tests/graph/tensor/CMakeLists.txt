# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/graph/tensor/CMakeLists.txt
# Tests for TensorGraph operations
#
# @version 1.1.0

set(TESTS_TENSOR_CATCH2 "add" "scale" "multiply" "relu" "sqrt" "copy")

foreach(test IN LISTS TESTS_TENSOR_CATCH2)
    add_test_set(TARGET_NAME tests_graph_tensor_${test}
        EXEC_NAME test_tensor_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_graph_tensor_${test}
        COV_GLOBAL coverage_graph coverage
    )
endforeach()
