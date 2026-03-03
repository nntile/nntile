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

set(TESTS_TENSOR_CATCH2
    "add"
    "add_fiber"
    "add_fiber_inplace"
    "add_slice"
    "add_slice_inplace"
    "add_inplace"
    "clear"
    "copy"
    "fill"
    "gelu"
    "gelu_backward"
    "gelutanh"
    "gelutanh_backward"
    "hypot"
    "multiply"
    "multiply_fiber"
    "multiply_inplace"
    "multiply_slice"
    "norm_fiber"
    "norm_fiber_inplace"
    "norm_slice"
    "norm_slice_inplace"
    "pow"
    "relu"
    "relu_backward"
    "relu_inplace"
    "scale"
    "scale_fiber"
    "scale_slice"
    "scale_inplace"
    "silu"
    "silu_backward"
    "sqrt"
    "sqrt_inplace"
    "sum_fiber"
    "sum_slice"
    "transpose"
)

foreach(test IN LISTS TESTS_TENSOR_CATCH2)
    add_test_set(TARGET_NAME tests_graph_tensor_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_graph_tensor_${test}
        COV_GLOBAL coverage_graph coverage
    )
    target_include_directories(tests_graph_tensor_${test} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..)
endforeach()
