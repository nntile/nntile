# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                 2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file tests/graph/nn/CMakeLists.txt
# Tests for NNGraph autograd operations
#
# @version 1.1.0

set(TESTS_NN_CATCH2
    "add"
    "add_fiber"
    "add_slice"
    "cross_entropy"
    "embedding"
    "gelu"
    "gelutanh"
    "gemm"
    "grad_mode"
    "multiply"
    "multiply_fiber"
    "multiply_slice"
    "norm"
    "norm_fiber"
    "norm_slice"
    "relu"
    "rope"
    "scale"
    "scale_fiber"
    "scale_slice"
    "silu"
    "softmax"
    "sum_fiber"
    "sum_slice"
)

set(TESTS_NN_PYTORCH
    "add"
    "add_fiber"
    "add_slice"
    "cross_entropy"
    "embedding"
    "gelu"
    "gelutanh"
    "gemm"
    "multiply"
    "multiply_fiber"
    "multiply_slice"
    "relu"
    "scale"
    "scale_fiber"
    "scale_slice"
    "silu"
    "softmax"
    "sum_fiber"
    "sum_slice"
)  # Tests with PyTorch comparison (need NNTILE_HAVE_TORCH)

foreach(test IN LISTS TESTS_NN_CATCH2)
    add_test_set(TARGET_NAME tests_graph_nn_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_graph_nn_${test}
        COV_GLOBAL coverage_graph coverage
    )
    target_include_directories(tests_graph_nn_${test} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..)
    if(NNTILE_HAVE_TORCH AND test IN_LIST TESTS_NN_PYTORCH)
        target_compile_definitions(tests_graph_nn_${test} PRIVATE NNTILE_HAVE_TORCH)
        target_link_libraries(tests_graph_nn_${test} PRIVATE ${TORCH_LIBRARIES})
        target_include_directories(tests_graph_nn_${test} PRIVATE
            ${TORCH_INSTALL_PREFIX}/include)
        if(TORCH_CXX_FLAGS)
            separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
            target_compile_options(tests_graph_nn_${test} PRIVATE ${TORCH_CXX_FLAGS_LIST})
        endif()
    endif()
endforeach()
