# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                 2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file nntile/tests/nn_graph/CMakeLists.txt
# Tests for NNGraph autograd operations
#
# @version 1.1.0

if(BUILD_COVERAGE)
    setup_target_for_coverage_lcov(NAME coverage_nn
        EXECUTABLE ctest -R tests_nn_
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
endif()

set(TESTS_NN_CATCH2
    "add"
    "add_fiber"
    "add_slice"
    "concat"
    "cross_entropy"
    "embedding"
    "gelu"
    "gelutanh"
    "gemm"
    "hypot"
    "layer_norm"
    "mse_loss"
    "multiply"
    "multiply_fiber"
    "multiply_slice"
    "norm"
    "norm_fiber"
    "norm_slice"
    "relu"
    "rms_norm"
    "rope"
    "sdpa_eager"
    "scale"
    "scale_fiber"
    "scale_slice"
    "silu"
    "softmax"
    "sum_fiber"
    "sum_slice"
    "transpose"
)

set(TESTS_NN_ROOT_CATCH2
    "grad_mode"
    "nn_graph"
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
    "hypot"
    "layer_norm"
    "mse_loss"
    "multiply"
    "multiply_fiber"
    "multiply_slice"
    "relu"
    "rms_norm"
    "sdpa_eager"
    "scale"
    "scale_fiber"
    "scale_slice"
    "silu"
    "softmax"
    "sum_fiber"
    "sum_slice"
)  # Tests with PyTorch comparison (need NNTILE_HAVE_TORCH)

foreach(test IN LISTS TESTS_NN_CATCH2)
    add_test_set(TARGET_NAME tests_nn_ops_${test}
        EXEC_NAME test_${test}
        SOURCES ops/${test}.cc
        LINK_LIBRARIES nntile Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_nn_ops_${test}
        COV_GLOBAL coverage_nn coverage
    )
    target_include_directories(tests_nn_ops_${test} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..)
    if(NNTILE_HAVE_TORCH AND test IN_LIST TESTS_NN_PYTORCH)
        target_compile_definitions(tests_nn_ops_${test} PRIVATE NNTILE_HAVE_TORCH)
        target_link_libraries(tests_nn_ops_${test} PRIVATE ${TORCH_LIBRARIES})
        target_include_directories(tests_nn_ops_${test} PRIVATE
            ${TORCH_INSTALL_PREFIX}/include)
        if(TORCH_CXX_FLAGS)
            separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
            target_compile_options(tests_nn_ops_${test} PRIVATE ${TORCH_CXX_FLAGS_LIST})
        endif()
    endif()
endforeach()

foreach(test IN LISTS TESTS_NN_ROOT_CATCH2)
    add_test_set(TARGET_NAME tests_nn_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_nn_${test}
        COV_GLOBAL coverage_nn coverage
    )
    target_include_directories(tests_nn_${test} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..)
endforeach()
