# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file nntile/tests/tensor_graph/CMakeLists.txt
# Tests for TensorGraph operations
#
# @version 1.1.0

if(BUILD_COVERAGE)
    setup_target_for_coverage_lcov(NAME coverage_tensor
        EXECUTABLE ctest -R tests_tensor_
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
endif()

set(TESTS_TENSOR_CATCH2
    "add"
    "adam_step"
    "adamw_step"
    "add_fiber"
    "add_fiber_inplace"
    "add_slice"
    "add_slice_inplace"
    "add_inplace"
    "clear"
    "concat"
    "conv2d_bwd_input_inplace"
    "conv2d_bwd_weight_inplace"
    "conv2d_inplace"
    "copy"
    "copy_intersection"
    "embedding"
    "embedding_backward"
    "fill"
    "gemm"
    "gelu"
    "gelu_inplace"
    "gelu_backward"
    "gather"
    "gelutanh"
    "gelutanh_inplace"
    "gelutanh_backward"
    "hypot"
    "hypot_inplace"
    "hypot_scalar_inverse"
    "invalidate"
    "multiply"
    "multiply_fiber"
    "multiply_fiber_inplace"
    "multiply_inplace"
    "multiply_slice"
    "log_scalar"
    "logsumexp"
    "mask_scalar"
    "maxsumexp"
    "norm"
    "norm_fiber"
    "norm_fiber_inplace"
    "norm_slice"
    "norm_slice_inplace"
    "pow"
    "randn"
    "relu"
    "relu_backward"
    "relu_inplace"
    "rope"
    "rope_backward"
    "scale"
    "scale_fiber"
    "scale_slice"
    "scale_inplace"
    "scatter"
    "sgd_step"
    "silu"
    "silu_inplace"
    "silu_backward"
    "softmax"
    "softmax_inplace"
    "sqrt"
    "sqrt_inplace"
    "sum"
    "sum_fiber"
    "sum_slice"
    "sumprod_fiber"
    "sumprod_slice"
    "subtract_indexed_outputs"
    "swap_two_axes"
    "total_sum_accum"
    "transpose"
)

set(TESTS_TENSOR_ROOT_CATCH2
    "axis_descriptor"
)

# CUDA-only tests (flash_* operations require n_cuda=1)
set(TESTS_TENSOR_CUDA)
if(NNTILE_USE_CUDA AND NNTILE_USE_FLASH_SDPA)
    list(APPEND TESTS_TENSOR_CUDA
        "flash_sdpa_fwd_cudnn"
        "flash_sdpa_bwd_cudnn"
    )
endif()

foreach(test IN LISTS TESTS_TENSOR_CATCH2)
    set(_tg_labels graph)
    if(test STREQUAL "add")
        set(_tg_labels "graph smoke")
    endif()
    add_test_set(TARGET_NAME tests_tensor_ops_${test}
        EXEC_NAME test_${test}
        SOURCES ops/${test}.cc
        LINK_LIBRARIES nntile Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_tensor_ops_${test}
        COV_GLOBAL coverage_tensor coverage
        LABELS ${_tg_labels}
    )
    target_include_directories(tests_tensor_ops_${test} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..)
endforeach()

foreach(test IN LISTS TESTS_TENSOR_ROOT_CATCH2)
    add_test_set(TARGET_NAME tests_tensor_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_tensor_${test}
        COV_GLOBAL coverage_tensor coverage
    )
    target_include_directories(tests_tensor_${test} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..)
endforeach()

foreach(test IN LISTS TESTS_TENSOR_CUDA)
    add_test_set(TARGET_NAME tests_tensor_ops_${test}
        EXEC_NAME test_${test}
        SOURCES ops/${test}.cc
        LINK_LIBRARIES
            nntile
            Catch2::Catch2WithMain
            $<$<BOOL:${NNTILE_USE_CUDA}>:CUDA::cudart>
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_tensor_ops_${test}
        COV_GLOBAL coverage_tensor coverage
    )
    target_include_directories(tests_tensor_ops_${test} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..)
endforeach()
