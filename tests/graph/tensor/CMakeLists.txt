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
    "adam_step"
    "adamw_step"
    "add_fiber"
    "axis_descriptor"
    "dimension_group"
    "add_fiber_inplace"
    "add_slice"
    "add_slice_inplace"
    "add_inplace"
    "clear"
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
    "total_sum_accum"
    "transpose"
)

# CUDA-only tests (flash_* operations require n_cuda=1)
set(TESTS_TENSOR_CUDA)
if(NNTILE_USE_CUDA)
    list(APPEND TESTS_TENSOR_CUDA
        "flash_sdpa_fwd_cudnn"
        "flash_sdpa_bwd_cudnn"
    )
endif()

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

foreach(test IN LISTS TESTS_TENSOR_CUDA)
    add_test_set(TARGET_NAME tests_graph_tensor_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES
            nntile
            Catch2::Catch2WithMain
            $<$<BOOL:${NNTILE_USE_CUDA}>:CUDA::cudart>
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_graph_tensor_${test}
        COV_GLOBAL coverage_graph coverage
    )
    target_include_directories(tests_graph_tensor_${test} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..)
endforeach()
