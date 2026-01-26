# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                 2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/graph/compiled/CMakeLists.txt
# Tests for nntile::graph::compiled functions
#
# @version 1.1.0

# All compiled graph operation tests using Catch2
set(TESTS_COMPILED_CATCH2
    "add"
    "add_fiber"
    "add_fiber_inplace"
    "add_inplace"
    "add_slice"
    "add_slice_inplace"
    "clear"
    "copy"
    "embedding"
    "embedding_backward"
    "fill"
    "flash_sdpa_fwd_cudnn"
    # "gather"
    "gelu"
    "gelu_backward"
    "gelu_inplace"
    "gelutanh"
    "gelutanh_backward"
    "gelutanh_inplace"
    "gemm"
    "hypot"
    "hypot_inplace"
    "hypot_scalar_inverse"
    "mask_scalar"
    "multiply"
    "multiply_fiber"
    "multiply_fiber_inplace"
    "multiply_inplace"
    "multiply_slice"
    "norm"
    "norm_fiber"
    "norm_fiber_inplace"
    "norm_slice"
    "norm_slice_inplace"
    "pow"
    "pow_inplace"
    "relu"
    "relu_backward"
    "relu_inplace"
    "scale"
    "scale_fiber"
    "scale_slice"
    "adam_step"
    "adamw_step"
    "sgd_step"
    "scale_inplace"
    "silu"
    "silu_backward"
    "silu_inplace"
    "softmax"
    "softmax_inplace"
    "sqrt"
    "sqrt_inplace"
    "rope"
    "rope_backward"
    "sum"
    "sum_fiber"
    "sum_slice"
    "total_sum_accum"
    "transpose"
)

foreach(test IN LISTS TESTS_COMPILED_CATCH2)
    add_test_set(TARGET_NAME tests_graph_compiled_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_graph_compiled_${test}
        COV_GLOBAL coverage_graph coverage
    )
endforeach()