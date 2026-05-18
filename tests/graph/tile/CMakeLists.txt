# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/graph/tile/CMakeLists.txt
# Tests for TileGraph operations
#
# @version 1.1.0

set(TESTS_TILE_CATCH2
    "adam_step"
    "adamw_step"
    "add"
    "add_fiber"
    "add_fiber_inplace"
    "add_inplace"
    "add_mixed_tile"
    "add_slice"
    "add_slice_inplace"
    "clear"
    "copy"
    "copy_intersection"
    "conv2d_bwd_input_inplace"
    "conv2d_bwd_weight_inplace"
    "conv2d_inplace"
    "embedding"
    "embedding_backward"
    "fill"
    "gelu"
    "gelu_backward"
    "gelu_inplace"
    "gemm"
    "gelutanh"
    "gelutanh_backward"
    "gelutanh_inplace"
    "hypot"
    "hypot_inplace"
    "hypot_scalar_inverse"
    "log_scalar"
    "logsumexp"
    "mask_scalar"
    "maxsumexp"
    "mlp"
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
    "randn"
    "relu"
    "relu_backward"
    "relu_inplace"
    "rope"
    "rope_backward"
    "scale"
    "scale_fiber"
    "scale_inplace"
    "scale_slice"
    "sgd_step"
    "silu"
    "silu_backward"
    "silu_inplace"
    "softmax"
    "softmax_inplace"
    "sqrt"
    "sqrt_inplace"
    "subtract_indexed_outputs"
    "sum"
    "sum_fiber"
    "sum_slice"
    "sumprod_fiber"
    "sumprod_slice"
    "tile_graph"
    "total_sum_accum"
    "transpose"
)

foreach(test IN LISTS TESTS_TILE_CATCH2)
    add_test_set(TARGET_NAME tests_graph_tile_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_graph_tile_${test}
        COV_GLOBAL coverage_graph coverage
    )
    target_include_directories(tests_graph_tile_${test} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..)
endforeach()

if(NNTILE_USE_CUDA)
    foreach(test IN ITEMS flash_sdpa_fwd_cudnn flash_sdpa_bwd_cudnn)
        add_test_set(TARGET_NAME tests_graph_tile_${test}
            EXEC_NAME test_${test}
            SOURCES ${test}.cc
            LINK_LIBRARIES
                nntile
                Catch2::Catch2WithMain
                $<$<BOOL:${NNTILE_USE_CUDA}>:CUDA::cudart>
            COV_ENABLE ${BUILD_COVERAGE}
            COV_NAME coverage_graph_tile_${test}
            COV_GLOBAL coverage_graph coverage
        )
        target_include_directories(tests_graph_tile_${test} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/..)
    endforeach()
endif()
