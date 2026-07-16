# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file nntile/tests/tile_graph/CMakeLists.txt
# Tests for TileGraph operations
#
# @version 1.1.0

if(BUILD_COVERAGE)
    setup_target_for_coverage_lcov(NAME coverage_tile
        EXECUTABLE ctest -R tests_tile_
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
endif()

set(TESTS_TILE_CATCH2
    "adam_step"
    "adamw_step"
    "add"
    "add_fiber"
    "add_fiber_inplace"
    "add_inplace"
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
    "total_sum_accum"
    "transpose"
)

set(TESTS_TILE_ROOT_CATCH2
    "append_tensor_graph_phase"
    "add_mixed_tile"
    "execution_schedule"
    "tile_graph"
)

foreach(test IN LISTS TESTS_TILE_CATCH2)
    add_test_set(TARGET_NAME tests_tile_ops_${test}
        EXEC_NAME test_${test}
        SOURCES ops/${test}.cc
        LINK_LIBRARIES nntile::nntile Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_tile_ops_${test}
        COV_GLOBAL coverage_tile coverage
    )
    target_include_directories(tests_tile_ops_${test} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..)
endforeach()

foreach(test IN LISTS TESTS_TILE_ROOT_CATCH2)
    add_test_set(TARGET_NAME tests_tile_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile::nntile Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_tile_${test}
        COV_GLOBAL coverage_tile coverage
    )
    target_include_directories(tests_tile_${test} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..)
endforeach()

if(NNTILE_USE_CUDA AND NNTILE_USE_FLASH_SDPA)
    foreach(test IN ITEMS flash_sdpa_fwd_cudnn flash_sdpa_bwd_cudnn)
        add_test_set(TARGET_NAME tests_tile_ops_${test}
            EXEC_NAME test_${test}
            SOURCES ops/${test}.cc
            LINK_LIBRARIES
                nntile::nntile
                Catch2::Catch2WithMain
                $<$<BOOL:${NNTILE_USE_CUDA}>:CUDA::cudart>
            COV_ENABLE ${BUILD_COVERAGE}
            COV_NAME coverage_tile_ops_${test}
            COV_GLOBAL coverage_tile coverage
        )
        target_include_directories(tests_tile_ops_${test} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/..)
    endforeach()
endif()
