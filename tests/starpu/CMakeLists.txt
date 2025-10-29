# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/starpu/CMakeLists.txt
# Tests for nntile::starpu functions
#
# @version 1.1.0

# All unit tests without arguments to test executable
set(TESTS
    "accumulate"
    "accumulate_hypot"
    "accumulate_maxsumexp"
    "adam_step"
    "adamw_step"
    "add"
    "add_fiber"
    "add_fiber_inplace"
    "add_inplace"
    "add_slice"
    "add_slice_inplace"
    "clear"
    "codelet"
    "conv2d_bwd_input_inplace"
    "conv2d_bwd_weight_inplace"
    "conv2d_inplace"
    "copy"
    "embedding"
    "embedding_backward"
    "fill"
    "flash_sdpa_fwd_cudnn"
    "gelu"
    "gelu_inplace"
    "gelu_backward"
    "gelutanh"
    "gelutanh_backward"
    "gelutanh_inplace"
    "gemm"
    "handle"
    "hypot_inplace"
    "hypot"
    "hypot_scalar_inverse"
    "log_scalar"
    "logsumexp"
    "mask_scalar"
    "maxsumexp"
    "norm"
    "norm_fiber"
    "norm_fiber_inplace"
    "norm_slice_inplace"
    "pow"
    "multiply"
    "multiply_fiber_inplace"
    "multiply_fiber"
    "multiply_inplace"
    "multiply_slice"
    "randn"
    "relu_inplace"
    "relu_backward"
    "relu"
    "rope"
    "rope_backward"
    "scale"
    "scale_inplace"
    "scale_fiber"
    "scale_slice"
    "silu_backward"
    "silu"
    "silu_inplace"
    "softmax"
    "softmax_inplace"
    "sqrt"
    "sqrt_inplace"
    "subcopy"
    "subtract_indexed_column"
    "subtract_indexed_outputs"
    "sum_fiber"
    "sum_slice"
    "sumprod_fiber"
    "sumprod_slice"
    "total_sum_accum"
    "transpose"
    )

# Describe all tests that are not yet implemented
set(TESTS_NOT_IMPLEMENTED
    "accumulate"
    "accumulate_hypot"
    "accumulate_maxsumexp"
    "adam_step"
    "adamw_step"
    "add"
    "add_fiber"
    "add_fiber_inplace"
    "add_inplace"
    "add_slice"
    "codelet"
    "conv2d_bwd_input_inplace"
    "conv2d_bwd_weight_inplace"
    "conv2d_inplace"
    "copy"
    "embedding"
    "embedding_backward"
    "flash_sdpa_fwd_cudnn"
    "gelu_backward"
    "gelutanh"
    "gelutanh_backward"
    "handle"
    "hypot_inplace"
    "hypot"
    "hypot_scalar_inverse"
    "log_scalar"
    "logsumexp"
    "pow"
    "multiply"
    "multiply_fiber_inplace"
    "multiply_fiber"
    "multiply_slice"
    "relu_backward"
    "relu"
    "rope"
    "rope_backward"
    "scale"
    "scale_inplace"
    "scale_fiber"
    "scale_slice"
    "silu_backward"
    "silu"
    "silu_inplace"
    "softmax"
    "sqrt"
    "subtract_indexed_column"
    "subtract_indexed_outputs"
    "sum_fiber"
    "sumprod_fiber"
    "total_sum_accum"
)

# Add target for local coverage
if(BUILD_COVERAGE)
    setup_target_for_coverage_lcov(NAME coverage_starpu
        EXECUTABLE ctest -R tests_starpu_
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
endif()

#add_test_set(TARGET_NAME tests_starpu_config
#    EXEC_NAME test_config
#    SOURCES config.cc
#    LINK_LIBRARIES nntile
#    ARGS "1" "2" "3" "4" "5"
#    COV_ENABLE ${BUILD_COVERAGE}
#    COV_NAME coverage_starpu_config
#    COV_GLOBAL coverage_starpu coverage
#    )

foreach(test IN LISTS TESTS)
    set(labels)
    if(test IN_LIST TESTS_NOT_IMPLEMENTED)
        set(labels "NotImplemented")
    endif()
    add_test_set(TARGET_NAME tests_starpu_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile $<$<BOOL:${NNTILE_LINK_CUDA}>:CUDA::cudart>
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_starpu_${test}
        COV_GLOBAL coverage_starpu coverage
        LABELS ${labels}
        )
endforeach()

set(TESTS_CATCH2
)

foreach(test IN LISTS TESTS_CATCH2)
    set(labels)
    if(test IN_LIST TESTS_NOT_IMPLEMENTED)
        set(labels "NotImplemented")
    endif()
    add_test_set(TARGET_NAME tests_starpu_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES
            nntile
            $<$<BOOL:${NNTILE_USE_CUDA}>:CUDA::cudart>
            Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_starpu_${test}
        COV_GLOBAL coverage_starpu coverage
        LABELS ${labels}
    )
endforeach()
