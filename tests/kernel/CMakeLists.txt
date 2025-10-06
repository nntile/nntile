# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/kernel/CMakeLists.txt
# Tests for nntile::kernel functions
#
# @version 1.1.0

# All unit tests without arguments to test executable
set(TESTS
    "conv2d_bwd_input_inplace"
    "conv2d_bwd_weight_inplace"
    "conv2d_inplace"
    "fill"
    "gelu"
    "gelutanh_inplace"
    "hypot_scalar_inverse"
    "mask_scalar"
    "maximum"
    "norm_fiber_inplace"
    "randn"
    "rope"
    "rope_backward"
    "scal_inplace"
    "softmax_inplace"
    "sqrt_inplace"
    "subcopy"
    "subtract_indexed_column"
    "subtract_indexed_outputs"
    "sum_slice"
    "sumprod_fiber"
    "sumprod_slice"
    "total_sum_accum"
    "transpose"
)

# Describe all tests that are not yet implemented
set(TESTS_NOT_IMPLEMENTED
    "conv2d_bwd_input_inplace"
    "conv2d_bwd_weight_inplace"
    "conv2d_inplace"
    "hypot_scalar_inverse"
    "scal_inplace"
    "subtract_indexed_column"
    "subtract_indexed_outputs"
    "sumprod_fiber"
    "total_sum_accum"
)

# Setup a target for coverage of kernels
if(BUILD_COVERAGE)
    setup_target_for_coverage_lcov(NAME coverage_kernel
        EXECUTABLE ctest -R tests_kernel_
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
endif()

foreach(test IN LISTS TESTS)
    set(labels)
    if(test IN_LIST TESTS_NOT_IMPLEMENTED)
        set(labels "NotImplemented")
    endif()
    add_test_set(TARGET_NAME tests_kernel_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile $<$<BOOL:${NNTILE_USE_CUDA}>:CUDA::cudart>
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_kernel_${test}
        COV_GLOBAL coverage_kernel coverage
        LABELS ${labels}
    )
endforeach()

set(TESTS_CATCH2
    "add"
    "addcdiv"
    "accumulate_maxsumexp"
    "add_fiber"
    "add_fiber_inplace"
    "add_inplace"
    "add_slice_inplace"
    "adam_step"
    "adamw_step"
    "add_scalar"
    "add_slice"
    "embedding"
    "embedding_backward"
    "gelu_backward"
    "gelutanh"
    "gelutanh_backward"
    "hypot"
    "logsumexp"
    "norm_fiber"
    "norm_slice"
    "pow"
    "prod_fiber"
    "prod_fiber3"
    "prod_inplace"
    "prod_slice"
    "prod"
    "relu"
    "relu_backward"
    "relu_forward"
    "scal"
    "silu_backward"
    "silu_forward"
    "softmax"
    "sqrt"
    "sum_fiber"
)

foreach(test IN LISTS TESTS_CATCH2)
    set(labels)
    if(test IN_LIST TESTS_NOT_IMPLEMENTED)
        set(labels "NotImplemented")
    endif()
    add_test_set(TARGET_NAME tests_kernel_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES
            nntile
            $<$<BOOL:${NNTILE_USE_CUDA}>:CUDA::cudart>
            Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_kernel_${test}
        COV_GLOBAL coverage_kernel coverage
        LABELS ${labels}
    )
endforeach()

#add_executable("tests_kernel_maxsumexp" maxsumexp.cc)
#target_link_libraries("tests_kernel_maxsumexp" PRIVATE
#    GTest::gtest_main
#    nntile
#    $<$<BOOL:${NNTILE_USE_CUDA}>:CUDA::cudart>)
#gtest_discover_tests("tests_kernel_maxsumexp")
