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
    "adam_step"
    "adamw_step"
    "add"
    "add_fiber"
    "add_slice"
    "add_slice3"
    "addcdiv"
    "dgelu"
    "dgelutanh"
    "drelu"
    "fill"
    "gelu"
    "gelu_backward"
    "gelutanh"
    "gelutanh_inplace"
    "gelutanh_backward"
    "hypot"
    "logsumexp"
    "maximum"
    "norm_slice"
    "normalize"
    "pow"
    "prod_inplace"
    "prod_fiber"
    "prod_fiber3"
    "prod_slice"
    "randn"
    "relu"
    "relu_backward"
    "rope"
    "rope_backward"
    "softmax"
    "softmax_inplace"
    "sqrt"
    "sqrt_inplace"
    "subcopy"
    "subtract_indexed_column"
    "sum_fiber"
    "norm_fiber"
    "sum_slice"
    "sumnorm"
    "sumprod_fiber"
    "sumprod_slice"
    "total_sum_accum"
    "mask_scalar"
    "scal"
    "transpose"
    )

# Describe all tests that are not yet implemented
set(TESTS_NOT_IMPLEMENTED
    "adam_step"
    "adamw_step"
    "add_fiber"
    "gelu_backward"
    "gelutanh"
    "gelutanh_backward"
    "logsumexp"
    "pow"
    "prod_fiber"
    "prod_slice"
    "relu_backward"
    "subtract_indexed_column"
    "sum_fiber"
    "sumprod_fiber"
    "total_sum_accum"
    "sqrt"
    "scal"
    "softmax"
    "hypot"
    "add_slice3"
    "prod_fiber3"
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

#add_executable("tests_kernel_maxsumexp" maxsumexp.cc)
#target_link_libraries("tests_kernel_maxsumexp" PRIVATE
#    GTest::gtest_main
#    nntile
#    $<$<BOOL:${NNTILE_USE_CUDA}>:CUDA::cudart>)
#gtest_discover_tests("tests_kernel_maxsumexp")
