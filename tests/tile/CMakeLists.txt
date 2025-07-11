# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/tile/CMakeLists.txt
# Tests for nntile::tile functions
#
# @version 1.1.0

# All unit tests without arguments to test executable
set(TESTS
    "add"
    "add_fiber_inplace"
    "add_fiber"
    "add_slice_inplace"
    "add_slice"
    "addcdiv"
    "clear"
    "copy"
    "copy_intersection"
    "dgelu"
    "dgelutanh"
    "drelu"
    "fill"
    "gelu"
    "gelu_backward"
    "gelutanh"
    "gelutanh_inplace"
    "gelutanh_backward"
    "gemm"
    "logsumexp"
    "maximum"
    "maxsumexp"
    "norm_fiber"
    "norm_fiber_inplace"
    "norm_slice"
    "pow"
    "prod_inplace"
    "prod_fiber"
    "prod_fiber3"
    "prod_slice"
    "randn"
    "relu"
    "relu_backward"
    "scal_inplace"
    "softmax"
    "softmax_inplace"
    "sqrt"
    "sqrt_inplace"
    "subtract_indexed_column"
    "sum_fiber"
    "sum_slice"
    "sumprod_fiber"
    "sumprod_slice"
    "tile"
    "total_sum_accum"
    "traits"
    "scal"
    "hypot"
)

# Describe all tests that are not yet implemented
set(TESTS_NOT_IMPLEMENTED
    "add"
    "add_fiber_inplace"
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
    "add_slice"
    "prod_fiber3"
)

# Add target for local coverage
if(BUILD_COVERAGE)
    setup_target_for_coverage_lcov(NAME coverage_tile
        EXECUTABLE ctest -R tests_tile_
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
endif()

foreach(test IN LISTS TESTS)
    set(labels)
    if(test IN_LIST TESTS_NOT_IMPLEMENTED)
        set(labels "NotImplemented")
    endif()
    add_test_set(TARGET_NAME tests_tile_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_tile_${test}
        COV_GLOBAL coverage_tile coverage
        LABELS ${labels}
    )
endforeach()
