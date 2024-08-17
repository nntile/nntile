# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/tensor/CMakeLists.txt
# Tests for nntile::tensor functions
#
# @version 1.1.0

# All unit tests without arguments to test executable
set(TESTS
    "traits"
    "distributions"
    )

set(TESTS_MPI
    "add"
    "add_fiber"
    "add_slice"
    "add_slice3"
    "addcdiv"
    "axpy"
    "clear"
    "copy"
    "copy_intersection"
    "dgelu"
    "dgelutanh"
    "drelu"
    "fill"
    "gather"
    "gelu"
    "gelu_backward"
    "gelutanh"
    "gelutanh_inplace"
    "gelutanh_backward"
    "gemm"
    "logsumexp"
    "maximum"
    "maxsumexp"
    "norm_slice"
    "normalize"
    "nrm2"
    "pow"
    "prod_inplace"
    "prod_fiber"
    "prod_fiber3"
    "prod_slice"
    "randn"
    "relu"
    "relu_backward"
    "scal_inplace"
    "scatter"
    "softmax"
    "softmax_inplace"
    "sqrt"
    "sqrt_inplace"
    "subtract_indexed_column"
    "sum_fiber"
    "sum_slice"
    "sumnorm"
    "sumprod_fiber"
    "sumprod_slice"
    "tensor"
    "total_sum_accum"
    "mask_scalar"
    "scal"
    "hypot"
    "transpose"
    )

# Describe all tests that are not yet implemented
set(TESTS_NOT_IMPLEMENTED
    "add"
    "gelu_backward"
    "gelutanh"
    "gelutanh_backward"
    "logsumexp"
    "pow"
    "relu_backward"
    "subtract_indexed_column"
    "total_sum_accum"
    "sqrt"
    "scal"
    "softmax"
    "hypot"
    "add_slice3"
    "prod_fiber3"
    "tensor"
    )

# Add target for local coverage
if(BUILD_COVERAGE)
    setup_target_for_coverage_lcov(NAME coverage_tensor
        EXECUTABLE ctest -R tests_tensor_
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
endif()

foreach(test IN LISTS TESTS)
    add_test_set(TARGET_NAME tests_tensor_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_tensor_${test}
        COV_GLOBAL coverage_tensor coverage
        )
endforeach()

foreach(test IN LISTS TESTS_MPI)
    set(labels)
    if(test IN_LIST TESTS_NOT_IMPLEMENTED)
        set(labels "NotImplemented")
    endif()
    # Add non-mpirun test
    add_test_set(TARGET_NAME tests_tensor_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_tensor_${test}
        COV_GLOBAL coverage_tensor coverage
        LABELS ${labels}
        )
    # Add mpirun test (the same source, but different output executable)
    add_test_set(TARGET_NAME tests_tensor_${test}_mpi
        EXEC_NAME test_${test}_mpi
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile
        MPI_NUMPROC 4
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_tensor_${test}
        COV_GLOBAL coverage_tensor coverage
        LABELS ${labels} MPI
        )
endforeach()
