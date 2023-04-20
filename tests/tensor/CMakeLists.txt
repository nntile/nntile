# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                          (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/tensor/CMakeLists.txt
# Tests for nntile::tensor functions
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @author Aleksandr Katrutsa
# @author Konstantin Sozykin
# @date 2023-04-20

# All unit tests without arguments to test executable
set(TESTS
    "traits"
    "distributions"
    )

set(TESTS_MPI
    "tensor"
    "copy"
    "copy_intersection"
    "scatter"
    "gather"
    "clear"
    "axpy"
    "bias"
    "randn"
    "gemm"
    "gelu"
    "gelutanh"
    "dgelu"
    "dgelutanh"
    "drelu"
    "relu"
    "sumnorm"
    "set"
    "sum"
    "sum_outer"
    "norm"
    "nrm2"
    "normalize"
    "prod"
    "maxsumexp"
    "softmax"
    "sqrt"
    "maximum"
    "addcdiv"
    "scal"
    "scalprod"
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
    add_test_set(TARGET_NAME tests_tensor_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile
        MPI_NUMPROC 4
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_tensor_${test}
        COV_GLOBAL coverage_tensor coverage
        )
endforeach()

