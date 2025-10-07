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
    "add_fiber_inplace"
    "add_inplace"
    "add_slice"
    "add_slice_inplace"
    "adam_step"
    "adamw_step"
    "clear"
    "conv2d_bwd_input_inplace"
    "conv2d_bwd_weight_inplace"
    "conv2d_inplace"
    "copy"
    "copy_intersection"
    "embedding"
    "embedding_backward"
    "fill"
    "gather"
    "gelu"
    "gelu_backward"
    "gelutanh"
    "gelutanh_backward"
    "gelutanh_inplace"
    "gemm"
    "hypot"
    "hypot_scalar_inverse"
    "log_scalar"
    "logsumexp"
    "mask_scalar"
    "maximum"
    "maxsumexp"
    "norm_fiber"
    "norm_fiber_inplace"
    "norm_slice"
    "pow"
    "prod"
    "prod_fiber"
    "prod_fiber3"
    "prod_inplace"
    "prod_slice"
    "randn"
    "relu"
    "relu_backward"
    "relu_forward"
    "rope"
    "rope_backward"
    "scal"
    "scal_inplace"
    "scatter"
    "silu_backward"
    "silu_forward"
    "softmax"
    "softmax_inplace"
    "sqrt"
    "sqrt_inplace"
    "subtract_indexed_column"
    "subtract_indexed_outputs"
    "sum_fiber"
    "sum_slice"
    "sumprod_fiber"
    "sumprod_slice"
    "tensor"
    "total_sum_accum"
    "transpose"
    )

# Describe all tests that are not yet implemented
set(TESTS_NOT_IMPLEMENTED
    "add_slice"
    "adam_step"
    "adamw_step"
    "conv2d_bwd_input_inplace"
    "conv2d_bwd_weight_inplace"
    "conv2d_inplace"
    "embedding"
    "embedding_backward"
    "gelu_backward"
    "gelutanh"
    "gelutanh_backward"
    "hypot"
    "hypot_scalar_inverse"
    "log_scalar"
    "logsumexp"
    "norm_fiber"
    "norm_fiber_inplace"
    "pow"
    "prod"
    "prod_fiber3"
    "relu_backward"
    "relu_forward"
    "rope"
    "rope_backward"
    "scal"
    "silu_backward"
    "silu_forward"
    "softmax"
    "sqrt"
    "subtract_indexed_column"
    "subtract_indexed_outputs"
    "tensor"
    "total_sum_accum"
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
    # # Add mpirun test (the same source, but different output executable)
    # add_test_set(TARGET_NAME tests_tensor_${test}_mpi
    #     EXEC_NAME test_${test}_mpi
    #     SOURCES ${test}.cc
    #     LINK_LIBRARIES nntile
    #     MPI_NUMPROC 4
    #     COV_ENABLE ${BUILD_COVERAGE}
    #     COV_NAME coverage_tensor_${test}
    #     COV_GLOBAL coverage_tensor coverage
    #     LABELS ${labels} MPI
    #     )
endforeach()
