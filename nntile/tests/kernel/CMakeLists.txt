# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems with GPU acceleration.
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
    "mask_scalar"
    "norm_fiber_inplace"
    "randn"
    "rope"
    "rope_backward"
    "scale_inplace"
    "softmax_inplace"
    "sqrt_inplace"
    "subcopy"
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
    "scale_inplace"
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

nntile_filter_disabled_kernel_tests(TESTS TESTS_NOT_IMPLEMENTED TESTS_CATCH2)

foreach(test IN LISTS TESTS)
    set(labels core)
    if(test IN_LIST TESTS_NOT_IMPLEMENTED)
        set(labels "NotImplemented core")
    endif()
    add_test_set(TARGET_NAME tests_kernel_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile::nntile $<$<BOOL:${NNTILE_USE_CUDA}>:CUDA::cudart>
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_kernel_${test}
        COV_GLOBAL coverage_kernel coverage
        LABELS ${labels}
    )
endforeach()

set(TESTS_CATCH2
    "add"
    "add_fiber"
    "add_fiber_inplace"
    "scale_fiber"
    "add_inplace"
    "add_slice_inplace"
    "adam_step"
    "adamw_step"
    "sgd_step"
    "add_slice"
    "embedding"
    "embedding_backward"
    "fill"
    "gelu"
    "gelu_inplace"
    "gelu_backward"
    "gelutanh"
    "gelutanh_inplace"
    "gelutanh_backward"
    "hypot_inplace"
    "hypot"
    "hypot_scalar_inverse"
    "logsumexp"
    "maxsumexp"
    "norm"
    "norm_fiber"
    "norm_slice_inplace"
    "norm_slice"
    "pow"
    "multiply_fiber_inplace"
    "multiply_fiber"
    "multiply_inplace"
    "multiply_slice"
    "multiply"
    "relu_inplace"
    "relu_backward"
    "relu"
    "scale"
    "scale_slice"
    "silu_backward"
    "silu"
    "silu_inplace"
    "softmax"
    "sqrt"
    "sum"
    "sum_fiber"
)

nntile_filter_disabled_kernel_tests(TESTS_CATCH2)

foreach(test IN LISTS TESTS_CATCH2)
    set(labels core)
    if(test IN_LIST TESTS_NOT_IMPLEMENTED)
        set(labels "NotImplemented core")
    endif()
    if(test STREQUAL "add")
        set(labels "${labels} smoke")
    endif()
    add_test_set(TARGET_NAME tests_kernel_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES
            nntile::nntile
            $<$<BOOL:${NNTILE_USE_CUDA}>:CUDA::cudart>
            Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_kernel_${test}
        COV_GLOBAL coverage_kernel coverage
        LABELS ${labels}
    )
endforeach()

# Chunked-versus-contiguous A/B. Backend-agnostic: the chunked entry points
# compile in both products, so this runs under NNHaul and under NNHaul with
# CUDA. Not a Catch2 test, so it needs no fixture framework.
if(NNTILE_USE_CUDA)
    add_test_set(TARGET_NAME tests_kernel_chunked_ab
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/chunked_ab.cc
        LINK_LIBRARIES
            nntile::nntile
            CUDA::cudart
        LABELS "SUBSYSTEM_KERNEL;NNHAUL"
    )
endif()

# The same A/B for the chunked entry points that the small wrapper tests
# never reach with more than one chunk: unary, in-place, reduction,
# per-fiber max and softmax, all over 10 class pieces plus a tail.
if(NNTILE_USE_CUDA)
    add_test_set(TARGET_NAME tests_kernel_chunked_ops
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/chunked_ops.cc
        LINK_LIBRARIES
            nntile::nntile
            CUDA::cudart
        LABELS "SUBSYSTEM_KERNEL;NNHAUL"
    )
endif()

# The probe that the suite's chunk-class environment actually reached the
# kernel-side class. A 1 MiB suite leg is meaningless without it.
if(NNTILE_USE_CUDA)
    add_test_set(TARGET_NAME tests_kernel_chunk_class_probe
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/chunk_class_probe.cc
        LINK_LIBRARIES
            nntile::nntile
            CUDA::cudart
        LABELS "SUBSYSTEM_KERNEL;NNHAUL"
    )
endif()

# Manual benchmark: one-launch chunked GeLU vs one contiguous launch per
# chunk. Not registered with ctest; run by hand on a GPU.
if(NNTILE_USE_CUDA)
    add_executable(gelu_chunk_benchmark
        ${CMAKE_CURRENT_SOURCE_DIR}/gelu_chunk_benchmark.cc)
    target_link_libraries(gelu_chunk_benchmark PRIVATE
        nntile::nntile
        CUDA::cudart)
endif()
