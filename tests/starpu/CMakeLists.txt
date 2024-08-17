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
    "add"
    "add_fiber"
    "add_slice"
    "add_slice3"
    "addcdiv"
    "clear"
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
    "maxsumexp"
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
    "softmax"
    "softmax_inplace"
    "sqrt"
    "sqrt_inplace"
    "subcopy"
    "subtract_indexed_column"
    "sum_fiber"
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
    "add"
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

# Add include path for generated axpy.cc
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/axpy.cc" PROPERTIES
    INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")

# Explicitly add axpy test since it was generated from axpy.cc.in
add_test_set(TARGET_NAME tests_starpu_axpy
    EXEC_NAME test_axpy
    SOURCES "${CMAKE_CURRENT_BINARY_DIR}/axpy.cc"
    LINK_LIBRARIES nntile $<$<BOOL:${NNTILE_LINK_CBLAS}>:BLAS::BLAS>
    "$<$<BOOL:${NNTILE_LINK_CUDA}>:CUDA::cublas;CUDA::cudart>"
    COV_ENABLE ${BUILD_COVERAGE}
    COV_NAME coverage_starpu_axpy
    COV_GLOBAL coverage_starpu
    )

# Add include path for generated nrm2.cc
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/nrm2.cc" PROPERTIES
    INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")

# Explicitly add axpy test since it was generated from nrm2.cc.in
add_test_set(TARGET_NAME tests_starpu_nrm2
    EXEC_NAME test_nrm2
    SOURCES "${CMAKE_CURRENT_BINARY_DIR}/nrm2.cc"
    LINK_LIBRARIES nntile $<$<BOOL:${NNTILE_LINK_CBLAS}>:BLAS::BLAS>
    "$<$<BOOL:${NNTILE_LINK_CUDA}>:CUDA::cublas;CUDA::cudart>"
    COV_ENABLE ${BUILD_COVERAGE}
    COV_NAME coverage_starpu_nrm2
    COV_GLOBAL coverage_starpu
    )

# Add include path for generated gemm.cc
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/gemm.cc" PROPERTIES
    INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")

# Explicitly add gemm test since it was generated from gemm.cc.in
add_test_set(TARGET_NAME tests_starpu_gemm
    EXEC_NAME test_gemm
    SOURCES "${CMAKE_CURRENT_BINARY_DIR}/gemm.cc"
    LINK_LIBRARIES nntile $<$<BOOL:${NNTILE_LINK_CBLAS}>:BLAS::BLAS>
    "$<$<BOOL:${NNTILE_LINK_CUDA}>:CUDA::cublas;CUDA::cudart>"
    COV_ENABLE ${BUILD_COVERAGE}
    COV_NAME coverage_starpu_gemm
    COV_GLOBAL coverage_starpu
    )

# Add include path for generated scal_inplace.cc
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/scal_inplace.cc"
    PROPERTIES INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")

# Explicitly add axpy test since it was generated from scal_inplace.cc.in
add_test_set(TARGET_NAME tests_starpu_scal_inplace
    EXEC_NAME test_scal_inplace
    SOURCES "${CMAKE_CURRENT_BINARY_DIR}/scal_inplace.cc"
    LINK_LIBRARIES nntile $<$<BOOL:${NNTILE_LINK_CBLAS}>:BLAS::BLAS>
    "$<$<BOOL:${NNTILE_LINK_CUDA}>:CUDA::cublas;CUDA::cudart>"
    COV_ENABLE ${BUILD_COVERAGE}
    COV_NAME coverage_starpu_scal_inplace
    COV_GLOBAL coverage_starpu
    )
