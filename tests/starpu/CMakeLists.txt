# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                          (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/starpu/CMakeLists.txt
# Tests for nntile::starpu functions
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @author Aleksandr Katrutsa
# @author Konstantin Sozykin
# @date 2023-05-02

# All unit tests without arguments to test executable
set(TESTS
    "add_slice"
    "add_fiber"
    "clear"
    "subcopy"
    "gelu"
    "gelutanh"
    "dgelu"
    "dgelutanh"
    "drelu"
    "hypot"
    "normalize"
    "randn"
    "relu"
    "sumnorm"
    "prod"
    "prod_slice"
    "maxsumexp"
    "softmax"
    "sqrt"
    "maximum"
    "fill"
    "sum_slice"
    "sum_fiber"
    "norm"
    "addcdiv"
    "sumprod_slice"
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
    add_test_set(TARGET_NAME tests_starpu_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile $<$<BOOL:${NNTILE_USE_CUDA}>:CUDA::cudart>
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_starpu_${test}
        COV_GLOBAL coverage_starpu coverage
        )
endforeach()

# Add include path for generated axpy.cc
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/axpy.cc" PROPERTIES
    INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")

# Explicitly add axpy test since it was generated from axpy.cc.in
add_test_set(TARGET_NAME tests_starpu_axpy
    EXEC_NAME test_axpy
    SOURCES "${CMAKE_CURRENT_BINARY_DIR}/axpy.cc"
    LINK_LIBRARIES nntile $<$<BOOL:${NNTILE_USE_CBLAS}>:BLAS::BLAS>
    "$<$<BOOL:${NNTILE_USE_CUDA}>:CUDA::cublas;CUDA::cudart>"
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
    LINK_LIBRARIES nntile $<$<BOOL:${NNTILE_USE_CBLAS}>:BLAS::BLAS>
    "$<$<BOOL:${NNTILE_USE_CUDA}>:CUDA::cublas;CUDA::cudart>"
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
    LINK_LIBRARIES nntile $<$<BOOL:${NNTILE_USE_CBLAS}>:BLAS::BLAS>
    "$<$<BOOL:${NNTILE_USE_CUDA}>:CUDA::cublas;CUDA::cudart>"
    COV_ENABLE ${BUILD_COVERAGE}
    COV_NAME coverage_starpu_gemm
    COV_GLOBAL coverage_starpu
    )

# Add include path for generated scal.cc
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/scal.cc" PROPERTIES
    INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")

# Explicitly add axpy test since it was generated from scal.cc.in
add_test_set(TARGET_NAME tests_starpu_scal
    EXEC_NAME test_scal
    SOURCES "${CMAKE_CURRENT_BINARY_DIR}/scal.cc"
    LINK_LIBRARIES nntile $<$<BOOL:${NNTILE_USE_CBLAS}>:BLAS::BLAS>
    "$<$<BOOL:${NNTILE_USE_CUDA}>:CUDA::cublas;CUDA::cudart>"
    COV_ENABLE ${BUILD_COVERAGE}
    COV_NAME coverage_starpu_scal
    COV_GLOBAL coverage_starpu
    )

