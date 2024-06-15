# @copyright (c) 2022-2023 Skolkovo Institute of Science and Technology
#                          (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file CMakeLists.txt
# This is a top-level CMake script, that is tested to work only with a
# single-configuration build systems like Ninja or Unix Makefile.
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-09-26


###############################################################################
#                       THIS IS A TOP-LEVEL CMAKELISTS.txt                    #
#                                                                             #
#        It is intended to find all dependencies (required or optional)       #
#                    and set up corresponding variables                       #
###############################################################################

# Set minimum CMake version to avoid configuration problems
cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0148 NEW)

# Define the project
project(NNTile VERSION 1.0.0 LANGUAGES CXX)

# Show a path to additional auxiliary CMake scripts
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules")

# List of options and cache variables
# Build shared libraries instead of static
option(BUILD_SHARED_LIBS "Build shared libraries instead of static" ON)
option(USE_CUDA "Use CUDA toolkit" OFF)
option(BUILD_COVERAGE "Generate code coverage report" OFF)
option(BUILD_DOCS "Build Doxygen-based documentation" OFF)
option(BUILD_EXAMPLES "Build examples" OFF)
option(BUILD_PYTHON_WRAPPERS "Generate Python wrappers" ON)
option(BUILD_TESTS "Build tests" OFF)

option(SYSTEM_FXT "Use system FxT library" OFF)
option(SYSTEM_STARPU "Use system StarPU library" OFF)

# Allowed values of NNTILE_BLAS_VENDOR corresponds to BLA_VENDOR of FindBLAS package
# with the only exception: auxiliary case-sensitive value None in order to
# not use BLAS/LAPACK routines at all.
#
# https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors
set(NNTILE_BLAS_VENDOR "None" CACHE STRING "BLAS/LAPACK vendors to link against")

# This variable is a way to inform CMake where to symlink or copy Python
# extension library.
set(NNTILE_CORE_SYMLINK_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/wrappers/python/nntile"
    CACHE PATH "Where to symlink extension in Python")

# For easier code navigation and interaction in editors.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++-17 standard publicly
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Specify the library
add_library(nntile)

# Add certain compilation flags for warnings etc
#add_compile_options(-Wall -Wextra)# -Wpedantic-errors)

# Check if CUDA is available
set(NNTILE_USE_CUDA OFF)
if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        # Get cublas
        find_package(CUDAToolkit 10.1)
	target_link_libraries(nntile PUBLIC CUDA::cublas)
        set(NNTILE_USE_CUDA ON)
    endif()
endif()

if (NNTILE_BLAS_VENDOR STREQUAL "None")
    set(NNTILE_USE_CBLAS OFF)
else()
    set(NNTILE_USE_CBLAS ON)
endif()

# TODO(@daskol): There is a CMake package to find BLAS/LAPACK with CBLAS
# interface. Thus proper wording is required here.
if (NNTILE_USE_CBLAS)
    set(BLA_VENDOR "${NNTILE_BLAS_VENDOR}")
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    target_link_libraries(nntile PRIVATE BLAS::BLAS LAPACK::LAPACK)

    # TODO(@daskol): Configure integer types and static/shared linkage.
    # https://cmake.org/cmake/help/latest/module/FindBLAS.html#input-variables.
    set(CBLAS_INT_TYPE "int" CACHE STRING
        "CBLAS integer type: int, int32_t or int64_t")
    set_property(CACHE CBLAS_INT_TYPE PROPERTY STRINGS "int" "int32_t"
        "int64_t")
    set(CBLAS_H_NAME "cblas.h" CACHE STRING
        "Name of header file containing cblas routines")
endif()

if(BUILD_EXAMPLES)
    add_subdirectory("examples")
endif()

add_subdirectory("external")
add_subdirectory("include")
add_subdirectory("src")

if(BUILD_PYTHON_WRAPPERS)
    add_subdirectory("wrappers/python")
endif()

# Get MPI, disabled for StarPU master-slave option
#find_package(MPI REQUIRED)
#target_link_libraries(nntile PUBLIC MPI::MPI_CXX)

# TODO(@daskol): Can we replace PUBLIC with PRIVATE?
target_link_libraries(nntile PUBLIC starpu::starpu)
target_include_directories(nntile
    PUBLIC "${PROJECT_SOURCE_DIR}/include"
    PUBLIC "${PROJECT_BINARY_DIR}/include")
target_include_directories(nntile PRIVATE "${PROJECT_SOURCE_DIR}/external")

# Configure list of definitions
configure_file("${PROJECT_SOURCE_DIR}/include/nntile/defs.h.in"
    "${PROJECT_BINARY_DIR}/include/nntile/defs.h" @ONLY)

# TODO(@daskol): Replace configuration file generation with proper compilation
# definitions.

# Configure src/starpu/axpy.cc that relies on cblas
configure_file("${PROJECT_SOURCE_DIR}/src/starpu/axpy.cc.in"
    "${PROJECT_BINARY_DIR}/src/starpu/axpy.cc" @ONLY)

# Configure tests/starpu/axpy.cc that relies on cblas
configure_file("${PROJECT_SOURCE_DIR}/tests/starpu/axpy.cc.in"
    "${PROJECT_BINARY_DIR}/tests/starpu/axpy.cc" @ONLY)

# Configure src/starpu/gemm.cc that relies on cblas
configure_file("${PROJECT_SOURCE_DIR}/src/starpu/gemm.cc.in"
    "${PROJECT_BINARY_DIR}/src/starpu/gemm.cc" @ONLY)

# Configure tests/starpu/gemm.cc that relies on cblas
configure_file("${PROJECT_SOURCE_DIR}/tests/starpu/gemm.cc.in"
    "${PROJECT_BINARY_DIR}/tests/starpu/gemm.cc" @ONLY)

# Configure src/starpu/flash_maxsumexp.cc that relies on cblas
configure_file("${PROJECT_SOURCE_DIR}/src/starpu/flash_maxsumexp.cc.in"
    "${PROJECT_BINARY_DIR}/src/starpu/flash_maxsumexp.cc" @ONLY)

# Configure src/starpu/flash_softmax_gemm.cc that relies on cblas
configure_file("${PROJECT_SOURCE_DIR}/src/starpu/flash_softmax_gemm.cc.in"
    "${PROJECT_BINARY_DIR}/src/starpu/flash_softmax_gemm.cc" @ONLY)

# Configure src/starpu/flash_softmax_gemm_backward_sumprod_slice.cc that relies on cblas
configure_file("${PROJECT_SOURCE_DIR}/src/starpu/flash_softmax_gemm_backward_sumprod_slice.cc.in"
    "${PROJECT_BINARY_DIR}/src/starpu/flash_softmax_gemm_backward_sumprod_slice.cc" @ONLY)

# Configure src/starpu/flash_softmax_gemm_backward_dq_dk.cc that relies on cblas
configure_file("${PROJECT_SOURCE_DIR}/src/starpu/flash_softmax_gemm_backward_dq_dk.cc.in"
    "${PROJECT_BINARY_DIR}/src/starpu/flash_softmax_gemm_backward_dq_dk.cc" @ONLY)

# Configure src/starpu/nrm2.cc that relies on cblas
configure_file("${PROJECT_SOURCE_DIR}/src/starpu/nrm2.cc.in"
    "${PROJECT_BINARY_DIR}/src/starpu/nrm2.cc" @ONLY)

# Configure tests/starpu/nrm2.cc that relies on cblas
configure_file("${PROJECT_SOURCE_DIR}/tests/starpu/nrm2.cc.in"
    "${PROJECT_BINARY_DIR}/tests/starpu/nrm2.cc" @ONLY)

# Configure src/starpu/scal_inplace.cc that relies on cblas
configure_file("${PROJECT_SOURCE_DIR}/src/starpu/scal_inplace.cc.in"
    "${PROJECT_BINARY_DIR}/src/starpu/scal_inplace.cc" @ONLY)

# Configure tests/starpu/scal_inplace.cc that relies on cblas
configure_file("${PROJECT_SOURCE_DIR}/tests/starpu/scal_inplace.cc.in"
    "${PROJECT_BINARY_DIR}/tests/starpu/scal_inplace.cc" @ONLY)

# Check if code coverage report is needed
if(BUILD_COVERAGE)
    # Tell user what we are doing here
    message(STATUS "Code coverage report was requested, so option BUILD_TESTS"
        " is ON."
        " The report itself can be generated by \"make coverage\" command.")
    # Enable tests even if they were manually disabled
    set(BUILD_TESTS ON)
    # Use CodeCoverage.cmake from cmake_modules
    include(CodeCoverage)
    # Append coverage flags
    append_coverage_compiler_flags()
    # Set global coverage exclude
    set(COVERAGE_EXCLUDES "${PROJECT_SOURCE_DIR}/tests/*"
        "${PROJECT_BINARY_DIR}/tests/*")
    # Setup a target for an overall coverage
    setup_target_for_coverage_lcov(NAME coverage
        EXECUTABLE ctest
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
endif()

if(BUILD_TESTS)
    add_subdirectory("tests")
endif()
