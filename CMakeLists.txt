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
message(STATUS "Using CMake ${CMAKE_VERSION}")

# Define the project
project(NNTile VERSION 1.0.0 LANGUAGES CXX)

# Show a path to additional auxiliary CMake scripts
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules")

# List of options and cache variables
# Build shared libraries instead of static
option(BUILD_SHARED_LIBS "Build shared libraries instead of static" ON)
option(USE_CUDA "Use CUDA toolkit" ON)
option(USE_CBLAS "Use CPU CBLAS" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_DOCS "Build Doxygen-based documentation" OFF)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_COVERAGE "Generate code coverage report" OFF)
option(BUILD_PYTHON_WRAPPERS "Generate Python wrappers" ON)

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

# Get CBLAS
set(NNTILE_USE_CBLAS OFF)
if(USE_CBLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        target_link_libraries(nntile PRIVATE BLAS::BLAS)
        set(NNTILE_USE_CBLAS ON)
        set(CBLAS_INT_TYPE "int" CACHE STRING
            "CBLAS integer type: int, int32_t or int64_t")
        set_property(CACHE CBLAS_INT_TYPE PROPERTY STRINGS "int" "int32_t"
            "int64_t")
        set(CBLAS_H_NAME "cblas.h" CACHE STRING
            "Name of header file containing cblas routines")
    endif()
endif()

# Get MPI, disabled for StarPU master-slave option
#find_package(MPI REQUIRED)
#target_link_libraries(nntile PUBLIC MPI::MPI_CXX)

# Get the pkg-config
find_package(PkgConfig REQUIRED)

# Get the StarPU
pkg_check_modules(StarPU REQUIRED starpu-1.4)
target_link_libraries(nntile PUBLIC ${StarPU_LDFLAGS})
target_include_directories(nntile PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_BINARY_DIR}/include"
    ${StarPU_INCLUDE_DIRS}
    )
target_include_directories(nntile PRIVATE
    "${PROJECT_SOURCE_DIR}/external"
    )

# Configure list of definitions
configure_file("${PROJECT_SOURCE_DIR}/include/nntile/defs.h.in"
    "${PROJECT_BINARY_DIR}/include/nntile/defs.h" @ONLY)

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

# Add directory with the sources
add_subdirectory("src")

# Add sirectory with headers
add_subdirectory("include")

# Add external sources/headers
add_subdirectory("external")

# Add subdirectory with tests
if(BUILD_TESTS)
    include(CTest)
    add_subdirectory("tests")
endif()

# Add subdirectory with examples
if(BUILD_EXAMPLES)
    add_subdirectory("examples")
endif()

# Check if Python wrappers are requested
if(BUILD_PYTHON_WRAPPERS)
    add_subdirectory("wrappers/python")
endif()

