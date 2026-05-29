# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file CMakeLists.txt
# Top-level CMake (Plan B unified nntile package).
#
# @version 1.1.0

cmake_minimum_required(VERSION 3.24)
message(STATUS "Using CMake ${CMAKE_VERSION}")

project(NNTile VERSION 1.1.0 LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules")

set(NNTILE_COMPILE_CHECK_TESTS_SUBSYSTEM "" CACHE STRING
    "Compile-check test sources for one subsystem (no libnntile link/run)")

option(BUILD_SHARED_LIBS "Build shared libraries instead of static" ON)
option(BUILD_NNTILE "Build unified nntile library" ON)
option(USE_CUDA "Use CUDA toolkit (cuBLAS and cuDNN are required)" ON)
option(USE_CUDA_FP16 "Enable FP16 CUDA support" ON)
option(USE_CUDA_TF32 "Enable TF32 CUDA support" ON)
option(USE_CUDA_BF16 "Enable BF16 CUDA support" ON)
option(USE_CUDA_FP8 "Enable FP8 CUDA support" ON)
option(USE_CBLAS "Use CPU CBLAS" ON)
option(BUILD_TESTS "Build and run tests (requires linked libnntile)" ON)
option(BUILD_TESTS_PYTORCH "Build NNGraph tests comparing against PyTorch" ON)
option(BUILD_DOCS "Build Doxygen-based documentation" OFF)
option(BUILD_EXAMPLES "Build C++ examples under nntile/examples" ON)
option(BUILD_COVERAGE "Generate code coverage report" OFF)
option(BUILD_PYTHON_WRAPPERS "Build python/ package (not wrappers/python)" ON)
option(BUILD_LEGACY_PYTHON_WRAPPERS "Build legacy wrappers/python (reference)" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

macro(nntile_apply_common_includes target)
    target_include_directories(${target} PUBLIC
        "${PROJECT_SOURCE_DIR}/nntile/include"
        "${PROJECT_BINARY_DIR}/include"
        ${StarPU_INCLUDE_DIRS}
    )
    target_include_directories(${target} PRIVATE
        "${PROJECT_SOURCE_DIR}/external"
    )
endmacro()

macro(nntile_propagation_scope target out_var)
    get_target_property(_imported ${target} IMPORTED)
    if(_imported)
        set(${out_var} INTERFACE)
    else()
        set(${out_var} PUBLIC)
    endif()
endmacro()

macro(nntile_apply_cuda target)
    nntile_propagation_scope(${target} _scope)
    if(NNTILE_LINK_CUDA)
        target_link_libraries(${target} ${_scope} CUDA::cublas CUDNN::cudnn_all)
        target_include_directories(${target} ${_scope}
            ${cudnn_frontend_SOURCE_DIR}/include)
    endif()
endmacro()

macro(nntile_apply_cblas target)
    nntile_propagation_scope(${target} _scope)
    if(NNTILE_LINK_CBLAS)
        target_link_libraries(${target} ${_scope} BLAS::BLAS)
    endif()
endmacro()

find_package(PkgConfig REQUIRED)
add_subdirectory("external")

pkg_check_modules(StarPU REQUIRED starpu-1.4)

include(CheckSymbolExists)
set(CMAKE_REQUIRED_INCLUDES ${StarPU_INCLUDE_DIRS})
check_symbol_exists(STARPU_SIMGRID "starpu_config.h" HAVE_STARPU_SIMGRID)
if(HAVE_STARPU_SIMGRID)
    message(STATUS "StarPU is compiled in SIMGRID mode")
endif()

set(NNTILE_USE_CUDA OFF)
set(NNTILE_LINK_CUDA OFF)
if(HAVE_STARPU_SIMGRID)
    if(USE_CUDA)
        set(NNTILE_USE_CUDA ON)
    endif()
elseif(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit 11.0 REQUIRED)
        add_subdirectory("external/cudnn_frontend")
        include(${cudnn_frontend_SOURCE_DIR}/cmake/cuDNN.cmake)
        set(NNTILE_USE_CUDA ON)
        set(NNTILE_LINK_CUDA ON)
        if(USE_CUDA_FP16)
            set(NNTILE_USE_CUDA_FP16 ON)
        else()
            set(NNTILE_USE_CUDA_FP16 OFF)
        endif()
        if(USE_CUDA_TF32)
            set(NNTILE_USE_CUDA_TF32 ON)
        else()
            set(NNTILE_USE_CUDA_TF32 OFF)
        endif()
        if(USE_CUDA_BF16)
            set(NNTILE_USE_CUDA_BF16 ON)
        else()
            set(NNTILE_USE_CUDA_BF16 OFF)
        endif()
        if(USE_CUDA_FP8 AND CUDAToolkit_VERSION GREATER_EQUAL 11.8)
            set(NNTILE_USE_CUDA_FP8 ON)
        else()
            set(NNTILE_USE_CUDA_FP8 OFF)
        endif()
    endif()
endif()

set(NNTILE_USE_CBLAS OFF)
set(NNTILE_LINK_CBLAS OFF)
if(HAVE_STARPU_SIMGRID)
    if(USE_CBLAS)
        set(NNTILE_USE_CBLAS ON)
    endif()
elseif(USE_CBLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        set(NNTILE_USE_CBLAS ON)
        set(NNTILE_LINK_CBLAS ON)
    endif()
endif()

if(BUILD_TESTS AND BUILD_NNTILE AND NOT HAVE_STARPU_SIMGRID
        AND NOT NNTILE_COMPILE_CHECK_SUBSYSTEM
        AND NOT NNTILE_COMPILE_CHECK_TESTS_SUBSYSTEM)
    include(CTest)
endif()

if(BUILD_COVERAGE)
    set(BUILD_TESTS ON)
    include(CodeCoverage)
    append_coverage_compiler_flags()
    set(COVERAGE_EXCLUDES "${PROJECT_SOURCE_DIR}/nntile/tests/*"
        "${PROJECT_BINARY_DIR}/nntile/tests/*")
    setup_target_for_coverage_lcov(NAME coverage
        EXECUTABLE ctest
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
endif()

if(BUILD_NNTILE OR NNTILE_COMPILE_CHECK_TESTS_SUBSYSTEM)
    add_subdirectory(nntile)
endif()

set(NNTILE_HAVE_TORCH OFF)
if(BUILD_TESTS_PYTORCH AND BUILD_TESTS AND BUILD_NNTILE AND NOT HAVE_STARPU_SIMGRID)
    find_package(Torch QUIET)
    if(Torch_FOUND)
        set(NNTILE_HAVE_TORCH ON)
        message(STATUS "LibTorch found: ${TORCH_INSTALL_PREFIX}")
    else()
        message(WARNING "BUILD_TESTS_PYTORCH=ON but LibTorch not found.")
    endif()
endif()

if(BUILD_PYTHON_WRAPPERS AND BUILD_NNTILE)
    add_subdirectory(python)
endif()

if(BUILD_LEGACY_PYTHON_WRAPPERS)
    add_subdirectory(wrappers/python)
endif()
