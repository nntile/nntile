# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file CMakeLists.txt
# Top-level CMake (unified nntile package).
#
# @version 1.1.0

cmake_minimum_required(VERSION 3.24)
message(STATUS "Using CMake ${CMAKE_VERSION}")

project(NNTile VERSION 1.1.0 LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules")

option(BUILD_SHARED_LIBS "Build shared libraries instead of static" ON)
option(BUILD_NNTILE "Build the core C++ libnntile library" ON)
option(USE_CUDA "Use CUDA toolkit (cuBLAS and cuDNN are required)" ON)
option(USE_CUDA_FP16 "Enable FP16 CUDA support" ON)
option(USE_CUDA_TF32 "Enable TF32 CUDA support" ON)
option(USE_CUDA_BF16 "Enable BF16 CUDA support" ON)
option(USE_CUDA_FP8 "Enable FP8 CUDA support" ON)
option(USE_FLASH_SDPA "Build cuDNN flash SDPA (disabled on graph_api until C-order migration)" OFF)
option(USE_CBLAS "Use CPU CBLAS" ON)
option(BUILD_DOCS "Build Doxygen-based documentation" OFF)
# Default ON: the user-facing product is the torch_nntile pip wheel (needs
# LibTorch on CMAKE_PREFIX_PATH). Layered CI and Torch-less configures must
# pass -DBUILD_TORCH_NNTILE=OFF -DBUILD_TORCH_NNTILE_WHEEL=OFF.
option(BUILD_TORCH_NNTILE
    "Build LibTorch bridge libtorch_nntile (requires LibTorch)" ON)
option(BUILD_TORCH_NNTILE_EXAMPLES "Build libtorch_nntile C++ examples" OFF)
option(BUILD_TORCH_NNTILE_WHEEL
    "Build installable torch_nntile pip wheel (user-facing product)" ON)
option(BUILD_COVERAGE "Generate code coverage report" OFF)

# Standard CMake testing switch (default ON). Tests are added only for
# enabled components: BUILD_NNTILE, BUILD_TORCH_NNTILE, and/or
# BUILD_TORCH_NNTILE_WHEEL. Pass -DBUILD_TESTING=OFF to skip all tests.
include(CTest)
if(BUILD_COVERAGE)
    set(BUILD_TESTING ON CACHE BOOL "Build the testing tree" FORCE)
endif()

# In-tree wheel needs libtorch_nntile. Against an install prefix, keep
# BUILD_TORCH_NNTILE=OFF and pass -DNNTILE_PREFIX=... (and optionally
# -DTORCH_NNTILE_PREFIX=...).
if(BUILD_TORCH_NNTILE_WHEEL AND NOT BUILD_TORCH_NNTILE)
    set(_wheel_has_prefix FALSE)
    if(DEFINED NNTILE_PREFIX AND NOT NNTILE_PREFIX STREQUAL "")
        set(_wheel_has_prefix TRUE)
    elseif(DEFINED ENV{NNTILE_PREFIX} AND NOT "$ENV{NNTILE_PREFIX}" STREQUAL "")
        set(_wheel_has_prefix TRUE)
    endif()
    if(NOT _wheel_has_prefix)
        set(BUILD_TORCH_NNTILE ON CACHE BOOL
            "Build LibTorch bridge libtorch_nntile (requires LibTorch)"
            FORCE)
        message(STATUS
            "BUILD_TORCH_NNTILE_WHEEL=ON -> enabling BUILD_TORCH_NNTILE")
    else()
        message(STATUS
            "BUILD_TORCH_NNTILE_WHEEL=ON with install prefix "
            "(not building libtorch_nntile)")
    endif()
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

macro(nntile_apply_common_includes target)
    # BUILD_INTERFACE only: install export uses INSTALL_INTERFACE on
    # libnntile. StarPU headers come from StarPU::starpu.
    target_include_directories(${target} PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/nntile/include>"
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>"
    )
    target_include_directories(${target} PRIVATE
        "${PROJECT_SOURCE_DIR}/external"
    )
    target_link_libraries(${target} PUBLIC StarPU::starpu)
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
        if(NNTILE_CUDA_ARCHITECTURES)
            set_target_properties(${target} PROPERTIES
                CUDA_ARCHITECTURES "${NNTILE_CUDA_ARCHITECTURES}")
        endif()
        target_link_libraries(${target} ${_scope} CUDA::cublas CUDNN::cudnn_all)
        target_include_directories(${target} ${_scope}
            "$<BUILD_INTERFACE:${cudnn_frontend_SOURCE_DIR}/include>")
    endif()
endmacro()

macro(nntile_apply_cblas target)
    nntile_propagation_scope(${target} _scope)
    if(NNTILE_LINK_CBLAS)
        target_link_libraries(${target} ${_scope} BLAS::BLAS)
        if(CBLAS_INCLUDE_DIR)
            target_include_directories(${target} ${_scope} "${CBLAS_INCLUDE_DIR}")
        endif()
    endif()
endmacro()

find_package(PkgConfig REQUIRED)

# Install nlohmann_json with NNTile so find_dependency(nlohmann_json) works.
set(JSON_Install ON CACHE BOOL "Install nlohmann_json with NNTile" FORCE)
add_subdirectory("external")

# StarPU::starpu (pkg-config); shared with installed nntileConfig.cmake.
include("${PROJECT_SOURCE_DIR}/cmake/NNTileFindStarPU.cmake")

include(CheckSymbolExists)
set(CMAKE_REQUIRED_INCLUDES ${StarPU_INCLUDE_DIRS})
check_symbol_exists(STARPU_SIMGRID "starpu_config.h" HAVE_STARPU_SIMGRID)
if(HAVE_STARPU_SIMGRID)
    message(STATUS "StarPU is compiled in SIMGRID mode")
endif()

set(NNTILE_USE_CUDA OFF)
set(NNTILE_LINK_CUDA OFF)
set(NNTILE_CUDA_ARCHITECTURES "")
if(HAVE_STARPU_SIMGRID)
    if(USE_CUDA)
        set(NNTILE_USE_CUDA ON)
    endif()
elseif(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        if(CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
                CACHE STRING "Semicolon-separated CUDA SM versions for NVCC"
                FORCE)
        else()
            set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90"
                CACHE STRING "Semicolon-separated CUDA SM versions for NVCC"
                FORCE)
        endif()
        enable_language(CUDA)
        find_package(CUDAToolkit 11.0 REQUIRED)
        # Prefer pip nvidia-*-cu12 for cublas/cudart when wheel CI (or a
        # local build) exports NVIDIA_* paths / NNTILE_CUDA_FROM_PIP.
        include("${PROJECT_SOURCE_DIR}/cmake/NNTilePreferPipCuda.cmake")
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
        set(NNTILE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()

set(NNTILE_USE_FLASH_SDPA OFF)
if(USE_FLASH_SDPA AND NNTILE_USE_CUDA)
    set(NNTILE_USE_FLASH_SDPA ON)
endif()

set(NNTILE_USE_CBLAS OFF)
set(NNTILE_LINK_CBLAS OFF)
if(HAVE_STARPU_SIMGRID)
    if(USE_CBLAS)
        set(NNTILE_USE_CBLAS ON)
    endif()
elseif(USE_CBLAS)
    find_package(BLAS)
    if(APPLE)
        find_path(CBLAS_INCLUDE_DIR NAMES Accelerate/Accelerate.h)
    else()
        find_path(CBLAS_INCLUDE_DIR NAMES cblas.h PATH_SUFFIXES openblas)
    endif()
    if(BLAS_FOUND AND CBLAS_INCLUDE_DIR)
        set(NNTILE_USE_CBLAS ON)
        set(NNTILE_LINK_CBLAS ON)
    elseif(BLAS_FOUND)
        message(WARNING "BLAS found but CBLAS headers were not found; disabling CBLAS")
    endif()
endif()

if(BUILD_COVERAGE)
    include(CodeCoverage)
    append_coverage_compiler_flags()
    set(COVERAGE_EXCLUDES "${PROJECT_SOURCE_DIR}/nntile/tests/*"
        "${PROJECT_BINARY_DIR}/nntile/tests/*")
    setup_target_for_coverage_lcov(NAME coverage
        EXECUTABLE ctest
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
endif()

# LibTorch is required only when building libtorch_nntile. libnntile and its
# C++ tests configure and build without Torch.
set(NNTILE_HAVE_TORCH OFF)
if(BUILD_TORCH_NNTILE)
    find_package(Torch REQUIRED)
    set(NNTILE_HAVE_TORCH ON)
    message(STATUS "LibTorch found: ${TORCH_INSTALL_PREFIX}")
endif()

if(BUILD_NNTILE)
    add_subdirectory(nntile)
endif()

if(BUILD_TORCH_NNTILE)
    if(NOT TARGET nntile AND NOT TARGET nntile::nntile)
        find_package(nntile REQUIRED CONFIG)
    endif()
    add_subdirectory(torch_nntile)
endif()

# Wheel against installed libs only (no libtorch_nntile rebuild).
if(BUILD_TORCH_NNTILE_WHEEL AND NOT BUILD_TORCH_NNTILE)
    if(NOT TARGET torch_nntile AND NOT TARGET torch_nntile::torch_nntile)
        find_package(torch_nntile REQUIRED CONFIG)
    endif()
    include("${PROJECT_SOURCE_DIR}/torch_nntile/cmake/TorchNntileWheel.cmake")
endif()

# C++ tests follow BUILD_TESTING and only for enabled components.
# Layered CI can also pass all of BUILD_NNTILE / BUILD_TORCH_NNTILE /
# BUILD_TORCH_NNTILE_WHEEL OFF and link tests against an install prefix:
# nntile suite if only libnntile is installed, torch suite if
# libtorch_nntile is present.
if(BUILD_TESTING AND NOT HAVE_STARPU_SIMGRID)
    if(NOT BUILD_NNTILE AND NOT BUILD_TORCH_NNTILE
            AND NOT BUILD_TORCH_NNTILE_WHEEL)
        if(NOT TARGET torch_nntile AND NOT TARGET torch_nntile::torch_nntile)
            find_package(torch_nntile QUIET CONFIG)
        endif()
        if(TARGET torch_nntile OR TARGET torch_nntile::torch_nntile
                OR torch_nntile_FOUND)
            if(NOT Torch_FOUND)
                find_package(Torch REQUIRED)
                set(NNTILE_HAVE_TORCH ON)
            endif()
            add_subdirectory(
                "${PROJECT_SOURCE_DIR}/torch_nntile/tests"
                "${PROJECT_BINARY_DIR}/torch_nntile/tests")
        else()
            if(NOT TARGET nntile AND NOT TARGET nntile::nntile)
                find_package(nntile REQUIRED CONFIG)
            endif()
            if(DEFINED NNTILE_BUILT_WITH_CUDA)
                set(NNTILE_USE_CUDA "${NNTILE_BUILT_WITH_CUDA}")
                set(NNTILE_LINK_CUDA "${NNTILE_BUILT_WITH_CUDA}")
            endif()
            add_subdirectory(
                "${PROJECT_SOURCE_DIR}/nntile/tests"
                "${PROJECT_BINARY_DIR}/nntile/tests")
        endif()
    endif()

    if(BUILD_TORCH_NNTILE)
        add_subdirectory(
            "${PROJECT_SOURCE_DIR}/torch_nntile/tests"
            "${PROJECT_BINARY_DIR}/torch_nntile/tests")
    endif()
endif()
