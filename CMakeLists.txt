# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems with GPU acceleration.
#
# @file CMakeLists.txt
# Top-level CMake (unified nntile package).
#
# @version 2.0.0

cmake_minimum_required(VERSION 3.24)
message(STATUS "Using CMake ${CMAKE_VERSION}")

project(NNTile VERSION 2.0.0 LANGUAGES CXX C)

# Installed shared libraries resolve their siblings (libnnhaul beside
# libnntile) without LD_LIBRARY_PATH.
if(UNIX AND NOT APPLE)
    if(NOT CMAKE_INSTALL_RPATH MATCHES "\\$ORIGIN")
        if(DEFINED CMAKE_INSTALL_RPATH AND NOT CMAKE_INSTALL_RPATH STREQUAL "")
            set(CMAKE_INSTALL_RPATH "$ORIGIN;${CMAKE_INSTALL_RPATH}")
        else()
            set(CMAKE_INSTALL_RPATH "$ORIGIN")
        endif()
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

option(BUILD_SHARED_LIBS "Build shared libraries instead of static" ON)
# libnntile's object libraries are exported only through the shared build;
# a static archive would need them installed as objects. Fail clearly.
if(NOT BUILD_SHARED_LIBS)
    message(FATAL_ERROR
        "Static libnntile is not supported: the object libraries that make "
        "up nntile are not part of the install export. Build with "
        "-DBUILD_SHARED_LIBS=ON.")
endif()
option(BUILD_LIBNNTILE "Build the core C++ libnntile library" ON)
# NNTile 2.0 has exactly one runtime: NNHaul. It is CUDA-only and its
# codelets are chunked-only. There is no backend switch.
# NNTile 2.0 has no CUDA switch: the NNHaul backend and every chunked
# kernel entry point require CUDA, enabled unconditionally below.
# Default ON: the user-facing product is the torch_nntile pip wheel (needs
# LibTorch on CMAKE_PREFIX_PATH). Layered CI and Torch-less configures must
# pass -DBUILD_LIBTORCH_NNTILE=OFF -DBUILD_TORCH_NNTILE=OFF.
option(BUILD_LIBTORCH_NNTILE
    "Build LibTorch bridge libtorch_nntile (requires LibTorch)" ON)
option(BUILD_TORCH_NNTILE
    "Build torch_nntile Python product (_C extension + pip wheel)" ON)
option(BUILD_COVERAGE "Generate code coverage report" OFF)
# NNTile 2.0 has one torch_nntile stack: classic nntile kernels and models.
# There is no ATen-on-nntile dispatch family to select.

# NNTILE_PREFIX / TORCH_NNTILE_PREFIX are the documented layered-install
# knobs; fold them into CMAKE_PREFIX_PATH so config-mode find_package sees
# <prefix>/lib/cmake/<package> without the caller repeating them.
foreach(_nntile_prefix IN ITEMS "${NNTILE_PREFIX}" "${TORCH_NNTILE_PREFIX}")
    if(NOT "${_nntile_prefix}" STREQUAL "")
        list(APPEND CMAKE_PREFIX_PATH "${_nntile_prefix}")
    endif()
endforeach()

# The disabled-family list and its test filter must exist before either
# the product subdirectory or a tests-only test tree is added.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/NNTileKernelScope.cmake")

# Standard CMake testing switch (default ON). Tests are added only for
# enabled components: BUILD_LIBNNTILE, BUILD_LIBTORCH_NNTILE, and/or
# BUILD_TORCH_NNTILE. Pass -DBUILD_TESTING=OFF to skip all tests.
include(CTest)
if(BUILD_COVERAGE)
    set(BUILD_TESTING ON CACHE BOOL "Build the testing tree" FORCE)
endif()

# In-tree wheel needs libtorch_nntile. Against an install prefix, keep
# BUILD_LIBTORCH_NNTILE=OFF and either put the prefix on CMAKE_PREFIX_PATH
# or pass -DNNTILE_PREFIX=... (and -DTORCH_NNTILE_PREFIX=... for the bridge
# package); the prefixes below are used as find_package hints.
if(BUILD_TORCH_NNTILE AND NOT BUILD_LIBTORCH_NNTILE)
    set(_wheel_has_prefix FALSE)
    if(DEFINED NNTILE_PREFIX AND NOT NNTILE_PREFIX STREQUAL "")
        set(_wheel_has_prefix TRUE)
    elseif(DEFINED ENV{NNTILE_PREFIX} AND NOT "$ENV{NNTILE_PREFIX}" STREQUAL "")
        set(_wheel_has_prefix TRUE)
    endif()
    if(NOT _wheel_has_prefix)
        set(BUILD_LIBTORCH_NNTILE ON CACHE BOOL
            "Build LibTorch bridge libtorch_nntile (requires LibTorch)"
            FORCE)
        message(STATUS
            "BUILD_TORCH_NNTILE=ON -> enabling BUILD_LIBTORCH_NNTILE")
    else()
        message(STATUS
            "BUILD_TORCH_NNTILE=ON with install prefix "
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
    # libnntile.
    target_include_directories(${target} PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/nntile/include>"
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>"
    )
    target_include_directories(${target} PRIVATE
        "${PROJECT_SOURCE_DIR}/external"
    )
    target_link_libraries(${target} PUBLIC
        "$<BUILD_INTERFACE:nnhaul>"
        "$<INSTALL_INTERFACE:nnhaul::nnhaul>")
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
            "$<BUILD_INTERFACE:${cudnn_frontend_SOURCE_DIR}/include>")
    endif()
endmacro()

# Install nlohmann_json with NNTile so find_dependency(nlohmann_json) works.
if(DEFINED JSON_Install AND NOT JSON_Install)
    message(WARNING
        "JSON_Install=OFF is not supported: the installed nntile package "
        "exports a find_dependency(nlohmann_json); forcing it ON.")
endif()
set(JSON_Install ON CACHE BOOL "Install nlohmann_json with NNTile" FORCE)
add_subdirectory("external")

# Tests-only layered build against an installed nntile: adopt the
# installed product's CUDA choice before it is enabled
# against this machine, otherwise the test tree can build NNHaul tests
# for an NNHaul install (or the reverse).
if(BUILD_TESTING AND NOT BUILD_LIBNNTILE AND NOT BUILD_LIBTORCH_NNTILE
        AND NOT BUILD_TORCH_NNTILE)
    find_package(nntile QUIET CONFIG HINTS "${NNTILE_PREFIX}")
    if(DEFINED NNTILE_BUILT_WITH_CUDA AND NOT NNTILE_BUILT_WITH_CUDA)
        message(FATAL_ERROR
            "the installed nntile was built without CUDA; NNTile 2.0 is "
            "CUDA-only, so its tests cannot run against it")
    endif()
    # The test lists must reflect the installed library's disabled families,
    # not this source tree's default. An install configured with
    # -DNNTILE_DISABLED_KERNELS= (or a superset) must register exactly the
    # matching tests.
    if(DEFINED NNTILE_BUILT_WITH_DISABLED_KERNELS)
        set(NNTILE_DISABLED_KERNELS
            "${NNTILE_BUILT_WITH_DISABLED_KERNELS}" CACHE STRING
            "Semicolon-separated kernel families disabled in this build"
            FORCE)
    endif()
endif()

set(NNTILE_USE_CUDA ON)
set(NNTILE_LINK_CUDA ON)
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
    find_package(CUDAToolkit 12.0 REQUIRED)
    # Prefer pip nvidia-*-cu12 for cublas/cudart when wheel CI (or a
    # local build) exports NVIDIA_* paths / NNTILE_CUDA_FROM_PIP.
    include("${PROJECT_SOURCE_DIR}/cmake/NNTilePreferPipCuda.cmake")
    add_subdirectory("external/cudnn_frontend")
    # An installed nntile found by find_package already imported
    # CUDNN::cudnn_all; importing it twice is a CMake error.
    if(NOT TARGET CUDNN::cudnn_all)
        include(${cudnn_frontend_SOURCE_DIR}/cmake/cuDNN.cmake)
    endif()
else()
    message(FATAL_ERROR
        "no CUDA compiler was found; NNTile 2.0 is CUDA-only.")
endif()

# A layered build (BUILD_LIBNNTILE=OFF) must discover the installed
# nntile/nnhaul targets before NNTileBackends.cmake decides to vendor the
# in-tree engine; otherwise the imported nnhaul::nnhaul target collides with
# the in-tree alias from add_subdirectory(nnhaul).
if(NOT BUILD_LIBNNTILE)
    find_package(nntile QUIET CONFIG HINTS "${NNTILE_PREFIX}")
endif()

# Shared cudart for every consumer of libnntile: the nnhaul branch already
# pulls CUDA::cudart (shared) via the imported nnhaul target, while the
# NNHaul-only path silently took nvcc's static default - two different
# cudart instances across wheels of the same library, and a baked-in launch
# path that no interposition or profiler can see. Set before the NNHaul
# targets are created: a target-free variable is captured at creation.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

include("${PROJECT_SOURCE_DIR}/cmake/NNTileBackends.cmake")

# An external NNHAUL_ROOT is not copied into this install prefix, so the
# install-verification test must not demand its headers/library there.
set(NNTILE_NNHAUL_EXTERNAL FALSE)
if(NOT NNHAUL_ROOT STREQUAL "")
    set(NNTILE_NNHAUL_EXTERNAL TRUE)
endif()


# No CPU worker exists, so no host BLAS is linked.

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

# LibTorch is required for libtorch_nntile.
if(BUILD_LIBTORCH_NNTILE)
    find_package(Torch REQUIRED)
    message(STATUS "LibTorch found: ${TORCH_INSTALL_PREFIX}")
endif()

if(BUILD_LIBNNTILE)
    add_subdirectory(nntile)
endif()

if(BUILD_LIBTORCH_NNTILE)
    if(NOT TARGET nntile AND NOT TARGET nntile::nntile)
        find_package(nntile REQUIRED CONFIG HINTS "${NNTILE_PREFIX}")
    endif()
    add_subdirectory(torch_nntile)
endif()

# Wheel against installed libs only (no libtorch_nntile rebuild).
if(BUILD_TORCH_NNTILE AND NOT BUILD_LIBTORCH_NNTILE)
    if(NOT TARGET torch_nntile AND NOT TARGET torch_nntile::torch_nntile)
        find_package(torch_nntile REQUIRED CONFIG HINTS "${TORCH_NNTILE_PREFIX}" "${NNTILE_PREFIX}")
    endif()
    include("${PROJECT_SOURCE_DIR}/torch_nntile/cmake/TorchNntileWheel.cmake")
endif()

# C++ tests follow BUILD_TESTING and only for enabled components.
# Layered CI can also pass all of BUILD_LIBNNTILE / BUILD_LIBTORCH_NNTILE /
# BUILD_TORCH_NNTILE OFF and link tests against an install prefix:
# nntile suite if only libnntile is installed, torch suite if
# libtorch_nntile is present.
if(BUILD_TESTING)
    if(NOT BUILD_LIBNNTILE AND NOT BUILD_LIBTORCH_NNTILE
            AND NOT BUILD_TORCH_NNTILE)
        if(NOT TARGET torch_nntile AND NOT TARGET torch_nntile::torch_nntile)
            find_package(torch_nntile QUIET CONFIG HINTS "${TORCH_NNTILE_PREFIX}" "${NNTILE_PREFIX}")
        endif()
        if(TARGET torch_nntile OR TARGET torch_nntile::torch_nntile
                OR torch_nntile_FOUND)
            if(NOT Torch_FOUND)
                find_package(Torch REQUIRED)
            endif()
            add_subdirectory(
                "${PROJECT_SOURCE_DIR}/torch_nntile/tests"
                "${PROJECT_BINARY_DIR}/torch_nntile/tests")
        else()
            if(NOT TARGET nntile AND NOT TARGET nntile::nntile)
                find_package(nntile REQUIRED CONFIG HINTS "${NNTILE_PREFIX}")
            endif()
            add_subdirectory(
                "${PROJECT_SOURCE_DIR}/nntile/tests"
                "${PROJECT_BINARY_DIR}/nntile/tests")
        endif()
    endif()

    if(BUILD_LIBTORCH_NNTILE)
        add_subdirectory(
            "${PROJECT_SOURCE_DIR}/torch_nntile/tests"
            "${PROJECT_BINARY_DIR}/torch_nntile/tests")
    elseif(BUILD_TORCH_NNTILE AND NOT BUILD_LIBNNTILE)
        if(NOT TARGET torch_nntile AND NOT TARGET torch_nntile::torch_nntile)
            find_package(torch_nntile REQUIRED CONFIG HINTS "${TORCH_NNTILE_PREFIX}" "${NNTILE_PREFIX}")
        endif()
        if(NOT Torch_FOUND)
            find_package(Torch REQUIRED)
        endif()
        add_subdirectory(
            "${PROJECT_SOURCE_DIR}/torch_nntile/tests"
            "${PROJECT_BINARY_DIR}/torch_nntile/tests")
    endif()
endif()
