# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file CMakeLists.txt
# This is a top-level CMake script, that is tested to work only with a
# single-configuration build systems like Ninja or Unix Makefile.
#
# @version 1.1.0


###############################################################################
#                       THIS IS A TOP-LEVEL CMAKELISTS.txt                    #
#                                                                             #
#        It is intended to find all dependencies (required or optional)       #
#                    and set up corresponding variables                       #
###############################################################################

cmake_minimum_required(VERSION 3.24)
message(STATUS "Using CMake ${CMAKE_VERSION}")

project(NNTile VERSION 1.1.0 LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules")

option(BUILD_SHARED_LIBS "Build shared libraries instead of static" ON)
option(BUILD_CORE "Build nntile_core library" ON)
option(BUILD_GRAPH "Build nntile_graph library" ON)
option(USE_CUDA "Use CUDA toolkit (cuBLAS and cuDNN are required)" ON)
option(USE_CUDA_FP16 "Enable FP16 CUDA support" ON)
option(USE_CUDA_TF32 "Enable TF32 CUDA support" ON)
option(USE_CUDA_BF16 "Enable BF16 CUDA support" ON)
option(USE_CUDA_FP8 "Enable FP8 CUDA support" ON)
option(USE_CBLAS "Use CPU CBLAS" ON)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_TESTS_CORE "Build core C++ tests" ON)
option(BUILD_TESTS_GRAPH "Build graph C++ tests" ON)
option(BUILD_TESTS_PYTORCH "Build NNGraph tests comparing against PyTorch C++ frontend (LibTorch)" ON)
option(BUILD_DOCS "Build Doxygen-based documentation" OFF)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_COVERAGE "Generate code coverage report" OFF)
option(BUILD_PYTHON_WRAPPERS "Generate Python wrappers" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

macro(nntile_apply_common_includes target)
    target_include_directories(${target} PUBLIC
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_BINARY_DIR}/include"
        ${StarPU_INCLUDE_DIRS}
    )
    target_include_directories(${target} PRIVATE
        "${PROJECT_SOURCE_DIR}/external"
    )
endmacro()

macro(nntile_apply_cuda target)
    if(NNTILE_LINK_CUDA)
        target_link_libraries(${target} PUBLIC CUDA::cublas CUDNN::cudnn_all)
        target_include_directories(${target} PUBLIC ${cudnn_frontend_SOURCE_DIR}/include)
    endif()
endmacro()

macro(nntile_apply_cblas target)
    if(NNTILE_LINK_CBLAS)
        target_link_libraries(${target} PUBLIC BLAS::BLAS)
    endif()
endmacro()

if(BUILD_CORE)
    add_library(nntile_core)
endif()
if(BUILD_GRAPH)
    add_library(nntile_graph)
endif()

find_package(PkgConfig REQUIRED)
add_subdirectory("external")

pkg_check_modules(StarPU REQUIRED starpu-1.4)

if(NOT BUILD_CORE AND BUILD_GRAPH)
    set(NNTILE_CORE_PREFIX "" CACHE PATH
        "Install prefix of a pre-built nntile_core (lib/ and include/)")
    if(NOT NNTILE_CORE_PREFIX)
        message(FATAL_ERROR
            "NNTILE_CORE_PREFIX must point at a core install when BUILD_CORE=OFF")
    endif()
    include("${PROJECT_SOURCE_DIR}/cmake/NNTileCoreImported.cmake")
    nntile_import_installed_core("${NNTILE_CORE_PREFIX}")
endif()

if(BUILD_CORE)
    target_link_libraries(nntile_core PUBLIC ${StarPU_LDFLAGS} nlohmann_json::nlohmann_json)
    nntile_apply_common_includes(nntile_core)
endif()
if(BUILD_GRAPH)
    if(NOT TARGET nntile_core)
        message(FATAL_ERROR "nntile_graph requires target nntile_core")
    endif()
    target_link_libraries(nntile_graph PUBLIC nntile_core)
    nntile_apply_common_includes(nntile_graph)
endif()

include(CheckSymbolExists)
set(CMAKE_REQUIRED_INCLUDES ${StarPU_INCLUDE_DIRS})
check_symbol_exists(STARPU_SIMGRID "starpu_config.h" HAVE_STARPU_SIMGRID)
if(HAVE_STARPU_SIMGRID)
    message(STATUS "StarPU is compiled in SIMGRID mode")
    if(BUILD_CORE)
        target_compile_definitions(nntile_core PUBLIC STARPU_SIMGRID)
    endif()
    if(BUILD_GRAPH)
        target_compile_definitions(nntile_graph PUBLIC STARPU_SIMGRID)
    endif()
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
        message(STATUS "cuDNN frontend SOURCE: ${cudnn_frontend_SOURCE_DIR}")
        message(STATUS "cuDNN frontend BINARY: ${cudnn_frontend_BINARY_DIR}")
        include(${cudnn_frontend_SOURCE_DIR}/cmake/cuDNN.cmake)
        set(NNTILE_USE_CUDA ON)
        set(NNTILE_LINK_CUDA ON)
        if(USE_CUDA_FP16)
            set(NNTILE_USE_CUDA_FP16 ON)
            message(STATUS "CUDA FP16 support enabled")
        else()
            set(NNTILE_USE_CUDA_FP16 OFF)
            message(STATUS "CUDA FP16 support disabled")
        endif()
        if(USE_CUDA_TF32)
            set(NNTILE_USE_CUDA_TF32 ON)
            message(STATUS "CUDA TF32 support enabled")
        else()
            set(NNTILE_USE_CUDA_TF32 OFF)
            message(STATUS "CUDA TF32 support disabled")
        endif()
        if(USE_CUDA_BF16)
            set(NNTILE_USE_CUDA_BF16 ON)
            message(STATUS "CUDA BF16 support enabled")
        else()
            set(NNTILE_USE_CUDA_BF16 OFF)
            message(STATUS "CUDA BF16 support disabled")
        endif()
        if(USE_CUDA_FP8 AND CUDAToolkit_VERSION GREATER_EQUAL 11.8)
            set(NNTILE_USE_CUDA_FP8 ON)
            message(STATUS "CUDA FP8 support enabled")
        else()
            set(NNTILE_USE_CUDA_FP8 OFF)
            message(STATUS "CUDA FP8 support disabled")
        endif()
    endif()
endif()

set(NNTILE_USE_CBLAS OFF)
set(NNTILE_LINK_CBLAS OFF)
if(HAVE_STARPU_SIMGRID)
    if(USE_CBLAS)
        set(NNTILE_USE_CBLAS ON)
        set(NNTILE_LINK_CBLAS OFF)
    endif()
elseif(USE_CBLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        set(NNTILE_USE_CBLAS ON)
        set(NNTILE_LINK_CBLAS ON)
    endif()
endif()

if(BUILD_CORE)
    nntile_apply_cuda(nntile_core)
    nntile_apply_cblas(nntile_core)
endif()

if(BUILD_CORE)
    configure_file("${PROJECT_SOURCE_DIR}/include/nntile/core/defs.h.in"
        "${PROJECT_BINARY_DIR}/include/nntile/core/defs.h" @ONLY)
endif()

if(BUILD_COVERAGE)
    message(STATUS "Code coverage report was requested, so option BUILD_TESTS"
        " is ON."
        " The report itself can be generated by \"make coverage\" command.")
    set(BUILD_TESTS ON)
    include(CodeCoverage)
    append_coverage_compiler_flags()
    set(COVERAGE_EXCLUDES "${PROJECT_SOURCE_DIR}/tests/*"
        "${PROJECT_BINARY_DIR}/tests/*")
    setup_target_for_coverage_lcov(NAME coverage
        EXECUTABLE ctest
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
endif()

add_subdirectory("src")
add_subdirectory("include")

set(NNTILE_HAVE_TORCH OFF)
if(BUILD_TESTS_PYTORCH AND BUILD_TESTS AND BUILD_GRAPH AND NOT HAVE_STARPU_SIMGRID)
    find_package(Torch QUIET)
    if(Torch_FOUND)
        set(NNTILE_HAVE_TORCH ON)
        message(STATUS "LibTorch found: ${TORCH_INSTALL_PREFIX}")
    else()
        message(WARNING "BUILD_TESTS_PYTORCH=ON but LibTorch not found. "
            "Set CMAKE_PREFIX_PATH to LibTorch root, e.g.: "
            "cmake -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') ..")
    endif()
endif()

if(BUILD_TESTS AND NOT HAVE_STARPU_SIMGRID)
    include(CTest)
    add_subdirectory("tests")
endif()

if(BUILD_EXAMPLES)
    add_subdirectory("examples")
endif()

if(BUILD_PYTHON_WRAPPERS)
    add_subdirectory("wrappers/python")
endif()

if(BUILD_CORE)
    include(GNUInstallDirs)

    install(
        TARGETS nntile_core
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(
        DIRECTORY "${PROJECT_SOURCE_DIR}/include/nntile"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hh"
    )
    install(
        FILES "${PROJECT_SOURCE_DIR}/include/nntile.hh"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(
        FILES "${PROJECT_BINARY_DIR}/include/nntile/core/defs.h"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nntile/core
    )
endif()
