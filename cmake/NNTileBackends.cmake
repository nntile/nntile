# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# NNTile 2.0 backend: NNHaul only. libnnhaul is a GPU-only runtime with one
# CUDA worker and no CPU workers; every codelet is chunk-addressable.
#
# Exports the `nnhaul` link target (in-tree subproject or NNHAUL_ROOT).

# NNTile 2.0 has no CPU backend and no second runtime.
if(NOT USE_CUDA)
    message(FATAL_ERROR
        "NNTile 2.0 is CUDA-only: NNHaul creates one CUDA worker and no "
        "CPU workers. Enable USE_CUDA.")
endif()

# No CPU worker exists, so no host BLAS is linked.

# The parent's architecture list governs the subproject; set the default
# before nnhaul enables CUDA so it is not detected per build.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING
        "Semicolon-separated CUDA SM versions for NVCC")
endif()

# -D wins. Without -D the environment seeds the cache, and a later changed
# environment value replaces an entry that still holds the previously seen
# environment value (never a user's -D choice).
if(DEFINED ENV{NNHAUL_ROOT})
    set(_nnhaul_root_env "$ENV{NNHAUL_ROOT}")
    if(NOT DEFINED NNHAUL_ROOT)
        set(NNHAUL_ROOT "${_nnhaul_root_env}" CACHE PATH
            "nnhaul build directory or prefix; empty = use the in-tree nnhaul/")
    elseif(DEFINED NNHAUL_ROOT_ENV_SEEN
            AND "${NNHAUL_ROOT}" STREQUAL "${NNHAUL_ROOT_ENV_SEEN}"
            AND NOT "${_nnhaul_root_env}" STREQUAL "${NNHAUL_ROOT_ENV_SEEN}")
        set(NNHAUL_ROOT "${_nnhaul_root_env}" CACHE PATH
            "nnhaul build directory or prefix; empty = use the in-tree nnhaul/"
            FORCE)
    endif()
    set(NNHAUL_ROOT_ENV_SEEN "${_nnhaul_root_env}" CACHE INTERNAL
        "last NNHAUL_ROOT value seen in the environment")
elseif(NOT DEFINED NNHAUL_ROOT)
    set(NNHAUL_ROOT "" CACHE PATH
        "nnhaul build directory or prefix; empty = use the in-tree nnhaul/")
endif()

if(NNHAUL_ROOT STREQUAL "")
    if(NOT EXISTS "${PROJECT_SOURCE_DIR}/nnhaul/CMakeLists.txt")
        message(FATAL_ERROR
            "Neither the in-tree nnhaul/ directory nor NNHAUL_ROOT is available.")
    endif()
    # A parent that installs nntile must install the nnhaul headers its
    # public headers include, and a test build must build the nnhaul suites.
    if(DEFINED NNHAUL_INSTALL AND NOT NNHAUL_INSTALL)
        message(WARNING
            "NNHAUL_INSTALL=OFF would break consumers of the installed "
            "nntile package; forcing it ON.")
    endif()
    set(NNHAUL_INSTALL ON CACHE BOOL
        "Install and export the nnhaul package" FORCE)
    if(BUILD_TESTING AND NOT DEFINED NNHAUL_BUILD_TESTS)
        set(NNHAUL_BUILD_TESTS ON CACHE BOOL
            "Build the nnhaul test suites" FORCE)
    endif()
    # Embedded builds share ONE engine: libnntile links a shared
    # libnnhaul, so a consumer that also links nnhaul::nnhaul gets the
    # same DSO. Standalone nnhaul stays static.
    if(NOT TARGET nnhaul::nnhaul)
        if(DEFINED NNHAUL_LIBRARY_TYPE
                AND NOT NNHAUL_LIBRARY_TYPE STREQUAL "SHARED")
            message(STATUS
                "nnhaul: embedded build forces SHARED (was "
                "${NNHAUL_LIBRARY_TYPE}); a static copy would give "
                "consumers a second engine")
        endif()
        set(NNHAUL_LIBRARY_TYPE SHARED CACHE STRING
            "nnhaul library type" FORCE)
    endif()
    if(TARGET nnhaul::nnhaul)
        get_target_property(_nnhaul_existing_type nnhaul::nnhaul TYPE)
        if(_nnhaul_existing_type STREQUAL "STATIC_LIBRARY")
            message(FATAL_ERROR
                "NNTile needs a shared nnhaul::nnhaul; the enclosing "
                "project supplied a static archive, which would give "
                "consumers a second engine.")
        endif()
        if(NOT TARGET nnhaul)
            add_library(nnhaul ALIAS nnhaul::nnhaul)
        endif()
        message(STATUS
            "nnhaul: using the nnhaul::nnhaul target from the enclosing project")
    else()
        # NNHaul is the only backend: it is always built. Its own
        # test/example toggles follow NNTile's BUILD_TESTING instead of
        # being independent cache switches; keep them INTERNAL so the
        # configure surface has no way to look like a backend choice.
        set(NNHAUL_BUILD_TESTS ${BUILD_TESTING} CACHE INTERNAL
            "embedded nnhaul: follows BUILD_TESTING" FORCE)
        set(NNHAUL_BUILD_EXAMPLES OFF CACHE INTERNAL
            "embedded nnhaul: examples are not installed" FORCE)
        add_subdirectory("${PROJECT_SOURCE_DIR}/nnhaul")
        message(STATUS "nnhaul: using the in-tree subproject")
    endif()
else()
    unset(NNHAUL_INCLUDE_DIR CACHE)
    unset(NNHAUL_LIBRARY CACHE)
    find_path(NNHAUL_INCLUDE_DIR nnhaul/nnhaul.hh
        HINTS "${NNHAUL_ROOT}/include" "${NNHAUL_ROOT}/../include"
              "${NNHAUL_ROOT}/nnhaul/include"
              "${NNHAUL_ROOT}/nnhaul/../include")
    find_library(NNHAUL_LIBRARY nnhaul
        HINTS "${NNHAUL_ROOT}" "${NNHAUL_ROOT}/lib"
              "${NNHAUL_ROOT}/nnhaul")
    if(NOT NNHAUL_INCLUDE_DIR OR NOT NNHAUL_LIBRARY)
        message(FATAL_ERROR
            "libnnhaul was not found under NNHAUL_ROOT=${NNHAUL_ROOT}.")
    endif()
    if(NNHAUL_LIBRARY MATCHES "\\.a$")
        message(FATAL_ERROR
            "NNHAUL_ROOT=${NNHAUL_ROOT} holds a static libnnhaul. The "
            "embedded nntile links a shared libnnhaul so consumers get one "
            "engine; point NNHAUL_ROOT at a shared build/install, or drop "
            "NNHAUL_ROOT to use the in-tree shared nnhaul.")
    endif()
    find_package(CUDAToolkit REQUIRED)
    find_path(CUDNN_INCLUDE_DIR cudnn.h REQUIRED)
    find_library(CUDNN_LIBRARY cudnn REQUIRED)
    get_filename_component(_nnhaul_libdir "${NNHAUL_LIBRARY}" DIRECTORY)
    list(APPEND CMAKE_INSTALL_RPATH "${_nnhaul_libdir}")
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}")
    add_library(nnhaul UNKNOWN IMPORTED)
    set_target_properties(nnhaul PROPERTIES
        IMPORTED_LOCATION "${NNHAUL_LIBRARY}")
    target_include_directories(nnhaul INTERFACE
        "${NNHAUL_INCLUDE_DIR}"
        ${CUDAToolkit_INCLUDE_DIRS}
        "${CUDNN_INCLUDE_DIR}")
    # defs.h carries the configured chunk class; linking a libnnhaul built
    # with a different one would dispatch with mismatched chunk arithmetic.
    if(EXISTS "${NNHAUL_ROOT}/include/nnhaul/defs.h")
        target_include_directories(nnhaul INTERFACE
            "${NNHAUL_ROOT}/include")
    elseif(EXISTS "${NNHAUL_ROOT}/nnhaul/include/nnhaul/defs.h")
        target_include_directories(nnhaul INTERFACE
            "${NNHAUL_ROOT}/nnhaul/include")
    elseif(EXISTS "${NNHAUL_ROOT}/../include/nnhaul/defs.h")
        target_include_directories(nnhaul INTERFACE
            "${NNHAUL_ROOT}/../include")
    else()
        message(FATAL_ERROR
            "NNHAUL_ROOT=${NNHAUL_ROOT} has no include/nnhaul/defs.h. "
            "Point it at a nnhaul build directory or install prefix; the "
            "generated defs.h carries the chunk class and must match.")
    endif()
    target_link_libraries(nnhaul INTERFACE
        CUDA::cudart
        CUDA::cublas
        "${CUDNN_LIBRARY}")
endif()
