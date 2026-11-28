# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# CTest script: run cmake --install and verify the install tree layout.
# @version 2.0.0

if(NOT DEFINED BUILD_DIR)
    message(FATAL_ERROR "BUILD_DIR is required")
endif()
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    message(FATAL_ERROR "CMAKE_INSTALL_PREFIX is required")
endif()
if(NOT CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR "lib")
endif()
if(NOT DEFINED NNTILE_INSTALL_LIB_BASENAME)
    message(FATAL_ERROR
        "NNTILE_INSTALL_LIB_BASENAME is required "
        "(set in CMakeLists.txt or passed with -D; WIN32/APPLE are unset "
        "in cmake -P script mode)")
endif()

file(REMOVE_RECURSE "${CMAKE_INSTALL_PREFIX}")

execute_process(
    COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}"
        --prefix "${CMAKE_INSTALL_PREFIX}"
    RESULT_VARIABLE _install_result
    ERROR_VARIABLE _install_err
)
if(NOT _install_result EQUAL 0)
    message(FATAL_ERROR "cmake --install failed:\n${_install_err}")
endif()

set(_include_root "${CMAKE_INSTALL_PREFIX}/include")
set(_required_headers
    "${_include_root}/nntile.hh"
    "${_include_root}/nntile/core.hh"
    "${_include_root}/nntile/defs.h"
)
foreach(_header IN LISTS _required_headers)
    if(NOT EXISTS "${_header}")
        message(FATAL_ERROR "Missing installed header: ${_header}")
    endif()
endforeach()

if(EXISTS "${_include_root}/nntile/nntile/core.hh")
    message(FATAL_ERROR
        "Headers were installed under include/nntile/nntile/ "
        "(duplicate nntile prefix)")
endif()

set(_lib_dir "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
set(_lib_path "${_lib_dir}/${NNTILE_INSTALL_LIB_BASENAME}")
if(NOT EXISTS "${_lib_path}")
    message(FATAL_ERROR "Missing installed library: ${_lib_path}")
endif()

if(NOT NNTILE_NNHAUL_EXTERNAL)
    foreach(_h IN ITEMS nnhaul/nnhaul.hh nnhaul/planner.hh
            nnhaul/defs.h)
        if(NOT EXISTS "${_include_root}/${_h}")
            message(FATAL_ERROR "Missing installed nnhaul header: "
                "${_include_root}/${_h}")
        endif()
    endforeach()
    foreach(_f IN ITEMS nnhaulConfig.cmake nnhaulConfigVersion.cmake
            nnhaulTargets.cmake)
        if(NOT EXISTS "${_lib_dir}/cmake/nnhaul/${_f}")
            message(FATAL_ERROR "Missing installed nnhaul package file: "
                "${_lib_dir}/cmake/nnhaul/${_f}")
        endif()
    endforeach()
    file(GLOB _nnhaul_libs "${_lib_dir}/*nnhaul*")
    if(NOT _nnhaul_libs)
        message(FATAL_ERROR "Missing installed nnhaul library under "
            "${_lib_dir}")
    endif()
endif()

set(_nntile_cmake_dir "${_lib_dir}/cmake/nntile")
foreach(_cmake_file IN ITEMS
        nntileConfig.cmake
        nntileConfigVersion.cmake
        nntileTargets.cmake)
    set(_path "${_nntile_cmake_dir}/${_cmake_file}")
    if(NOT EXISTS "${_path}")
        message(FATAL_ERROR "Missing installed CMake package file: ${_path}")
    endif()
endforeach()

if(DEFINED TORCH_NNTILE_INSTALL_LIB_BASENAME)
    set(_tn_lib "${_lib_dir}/${TORCH_NNTILE_INSTALL_LIB_BASENAME}")
    if(NOT EXISTS "${_tn_lib}")
        message(FATAL_ERROR "Missing installed library: ${_tn_lib}")
    endif()
    set(_tn_cmake_dir "${_lib_dir}/cmake/torch_nntile")
    foreach(_cmake_file IN ITEMS
            torch_nntileConfig.cmake
            torch_nntileConfigVersion.cmake
            torch_nntileTargets.cmake)
        set(_path "${_tn_cmake_dir}/${_cmake_file}")
        if(NOT EXISTS "${_path}")
            message(FATAL_ERROR
                "Missing installed CMake package file: ${_path}")
        endif()
    endforeach()
    set(_tn_header
        "${_include_root}/torch_nntile/torch_nntile.hh")
    if(NOT EXISTS "${_tn_header}")
        message(FATAL_ERROR "Missing installed header: ${_tn_header}")
    endif()
endif()

message(STATUS "nntile install verification passed (${CMAKE_INSTALL_PREFIX})")
