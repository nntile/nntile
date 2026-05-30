# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# CTest script: run cmake --install and verify the install tree layout.

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

message(STATUS "nntile install verification passed (${CMAKE_INSTALL_PREFIX})")
