# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# CTest script: run cmake --install for libtorch_nntile and verify the
# install tree layout (expects libnntile already present in the prefix).

if(NOT DEFINED BUILD_DIR)
    message(FATAL_ERROR "BUILD_DIR is required")
endif()
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
    message(FATAL_ERROR "CMAKE_INSTALL_PREFIX is required")
endif()
if(NOT CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR "lib")
endif()
if(NOT DEFINED TORCH_NNTILE_INSTALL_LIB_BASENAME)
    set(TORCH_NNTILE_INSTALL_LIB_BASENAME "libtorch_nntile.so")
endif()

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
set(_lib_dir "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")

set(_tn_lib "${_lib_dir}/${TORCH_NNTILE_INSTALL_LIB_BASENAME}")
if(NOT EXISTS "${_tn_lib}")
    # Accept SONAME variants (libtorch_nntile.so.1, ...).
    file(GLOB _tn_libs "${_lib_dir}/libtorch_nntile.so*")
    if(NOT _tn_libs)
        message(FATAL_ERROR "Missing installed library under ${_lib_dir}: "
            "${TORCH_NNTILE_INSTALL_LIB_BASENAME}")
    endif()
endif()

set(_tn_header "${_include_root}/torch_nntile/torch_nntile.hh")
if(NOT EXISTS "${_tn_header}")
    message(FATAL_ERROR "Missing installed header: ${_tn_header}")
endif()

set(_tn_cmake_dir "${_lib_dir}/cmake/torch_nntile")
foreach(_cmake_file IN ITEMS
        torch_nntileConfig.cmake
        torch_nntileConfigVersion.cmake
        torch_nntileTargets.cmake)
    set(_path "${_tn_cmake_dir}/${_cmake_file}")
    if(NOT EXISTS "${_path}")
        message(FATAL_ERROR "Missing installed CMake package file: ${_path}")
    endif()
endforeach()

# libnntile must be present in the same prefix for consumers.
set(_nntile_lib "${_lib_dir}/libnntile.so")
if(NOT EXISTS "${_nntile_lib}")
    file(GLOB _nntile_libs "${_lib_dir}/libnntile.so*")
    if(NOT _nntile_libs)
        message(FATAL_ERROR
            "Missing libnntile in the same prefix (${_lib_dir})")
    endif()
endif()

message(STATUS
    "libtorch_nntile install verification passed (${CMAKE_INSTALL_PREFIX})")
