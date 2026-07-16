# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/cmake/TorchNntileWheel.cmake
# Custom target: build torch_nntile Python wheel against in-tree libs or an
# install prefix (-DNNTILE_PREFIX / -DTORCH_NNTILE_PREFIX).

if(NOT BUILD_TORCH_NNTILE_WHEEL)
    return()
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(TORCH_NNTILE_WHEELHOUSE "${CMAKE_BINARY_DIR}/wheelhouse"
    CACHE PATH "Output directory for torch_nntile wheels")
set(TORCH_NNTILE_WHEEL_VERSION "0.0.5"
    CACHE STRING "torch_nntile wheel version (TORCH_NNTILE_WHEEL_VERSION)")
option(TORCH_NNTILE_WHEEL_REPAIR
    "Run auditwheel/delocate repair after pip wheel" ON)

set(NNTILE_PREFIX "" CACHE PATH
    "Install prefix containing libnntile (wheel / tests)")
set(TORCH_NNTILE_PREFIX "" CACHE PATH
    "Install prefix containing libtorch_nntile (defaults to NNTILE_PREFIX)")

file(MAKE_DIRECTORY "${TORCH_NNTILE_WHEELHOUSE}")
set(_torch_nntile_wheel_raw "${TORCH_NNTILE_WHEELHOUSE}/raw")
file(MAKE_DIRECTORY "${_torch_nntile_wheel_raw}")

# --- resolve libnntile -------------------------------------------------------
set(_wheel_nntile_build_dir "")
set(_wheel_nntile_prefix "")
if(TARGET nntile)
    set(_wheel_nntile_build_dir "${CMAKE_BINARY_DIR}")
elseif(NOT NNTILE_PREFIX STREQUAL "")
    set(_wheel_nntile_prefix "${NNTILE_PREFIX}")
elseif(DEFINED ENV{NNTILE_PREFIX} AND NOT "$ENV{NNTILE_PREFIX}" STREQUAL "")
    set(_wheel_nntile_prefix "$ENV{NNTILE_PREFIX}")
elseif(TARGET nntile::nntile AND DEFINED nntile_DIR)
    get_filename_component(_wheel_nntile_prefix
        "${nntile_DIR}/../../.." ABSOLUTE)
else()
    message(FATAL_ERROR
        "BUILD_TORCH_NNTILE_WHEEL needs in-tree nntile or "
        "-DNNTILE_PREFIX=... (install prefix with libnntile)")
endif()

# --- resolve libtorch_nntile -------------------------------------------------
set(_wheel_torch_build_dir "")
set(_wheel_torch_prefix "")
if(TARGET torch_nntile)
    set(_wheel_torch_build_dir "${CMAKE_BINARY_DIR}")
elseif(NOT TORCH_NNTILE_PREFIX STREQUAL "")
    set(_wheel_torch_prefix "${TORCH_NNTILE_PREFIX}")
elseif(NOT _wheel_nntile_prefix STREQUAL "")
    set(_wheel_torch_prefix "${_wheel_nntile_prefix}")
elseif(DEFINED ENV{TORCH_NNTILE_PREFIX}
        AND NOT "$ENV{TORCH_NNTILE_PREFIX}" STREQUAL "")
    set(_wheel_torch_prefix "$ENV{TORCH_NNTILE_PREFIX}")
elseif(TARGET torch_nntile::torch_nntile AND DEFINED torch_nntile_DIR)
    get_filename_component(_wheel_torch_prefix
        "${torch_nntile_DIR}/../../.." ABSOLUTE)
else()
    message(FATAL_ERROR
        "BUILD_TORCH_NNTILE_WHEEL needs in-tree torch_nntile or "
        "-DTORCH_NNTILE_PREFIX=... / -DNNTILE_PREFIX=...")
endif()

set(_wheel_env
    "TORCH_NNTILE_WHEEL=1"
    "TORCH_NNTILE_WHEEL_VERSION=${TORCH_NNTILE_WHEEL_VERSION}"
    "NNTILE_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
)
if(_wheel_nntile_build_dir)
    list(APPEND _wheel_env "NNTILE_BUILD_DIR=${_wheel_nntile_build_dir}")
endif()
if(_wheel_torch_build_dir)
    list(APPEND _wheel_env
        "TORCH_NNTILE_BUILD_DIR=${_wheel_torch_build_dir}")
endif()
if(_wheel_nntile_prefix)
    list(APPEND _wheel_env "NNTILE_PREFIX=${_wheel_nntile_prefix}")
endif()
if(_wheel_torch_prefix)
    list(APPEND _wheel_env "TORCH_NNTILE_PREFIX=${_wheel_torch_prefix}")
endif()
if(DEFINED ENV{PKG_CONFIG_PATH})
    list(APPEND _wheel_env "PKG_CONFIG_PATH=$ENV{PKG_CONFIG_PATH}")
endif()
if(DEFINED ENV{LD_LIBRARY_PATH})
    list(APPEND _wheel_env "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}")
endif()
if(DEFINED ENV{DYLD_LIBRARY_PATH})
    list(APPEND _wheel_env "DYLD_LIBRARY_PATH=$ENV{DYLD_LIBRARY_PATH}")
endif()
if(DEFINED ENV{TORCH_NNTILE_USE_CUDA})
    list(APPEND _wheel_env "TORCH_NNTILE_USE_CUDA=$ENV{TORCH_NNTILE_USE_CUDA}")
endif()
if(DEFINED ENV{CUDA_HOME})
    list(APPEND _wheel_env "CUDA_HOME=$ENV{CUDA_HOME}")
endif()
if(DEFINED ENV{CC})
    list(APPEND _wheel_env "CC=$ENV{CC}")
endif()
if(DEFINED ENV{CXX})
    list(APPEND _wheel_env "CXX=$ENV{CXX}")
endif()

set(_wheel_build_script
    "${CMAKE_CURRENT_BINARY_DIR}/build_torch_nntile_wheel.sh")
set(_repair_linux
    "${PROJECT_SOURCE_DIR}/torch_nntile/tools/repair_wheel_linux.sh")
set(_repair_macos
    "${PROJECT_SOURCE_DIR}/torch_nntile/tools/repair_wheel_macos.sh")
set(_wheel_pkg_dir "${PROJECT_SOURCE_DIR}/torch_nntile")

# Prefer CMAKE_CURRENT_SOURCE_DIR when included from torch_nntile/.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/repair_wheel_linux.sh")
    set(_repair_linux
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/repair_wheel_linux.sh")
    set(_repair_macos
        "${CMAKE_CURRENT_SOURCE_DIR}/tools/repair_wheel_macos.sh")
    set(_wheel_pkg_dir "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

set(_wheel_lib_path_parts "")
if(_wheel_nntile_build_dir)
    list(APPEND _wheel_lib_path_parts
        "${_wheel_nntile_build_dir}/nntile")
endif()
if(_wheel_torch_build_dir)
    list(APPEND _wheel_lib_path_parts
        "${_wheel_torch_build_dir}/torch_nntile")
endif()
if(_wheel_nntile_prefix)
    list(APPEND _wheel_lib_path_parts "${_wheel_nntile_prefix}/lib")
endif()
if(_wheel_torch_prefix AND NOT _wheel_torch_prefix STREQUAL _wheel_nntile_prefix)
    list(APPEND _wheel_lib_path_parts "${_wheel_torch_prefix}/lib")
endif()
string(REPLACE ";" ":" _wheel_lib_path "${_wheel_lib_path_parts}")

file(WRITE "${_wheel_build_script}" "#!/usr/bin/env bash\n")
file(APPEND "${_wheel_build_script}" "set -euo pipefail\n")
foreach(_kv IN LISTS _wheel_env)
    file(APPEND "${_wheel_build_script}" "export ${_kv}\n")
endforeach()
file(APPEND "${_wheel_build_script}"
    "export LD_LIBRARY_PATH=\"${_wheel_lib_path}:\${LD_LIBRARY_PATH:-}\"\n"
    "export DYLD_LIBRARY_PATH=\"${_wheel_lib_path}:\${DYLD_LIBRARY_PATH:-}\"\n"
    "rm -rf \"${_torch_nntile_wheel_raw}\"\n"
    "mkdir -p \"${_torch_nntile_wheel_raw}\" \"${TORCH_NNTILE_WHEELHOUSE}\"\n"
    "\"${Python3_EXECUTABLE}\" -m pip wheel \\\n"
    "  \"${_wheel_pkg_dir}\" \\\n"
    "  -w \"${_torch_nntile_wheel_raw}\" \\\n"
    "  --no-build-isolation --no-deps\n"
    "shopt -s nullglob\n"
    "raw_wheels=(\"${_torch_nntile_wheel_raw}\"/*.whl)\n"
    "if [ \"\${#raw_wheels[@]}\" -eq 0 ]; then\n"
    "  echo \"pip wheel produced no .whl under ${_torch_nntile_wheel_raw}\" >&2\n"
    "  exit 1\n"
    "fi\n"
)

if(TORCH_NNTILE_WHEEL_REPAIR)
    if(APPLE)
        file(APPEND "${_wheel_build_script}"
            "for w in \"\${raw_wheels[@]}\"; do\n"
            "  bash \"${_repair_macos}\" \"\$w\" \"${TORCH_NNTILE_WHEELHOUSE}\" arm64\n"
            "done\n"
        )
    elseif(UNIX)
        file(APPEND "${_wheel_build_script}"
            "for w in \"\${raw_wheels[@]}\"; do\n"
            "  bash \"${_repair_linux}\" \"\$w\" \"${TORCH_NNTILE_WHEELHOUSE}\"\n"
            "done\n"
        )
    else()
        file(APPEND "${_wheel_build_script}"
            "cp -f \"\${raw_wheels[@]}\" \"${TORCH_NNTILE_WHEELHOUSE}/\"\n"
        )
    endif()
else()
    file(APPEND "${_wheel_build_script}"
        "cp -f \"\${raw_wheels[@]}\" \"${TORCH_NNTILE_WHEELHOUSE}/\"\n"
    )
endif()

file(APPEND "${_wheel_build_script}"
    "echo \"torch_nntile wheels in ${TORCH_NNTILE_WHEELHOUSE}:\"\n"
    "ls -la \"${TORCH_NNTILE_WHEELHOUSE}\"/*.whl\n"
)

file(CHMOD "${_wheel_build_script}" PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE)

set(_wheel_depends "")
if(TARGET torch_nntile)
    list(APPEND _wheel_depends torch_nntile)
endif()
if(TARGET nntile)
    list(APPEND _wheel_depends nntile)
endif()

add_custom_target(torch_nntile_wheel ALL
    COMMAND "${_wheel_build_script}"
    DEPENDS ${_wheel_depends}
    WORKING_DIRECTORY "${_wheel_pkg_dir}"
    COMMENT "Building torch_nntile Python wheel"
    VERBATIM
)

include(GNUInstallDirs)
install(
    DIRECTORY "${TORCH_NNTILE_WHEELHOUSE}/"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/torch_nntile/wheels"
    FILES_MATCHING PATTERN "*.whl"
)

message(STATUS
    "torch_nntile wheel target enabled -> ${TORCH_NNTILE_WHEELHOUSE} "
    "(build target: torch_nntile_wheel)")
