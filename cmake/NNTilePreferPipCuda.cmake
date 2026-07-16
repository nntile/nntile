# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file cmake/NNTilePreferPipCuda.cmake
# Prefer pip nvidia-*-cu12 libraries over system CUDA toolkit copies.
#
# Call after find_package(CUDAToolkit). When NVIDIA_* / CUDNN_* paths are set
# (typically by torch_nntile/tools/setup_torch_cuda_env.sh), point CUDA::cublas
# / CUDA::cublasLt / optionally CUDA::cudart at those trees so libnntile does
# not link a second copy under /usr/local/cuda/lib64.
#
# Enable with -DNNTILE_CUDA_FROM_PIP=ON or env NNTILE_CUDA_FROM_PIP=1.

if(DEFINED ENV{NNTILE_CUDA_FROM_PIP} AND NOT DEFINED NNTILE_CUDA_FROM_PIP)
    if("$ENV{NNTILE_CUDA_FROM_PIP}" STREQUAL "1"
            OR "$ENV{NNTILE_CUDA_FROM_PIP}" STREQUAL "ON")
        set(NNTILE_CUDA_FROM_PIP ON)
    endif()
endif()
option(NNTILE_CUDA_FROM_PIP
    "Link cublas/cudart from pip nvidia-* paths when set" OFF)

macro(_nntile_pip_cuda_path out_var cache_var env_var)
    if(DEFINED ${cache_var} AND NOT "${${cache_var}}" STREQUAL "")
        set(${out_var} "${${cache_var}}")
    elseif(DEFINED ENV{${env_var}} AND NOT "$ENV{${env_var}}" STREQUAL "")
        set(${out_var} "$ENV{${env_var}}")
        set(${cache_var} "${${out_var}}" CACHE PATH
            "pip nvidia path (${env_var})" FORCE)
    else()
        set(${out_var} "")
    endif()
endmacro()

_nntile_pip_cuda_path(_nntile_cublas_lib NVIDIA_CUBLAS_LIBRARY_PATH
    NVIDIA_CUBLAS_LIBRARY_PATH)
_nntile_pip_cuda_path(_nntile_cublas_inc NVIDIA_CUBLAS_INCLUDE_PATH
    NVIDIA_CUBLAS_INCLUDE_PATH)
_nntile_pip_cuda_path(_nntile_cudart_lib NVIDIA_CUDA_RUNTIME_LIBRARY_PATH
    NVIDIA_CUDA_RUNTIME_LIBRARY_PATH)
_nntile_pip_cuda_path(_nntile_cudart_inc NVIDIA_CUDA_RUNTIME_INCLUDE_PATH
    NVIDIA_CUDA_RUNTIME_INCLUDE_PATH)

if(NOT NNTILE_CUDA_FROM_PIP)
    if(_nntile_cublas_lib OR _nntile_cudart_lib)
        set(NNTILE_CUDA_FROM_PIP ON)
        message(STATUS
            "NNTILE_CUDA_FROM_PIP enabled (pip nvidia paths present)")
    else()
        return()
    endif()
endif()

function(nntile_pip_cuda_import name)
    # nntile_pip_cuda_import(<cmake_name> <find_names...> LIB_DIR <dir>
    #     [INCLUDE_DIR <dir>])
    set(options)
    set(one_value LIB_DIR INCLUDE_DIR)
    set(multi_value FIND_NAMES)
    cmake_parse_arguments(ARG "${options}" "${one_value}" "${multi_value}"
        ${ARGN})
    if(NOT ARG_LIB_DIR)
        return()
    endif()
    if(NOT ARG_FIND_NAMES)
        set(ARG_FIND_NAMES "${name}")
    endif()

    find_library(NNTILE_PIP_${name}_LIBRARY
        NAMES ${ARG_FIND_NAMES}
        HINTS "${ARG_LIB_DIR}"
        NO_DEFAULT_PATH
        NO_CACHE)
    if(NOT NNTILE_PIP_${name}_LIBRARY)
        find_library(NNTILE_PIP_${name}_LIBRARY
            NAMES ${ARG_FIND_NAMES}
            HINTS "${ARG_LIB_DIR}"
            PATH_SUFFIXES lib lib64
            NO_CACHE)
    endif()
    if(NOT NNTILE_PIP_${name}_LIBRARY)
        message(WARNING
            "pip CUDA lib '${name}' not found under ${ARG_LIB_DIR}")
        return()
    endif()

    if(NOT TARGET CUDA::${name})
        add_library(CUDA::${name} SHARED IMPORTED)
    endif()
    set_property(TARGET CUDA::${name} PROPERTY
        IMPORTED_LOCATION "${NNTILE_PIP_${name}_LIBRARY}")
    if(ARG_INCLUDE_DIR)
        set_property(TARGET CUDA::${name} APPEND PROPERTY
            INTERFACE_INCLUDE_DIRECTORIES "${ARG_INCLUDE_DIR}")
    endif()
    message(STATUS
        "CUDA::${name} -> ${NNTILE_PIP_${name}_LIBRARY} (pip nvidia)")
endfunction()

if(_nntile_cublas_lib)
    nntile_pip_cuda_import(cublasLt
        FIND_NAMES cublasLt libcublasLt.so.12
        LIB_DIR "${_nntile_cublas_lib}"
        INCLUDE_DIR "${_nntile_cublas_inc}")
    nntile_pip_cuda_import(cublas
        FIND_NAMES cublas libcublas.so.12
        LIB_DIR "${_nntile_cublas_lib}"
        INCLUDE_DIR "${_nntile_cublas_inc}")
    if(TARGET CUDA::cublas AND TARGET CUDA::cublasLt)
        set_property(TARGET CUDA::cublas APPEND PROPERTY
            INTERFACE_LINK_LIBRARIES CUDA::cublasLt)
    endif()
endif()

if(_nntile_cudart_lib)
    nntile_pip_cuda_import(cudart
        FIND_NAMES cudart cudart_static libcudart.so.12
        LIB_DIR "${_nntile_cudart_lib}"
        INCLUDE_DIR "${_nntile_cudart_inc}")
endif()

if(NOT TARGET CUDA::cublas)
    message(FATAL_ERROR
        "NNTILE_CUDA_FROM_PIP=ON but CUDA::cublas could not be created. "
        "Set NVIDIA_CUBLAS_LIBRARY_PATH (see setup_torch_cuda_env.sh).")
endif()
