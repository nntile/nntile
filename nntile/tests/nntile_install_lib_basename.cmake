# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# Resolve installed nntile library basename for the current
# configure host. Use from CMakeLists.txt (WIN32/APPLE are defined), not from
# cmake -P scripts.

function(nntile_install_lib_basename out_var)
    if(BUILD_SHARED_LIBS)
        if(WIN32)
            set(_name "nntile.dll")
        elseif(APPLE)
            set(_name "libnntile.dylib")
        else()
            set(_name "libnntile.so")
        endif()
    else()
        if(WIN32)
            set(_name "nntile.lib")
        else()
            set(_name "libnntile.a")
        endif()
    endif()
    set(${out_var} "${_name}" PARENT_SCOPE)
endfunction()
