# Import a pre-built nntile from an install prefix or a CI build tree.

include(GNUInstallDirs)

# Link tests against libnntile.so produced by build-nntile (no per-subsystem .a).
function(nntile_import_prebuilt_library lib_path)
    if(NOT IS_ABSOLUTE "${lib_path}")
        get_filename_component(_lib "${lib_path}" ABSOLUTE BASE_DIR
            "${CMAKE_CURRENT_BINARY_DIR}")
    else()
        set(_lib "${lib_path}")
    endif()
    if(NOT EXISTS "${_lib}")
        message(FATAL_ERROR "NNTILE_PREBUILT_LIBRARY not found: ${_lib}")
    endif()

    if(NOT TARGET nntile)
        add_library(nntile SHARED IMPORTED GLOBAL)
    endif()
    set_target_properties(nntile PROPERTIES IMPORTED_LOCATION "${_lib}")
    target_include_directories(nntile INTERFACE
        "${PROJECT_SOURCE_DIR}/nntile/include"
        "${PROJECT_BINARY_DIR}/include"
        ${StarPU_INCLUDE_DIRS}
    )
    target_link_libraries(nntile INTERFACE
        ${StarPU_LDFLAGS}
        nlohmann_json::nlohmann_json
    )
    if(HAVE_STARPU_SIMGRID)
        target_compile_definitions(nntile INTERFACE STARPU_SIMGRID)
    endif()
    if(NNTILE_LINK_CUDA)
        target_link_libraries(nntile INTERFACE CUDA::cublas CUDNN::cudnn_all)
        target_include_directories(nntile INTERFACE
            ${cudnn_frontend_SOURCE_DIR}/include)
    endif()
    if(NNTILE_LINK_CBLAS)
        target_link_libraries(nntile INTERFACE BLAS::BLAS)
    endif()
endfunction()

function(nntile_import_installed_core prefix)
    if(NOT IS_ABSOLUTE "${prefix}")
        get_filename_component(prefix "${prefix}" ABSOLUTE BASE_DIR
            "${CMAKE_CURRENT_BINARY_DIR}")
    endif()

    set(_lib "${prefix}/${CMAKE_INSTALL_LIBDIR}/libnntile.so")
    if(NOT EXISTS "${_lib}")
        set(_lib "${prefix}/lib/libnntile.so")
    endif()
    if(NOT EXISTS "${_lib}")
        message(FATAL_ERROR "libnntile.so not found under ${prefix}")
    endif()

    set(_inc "${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
    if(NOT IS_DIRECTORY "${_inc}")
        set(_inc "${prefix}/include")
    endif()

    if(NOT TARGET nntile)
        add_library(nntile SHARED IMPORTED GLOBAL)
    endif()
    set_target_properties(nntile PROPERTIES
        IMPORTED_LOCATION "${_lib}"
        INTERFACE_INCLUDE_DIRECTORIES "${_inc}"
    )
    target_link_libraries(nntile INTERFACE
        ${StarPU_LDFLAGS}
        nlohmann_json::nlohmann_json
    )
endfunction()

# Read installed defs.h from a core prefix (graph-only builds).
function(nntile_installed_core_defs_has prefix symbol out_var)
    set(_inc "${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
    if(NOT IS_DIRECTORY "${_inc}")
        set(_inc "${prefix}/include")
    endif()
    set(_defs "${_inc}/nntile/defs.h")
    if(NOT EXISTS "${_defs}")
        set(${out_var} FALSE PARENT_SCOPE)
        return()
    endif()
    file(READ "${_defs}" _content)
    if(_content MATCHES "#define ${symbol}([^a-zA-Z0-9_]|$)")
        set(${out_var} TRUE PARENT_SCOPE)
    else()
        set(${out_var} FALSE PARENT_SCOPE)
    endif()
endfunction()
