# Import a pre-built nntile from an install prefix (CI graph-only builds).

include(GNUInstallDirs)

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
