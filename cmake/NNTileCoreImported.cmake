# Import a pre-built nntile_core from an install prefix (CI graph-only builds).

function(nntile_import_installed_core prefix)
    if(NOT IS_ABSOLUTE "${prefix}")
        get_filename_component(prefix "${prefix}" ABSOLUTE BASE_DIR
            "${CMAKE_CURRENT_BINARY_DIR}")
    endif()

    set(_lib "${prefix}/${CMAKE_INSTALL_LIBDIR}/libnntile_core.so")
    if(NOT EXISTS "${_lib}")
        set(_lib "${prefix}/lib/libnntile_core.so")
    endif()
    if(NOT EXISTS "${_lib}")
        message(FATAL_ERROR "libnntile_core.so not found under ${prefix}")
    endif()

    set(_inc "${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
    if(NOT IS_DIRECTORY "${_inc}")
        set(_inc "${prefix}/include")
    endif()

    if(NOT TARGET nntile_core)
        add_library(nntile_core SHARED IMPORTED GLOBAL)
    endif()
    set_target_properties(nntile_core PROPERTIES
        IMPORTED_LOCATION "${_lib}"
        INTERFACE_INCLUDE_DIRECTORIES "${_inc}"
    )
    target_link_libraries(nntile_core INTERFACE
        ${StarPU_LDFLAGS}
        nlohmann_json::nlohmann_json
    )
endfunction()
