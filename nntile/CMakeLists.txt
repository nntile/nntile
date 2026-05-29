# @file nntile/CMakeLists.txt
# Unified nntile library (Plan B monorepo layout).

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(NNTileSubsystems)

if(NOT BUILD_NNTILE)
    return()
endif()

if(NOT TARGET nntile)
    add_library(nntile)
    target_link_libraries(nntile PUBLIC ${StarPU_LDFLAGS} nlohmann_json::nlohmann_json)
    nntile_apply_common_includes(nntile)
    nntile_apply_cuda(nntile)
    nntile_apply_cblas(nntile)
    if(HAVE_STARPU_SIMGRID)
        target_compile_definitions(nntile PUBLIC STARPU_SIMGRID)
    endif()
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/nntile/defs.h.in"
    "${PROJECT_BINARY_DIR}/include/nntile/defs.h"
    @ONLY
)

add_subdirectory(src)

if(BUILD_TESTS AND NOT NNTILE_COMPILE_CHECK_SUBSYSTEM)
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES AND NOT NNTILE_COMPILE_CHECK_SUBSYSTEM)
    add_subdirectory(examples)
endif()

if(NOT NNTILE_COMPILE_CHECK_SUBSYSTEM)
    include(GNUInstallDirs)
    install(
        TARGETS nntile
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/nntile"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nntile
        FILES_MATCHING PATTERN "*.hh"
    )
    install(
        FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/nntile.hh"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(
        FILES "${PROJECT_BINARY_DIR}/include/nntile/defs.h"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/nntile
    )
endif()
