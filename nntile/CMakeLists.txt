# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file nntile/CMakeLists.txt
# CMakeLists.txt
#
# @version 1.1.0
# @file nntile/CMakeLists.txt
# Unified nntile library.

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(NNTileSubsystems)
include(NNTileTests)

if(NNTILE_PREBUILT_LIBRARY AND NNTILE_LINK_CACHED_OBJECTS)
    message(FATAL_ERROR
        "Use either NNTILE_PREBUILT_LIBRARY or NNTILE_LINK_CACHED_OBJECTS, not both")
endif()

# Test compile-check: compile test .cc only; no libnntile, no CTest run.
if(NNTILE_COMPILE_CHECK_TESTS_SUBSYSTEM)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/include/nntile/defs.h.in"
        "${PROJECT_BINARY_DIR}/include/nntile/defs.h"
        @ONLY
    )
    add_subdirectory(tests)
    return()
endif()

if(NOT BUILD_NNTILE)
    return()
endif()

set(_nntile_defs "${PROJECT_BINARY_DIR}/include/nntile/defs.h")
if(NNTILE_PREBUILT_LIBRARY AND EXISTS "${_nntile_defs}")
    list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
    include(NNTileCoreImported)
    nntile_apply_link_options_from_defs("${_nntile_defs}")
else()
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/include/nntile/defs.h.in"
        "${_nntile_defs}"
        @ONLY
    )
endif()

if(NNTILE_PREBUILT_LIBRARY)
    if(NOT COMMAND nntile_import_prebuilt_library)
        list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
        include(NNTileCoreImported)
    endif()
    nntile_import_prebuilt_library("${NNTILE_PREBUILT_LIBRARY}")
elseif(NOT TARGET nntile AND NOT NNTILE_COMPILE_CHECK_SUBSYSTEM)
    add_library(nntile)
    target_link_libraries(nntile PUBLIC ${StarPU_LDFLAGS} nlohmann_json::nlohmann_json)
    nntile_apply_common_includes(nntile)
    nntile_apply_cuda(nntile)
    nntile_apply_cblas(nntile)
    if(HAVE_STARPU_SIMGRID)
        target_compile_definitions(nntile PUBLIC STARPU_SIMGRID)
    endif()
    add_subdirectory(src)
else()
    add_subdirectory(src)
endif()

if(BUILD_TESTS AND NOT NNTILE_COMPILE_CHECK_SUBSYSTEM)
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES AND NOT NNTILE_COMPILE_CHECK_SUBSYSTEM)
    add_subdirectory(examples)
endif()

if(NOT NNTILE_COMPILE_CHECK_SUBSYSTEM AND NOT NNTILE_PREBUILT_LIBRARY)
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
