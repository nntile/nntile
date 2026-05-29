# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file nntile/CMakeLists.txt
# Unified nntile library.
#
# @version 1.1.0

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(NNTileSubsystems)
include(NNTileTests)

if(NOT BUILD_NNTILE)
    return()
endif()

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/nntile/defs.h.in"
    "${PROJECT_BINARY_DIR}/include/nntile/defs.h"
    @ONLY
)

add_library(nntile)
if(NNTILE_BUILD_KERNEL)
    target_sources(nntile PRIVATE "${PROJECT_SOURCE_DIR}/external/random.h")
endif()
target_link_libraries(nntile PUBLIC ${StarPU_LDFLAGS} nlohmann_json::nlohmann_json)
nntile_apply_common_includes(nntile)
nntile_apply_cuda(nntile)
nntile_apply_cblas(nntile)
if(HAVE_STARPU_SIMGRID)
    target_compile_definitions(nntile PUBLIC STARPU_SIMGRID)
endif()
add_subdirectory(src)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

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
