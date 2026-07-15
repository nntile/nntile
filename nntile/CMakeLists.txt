# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file nntile/CMakeLists.txt
# libnntile_tensorgraph (kernel..tensor) and optional libnntile (NNGraph+).
#
# @version 1.1.0

if(NOT BUILD_NNTILE)
    return()
endif()

include(GNUInstallDirs)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/nntile/defs.h.in"
    "${PROJECT_BINARY_DIR}/include/nntile/defs.h"
    @ONLY
)

# TensorGraph stack: kernel -> starpu -> core -> tile -> tensor -> runtime.
add_library(nntile_tensorgraph)
target_sources(nntile_tensorgraph PRIVATE
    "${PROJECT_SOURCE_DIR}/external/random.h")
target_link_libraries(nntile_tensorgraph PUBLIC
    ${StarPU_LDFLAGS}
    nlohmann_json::nlohmann_json)
nntile_apply_common_includes(nntile_tensorgraph)
nntile_apply_cuda(nntile_tensorgraph)
nntile_apply_cblas(nntile_tensorgraph)
if(HAVE_STARPU_SIMGRID)
    target_compile_definitions(nntile_tensorgraph PUBLIC STARPU_SIMGRID)
endif()

# High-level stack: NNGraph, modules, models, optim, io, dataset.
set(NNTILE_BUILD_HIGH_LEVEL OFF)
if(BUILD_NNTILE_NNGRAPH
        OR BUILD_NNTILE_MODULES
        OR BUILD_NNTILE_MODELS
        OR BUILD_NNTILE_OPTIM
        OR BUILD_NNTILE_IO
        OR BUILD_NNTILE_DATASET)
    set(NNTILE_BUILD_HIGH_LEVEL ON)
endif()

if(NNTILE_BUILD_HIGH_LEVEL)
    add_library(nntile)
    target_link_libraries(nntile PUBLIC nntile_tensorgraph)
    nntile_apply_common_includes(nntile)
    nntile_apply_cuda(nntile)
    nntile_apply_cblas(nntile)
    if(HAVE_STARPU_SIMGRID)
        target_compile_definitions(nntile PUBLIC STARPU_SIMGRID)
    endif()
endif()

add_subdirectory(src)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

set(_nntile_install_targets nntile_tensorgraph)
if(NNTILE_BUILD_HIGH_LEVEL)
    list(APPEND _nntile_install_targets nntile)
endif()
install(
    TARGETS ${_nntile_install_targets}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
# Trailing slash installs directory *contents* into include/nntile/.
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/nntile/"
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
