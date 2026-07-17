# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file nntile/CMakeLists.txt
# libnntile: kernel -> starpu -> core -> TileGraph -> TensorGraph -> Runtime.
#
# @version 1.1.0

if(NOT BUILD_LIBNNTILE)
    return()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/nntile/defs.h.in"
    "${PROJECT_BINARY_DIR}/include/nntile/defs.h"
    @ONLY
)

# TensorGraph stack: kernel -> starpu -> core -> tile -> tensor -> runtime.
add_library(nntile)
add_library(nntile::nntile ALIAS nntile)
target_sources(nntile PRIVATE
    "${PROJECT_SOURCE_DIR}/external/random.h")
# nlohmann_json is built in-tree (FetchContent) but not in this export set;
# BUILD/INSTALL_INTERFACE keeps install(EXPORT) valid while Config.cmake
# find_dependency(nlohmann_json) recreates the imported target for consumers.
target_link_libraries(nntile PUBLIC
    "$<BUILD_INTERFACE:nlohmann_json::nlohmann_json>"
    "$<INSTALL_INTERFACE:nlohmann_json::nlohmann_json>"
)
# INSTALL_INTERFACE include root (nntile.hh + nntile/).
target_include_directories(nntile PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
nntile_apply_common_includes(nntile)
nntile_apply_cuda(nntile)
nntile_apply_cblas(nntile)
if(HAVE_STARPU_SIMGRID)
    target_compile_definitions(nntile PUBLIC STARPU_SIMGRID)
endif()
if(NNTILE_TORCH_NATIVE_OPS)
    if(NOT Torch_FOUND)
        find_package(Torch REQUIRED)
    endif()
    target_link_libraries(nntile PUBLIC ${TORCH_LIBRARIES})
    # defs.h already #cmakedefine's this; avoid -D redefinition warnings.
    message(STATUS "libnntile links LibTorch (NNTILE_TORCH_NATIVE_OPS)")
endif()

add_subdirectory(src)

# Classic layer tests expect the full kernel/op set.
if(BUILD_TESTING AND NOT NNTILE_TORCH_NATIVE_OPS)
    add_subdirectory(tests)
endif()

set(_nntile_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/nntile")

install(
    TARGETS nntile
    EXPORT nntileTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
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

install(
    EXPORT nntileTargets
    FILE nntileTargets.cmake
    NAMESPACE nntile::
    DESTINATION ${_nntile_cmake_dir}
)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/nntileConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/nntileConfig.cmake"
    INSTALL_DESTINATION ${_nntile_cmake_dir}
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/nntileConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/nntileConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/nntileConfigVersion.cmake"
        "${PROJECT_SOURCE_DIR}/cmake/NNTileFindStarPU.cmake"
    DESTINATION ${_nntile_cmake_dir}
)
