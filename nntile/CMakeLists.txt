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

# CUTLASS 2.11 GEMM engines are option-gated (USE_CUTLASS, root
# CMakeLists) and live inside the native chunked-GEMM family's
# translation unit (USE_CHUNKED_GEMM) -- with the family OFF there is
# nothing for them to compile into. ON with the family and CUDA kernels:
# fetch the SHA256-pinned release tarball (header-only: SOURCE_SUBDIR
# names a directory that has no CMakeLists.txt, so
# FetchContent_MakeAvailable populates without add_subdirectory and
# CUTLASS's own CMake never runs -- same idiom as
# external/cudnn_frontend) and put its include/ on the path PRIVATE to
# the kernel object library that compiles the gemm_chunked CUDA TU;
# that TU's own __has_include(<cutlass/...>) then arms the engine
# compilation. OFF, or a configure without the family/CUDA kernels:
# nothing is fetched and GEMM routing keeps cuBLAS (native family, or
# the staged path when USE_CHUNKED_GEMM=OFF). The populate is cached
# under _deps and pre-seedable offline via
# FETCHCONTENT_SOURCE_DIR_CUTLASS=<extracted source dir> (a source-dir
# override bypasses URL_HASH -- the header check below is the sanity
# gate on a seed).
if(USE_CHUNKED_GEMM AND USE_CUTLASS AND NNTILE_USE_CUDA
        AND NOT HAVE_STARPU_SIMGRID)
    include(FetchContent)
    FetchContent_Declare(cutlass
        URL https://github.com/NVIDIA/cutlass/archive/refs/tags/v2.11.0.tar.gz
        URL_HASH SHA256=b4394f1e080b63cfc54163069334096324c1262dfc66e67099880005d51b8af9
        SOURCE_SUBDIR header_only_no_cmakelists)
    FetchContent_MakeAvailable(cutlass)
    if(NOT EXISTS "${cutlass_SOURCE_DIR}/include/cutlass/gemm/device/gemm.h")
        message(FATAL_ERROR "USE_CUTLASS: populated source at "
            "'${cutlass_SOURCE_DIR}' carries no cutlass headers "
            "(bad FETCHCONTENT_SOURCE_DIR_CUTLASS seed or tarball?)")
    endif()
    if(TARGET nntile_objs_kernel)
        target_include_directories(nntile_objs_kernel PRIVATE
            "${cutlass_SOURCE_DIR}/include")
        message(STATUS "libnntile compiles CUTLASS 2.11 GEMM engines "
            "(USE_CUTLASS)")
    endif()
elseif(USE_CUTLASS AND NOT USE_CHUNKED_GEMM)
    message(WARNING "USE_CUTLASS=ON ignored: the native chunked-GEMM "
        "family is not compiled (USE_CHUNKED_GEMM=OFF) and the CUTLASS "
        "engines are instantiated inside that family's translation "
        "unit. Nothing is fetched.")
elseif(USE_CUTLASS)
    message(WARNING "USE_CUTLASS=ON ignored: the CUTLASS GEMM engines "
        "are CUDA-only and this configure compiles no CUDA kernels "
        "(NNTILE_USE_CUDA=${NNTILE_USE_CUDA}, "
        "HAVE_STARPU_SIMGRID=${HAVE_STARPU_SIMGRID}). Building without "
        "CUTLASS (engine stubs keep cuBLAS/native routing).")
endif()

# Classic kernel/op suites always register. When NNTILE_TORCH_NATIVE_OPS
# is ON, the torch_native layer suite is added as well.
if(BUILD_TESTING)
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
