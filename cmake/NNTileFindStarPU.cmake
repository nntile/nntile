# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file cmake/NNTileFindStarPU.cmake
# Create StarPU::starpu from pkg-config (build tree and installed package).

if(TARGET StarPU::starpu)
    return()
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(StarPU REQUIRED IMPORTED_TARGET starpu-1.4)

add_library(StarPU::starpu INTERFACE IMPORTED)
target_link_libraries(StarPU::starpu INTERFACE PkgConfig::StarPU)
