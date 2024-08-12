# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file PreLoad.cmake
# Cmake script to set generator to Ninja
#
# @version 1.1.0

if (NOT DEFINED ENV{CLION_IDE})
    find_program(NINJA_PATH ninja)
    if (NINJA_PATH)
        set(CMAKE_GENERATOR "Ninja" CACHE INTERNAL "" FORCE)
    endif()
endif()
