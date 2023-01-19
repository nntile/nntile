# @copyright (c) 2022-2022 Skolkovo Institute of Science and Technology
#                          (Skoltech). All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/CMakeLists.txt
# Python wrappers for High-level API of NNTile library
#
# @version 1.0.0
# @author Aleksandr Mikhalev
# @date 2023-01-19

# Get pybind11 library
add_subdirectory("../../external/pybind11" "pybind11")

#pybind11_add_module(starpu "starpu.cc")
#target_link_libraries(starpu PRIVATE nntile)

# Define module tile
set(TILE_SRC
    "tile.cc"
    "tile/traits.cc"
    "tile/tile.cc"
    )
pybind11_add_module(tile ${TILE_SRC})
target_link_libraries(tile PRIVATE nntile)
target_include_directories(tile PRIVATE "${CMAKE_CURRENT_DIR}")


set(TENSOR_SRC
    "tensor.cc"
    "tensor/traits.cc"
    "tensor/distributions.cc"
    "tensor/tensor.cc"
    )
pybind11_add_module(tensor ${TENSOR_SRC})
target_link_libraries(tensor PRIVATE nntile)
target_include_directories(tensor PRIVATE "${CMAKE_CURRENT_DIR}")

#pybind11_add_module(layer "layer/fully_connected.cc")
#target_link_libraries(layer PRIVATE nntile)

