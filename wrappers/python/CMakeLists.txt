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
# @date 2022-08-09

# Get pybind11 library
add_subdirectory("../../external/pybind11" "pybind11")

pybind11_add_module(starpu "starpu.cc")
target_link_libraries(starpu PRIVATE nntile)

pybind11_add_module(tensor "tensor.cc")
target_link_libraries(tensor PRIVATE nntile)

pybind11_add_module(layer "layer/fully_connected.cc")
target_link_libraries(layer PRIVATE nntile)

