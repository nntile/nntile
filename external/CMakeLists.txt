# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file external/CMakeLists.txt
# External header libraires to be used
#
# @version 1.1.0

include(FetchContent)

#add_subdirectory(googletest)
#add_subdirectory(nvbench)
add_subdirectory(pybind11)

#FetchContent_MakeAvailable(googletest nvbench pybind11)
FetchContent_MakeAvailable(pybind11)

#include(GoogleTest)

# Append external sources and headers.
target_sources(nntile PRIVATE "random.h")
