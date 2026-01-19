# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/graph/CMakeLists.txt
# CMake configuration for graph tests
#
# @version 1.1.0

# Graph tests
add_executable(test_logical_graph test_logical_graph.cc)
target_link_libraries(test_logical_graph nntile Catch2::Catch2WithMain)

add_executable(test_compiled_graph test_compiled_graph.cc)
target_link_libraries(test_compiled_graph nntile Catch2::Catch2WithMain)

# Add tests to CTest
add_test(NAME LogicalGraphTest COMMAND test_logical_graph)
add_test(NAME CompiledGraphTest COMMAND test_compiled_graph)
