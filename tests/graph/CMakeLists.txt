# Graph tests
add_executable(test_logical_graph test_logical_graph.cc)
target_link_libraries(test_logical_graph nntile gtest_main)

add_executable(test_compiled_graph test_compiled_graph.cc)
target_link_libraries(test_compiled_graph nntile gtest_main)

# Add tests to CTest
add_test(NAME LogicalGraphTest COMMAND test_logical_graph)
add_test(NAME CompiledGraphTest COMMAND test_compiled_graph)