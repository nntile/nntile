# Graph library sources
set(GRAPH_SOURCES
    tensor_spec.cc
    tensor_node.cc
    op_node.cc
    logical_graph.cc
    compiled_graph.cc
)

# Create library
add_library(nntile_graph ${GRAPH_SOURCES})

target_include_directories(nntile_graph PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(nntile_graph PUBLIC
    nntile_tensor
)