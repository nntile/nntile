# Set list of examples
set(EXAMPLES
    "tile.cc"
    "tensor.cc"
    )

add_executable(examples_tile "tile.cc")
add_executable(examples_tensor "tensor.cc")
target_link_libraries(examples_tile nntile)
target_link_libraries(examples_tensor nntile)

