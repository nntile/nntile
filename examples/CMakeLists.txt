# Set list of examples
set(EXAMPLES
    "tile.cc"
    )

add_executable(examples_gemm "tile.cc")
target_link_libraries(examples_gemm nntile)

