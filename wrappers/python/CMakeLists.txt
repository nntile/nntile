# Get pybind11 library
add_subdirectory("../../external/pybind11" "pybind11")

pybind11_add_module(starpu "starpu.cc")
target_link_libraries(starpu PRIVATE nntile)

pybind11_add_module(tensor "tensor.cc")
target_link_libraries(tensor PRIVATE nntile)

pybind11_add_module(layer "layer/fully_connected.cc")
target_link_libraries(layer PRIVATE nntile)

