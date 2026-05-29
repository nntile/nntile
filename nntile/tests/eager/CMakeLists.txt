# @file nntile/tests/eager/CMakeLists.txt
# Core package tests (kernel, starpu, tile, tensor)

if(NOT HAVE_STARPU_SIMGRID)
    add_subdirectory("kernel")
endif()
add_subdirectory("starpu")
add_subdirectory("tile")
add_subdirectory("tensor")
