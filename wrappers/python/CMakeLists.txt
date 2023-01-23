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
# @date 2023-01-23

# Get pybind11 library and set up binary directory pybind11
add_subdirectory("../../external/pybind11" "pybind11")

# If a single source file will become huge, it will be split into several files
set(EXT_SRC
    "nntile/nntile_core.cc"
    )

# Set up python extension module nntile_core and its properties
pybind11_add_module(nntile_core ${EXT_SRC})
set_target_properties(nntile_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "nntile"
    SUFFIX ".so")
target_link_libraries(nntile_core PRIVATE nntile)

# Collect all *.py source files and copy them into build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/nntile DESTINATION
    ${CMAKE_CURRENT_BINARY_DIR} FILES_MATCHING PATTERN "*.py"
    FOLLOW_SYMLINK_CHAIN)

# Configure installation scripts in build directory
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in"
    "${CMAKE_CURRENT_BINARY_DIR}/setup.py" @ONLY)

install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install)")

