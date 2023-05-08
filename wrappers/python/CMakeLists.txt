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
# @date 2023-02-15

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

# Copy tests into build directory also to avoid __pycache__ in sources
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tests DESTINATION
    ${CMAKE_CURRENT_BINARY_DIR}
    FOLLOW_SYMLINK_CHAIN)

# Copy examples into build directory also to avoid __pycache__ in sources
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/examples DESTINATION
    ${CMAKE_CURRENT_BINARY_DIR}
    FOLLOW_SYMLINK_CHAIN)

# Collect all the tests and add them to ctest
file(GLOB_RECURSE PY_TESTS FOLLOW_SYMLINKS LIST_DIRECTORIES FALSE
    RELATIVE ${CMAKE_CURRENT_BINARY_DIR}/tests
    ${CMAKE_CURRENT_BINARY_DIR}/tests/test_*.py)
foreach(py_test ${PY_TESTS})
    get_filename_component(py_dir ${py_test} DIRECTORY)
    get_filename_component(py_name ${py_test} NAME_WE)
    string(SUBSTRING ${py_name} 5 -1 py_name)
    string(REPLACE "/" "_" py_name "${py_name}")
    set(test_name wrappers_python_tests_${py_dir}_${py_name})
    message(STATUS "Adding 1 test ${test_name}")
    add_test(NAME ${test_name} COMMAND ${PYTHON_EXECUTABLE} -m pytest -rx
        ${CMAKE_CURRENT_BINARY_DIR}/tests/${py_test})
endforeach()

# Install Python module
install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install)")

