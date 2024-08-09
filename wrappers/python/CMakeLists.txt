# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file wrappers/python/CMakeLists.txt
# Python wrappers for High-level API of NNTile library
#
# @version 1.1.0

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
    ${CMAKE_CURRENT_BINARY_DIR}/tests/*/test_*.py)

# Describe all tests that are not yet implemented
set(TESTS_NOT_IMPLEMENTED
    "nntile_core/test_tensor_add_fiber.py"
    "nntile_core/test_tensor_sum_fiber.py"
    "nntile_core/test_tensor_sum_slice.py"
    "nntile_core/test_tensor_fill.py"
    "nntile_core/test_tensor_gelu_backward.py"
    "nntile_core/test_tensor_gelutanh_backward.py"
    "nntile_core/test_tensor_relu_backward.py"
    "nntile_core/test_tensor_logsumexp.py"
    "nntile_core/test_tensor_norm_slice.py"
    "nntile_core/test_tensor_pow.py"
    "nntile_core/test_tensor_prod_fiber.py"
    "nntile_core/test_tensor_prod_slice.py"
    "nntile_core/test_tensor_sumprod_fiber.py"
    "nntile_core/test_tensor_sumprod_slice.py"
    "nntile_core/test_tensor_subtract_indexed_column.py"
    "nntile_core/test_tensor_total_sum_accum.py"
    "nntile_core/test_tensor_add_slice3.py"
    "nntile_core/test_tensor_hypot.py"
    "nntile_core/test_tensor_prod_fiber3.py"
    "nntile_core/test_tensor_scal.py"
    "layer/test_add.py"
    "layer/test_add_slice.py"
    "layer/test_gpt2mlp.py"
    "model/test_deep_relu.py"
    "model/test_gpt2.py"
    "optimizer/test_adam.py"
    )

foreach(py_test IN LISTS PY_TESTS)
    get_filename_component(py_dir ${py_test} DIRECTORY)
    get_filename_component(py_name ${py_test} NAME_WE)
    string(SUBSTRING ${py_name} 5 -1 py_name)
    string(REPLACE "/" "_" py_name "${py_name}")
    set(test_name wrappers_python_tests_${py_dir}_${py_name})
    message(STATUS "Adding 1 test ${test_name}")
    add_test(NAME ${test_name} COMMAND ${PYTHON_EXECUTABLE} -m pytest -rx
        ${CMAKE_CURRENT_BINARY_DIR}/tests/${py_test})
    if(py_test IN_LIST TESTS_NOT_IMPLEMENTED)
        set_tests_properties(${test_name} PROPERTIES LABELS "NotImplemented")
    endif()
endforeach()

# Install Python module
install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install)")
