# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file python/CMakeLists.txt
# CMakeLists.txt
#
# @version 1.1.0
# @file python/CMakeLists.txt
# Graph-first Python bindings (nntile_py -> nntile.so inside package).

if(NOT BUILD_PYTHON_WRAPPERS OR NOT BUILD_NNTILE)
    return()
endif()

add_subdirectory("${PROJECT_SOURCE_DIR}/external/pybind11" "pybind11")

set(PY_EXT_SRC
    "nntile/_bindings/nntile.cc"
    "nntile/_bindings/nntile_models.cc"
)

pybind11_add_module(nntile_py ${PY_EXT_SRC})
set_target_properties(nntile_py PROPERTIES
    OUTPUT_NAME "nntile"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/nntile"
    SUFFIX ".so")
target_link_libraries(nntile_py PRIVATE nntile)
target_include_directories(nntile_py PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/nntile/_bindings")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/nntile DESTINATION
    ${CMAKE_CURRENT_BINARY_DIR}
    FILES_MATCHING PATTERN "*.py"
    FOLLOW_SYMLINK_CHAIN)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in"
    "${CMAKE_CURRENT_BINARY_DIR}/setup.py" @ONLY)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/tests DESTINATION
        ${CMAKE_CURRENT_BINARY_DIR} FOLLOW_SYMLINK_CHAIN)
endif()
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/examples DESTINATION
    ${CMAKE_CURRENT_BINARY_DIR} FOLLOW_SYMLINK_CHAIN)
