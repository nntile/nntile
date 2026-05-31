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
    "${CMAKE_CURRENT_SOURCE_DIR}/nntile/_bindings"
    "${PROJECT_SOURCE_DIR}/nntile/examples")
set_target_properties(nntile_py PROPERTIES
    INSTALL_RPATH "$ORIGIN/../../nntile:$ORIGIN/../../../nntile"
    BUILD_WITH_INSTALL_RPATH TRUE)

include(GNUInstallDirs)
install(TARGETS nntile_py
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/nntile
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR}/nntile)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/nntile"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}
    FILES_MATCHING PATTERN "*.py"
    PATTERN "_bindings" EXCLUDE)

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

find_package(Python3 COMPONENTS Interpreter REQUIRED)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    file(GLOB_RECURSE NNTILE_PY_TESTS
        LIST_DIRECTORIES false
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.py")
    foreach(py_test IN LISTS NNTILE_PY_TESTS)
        get_filename_component(py_test_name "${py_test}" NAME_WE)
        get_filename_component(py_test_dir "${py_test}" DIRECTORY)
        file(RELATIVE_PATH py_test_rel "${CMAKE_CURRENT_SOURCE_DIR}/tests"
            "${py_test_dir}")
        if(py_test_rel STREQUAL "." OR py_test_rel STREQUAL "")
            set(ctest_suffix "${py_test_name}")
        else()
            string(REPLACE "/" "_" py_test_rel_underscore "${py_test_rel}")
            set(ctest_suffix "${py_test_rel_underscore}_${py_test_name}")
        endif()
        add_test(
            NAME "python_tests_${ctest_suffix}"
            COMMAND "${Python3_EXECUTABLE}" -m pytest -rx "${py_test}"
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tests")
        # Prepend libnntile and StarPU to inherited LD_LIBRARY_PATH (do not replace).
        set(_nntile_py_ld_prepend "${CMAKE_BINARY_DIR}/nntile")
        foreach(_starpu_lib_dir IN LISTS StarPU_LIBRARY_DIRS)
            set(_nntile_py_ld_prepend
                "${_nntile_py_ld_prepend}:${_starpu_lib_dir}")
        endforeach()
        set_tests_properties("python_tests_${ctest_suffix}" PROPERTIES
            ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
            ENVIRONMENT_MODIFICATION
            "LD_LIBRARY_PATH=path_list_prepend:${_nntile_py_ld_prepend}"
            LABELS "python")
    endforeach()
endif()
