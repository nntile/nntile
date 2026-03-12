# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/graph/io/CMakeLists.txt
# Tests for nntile::graph::io components
#
# @version 1.1.0

set(TESTS_IO
    "safetensors"
    "module_save_load"
)

if(BUILD_COVERAGE)
    setup_target_for_coverage_lcov(NAME coverage_graph_io
        EXECUTABLE ctest -R tests_graph_io_
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
endif()

foreach(test IN LISTS TESTS_IO)
    add_test_set(TARGET_NAME tests_graph_io_${test}
        EXEC_NAME test_io_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_graph_io_${test}
        COV_GLOBAL coverage_graph_io coverage
    )
    target_include_directories(tests_graph_io_${test} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..)
endforeach()
