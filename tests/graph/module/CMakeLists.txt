# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/graph/module/CMakeLists.txt
# Tests for nntile::graph::module components
#
# @version 1.1.0

# All module tests using Catch2
set(TESTS_CATCH2
    "activation"
    "embedding"
    "gated_mlp"
    "linear"
    "mlp"
    "module"
)

# Module tests with PyTorch comparison
set(TESTS_MODULE_PYTORCH
    "activation"
    "embedding"
    "gated_mlp"
    "linear"
    "mlp"
)

# Setup a target for coverage of module tests
if(BUILD_COVERAGE)
    setup_target_for_coverage_lcov(NAME coverage_module
        EXECUTABLE ctest -R tests_module_
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
endif()

foreach(test IN LISTS TESTS_CATCH2)
    add_test_set(TARGET_NAME tests_module_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_module_${test}
        COV_GLOBAL coverage_module coverage
    )
    target_include_directories(tests_module_${test} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../nn)
    if(NNTILE_HAVE_TORCH AND test IN_LIST TESTS_MODULE_PYTORCH)
        target_compile_definitions(tests_module_${test} PRIVATE NNTILE_HAVE_TORCH)
        target_link_libraries(tests_module_${test} PRIVATE ${TORCH_LIBRARIES})
        target_include_directories(tests_module_${test} PRIVATE
            ${TORCH_INSTALL_PREFIX}/include)
        if(TORCH_CXX_FLAGS)
            separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND
                "${TORCH_CXX_FLAGS}")
            target_compile_options(tests_module_${test} PRIVATE
                ${TORCH_CXX_FLAGS_LIST})
        endif()
    endif()
endforeach()
