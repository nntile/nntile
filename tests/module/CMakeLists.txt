# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/module/CMakeLists.txt
# Tests for nntile::module components
#
# @version 1.1.0

# All module tests using Catch2
set(TESTS_CATCH2
    "gelu"
    "module"
    "linear"
    "linear_manual"
    "mlp"
    "mse_loss"
    "sdpa"
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
endforeach()
