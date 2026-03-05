# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/model/CMakeLists.txt
# Tests for nntile::model components
#
# @version 1.1.0

add_subdirectory(llama)

# Setup a target for coverage of model tests
if(BUILD_COVERAGE)
    setup_target_for_coverage_lcov(NAME coverage_model
        EXECUTABLE ctest -R tests_model_
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
endif()
