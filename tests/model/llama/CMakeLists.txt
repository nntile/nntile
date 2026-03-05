# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/model/llama/CMakeLists.txt
# Tests for nntile::model::llama components
#
# @version 1.1.0

add_test_set(TARGET_NAME tests_model_llama_llama_mlp
    EXEC_NAME test_llama_mlp
    SOURCES llama_mlp.cc
    LINK_LIBRARIES nntile Catch2::Catch2WithMain
    COV_ENABLE ${BUILD_COVERAGE}
    COV_NAME coverage_model_llama_llama_mlp
    COV_GLOBAL coverage_model coverage
)
