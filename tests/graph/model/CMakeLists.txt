# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                 2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file tests/graph/model/CMakeLists.txt
# Tests for nntile::graph::model components
#
# @version 1.1.0

# Llama test data: generated at test time via CTest fixtures (one per block)
set(LLAMA_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/llama_data")
find_package(Python3 COMPONENTS Interpreter QUIET)
# Prefer venv Python (has torch, transformers, safetensors); fallback to system
set(LLAMA_DATA_PYTHON "${Python3_EXECUTABLE}")

set(LLAMA_GENERATE_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/llama/generate_test_data.py")

set(TESTS_MODEL
    "llama/llama_config"
    "llama/llama_mlp"
    "llama/llama_attention"
    "llama/llama_decoder"
    "llama/llama_model"
    "llama/llama_causal"
)

foreach(test_path IN LISTS TESTS_MODEL)
    # Get test name (e.g., "llama_mlp") and create a CTest test
    get_filename_component(test_name ${test_path} NAME)
    add_test_set(TARGET_NAME tests_graph_model_${test_name}
        EXEC_NAME test_${test_name}
        SOURCES ${test_path}.cc
        LINK_LIBRARIES nntile Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_graph_model_${test_name}
        COV_GLOBAL coverage_graph_model coverage
    )
    target_include_directories(tests_graph_model_${test_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../nn
    )
    set_target_properties(tests_graph_model_${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/llama)
    # Data-driven tests: everything except llama_config
    if(NOT test_name STREQUAL "llama_config")
        # block name = test_name without "llama_" prefix (e.g. "mlp")
        string(REPLACE "llama_" "" block_name "${test_name}")
        target_compile_definitions(tests_graph_model_${test_name} PRIVATE
            LLAMA_DATA_DIR="${LLAMA_DATA_DIR}"
        )
        if(Python3_Interpreter_FOUND)
            add_test(NAME tests_graph_model_${test_name}_data_setup
                COMMAND "${LLAMA_DATA_PYTHON}" "${LLAMA_GENERATE_SCRIPT}"
                    --block "${block_name}" --output "${LLAMA_DATA_DIR}"
                    --seed 42
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            )
            set_tests_properties(tests_graph_model_${test_name}_data_setup
                PROPERTIES
                    FIXTURES_SETUP "tests_graph_model_${test_name}_data"
            )
            set(required_fixtures "tests_graph_model_${test_name}_data")

            # GQA data fixture (for blocks with attention)
            if(NOT block_name STREQUAL "mlp")
                add_test(NAME tests_graph_model_${test_name}_gqa_data_setup
                    COMMAND "${LLAMA_DATA_PYTHON}" "${LLAMA_GENERATE_SCRIPT}"
                        --block "${block_name}_gqa"
                        --output "${LLAMA_DATA_DIR}" --seed 42
                    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                )
                set_tests_properties(
                    tests_graph_model_${test_name}_gqa_data_setup
                    PROPERTIES
                        FIXTURES_SETUP
                            "tests_graph_model_${test_name}_gqa_data"
                )
                list(APPEND required_fixtures
                    "tests_graph_model_${test_name}_gqa_data")
            endif()

            set_tests_properties(tests_graph_model_${test_name} PROPERTIES
                FIXTURES_REQUIRED "${required_fixtures}"
            )
        endif()
    endif()
endforeach()
