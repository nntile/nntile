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
set(GRAPH_MODEL_DATA_PYTHON "${Python3_EXECUTABLE}")

set(LLAMA_GENERATE_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/llama/generate_test_data.py")

set(TESTS_MODEL
    "llama/llama_config"
    "llama/llama_mlp"
    "llama/llama_attention"
    "llama/llama_rope"
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
        ${CMAKE_CURRENT_SOURCE_DIR}/../nn
    )
    set_target_properties(tests_graph_model_${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/llama)
    # Data-driven tests: everything except llama_config
    if(NOT test_name STREQUAL "llama_config")
        # block name = test_name without "llama_" prefix (e.g. "mlp")
        string(REPLACE "llama_" "" block_name "${test_name}")
        if(block_name STREQUAL "rope")
            set(block_name "attention")
        endif()
        target_compile_definitions(tests_graph_model_${test_name} PRIVATE
            LLAMA_DATA_DIR="${LLAMA_DATA_DIR}"
        )
        if(Python3_Interpreter_FOUND)
            add_test(NAME tests_graph_model_${test_name}_data_setup
                COMMAND "${GRAPH_MODEL_DATA_PYTHON}" "${LLAMA_GENERATE_SCRIPT}"
                    --block "${block_name}" --output "${LLAMA_DATA_DIR}"
                    --seed 42
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            )
            set_tests_properties(tests_graph_model_${test_name}_data_setup
                PROPERTIES
                    FIXTURES_SETUP "tests_graph_model_${test_name}_data"
                    ENVIRONMENT
                        "PYTHONPATH=${CMAKE_BINARY_DIR}/wrappers/python:$ENV{PYTHONPATH}"
            )
            set(required_fixtures "tests_graph_model_${test_name}_data")

            # GQA data fixture (for blocks with attention)
            if(NOT block_name STREQUAL "mlp")
                add_test(NAME tests_graph_model_${test_name}_gqa_data_setup
                    COMMAND "${GRAPH_MODEL_DATA_PYTHON}" "${LLAMA_GENERATE_SCRIPT}"
                        --block "${block_name}_gqa"
                        --output "${LLAMA_DATA_DIR}" --seed 42
                    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                )
                set_tests_properties(
                    tests_graph_model_${test_name}_gqa_data_setup
                    PROPERTIES
                        FIXTURES_SETUP
                            "tests_graph_model_${test_name}_gqa_data"
                        ENVIRONMENT
                            "PYTHONPATH=${CMAKE_BINARY_DIR}/wrappers/python:$ENV{PYTHONPATH}"
                )
                list(APPEND required_fixtures
                    "tests_graph_model_${test_name}_gqa_data")
            endif()

            # Extra LlamaAttention safetensors (no-RoPE / causal / both) for graph tests
            if(test_name STREQUAL "llama_attention")
                add_test(NAME tests_graph_model_llama_attention_variant_data_setup
                    COMMAND "${GRAPH_MODEL_DATA_PYTHON}" "${LLAMA_GENERATE_SCRIPT}"
                        --write-attention-rope-mask-variants
                        --output "${LLAMA_DATA_DIR}" --seed 42
                    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                )
                set_tests_properties(
                    tests_graph_model_llama_attention_variant_data_setup
                    PROPERTIES
                        FIXTURES_SETUP
                            "tests_graph_model_llama_attention_variant_data"
                        ENVIRONMENT
                            "PYTHONPATH=${CMAKE_BINARY_DIR}/wrappers/python:$ENV{PYTHONPATH}"
                )
                list(APPEND required_fixtures
                    "tests_graph_model_llama_attention_variant_data")
            endif()

            set_tests_properties(tests_graph_model_${test_name} PROPERTIES
                FIXTURES_REQUIRED "${required_fixtures}"
            )
        endif()
    endif()
endforeach()

# GPT-2 test data: generated at test time via CTest fixtures (one per block)
set(GPT2_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/gpt2_data")
set(GPT2_GENERATE_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/gpt2/generate_test_data.py")

set(TESTS_MODEL_GPT2
    "gpt2/gpt2_config"
    "gpt2/gpt2_mlp"
    "gpt2/gpt2_attention"
    "gpt2/gpt2_block"
    "gpt2/gpt2_model"
    "gpt2/gpt2_causal"
)

foreach(test_path IN LISTS TESTS_MODEL_GPT2)
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
        ${CMAKE_CURRENT_SOURCE_DIR}/../nn
    )
    set_target_properties(tests_graph_model_${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gpt2)
    if(NOT test_name STREQUAL "gpt2_config")
        string(REPLACE "gpt2_" "" block_name "${test_name}")
        target_compile_definitions(tests_graph_model_${test_name} PRIVATE
            GPT2_DATA_DIR="${GPT2_DATA_DIR}"
        )
        if(Python3_Interpreter_FOUND)
            add_test(NAME tests_graph_model_${test_name}_data_setup
                COMMAND "${GRAPH_MODEL_DATA_PYTHON}" "${GPT2_GENERATE_SCRIPT}"
                    --block "${block_name}" --output "${GPT2_DATA_DIR}"
                    --seed 42
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            )
            set_tests_properties(tests_graph_model_${test_name}_data_setup
                PROPERTIES
                    FIXTURES_SETUP "tests_graph_model_${test_name}_data"
                    ENVIRONMENT
                        "PYTHONPATH=${CMAKE_BINARY_DIR}/wrappers/python:$ENV{PYTHONPATH}"
            )
            set(required_fixtures "tests_graph_model_${test_name}_data")

            if(test_name STREQUAL "gpt2_attention")
                add_test(NAME tests_graph_model_gpt2_attention_causal_data_setup
                    COMMAND "${GRAPH_MODEL_DATA_PYTHON}" "${GPT2_GENERATE_SCRIPT}"
                        --block attention_causal
                        --output "${GPT2_DATA_DIR}" --seed 42
                    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                )
                set_tests_properties(
                    tests_graph_model_gpt2_attention_causal_data_setup
                    PROPERTIES
                        FIXTURES_SETUP
                            "tests_graph_model_gpt2_attention_causal_data"
                        ENVIRONMENT
                            "PYTHONPATH=${CMAKE_BINARY_DIR}/wrappers/python:$ENV{PYTHONPATH}"
                )
                list(APPEND required_fixtures
                    "tests_graph_model_gpt2_attention_causal_data")
            endif()
            set_tests_properties(tests_graph_model_${test_name} PROPERTIES
                FIXTURES_REQUIRED "${required_fixtures}"
            )
        endif()
    endif()
endforeach()
# GPT-Neo test data: generated at test time via CTest fixtures (one per block)
set(GPTNEO_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/gptneo_data")
set(GPTNEO_GENERATE_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/gptneo/generate_test_data.py")

set(TESTS_MODEL_GPTNEO
    "gptneo/gptneo_config"
    "gptneo/gptneo_mlp"
    "gptneo/gptneo_attention"
    "gptneo/gptneo_decoder"
    "gptneo/gptneo_model"
    "gptneo/gptneo_causal"
)

foreach(test_path IN LISTS TESTS_MODEL_GPTNEO)
    get_filename_component(test_name ${test_path} NAME)
    string(REPLACE "gptneo_" "" block_name "${test_name}")
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
        ${CMAKE_CURRENT_SOURCE_DIR}/../nn
    )
    set_target_properties(tests_graph_model_${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gptneo)
    if(NOT test_name STREQUAL "gptneo_config")
        target_compile_definitions(tests_graph_model_${test_name} PRIVATE
            GPTNEO_DATA_DIR="${GPTNEO_DATA_DIR}"
        )
        if(Python3_Interpreter_FOUND)
            add_test(NAME tests_graph_model_${test_name}_data_setup
                COMMAND "${GRAPH_MODEL_DATA_PYTHON}" "${GPTNEO_GENERATE_SCRIPT}"
                    --block "${block_name}" --output "${GPTNEO_DATA_DIR}"
                    --seed 42
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            )
            set_tests_properties(tests_graph_model_${test_name}_data_setup
                PROPERTIES
                    FIXTURES_SETUP "tests_graph_model_${test_name}_data"
                    ENVIRONMENT
                        "PYTHONPATH=${CMAKE_BINARY_DIR}/wrappers/python:$ENV{PYTHONPATH}"
            )
            set(required_fixtures "tests_graph_model_${test_name}_data")

            if(test_name STREQUAL "gptneo_attention")
                add_test(NAME tests_graph_model_gptneo_attention_causal_data_setup
                    COMMAND "${GRAPH_MODEL_DATA_PYTHON}" "${GPTNEO_GENERATE_SCRIPT}"
                        --block attention_causal
                        --output "${GPTNEO_DATA_DIR}" --seed 42
                    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                )
                set_tests_properties(
                    tests_graph_model_gptneo_attention_causal_data_setup
                    PROPERTIES
                        FIXTURES_SETUP
                            "tests_graph_model_gptneo_attention_causal_data"
                        ENVIRONMENT
                            "PYTHONPATH=${CMAKE_BINARY_DIR}/wrappers/python:$ENV{PYTHONPATH}"
                )
                list(APPEND required_fixtures
                    "tests_graph_model_gptneo_attention_causal_data")
                add_test(NAME tests_graph_model_gptneo_attention_local_data_setup
                    COMMAND "${GRAPH_MODEL_DATA_PYTHON}" "${GPTNEO_GENERATE_SCRIPT}"
                        --block attention_local
                        --output "${GPTNEO_DATA_DIR}" --seed 42
                    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                )
                set_tests_properties(
                    tests_graph_model_gptneo_attention_local_data_setup
                    PROPERTIES
                        FIXTURES_SETUP
                            "tests_graph_model_gptneo_attention_local_data"
                        ENVIRONMENT
                            "PYTHONPATH=${CMAKE_BINARY_DIR}/wrappers/python:$ENV{PYTHONPATH}"
                )
                list(APPEND required_fixtures
                    "tests_graph_model_gptneo_attention_local_data")
            endif()
            set_tests_properties(tests_graph_model_${test_name} PROPERTIES
                FIXTURES_REQUIRED "${required_fixtures}"
            )
        endif()
    endif()
endforeach()

# GPT-NeoX test data: generated at test time via CTest fixtures (one per block)
set(GPTNEOX_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/gptneox_data")
set(GPTNEOX_GENERATE_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/gptneox/generate_test_data.py")

set(TESTS_MODEL_GPTNEOX
    "gptneox/gptneox_config"
    "gptneox/gptneox_mlp"
    "gptneox/gptneox_attention"
    "gptneox/gptneox_decoder"
    "gptneox/gptneox_model"
    "gptneox/gptneox_causal"
)

foreach(test_path IN LISTS TESTS_MODEL_GPTNEOX)
    get_filename_component(test_name ${test_path} NAME)
    string(REPLACE "gptneox_" "" block_name "${test_name}")
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
        ${CMAKE_CURRENT_SOURCE_DIR}/../nn
    )
    set_target_properties(tests_graph_model_${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gptneox)
    if(NOT test_name STREQUAL "gptneox_config")
        target_compile_definitions(tests_graph_model_${test_name} PRIVATE
            GPTNEOX_DATA_DIR="${GPTNEOX_DATA_DIR}"
        )
        if(Python3_Interpreter_FOUND)
            add_test(NAME tests_graph_model_${test_name}_data_setup
                COMMAND "${GRAPH_MODEL_DATA_PYTHON}" "${GPTNEOX_GENERATE_SCRIPT}"
                    --block "${block_name}" --output "${GPTNEOX_DATA_DIR}"
                    --seed 42
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            )
            set_tests_properties(tests_graph_model_${test_name}_data_setup
                PROPERTIES
                    FIXTURES_SETUP "tests_graph_model_${test_name}_data"
                    ENVIRONMENT
                        "PYTHONPATH=${CMAKE_BINARY_DIR}/wrappers/python:$ENV{PYTHONPATH}"
            )
            set(required_fixtures "tests_graph_model_${test_name}_data")

            if(test_name STREQUAL "gptneox_attention")
                add_test(NAME tests_graph_model_gptneox_attention_causal_data_setup
                    COMMAND "${GRAPH_MODEL_DATA_PYTHON}" "${GPTNEOX_GENERATE_SCRIPT}"
                        --block attention_causal
                        --output "${GPTNEOX_DATA_DIR}" --seed 42
                    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                )
                set_tests_properties(
                    tests_graph_model_gptneox_attention_causal_data_setup
                    PROPERTIES
                        FIXTURES_SETUP
                            "tests_graph_model_gptneox_attention_causal_data"
                        ENVIRONMENT
                            "PYTHONPATH=${CMAKE_BINARY_DIR}/wrappers/python:$ENV{PYTHONPATH}"
                )
                list(APPEND required_fixtures
                    "tests_graph_model_gptneox_attention_causal_data")
            endif()
            set_tests_properties(tests_graph_model_${test_name} PROPERTIES
                FIXTURES_REQUIRED "${required_fixtures}"
            )
        endif()
    endif()
endforeach()
