# Model block tests use CTest fixtures: *_data_setup generates safetensors,
# then the C++ test runs with FIXTURES_REQUIRED.

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

add_test_set(TARGET_NAME tests_graph_model_safetensors_nntile_layout
    EXEC_NAME test_safetensors_nntile_layout
    SOURCES safetensors_nntile_layout.cc
    LINK_LIBRARIES nntile Catch2::Catch2WithMain
    COV_ENABLE ${BUILD_COVERAGE}
    COV_NAME coverage_graph_model_safetensors_nntile_layout
    COV_GLOBAL coverage_graph_model coverage
)
target_include_directories(tests_graph_model_safetensors_nntile_layout PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# GPT-NeoX test data: generated at test time via CTest fixtures (one per block)
set(GPTNEOX_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/gptneox_data")
set(GPTNEOX_GENERATE_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/gptneox/generate_test_data.py")

set(TESTS_MODEL_GPTNEOX
    "gptneox/gptneox_config"
    "gptneox/gptneox_mlp"
    "gptneox/gptneox_rope"
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
        if(Python3_Interpreter_FOUND AND NOT test_name STREQUAL "gptneox_rope")
            add_test(NAME tests_graph_model_${test_name}_data_setup
                COMMAND "${GRAPH_MODEL_DATA_PYTHON}" "${GPTNEOX_GENERATE_SCRIPT}"
                    --block "${block_name}" --output "${GPTNEOX_DATA_DIR}"
                    --seed 42
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            )
            set_tests_properties(tests_graph_model_${test_name}_data_setup
                PROPERTIES
                    FIXTURES_SETUP "tests_graph_model_${test_name}_data"
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
                )
                list(APPEND required_fixtures
                    "tests_graph_model_gptneox_attention_causal_data")

                add_test(
                    NAME tests_graph_model_gptneox_attention_variant_data_setup
                    COMMAND "${GRAPH_MODEL_DATA_PYTHON}" "${GPTNEOX_GENERATE_SCRIPT}"
                        --write-attention-rope-mask-variants
                        --output "${GPTNEOX_DATA_DIR}" --seed 42
                    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                )
                set_tests_properties(
                    tests_graph_model_gptneox_attention_variant_data_setup
                    PROPERTIES
                        FIXTURES_SETUP
                            "tests_graph_model_gptneox_attention_variant_data"
                )
                list(APPEND required_fixtures
                    "tests_graph_model_gptneox_attention_variant_data")
            endif()
            set_tests_properties(tests_graph_model_${test_name} PROPERTIES
                FIXTURES_REQUIRED "${required_fixtures}"
            )
        endif()
        if(test_name STREQUAL "gptneox_rope" AND Python3_Interpreter_FOUND)
            set_tests_properties(tests_graph_model_${test_name} PROPERTIES
                FIXTURES_REQUIRED "tests_graph_model_gptneox_attention_data"
            )
        endif()
    endif()
endforeach()

# T5 test data: generated at test time via CTest fixtures (one per block)
set(T5_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/t5_data")
set(T5_GENERATE_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/t5/generate_test_data.py")

set(TESTS_MODEL_T5
    "t5/t5_config"
    "t5/t5_ff"
    "t5/t5_attention"
    "t5/t5_cross_attention"
    "t5/t5_encoder_block"
    "t5/t5_decoder_block"
    "t5/t5_model"
    "t5/t5_conditional"
)

foreach(test_path IN LISTS TESTS_MODEL_T5)
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
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/t5)
    if(NOT test_name STREQUAL "t5_config")
        string(REPLACE "t5_" "" block_name "${test_name}")
        target_compile_definitions(tests_graph_model_${test_name} PRIVATE
            T5_DATA_DIR="${T5_DATA_DIR}"
        )
        if(Python3_Interpreter_FOUND)
            add_test(NAME tests_graph_model_${test_name}_data_setup
                COMMAND "${GRAPH_MODEL_DATA_PYTHON}" "${T5_GENERATE_SCRIPT}"
                    --block "${block_name}" --output "${T5_DATA_DIR}"
                    --seed 42
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            )
            set_tests_properties(tests_graph_model_${test_name}_data_setup
                PROPERTIES
                    FIXTURES_SETUP "tests_graph_model_${test_name}_data"
            )
            set(required_fixtures "tests_graph_model_${test_name}_data")

            if(test_name STREQUAL "t5_attention")
                add_test(NAME tests_graph_model_t5_attention_causal_data_setup
                    COMMAND "${GRAPH_MODEL_DATA_PYTHON}" "${T5_GENERATE_SCRIPT}"
                        --block attention_causal
                        --output "${T5_DATA_DIR}" --seed 42
                    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                )
                set_tests_properties(
                    tests_graph_model_t5_attention_causal_data_setup
                    PROPERTIES
                        FIXTURES_SETUP
                            "tests_graph_model_t5_attention_causal_data"
                )
                list(APPEND required_fixtures
                    "tests_graph_model_t5_attention_causal_data")
            endif()
            set_tests_properties(tests_graph_model_${test_name} PROPERTIES
                FIXTURES_REQUIRED "${required_fixtures}"
            )
        endif()
    endif()
endforeach()

# BERT test data
set(BERT_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/bert_data")
set(BERT_GENERATE_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/bert/generate_test_data.py")

set(TESTS_MODEL_BERT
    "bert/bert_config"
    "bert/bert_intermediate"
    "bert/bert_attention"
    "bert/bert_layer"
    "bert/bert_embeddings"
    "bert/bert_model"
    "bert/bert_mlm"
)

foreach(test_path IN LISTS TESTS_MODEL_BERT)
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
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bert)
    if(NOT test_name STREQUAL "bert_config")
        string(REPLACE "bert_" "" block_name "${test_name}")
        target_compile_definitions(tests_graph_model_${test_name} PRIVATE
            BERT_DATA_DIR="${BERT_DATA_DIR}"
        )
        if(Python3_Interpreter_FOUND)
            add_test(NAME tests_graph_model_${test_name}_data_setup
                COMMAND "${GRAPH_MODEL_DATA_PYTHON}" "${BERT_GENERATE_SCRIPT}"
                    --block "${block_name}" --output "${BERT_DATA_DIR}"
                    --seed 42
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            )
            set_tests_properties(tests_graph_model_${test_name}_data_setup
                PROPERTIES
                    FIXTURES_SETUP "tests_graph_model_${test_name}_data"
            )
            set_tests_properties(tests_graph_model_${test_name} PROPERTIES
                FIXTURES_REQUIRED "tests_graph_model_${test_name}_data"
            )
        endif()
    endif()
endforeach()

# RoBERTa test data
set(ROBERTA_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/roberta_data")
set(ROBERTA_GENERATE_SCRIPT
    "${CMAKE_CURRENT_SOURCE_DIR}/roberta/generate_test_data.py")

set(TESTS_MODEL_ROBERTA
    "roberta/roberta_config"
    "roberta/roberta_intermediate"
    "roberta/roberta_attention"
    "roberta/roberta_layer"
    "roberta/roberta_embeddings"
    "roberta/roberta_model"
    "roberta/roberta_mlm"
)

foreach(test_path IN LISTS TESTS_MODEL_ROBERTA)
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
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/roberta)
    if(NOT test_name STREQUAL "roberta_config")
        string(REPLACE "roberta_" "" block_name "${test_name}")
        target_compile_definitions(tests_graph_model_${test_name} PRIVATE
            ROBERTA_DATA_DIR="${ROBERTA_DATA_DIR}"
        )
        if(Python3_Interpreter_FOUND)
            add_test(NAME tests_graph_model_${test_name}_data_setup
                COMMAND "${GRAPH_MODEL_DATA_PYTHON}" "${ROBERTA_GENERATE_SCRIPT}"
                    --block "${block_name}" --output "${ROBERTA_DATA_DIR}"
                    --seed 42
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            )
            set_tests_properties(tests_graph_model_${test_name}_data_setup
                PROPERTIES
                    FIXTURES_SETUP "tests_graph_model_${test_name}_data"
            )
            set_tests_properties(tests_graph_model_${test_name} PROPERTIES
                FIXTURES_REQUIRED "tests_graph_model_${test_name}_data"
            )
        endif()
    endif()
endforeach()
