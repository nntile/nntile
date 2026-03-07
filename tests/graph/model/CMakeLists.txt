# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                 2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file tests/graph/model/CMakeLists.txt
# Tests for nntile::graph::model components
#
# @version 1.1.0

# Llama test data: generated at test time via CTest fixture (no binary in repo)
set(LLAMA_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/llama_data")
find_package(Python3 COMPONENTS Interpreter QUIET)
# Prefer venv Python (has numpy, safetensors from uv sync); fallback to system
set(LLAMA_DATA_PYTHON "${Python3_EXECUTABLE}")
if(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    set(LLAMA_DATA_PYTHON "${CMAKE_SOURCE_DIR}/.venv/bin/python")
endif()
if(Python3_Interpreter_FOUND)
    add_test(NAME llama_data_setup
        COMMAND "${LLAMA_DATA_PYTHON}"
            "${CMAKE_SOURCE_DIR}/scripts/generate_llama_test_data.py"
            --output "${LLAMA_DATA_DIR}" --seed 42 --no-torch
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
    set_tests_properties(llama_data_setup PROPERTIES
        FIXTURES_SETUP "llama_data"
    )
endif()

set(TESTS_MODEL
    "llama/llama_config"
    "llama/llama_mlp"
    "llama/llama_attention"
    "llama/llama_decoder"
    "llama/llama_model"
    "llama/llama_causal"
)

foreach(test_path IN LISTS TESTS_MODEL)
    get_filename_component(test_name ${test_path} NAME)
    add_test_set(TARGET_NAME tests_model_${test_name}
        EXEC_NAME test_${test_name}
        SOURCES ${test_path}.cc
        LINK_LIBRARIES nntile Catch2::Catch2WithMain
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_model_${test_name}
        COV_GLOBAL coverage_model coverage
    )
    target_include_directories(tests_model_${test_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../nn
    )
    set_target_properties(tests_model_${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/llama)
    if(test_name STREQUAL "llama_attention")
        target_compile_definitions(tests_model_${test_name} PRIVATE
            LLAMA_DATA_DIR="${LLAMA_DATA_DIR}"
        )
        if(Python3_Interpreter_FOUND)
            set_tests_properties(tests_model_${test_name} PROPERTIES
                FIXTURES_REQUIRED "llama_data"
            )
        endif()
    endif()
endforeach()
