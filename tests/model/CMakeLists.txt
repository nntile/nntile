# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                 2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# @file tests/model/CMakeLists.txt
# Tests for nntile::model components
#
# @version 1.1.0

# Generate Llama test data (safetensors) for load/save roundtrip tests
set(LLAMA_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama/data")
file(MAKE_DIRECTORY "${LLAMA_DATA_DIR}")
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/generate_llama_test_data.py"
            --output "${LLAMA_DATA_DIR}" --seed 42 --no-torch
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        RESULT_VARIABLE GEN_RESULT
        OUTPUT_QUIET ERROR_QUIET
    )
    if(GEN_RESULT EQUAL 0)
        message(STATUS "Generated Llama test data in ${LLAMA_DATA_DIR}")
    endif()
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
    set_target_properties(tests_model_${test_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/llama)
    target_include_directories(tests_model_${test_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../graph
        ${CMAKE_CURRENT_SOURCE_DIR}/../graph/nn
    )
    if(test_name STREQUAL "llama_attention")
        target_compile_definitions(tests_model_${test_name} PRIVATE
            LLAMA_DATA_DIR="${LLAMA_DATA_DIR}"
        )
    endif()
endforeach()
