# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file examples/CMakeLists.txt
# Add examples to the CMake
#
# @version 1.1.0

# Add an example, which is NOT controlled by the ctest. Example executable
# shall not take any arguments, as all of the inputs shall be within the
# executable. The following sample code generates an example:
# add_example(TARGET_NAME target_name
#             EXEC_NAME executable_name
#             SOURCES ${executable_sources}
#             LINK_LIBRARIES ${link_librariess})
# Here, "target_name" is the name of example within CMake build system which
# shall be unique. The next argument "EXEC_NAME" simply defines name for the
# executable which is equal to target_name in a case "EXEC_NAME" is not
# provided. "SOURCES" argument contains list of all sources of the executable.
# Finally, "LINK_LIBRARIES" defines a set of libraries to be linked against the
# example executable.
function(add_example)
    # List of non-value arguments
    set(options)
    # List of single-value arguments
    set(one_value_args TARGET_NAME EXEC_NAME)
    # List of multi-value arguments
    set(multi_value_args SOURCES LINK_LIBRARIES)
    # Parse arguments
    cmake_parse_arguments(_args "${options}" "${one_value_args}"
        "${multi_value_args}" ${ARGN})
    # Check if the function arguments were properly defined
    if(DEFINED _args_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "add_test_set() had the following unpared \
        arguments: ${_args_UNPARSED_ARGUMENTS}")
    endif()
    # Register executable with its unique target name
    add_executable(${_args_TARGET_NAME} ${_args_SOURCES})
    # Set name of executable file
    if(_args_EXEC_NAME)
        set_target_properties(${_args_TARGET_NAME} PROPERTIES OUTPUT_NAME
            ${_args_EXEC_NAME})
    endif()
    # Link to provided libraries
    target_link_libraries(${_args_TARGET_NAME} PRIVATE ${_args_LINK_LIBRARIES})
    # Try to set environment for a test on Apple MacOS
    if(APPLE AND DEFINED ADD_RPATH)
        set_target_properties(${_args_TARGET_NAME} PROPERTIES BUILD_RPATH
            "${ADD_RPATH}")
    endif()
    message(STATUS "Adding example ${_args_TARGET_NAME}")
endfunction()

add_example(TARGET_NAME examples_autograd_add
    EXEC_NAME "autograd_add_example"
    SOURCES "autograd_add_example.cc"
    LINK_LIBRARIES nntile)

add_example(TARGET_NAME examples_graph_mlp
    EXEC_NAME "graph_mlp_example"
    SOURCES "graph_mlp_example.cc"
    LINK_LIBRARIES nntile)

add_example(TARGET_NAME examples_linear_layer
    EXEC_NAME "linear_layer_example"
    SOURCES "linear_layer_example.cc"
    LINK_LIBRARIES nntile)

