# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/CMakeLists.txt
# Function to register test sets with CTest
#
# @version 1.1.0

# Extend set of ctest-controlled tests and enable code coverage if required.
# All of the tests shall be launched by the same executable but with different
# arguments. All the arguments are passed by corresponding keyword ARGS. All
# the supported keywords are: TARGET_NAME, EXEC_NAME, SOURCES, LINK_LIBRARIES,
# ARGS, COV_ENABLE, COV_NAME, COV_GLOBAL and MPI_NUMPROC. The following
# example generates a test set of 4 ctest tests:
# add_test_set(TARGET_NAME target_name
#              EXEC_NAME executable_name
#              SOURCES ${executable_sources}
#              LINK_LIBRARIES ${link_librariess}
#              ARGS " " "arg1 arg2 arg3" "arg1 arg2" "arg1"
#              COV_ENABLE ON
#              COV_NAME coverage_name
#              COV_GLOBAL coverage
#              MPI_NUMPROC 4)
# Here, "target_name" is the base name for all generated tests and it shall be
# unique across entire CMake build. Actual target names will be: "target_name"
# in a case of a single test and "target_name_${i}" with integer values ${i} in
# a case of multiple tests. The above mentioned example creates 4 tests:
# target_name_1, target_name_2, target_name_3 and target_name_4. The first
# test executes command "executable_name", the second one runs
# "executable_name arg1 arg2 arg3", and so on. It is then easy to launch all of
# these tests by a single command "ctest -R target_name". User shall be careful
# when choosing "target_name" as it shall be unique across entire CMake build.
# The next argument, namely "EXEC_NAME", simply defines name for the executable
# which is equal to "target_name" in a case "EXEC_NAME" is not provided. The
# "SOURCES" argument contains list of all sources of the test executable. The
# argument "LINK_LIBRARIES" defines a set of libraries to be linked against the
# test executable. The variable "ARGS" is a list of strings, where each string
# contains arguments for a single test. "COV_ENABLE" enables coverage target
# with a name "COV_NAME" which executes "ctest -R target_name" and creates
# coverage report, based on gcov, lcov and genhtml executables. Additionally,
# global coverage target "COV_GLOBAL" (which simply executes "ctest") is
# updated to have executable "executable_name" as a dependency.
function(add_test_set)
    # List of non-value arguments
    set(options)
    # List of single-value arguments
    set(one_value_args TARGET_NAME EXEC_NAME MPI_NUMPROC COV_ENABLE COV_NAME)
    # List of multi-value arguments
    set(multi_value_args SOURCES LINK_LIBRARIES ARGS COV_GLOBAL LABELS)
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
    target_link_libraries(${_args_TARGET_NAME} PRIVATE
        "${_args_LINK_LIBRARIES}")
    # Try to set environment for a test on Apple MacOS
    if(APPLE AND DEFINED ADD_RPATH)
        set_target_properties(${_args_TARGET_NAME} PROPERTIES BUILD_RPATH
            "${ADD_RPATH}")
    endif()
    # Check if coverage needed
    if(_args_COV_ENABLE)
        # Extend dependencies list for global coverage
        foreach(global_arg IN LISTS ${_args_COV_GLOBAL})
            add_dependencies(${global_arg} ${_args_TARGET_NAME})
        endforeach()
        # Add target for local coverage
        setup_target_for_coverage_lcov(NAME ${_args_COV_NAME}
            EXECUTABLE ctest -R ${_args_TARGET_NAME}
            DEPENDENCIES ${_args_TARGET_NAME}
            LCOV_ARGS --no-external
            GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
    endif()
    # Add test suite to the CTest
    list(LENGTH _args_ARGS ntests)
    if(DEFINED _args_MPI_NUMPROC)
        set(exec_cmd ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG}
            ${_args_MPI_NUMPROC} $<TARGET_FILE:${_args_TARGET_NAME}>)
    else()
        set(exec_cmd $<TARGET_FILE:${_args_TARGET_NAME}>)
    endif()
    if(ntests EQUAL 0)
        # Test without arguments
        message(STATUS "Adding 1 test ${_args_TARGET_NAME}")
        add_test(NAME ${_args_TARGET_NAME} COMMAND ${exec_cmd})
        # Add labels to test if they are actually provided
        if(DEFINED _args_LABELS)
            set_tests_properties(${_args_TARGET_NAME} PROPERTIES LABELS
                "${_args_LABELS}")
        endif()
    elseif(ntests EQUAL 1)
        # Test with a single set of arguments
        message(STATUS "Adding 1 test ${_args_TARGET_NAME}")
        # Convert string of arguments to a list
        string(REPLACE " " ";" _args_ARGS "${_args_ARGS}")
        # Test with properly parsed arguments
        add_test(NAME ${_args_TARGET_NAME} COMMAND ${exec_cmd}
            ${_args_ARGS})
        # Add labels to test if they are actually provided
        if(DEFINED _args_LABELS)
            set_tests_properties(${_args_TARGET_NAME} PROPERTIES LABELS
                "${_args_LABELS}")
        endif()
    else()
        # Multiple sets of arguments
        message(STATUS "Adding ${ntests} tests ${_args_TARGET_NAME}")
        foreach(i RANGE 1 ${ntests})
            # Read current set of arguments as a string
            list(POP_FRONT _args_ARGS current_ARGS)
            # Convert string of arguments to a list
            string(REPLACE " " ";" current_ARGS "${current_ARGS}")
            # Test with properly parsed arguments
            add_test(NAME ${_args_TARGET_NAME}_${i} COMMAND ${exec_cmd}
                ${current_ARGS})
            # Add labels to test if they are actually provided
            if(DEFINED _args_LABELS)
                set_tests_properties(${_args_TARGET_NAME}_${i} PROPERTIES
                    LABELS "${_args_LABELS}")
            endif()
        endforeach()
    endif()
endfunction()

add_test_set(TARGET_NAME tests_constants
    EXEC_NAME test_constants
    SOURCES constants.cc
    LINK_LIBRARIES nntile
    COV_ENABLE ${BUILD_COVERAGE}
    COV_NAME coverage_constants
    COV_GLOBAL coverage
    )

# Kernels are not compiled in case of SimGrid mode
if(NOT HAVE_STARPU_SIMGRID)
    add_subdirectory("kernel")
endif()
add_subdirectory("starpu")
add_subdirectory("tile")
add_subdirectory("tensor")
#add_subdirectory("layer")
#add_subdirectory("model")
