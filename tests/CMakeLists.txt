# Extend set of ctest-controlled tests and enable code coverage. All of the
# tests shall be launched by the same executable with different arguments. All
# the arguments are passed by corresponding keyword. The keywords are:
# TEST_NAME, EXEC_NAME, SOURCES, LINK_LIBRARIES, ARGS, COV_ENABLE, COV_NAME and
# COV_GLOBAL. The following example generates a test set:
# add_test_set(TEST_NAME test_name
#              EXEC_NAME executable_name
#              SOURCES ${executable_sources}
#              LINK_LIBRARIES ${link_librariess}
#              ARGS " " "arg1 arg2 arg3" "arg1 arg2" "arg1"
#              COV_ENABLE ON
#              COV_NAME coverage_name
#              COV_GLOBAL coverage)
# Here, test_name is the base name of generated tests. Actual target names
# will be test_name in a case of a single test and test_name_${i} with integer
# values ${i} in a case of multiple tests. The above mentioned example creates
# 4 tests: test_name_1, test_name_2, test_name_3 and test_name_4. The first
# test executes command executable_name, the second one runs executable_name
# with arguments arg1 arg2 arg3. It is then easy to launch all of these tests
# by a single command ctest -R test_name. User shall be careful when choosing
# test_name as it shall be unique across entire CMake build. The next argument
# EXEC_NAME simply defines name for the executable which is equal to test_name
# in case EXEC_NAME is not provided. SOURCES argument contains list of all
# sources of the test executable. LINK_LIBRARIES defines a set of libraries to
# be linked against the test executable. ARGS is a list of strings, each string
# contains arguments for a single test. COV_ENABLE enables coverage target with
# a name COV_NAME which executes ctest -R test_name. Additionally, global
# coverage target COV_GLOBAL (which simply executes ctest) is updated to have
# executable executable_name as a dependency.
function(add_test_set)
    # List of non-value arguments
    set(options)
    # List of single-value arguments
    set(one_value_args TEST_NAME EXEC_NAME COV_ENABLE COV_NAME COV_GLOBAL)
    # List of multi-value arguments
    set(multi_value_args SOURCES LINK_LIBRARIES ARGS)
    # Parse arguments
    cmake_parse_arguments(_args "${options}" "${one_value_args}"
        "${multi_value_args}" ${ARGN})
    # Check if the function arguments were properly defined
    if(DEFINED _args_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "add_test_set() had the following unpared \
        arguments: ${_args_UNPARSED_ARGUMENTS}")
    endif()
    # Register executable with its unique target name
    add_executable(${_args_TEST_NAME} ${_args_SOURCES})
    # Set name of executable file
    if(_args_EXEC_NAME)
        set_target_properties(${_args_TEST_NAME} PROPERTIES OUTPUT_NAME
            ${_args_EXEC_NAME})
    endif()
    # Link to provided libraries
    target_link_libraries(${_args_TEST_NAME} PRIVATE ${_args_LINK_LIBRARIES})
    # Try to set environment for a test on Apple MacOS
    if(APPLE AND DEFINED ADD_RPATH)
        set_target_properties(${_args_TEST_NAME} PROPERTIES BUILD_RPATH
            "${ADD_RPATH}")
    endif()
    # Check if coverage needed
    if(_args_COV_ENABLE)
        # Extend dependencies list for global coverage
        add_dependencies(${_args_COV_GLOBAL} ${_args_TEST_NAME})
        # Add target for local coverage
        setup_target_for_coverage_lcov(NAME ${_args_COV_NAME}
            EXECUTABLE ctest -R ${_args_TEST_NAME}
            DEPENDENCIES ${_args_TEST_NAME}
            LCOV_ARGS --no-external)
    endif()
    # Add test suite to the CTest
    list(LENGTH _args_ARGS ntests)
    if(ntests EQUAL 0)
        # Test without arguments
        message(STATUS "Adding 1 test for ${_args_TEST_NAME}")
        add_test(NAME ${_args_TEST_NAME} COMMAND ${_args_EXEC_NAME})
    elseif(ntests EQUAL 1)
        # Test with a single set of arguments
        message(STATUS "Adding 1 test for ${_args_TEST_NAME}")
        # Convert string of arguments to a list
        string(REPLACE " " ";" _args_ARGS "${_args_ARGS}")
        # Test with properly parsed arguments
        add_test(NAME ${_args_TEST_NAME} COMMAND ${_args_EXEC_NAME}
            ${_args_ARGS})
    else()
        # Multiple sets of arguments
        message(STATUS "Adding ${ntests} test for ${_args_TEST_NAME}")
        foreach(i RANGE 1 ${ntests})
            # Read current set of arguments as a string
            list(POP_FRONT _args_ARGS current_ARGS)
            # Convert string of arguments to a list
            string(REPLACE " " ";" current_ARGS "${current_ARGS}")
            # Test with properly parsed arguments
            add_test(NAME ${_args_TEST_NAME}_${i} COMMAND ${_args_EXEC_NAME}
                ${current_ARGS})
        endforeach()
    endif()
endfunction()

add_test_set(TEST_NAME tests_constants
    EXEC_NAME constants
    SOURCES constants.cc
    LINK_LIBRARIES nntile
    COV_ENABLE ${COVERAGE}
    COV_NAME coverage_constants
    COV_GLOBAL coverage
    )

add_test_set(TEST_NAME tests_starpu
    EXEC_NAME starpu
    SOURCES starpu.cc
    LINK_LIBRARIES nntile
    ARGS "1" "2" "3" "4" "5"
    COV_ENABLE ${COVERAGE}
    COV_NAME coverage_starpu
    COV_GLOBAL coverage
    )

add_subdirectory("tile")
add_subdirectory("tensor")

