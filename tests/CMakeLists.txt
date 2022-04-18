# Set list of tests
set(EXAMPLES
    "starpu.cc"
    )

# Add single test executable, presented by its single source, along with a list
# of combinations of parameters (stored as strings). One test executable = one
# source file. Number of tests = number of combinations. These tests are then
# available for ctest.
function(add_test_set)
    # Parse arguments
    set(options)
    set(one_value_args TARGET COVERAGE EXECUTABLE)
    set(multi_value_args SOURCES LINK_LIBRARIES ARGS)
    cmake_parse_arguments(test "${options}" "${one_value_args}"
        "${multi_value_args}" ${ARGN})
    # Register executable with its unique target name
    add_executable(${test_TARGET} ${test_SOURCES})
    # Set name of executable file
    set_target_properties(${test_TARGET} PROPERTIES OUTPUT_NAME
        ${test_EXECUTABLE})
    # Link to provided libraries
    target_link_libraries(${test_TARGET} PRIVATE ${test_LINK_LIBRARIES})
    # Try to set environment for a test on Apple MacOS
    if(APPLE AND DEFINED ADD_RPATH)
        set_target_properties(${test_TARGET} PROPERTIES BUILD_RPATH
            "${ADD_RPATH}")
    endif()
    # Check if coverage needed
    if(COVERAGE)
        # Extend dependencies list for global coverage
        add_dependencies(coverage ${test_TARGET})
        # Add target for local coverage
        setup_target_for_coverage_lcov(NAME ${test_COVERAGE}
            EXECUTABLE ctest -R ${test_TARGET}
            DEPENDENCIES ${test_TARGET}
            EXCLUDE "${CMAKE_SOURCE_DIR}/tests/*"
            LCOV_ARGS --no-external)
    endif()
    # Add test suite to the CTest
    list(LENGTH test_ARGS ntests)
    if(ntests EQUAL 0)
        # Test without arguments
        message(STATUS "Adding 1 test for ${test_TARGET}")
        add_test(NAME ${test_TARGET} COMMAND ${test_EXECUTABLE})
    elseif(ntests EQUAL 1)
        # Test with a single set of arguments
        message(STATUS "Adding 1 test for ${test_TARGET}")
        # Convert string of arguments to a list
        string(REPLACE " " ";" test_ARGS "${test_ARGS}")
        # Test with properly parsed arguments
        add_test(NAME ${test_TARGET} COMMAND ${test_EXECUTABLE} ${test_ARGS})
    else()
        # Multiple sets of arguments
        message(STATUS "Adding ${ntests} test for ${test_TARGET}")
        foreach(i RANGE 1 ${ntests})
            # Read current set of arguments as a string
            list(POP_FRONT test_ARGS current_test_ARGS)
            # Convert string of arguments to a list
            string(REPLACE " " ";" current_test_ARGS "${current_test_ARGS}")
            # Test with properly parsed arguments
            add_test(NAME ${test_TARGET}_${i} COMMAND ${test_EXECUTABLE}
                ${current_test_ARGS})
        endforeach()
    endif()
endfunction()

add_test_set(TARGET tests_constants
    COVERAGE coverage_constants
    EXECUTABLE constants
    SOURCES constants.cc
    LINK_LIBRARIES nntile
    )

add_test_set(TARGET tests_starpu
    COVERAGE coverage_starpu
    EXECUTABLE starpu
    SOURCES starpu.cc
    LINK_LIBRARIES nntile
    ARGS "1" "2" "3" "4"
    )

add_subdirectory("tile")
#add_subdirectory("tensor")

