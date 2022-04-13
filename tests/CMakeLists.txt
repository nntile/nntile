# Set list of tests
set(EXAMPLES
    "starpu.cc"
    )

# Add single test executable, presented by its single source, along with a list
# of combinations of parameters (stored as strings). One test executable = one
# source file. Number of tests = number of combinations. These tests are then
# available for ctest.
function(custom_add_test src deps)
    # Parse path to get name of executable and its corresponding unique logical
    # name, based on its path and name
    get_filename_component(exec ${src} NAME_WE)
    file(RELATIVE_PATH path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
    string(REPLACE "/" "_" suffix ${path})
    set(logical ${suffix}_${exec})
    # Register executable with its unique logical name
    add_executable(${logical} ${src})
    # Set name of executable file
    set_target_properties(${logical} PROPERTIES OUTPUT_NAME ${exec})
    # Add dependency to the global coverage target
    if(COVERAGE)
        add_dependencies(coverage ${logical})
    endif()
    # Try to set environment for a test on Apple MacOS
    if(APPLE AND DEFINED ADD_RPATH)
        set_target_properties(${logical} PROPERTIES BUILD_RPATH "${ADD_RPATH}")
    endif()
    # Link to dependencies
    target_link_libraries(${logical} PRIVATE ${deps})
    # If no arguments were provided, add test without parameters
    if(${ARGC} EQUAL 2)
        add_test(NAME ${logical} COMMAND ${exec})
        if(COVERAGE)
            # Add coverage by a single test
            setup_target_for_coverage_lcov(NAME coverage_${logical}
                EXECUTABLE ${path}/${exec}
                DEPENDENCIES ${logical}
                EXCLUDE "${CMAKE_SOURCE_DIR}/tests/*"
                LCOV_ARGS --no-external)
        endif()
        message(STATUS "Adding 1 test: ${path}/${exec}")
    # If only one test with arguments must be added
    elseif(${ARGC} EQUAL 3)
        # Convert arguments from string to list
        string(REPLACE " " ";" param ${ARGV2})
        add_test(NAME ${logical} COMMAND ${exec} ${param})
        message(STATUS "Adding 1 test: ${path}/${exec}")
    # Add multiple tests and add indexing for better readability of output
    else()
        math(EXPR ntests ${ARGC}-2)
        message(STATUS "Adding ${ntests} tests: ${path}/${exec}")
        # For every string of parameters, register corresponding test
        foreach(i RANGE 1 ${ntests})
            # Get corresponding arguments as a string
            math(EXPR ip1 ${i}+1)
            set(test_argv ${ARGV${ip1}})
            # Convert arguments from string to list
            string(REPLACE " " ";" param ${test_argv})
            # Add test to the pool of tests
            add_test(NAME ${logical}_${i} COMMAND ${exec} ${param})
        endforeach()
    endif()
endfunction(custom_add_test)

custom_add_test("constants.cc" nntile)
custom_add_test("starpu.cc" nntile "1" "2" "3" "4")

add_subdirectory("tile")
add_subdirectory("tensor")

