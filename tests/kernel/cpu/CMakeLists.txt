# All unit tests without arguments to test executable
set(TESTS
    "sumnorm"
    )

foreach(test IN LISTS TESTS)
    add_test_set(TARGET_NAME tests_kernel_cpu_${test}
        EXEC_NAME ${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_kernel_cpu_${test}
        COV_GLOBAL coverage
        )
endforeach()

