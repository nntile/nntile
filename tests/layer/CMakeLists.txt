# All unit tests for layers
set(TESTS
    "fully_connected"
    )

foreach(test IN LISTS TESTS)
    add_test_set(TARGET_NAME tests_layer_${test}
        EXEC_NAME ${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_layer_${test}
        COV_GLOBAL coverage
        )
endforeach()

