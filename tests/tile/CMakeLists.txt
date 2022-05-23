# All unit tests without arguments to test executable
set(TESTS
    "traits"
    "tile"
    "randn"
    "copy"
    "gemm"
    "bias"
    "gelu"
    "relu"
    "norm"
    )

foreach(test IN LISTS TESTS)
    add_test_set(TARGET_NAME tests_tile_${test}
        EXEC_NAME ${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_tile_${test}
        COV_GLOBAL coverage
        )
endforeach()

