# All unit tests without arguments to test executable
set(TESTS
    "traits"
    "tensor"
    "randn"
    "copy"
    "gemm"
    "bias"
    "gelu"
    "norm"
    )

foreach(test IN LISTS TESTS)
    add_test_set(TEST_NAME tests_tensor_${test}
        EXEC_NAME ${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile
        COV_ENABLE ${COVERAGE}
        COV_NAME coverage_tensor_${test}
        COV_GLOBAL coverage
        )
endforeach()
