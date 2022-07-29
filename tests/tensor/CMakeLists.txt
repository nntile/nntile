# All unit tests without arguments to test executable
set(TESTS
    "traits"
    "tensor"
    "randn"
    "copy"
    "clear"
    "gemm"
    "bias"
    "gelu"
    "relu"
    #"norm"
    "sumnorm"
    )

foreach(test IN LISTS TESTS)
    add_test_set(TARGET_NAME tests_tensor_${test}
        EXEC_NAME ${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_tensor_${test}
        COV_GLOBAL coverage
        )
endforeach()

