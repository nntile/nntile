# All unit tests without arguments to test executable
set(TESTS
    "traits"
    "tensor"
    "randn"
    "copy"
    "gemm"
    "bias"
    "gelu"
    )

foreach(test IN LISTS TESTS)
    add_test_set(TARGET tests_tensor_${test}
        COVERAGE coverage_tensor_${test}
        EXECUTABLE ${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile
        )
endforeach()
