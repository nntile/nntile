# All unit tests without arguments to test executable
set(TESTS
    "traits"
    "tile"
    "randn"
    "copy"
    "gemm"
    "bias"
    "gelu"
    )

foreach(test IN LISTS TESTS)
    add_test_set(TARGET tests_tile_${test}
        COVERAGE coverage_tile_${test}
        EXECUTABLE ${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile
        )
endforeach()

