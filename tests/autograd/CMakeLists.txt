# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file tests/autograd/CMakeLists.txt
# Tests for nntile::autograd functions
#
# @version 1.1.0

# All unit tests without arguments to test executable
set(TESTS
    "autograd"
    )

set(TESTS_MPI
    )

# Describe all tests that are not yet implemented
set(TESTS_NOT_IMPLEMENTED
    )

# Add target for local coverage
if(BUILD_COVERAGE)
    setup_target_for_coverage_lcov(NAME coverage_autograd
        EXECUTABLE ctest -R tests_autograd_
        LCOV_ARGS --no-external
        GENHTML_ARGS --prefix ${PROJECT_SOURCE_DIR})
endif()

foreach(test IN LISTS TESTS)
    add_test_set(TARGET_NAME tests_autograd_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_autograd_${test}
        COV_GLOBAL coverage_autograd coverage
        )
endforeach()

foreach(test IN LISTS TESTS_MPI)
    set(labels)
    if(test IN_LIST TESTS_NOT_IMPLEMENTED)
        set(labels "NotImplemented")
    endif()
    # Add non-mpirun test
    add_test_set(TARGET_NAME tests_autograd_${test}
        EXEC_NAME test_${test}
        SOURCES ${test}.cc
        LINK_LIBRARIES nntile
        COV_ENABLE ${BUILD_COVERAGE}
        COV_NAME coverage_autograd_${test}
        COV_GLOBAL coverage_autograd coverage
        LABELS ${labels}
        )
    # # Add mpirun test (the same source, but different output executable)
    # add_test_set(TARGET_NAME tests_tensor_${test}_mpi
    #     EXEC_NAME test_${test}_mpi
    #     SOURCES ${test}.cc
    #     LINK_LIBRARIES nntile
    #     MPI_NUMPROC 4
    #     COV_ENABLE ${BUILD_COVERAGE}
    #     COV_NAME coverage_tensor_${test}
    #     COV_GLOBAL coverage_tensor coverage
    #     LABELS ${labels} MPI
    #     )
endforeach()
