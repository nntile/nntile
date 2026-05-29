# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file nntile/cmake/NNTileTests.cmake
# Per-subsystem test options and CTest registration.
#
# @version 1.1.0

set(NNTILE_TEST_SUBSYSTEMS
    kernel
    starpu
    core
    tile
    tensor
    nn
    module
    model
    io
)

foreach(_sub IN LISTS NNTILE_TEST_SUBSYSTEMS)
    string(TOUPPER "${_sub}" _u)
    option(BUILD_TESTS_${_u} "Build and run tests for ${_sub}" OFF)
endforeach()

# Local partial test builds: -DNNTILE_TEST_SUBSYSTEM=tensor
set(NNTILE_TEST_SUBSYSTEM "" CACHE STRING
    "Enable BUILD_TESTS_* only for this subsystem (and its test deps)")

function(_nntile_force_build_tests)
    foreach(_sub IN LISTS ARGN)
        string(TOUPPER "${_sub}" _u)
        set(BUILD_TESTS_${_u} ON CACHE BOOL "Build tests for ${_sub}" FORCE)
    endforeach()
endfunction()

if(NNTILE_TEST_SUBSYSTEM)
    foreach(_sub IN LISTS NNTILE_TEST_SUBSYSTEMS)
        string(TOUPPER "${_sub}" _u)
        set(BUILD_TESTS_${_u} OFF CACHE BOOL "Build tests for ${_sub}" FORCE)
    endforeach()
    string(TOLOWER "${NNTILE_TEST_SUBSYSTEM}" _req)
    if(_req STREQUAL "kernel")
        _nntile_force_build_tests(kernel)
    elseif(_req STREQUAL "starpu")
        _nntile_force_build_tests(kernel starpu)
    elseif(_req STREQUAL "core")
        _nntile_force_build_tests(kernel starpu core)
    elseif(_req STREQUAL "tile")
        _nntile_force_build_tests(tile)
    elseif(_req STREQUAL "tensor")
        _nntile_force_build_tests(tensor)
    elseif(_req STREQUAL "nn")
        _nntile_force_build_tests(nn)
    elseif(_req STREQUAL "module")
        _nntile_force_build_tests(module)
    elseif(_req STREQUAL "model")
        _nntile_force_build_tests(model)
    elseif(_req STREQUAL "io")
        _nntile_force_build_tests(io)
    else()
        message(FATAL_ERROR "Unknown NNTILE_TEST_SUBSYSTEM=${NNTILE_TEST_SUBSYSTEM}")
    endif()
elseif(BUILD_TESTS)
    set(_nntile_any_test_subsystem OFF)
    foreach(_sub IN LISTS NNTILE_TEST_SUBSYSTEMS)
        string(TOUPPER "${_sub}" _u)
        if(BUILD_TESTS_${_u})
            set(_nntile_any_test_subsystem ON)
        endif()
    endforeach()
    if(NOT _nntile_any_test_subsystem)
        _nntile_force_build_tests(
            kernel starpu core tile tensor nn module model io)
    endif()
endif()
