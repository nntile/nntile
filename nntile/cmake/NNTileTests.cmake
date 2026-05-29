# @file nntile/cmake/NNTileTests.cmake
# Per-subsystem test options, test compile-check, and CTest registration control.

# Subsystems that have a test tree under nntile/tests/<dir>/
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
    option(BUILD_TESTS_${_u} "Build and run tests for ${_sub}" ON)
endforeach()

# When compiling tests only (no link/run), skip CTest registration.
if(NNTILE_COMPILE_CHECK_TESTS_SUBSYSTEM)
    set(NNTILE_REGISTER_CTEST OFF)
else()
    set(NNTILE_REGISTER_CTEST ON)
endif()

function(nntile_add_test_compile_check subsystem)
    string(TOLOWER "${subsystem}" _sub)
    list(FIND NNTILE_TEST_SUBSYSTEMS "${_sub}" _idx)
    if(_idx LESS 0)
        message(FATAL_ERROR
            "Unknown NNTILE_COMPILE_CHECK_TESTS_SUBSYSTEM=${subsystem}")
    endif()
    set(_dir "${CMAKE_CURRENT_SOURCE_DIR}/${_sub}")
    file(GLOB_RECURSE _test_src CONFIGURE_DEPENDS "${_dir}/*.cc")
    if(_sub STREQUAL "starpu")
        list(FILTER _test_src EXCLUDE REGEX "/config\\.cc$")
    endif()
    if(NOT NNTILE_USE_CUDA)
        list(FILTER _test_src EXCLUDE REGEX "flash_sdpa_(fwd|bwd)_cudnn\\.cc$")
    endif()
    if(NOT _test_src)
        message(STATUS "No test sources under ${_dir}; "
            "nntile_compile_check_tests_${_sub} is empty")
        add_custom_target(nntile_compile_check_tests_${_sub})
        return()
    endif()
    add_library(nntile_test_objs_${_sub} OBJECT ${_test_src})
    nntile_apply_common_includes(nntile_test_objs_${_sub})
    nntile_apply_cuda(nntile_test_objs_${_sub})
    set(_test_inc_dirs
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/${_sub}")
    target_include_directories(nntile_test_objs_${_sub} PRIVATE
        ${_test_inc_dirs})
    target_link_libraries(nntile_test_objs_${_sub} PRIVATE
        Catch2::Catch2WithMain
        nlohmann_json::nlohmann_json)
    add_custom_target(nntile_compile_check_tests_${_sub}
        DEPENDS nntile_test_objs_${_sub})
endfunction()
