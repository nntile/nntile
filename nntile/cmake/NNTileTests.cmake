# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file nntile/cmake/NNTileTests.cmake
# NNTileTests.cmake
#
# @version 1.1.0
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
    option(BUILD_TESTS_${_u} "Build and run tests for ${_sub}" OFF)
endforeach()

# CI / per-subsystem jobs: -DNNTILE_TEST_SUBSYSTEM=tensor (see ctest-run-subsystem.sh).
set(NNTILE_TEST_SUBSYSTEM "" CACHE STRING
    "Enable BUILD_TESTS_* only for this subsystem (and its test deps)")

set(NNTILE_LINK_CACHED_TEST_OBJECTS OFF CACHE BOOL
    "Link test executables from prebuilt objects (CI build-tests)")

function(nntile_test_cached_object_path src out_var)
    if(NOT NNTILE_TEST_SUBSYSTEM)
        message(FATAL_ERROR "nntile_test_cached_object_path requires NNTILE_TEST_SUBSYSTEM")
    endif()
    string(TOLOWER "${NNTILE_TEST_SUBSYSTEM}" _sub)
    get_filename_component(_name "${src}" NAME)
    set(_obj_dir
        "${CMAKE_BINARY_DIR}/nntile/tests/CMakeFiles/nntile_test_objs_${_sub}.dir")
    file(GLOB_RECURSE _candidates
        "${_obj_dir}/${_name}.o"
        "${_obj_dir}/*/${_name}.o")
    list(LENGTH _candidates _n)
    if(_n EQUAL 1)
        set(${out_var} "${_candidates}" PARENT_SCOPE)
    elseif(_n GREATER 1)
        message(FATAL_ERROR
            "Ambiguous cached test object for ${src}: ${_candidates}")
    else()
        message(FATAL_ERROR
            "Missing cached test object for ${src} under ${_obj_dir}")
    endif()
endfunction()

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

# When compiling tests only (no link/run), skip CTest registration.
if(NNTILE_COMPILE_CHECK_TESTS_SUBSYSTEM)
    set(NNTILE_REGISTER_CTEST OFF)
else()
    set(NNTILE_REGISTER_CTEST ON)
endif()

# OBJECT compile-check: Catch2 headers only (no Catch2 library build/link).
function(nntile_apply_catch2_compile_headers target)
    if(NNTILE_FETCHCONTENT_DISCONNECTED)
        set(_catch2_src "${CMAKE_BINARY_DIR}/_deps/catch2-src/src")
        set(_catch2_gen
            "${CMAKE_BINARY_DIR}/_deps/catch2-build/generated-includes")
        if(NOT EXISTS "${_catch2_src}/catch2/catch_test_macros.hpp")
            message(FATAL_ERROR "Catch2 sources missing under ${_catch2_src} "
                "(populate build/_deps from build-test-prerequisites)")
        endif()
        if(NOT EXISTS "${_catch2_gen}/catch2/catch_user_config.hpp")
            message(FATAL_ERROR "Catch2 generated headers missing under ${_catch2_gen} "
                "(build-test-prerequisites must configure Catch2 once)")
        endif()
        target_include_directories(${target} PRIVATE "${_catch2_src}" "${_catch2_gen}")
        return()
    endif()
    if(NOT TARGET Catch2::Catch2WithMain)
        message(FATAL_ERROR
            "Catch2::Catch2WithMain missing; configure external/Catch2 first")
    endif()
    get_target_property(_catch2_inc Catch2::Catch2WithMain
        INTERFACE_INCLUDE_DIRECTORIES)
    if(_catch2_inc MATCHES ";")
        list(GET _catch2_inc 0 _catch2_inc)
    endif()
    if(NOT _catch2_inc OR NOT EXISTS "${_catch2_inc}/catch2/catch_test_macros.hpp")
        message(FATAL_ERROR "Catch2 headers not found (expected under ${_catch2_inc})")
    endif()
    target_include_directories(${target} PRIVATE "${_catch2_inc}")
endfunction()

function(nntile_add_test_compile_check subsystem)
    string(TOLOWER "${subsystem}" _sub)
    list(FIND NNTILE_TEST_SUBSYSTEMS "${_sub}" _idx)
    if(_idx LESS 0)
        message(FATAL_ERROR
            "Unknown NNTILE_COMPILE_CHECK_TESTS_SUBSYSTEM=${subsystem}")
    endif()
    set(_dir "${CMAKE_CURRENT_SOURCE_DIR}/${_sub}")
    file(GLOB_RECURSE _test_src CONFIGURE_DEPENDS "${_dir}/*.cc")
    if(_sub STREQUAL "kernel")
        list(APPEND _test_src "${CMAKE_CURRENT_SOURCE_DIR}/constants.cc")
    endif()
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
    nntile_apply_catch2_compile_headers(nntile_test_objs_${_sub})
    target_link_libraries(nntile_test_objs_${_sub} PRIVATE
        nlohmann_json::nlohmann_json)
    add_custom_target(nntile_compile_check_tests_${_sub}
        DEPENDS nntile_test_objs_${_sub})
endfunction()
