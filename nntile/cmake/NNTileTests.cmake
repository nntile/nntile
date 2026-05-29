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

function(nntile_test_cached_object_subsystems out_var)
    if(NOT NNTILE_TEST_SUBSYSTEM)
        message(FATAL_ERROR
            "nntile_test_cached_object_subsystems requires NNTILE_TEST_SUBSYSTEM")
    endif()
    string(TOLOWER "${NNTILE_TEST_SUBSYSTEM}" _req)
    if(_req STREQUAL "kernel")
        set(_subs kernel)
    elseif(_req STREQUAL "starpu")
        set(_subs kernel starpu)
    elseif(_req STREQUAL "core")
        set(_subs kernel starpu core)
    elseif(_req MATCHES "^(tile|tensor|nn|module|model|io)$")
        set(_subs ${_req})
    else()
        message(FATAL_ERROR "Unknown NNTILE_TEST_SUBSYSTEM=${NNTILE_TEST_SUBSYSTEM}")
    endif()
    set(${out_var} ${_subs} PARENT_SCOPE)
endfunction()

function(nntile_test_cached_object_path src out_var)
    if(NOT NNTILE_TEST_SUBSYSTEM)
        message(FATAL_ERROR "nntile_test_cached_object_path requires NNTILE_TEST_SUBSYSTEM")
    endif()
    nntile_test_cached_object_subsystems(_subs)

    if(IS_ABSOLUTE "${src}")
        file(RELATIVE_PATH _obj_rel "${CMAKE_SOURCE_DIR}/nntile/tests" "${src}")
        if(_obj_rel MATCHES "^\\.\\.")
            get_filename_component(_obj_rel "${src}" NAME)
        endif()
    else()
        set(_obj_rel "${src}")
    endif()
    string(REGEX REPLACE "\\.cc$" ".cc.o" _obj_rel "${_obj_rel}")

    get_filename_component(_src_name "${src}" NAME)
    set(_search_order "")
    if(_src_name STREQUAL "constants.cc")
        list(APPEND _search_order kernel)
    else()
        get_filename_component(_caller_sub "${CMAKE_CURRENT_LIST_DIR}" NAME)
        if(_caller_sub IN_LIST NNTILE_TEST_SUBSYSTEMS)
            list(APPEND _search_order "${_caller_sub}")
        endif()
        foreach(_sub IN LISTS _subs)
            if(NOT _sub IN_LIST _search_order)
                list(APPEND _search_order "${_sub}")
            endif()
        endforeach()
    endif()

    set(_candidates "")
    foreach(_sub IN LISTS _search_order)
        set(_obj_dir
            "${CMAKE_BINARY_DIR}/nntile/tests/CMakeFiles/nntile_test_objs_${_sub}.dir")
        if(NOT IS_DIRECTORY "${_obj_dir}")
            continue()
        endif()
        set(_sub_candidates "")
        file(GLOB_RECURSE _objs "${_obj_dir}/*.o")
        string(LENGTH "${_obj_rel}" _rel_len)
        foreach(_obj IN LISTS _objs)
            file(RELATIVE_PATH _path "${_obj_dir}" "${_obj}")
            string(LENGTH "${_path}" _path_len)
            if(_path_len LESS _rel_len)
                continue()
            endif()
            math(EXPR _start "${_path_len} - ${_rel_len}")
            string(SUBSTRING "${_path}" ${_start} -1 _suffix)
            if(NOT _suffix STREQUAL "${_obj_rel}")
                continue()
            endif()
            if(_start EQUAL 0)
                list(APPEND _sub_candidates "${_obj}")
            else()
                math(EXPR _slash_pos "${_start} - 1")
                string(SUBSTRING "${_path}" ${_slash_pos} 1 _sep)
                if(_sep STREQUAL "/")
                    list(APPEND _sub_candidates "${_obj}")
                endif()
            endif()
        endforeach()
        list(LENGTH _sub_candidates _sub_n)
        if(_sub_n EQUAL 1)
            set(${out_var} "${_sub_candidates}" PARENT_SCOPE)
            return()
        elseif(_sub_n GREATER 1)
            message(FATAL_ERROR
                "Ambiguous cached test object for ${src} in nntile_test_objs_${_sub}.dir: "
                "${_sub_candidates}")
        endif()
    endforeach()

    message(FATAL_ERROR
        "Missing cached test object for ${src} (expected */${_obj_rel} under "
        "nntile_test_objs_{${_search_order}}.dir)")
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
