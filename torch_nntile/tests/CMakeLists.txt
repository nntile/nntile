# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/CMakeLists.txt
# C++ tests for libtorch_nntile (PrivateUse1 vs CPU).

cmake_minimum_required(VERSION 3.24)

include("${PROJECT_SOURCE_DIR}/cmake/NntileCatch2.cmake")

if(NOT TARGET torch_nntile::torch_nntile)
    find_package(torch_nntile REQUIRED CONFIG)
endif()

if(NOT Torch_FOUND)
    find_package(Torch REQUIRED)
endif()

set(_torch_nntile_test_include
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

# catch_discover_tests runs the binary at build time; DT_RUNPATH is not
# transitive, so libnntile must be on LD_LIBRARY_PATH / DL_PATHS.
set(_torch_nntile_dl_paths "")
foreach(_tgt IN ITEMS torch_nntile::torch_nntile nntile::nntile)
    if(NOT TARGET ${_tgt})
        continue()
    endif()
    foreach(_prop IN ITEMS
        IMPORTED_LOCATION
        IMPORTED_LOCATION_RELEASE
        IMPORTED_LOCATION_RELWITHDEBINFO
        IMPORTED_LOCATION_DEBUG
    )
        get_target_property(_loc ${_tgt} ${_prop})
        if(_loc AND NOT _loc STREQUAL "_loc-NOTFOUND")
            get_filename_component(_dir "${_loc}" DIRECTORY)
            list(APPEND _torch_nntile_dl_paths "${_dir}")
            break()
        endif()
    endforeach()
    # In-tree SHARED target: use generator-safe LOCATION via TARGET_FILE_DIR
    # after build; for discover-time DL_PATHS use BINARY_DIR.
    get_target_property(_tgt_type ${_tgt} TYPE)
    if(_tgt_type STREQUAL "SHARED_LIBRARY" OR _tgt_type STREQUAL "MODULE_LIBRARY")
        get_target_property(_bin ${_tgt} BINARY_DIR)
        if(_bin AND NOT _bin STREQUAL "_bin-NOTFOUND")
            list(APPEND _torch_nntile_dl_paths "${_bin}")
        endif()
    endif()
endforeach()
if(DEFINED ENV{STARPU_PREFIX})
    list(APPEND _torch_nntile_dl_paths "$ENV{STARPU_PREFIX}/lib")
elseif(EXISTS "/opt/starpu/lib")
    list(APPEND _torch_nntile_dl_paths "/opt/starpu/lib")
endif()
list(REMOVE_DUPLICATES _torch_nntile_dl_paths)

set(_torch_nntile_ld_path "")
if(_torch_nntile_dl_paths)
    string(JOIN ":" _torch_nntile_ld_path ${_torch_nntile_dl_paths})
endif()
if(DEFINED ENV{LD_LIBRARY_PATH} AND NOT "$ENV{LD_LIBRARY_PATH}" STREQUAL "")
    if(_torch_nntile_ld_path)
        set(_torch_nntile_ld_path
            "${_torch_nntile_ld_path}:$ENV{LD_LIBRARY_PATH}")
    else()
        set(_torch_nntile_ld_path "$ENV{LD_LIBRARY_PATH}")
    endif()
endif()

set(_torch_nntile_test_env
    "STARPU_SILENT=1;STARPU_FXT_TRACE=0;STARPU_WORKERS_NOBIND=1;OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1"
)
if(_torch_nntile_ld_path)
    string(APPEND _torch_nntile_test_env
        ";LD_LIBRARY_PATH=${_torch_nntile_ld_path}")
endif()

function(torch_nntile_add_test name source)
    add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE
        ${_torch_nntile_test_include})
    target_link_libraries(${name} PRIVATE
        torch_nntile::torch_nntile
        Catch2::Catch2WithMain
        ${TORCH_LIBRARIES}
    )
    set_property(TARGET ${name} PROPERTY CXX_STANDARD 17)
    if(TORCH_CXX_FLAGS)
        target_compile_options(${name} PRIVATE ${TORCH_CXX_FLAGS})
    endif()
    # One CTest entry per Catch2 TEST_CASE (not just the binary).
    catch_discover_tests(${name}
        TEST_PREFIX "${name}."
        DL_PATHS ${_torch_nntile_dl_paths}
        PROPERTIES
            LABELS "libtorch_nntile"
            ENVIRONMENT "${_torch_nntile_test_env}"
    )
endfunction()

torch_nntile_add_test(tests_libtorch_nntile_smoke smoke_add.cc)
if(NNTILE_TORCH_NATIVE_OPS)
    torch_nntile_add_test(
        tests_libtorch_nntile_aten_ops
        aten_ops_parity.cc)
endif()
if(NNTILE_NNTILE_NATIVE_OPS)
    torch_nntile_add_test(
        tests_libtorch_nntile_models_smoke
        models_fwd_bwd_smoke.cc)
endif()
