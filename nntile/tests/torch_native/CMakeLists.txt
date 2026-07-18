# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file nntile/tests/torch_native/CMakeLists.txt
# Layer tests for torch-native StarPU / core / tile / tensor ops.

include("${PROJECT_SOURCE_DIR}/cmake/NntileCatch2.cmake")

if(NOT Torch_FOUND)
    find_package(Torch REQUIRED)
endif()

set(_torch_native_env
    "STARPU_SILENT=1;STARPU_FXT_TRACE=0;STARPU_WORKERS_NOBIND=1;OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1"
)

function(nntile_torch_native_add_test name source labels)
    add_test_set(
        TARGET_NAME ${name}
        EXEC_NAME ${name}
        SOURCES ${source}
        LINK_LIBRARIES nntile::nntile Catch2::Catch2WithMain
            ${TORCH_LIBRARIES}
        LABELS "${labels}"
    )
    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..)
    if(TORCH_CXX_FLAGS)
        target_compile_options(${name} PRIVATE ${TORCH_CXX_FLAGS})
    endif()
    set_tests_properties(${name} PROPERTIES
        ENVIRONMENT "${_torch_native_env}")
endfunction()

nntile_torch_native_add_test(
    tests_torch_native_starpu_torch_dispatch
    starpu/torch_dispatch.cc
    "torch_native;starpu;smoke")
nntile_torch_native_add_test(
    tests_torch_native_core_torch_dispatch
    core/torch_dispatch.cc
    "torch_native;core;smoke")
nntile_torch_native_add_test(
    tests_torch_native_tile_torch_dispatch
    tile/ops/torch_dispatch.cc
    "torch_native;graph;tile;smoke")
nntile_torch_native_add_test(
    tests_torch_native_tensor_torch_dispatch
    tensor/ops/torch_dispatch.cc
    "torch_native;graph;tensor;smoke")
