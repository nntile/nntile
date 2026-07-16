# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/tests/CMakeLists.txt
# C++ tests for libtorch_nntile (PrivateUse1 vs CPU).

cmake_minimum_required(VERSION 3.24)

if(NOT TARGET Catch2::Catch2WithMain)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.11.0
    )
    FetchContent_MakeAvailable(Catch2)
endif()

if(NOT TARGET torch_nntile::torch_nntile)
    find_package(torch_nntile REQUIRED CONFIG)
endif()

if(NOT Torch_FOUND)
    find_package(Torch REQUIRED)
endif()

include(CTest)

set(_torch_nntile_test_include
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

function(torch_nntile_add_test name source)
    add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE ${_torch_nntile_test_include})
    target_link_libraries(${name} PRIVATE
        torch_nntile::torch_nntile
        Catch2::Catch2WithMain
        ${TORCH_LIBRARIES}
    )
    set_property(TARGET ${name} PROPERTY CXX_STANDARD 17)
    if(TORCH_CXX_FLAGS)
        target_compile_options(${name} PRIVATE ${TORCH_CXX_FLAGS})
    endif()
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES
        LABELS "libtorch_nntile"
        ENVIRONMENT
            "STARPU_SILENT=1;STARPU_FXT_TRACE=0;STARPU_WORKERS_NOBIND=1;OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1"
    )
endfunction()

torch_nntile_add_test(tests_libtorch_nntile_smoke smoke_add.cc)
torch_nntile_add_test(tests_libtorch_nntile_aten_ops aten_ops_parity.cc)
