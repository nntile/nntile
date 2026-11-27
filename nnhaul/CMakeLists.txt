# SPDX-License-Identifier: MIT
# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# NNHaul is a task-based dynamic execution engine: a virtual planner
# builds one plan per batch from a drained snapshot, and a literal
# executor issues exactly the planned operations.
#
# @file CMakeLists.txt
# Top-level build: the libnnhaul target, its install/export rules, and
# the test suites. Works standalone and as an add_subdirectory /
# FetchContent subproject (tests off by default then).
#
# @version 1.0.0

cmake_minimum_required(VERSION 3.18)

# Top-level or subproject. Computed before anything that must not leak
# into a parent build (CUDA arch default, standards, PIC, rpath).
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(NNHAUL_IS_TOP_LEVEL ON)
else()
    set(NNHAUL_IS_TOP_LEVEL OFF)
endif()

# Set before project() so CMake does not detect a native architecture.
# Only as a top-level project: a parent owns the architecture list.
if(NNHAUL_IS_TOP_LEVEL AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures")
endif()

project(nnhaul VERSION 1.0.0 LANGUAGES CXX CUDA)

# ---- the chunk class (runtime configuration) ----
# The chunk size is the engine's page. It is a RUNTIME value, read on
# first use from the NNHAUL_CHUNK_SIZE_MB environment variable (MiB, a
# power of two). This option only fixes the DEFAULT used when the env
# variable is unset; the generated nnhaul/defs.h carries the runtime
# accessors NNTile also reads, so the two sides cannot disagree.
set(NNHAUL_DEFAULT_CHUNK_SIZE_MB 4 CACHE STRING
    "Default chunk class size in MiB (a positive power of two)")
if(NOT NNHAUL_DEFAULT_CHUNK_SIZE_MB MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR
        "NNHAUL_DEFAULT_CHUNK_SIZE_MB='${NNHAUL_DEFAULT_CHUNK_SIZE_MB}' must "
        "be a positive integer number of MiB")
endif()
math(EXPR _nnhaul_chunk_minus_one "${NNHAUL_DEFAULT_CHUNK_SIZE_MB} - 1")
math(EXPR _nnhaul_chunk_not_pow2 "${NNHAUL_DEFAULT_CHUNK_SIZE_MB} & ${_nnhaul_chunk_minus_one}")
if(NOT _nnhaul_chunk_not_pow2 EQUAL 0)
    message(FATAL_ERROR
        "NNHAUL_DEFAULT_CHUNK_SIZE_MB=${NNHAUL_DEFAULT_CHUNK_SIZE_MB} is not "
        "a power of two; the chunk-addressing path is shift/mask and "
        "requires one")
endif()
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/defs.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/nnhaul/defs.h"
    @ONLY
)
# The generated header must be visible to every in-tree consumer,
# including the wrapper tests that include nnhaul/nnhaul.hh through
# nntile's harness without linking the nnhaul target directly.
# Copy the public headers into the build tree so NNHAUL_ROOT may point at
# a build directory (not only at an install prefix).
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/include/nnhaul"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/include")
include_directories("${CMAKE_CURRENT_BINARY_DIR}/include")

if(NNHAUL_IS_TOP_LEVEL)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# When a parent (NNTile) pulls NNHaul in via add_subdirectory the tests
# default off and only the nnhaul::nnhaul target is contributed; install
# rules follow NNHAUL_INSTALL.
option(NNHAUL_BUILD_TESTS "Build the nnhaul test suites" ${NNHAUL_IS_TOP_LEVEL})
option(NNHAUL_INSTALL "Install and export the nnhaul package" ${NNHAUL_IS_TOP_LEVEL})
option(NNHAUL_BUILD_EXAMPLES "Build the public-API example planner"
    ${NNHAUL_BUILD_TESTS})

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

include(GNUInstallDirs)

set(_nnhaul_cudnn_hints
    "$ENV{CUDNN_ROOT}/include"
    "$ENV{CUDNN_INCLUDE_PATH}"
    "$ENV{CUDNN_PATH}/include"
    "${CUDAToolkit_ROOT}/include"
    "${CUDAToolkit_LIBRARY_ROOT}/include"
)
find_path(
    CUDNN_INCLUDE_DIR
    cudnn.h
    HINTS ${_nnhaul_cudnn_hints}
)
set(_nnhaul_cudnn_libs
    "$ENV{CUDNN_ROOT}/lib"
    "$ENV{CUDNN_LIBRARY_PATH}"
    "$ENV{CUDNN_PATH}/lib"
    "${CUDAToolkit_ROOT}/lib"
)
find_library(
    CUDNN_LIBRARY
    NAMES cudnn
    HINTS ${_nnhaul_cudnn_libs}
)
if(NOT CUDNN_LIBRARY)
    # Linux sonames first (the declared target), then the macOS and
    # Windows names so a toolkit-local cuDNN wins on those too.
    find_file(
        CUDNN_LIBRARY
        NAMES libcudnn.so.9 libcudnn.so
            libcudnn.9.dylib libcudnn.dylib
            cudnn64_9.dll cudnn.lib
        HINTS ${_nnhaul_cudnn_libs}
    )
endif()
if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(
        FATAL_ERROR
        "cuDNN is required (cudnn.h and libcudnn). "
        "Set CUDNN_ROOT, CUDNN_INCLUDE_DIR, and CUDNN_LIBRARY."
    )
endif()
get_filename_component(CUDNN_LIB_DIR "${CUDNN_LIBRARY}" DIRECTORY)
if(NNHAUL_IS_TOP_LEVEL)
    set(CMAKE_BUILD_RPATH "${CUDAToolkit_LIBRARY_DIR};${CUDNN_LIB_DIR}")
endif()

# cuDNN as an imported target so the export stays relocatable: the
# installed nnhaulConfig.cmake re-finds cuDNN and defines the same
# target before loading nnhaulTargets.cmake.
if(NOT TARGET nnhaul::cudnn)
    add_library(nnhaul::cudnn UNKNOWN IMPORTED)
    set_target_properties(
        nnhaul::cudnn
        PROPERTIES
            IMPORTED_LOCATION "${CUDNN_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}"
    )
endif()

# The chunk-graph executor (the plan format): the
# one engine of this tree. Sources are flat under src/ in dependency
# order — pools and the binding table, the inventory, the plan and its
# checker, the derivation, the timing pass, the host worker and the
# tracer, the executor, the planner, the CUDA backend, then the engine
# and the public API over them.
set(NNHAUL_LIBRARY_TYPE "STATIC" CACHE STRING
    "nnhaul library type (STATIC standalone; SHARED when embedded in nntile)")
add_library(nnhaul ${NNHAUL_LIBRARY_TYPE}
    src/host_pool.cc
    src/binding_table.cc
    src/inventory.cc
    src/plan.cc
    src/plan_checker.cc
    src/dependency.cc
    src/timing.cc
    src/host_worker.cc
    src/plan_trace.cc
    src/plan_executor.cc
    src/lru_planner.cc
    src/planner_api.cc
    src/cuda_backend.cc
    src/engine.cc
    src/api.cc
)
set_target_properties(nnhaul PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(NNHAUL_LIBRARY_TYPE STREQUAL "SHARED")
    target_compile_definitions(nnhaul PRIVATE NNHAUL_BUILT_SHARED=1)
    if(UNIX AND NOT APPLE)
        # Bind the engine's own references inside the DSO: a preloaded copy
        # of another nnhaul must not interpose on its internal symbols.
        target_link_options(nnhaul PRIVATE "-Wl,-Bsymbolic-functions")
    endif()
endif()
target_link_libraries(nnhaul PRIVATE ${CMAKE_DL_LIBS})

# The example planner is the SDK's executable specification: it links only
# nnhaul and includes only public headers. It is the same artifact a third
# party ships.
if(NNHAUL_BUILD_EXAMPLES)
    add_library(nnhaul_example_greedy_planner SHARED
        examples/planner/greedy_planner.cc)
    target_link_libraries(nnhaul_example_greedy_planner PRIVATE nnhaul)
    set_target_properties(nnhaul_example_greedy_planner PROPERTIES
        OUTPUT_NAME "greedy_planner"
        PREFIX "")
endif()
target_compile_features(nnhaul PUBLIC cxx_std_17)
add_library(nnhaul::nnhaul ALIAS nnhaul)
target_include_directories(
    nnhaul
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(
    nnhaul
    PUBLIC
        Threads::Threads
        CUDA::cudart
        CUDA::cublas
        nnhaul::cudnn
)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(
        nnhaul
        PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall>
            $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
            $<$<COMPILE_LANGUAGE:CXX>:-Werror>
            $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>
    )
endif()

# The trace metadata block reports the runtime version; PRIVATE
# because only nnhaul's own sources read it.
target_compile_definitions(nnhaul PRIVATE NNHAUL_VERSION="${PROJECT_VERSION}")

# ---- install / export ----

if(NNHAUL_INSTALL)

include(CMakePackageConfigHelpers)

set(NNHAUL_INSTALL_CMAKEDIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/nnhaul"
    CACHE PATH "CMake package config location for nnhaul")

install(
    TARGETS nnhaul
    EXPORT nnhaulTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/nnhaul"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/include/nnhaul/defs.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/nnhaul"
)

install(
    EXPORT nnhaulTargets
    FILE nnhaulTargets.cmake
    NAMESPACE nnhaul::
    DESTINATION "${NNHAUL_INSTALL_CMAKEDIR}"
)

# The build tree is also usable as NNHAUL_ROOT: nnhaulConfig.cmake is
# generated here, so the targets file must be here too. Without it a
# consumer pointing NNHAUL_ROOT at this directory fails at include().
export(
    EXPORT nnhaulTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/nnhaulTargets.cmake"
    NAMESPACE nnhaul::
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/nnhaulConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/nnhaulConfig.cmake"
    INSTALL_DESTINATION "${NNHAUL_INSTALL_CMAKEDIR}"
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/nnhaulConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/nnhaulConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/nnhaulConfigVersion.cmake"
    DESTINATION "${NNHAUL_INSTALL_CMAKEDIR}"
)

endif() # NNHAUL_INSTALL

# ---- tests ----

if(NNHAUL_BUILD_TESTS)
    if(NNHAUL_IS_TOP_LEVEL)
        enable_testing()
    endif()

    function(nnhaul_test name source)
        set(_nnhaul_test_target "nnhaul_${name}")
        add_executable(${_nnhaul_test_target} ${source})
        target_link_libraries(${_nnhaul_test_target} PRIVATE nnhaul)
        target_include_directories(
            ${_nnhaul_test_target}
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/tests
        )
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(
                ${_nnhaul_test_target}
                PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                    $<$<COMPILE_LANGUAGE:CXX>:-Werror>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
                    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
                    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror,-Wno-pedantic>
            )
        else()
            target_compile_options(
                ${_nnhaul_test_target}
                PRIVATE
                    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
            )
        endif()
        add_test(NAME ${_nnhaul_test_target} COMMAND ${_nnhaul_test_target})
        set_tests_properties(${_nnhaul_test_target} PROPERTIES TIMEOUT 60)
    endfunction()

    # The unit suites over the engine's own types.
    nnhaul_test(host_pool tests/test_host_pool.cc)
    nnhaul_test(binding_table tests/test_binding_table.cc)
    nnhaul_test(plan_checker tests/test_plan_checker.cc)
    nnhaul_test(dependency tests/test_dependency.cc)
    nnhaul_test(timing tests/test_timing.cc)
    nnhaul_test(plan_trace tests/test_plan_trace.cc)
    nnhaul_test(plan_executor tests/test_plan_executor.cu)
    nnhaul_test(lru_planner tests/test_lru_planner.cc)

    # Retained runtime units  beside the engine: the
    # worker resolution and setters, the arena cap precedence, the
    # footprint key, and the tracer's knob/lane surfaces adapted to
    # the chunk-graph engine (the old subjects that died with it —
    # chunked A/B, the plan engine, the old pipeline, the chunk
    # manager, the C-counter memory trace — are deleted with their
    # sources).
    nnhaul_test(default_worker tests/test_default_worker.cc)
    nnhaul_test(gpu_cap tests/test_gpu_cap.cc)
    nnhaul_test(footprint tests/test_footprint.cc)
    nnhaul_test(trace_knobs tests/test_trace_knobs.cc)
    nnhaul_test(trace_lanes tests/test_trace_lanes.cc)

    # The multi-device integration suite : the six named
    # cases over the public API, the checker, and the executor,
    # skipping loudly where the host lacks the devices a case needs.
    nnhaul_test(multi_device tests/test_multi_device.cu)

    # The backend conformance suite: one case per backend rule over
    # real lanes, the canaries harness-planted and read back at
    # barriers. Case 4
    # (the static no-readback audit) reads the engine plane's sources
    # at test time, so the source tree rides in as a define.
    nnhaul_test(backend_conformance tests/test_backend_conformance.cu)
    add_library(nnhaul_test_bad_planner SHARED tests/bad_planner.cc)
    target_link_libraries(nnhaul_test_bad_planner PRIVATE nnhaul)
    set_target_properties(nnhaul_test_bad_planner PROPERTIES
        OUTPUT_NAME "bad_planner" PREFIX "")
    nnhaul_test(planner_version tests/test_planner_version.cc)
    target_link_libraries(nnhaul_planner_version PRIVATE ${CMAKE_DL_LIBS})
    target_compile_definitions(nnhaul_planner_version PRIVATE
        NNHAUL_BAD_PLANNER="$<TARGET_FILE:nnhaul_test_bad_planner>")
    add_dependencies(nnhaul_planner_version nnhaul_test_bad_planner)

    if(TARGET nnhaul_example_greedy_planner)
        nnhaul_test(planner_plugin tests/test_planner_plugin.cc)
        target_include_directories(nnhaul_planner_plugin PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(nnhaul_planner_plugin PRIVATE ${CMAKE_DL_LIBS})
        target_compile_definitions(nnhaul_planner_plugin PRIVATE
            NNHAUL_EXAMPLE_PLANNER="$<TARGET_FILE:nnhaul_example_greedy_planner>")
        add_dependencies(nnhaul_planner_plugin nnhaul_example_greedy_planner)
    endif()
    target_compile_definitions(nnhaul_backend_conformance PRIVATE
        NNHAUL_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )

    # The engine's own suites: the smoke test (the end-to-end proof),
    # the depth-2 pipeline test (issue without join, the live-lane late
    # record, the loud
    # third-batch abort), and the old runtime tests whose pinned
    # behaviors survive the new design unmodified (test_cap_throw's
    # census message strings carry the new caller-defect wording).
    nnhaul_test(engine_smoke tests/test_engine_smoke.cc)
    # The same batch, planned by the shipped external planner through
    # NNHAUL_PLANNER: proves the engine accepts and executes a plugin's
    # plan end to end, not only the checker's verdict.
    if(TARGET nnhaul_example_greedy_planner)
        add_test(NAME nnhaul_engine_external_planner
            COMMAND $<TARGET_FILE:nnhaul_engine_smoke>)
        set_tests_properties(nnhaul_engine_external_planner PROPERTIES
            TIMEOUT 60
            ENVIRONMENT "NNHAUL_PLANNER=$<TARGET_FILE:nnhaul_example_greedy_planner>")
    endif()
    nnhaul_test(pipeline_depth2 tests/test_pipeline_depth2.cc)
    nnhaul_test(cross_batch_survival tests/test_cross_batch_survival.cc)
    nnhaul_test(env_robustness tests/test_env_robustness.cc)
    nnhaul_test(runtime_cycles tests/test_runtime_cycles.cc)
    nnhaul_test(sgemm tests/test_sgemm.cc)
    nnhaul_test(cpu_gpu_copy tests/test_cpu_gpu_copy.cc)
    nnhaul_test(queue_admission tests/test_queue_admission.cc)
    nnhaul_test(dtd_versions tests/test_dtd_versions.cc)
    nnhaul_test(cap_throw tests/test_cap_throw.cc)

    # Python-driven trace tests over the format-1 dump: the analyzer
    # tooling's acceptance suite over a hand-computed fixture, and the
    # end-to-end pipeline (the trace_lanes workload's dump, validated
    # and cross-checked against its test_hooks sidecar).
    find_program(NNHAUL_PYTHON3 python3)
    if(NNHAUL_PYTHON3)
        add_test(
            NAME trace_analyzer
            COMMAND ${NNHAUL_PYTHON3}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/trace_analyzer_test.py
        )
        set_tests_properties(trace_analyzer PROPERTIES TIMEOUT 60)
        add_test(
            NAME trace_pipeline
            COMMAND ${NNHAUL_PYTHON3}
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/trace_pipeline_test.py
                $<TARGET_FILE:nnhaul_trace_lanes>
                ${CMAKE_CURRENT_SOURCE_DIR}
        )
        set_tests_properties(trace_pipeline PROPERTIES TIMEOUT 120)
    endif()

endif()
