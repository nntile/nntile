# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file external/CMakeLists.txt
# External header libraires to be used
#
# @version 1.1.0

include(FetchContent)

# CI: build-test-prerequisites populates build/_deps; test jobs pass
# -DNNTILE_FETCHCONTENT_DISCONNECTED=ON to skip re-downloading Catch2/json.
if(NNTILE_FETCHCONTENT_DISCONNECTED)
    set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL "" FORCE)
endif()

#add_subdirectory(googletest)
#add_subdirectory(nvbench)
if(BUILD_PYTHON_WRAPPERS)
    add_subdirectory(pybind11)
    FetchContent_MakeAvailable(pybind11)
endif()

set(_nntile_fetch_catch2 ON)
if(NNTILE_COMPILE_CHECK_TESTS_SUBSYSTEM AND NNTILE_FETCHCONTENT_DISCONNECTED)
    # compile-check-tests: Catch2 headers come from build/_deps artifact only.
    set(_nntile_fetch_catch2 OFF)
endif()

if(_nntile_fetch_catch2)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.11.0
    )
    FetchContent_MakeAvailable(Catch2)
endif()

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

if(_nntile_fetch_catch2)
    add_custom_target(nntile_test_prerequisites
        DEPENDS Catch2 Catch2WithMain
        COMMENT "Catch2 libraries for NNTile tests (CI cache seed)")
endif()

#add_subdirectory(cudnn_frontend)

#include(GoogleTest)

# Append external sources and headers (not for IMPORTED prebuilt nntile).
if(TARGET nntile)
    get_target_property(_nntile_imported nntile IMPORTED)
    if(NOT _nntile_imported)
        target_sources(nntile PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/random.h")
    endif()
endif()
