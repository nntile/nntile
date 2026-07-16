# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file cmake/NntileCatch2.cmake
# Prefer an installed Catch2 (CI prefix / CMAKE_PREFIX_PATH), else FetchContent.
# GLOBAL: find_package in external/ must expose targets to nntile/tests/.
#
# Pin: keep NNTILE_CATCH2_GIT_TAG in sync with
# .github/actions/nntile-catch2-setup (default v3.11.0).

if(TARGET Catch2::Catch2WithMain)
    return()
endif()

set(NNTILE_CATCH2_GIT_TAG "v3.11.0" CACHE STRING
    "Catch2 git tag when FetchContent is used")

find_package(Catch2 3.11 CONFIG QUIET GLOBAL)
if(Catch2_FOUND)
    message(STATUS "Using installed Catch2 ${Catch2_VERSION}")
else()
    message(STATUS
        "Catch2 not found in CMAKE_PREFIX_PATH; "
        "FetchContent ${NNTILE_CATCH2_GIT_TAG}")
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG ${NNTILE_CATCH2_GIT_TAG}
    )
    FetchContent_MakeAvailable(Catch2)
endif()

if(NOT COMMAND catch_discover_tests)
    # FetchContent: extras live next to the source tree.
    if(DEFINED Catch2_SOURCE_DIR
            AND EXISTS "${Catch2_SOURCE_DIR}/extras/Catch.cmake")
        list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
    endif()
    # Installed Catch2: Catch2Config appends its cmake dir (has Catch.cmake).
    include(Catch)
endif()
