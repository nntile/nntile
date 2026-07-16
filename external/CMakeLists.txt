# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file external/CMakeLists.txt
# External header libraries to be used
#
# @version 1.1.0

include(FetchContent)

if(BUILD_TESTS)
    include(NntileCatch2)
endif()

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)
