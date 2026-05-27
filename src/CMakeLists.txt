# @file src/CMakeLists.txt
if(BUILD_CORE)
    add_subdirectory(core)
endif()
if(BUILD_GRAPH)
    add_subdirectory(graph)
endif()
