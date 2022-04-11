# Set minimum CMake version to avoid configuration problems
cmake_minimum_required(VERSION 3.16.3)

# Define the project
project(NNTile VERSION 0.0.1 LANGUAGES CXX)

# List of options
# Build shared libraries instead of static
option(BUILD_SHARED_LIBS "Build shared libraries instead of static" ON)
option(USE_CUDA "Use CUDA toolkit" ON)
option(USE_CBLAS "Use CPU BLAS" ON)
option(BUILD_TESTS "Build tests" ON)

# Specify the library
add_library(nntile)

# Set C++ standard for all sources
#set_property(TARGET nntile PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check if CUDA is available
set(NNTILE_USE_CUDA OFF)
if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        # Get cublas
        find_package(CUDAToolkit >=10.1)
        target_link_libraries(nntile PRIVATE CUDA::cublas)
        set(NNTILE_USE_CUDA ON)
    endif()
endif()

# Get CBLAS
set(NNTILE_USE_CBLAS OFF)
if(USE_CBLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        target_link_libraries(nntile PRIVATE BLAS::BLAS)
        set(NNTILE_USE_CBLAS ON)
    endif()
endif()

# Get the pkg-config
find_package(PkgConfig REQUIRED)

# Get the StarPU-1.3
pkg_check_modules(StarPU REQUIRED starpu-1.3)
target_link_libraries(nntile PUBLIC ${StarPU_LDFLAGS})
target_include_directories(nntile PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    ${StarPU_INCLUDE_DIRS}
    )

# Configure list of definitions
configure_file("include/nntile/defs.h.in"
    "${CMAKE_BINARY_DIR}/include/nntile/defs.h" @ONLY)

# Add directory with the sources
add_subdirectory("src")

# Add sirectory with headers
add_subdirectory("include")

# Add external sources/headers
add_subdirectory("external")
target_include_directories(nntile PRIVATE "${CMAKE_SOURCE_DIR}/external")

# Add subdirectory with examples
#add_subdirectory("examples")

# Add subdirectory with tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory("tests")
endif()

