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

#add_subdirectory(googletest)
#add_subdirectory(nvbench)
add_subdirectory(pybind11)

# Configure CUTLASS
set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "")
set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "")
set(CUTLASS_ENABLE_PROFILER OFF CACHE BOOL "")
set(CUTLASS_ENABLE_LIBRARY OFF CACHE BOOL "")
set(CUTLASS_ENABLE_CUDA_NATIVE_BFLOAT16 ON CACHE BOOL "")
add_subdirectory(cutlass)

#FetchContent_MakeAvailable(googletest nvbench pybind11)
FetchContent_MakeAvailable(pybind11 cutlass)

# Add CUTLASS include directories and compile definitions
target_include_directories(nntile PUBLIC
    ${cutlass_SOURCE_DIR}/include
)

target_compile_definitions(nntile PUBLIC
    CUTLASS_ENABLE_TENSOR_CORE_MMA=1
    CUTLASS_USE_CUDA_TOOLKIT=1
)

# Append external sources and headers.
target_sources(nntile PRIVATE "random.h")
