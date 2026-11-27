# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems with GPU acceleration.
#
# @file nntile/tests/nnhaul/CMakeLists.txt
# Kernel-versus-wrapper comparisons on the NNHaul backend. These test the
# nntile side of the integration (nntile + nnhaul), not NNHaul alone.
#
# @version 1.1.0

function(nnhaul_wrapper_test name)
    set(target "tests_nnhaul_${name}")
    add_executable(${target} "${CMAKE_CURRENT_SOURCE_DIR}/${name}.cc")
    target_link_libraries(${target} PRIVATE nntile::nntile CUDA::cudart
        CUDA::cublas)
    target_include_directories(${target} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror -Wpedantic>)
    endif()
    add_test(NAME ${target} COMMAND ${target})
    set_tests_properties(${target} PROPERTIES
        TIMEOUT 120
        LABELS "NNHAUL;SUBSYSTEM_KERNEL")
endfunction()

set(NNHAUL_WRAPPER_TESTS
    "add"
    "add_inplace"
    "add_slice_inplace"
    "clear"
    "copy"
    "fill"
    "gelu"
    "gelu_inplace"
    "gelutanh"
    "gelutanh_inplace"
    "hypot"
    "hypot_inplace"
    "mask_scalar"
    "maxsumexp"
    "multiply"
    "multiply_inplace"
    "norm"
    "norm_fiber"
    "norm_fiber_inplace"
    "norm_slice_inplace"
    "pow"
    "relu"
    "relu_inplace"
    "scale"
    "scale_inplace"
    "silu"
    "silu_inplace"
    "softmax_inplace"
    "sqrt"
    "sqrt_inplace"
    "subcopy"
    "sum"
    "sum_slice"
    "sumprod_slice"
    "swap_two_axes"
    "transpose"
)

nntile_filter_disabled_kernel_tests(NNHAUL_WRAPPER_TESTS)

foreach(name IN LISTS NNHAUL_WRAPPER_TESTS)
    nnhaul_wrapper_test(${name})
endforeach()
