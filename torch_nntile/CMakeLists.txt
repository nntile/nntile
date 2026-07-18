# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/CMakeLists.txt
# libtorch_nntile (LibTorch PrivateUse1 + models).

option(BUILD_LIBTORCH_NNTILE
    "Build libtorch_nntile (requires LibTorch)" OFF)

if(NOT BUILD_LIBTORCH_NNTILE)
    return()
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(NOT TARGET nntile AND NOT TARGET nntile::nntile)
    find_package(nntile REQUIRED CONFIG)
endif()

if(NOT Torch_FOUND)
    find_package(Torch REQUIRED)
    message(STATUS "LibTorch found: ${TORCH_INSTALL_PREFIX}")
endif()

if(NNTILE_TORCH_NATIVE_OPS)
    message(STATUS
        "libtorch_nntile: NNTILE_TORCH_NATIVE_OPS "
        "(PrivateUse1 aten ops via torch-native StarPU path)")
    set(TORCH_NNTILE_CSRC
        csrc/nntile_allocator.cpp
        csrc/nntile_tensor_gc.cpp
        csrc/nntile_tensor_meta.cpp
        csrc/nntile_kernels.cpp
        csrc/nntile_guard.cpp
        csrc/nntile_generator.cpp
        csrc/nntile_hooks.cpp
        csrc/nntile_context.cpp
        csrc/nntile_graph_recorder.cpp
        csrc/nntile_gemm_layout.cpp
        csrc/nntile_executor_torch_native.cpp
        csrc/nntile_add.cpp
        csrc/nntile_sub.cpp
        csrc/nntile_mul.cpp
        csrc/nntile_host_aten.cpp
        csrc/nntile_cat.cpp
        csrc/nntile_narrow.cpp
        csrc/nntile_split.cpp
        csrc/nntile_hypot.cpp
        csrc/nntile_linear.cpp
        csrc/nntile_relu.cpp
        csrc/nntile_threshold_backward.cpp
        csrc/nntile_silu.cpp
        csrc/nntile_trig.cpp
        csrc/nntile_silu_backward.cpp
        csrc/nntile_gelu.cpp
        csrc/nntile_gelu_backward.cpp
        csrc/nntile_softmax.cpp
        csrc/nntile_softmax_backward.cpp
        csrc/nntile_log_softmax.cpp
        csrc/nntile_nll_loss.cpp
        csrc/nntile_avg_pool2d.cpp
        csrc/nntile_adaptive_avg_pool2d.cpp
        csrc/nntile_max_pool2d.cpp
        csrc/nntile_convolution.cpp
        csrc/nntile_batch_norm.cpp
        csrc/nntile_mm.cpp
        csrc/nntile_bmm.cpp
        csrc/nntile_addmm.cpp
        csrc/nntile_sum.cpp
        csrc/nntile_gemm.cpp
        csrc/nntile_layer_norm.cpp
        csrc/nntile_norm.cpp
        csrc/nntile_broadcast_torch_native.cpp
        csrc/nntile_repeat.cpp
        csrc/nntile_embedding.cpp
        csrc/nntile_sdpa.cpp
        csrc/nntile_sdpa_aten.cpp
        csrc/nntile_transpose.cpp
    )
else()
    set(TORCH_NNTILE_CSRC
        csrc/nntile_allocator.cpp
        csrc/nntile_tensor_gc.cpp
        csrc/nntile_tensor_meta.cpp
        csrc/nntile_kernels.cpp
        csrc/nntile_guard.cpp
        csrc/nntile_generator.cpp
        csrc/nntile_hooks.cpp
        csrc/nntile_context.cpp
        csrc/nntile_graph_recorder.cpp
        csrc/nntile_gemm_layout.cpp
        csrc/nntile_gemm.cpp
        csrc/nntile_executor.cpp
        csrc/nntile_add.cpp
        csrc/nntile_add_fiber.cpp
        csrc/nntile_sum_slice.cpp
        csrc/nntile_mul.cpp
        csrc/nntile_cat.cpp
        csrc/nntile_narrow.cpp
        csrc/nntile_split.cpp
        csrc/nntile_hypot.cpp
        csrc/nntile_linear.cpp
        csrc/nntile_relu.cpp
        csrc/nntile_threshold_backward.cpp
        csrc/nntile_silu.cpp
        csrc/nntile_trig.cpp
        csrc/nntile_silu_backward.cpp
        csrc/nntile_gelu.cpp
        csrc/nntile_gelu_backward.cpp
        csrc/nntile_softmax.cpp
        csrc/nntile_softmax_backward.cpp
        csrc/nntile_mm.cpp
        csrc/nntile_bmm.cpp
        csrc/nntile_addmm.cpp
        csrc/nntile_sum.cpp
        csrc/nntile_mm_backward.cpp
        csrc/nntile_cross_entropy.cpp
        csrc/nntile_sgd_step.cpp
        csrc/nntile_adam_step.cpp
        csrc/nntile_layer_norm.cpp
        csrc/nntile_rms_norm.cpp
        csrc/nntile_norm.cpp
        csrc/nntile_broadcast.cpp
        csrc/nntile_repeat.cpp
        csrc/nntile_embedding.cpp
        csrc/nntile_sdpa.cpp
        csrc/nntile_sdpa_aten.cpp
        csrc/nntile_transpose.cpp
        csrc/nntile_rope.cpp
        csrc/nntile_mse_loss.cpp
        csrc/models/deep_relu.cpp
        csrc/models/gpt2.cpp
        csrc/models/gpt_neo.cpp
        csrc/models/gpt_neox.cpp
        csrc/models/llama.cpp
        csrc/models/bert.cpp
        csrc/models/roberta.cpp
        csrc/models/t5.cpp
        csrc/models/mlp_mixer.cpp
    )
endif()

add_library(torch_nntile SHARED ${TORCH_NNTILE_CSRC})
add_library(torch_nntile::torch_nntile ALIAS torch_nntile)

target_include_directories(torch_nntile
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/csrc>"
)
# In-tree `nntile` is a different export set; record nntile::nntile for
# install(EXPORT) and resolve via find_dependency(nntile) in Config.cmake.
if(TARGET nntile)
    target_link_libraries(torch_nntile PUBLIC
        "$<BUILD_INTERFACE:nntile>"
        "$<INSTALL_INTERFACE:nntile::nntile>"
    )
else()
    target_link_libraries(torch_nntile PUBLIC nntile::nntile)
endif()
target_link_libraries(torch_nntile PUBLIC ${TORCH_LIBRARIES})
set_property(TARGET torch_nntile PROPERTY CXX_STANDARD 17)

# Expose whether linked libnntile was built with CUDA (for consumers + wheels).
if(DEFINED NNTILE_BUILT_WITH_CUDA)
    set(TORCH_NNTILE_BUILT_WITH_CUDA "${NNTILE_BUILT_WITH_CUDA}")
elseif(NNTILE_USE_CUDA OR NNTILE_LINK_CUDA)
    set(TORCH_NNTILE_BUILT_WITH_CUDA ON)
else()
    set(TORCH_NNTILE_BUILT_WITH_CUDA OFF)
endif()
message(STATUS "libtorch_nntile CUDA: ${TORCH_NNTILE_BUILT_WITH_CUDA}")

if(BUILD_LIBTORCH_NNTILE_EXAMPLES)
    add_subdirectory(examples/cpp)
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/TorchNntileWheel.cmake")

set(_torch_nntile_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/torch_nntile")

install(
    TARGETS torch_nntile
    EXPORT torch_nntileTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/torch_nntile/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/torch_nntile
    FILES_MATCHING PATTERN "*.hh" PATTERN "*.h"
)

install(
    EXPORT torch_nntileTargets
    FILE torch_nntileTargets.cmake
    NAMESPACE torch_nntile::
    DESTINATION ${_torch_nntile_cmake_dir}
)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/torch_nntileConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/torch_nntileConfig.cmake"
    INSTALL_DESTINATION ${_torch_nntile_cmake_dir}
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/torch_nntileConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/torch_nntileConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/torch_nntileConfigVersion.cmake"
    DESTINATION ${_torch_nntile_cmake_dir}
)
