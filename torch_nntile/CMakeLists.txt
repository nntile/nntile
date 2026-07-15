# @copyright (c) 2026-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#
# @file torch_nntile/CMakeLists.txt
# Optional libtorch_nntile (LibTorch PrivateUse1 + models).

option(BUILD_TORCH_NNTILE
    "Build libtorch_nntile (requires LibTorch)" OFF)

if(NOT BUILD_TORCH_NNTILE)
    return()
endif()

if(NOT TARGET nntile)
    message(FATAL_ERROR "BUILD_TORCH_NNTILE requires libnntile")
endif()

find_package(Torch REQUIRED)

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
    csrc/nntile_mul.cpp
    csrc/nntile_cat.cpp
    csrc/nntile_narrow.cpp
    csrc/nntile_split.cpp
    csrc/nntile_hypot.cpp
    csrc/nntile_linear.cpp
    csrc/nntile_relu.cpp
    csrc/nntile_threshold_backward.cpp
    csrc/nntile_silu.cpp
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
    csrc/models/llama.cpp
    csrc/models/bert.cpp
    csrc/models/t5.cpp
)

add_library(torch_nntile ${TORCH_NNTILE_CSRC})
add_library(torch_nntile::torch_nntile ALIAS torch_nntile)

target_compile_definitions(torch_nntile PUBLIC TORCH_NNTILE_USE_LIBNNTILE=1)
target_include_directories(torch_nntile
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
    PRIVATE
        "${PROJECT_SOURCE_DIR}/nntile/include"
        "${PROJECT_BINARY_DIR}/include"
)
target_link_libraries(torch_nntile
    PUBLIC
        nntile
        ${TORCH_LIBRARIES}
)
set_property(TARGET torch_nntile PROPERTY CXX_STANDARD 17)

if(BUILD_TORCH_NNTILE_EXAMPLES)
    add_subdirectory(examples/cpp)
endif()

install(
    TARGETS torch_nntile
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/torch_nntile/"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/torch_nntile
    FILES_MATCHING PATTERN "*.hh" PATTERN "*.h"
)
