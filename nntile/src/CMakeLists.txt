# @file nntile/src/CMakeLists.txt
# Unified sources for target nntile (generated from legacy core/graph lists)

# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems based on StarPU runtime system.
#
# @file src/CMakeLists.txt
# Set NNTile sources to compile
#
# @version 1.1.0

# Base of the NNTile sources
set(BASE_SRC "context.cc")

# Ignore low-level kernel functions in case of SimGrid simulation
if(NOT HAVE_STARPU_SIMGRID)
    # Generic low-level CPU kernels
    set(KERNEL_CPU_SRC
        "kernel/accumulate_maxsumexp/cpu.cc"
        "kernel/accumulate_attn_output/cpu.cc"
        "kernel/adam_step/cpu.cc"
        "kernel/adamw_step/cpu.cc"
        "kernel/sgd_step/cpu.cc"
        "kernel/add/cpu.cc"
        "kernel/add_fiber/cpu.cc"
        "kernel/add_fiber_inplace/cpu.cc"
        "kernel/scale_fiber/cpu.cc"
        "kernel/add_inplace/cpu.cc"
        "kernel/add_slice/cpu.cc"
        "kernel/add_slice_inplace/cpu.cc"
        "kernel/cblas.cc"
        "kernel/conv2d_bwd_input_inplace/cpu.cc"
        "kernel/conv2d_bwd_weight_inplace/cpu.cc"
        "kernel/conv2d_inplace/cpu.cc"
        "kernel/embedding/cpu.cc"
        "kernel/embedding_backward/cpu.cc"
        "kernel/fill/cpu.cc"
        "kernel/gelu/cpu.cc"
        "kernel/gelu_inplace/cpu.cc"
        "kernel/gelu_backward/cpu.cc"
        "kernel/gelutanh/cpu.cc"
        "kernel/gelutanh_backward/cpu.cc"
        "kernel/gelutanh_inplace/cpu.cc"
        "kernel/hypot_inplace/cpu.cc"
        "kernel/hypot/cpu.cc"
        "kernel/hypot_scalar_inverse/cpu.cc"
        "kernel/logsumexp/cpu.cc"
        "kernel/mask_scalar/cpu.cc"
        "kernel/maxsumexp/cpu.cc"
        "kernel/norm/cpu.cc"
        "kernel/norm_fiber/cpu.cc"
        "kernel/norm_fiber_inplace/cpu.cc"
        "kernel/norm_slice_inplace/cpu.cc"
        "kernel/norm_slice/cpu.cc"
        "kernel/pow/cpu.cc"
        "kernel/multiply/cpu.cc"
        "kernel/multiply_fiber_inplace/cpu.cc"
        "kernel/multiply_fiber/cpu.cc"
        "kernel/multiply_inplace/cpu.cc"
        "kernel/multiply_slice/cpu.cc"
        "kernel/randn/cpu.cc"
        "kernel/relu_inplace/cpu.cc"
        "kernel/relu_backward/cpu.cc"
        "kernel/relu/cpu.cc"
        "kernel/rope/cpu.cc"
        "kernel/rope_backward/cpu.cc"
        "kernel/scale/cpu.cc"
        "kernel/scale_inplace/cpu.cc"
        "kernel/scale_slice/cpu.cc"
        "kernel/silu_backward/cpu.cc"
        "kernel/silu/cpu.cc"
        "kernel/silu_inplace/cpu.cc"
        "kernel/softmax/cpu.cc"
        "kernel/softmax_inplace/cpu.cc"
        "kernel/sqrt/cpu.cc"
        "kernel/sqrt_inplace/cpu.cc"
        "kernel/subcopy/cpu.cc"
        "kernel/subtract_indexed_outputs/cpu.cc"
        "kernel/sum_fiber/cpu.cc"
        "kernel/sum_slice/cpu.cc"
        "kernel/sum/cpu.cc"
        "kernel/sumprod_fiber/cpu.cc"
        "kernel/sumprod_slice/cpu.cc"
        "kernel/total_sum_accum/cpu.cc"
        "kernel/transpose/cpu.cc"
        "kernel/swap_two_axes/cpu.cc"
        "kernel/isfinite/cpu.cc"
    )

    # CUDA-specific low-level kernels
    if(NNTILE_USE_CUDA)
        set(KERNEL_CUDA_SRC
            "kernel/accumulate_maxsumexp/cuda.cu"
            "kernel/accumulate_attn_output/cuda.cu"
            "kernel/adam_step/cuda.cu"
            "kernel/adamw_step/cuda.cu"
            "kernel/sgd_step/cuda.cu"
            "kernel/add/cuda.cu"
            "kernel/add_fiber/cuda.cu"
            "kernel/add_fiber_inplace/cuda.cu"
            "kernel/scale_fiber/cuda.cu"
            "kernel/add_inplace/cuda.cu"
            "kernel/add_slice/cuda.cu"
            "kernel/add_slice_inplace/cuda.cu"
            "kernel/cublas.cc"
            "kernel/conv2d_bwd_input_inplace/cuda.cu"
            "kernel/conv2d_bwd_weight_inplace/cuda.cu"
            "kernel/conv2d_inplace/cuda.cu"
            "kernel/embedding/cuda.cu"
            "kernel/embedding_backward/cuda.cu"
            "kernel/fill/cuda.cu"
            "kernel/gelu/cuda.cu"
            "kernel/gelu_inplace/cuda.cu"
            "kernel/gelu_backward/cuda.cu"
            "kernel/gelutanh/cuda.cu"
            "kernel/gelutanh_backward/cuda.cu"
            "kernel/gelutanh_inplace/cuda.cu"
            "kernel/hypot_inplace/cuda.cu"
            "kernel/hypot/cuda.cu"
            "kernel/hypot_scalar_inverse/cuda.cu"
            "kernel/logsumexp/cuda.cu"
            "kernel/mask_scalar/cuda.cu"
            "kernel/maxsumexp/cuda.cu"
            "kernel/norm/cuda.cu"
            "kernel/norm_fiber/cuda.cu"
            "kernel/norm_fiber_inplace/cuda.cu"
            "kernel/norm_slice_inplace/cuda.cu"
            "kernel/norm_slice/cuda.cu"
            "kernel/pow/cuda.cu"
            "kernel/multiply/cuda.cu"
            "kernel/multiply_fiber_inplace/cuda.cu"
            "kernel/multiply_fiber/cuda.cu"
            "kernel/multiply_inplace/cuda.cu"
            "kernel/multiply_slice/cuda.cu"
            "kernel/relu_inplace/cuda.cu"
            "kernel/relu_backward/cuda.cu"
            "kernel/relu/cuda.cu"
            "kernel/rope/cuda.cu"
            "kernel/rope_backward/cuda.cu"
            "kernel/scale/cuda.cu"
            "kernel/scale_inplace/cuda.cu"
            "kernel/scale_slice/cuda.cu"
            "kernel/silu_backward/cuda.cu"
            "kernel/silu/cuda.cu"
            "kernel/silu_inplace/cuda.cu"
            "kernel/softmax/cuda.cu"
            "kernel/softmax_inplace/cuda.cu"
            "kernel/sqrt/cuda.cu"
            "kernel/sqrt_inplace/cuda.cu"
            "kernel/subcopy/cuda.cu"
            "kernel/subtract_indexed_outputs/cuda.cu"
            "kernel/sum_fiber/cuda.cu"
            "kernel/sum_slice/cuda.cu"
            "kernel/sum/cuda.cu"
            "kernel/sumprod_fiber/cuda.cu"
            "kernel/sumprod_slice/cuda.cu"
            "kernel/total_sum_accum/cuda.cu"
            "kernel/transpose/cuda.cu"
            "kernel/swap_two_axes/cuda.cu"
            "kernel/isfinite/cuda.cu"
        )
        if(NNTILE_USE_FLASH_SDPA)
            list(APPEND KERNEL_CUDA_SRC
                "kernel/flash_sdpa_fwd_cudnn/cuda.cc"
                "kernel/flash_sdpa_bwd_cudnn/cuda.cc"
            )
        endif()
    endif(NNTILE_USE_CUDA)
endif(NOT HAVE_STARPU_SIMGRID)

# Basic StarPU-related sources
set(STARPU_BASE_SRC
    "starpu/handle.cc"
    "starpu/codelet.cc"
)

# StarPU-related wrappers (codelets) for low-level kernels
set(STARPU_CODELET_SRC
    "starpu/accumulate.cc"
    "starpu/accumulate_hypot.cc"
    "starpu/hypot.cc"
    "starpu/accumulate_maxsumexp.cc"
    "starpu/adam_step.cc"
    "starpu/adamw_step.cc"
    "starpu/sgd_step.cc"
    "starpu/add.cc"
    "starpu/add_fiber.cc"
    "starpu/add_fiber_inplace.cc"
    "starpu/scale_fiber.cc"
    "starpu/add_inplace.cc"
    "starpu/add_slice.cc"
    "starpu/add_slice_inplace.cc"
    "starpu/clear.cc"
    "starpu/conv2d_bwd_input_inplace.cc"
    "starpu/conv2d_bwd_weight_inplace.cc"
    "starpu/conv2d_inplace.cc"
    "starpu/copy.cc"
    "starpu/embedding.cc"
    "starpu/embedding_backward.cc"
    "starpu/fill.cc"
    "starpu/gelu.cc"
    "starpu/gelu_inplace.cc"
    "starpu/gelu_backward.cc"
    "starpu/gelutanh.cc"
    "starpu/gelutanh_backward.cc"
    "starpu/gelutanh_inplace.cc"
    "starpu/gemm.cc"
    "starpu/hypot_inplace.cc"
    "starpu/hypot_scalar_inverse.cc"
    "starpu/log_scalar.cc"
    "starpu/logsumexp.cc"
    "starpu/mask_scalar.cc"
    "starpu/maxsumexp.cc"
    "starpu/norm.cc"
    "starpu/norm_fiber.cc"
    "starpu/norm_fiber_inplace.cc"
    "starpu/norm_slice_inplace.cc"
    "starpu/norm_slice.cc"
    "starpu/pow.cc"
    "starpu/multiply.cc"
    "starpu/multiply_fiber_inplace.cc"
    "starpu/multiply_fiber.cc"
    "starpu/multiply_inplace.cc"
    "starpu/multiply_slice.cc"
    "starpu/randn.cc"
    "starpu/relu_inplace.cc"
    "starpu/relu_backward.cc"
    "starpu/relu.cc"
    "starpu/rope.cc"
    "starpu/rope_backward.cc"
    "starpu/scale.cc"
    "starpu/scale_inplace.cc"
    "starpu/scale_slice.cc"
    "starpu/silu_backward.cc"
    "starpu/silu.cc"
    "starpu/silu_inplace.cc"
    "starpu/softmax.cc"
    "starpu/softmax_inplace.cc"
    "starpu/sqrt.cc"
    "starpu/sqrt_inplace.cc"
    "starpu/subcopy.cc"
    "starpu/subtract_indexed_outputs.cc"
    "starpu/sum_fiber.cc"
    "starpu/sum_slice.cc"
    "starpu/sum.cc"
    "starpu/sumprod_fiber.cc"
    "starpu/sumprod_slice.cc"
    "starpu/total_sum_accum.cc"
    "starpu/transpose.cc"
    "starpu/swap_two_axes.cc"
    "starpu/isfinite.cc"
)

# Basic Tile-related sources
set(CORE_BASE_SRC "core/traits.cc")

# Tile-related operations
set(CORE_OPERATION_SRC
    "core/adam_step.cc"
    "core/adamw_step.cc"
    "core/sgd_step.cc"
    "core/add.cc"
    "core/add_inplace.cc"
    "core/add_fiber.cc"
    "core/add_fiber_inplace.cc"
    "core/scale_fiber.cc"
    "core/add_slice.cc"
    "core/add_slice_inplace.cc"
    "core/clear.cc"
    "core/conv2d_bwd_input_inplace.cc"
    "core/conv2d_bwd_weight_inplace.cc"
    "core/conv2d_inplace.cc"
    "core/copy.cc"
    "core/copy_intersection.cc"
    "core/embedding.cc"
    "core/embedding_backward.cc"
    "core/fill.cc"
    "core/gelu.cc"
    "core/gelu_inplace.cc"
    "core/gelu_backward.cc"
    "core/gelutanh.cc"
    "core/gelutanh_backward.cc"
    "core/gelutanh_inplace.cc"
    "core/gemm.cc"
    "core/hypot_inplace.cc"
    "core/hypot.cc"
    "core/hypot_scalar_inverse.cc"
    "core/log_scalar.cc"
    "core/logsumexp.cc"
    "core/mask_scalar.cc"
    "core/maxsumexp.cc"
    "core/norm.cc"
    "core/norm_fiber.cc"
    "core/norm_fiber_inplace.cc"
    "core/norm_slice_inplace.cc"
    "core/norm_slice.cc"
    "core/pow.cc"
    "core/multiply_fiber_inplace.cc"
    "core/multiply_fiber.cc"
    "core/multiply.cc"
    "core/multiply_inplace.cc"
    "core/multiply_slice.cc"
    "core/randn.cc"
    "core/relu_inplace.cc"
    "core/relu_backward.cc"
    "core/relu.cc"
    "core/rope.cc"
    "core/rope_backward.cc"
    "core/transpose.cc"
    "core/swap_two_axes.cc"
    "core/scale.cc"
    "core/scale_inplace.cc"
    "core/scale_slice.cc"
    "core/silu_backward.cc"
    "core/silu.cc"
    "core/silu_inplace.cc"
    "core/softmax.cc"
    "core/softmax_inplace.cc"
    "core/sqrt.cc"
    "core/sqrt_inplace.cc"
    "core/subtract_indexed_outputs.cc"
    "core/sum_fiber.cc"
    "core/sum_slice.cc"
    "core/sum.cc"
    "core/sumprod_fiber.cc"
    "core/sumprod_slice.cc"
    "core/total_sum_accum.cc"
    "core/isfinite.cc"
)

# Tensor-related sources

# Logger-related sources
set(LOGGER_SRC
    "logger/logger_thread.cc"
    "logger/websocket_client.cc"
)

set(GRAPH_TENSOR_LAYER_SRC
    "tensor/axis_descriptor.cc"
    "tensor/tiling_spec_json.cc"
    "tensor/tensor_graph_tiling.cc"
    "tensor/graph_data_node.cc"
    "tensor/graph_op_node.cc"
    "tensor/tile_lowering_helpers.cc"
    "tensor/ops/add.cc"
    "tensor/ops/add_fiber.cc"
    "tensor/ops/add_fiber_inplace.cc"
    "tensor/ops/add_inplace.cc"
    "tensor/ops/add_slice.cc"
    "tensor/ops/add_slice_inplace.cc"
    "tensor/ops/adam_step.cc"
    "tensor/ops/adamw_step.cc"
    "tensor/ops/clear.cc"
    "tensor/ops/concat.cc"
    "tensor/ops/contiguous_view.cc"
    "tensor/ops/conv2d_bwd_input_inplace.cc"
    "tensor/ops/conv2d_bwd_weight_inplace.cc"
    "tensor/ops/conv2d_inplace.cc"
    "tensor/ops/copy.cc"
    "tensor/ops/copy_intersection.cc"
    "tensor/ops/embedding.cc"
    "tensor/ops/embedding_backward.cc"
    "tensor/ops/fill.cc"
    "tensor/ops/gather.cc"
    "tensor/ops/gemm.cc"
    "tensor/ops/gelu.cc"
    "tensor/ops/gelu_backward.cc"
    "tensor/ops/gelu_inplace.cc"
    "tensor/ops/gelutanh.cc"
    "tensor/ops/gelutanh_backward.cc"
    "tensor/ops/gelutanh_inplace.cc"
    "tensor/ops/hypot.cc"
    "tensor/ops/hypot_inplace.cc"
    "tensor/ops/hypot_scalar_inverse.cc"
    "tensor/ops/log_scalar.cc"
    "tensor/ops/logsumexp.cc"
    "tensor/ops/mask_scalar.cc"
    "tensor/ops/maxsumexp.cc"
    "tensor/ops/multiply.cc"
    "tensor/ops/multiply_fiber.cc"
    "tensor/ops/multiply_fiber_inplace.cc"
    "tensor/ops/multiply_inplace.cc"
    "tensor/ops/multiply_slice.cc"
    "tensor/ops/norm.cc"
    "tensor/ops/norm_fiber.cc"
    "tensor/ops/norm_fiber_inplace.cc"
    "tensor/ops/norm_slice.cc"
    "tensor/ops/norm_slice_inplace.cc"
    "tensor/ops/pow.cc"
    "tensor/ops/randn.cc"
    "tensor/ops/relu.cc"
    "tensor/ops/relu_backward.cc"
    "tensor/ops/relu_inplace.cc"
    "tensor/ops/rope.cc"
    "tensor/ops/rope_backward.cc"
    "tensor/ops/scale.cc"
    "tensor/ops/scale_fiber.cc"
    "tensor/ops/scale_inplace.cc"
    "tensor/ops/scale_slice.cc"
    "tensor/ops/scatter.cc"
    "tensor/ops/sgd_step.cc"
    "tensor/ops/silu.cc"
    "tensor/ops/silu_backward.cc"
    "tensor/ops/silu_inplace.cc"
    "tensor/ops/softmax.cc"
    "tensor/ops/softmax_inplace.cc"
    "tensor/ops/sqrt.cc"
    "tensor/ops/sqrt_inplace.cc"
    "tensor/ops/subtract_indexed_outputs.cc"
    "tensor/ops/sum.cc"
    "tensor/ops/sum_fiber.cc"
    "tensor/ops/sum_slice.cc"
    "tensor/ops/sumprod_fiber.cc"
    "tensor/ops/sumprod_slice.cc"
    "tensor/ops/total_sum_accum.cc"
    "tensor/ops/transpose.cc"
    "tensor/ops/swap_two_axes.cc"
)

set(GRAPH_NN_LAYER_SRC
    "nn/graph.cc"
    "nn/tensor_node.cc"
    "nn/ops/add.cc"
    "nn/ops/add_fiber.cc"
    "nn/ops/add_slice.cc"
    "nn/ops/adam_step.cc"
    "nn/ops/adamw_step.cc"
    "nn/ops/concat.cc"
    "nn/ops/cross_entropy.cc"
    "nn/ops/embedding.cc"
    "nn/ops/gemm.cc"
    "nn/ops/gelu.cc"
    "nn/ops/layer_norm.cc"
    "nn/ops/gelutanh.cc"
    "nn/ops/hypot.cc"
    "nn/ops/mse_loss.cc"
    "nn/ops/multiply.cc"
    "nn/ops/multiply_fiber.cc"
    "nn/ops/multiply_slice.cc"
    "nn/ops/norm.cc"
    "nn/ops/norm_fiber.cc"
    "nn/ops/norm_slice.cc"
    "nn/ops/relu.cc"
    "nn/ops/rms_norm.cc"
    "nn/ops/rope.cc"
    "nn/ops/scale.cc"
    "nn/ops/scale_fiber.cc"
    "nn/ops/scale_slice.cc"
    "nn/ops/sdpa_eager.cc"
    "nn/ops/sdpa_causal_mask.cc"
    "nn/ops/sgd_step.cc"
    "nn/ops/silu.cc"
    "nn/ops/softmax.cc"
    "nn/ops/sum_fiber.cc"
    "nn/ops/sum_slice.cc"
    "nn/ops/transpose.cc"
)

set(GRAPH_IO_SRC
    "io/safetensors.cc"
)

set(GRAPH_UTIL_SRC
    "kv_cache.cc"
)

set(GRAPH_MODULE_SRC
    "module/module.cc"
    "module/activation.cc"
    "module/gated_mlp.cc"
    "module/mlp.cc"
    "module/linear.cc"
    "module/embedding.cc"
    "module/rms_norm.cc"
    "module/layer_norm.cc"
)

set(GRAPH_DATASET_SRC
    "dataset/causal_lm_mmap.cc"
    "dataset/seq2seq_lm_mmap.cc"
    )

set(GRAPH_MODEL_SRC
    "model/llama/llama_mlp.cc"
    "model/llama/llama_attention.cc"
    "model/llama/llama_rope.cc"
    "model/llama/llama_decoder.cc"
    "model/llama/llama_model.cc"
    "model/llama/llama_causal.cc"
    "model/gptneo/gptneo_config.cc"
    "model/gptneo/gptneo_attention.cc"
    "model/gptneo/gptneo_mlp.cc"
    "model/gptneo/gptneo_decoder.cc"
    "model/gptneo/gptneo_model.cc"
    "model/gptneo/gptneo_causal.cc"
    "model/gpt2/gpt2_attention.cc"
    "model/gpt2/gpt2_mlp.cc"
    "model/gpt2/gpt2_block.cc"
    "model/gpt2/gpt2_model.cc"
    "model/gpt2/gpt2_causal.cc"
    "model/gptneox/gptneox_config.cc"
    "model/gptneox/gptneox_rope.cc"
    "model/gptneox/gptneox_mlp.cc"
    "model/gptneox/gptneox_attention.cc"
    "model/gptneox/gptneox_decoder.cc"
    "model/gptneox/gptneox_model.cc"
    "model/gptneox/gptneox_causal.cc"
    "model/t5/t5_attention.cc"
    "model/t5/t5_ff.cc"
    "model/t5/t5_block.cc"
    "model/t5/t5_model.cc"
    "model/t5/t5_for_conditional_generation.cc"
    "model/bert/bert_self_attention.cc"
    "model/bert/bert_self_output.cc"
    "model/bert/bert_attention.cc"
    "model/bert/bert_intermediate.cc"
    "model/bert/bert_output.cc"
    "model/bert/bert_layer.cc"
    "model/bert/bert_embeddings.cc"
    "model/bert/bert_model.cc"
    "model/bert/bert_mlm_head.cc"
    "model/bert/bert_mlm.cc"
    "model/roberta/roberta_embeddings.cc"
    "model/roberta/roberta_model.cc"
    "model/roberta/roberta_mlm_head.cc"
    "model/roberta/roberta_mlm.cc"
    )

set(GRAPH_OPTIM_SRC
    "optim/optimizer.cc"
    "optim/sgd.cc"
    "optim/adam.cc"
    "optim/adamw.cc"
)

set(GRAPH_RUNTIME_SRC
    "bind_parameters.cc"
    "core/execution_schedule.cc"
    "runtime.cc"
    )

set(GRAPH_TILE_LAYER_SRC
    "tile/append_tensor_graph_phase.cc"
    "tile/from_tensor_graph.cc"
    "tile/graph_data_node.cc"
    "tile/lower_from_tensor.cc"
    "tile/ops/adam_step.cc"
    "tile/ops/adamw_step.cc"
    "tile/ops/add.cc"
    "tile/ops/add_fiber.cc"
    "tile/ops/add_fiber_inplace.cc"
    "tile/ops/add_inplace.cc"
    "tile/ops/add_slice.cc"
    "tile/ops/add_slice_inplace.cc"
    "tile/ops/clear.cc"
    "tile/ops/conv2d_bwd_input_inplace.cc"
    "tile/ops/conv2d_bwd_weight_inplace.cc"
    "tile/ops/conv2d_inplace.cc"
    "tile/ops/copy.cc"
    "tile/ops/copy_intersection.cc"
    "tile/ops/embedding.cc"
    "tile/ops/embedding_backward.cc"
    "tile/ops/fill.cc"
    "tile/ops/gemm.cc"
    "tile/ops/gelu.cc"
    "tile/ops/gelu_backward.cc"
    "tile/ops/gelu_inplace.cc"
    "tile/ops/gelutanh.cc"
    "tile/ops/gelutanh_backward.cc"
    "tile/ops/gelutanh_inplace.cc"
    "tile/ops/hypot.cc"
    "tile/ops/hypot_inplace.cc"
    "tile/ops/hypot_scalar_inverse.cc"
    "tile/ops/log_scalar.cc"
    "tile/ops/logsumexp.cc"
    "tile/ops/mask_scalar.cc"
    "tile/ops/maxsumexp.cc"
    "tile/ops/multiply.cc"
    "tile/ops/multiply_fiber.cc"
    "tile/ops/multiply_fiber_inplace.cc"
    "tile/ops/multiply_inplace.cc"
    "tile/ops/multiply_slice.cc"
    "tile/ops/norm.cc"
    "tile/ops/norm_fiber.cc"
    "tile/ops/norm_fiber_inplace.cc"
    "tile/ops/norm_slice.cc"
    "tile/ops/norm_slice_inplace.cc"
    "tile/ops/pow.cc"
    "tile/ops/randn.cc"
    "tile/ops/relu.cc"
    "tile/ops/relu_backward.cc"
    "tile/ops/relu_inplace.cc"
    "tile/ops/rope.cc"
    "tile/ops/rope_backward.cc"
    "tile/ops/scale.cc"
    "tile/ops/scale_fiber.cc"
    "tile/ops/scale_inplace.cc"
    "tile/ops/scale_slice.cc"
    "tile/ops/sgd_step.cc"
    "tile/ops/silu.cc"
    "tile/ops/silu_backward.cc"
    "tile/ops/silu_inplace.cc"
    "tile/ops/softmax.cc"
    "tile/ops/softmax_inplace.cc"
    "tile/ops/sqrt.cc"
    "tile/ops/sqrt_inplace.cc"
    "tile/ops/subtract_indexed_outputs.cc"
    "tile/ops/sum.cc"
    "tile/ops/sum_fiber.cc"
    "tile/ops/sum_slice.cc"
    "tile/ops/sumprod_fiber.cc"
    "tile/ops/sumprod_slice.cc"
    "tile/ops/total_sum_accum.cc"
    "tile/ops/transpose.cc"
    "tile/ops/swap_two_axes.cc"
)

set(NNTILE_GRAPH_STACK_SRC
    ${GRAPH_TENSOR_LAYER_SRC}
    ${GRAPH_NN_LAYER_SRC}
    ${GRAPH_IO_SRC}
    ${GRAPH_UTIL_SRC}
    ${GRAPH_MODULE_SRC}
    ${GRAPH_DATASET_SRC}
    ${GRAPH_MODEL_SRC}
    ${GRAPH_OPTIM_SRC}
    ${GRAPH_RUNTIME_SRC}
    ${GRAPH_TILE_LAYER_SRC}
)

if(NNTILE_USE_FLASH_SDPA)
    list(APPEND STARPU_CODELET_SRC
        "starpu/flash_sdpa_fwd_cudnn.cc"
        "starpu/flash_sdpa_bwd_cudnn.cc"
    )
    list(APPEND CORE_OPERATION_SRC
        "core/flash_sdpa_fwd_cudnn.cc"
        "core/flash_sdpa_bwd_cudnn.cc"
    )
    list(APPEND GRAPH_TENSOR_LAYER_SRC
        "tensor/ops/flash_sdpa_bwd_cudnn.cc"
        "tensor/ops/flash_sdpa_fwd_cudnn.cc"
    )
    list(APPEND GRAPH_TILE_LAYER_SRC
        "tile/ops/flash_sdpa_bwd_cudnn.cc"
        "tile/ops/flash_sdpa_fwd_cudnn.cc"
    )
    set(NNTILE_GRAPH_STACK_SRC
        ${GRAPH_TENSOR_LAYER_SRC}
        ${GRAPH_NN_LAYER_SRC}
        ${GRAPH_IO_SRC}
        ${GRAPH_UTIL_SRC}
        ${GRAPH_MODULE_SRC}
        ${GRAPH_DATASET_SRC}
        ${GRAPH_MODEL_SRC}
        ${GRAPH_OPTIM_SRC}
        ${GRAPH_RUNTIME_SRC}
        ${GRAPH_TILE_LAYER_SRC}
    )
endif()

set(NNTILE_CORE_SRC
    ${BASE_SRC}
    ${KERNEL_CUDA_SRC}
    ${STARPU_BASE_SRC}
    ${STARPU_CODELET_SRC}
    ${CORE_BASE_SRC}
    ${CORE_OPERATION_SRC}
    ${LOGGER_SRC}
)

set(NNTILE_ALL_SRC ${NNTILE_CORE_SRC} ${NNTILE_GRAPH_STACK_SRC})

function(nntile_add_objs name)
    set(_sources ${ARGN})
    if(NOT _sources)
        return()
    endif()
    add_library(${name} OBJECT ${_sources})
    nntile_apply_common_includes(${name})
    nntile_apply_cuda(${name})
    nntile_apply_cblas(${name})
    if(name MATCHES "nntile_objs_(optim|io|tensor|runtime)$")
        target_link_libraries(${name} PUBLIC nlohmann_json::nlohmann_json)
    endif()
    target_link_libraries(nntile PRIVATE ${name})
endfunction()

if(KERNEL_CPU_SRC)
    nntile_add_objs(nntile_objs_kernel ${KERNEL_CPU_SRC} ${KERNEL_CUDA_SRC})
endif()
nntile_add_objs(nntile_objs_starpu ${STARPU_BASE_SRC} ${STARPU_CODELET_SRC})
nntile_add_objs(nntile_objs_core ${CORE_BASE_SRC} ${CORE_OPERATION_SRC})
nntile_add_objs(nntile_objs_graph_base dtype.cc kv_cache.cc)
nntile_add_objs(nntile_objs_tile ${GRAPH_TILE_LAYER_SRC})
nntile_add_objs(nntile_objs_tensor ${GRAPH_TENSOR_LAYER_SRC})
nntile_add_objs(nntile_objs_nn ${GRAPH_NN_LAYER_SRC})
nntile_add_objs(nntile_objs_runtime ${GRAPH_RUNTIME_SRC})
nntile_add_objs(nntile_objs_module ${GRAPH_MODULE_SRC})
nntile_add_objs(nntile_objs_model ${GRAPH_MODEL_SRC})
nntile_add_objs(nntile_objs_optim ${GRAPH_OPTIM_SRC})
nntile_add_objs(nntile_objs_io ${GRAPH_IO_SRC})
nntile_add_objs(nntile_objs_dataset ${GRAPH_DATASET_SRC})
target_sources(nntile PRIVATE ${LOGGER_SRC})
target_sources(nntile PRIVATE context.cc)
