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
            "kernel/flash_sdpa_fwd_cudnn/cuda.cc"
            "kernel/flash_sdpa_bwd_cudnn/cuda.cc"
            "kernel/isfinite/cuda.cu"
        )
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
    "starpu/flash_sdpa_fwd_cudnn.cc"
    "starpu/flash_sdpa_bwd_cudnn.cc"
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
    "starpu/isfinite.cc"
)

# Basic Tile-related sources
set(TILE_BASE_SRC "tile/traits.cc")

# Tile-related operations
set(TILE_OPERATION_SRC
    "tile/adam_step.cc"
    "tile/adamw_step.cc"
    "tile/sgd_step.cc"
    "tile/add.cc"
    "tile/add_inplace.cc"
    "tile/add_fiber.cc"
    "tile/add_fiber_inplace.cc"
    "tile/scale_fiber.cc"
    "tile/add_slice.cc"
    "tile/add_slice_inplace.cc"
    "tile/clear.cc"
    "tile/conv2d_bwd_input_inplace.cc"
    "tile/conv2d_bwd_weight_inplace.cc"
    "tile/conv2d_inplace.cc"
    "tile/copy.cc"
    "tile/copy_intersection.cc"
    "tile/embedding.cc"
    "tile/embedding_backward.cc"
    "tile/fill.cc"
    "tile/flash_sdpa_fwd_cudnn.cc"
    "tile/flash_sdpa_bwd_cudnn.cc"
    "tile/gelu.cc"
    "tile/gelu_inplace.cc"
    "tile/gelu_backward.cc"
    "tile/gelutanh.cc"
    "tile/gelutanh_backward.cc"
    "tile/gelutanh_inplace.cc"
    "tile/gemm.cc"
    "tile/hypot_inplace.cc"
    "tile/hypot.cc"
    "tile/hypot_scalar_inverse.cc"
    "tile/log_scalar.cc"
    "tile/logsumexp.cc"
    "tile/mask_scalar.cc"
    "tile/maxsumexp.cc"
    "tile/norm.cc"
    "tile/norm_fiber.cc"
    "tile/norm_fiber_inplace.cc"
    "tile/norm_slice_inplace.cc"
    "tile/norm_slice.cc"
    "tile/pow.cc"
    "tile/multiply_fiber_inplace.cc"
    "tile/multiply_fiber.cc"
    "tile/multiply.cc"
    "tile/multiply_inplace.cc"
    "tile/multiply_slice.cc"
    "tile/randn.cc"
    "tile/relu_inplace.cc"
    "tile/relu_backward.cc"
    "tile/relu.cc"
    "tile/rope.cc"
    "tile/rope_backward.cc"
    "tile/transpose.cc"
    "tile/scale.cc"
    "tile/scale_inplace.cc"
    "tile/scale_slice.cc"
    "tile/silu_backward.cc"
    "tile/silu.cc"
    "tile/silu_inplace.cc"
    "tile/softmax.cc"
    "tile/softmax_inplace.cc"
    "tile/sqrt.cc"
    "tile/sqrt_inplace.cc"
    "tile/subtract_indexed_outputs.cc"
    "tile/sum_fiber.cc"
    "tile/sum_slice.cc"
    "tile/sum.cc"
    "tile/sumprod_fiber.cc"
    "tile/sumprod_slice.cc"
    "tile/total_sum_accum.cc"
    "tile/isfinite.cc"
)

# Tensor-related sources
set(TENSOR_BASE_SRC "tensor/traits.cc")

# Tensor-related operations
set(TENSOR_OPERATION_SRC
    "tensor/adam_step.cc"
    "tensor/adamw_step.cc"
    "tensor/sgd_step.cc"
    "tensor/add.cc"
    "tensor/add_fiber.cc"
    "tensor/add_fiber_inplace.cc"
    "tensor/scale_fiber.cc"
    "tensor/add_inplace.cc"
    "tensor/add_slice.cc"
    "tensor/add_slice_inplace.cc"
    "tensor/clear.cc"
    "tensor/conv2d_bwd_input_inplace.cc"
    "tensor/conv2d_bwd_weight_inplace.cc"
    "tensor/conv2d_inplace.cc"
    "tensor/copy.cc"
    "tensor/copy_intersection.cc"
    "tensor/distributions.cc"
    "tensor/embedding.cc"
    "tensor/embedding_backward.cc"
    "tensor/fill.cc"
    "tensor/flash_sdpa_fwd_cudnn.cc"
    "tensor/flash_sdpa_bwd_cudnn.cc"
    "tensor/gather.cc"
    "tensor/gelu.cc"
    "tensor/gelu_inplace.cc"
    "tensor/gelu_backward.cc"
    "tensor/gelutanh.cc"
    "tensor/gelutanh_backward.cc"
    "tensor/gelutanh_inplace.cc"
    "tensor/gemm.cc"
    "tensor/hypot_inplace.cc"
    "tensor/hypot.cc"
    "tensor/hypot_scalar_inverse.cc"
    "tensor/log_scalar.cc"
    "tensor/logsumexp.cc"
    "tensor/mask_scalar.cc"
    "tensor/maxsumexp.cc"
    "tensor/norm.cc"
    "tensor/norm_fiber.cc"
    "tensor/norm_fiber_inplace.cc"
    "tensor/norm_slice_inplace.cc"
    "tensor/norm_slice.cc"
    "tensor/pow.cc"
    "tensor/multiply.cc"
    "tensor/multiply_fiber_inplace.cc"
    "tensor/multiply_fiber.cc"
    "tensor/multiply_inplace.cc"
    "tensor/multiply_slice.cc"
    "tensor/randn.cc"
    "tensor/relu_inplace.cc"
    "tensor/relu_backward.cc"
    "tensor/relu.cc"
    "tensor/rope.cc"
    "tensor/rope_backward.cc"
    "tensor/scale.cc"
    "tensor/scale_inplace.cc"
    "tensor/scale_slice.cc"
    "tensor/scatter.cc"
    "tensor/silu_backward.cc"
    "tensor/silu_inplace.cc"
    "tensor/silu.cc"
    "tensor/softmax.cc"
    "tensor/softmax_inplace.cc"
    "tensor/sqrt.cc"
    "tensor/sqrt_inplace.cc"
    "tensor/subtract_indexed_outputs.cc"
    "tensor/sum_fiber.cc"
    "tensor/sum_slice.cc"
    "tensor/sum.cc"
    "tensor/sumprod_fiber.cc"
    "tensor/sumprod_slice.cc"
    "tensor/total_sum_accum.cc"
    "tensor/transpose.cc"
    "tensor/isfinite.cc"
)

# Logger-related sources
set(LOGGER_SRC
    "logger/logger_thread.cc"
    "logger/websocket_client.cc"
)

set(GRAPH_TENSOR_SRC
    "tensor_graph/axis_descriptor.cc"
    "tensor_graph/tensor_graph_tiling.cc"
    "tensor_graph/graph_data_node.cc"
    "tensor_graph/graph_op_node.cc"
    "tensor_graph/tile_lowering_helpers.cc"
    "tensor_graph/ops/add.cc"
    "tensor_graph/ops/add_fiber.cc"
    "tensor_graph/ops/add_fiber_inplace.cc"
    "tensor_graph/ops/add_inplace.cc"
    "tensor_graph/ops/add_slice.cc"
    "tensor_graph/ops/add_slice_inplace.cc"
    "tensor_graph/ops/adam_step.cc"
    "tensor_graph/ops/adamw_step.cc"
    "tensor_graph/ops/clear.cc"
    "tensor_graph/ops/concat.cc"
    "tensor_graph/ops/conv2d_bwd_input_inplace.cc"
    "tensor_graph/ops/conv2d_bwd_weight_inplace.cc"
    "tensor_graph/ops/conv2d_inplace.cc"
    "tensor_graph/ops/copy.cc"
    "tensor_graph/ops/copy_intersection.cc"
    "tensor_graph/ops/embedding.cc"
    "tensor_graph/ops/embedding_backward.cc"
    "tensor_graph/ops/fill.cc"
    "tensor_graph/ops/flash_sdpa_bwd_cudnn.cc"
    "tensor_graph/ops/flash_sdpa_fwd_cudnn.cc"
    "tensor_graph/ops/gather.cc"
    "tensor_graph/ops/gemm.cc"
    "tensor_graph/ops/gelu.cc"
    "tensor_graph/ops/gelu_backward.cc"
    "tensor_graph/ops/gelu_inplace.cc"
    "tensor_graph/ops/gelutanh.cc"
    "tensor_graph/ops/gelutanh_backward.cc"
    "tensor_graph/ops/gelutanh_inplace.cc"
    "tensor_graph/ops/hypot.cc"
    "tensor_graph/ops/hypot_inplace.cc"
    "tensor_graph/ops/hypot_scalar_inverse.cc"
    "tensor_graph/ops/log_scalar.cc"
    "tensor_graph/ops/logsumexp.cc"
    "tensor_graph/ops/mask_scalar.cc"
    "tensor_graph/ops/maxsumexp.cc"
    "tensor_graph/ops/multiply.cc"
    "tensor_graph/ops/multiply_fiber.cc"
    "tensor_graph/ops/multiply_fiber_inplace.cc"
    "tensor_graph/ops/multiply_inplace.cc"
    "tensor_graph/ops/multiply_slice.cc"
    "tensor_graph/ops/norm.cc"
    "tensor_graph/ops/norm_fiber.cc"
    "tensor_graph/ops/norm_fiber_inplace.cc"
    "tensor_graph/ops/norm_slice.cc"
    "tensor_graph/ops/norm_slice_inplace.cc"
    "tensor_graph/ops/pow.cc"
    "tensor_graph/ops/randn.cc"
    "tensor_graph/ops/relu.cc"
    "tensor_graph/ops/relu_backward.cc"
    "tensor_graph/ops/relu_inplace.cc"
    "tensor_graph/ops/rope.cc"
    "tensor_graph/ops/rope_backward.cc"
    "tensor_graph/ops/scale.cc"
    "tensor_graph/ops/scale_fiber.cc"
    "tensor_graph/ops/scale_inplace.cc"
    "tensor_graph/ops/scale_slice.cc"
    "tensor_graph/ops/scatter.cc"
    "tensor_graph/ops/sgd_step.cc"
    "tensor_graph/ops/silu.cc"
    "tensor_graph/ops/silu_backward.cc"
    "tensor_graph/ops/silu_inplace.cc"
    "tensor_graph/ops/softmax.cc"
    "tensor_graph/ops/softmax_inplace.cc"
    "tensor_graph/ops/sqrt.cc"
    "tensor_graph/ops/sqrt_inplace.cc"
    "tensor_graph/ops/subtract_indexed_outputs.cc"
    "tensor_graph/ops/sum.cc"
    "tensor_graph/ops/sum_fiber.cc"
    "tensor_graph/ops/sum_slice.cc"
    "tensor_graph/ops/sumprod_fiber.cc"
    "tensor_graph/ops/sumprod_slice.cc"
    "tensor_graph/ops/total_sum_accum.cc"
    "tensor_graph/ops/transpose.cc"
)

set(GRAPH_NN_SRC
    "nn_graph/graph.cc"
    "nn_graph/tensor_node.cc"
    "nn_graph/ops/add.cc"
    "nn_graph/ops/add_fiber.cc"
    "nn_graph/ops/add_slice.cc"
    "nn_graph/ops/adam_step.cc"
    "nn_graph/ops/adamw_step.cc"
    "nn_graph/ops/concat.cc"
    "nn_graph/ops/cross_entropy.cc"
    "nn_graph/ops/embedding.cc"
    "nn_graph/ops/gemm.cc"
    "nn_graph/ops/gelu.cc"
    "nn_graph/ops/layer_norm.cc"
    "nn_graph/ops/gelutanh.cc"
    "nn_graph/ops/mse_loss.cc"
    "nn_graph/ops/multiply.cc"
    "nn_graph/ops/multiply_fiber.cc"
    "nn_graph/ops/multiply_slice.cc"
    "nn_graph/ops/norm.cc"
    "nn_graph/ops/norm_fiber.cc"
    "nn_graph/ops/norm_slice.cc"
    "nn_graph/ops/relu.cc"
    "nn_graph/ops/rms_norm.cc"
    "nn_graph/ops/rope.cc"
    "nn_graph/ops/scale.cc"
    "nn_graph/ops/scale_fiber.cc"
    "nn_graph/ops/scale_slice.cc"
    "nn_graph/ops/sdpa_eager.cc"
    "nn_graph/ops/sdpa_causal_mask.cc"
    "nn_graph/ops/sgd_step.cc"
    "nn_graph/ops/silu.cc"
    "nn_graph/ops/softmax.cc"
    "nn_graph/ops/sum_fiber.cc"
    "nn_graph/ops/sum_slice.cc"
    "nn_graph/ops/transpose.cc"
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
    "runtime.cc"
    )

set(GRAPH_TILE_SRC
    "tile_graph/append_tensor_graph_phase.cc"
    "tile_graph/from_tensor_graph.cc"
    "tile_graph/graph_data_node.cc"
    "tile_graph/lower_from_tensor.cc"
    "tile_graph/ops/adam_step.cc"
    "tile_graph/ops/adamw_step.cc"
    "tile_graph/ops/add.cc"
    "tile_graph/ops/add_fiber.cc"
    "tile_graph/ops/add_fiber_inplace.cc"
    "tile_graph/ops/add_inplace.cc"
    "tile_graph/ops/add_slice.cc"
    "tile_graph/ops/add_slice_inplace.cc"
    "tile_graph/ops/clear.cc"
    "tile_graph/ops/conv2d_bwd_input_inplace.cc"
    "tile_graph/ops/conv2d_bwd_weight_inplace.cc"
    "tile_graph/ops/conv2d_inplace.cc"
    "tile_graph/ops/copy.cc"
    "tile_graph/ops/copy_intersection.cc"
    "tile_graph/ops/embedding.cc"
    "tile_graph/ops/embedding_backward.cc"
    "tile_graph/ops/flash_sdpa_bwd_cudnn.cc"
    "tile_graph/ops/flash_sdpa_fwd_cudnn.cc"
    "tile_graph/ops/fill.cc"
    "tile_graph/ops/gemm.cc"
    "tile_graph/ops/gelu.cc"
    "tile_graph/ops/gelu_backward.cc"
    "tile_graph/ops/gelu_inplace.cc"
    "tile_graph/ops/gelutanh.cc"
    "tile_graph/ops/gelutanh_backward.cc"
    "tile_graph/ops/gelutanh_inplace.cc"
    "tile_graph/ops/hypot.cc"
    "tile_graph/ops/hypot_inplace.cc"
    "tile_graph/ops/hypot_scalar_inverse.cc"
    "tile_graph/ops/log_scalar.cc"
    "tile_graph/ops/logsumexp.cc"
    "tile_graph/ops/mask_scalar.cc"
    "tile_graph/ops/maxsumexp.cc"
    "tile_graph/ops/multiply.cc"
    "tile_graph/ops/multiply_fiber.cc"
    "tile_graph/ops/multiply_fiber_inplace.cc"
    "tile_graph/ops/multiply_inplace.cc"
    "tile_graph/ops/multiply_slice.cc"
    "tile_graph/ops/norm.cc"
    "tile_graph/ops/norm_fiber.cc"
    "tile_graph/ops/norm_fiber_inplace.cc"
    "tile_graph/ops/norm_slice.cc"
    "tile_graph/ops/norm_slice_inplace.cc"
    "tile_graph/ops/pow.cc"
    "tile_graph/ops/randn.cc"
    "tile_graph/ops/relu.cc"
    "tile_graph/ops/relu_backward.cc"
    "tile_graph/ops/relu_inplace.cc"
    "tile_graph/ops/rope.cc"
    "tile_graph/ops/rope_backward.cc"
    "tile_graph/ops/scale.cc"
    "tile_graph/ops/scale_fiber.cc"
    "tile_graph/ops/scale_inplace.cc"
    "tile_graph/ops/scale_slice.cc"
    "tile_graph/ops/sgd_step.cc"
    "tile_graph/ops/silu.cc"
    "tile_graph/ops/silu_backward.cc"
    "tile_graph/ops/silu_inplace.cc"
    "tile_graph/ops/softmax.cc"
    "tile_graph/ops/softmax_inplace.cc"
    "tile_graph/ops/sqrt.cc"
    "tile_graph/ops/sqrt_inplace.cc"
    "tile_graph/ops/subtract_indexed_outputs.cc"
    "tile_graph/ops/sum.cc"
    "tile_graph/ops/sum_fiber.cc"
    "tile_graph/ops/sum_slice.cc"
    "tile_graph/ops/sumprod_fiber.cc"
    "tile_graph/ops/sumprod_slice.cc"
    "tile_graph/ops/total_sum_accum.cc"
    "tile_graph/ops/transpose.cc"
)

set(NNTILE_GRAPH_STACK_SRC
    ${GRAPH_TENSOR_SRC}
    ${GRAPH_NN_SRC}
    ${GRAPH_IO_SRC}
    ${GRAPH_UTIL_SRC}
    ${GRAPH_MODULE_SRC}
    ${GRAPH_DATASET_SRC}
    ${GRAPH_MODEL_SRC}
    ${GRAPH_OPTIM_SRC}
    ${GRAPH_RUNTIME_SRC}
    ${GRAPH_TILE_SRC}
)


# Eager tensor is optional (deprecated)
if(NNTILE_BUILD_TENSOR)
    list(APPEND NNTILE_CORE_EAGER_SRC ${TENSOR_BASE_SRC} ${TENSOR_OPERATION_SRC})
else()
    set(NNTILE_CORE_EAGER_SRC "")
endif()

set(NNTILE_CORE_SRC
    ${BASE_SRC}
    ${KERNEL_CUDA_SRC}
    ${STARPU_BASE_SRC}
    ${STARPU_CODELET_SRC}
    ${TILE_BASE_SRC}
    ${TILE_OPERATION_SRC}
    ${NNTILE_CORE_EAGER_SRC}
    ${LOGGER_SRC}
)

set(NNTILE_ALL_SRC ${NNTILE_CORE_SRC} ${NNTILE_GRAPH_STACK_SRC})

if(NNTILE_COMPILE_CHECK_SUBSYSTEM)
    string(TOLOWER "${NNTILE_COMPILE_CHECK_SUBSYSTEM}" _cc_sub)
    if(_cc_sub STREQUAL "kernel")
        set(_cc_src ${KERNEL_CPU_SRC} ${KERNEL_CUDA_SRC})
    elseif(_cc_sub STREQUAL "starpu")
        set(_cc_src ${STARPU_BASE_SRC} ${STARPU_CODELET_SRC})
    elseif(_cc_sub STREQUAL "tile")
        set(_cc_src ${TILE_BASE_SRC} ${TILE_OPERATION_SRC})
    elseif(_cc_sub STREQUAL "tensor")
        set(_cc_src ${TENSOR_BASE_SRC} ${TENSOR_OPERATION_SRC})
    elseif(_cc_sub STREQUAL "tile_graph")
        set(_cc_src ${GRAPH_TILE_SRC})
    elseif(_cc_sub STREQUAL "tensor_graph")
        set(_cc_src ${GRAPH_TENSOR_SRC})
    elseif(_cc_sub STREQUAL "nn_graph")
        set(_cc_src ${GRAPH_NN_SRC})
    elseif(_cc_sub STREQUAL "runtime")
        set(_cc_src ${GRAPH_RUNTIME_SRC})
    elseif(_cc_sub STREQUAL "module")
        set(_cc_src ${GRAPH_MODULE_SRC})
    elseif(_cc_sub STREQUAL "model")
        set(_cc_src ${GRAPH_MODEL_SRC})
    elseif(_cc_sub STREQUAL "optim")
        set(_cc_src ${GRAPH_OPTIM_SRC})
    elseif(_cc_sub STREQUAL "io")
        set(_cc_src ${GRAPH_IO_SRC})
    elseif(_cc_sub STREQUAL "dataset")
        set(_cc_src ${GRAPH_DATASET_SRC})
    else()
        message(FATAL_ERROR "Unknown NNTILE_COMPILE_CHECK_SUBSYSTEM=${NNTILE_COMPILE_CHECK_SUBSYSTEM}")
    endif()
    add_library(nntile_objs_${_cc_sub} OBJECT ${_cc_src})
    nntile_apply_common_includes(nntile_objs_${_cc_sub})
    nntile_apply_cuda(nntile_objs_${_cc_sub})
    nntile_apply_cblas(nntile_objs_${_cc_sub})
    add_custom_target(nntile_compile_check_${_cc_sub} DEPENDS nntile_objs_${_cc_sub})
    return()
endif()

function(nntile_add_objs name)
    set(_sources ${ARGN})
    if(NOT _sources)
        return()
    endif()
    add_library(${name} OBJECT ${_sources})
    nntile_apply_common_includes(${name})
    nntile_apply_cuda(${name})
    nntile_apply_cblas(${name})
    target_link_libraries(nntile PRIVATE ${name})
endfunction()

if(NNTILE_BUILD_KERNEL AND KERNEL_CPU_SRC)
    nntile_add_objs(nntile_objs_kernel ${KERNEL_CPU_SRC} ${KERNEL_CUDA_SRC})
endif()
if(NNTILE_BUILD_STARPU)
    nntile_add_objs(nntile_objs_starpu ${STARPU_BASE_SRC} ${STARPU_CODELET_SRC})
endif()
if(NNTILE_BUILD_TILE)
    nntile_add_objs(nntile_objs_tile ${TILE_BASE_SRC} ${TILE_OPERATION_SRC})
endif()
if(NNTILE_BUILD_TENSOR)
    nntile_add_objs(nntile_objs_tensor ${TENSOR_BASE_SRC} ${TENSOR_OPERATION_SRC})
endif()
if(NNTILE_BUILD_GRAPH_BASE)
    nntile_add_objs(nntile_objs_graph_base dtype.cc kv_cache.cc)
endif()
if(NNTILE_BUILD_TILE_GRAPH)
    nntile_add_objs(nntile_objs_tile_graph ${GRAPH_TILE_SRC})
endif()
if(NNTILE_BUILD_TENSOR_GRAPH)
    nntile_add_objs(nntile_objs_tensor_graph ${GRAPH_TENSOR_SRC})
endif()
if(NNTILE_BUILD_NN_GRAPH)
    nntile_add_objs(nntile_objs_nn_graph ${GRAPH_NN_SRC})
endif()
if(NNTILE_BUILD_RUNTIME)
    nntile_add_objs(nntile_objs_runtime ${GRAPH_RUNTIME_SRC})
endif()
if(NNTILE_BUILD_MODULE)
    nntile_add_objs(nntile_objs_module ${GRAPH_MODULE_SRC})
endif()
if(NNTILE_BUILD_MODEL)
    nntile_add_objs(nntile_objs_model ${GRAPH_MODEL_SRC})
endif()
if(NNTILE_BUILD_OPTIM)
    nntile_add_objs(nntile_objs_optim ${GRAPH_OPTIM_SRC})
endif()
if(NNTILE_BUILD_IO)
    nntile_add_objs(nntile_objs_io ${GRAPH_IO_SRC})
endif()
if(NNTILE_BUILD_DATASET)
    nntile_add_objs(nntile_objs_dataset ${GRAPH_DATASET_SRC})
endif()
if(NNTILE_BUILD_LOGGER)
    target_sources(nntile PRIVATE ${LOGGER_SRC})
endif()
target_sources(nntile PRIVATE context.cc)
