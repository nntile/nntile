# Tensor Operations Inventory

This document catalogs all tensor operations in `include/nntile/tensor` for integration into the graph system.

## Operation Categories

### Element-wise Operations
- Unary: gelu, gelu_inplace, gelutanh, gelutanh_inplace, relu, relu_inplace, silu, silu_inplace, sqrt, sqrt_inplace, hypot, hypot_inplace, pow, log_scalar
- Binary: add, add_inplace, multiply, multiply_inplace, hypot_scalar_inverse, subtract_indexed_outputs
- Ternary: mask_scalar

### Reduction Operations
- sum: total sum of all elements (alpha*sum(src) + beta*dst)
- sum_fiber: sum over fibers along axis (alpha, beta, axis, batch_ndim, redux)
- sum_slice: sum over slices (alpha, beta, axis, batch_ndim, redux)
- sumprod_fiber: sum of products over fibers (alpha, beta, axis, batch_ndim, redux)
- sumprod_slice: sum of products over slices (alpha, beta, axis, batch_ndim, redux)
- norm: L2 norm (alpha, beta, dst)
- norm_fiber: L2 norm over fibers (alpha, beta, axis, batch_ndim, redux)
- norm_slice: L2 norm over slices (alpha, beta, axis, batch_ndim, redux)
- logsumexp: log(sum(exp(x))) over axis (alpha, beta, axis)
- maxsumexp: max + log(sum(exp(x - max))) over axis (alpha, beta, axis)

### Matrix Operations
- gemm (matrix multiplication)
- transpose

### Convolution Operations
- conv2d_inplace, conv2d_bwd_input_inplace, conv2d_bwd_weight_inplace

### Indexing/Embedding Operations
- embedding, embedding_backward, gather, scatter

### Optimizer Steps
- sgd_step, adam_step, adamw_step

### Utility Operations
- fill: fill tensor with scalar value
- clear: set tensor to zero
- copy: copy tensor data
- copy_intersection: copy overlapping regions
- randn: fill with random normal values (mean, stddev parameters)
- scale: element-wise scaling (alpha * src -> dst)
- scale_inplace: in-place scaling
- scale_fiber: scaling over fibers
- scale_slice: scaling over slices
- transpose: tensor transposition
- gather: gather elements by indices
- scatter: scatter elements by indices

### Flash Attention (CUDA-only)
- flash_sdpa_fwd_cudnn, flash_sdpa_bwd_cudnn

### Rotary Position Embedding
- rope, rope_backward

## Detailed Operation Specifications

### Element-wise Operations

#### gelu
- **Signature**: `gelu_async(const Tensor<T> &src, const Tensor<T> &dst)`
- **Inputs**: src
- **Outputs**: dst
- **Attributes**: none
- **Shape Rule**: dst.shape == src.shape
- **Dtypes**: fp32_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t, bf16_t, fp16_t

#### gelu_inplace
- **Signature**: `gelu_inplace_async(const Tensor<T> &src_dst)`
- **Inputs/Outputs**: src_dst
- **Attributes**: none
- **Shape Rule**: none
- **Dtypes**: fp32_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t, bf16_t, fp16_t

#### add
- **Signature**: `add_async(Scalar alpha, const Tensor<T> &src1, Scalar beta, const Tensor<T> &src2, const Tensor<T> &dst)`
- **Inputs**: alpha, src1, beta, src2
- **Outputs**: dst
- **Attributes**: alpha, beta
- **Shape Rule**: dst.shape == src1.shape == src2.shape
- **Dtypes**: fp32_t, bf16_t, fp16_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t

#### add_inplace
- **Signature**: `add_inplace_async(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &src_dst)`
- **Inputs**: alpha, src, beta
- **In/Outputs**: src_dst
- **Attributes**: alpha, beta
- **Shape Rule**: src_dst.shape == src.shape
- **Dtypes**: fp32_t, bf16_t, fp16_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t

#### multiply
- **Signature**: `multiply_async(const Tensor<T> &src1, const Tensor<T> &src2, const Tensor<T> &dst)`
- **Inputs**: src1, src2
- **Outputs**: dst
- **Attributes**: none
- **Shape Rule**: dst.shape == src1.shape == src2.shape
- **Dtypes**: fp32_t, bf16_t, fp16_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t

#### multiply_inplace
- **Signature**: `multiply_inplace_async(const Tensor<T> &src, const Tensor<T> &src_dst)`
- **Inputs**: src
- **In/Outputs**: src_dst
- **Attributes**: none
- **Shape Rule**: src_dst.shape == src.shape
- **Dtypes**: fp32_t, bf16_t, fp16_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t

#### multiply
- **Signature**: `multiply_async(const Tensor<T> &src1, const Tensor<T> &src2, const Tensor<T> &dst)`
- **Inputs**: src1, src2
- **Outputs**: dst
- **Attributes**: none
- **Shape Rule**: dst.shape == src1.shape == src2.shape
- **Dtypes**: [TBD]

### Matrix Operations

#### gemm
- **Signature**: `gemm_async(Scalar alpha, const TransOp &transA, const Tensor<T> &A, const TransOp &transB, const Tensor<T> &B, Scalar beta, const Tensor<T> &C, Index ndim, Index batch_ndim, int redux)`
- **Inputs**: alpha, transA, A, transB, B, beta
- **In/Outputs**: C
- **Attributes**: transA, transB, ndim, batch_ndim, redux
- **Shape Rule**: Complex matrix multiplication rules based on transA/transB, ndim, batch_ndim
- **Dtypes**: fp32_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t, bf16_t, fp16_t

### Convolution Operations

#### conv2d_inplace
- **Signature**: `conv2d_inplace_async(Scalar alpha, const Tensor<T> &X, const Tensor<T> &C, Scalar beta, const Tensor<T> &Y, std::array<Index, 2> padding, std::array<Index, 2> stride, std::array<Index, 2> dilation)`
- **Inputs**: alpha, X, C, beta
- **In/Outputs**: Y
- **Attributes**: padding, stride, dilation
- **Shape Rule**: 2D convolution shape rules: Y.shape[0] = (X.shape[0] + 2*padding[0] - dilation[0]*(C.shape[0]-1)-1)/stride[0] + 1, similar for H
- **Dtypes**: fp32_t, bf16_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t

#### conv2d_bwd_input_inplace
- **Signature**: Similar to conv2d_inplace but for backward pass w.r.t. input
- **Dtypes**: fp32_t, bf16_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t

#### conv2d_bwd_weight_inplace
- **Signature**: Similar to conv2d_inplace but for backward pass w.r.t. weights
- **Dtypes**: fp32_t, bf16_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t

### Embedding Operations

#### embedding
- **Signature**: `embedding_async(const Tensor<int64_t> &index, const Tensor<T> &vocab, const Tensor<T> &embed, Index axis)`
- **Inputs**: index (int64_t), vocab
- **Outputs**: embed
- **Attributes**: axis
- **Shape Rule**: index.ndim+1 == embed.ndim, vocab.ndim == 2, embed.shape[axis] == vocab.shape[0]
- **Dtypes**: vocab/embed: fp32_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t, bf16_t, fp16_t

#### embedding_backward
- **Signature**: `embedding_backward_async(const Tensor<T> &embed, const Tensor<int64_t> &index, const Tensor<T> &vocab, Index axis)`
- **Inputs**: embed, index (int64_t)
- **In/Outputs**: vocab
- **Attributes**: axis
- **Shape Rule**: embed.shape[axis] == vocab.shape[0], index.ndim+1 == embed.ndim
- **Dtypes**: embed/vocab: fp32_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t, bf16_t, fp16_t

### Mixed-Dtype Operations

#### mask_scalar
- **Signature**: `mask_scalar_async(const Tensor<bool_t> &mask, Scalar val, const Tensor<T> &A, Index batch_ndim)`
- **Inputs**: mask (bool_t), val, batch_ndim
- **In/Outputs**: A
- **Attributes**: val, batch_ndim
- **Shape Rule**: A.shape matches mask.shape for non-batch dimensions
- **Dtypes**: A: fp32_t, bf16_t, fp16_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t

#### total_sum_accum
- **Signature**: `total_sum_accum_async(Scalar alpha, const Tensor<T> &logsumexp, const Tensor<T> &src, const Tensor<int64_t> &class_labels, const Tensor<fp32_t> &val, Index ignore_index)`
- **Inputs**: alpha, logsumexp, src, class_labels (int64_t), val (fp32_t), ignore_index
- **Attributes**: alpha, ignore_index
- **Shape Rule**: Complex loss accumulation rules
- **Dtypes**: logsumexp/src: fp32_t, bf16_t, fp16_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t; val: fp32_t

### Reduction Operations

#### sum
- **Signature**: `sum_async(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &dst)`
- **Inputs**: alpha, src, beta
- **In/Outputs**: dst
- **Attributes**: alpha, beta
- **Shape Rule**: dst must be scalar (shape = [1])
- **Dtypes**: fp32_t, bf16_t, fp16_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t

#### sum_fiber
- **Signature**: `sum_fiber_async(Scalar alpha, const Tensor<T> &src, Scalar beta, const Tensor<T> &dst, Index axis, Index batch_ndim, int redux)`
- **Inputs**: alpha, src, beta
- **In/Outputs**: dst
- **Attributes**: alpha, beta, axis, batch_ndim, redux
- **Shape Rule**: dst.shape = src.shape with axis dimension removed, batch dims preserved
- **Dtypes**: fp32_t, bf16_t, fp16_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t

#### softmax
- **Signature**: `softmax_async(const Tensor<T> &maxsumexp, const Tensor<T> &src, Scalar alpha, const Tensor<T> &dst, Index axis)`
- **Inputs**: maxsumexp, src, alpha
- **Outputs**: dst
- **Attributes**: alpha, axis
- **Shape Rule**: dst.shape == src.shape == maxsumexp.shape
- **Dtypes**: fp32_t, bf16_t, fp16_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t

#### softmax_inplace
- **Signature**: `softmax_inplace_async(const Tensor<T> &maxsumexp, const Tensor<T> &src_dst, Index axis)`
- **Inputs**: maxsumexp
- **In/Outputs**: src_dst
- **Attributes**: axis
- **Shape Rule**: src_dst.shape == maxsumexp.shape
- **Dtypes**: fp32_t, bf16_t, fp16_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t

### Optimizer Operations

#### sgd_step
- **Signature**: `sgd_step_async(Index num_iter, Scalar momentum, Scalar lr, Scalar weight_decay, Scalar dampening, bool nesterov, const Tensor<T> &grad, const Tensor<T> &velocity, const Tensor<T> &p)`
- **Inputs**: num_iter, momentum, lr, weight_decay, dampening, nesterov, grad, velocity
- **In/Outputs**: p
- **Attributes**: num_iter, momentum, lr, weight_decay, dampening, nesterov
- **Shape Rule**: p.shape == grad.shape == velocity.shape
- **Dtypes**: fp32_t, bf16_t, fp16_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t

#### adam_step
- **Signature**: `adam_step_async(Index num_iter, Scalar beta_1, Scalar beta_2, Scalar eps, Scalar lr, Scalar weight_decay, const Tensor<T> &grad, const Tensor<T> &first_moment, const Tensor<T> &second_moment, const Tensor<T> &p)`
- **Inputs**: num_iter, beta_1, beta_2, eps, lr, weight_decay, grad, first_moment, second_moment
- **In/Outputs**: p
- **Attributes**: num_iter, beta_1, beta_2, eps, lr, weight_decay
- **Shape Rule**: p.shape == grad.shape == first_moment.shape == second_moment.shape
- **Dtypes**: fp32_t, bf16_t, fp16_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t

#### adamw_step
- **Signature**: Similar to adam_step but with different weight decay handling
- **Dtypes**: fp32_t, bf16_t, fp16_t, fp32_fast_tf32_t, fp32_fast_fp16_t, fp32_fast_bf16_t, fp64_t

## TODO: Complete Inventory

Need to fill in the [TBD] entries above and add all remaining operations.