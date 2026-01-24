/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical_graph_ops.hh
 * Logical graph operations.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>
#include <nntile/graph/logical/gelu.hh>
#include <nntile/graph/logical/gelu_inplace.hh>
#include <nntile/graph/logical/gelu_backward.hh>
#include <nntile/graph/logical/relu.hh>
#include <nntile/graph/logical/relu_inplace.hh>
#include <nntile/graph/logical/silu.hh>
#include <nntile/graph/logical/sqrt.hh>
#include <nntile/graph/logical/add.hh>
#include <nntile/graph/logical/add_inplace.hh>
#include <nntile/graph/logical/multiply.hh>
#include <nntile/graph/logical/multiply_inplace.hh>
#include <nntile/graph/logical/clear.hh>
#include <nntile/graph/logical/softmax.hh>
#include <nntile/graph/logical/softmax_inplace.hh>
#include <nntile/graph/logical/pow.hh>
#include <nntile/graph/logical/pow_inplace.hh>
#include <nntile/graph/logical/hypot.hh>
#include <nntile/graph/logical/hypot_inplace.hh>
#include <nntile/graph/logical/hypot_scalar_inverse.hh>
#include <nntile/graph/logical/fill.hh>
#include <nntile/graph/logical/copy.hh>
#include <nntile/graph/logical/transpose.hh>
#include <nntile/graph/logical/scatter.hh>
#include <nntile/graph/logical/copy_intersection.hh>
#include <nntile/graph/logical/log_scalar.hh>
#include <nntile/graph/logical/mask_scalar.hh>
#include <nntile/graph/logical/subtract_indexed_outputs.hh>
#include <nntile/graph/logical/gather.hh>
#include <nntile/graph/logical/scale_fiber.hh>
#include <nntile/graph/logical/scale_slice.hh>
#include <nntile/graph/logical/randn.hh>
#include <nntile/graph/logical/sum.hh>
#include <nntile/graph/logical/sum_fiber.hh>
#include <nntile/graph/logical/sum_slice.hh>
#include <nntile/graph/logical/norm.hh>
#include <nntile/graph/logical/logsumexp.hh>
#include <nntile/graph/logical/maxsumexp.hh>
#include <nntile/graph/logical/total_sum_accum.hh>
#include <nntile/graph/logical/sumprod_fiber.hh>
#include <nntile/graph/logical/sumprod_slice.hh>
#include <nntile/graph/logical/norm_fiber.hh>
#include <nntile/graph/logical/norm_fiber_inplace.hh>
#include <nntile/graph/logical/norm_slice.hh>
#include <nntile/graph/logical/norm_slice_inplace.hh>
#include <nntile/graph/logical/sgd_step.hh>
#include <nntile/graph/logical/adam_step.hh>
#include <nntile/graph/logical/adamw_step.hh>
#include <nntile/graph/logical/flash_sdpa_fwd_cudnn.hh>
#include <nntile/graph/logical/flash_sdpa_bwd_cudnn.hh>
#include <nntile/graph/logical/rope.hh>
#include <nntile/graph/logical/rope_backward.hh>
#include <nntile/graph/logical/conv2d_inplace.hh>
#include <nntile/graph/logical/conv2d_bwd_input_inplace.hh>
#include <nntile/graph/logical/conv2d_bwd_weight_inplace.hh>
#include <nntile/graph/logical/add_fiber.hh>
#include <nntile/graph/logical/add_fiber_inplace.hh>
#include <nntile/graph/logical/add_slice.hh>
#include <nntile/graph/logical/add_slice_inplace.hh>
#include <nntile/graph/logical/multiply_fiber.hh>
#include <nntile/graph/logical/multiply_fiber_inplace.hh>
#include <nntile/graph/logical/sqrt_inplace.hh>

namespace nntile::graph
{

//! Multiply slice operation: y = alpha * x * slice
//! @param x Input tensor
//! @param y Input/output tensor (modified in-place)
//! @param alpha Scaling factor (default: 1.0)
//! @param axis Axis along which to broadcast (default: -1, last axis)
void multiply_slice(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha = 1.0,
    Index axis = -1
);



//! Total sum accumulation: val = alpha * sum(logsumexp * src) + beta * val
//! @param logsumexp Log-sum-exp tensor
//! @param src Source tensor
//! @param class_labels Class labels tensor (int64)
//! @param val Output value tensor (fp32)
//! @param alpha Scaling factor (default: 1.0)
//! @param ignore_index Index to ignore (default: -1)


//! Sum of products along fibers: y = alpha * sum_fiber(x1 * x2) + beta * y
//! @param x1 First input tensor
//! @param x2 Second input tensor
//! @param y Output tensor to accumulate into
//! @param axis Axis along which to sum
//! @param redux Whether to use reduction (default: 0)
//! @param alpha Scaling factor for sum (default: 1.0)
//! @param beta Scaling factor for existing y (default: 0.0)

//! Sum of products along slices: y = alpha * sum_slice(x1 * x2) + beta * y
//! @param x1 First input tensor
//! @param x2 Second input tensor
//! @param y Output tensor to accumulate into
//! @param axis Axis along which to sum
//! @param redux Whether to use reduction (default: 0)
//! @param alpha Scaling factor for sum (default: 1.0)
//! @param beta Scaling factor for existing y (default: 0.0)

//! Norm along fibers: y = alpha * norm_fiber(x) + beta * y
//! @param x Input tensor
//! @param y Output tensor to accumulate into
//! @param axis Axis along which to compute norm
//! @param batch_ndim Number of trailing batch dimensions (default: 0)
//! @param redux Whether to use reduction (default: 0)
//! @param alpha Scaling factor for norm (default: 1.0)
//! @param beta Scaling factor for existing y (default: 0.0)

//! Norm along fibers (in-place): y = alpha * norm_fiber(x) + beta * y
//! @param x Input tensor
//! @param y Output tensor to accumulate into
//! @param axis Axis along which to compute norm
//! @param batch_ndim Number of trailing batch dimensions (default: 0)
//! @param redux Whether to use reduction (default: 0)
//! @param alpha Scaling factor for norm (default: 1.0)
//! @param beta Scaling factor for existing y (default: 0.0)

//! Norm along slices: y = alpha * norm_slice(x) + beta * y
//! @param x Input tensor
//! @param y Output tensor to accumulate into
//! @param axis Axis along which to compute norm
//! @param redux Whether to use reduction (default: 0)
//! @param alpha Scaling factor for norm (default: 1.0)
//! @param beta Scaling factor for existing y (default: 0.0)

//! Norm along slices (in-place): y = alpha * norm_slice(x) + beta * y
//! @param x Input tensor
//! @param y Output tensor to accumulate into
//! @param axis Axis along which to compute norm
//! @param redux Whether to use reduction (default: 0)
//! @param alpha Scaling factor for norm (default: 1.0)
//! @param beta Scaling factor for existing y (default: 0.0)


//! Scale operation: y = alpha * x
//! @param x Input tensor
//! @param output_name Name for the output tensor
//! @param alpha Scaling factor (default: 1.0)
//! @return Reference to the output tensor
LogicalGraph::TensorNode& scale(
    LogicalGraph::TensorNode& x,
    const std::string& output_name,
    Scalar alpha = 1.0
);

//! Scale in-place: x = alpha * x
//! @param x Input/output tensor (modified in-place)
//! @param alpha Scaling factor (default: 1.0)
void scale_inplace(
    LogicalGraph::TensorNode& x,
    Scalar alpha = 1.0
);

//! Embedding lookup: y = embedding(x, vocab)
//! @param index Index tensor (int64_t)
//! @param vocab Vocabulary tensor (float type)
//! @param output_name Name for the output tensor
//! @param axis Axis along which to perform embedding (default: 0)
//! @return Reference to the output tensor
LogicalGraph::TensorNode& embedding(
    LogicalGraph::TensorNode& index,
    LogicalGraph::TensorNode& vocab,
    const std::string& output_name,
    Index axis = 0
);

//! Embedding backward: vocab += embedding_backward(embed, index, vocab)
//! @param embed Embedding tensor (forward pass output)
//! @param index Index tensor (int64_t)
//! @param vocab Vocabulary tensor (modified in-place)
//! @param axis Axis along which to perform embedding (default: 0)
void embedding_backward(
    LogicalGraph::TensorNode& embed,
    LogicalGraph::TensorNode& index,
    LogicalGraph::TensorNode& vocab,
    Index axis = 0
);

//! Tensor contraction creating new output: C = alpha * op(A) @ op(B)
//! @param a First input tensor
//! @param b Second input tensor
//! @param output_name Name for the output tensor
//! @param alpha Scalar multiplier for A @ B (default: 1.0)
//! @param trans_a Swap M and K dimensions in A (default: false)
//! @param trans_b Swap K and N dimensions in B (default: false)
//! @param ndim Number of contraction dimensions (K) (default: 1)
//! @param batch_ndim Number of trailing batch dimensions (default: 0)
//! @return Reference to the created output tensor
LogicalGraph::TensorNode& gemm(
    LogicalGraph::TensorNode& a,
    LogicalGraph::TensorNode& b,
    const std::string& output_name,
    Scalar alpha = 1.0,
    bool trans_a = false,
    bool trans_b = false,
    Index ndim = 1,
    Index batch_ndim = 0
);

//! Tensor contraction with accumulation: C = alpha * op(A) @ op(B) + beta * C
//! @param a First input tensor
//! @param b Second input tensor
//! @param c Existing tensor to accumulate into (modified in-place)
//! @param alpha Scalar multiplier for A @ B (default: 1.0)
//! @param beta Scalar multiplier for existing C (default: 1.0)
//! @param trans_a Swap M and K dimensions in A (default: false)
//! @param trans_b Swap K and N dimensions in B (default: false)
//! @param ndim Number of contraction dimensions (K) (default: 1)
//! @param batch_ndim Number of trailing batch dimensions (default: 0)
void gemm(
    LogicalGraph::TensorNode& a,
    LogicalGraph::TensorNode& b,
    LogicalGraph::TensorNode& c,
    Scalar alpha = 1.0,
    Scalar beta = 1.0,
    bool trans_a = false,
    bool trans_b = false,
    Index ndim = 1,
    Index batch_ndim = 0
);



//! Flash attention forward pass (CUDA-only): A = flash_sdpa_fwd_cudnn(K, Q, mask, logsumexp, V)
//! @param K Key tensor
//! @param Q Query tensor
//! @param mask Attention mask tensor
//! @param logsumexp Log-sum-exp tensor (fp32)
//! @param V Value tensor
//! @param A Output attention tensor

//! Flash attention backward pass (CUDA-only): gradients w.r.t. K, Q, V
//! @param K Key tensor
//! @param Q Query tensor
//! @param V Value tensor
//! @param A Forward pass attention output
//! @param dA Gradient of attention output
//! @param mask Attention mask tensor
//! @param logsumexp Log-sum-exp tensor (fp32)
//! @param dK Gradient tensor for K (modified in-place)
//! @param dQ Gradient tensor for Q (modified in-place)
//! @param dV Gradient tensor for V (modified in-place)

//! Rotary position embedding: dst = rope(sin, cos, src)
//! @param sin_tensor Sine tensor for rotation
//! @param cos_tensor Cosine tensor for rotation
//! @param src Input tensor
//! @param dst Output tensor

//! Rotary position embedding backward: dx = rope_backward(sin, cos, dy)
//! @param sin_tensor Sine tensor for rotation
//! @param cos_tensor Cosine tensor for rotation
//! @param dy Gradient of output
//! @param dx Gradient of input (modified in-place)


} // namespace nntile::graph
