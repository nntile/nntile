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

namespace nntile::graph
{

//! Clear tensor: x = 0
//! @param x Tensor to clear (modified in-place)
void clear(LogicalGraph::TensorNode& x);

//! GeLU activation: y = gelu(x)
//! @param x Input tensor
//! @param output_name Name for the output tensor
//! @return Reference to the output tensor
LogicalGraph::TensorNode& gelu(
    LogicalGraph::TensorNode& x,
    const std::string& output_name
);

//! GeLU in-place: x = gelu(x)
//! @param x Input/output tensor (modified in-place)
void gelu_inplace(LogicalGraph::TensorNode& x);

//! GeLU tanh activation: y = gelu_tanh(x)
//! @param x Input tensor
//! @param output_name Name for the output tensor
//! @return Reference to the output tensor
LogicalGraph::TensorNode& gelutanh(
    LogicalGraph::TensorNode& x,
    const std::string& output_name
);

//! GeLU tanh in-place: x = gelu_tanh(x)
//! @param x Input/output tensor (modified in-place)
void gelutanh_inplace(LogicalGraph::TensorNode& x);

//! ReLU activation: y = relu(x)
//! @param x Input tensor
//! @param output_name Name for the output tensor
//! @return Reference to the output tensor
LogicalGraph::TensorNode& relu(
    LogicalGraph::TensorNode& x,
    const std::string& output_name
);

//! ReLU in-place: x = relu(x)
//! @param x Input/output tensor (modified in-place)
void relu_inplace(LogicalGraph::TensorNode& x);

//! SiLU activation: y = silu(x)
//! @param x Input tensor
//! @param output_name Name for the output tensor
//! @return Reference to the output tensor
LogicalGraph::TensorNode& silu(
    LogicalGraph::TensorNode& x,
    const std::string& output_name
);

//! SiLU in-place: x = silu(x)
//! @param x Input/output tensor (modified in-place)
void silu_inplace(LogicalGraph::TensorNode& x);

//! Sqrt activation: y = sqrt(x)
//! @param x Input tensor
//! @param output_name Name for the output tensor
//! @return Reference to the output tensor
LogicalGraph::TensorNode& sqrt(
    LogicalGraph::TensorNode& x,
    const std::string& output_name
);

//! Sqrt in-place: x = sqrt(x)
//! @param x Input/output tensor (modified in-place)
void sqrt_inplace(LogicalGraph::TensorNode& x);

//! GeLU backward: dx += gelu_backward(x, dy)
//! @param x Input tensor (forward pass activation)
//! @param dy Gradient of output (upstream gradient)
//! @param dx Gradient tensor to accumulate into (gradient of input)
void gelu_backward(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& dy,
    LogicalGraph::TensorNode& dx
);

//! GeLU tanh backward: dx += gelu_tanh_backward(x, dy)
//! @param x Input tensor (forward pass activation)
//! @param dy Gradient of output (upstream gradient)
//! @param dx Gradient tensor to accumulate into (gradient of input)
void gelutanh_backward(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& dy,
    LogicalGraph::TensorNode& dx
);

//! ReLU backward: dx += relu_backward(x, dy)
//! @param x Input tensor (forward pass activation)
//! @param dy Gradient of output (upstream gradient)
//! @param dx Gradient tensor to accumulate into (gradient of input)
void relu_backward(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& dy,
    LogicalGraph::TensorNode& dx
);

//! SiLU backward: dx += silu_backward(x, dy)
//! @param x Input tensor (forward pass activation)
//! @param dy Gradient of output (upstream gradient)
//! @param dx Gradient tensor to accumulate into (gradient of input)
void silu_backward(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& dy,
    LogicalGraph::TensorNode& dx
);

//! Add operation: z = alpha * x + beta * y
//! @param x First input tensor
//! @param y Second input tensor
//! @param output_name Name for the output tensor
//! @param alpha Scaling factor for x (default: 1.0)
//! @param beta Scaling factor for y (default: 1.0)
//! @return Reference to the output tensor
LogicalGraph::TensorNode& add(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    const std::string& output_name,
    Scalar alpha = 1.0,
    Scalar beta = 1.0
);

//! Add in-place: y = alpha * x + beta * y
//! @param x First input tensor
//! @param y Second input/output tensor (modified in-place)
//! @param alpha Scaling factor for x (default: 1.0)
//! @param beta Scaling factor for y (default: 1.0)
void add_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha = 1.0,
    Scalar beta = 1.0
);

//! Multiply operation: z = x * y
//! @param x First input tensor
//! @param y Second input tensor
//! @param output_name Name for the output tensor
//! @return Reference to the output tensor
LogicalGraph::TensorNode& multiply(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    const std::string& output_name
);

//! Multiply in-place: y = x * y
//! @param x First input tensor
//! @param y Second input/output tensor (modified in-place)
void multiply_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y
);

//! Total sum of all elements: y = alpha * sum(x) + beta * y
//! @param x Input tensor
//! @param y Output tensor to accumulate into
//! @param alpha Scaling factor for sum (default: 1.0)
//! @param beta Scaling factor for existing y (default: 0.0)
void sum(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha = 1.0,
    Scalar beta = 0.0
);

//! Sum along fibers: y = alpha * sum_fiber(x) + beta * y
//! @param x Input tensor
//! @param y Output tensor to accumulate into
//! @param axis Axis along which to sum (default: 0)
//! @param batch_ndim Number of trailing batch dimensions (default: 0)
//! @param redux Whether to use reduction (default: 0)
//! @param alpha Scaling factor for sum (default: 1.0)
//! @param beta Scaling factor for existing y (default: 0.0)
void sum_fiber(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Index axis = 0,
    Index batch_ndim = 0,
    int redux = 0,
    Scalar alpha = 1.0,
    Scalar beta = 0.0
);

//! Sum along slices: y = alpha * sum_slice(x) + beta * y
//! @param x Input tensor
//! @param y Output tensor to accumulate into
//! @param axis Axis along which to sum
//! @param redux Whether to use reduction (default: 0)
//! @param alpha Scaling factor for sum (default: 1.0)
//! @param beta Scaling factor for existing y (default: 0.0)
void sum_slice(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Index axis,
    int redux = 0,
    Scalar alpha = 1.0,
    Scalar beta = 0.0
);

//! Euclidean norm: y = alpha * norm(x) + beta * y
//! @param x Input tensor
//! @param y Output tensor to accumulate into (must be scalar)
//! @param alpha Scaling factor for norm (default: 1.0)
//! @param beta Scaling factor for existing y (default: 0.0)
void norm(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha = 1.0,
    Scalar beta = 0.0
);

//! Log sum exp along axis: y = log(sum(exp(x)))
//! @param x Input tensor
//! @param y Output tensor
//! @param axis Axis along which to compute logsumexp
void logsumexp(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Index axis
);

//! Max and sum of exponents along axis: y = max + log(sum(exp(x - max)))
//! @param x Input tensor
//! @param y Output tensor
//! @param axis Axis along which to compute maxsumexp
//! @param redux Whether to use reduction (default: 0)
void maxsumexp(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Index axis,
    int redux = 0
);

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

//! Hypot operation: z = hypot(alpha * x, beta * y)
//! @param x First input tensor
//! @param y Second input tensor
//! @param output_name Name for the output tensor
//! @param alpha Scaling factor for x (default: 1.0)
//! @param beta Scaling factor for y (default: 1.0)
//! @return Reference to the output tensor
LogicalGraph::TensorNode& hypot(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    const std::string& output_name,
    Scalar alpha = 1.0,
    Scalar beta = 1.0
);

//! Hypot in-place: y = hypot(alpha * x, beta * y)
//! @param x First input tensor
//! @param y Second input/output tensor (modified in-place)
//! @param alpha Scaling factor for x (default: 1.0)
//! @param beta Scaling factor for y (default: 1.0)
void hypot_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha = 1.0,
    Scalar beta = 1.0
);

//! Power operation: y = alpha * (x ^ exp)
//! @param x Input tensor
//! @param output_name Name for the output tensor
//! @param alpha Scaling factor (default: 1.0)
//! @param exp Exponent (default: 1.0)
//! @return Reference to the output tensor
LogicalGraph::TensorNode& pow(
    LogicalGraph::TensorNode& x,
    const std::string& output_name,
    Scalar alpha = 1.0,
    Scalar exp = 1.0
);

//! Power in-place: x = alpha * (x ^ exp)
//! @param x Input/output tensor (modified in-place)
//! @param alpha Scaling factor (default: 1.0)
//! @param exp Exponent (default: 1.0)
void pow_inplace(
    LogicalGraph::TensorNode& x,
    Scalar alpha = 1.0,
    Scalar exp = 1.0
);

//! Log scalar operation: log value with given name
//! @param x Input tensor
//! @param name Name for logging
void log_scalar(
    LogicalGraph::TensorNode& x,
    const std::string& name
);

//! Mask scalar operation: conditionally set values based on mask
//! @param mask Boolean mask tensor
//! @param x Input/output tensor (modified in-place)
//! @param val Value to set where mask is true (default: 0.0)
//! @param batch_ndim Number of batch dimensions (default: 0)
void mask_scalar(
    LogicalGraph::TensorNode& mask,
    LogicalGraph::TensorNode& x,
    Scalar val = 0.0,
    Index batch_ndim = 0
);

//! Fill operation: x = val
//! @param x Input/output tensor (modified in-place)
//! @param val Value to fill tensor with
void fill(
    LogicalGraph::TensorNode& x,
    Scalar val = 0.0
);

//! Copy operation: y = x
//! @param x Input tensor
//! @param output_name Name for the output tensor
//! @return Reference to the output tensor
LogicalGraph::TensorNode& copy(
    LogicalGraph::TensorNode& x,
    const std::string& output_name
);

//! Transpose operation: y = alpha * transpose(x)
//! @param x Input tensor
//! @param output_name Name for the output tensor
//! @param alpha Scaling factor (default: 1.0)
//! @param ndim Number of dimensions to transpose (default: 0)
//! @return Reference to the output tensor
LogicalGraph::TensorNode& transpose(
    LogicalGraph::TensorNode& x,
    const std::string& output_name,
    Scalar alpha = 1.0,
    Index ndim = 0
);

//! Gather operation: y = gather(x)
//! @param x Input tensor
//! @param output_name Name for the output tensor
//! @return Reference to the output tensor
LogicalGraph::TensorNode& gather(
    LogicalGraph::TensorNode& x,
    const std::string& output_name
);

//! Hypot scalar inverse operation: y = 1.0 / hypot(eps, alpha * y)
//! @param x Input/output tensor (modified in-place)
//! @param eps Epsilon value for numerical stability
//! @param alpha Scaling factor
void hypot_scalar_inverse(
    LogicalGraph::TensorNode& x,
    Scalar eps = 0.0,
    Scalar alpha = 1.0
);

//! Subtract indexed outputs operation: subtract val from elements indexed by labels
//! @param labels Index tensor (int64_t) indicating which elements to modify
//! @param x Input/output tensor (modified in-place)
//! @param val Value to subtract
//! @param ignore_index Index value to ignore (-1 by default)
void subtract_indexed_outputs(
    LogicalGraph::TensorNode& labels,
    LogicalGraph::TensorNode& x,
    Scalar val = 0.0,
    Index ignore_index = -1
);

} // namespace nntile::graph
