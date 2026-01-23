/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/arithmetic.hh
 * Logical graph arithmetic operations.
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

//! Add fiber operation: z = alpha * fiber + beta * x
//! @param x Input tensor
//! @param y Fiber tensor
//! @param output_name Name for the output tensor
//! @param alpha Scaling factor for fiber (default: 1.0)
//! @param beta Scaling factor for x (default: 1.0)
//! @param axis Axis along which to broadcast (default: -1, last axis)
//! @param batch_ndim Number of batch dimensions (default: 0)
//! @return Reference to the output tensor
LogicalGraph::TensorNode& add_fiber(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    const std::string& output_name,
    Scalar alpha = 1.0,
    Scalar beta = 1.0,
    Index axis = -1,
    Index batch_ndim = 0
);

//! Add fiber in-place: y = alpha * fiber + beta * y
//! @param x Fiber tensor
//! @param y Input/output tensor (modified in-place)
//! @param alpha Scaling factor for fiber (default: 1.0)
//! @param beta Scaling factor for y (default: 1.0)
//! @param axis Axis along which to broadcast (default: -1, last axis)
//! @param batch_ndim Number of batch dimensions (default: 0)
void add_fiber_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha = 1.0,
    Scalar beta = 1.0,
    Index axis = -1,
    Index batch_ndim = 0
);

//! Add slice operation: z = alpha * slice + beta * x
//! @param x Input tensor
//! @param y Slice tensor
//! @param output_name Name for the output tensor
//! @param alpha Scaling factor for slice (default: 1.0)
//! @param beta Scaling factor for x (default: 1.0)
//! @param axis Axis along which to broadcast (default: -1, last axis)
//! @return Reference to the output tensor
LogicalGraph::TensorNode& add_slice(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    const std::string& output_name,
    Scalar alpha = 1.0,
    Scalar beta = 1.0,
    Index axis = -1
);

//! Add slice in-place: y = alpha * slice + beta * y
//! @param x Slice tensor
//! @param y Input/output tensor (modified in-place)
//! @param alpha Scaling factor for slice (default: 1.0)
//! @param beta Scaling factor for y (default: 1.0)
//! @param axis Axis along which to broadcast (default: -1, last axis)
void add_slice_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha = 1.0,
    Scalar beta = 1.0,
    Index axis = -1
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

//! Hypot scalar inverse operation: y = 1.0 / hypot(eps, alpha * y)
//! @param x Input/output tensor (modified in-place)
//! @param eps Epsilon value for numerical stability
//! @param alpha Scaling factor
void hypot_scalar_inverse(
    LogicalGraph::TensorNode& x,
    Scalar eps = 0.0,
    Scalar alpha = 1.0
);

//! Multiply fiber operation: z = alpha * x * y
//! @param x Input tensor
//! @param y Fiber tensor
//! @param output_name Name for the output tensor
//! @param alpha Scaling factor (default: 1.0)
//! @param axis Axis along which to broadcast (default: -1, last axis)
//! @return Reference to the output tensor
LogicalGraph::TensorNode& multiply_fiber(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    const std::string& output_name,
    Scalar alpha = 1.0,
    Index axis = -1
);

//! Multiply fiber in-place: y = alpha * y * fiber
//! @param x Fiber tensor
//! @param y Input/output tensor (modified in-place)
//! @param alpha Scaling factor (default: 1.0)
//! @param axis Axis along which to broadcast (default: -1, last axis)
void multiply_fiber_inplace(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Scalar alpha = 1.0,
    Index axis = -1
);

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

} // namespace nntile::graph