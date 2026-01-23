/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical_graph_ops.cc
 * Logical graph operations.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical_graph_ops.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

namespace nntile::graph
{













//! Scale along fibers: y = alpha * scale_fiber(x, y)

//! Scale along slices: y = alpha * scale_slice(x, y)

//! Random normal generation: x = randn(start, underlying_shape, seed, mean, stddev)

//! SGD step: p = sgd_step(grad, velocity, p)

//! Adam step: p = adam_step(grad, first_moment, second_moment, p)

//! AdamW step: p = adamw_step(grad, first_moment, second_moment, p)

//! Flash attention forward pass (CUDA-only): A = flash_sdpa_fwd_cudnn(K, Q, mask, logsumexp, V)

//! Flash attention backward pass (CUDA-only): gradients w.r.t. K, Q, V = flash_sdpa_bwd_cudnn(...)

//! Rotary position embedding: dst = rope(sin, cos, src)

//! Rotary position embedding backward: dx = rope_backward(sin, cos, dy)

//! Total sum of all elements: y = alpha * sum(x) + beta * y

//! Sum along fibers: y = alpha * sum_fiber(x) + beta * y

//! Sum along slices: y = alpha * sum_slice(x) + beta * y

//! Total sum accumulation: val = alpha * sum(logsumexp * src) + beta * val

//! Euclidean norm: y = alpha * norm(x) + beta * y

//! Log sum exp along axis: y = log(sum(exp(x)))

//! Max and sum of exponents along axis: y = max + log(sum(exp(x - max)))

//! Sum of products along fibers: y = alpha * sum_fiber(x1 * x2) + beta * y

//! Sum of products along slices: y = alpha * sum_slice(x1 * x2) + beta * y

//! Norm along fibers: y = alpha * norm_fiber(x) + beta * y

//! Norm along fibers (in-place): y = alpha * norm_fiber(x) + beta * y

//! Norm along slices: y = alpha * norm_slice(x) + beta * y

//! Norm along slices (in-place): y = alpha * norm_slice(x) + beta * y

//! 2D Convolution forward: Y = alpha * conv2d(X, C) + beta * Y

//! 2D Convolution backward w.r.t. input: dX = alpha * conv2d_bwd_input(dY, C) + beta * dX

//! 2D Convolution backward w.r.t. weights: dC = alpha * conv2d_bwd_weight(X, dY) + beta * dC




} // namespace nntile::graph
