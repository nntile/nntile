/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/rope.hh
 * Logical graph RoPE operation.
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

//! Rotary position embedding: dst = rope(sin, cos, src)
//! @param sin_tensor Sine tensor for rotation
//! @param cos_tensor Cosine tensor for rotation
//! @param src Input tensor
//! @param dst Output tensor
void rope(
    LogicalGraph::TensorNode& sin_tensor,
    LogicalGraph::TensorNode& cos_tensor,
    LogicalGraph::TensorNode& src,
    LogicalGraph::TensorNode& dst
);

} // namespace nntile::graph