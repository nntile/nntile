/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/maxsumexp.hh
 * Logical graph maxsumexp operation.
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

} // namespace nntile::graph