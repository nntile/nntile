/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/logsumexp.hh
 * Logical graph logsumexp operation.
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

//! Log sum exp from maxsumexp output: y = max + log(sumexp)
//! @param x Input tensor from maxsumexp (shape [2, ...], x[0]=max, x[1]=sumexp)
//! @param y Output tensor
//! @param axis Axis that was reduced by maxsumexp
void logsumexp(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& y,
    Index axis
);

} // namespace nntile::graph
