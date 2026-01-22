/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/clear.hh
 * CLEAR operation for logical graph.
 *
 * @version 1.1.0
 * */

#pragma once

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Clear tensor: x = 0
//! @param x Tensor to clear (modified in-place)
void clear(LogicalGraph::TensorNode& x);

} // namespace nntile::graph
