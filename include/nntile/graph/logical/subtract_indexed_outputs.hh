/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/subtract_indexed_outputs.hh
 * Logical graph subtract_indexed_outputs operation.
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
