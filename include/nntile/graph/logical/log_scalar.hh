/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/log_scalar.hh
 * Logical graph log_scalar operation.
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

struct LogScalarAttrs
{
    std::string name;
};

//! Log scalar operation: log value with given name
//! @param x Input tensor
//! @param name Name for logging
void log_scalar(
    LogicalGraph::TensorNode& x,
    const std::string& name
);

} // namespace nntile::graph
