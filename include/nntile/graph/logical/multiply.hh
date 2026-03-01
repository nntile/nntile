/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/multiply.hh
 * Logical graph multiply operation.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// Include other NNTile headers
#include <nntile/graph/logical/add.hh>
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

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

} // namespace nntile::graph
