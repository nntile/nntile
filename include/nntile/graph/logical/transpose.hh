/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/transpose.hh
 * Logical graph transpose operation.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// Include other NNTile headers
#include <nntile/base_types.hh>
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

struct TransposeAttrs
{
    Scalar alpha = 1.0;
    Index ndim = 0;
};

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

} // namespace nntile::graph
