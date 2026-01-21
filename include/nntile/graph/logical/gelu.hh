/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file include/nntile/graph/logical/gelu.hh
 * GELU operation for logical graph.
 *
 * @version 1.1.0
 * */

#pragma once

// Include standard headers
#include <string>

// Include third-party headers

// Include other NNTile headers
#include <nntile/graph/tensor_node.hh>

namespace nntile::graph
{

//! GeLU activation: y = gelu(x)
//! @param x Input tensor
//! @param output_name Name for the output tensor
//! @return Reference to the output tensor
LogicalGraphTensorNode& gelu(
    LogicalGraphTensorNode& x,
    const std::string& output_name
);

} // namespace nntile::graph
