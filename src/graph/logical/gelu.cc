/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/gelu.cc
 * GELU operation implementation for logical graph.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/gelu.hh"

// Include standard headers
#include <utility>

// Include third-party headers

// Include other NNTile headers
#include "nntile/graph/logical_graph.hh"

namespace nntile::graph
{

//! GeLU activation: y = gelu(x)
LogicalGraph::TensorNode& gelu(
    LogicalGraph::TensorNode& x,
    const std::string& output_name)
{
    // Output shape = input shape
    std::vector<Index> output_shape = x.shape();

    // Create output tensor
    LogicalGraph::TensorNode& output = x.graph().tensor(
        std::move(output_shape),
        output_name,
        x.dtype());

    // Create operation attributes
    OpAttrs attrs = GeluAttrs{};

    // Add operation to graph using public builder API
    x.graph().add_op(
        OpType::GELU,
        attrs,
        {&x},
        {&output}
    );

    return output;
}

} // namespace nntile::graph
