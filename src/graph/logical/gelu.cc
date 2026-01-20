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

// Include third-party headers

// Include other NNTile headers
#include "nntile/graph/logical_graph.hh"
#include "nntile/graph/op_node.hh"

namespace nntile::graph
{

//! GeLU activation: y = gelu(x)
TensorNode& gelu(
    TensorNode& x,
    const std::string& output_name)
{
    // Output shape = input shape
    TensorSpec output_spec = TensorSpec(x.shape(), x.dtype());

    // Create operation attributes
    OpAttrs attrs = GeluAttrs{};

    // Add operation to graph using public builder API
    return x.graph().add_op(
        OpType::GELU,
        attrs,
        {&x},
        output_spec,
        output_name
    );
}

} // namespace nntile::graph
