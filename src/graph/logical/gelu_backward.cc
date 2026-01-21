/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/gelu_backward.cc
 * GELU_BACKWARD operation implementation for logical graph.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/gelu_backward.hh"

// Include standard headers

// Include third-party headers

// Include other NNTile headers
#include "nntile/graph/logical_graph.hh"
#include "nntile/graph/op_node.hh"

namespace nntile::graph
{

//! GeLU backward: dx += gelu_backward(x, dy)
void gelu_backward(
    TensorNode& x,
    TensorNode& dy,
    TensorNode& dx)
{
    // Create operation attributes
    OpAttrs attrs = GeluBackwardAttrs{};

    // Add operation to graph using public builder API
    // Note: dx is both input and output (accumulates gradients)
    x.graph().add_op(
        OpType::GELU_BACKWARD,
        attrs,
        {&x, &dy, &dx},
        {&dx}
    );
}

} // namespace nntile::graph