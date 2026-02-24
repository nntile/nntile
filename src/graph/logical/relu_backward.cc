/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/relu_backward.cc
 * Logical graph ReLU backward operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/relu_backward.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! ReLU backward: dx += relu_backward(x, dy)
void relu_backward(
    LogicalGraph::TensorNode& x,
    LogicalGraph::TensorNode& dy,
    LogicalGraph::TensorNode& dx)
{
    // Create operation attributes
    OpAttrs attrs = ReluBackwardAttrs{};

    // Add operation to graph using public builder API
    // Note: dx is both input and output (accumulates gradients)
    x.graph().add_op(
        OpType::RELU_BACKWARD,
        attrs,
        {&x, &dy, &dx},
        {&dx}
    );
}

} // namespace nntile::graph
