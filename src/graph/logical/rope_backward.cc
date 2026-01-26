/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/rope_backward.cc
 * Logical graph RoPE backward operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/rope_backward.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Rotary position embedding backward: dx = rope_backward(sin, cos, dy)
void rope_backward(
    LogicalGraph::TensorNode& sin_tensor,
    LogicalGraph::TensorNode& cos_tensor,
    LogicalGraph::TensorNode& dy,
    LogicalGraph::TensorNode& dx)
{
    if(&sin_tensor.graph() != &cos_tensor.graph() || &sin_tensor.graph() != &dy.graph() || &sin_tensor.graph() != &dx.graph())
    {
        throw std::invalid_argument(
            "rope_backward: tensors must belong to the same graph");
    }

    if(sin_tensor.dtype() != cos_tensor.dtype() || sin_tensor.dtype() != dy.dtype() || sin_tensor.dtype() != dx.dtype())
    {
        throw std::invalid_argument(
            "rope_backward: all tensors must have the same dtype");
    }

    OpAttrs attrs = ClearAttrs{};  // No additional attributes needed
    sin_tensor.graph().add_op(
        OpType::ROPE_BACKWARD,
        attrs,
        {&sin_tensor, &cos_tensor, &dy, &dx},
        {&dx}
    );
}

} // namespace nntile::graph