/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/logical/rope.cc
 * Logical graph RoPE operation.
 *
 * @version 1.1.0
 * */

// Include corresponding header
#include "nntile/graph/logical/rope.hh"

// Include standard headers
#include <stdexcept>
#include <utility>

// Include other NNTile headers
#include <nntile/graph/logical_graph.hh>

namespace nntile::graph
{

//! Rotary position embedding: dst = rope(sin, cos, src)
void rope(
    LogicalGraph::TensorNode& sin_tensor,
    LogicalGraph::TensorNode& cos_tensor,
    LogicalGraph::TensorNode& src,
    LogicalGraph::TensorNode& dst)
{
    if(&sin_tensor.graph() != &cos_tensor.graph() || &sin_tensor.graph() != &src.graph() || &sin_tensor.graph() != &dst.graph())
    {
        throw std::invalid_argument(
            "rope: tensors must belong to the same graph");
    }

    if(sin_tensor.dtype() != cos_tensor.dtype() || sin_tensor.dtype() != src.dtype() || sin_tensor.dtype() != dst.dtype())
    {
        throw std::invalid_argument(
            "rope: all tensors must have the same dtype");
    }

    auto attrs = std::make_shared<ClearAttrs>(ClearAttrs{});
    sin_tensor.graph().add_op(
        OpType::ROPE,
        attrs,
        {&sin_tensor, &cos_tensor, &src},
        {&dst}
    );
}

} // namespace nntile::graph
