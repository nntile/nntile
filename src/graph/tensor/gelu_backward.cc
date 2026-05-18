/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/gelu_backward.cc
 * TensorGraph GeLU backward operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/gelu_backward.hh"

#include <stdexcept>

#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/gelu_backward.hh"

#include <nntile/graph/tile/graph_ops.hh>
#include <nntile/graph/tensor/tile_lowering_helpers.hh>

namespace nntile::graph::tensor
{



void gelu_backward(
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* dy,
    TensorGraph::TensorNode* dx)
{
    if(x == nullptr || dy == nullptr || dx == nullptr)
    {
        throw std::invalid_argument(
            "gelu_backward: input tensors must be non-null");
    }
    if(x->graph() != dy->graph() || x->graph() != dx->graph())
    {
        throw std::invalid_argument(
            "gelu_backward: input tensors must belong to the same graph");
    }
    if(x->dtype() != dy->dtype() || x->dtype() != dx->dtype())
    {
        throw std::invalid_argument(
            "gelu_backward: input tensors must have the same dtype");
    }
    if(x == dy || x == dx || dy == dx)
    {
        throw std::invalid_argument(
            "gelu_backward: x, dy, and dx must be distinct tensors");
    }
    validate_same_shape_and_merge(x, dy, "gelu_backward");
    validate_same_shape_and_merge(x, dx, "gelu_backward");

    auto op = std::make_shared<TensorGeluBackwardOp>(x, dy, dx);
    x->graph()->add_op(op);
}

void TensorGeluBackwardOp::lower_to_tile(const LoweringContext& ctx) const
{
    tile_lower::lower_backward3(x, dy, dx, ctx.tile_map, "GELU_BACKWARD",
        tile_graph::gelu_backward);
}

} // namespace nntile::graph::tensor
