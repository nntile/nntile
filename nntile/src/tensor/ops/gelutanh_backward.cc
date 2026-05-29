/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/gelutanh_backward.cc
 * TensorGraph gelutanh_backward operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/gelutanh_backward.hh"

#include "nntile/dtype.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/ops/gelutanh_backward.hh"

#include <nntile/tensor/tile_lowering_helpers.hh>
#include <nntile/tile/graph_ops.hh>
#include <stdexcept>
#include <utility>

namespace nntile::tensor
{

TensorGraph::TensorNode *gelutanh_backward(
    TensorGraph::TensorNode *x, TensorGraph::TensorNode *dy)
{
    if (x == nullptr || dy == nullptr)
    {
        throw std::invalid_argument(
            "gelutanh_backward: input tensors must be non-null");
    }
    if (x->graph() != dy->graph())
    {
        throw std::invalid_argument(
            "gelutanh_backward: input tensors must belong to the same graph");
    }
    if (x->dtype() != dy->dtype())
    {
        throw std::invalid_argument(
            "gelutanh_backward: input tensors must have the same dtype");
    }
    if (x == dy)
    {
        throw std::invalid_argument(
            "gelutanh_backward: x and dy must be distinct tensors");
    }
    validate_same_shape_and_merge(x, dy, "gelutanh_backward");

    TensorGraph::TensorNode *dx = x->graph()->data(x->shape(), x->dtype());
    dx->set_axes(x->axes());

    gelutanh_backward(x, dy, dx);

    return dx;
}

void gelutanh_backward(TensorGraph::TensorNode *x,
    TensorGraph::TensorNode *dy,
    TensorGraph::TensorNode *dx)
{
    if (x == nullptr || dy == nullptr || dx == nullptr)
    {
        throw std::invalid_argument(
            "gelutanh_backward: input tensors must be non-null");
    }
    if (x->graph() != dy->graph() || x->graph() != dx->graph())
    {
        throw std::invalid_argument(
            "gelutanh_backward: input tensors must belong to the same graph");
    }
    if (x->dtype() != dy->dtype() || x->dtype() != dx->dtype())
    {
        throw std::invalid_argument(
            "gelutanh_backward: input tensors must have the same dtype");
    }
    if (x == dy || x == dx || dy == dx)
    {
        throw std::invalid_argument(
            "gelutanh_backward: x, dy, and dx must be distinct tensors");
    }
    validate_same_shape_and_merge(x, dy, "gelutanh_backward");
    validate_same_shape_and_merge(x, dx, "gelutanh_backward");

    auto op = std::make_shared<TensorGelutanhBackwardOp>(x, dy, dx);
    x->graph()->add_op(op);
}

void TensorGelutanhBackwardOp::lower_to_tile(const LoweringContext &ctx) const
{
    tile_lower::lower_backward3(x,
        dy,
        dx,
        ctx.tile_map,
        "GELUTANH_BACKWARD",
        tile_graph::gelutanh_backward);
}

} // namespace nntile::tensor
