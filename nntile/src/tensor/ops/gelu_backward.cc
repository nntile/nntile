/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/gelu_backward.cc
 * TensorGraph GeLU backward operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/gelu_backward.hh"

#include <stdexcept>

#include "nntile/dtype.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/ops/gelu_backward.hh"

#include <nntile/tile/graph_ops.hh>
#include <nntile/tensor/tile_lowering_helpers.hh>

namespace nntile::tensor
{

TensorGraph::TensorNode* gelu_backward(
    Scalar alpha, TensorGraph::TensorNode* x, TensorGraph::TensorNode* dy)
{
    if(x == nullptr || dy == nullptr)
    {
        throw std::invalid_argument(
            "gelu_backward: input tensors must be non-null");
    }
    if(x->graph() != dy->graph())
    {
        throw std::invalid_argument(
            "gelu_backward: input tensors must belong to the same graph");
    }
    if(x->dtype() != dy->dtype())
    {
        throw std::invalid_argument(
            "gelu_backward: input tensors must have the same dtype");
    }
    if(x == dy)
    {
        throw std::invalid_argument(
            "gelu_backward: x and dy must be distinct tensors");
    }
    validate_same_shape_and_merge(x, dy, "gelu_backward");

    TensorGraph::TensorNode* dx = x->graph()->emplace_data(x->shape(), x->dtype());
    dx->set_axes(x->axes());

    gelu_backward(alpha, x, dy, Scalar{0.0}, dx);

    return dx;
}

void gelu_backward(
    Scalar alpha,
    TensorGraph::TensorNode* x,
    TensorGraph::TensorNode* dy,
    Scalar beta,
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

    auto op = std::make_shared<TensorGeluBackwardOp>(x, dy, dx, alpha, beta);
    x->graph()->add_op(op);
}

void TensorGeluBackwardOp::lower_to_tile(const LoweringContext& ctx) const
{
    const auto& m = ctx.tile_map;
    const auto& vx = tile_lower::tiles_of(m, x);
    const auto& vdy = tile_lower::tiles_of(m, dy);
    const auto& vdx = tile_lower::tiles_of(m, dx);
    if(vx.size() != vdy.size() || vx.size() != vdx.size())
    {
        throw std::runtime_error(
            "lower_to_tile GELU_BACKWARD: tile count mismatch");
    }
    tile_lower::assert_same_elementwise_layout(x, dy, "GELU_BACKWARD x/dy");
    tile_lower::assert_same_elementwise_layout(x, dx, "GELU_BACKWARD x/dx");
    for(size_t i = 0; i < vx.size(); ++i)
    {
        tile::gelu_backward(alpha, vx[i], vdy[i], beta, vdx[i]);
    }
}

} // namespace nntile::tensor
