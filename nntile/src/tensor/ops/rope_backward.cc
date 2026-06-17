#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/rope_backward.cc
 * TensorGraph rope_backward operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/rope_backward.hh"

#include "nntile/dtype.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/tensor_graph_tiling.hh"
#include "nntile/tensor/tile_lowering_helpers.hh"
#include "nntile/tile/lowering_context.hh"
#include "nntile/tile/ops/rope_backward.hh"
#include "nntile/tensor/ops/rope_backward.hh"

#include <stdexcept>
#include <utility>

namespace nntile::tensor
{

namespace
{

inline Index graph_axis_to_storage(Index graph_axis, Index ndim)
{
    return ndim - 1 - graph_axis;
}

inline Index storage_axis_to_graph(Index storage_axis, Index ndim)
{
    return ndim - 1 - storage_axis;
}

} // anonymous namespace

TensorGraph::TensorNode *rope_backward(TensorGraph::TensorNode *sin,
    TensorGraph::TensorNode *cos,
    TensorGraph::TensorNode *dy)
{
    if (sin == nullptr || cos == nullptr || dy == nullptr)
    {
        throw std::invalid_argument(
            "rope_backward: input tensors must be non-null");
    }
    if (sin->graph() != cos->graph() || sin->graph() != dy->graph())
    {
        throw std::invalid_argument(
            "rope_backward: input tensors must belong to the same graph");
    }
    if (sin->dtype() != cos->dtype() || sin->dtype() != dy->dtype())
    {
        throw std::invalid_argument(
            "rope_backward: input tensors must have the same dtype");
    }

    TensorGraph::TensorNode *dx = dy->graph()->data(dy->shape(), dy->dtype());
    dx->set_axes(dy->axes());

    rope_backward(sin, cos, dy, dx);

    return dx;
}

void rope_backward(TensorGraph::TensorNode *sin,
    TensorGraph::TensorNode *cos,
    TensorGraph::TensorNode *dy,
    TensorGraph::TensorNode *dx)
{
    if (sin == nullptr || cos == nullptr || dy == nullptr || dx == nullptr)
    {
        throw std::invalid_argument(
            "rope_backward: input tensors must be non-null");
    }
    if (sin->graph() != cos->graph() || sin->graph() != dy->graph() ||
        sin->graph() != dx->graph())
    {
        throw std::invalid_argument(
            "rope_backward: input tensors must belong to the same graph");
    }
    if (sin->dtype() != cos->dtype() || sin->dtype() != dy->dtype() ||
        sin->dtype() != dx->dtype())
    {
        throw std::invalid_argument(
            "rope_backward: input tensors must have the same dtype");
    }
    validate_same_shape_and_merge(dy, dx, "rope_backward");

    auto op = std::make_shared<TensorRopeBackwardOp>(sin, cos, dy, dx);
    dy->graph()->add_op(op);
}

void TensorRopeBackwardOp::lower_to_tile(const LoweringContext &ctx) const
{
    // Match nntile::tensor::rope_backward_async (src/tensor/rope_backward.cc).
    tile_lower::assert_same_elementwise_layout(dy, dx, "ROPE_BACKWARD dy/dx");

    const TensorAxisLayout *lay_dy = ctx.tiling.find(dy);
    const TensorAxisLayout *lay_sin = ctx.tiling.find(sin);
    if (lay_dy == nullptr || lay_sin == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile ROPE_BACKWARD: missing tiling for dy and/or sin");
    }

    const auto &tiles_sin = tile_lower::tiles_of(ctx.tile_map, sin);
    const auto &tiles_cos = tile_lower::tiles_of(ctx.tile_map, cos);
    const auto &tiles_dy = tile_lower::tiles_of(ctx.tile_map, dy);
    const auto &tiles_dx = tile_lower::tiles_of(ctx.tile_map, dx);

    const Index sin_ndim = sin->ndim();
    const Index dy_ndim = dy->ndim();
    std::vector<Index> dydx_coord;
    std::vector<Index> sincos_coord(static_cast<size_t>(sin_ndim));

    for (Index lin = 0; lin < lay_dy->grid_volume(); ++lin)
    {
        lay_dy->grid_coord_from_linear(lin, dydx_coord);
        for (Index d = 0; d < sin_ndim; ++d)
        {
            const Index g = storage_axis_to_graph(d, sin_ndim);
            const Index dy_s = graph_axis_to_storage(g, dy_ndim);
            sincos_coord[static_cast<size_t>(d)] =
                dydx_coord[static_cast<size_t>(dy_s)];
        }
        const Index j = lay_sin->grid_linear(sincos_coord);
        tile::rope_backward(tiles_sin[static_cast<size_t>(j)],
            tiles_cos[static_cast<size_t>(j)],
            tiles_dy[static_cast<size_t>(lin)],
            tiles_dx[static_cast<size_t>(lin)]);
    }
}

} // namespace nntile::tensor
