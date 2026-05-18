/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/scale_fiber.cc
 * TensorGraph scale_fiber operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/scale_fiber.hh"

#include <stdexcept>
#include <utility>

#include "nntile/base_types.hh"
#include "nntile/graph/dtype.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/tensor/scale_fiber.hh"

#include "nntile/graph/tile/clear.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/scale_fiber.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"

namespace nntile::graph::tensor
{



TensorGraph::TensorNode* scale_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    const std::string& output_name,
    const std::vector<Index>& dst_shape,
    Index axis,
    Index batch_ndim)
{
    if(src == nullptr)
    {
        throw std::invalid_argument(
            "scale_fiber: input tensor must be non-null");
    }

    TensorGraph::TensorNode* dst = src->graph()->data(
        dst_shape,
        output_name,
        src->dtype());

    scale_fiber(alpha, src, dst, axis, batch_ndim);

    return dst;
}

void scale_fiber(
    Scalar alpha,
    TensorGraph::TensorNode* src,
    TensorGraph::TensorNode* dst,
    Index axis,
    Index batch_ndim)
{
    if(src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "scale_fiber: input tensors must be non-null");
    }
    if(src == dst)
    {
        throw std::invalid_argument(
            "scale_fiber: src and dst must be distinct tensors");
    }
    if(src->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "scale_fiber: input tensors must belong to the same graph");
    }
    if(src->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "scale_fiber: input tensors must have the same dtype");
    }
    validate_fiber_shape_and_merge(src, dst, axis, batch_ndim, "scale_fiber");

    auto op = std::make_shared<TensorScaleFiberOp>(
        alpha, src, dst, axis, batch_ndim);
    src->graph()->add_op(op);
}

void TensorScaleFiberOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::scale_fiber_async (src/tensor/scale_fiber.cc).
    const TensorAxisLayout* lay_d = ctx.tiling.find(dst);
    if(lay_d == nullptr)
    {
        throw std::runtime_error("lower_to_tile SCALE_FIBER: missing tiling for dst");
    }

    const auto& tiles_s = tile_lower::tiles_of(ctx.tile_map, src);
    const auto& tiles_d = tile_lower::tiles_of(ctx.tile_map, dst);

    if(alpha == 0.0)
    {
        for(TileGraph::TileNode* t : tiles_d)
        {
            tile_graph::clear(t);
        }
        return;
    }

    const TensorAxisLayout* lay_s = ctx.tiling.find(src);
    if(lay_s == nullptr)
    {
        throw std::runtime_error("lower_to_tile SCALE_FIBER: missing tiling for src");
    }

    std::vector<Index> dst_coord;
    std::vector<Index> src_coord(static_cast<size_t>(src->ndim()));

    for(Index lin_d = 0; lin_d < lay_d->grid_volume(); ++lin_d)
    {
        lay_d->grid_coord_from_linear(lin_d, dst_coord);
        src_coord[0] = dst_coord[static_cast<size_t>(axis)];
        for(Index b = 0; b < batch_ndim; ++b)
        {
            src_coord[static_cast<size_t>(b + 1)] =
                dst_coord[static_cast<size_t>(dst->ndim() - batch_ndim + b)];
        }
        const Index lin_s = lay_s->grid_linear(src_coord);
        tile_graph::scale_fiber(
            alpha,
            tiles_s[static_cast<size_t>(lin_s)],
            tiles_d[static_cast<size_t>(lin_d)],
            axis,
            batch_ndim);
    }
}

} // namespace nntile::graph::tensor
