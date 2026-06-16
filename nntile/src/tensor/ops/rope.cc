#include <nntile/common.hh>
/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file nntile/src/tensor_graph/rope.cc
 * TensorGraph rope operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/tensor/ops/rope.hh"

#include "nntile/dtype.hh"
#include "nntile/tensor.hh"
#include "nntile/tensor/tensor_graph_tiling.hh"
#include "nntile/tensor/tile_lowering_helpers.hh"
#include "nntile/tile/lowering_context.hh"
#include "nntile/tile/ops/rope.hh"
#include "nntile/tensor/ops/rope.hh"

#include <stdexcept>
#include <utility>

namespace nntile::tensor
{

TensorGraph::TensorNode *rope(TensorGraph::TensorNode *sin,
    TensorGraph::TensorNode *cos,
    TensorGraph::TensorNode *src)
{
    if (sin == nullptr || cos == nullptr || src == nullptr)
    {
        throw std::invalid_argument("rope: input tensors must be non-null");
    }
    if (sin->graph() != cos->graph() || sin->graph() != src->graph())
    {
        throw std::invalid_argument(
            "rope: input tensors must belong to the same graph");
    }
    if (sin->dtype() != cos->dtype() || sin->dtype() != src->dtype())
    {
        throw std::invalid_argument(
            "rope: input tensors must have the same dtype");
    }

    TensorGraph::TensorNode *dst =
        src->graph()->data(src->shape(), src->dtype());
    dst->set_axes(src->axes());

    rope(sin, cos, src, dst);

    return dst;
}

void rope(TensorGraph::TensorNode *sin,
    TensorGraph::TensorNode *cos,
    TensorGraph::TensorNode *src,
    TensorGraph::TensorNode *dst)
{
    if (sin == nullptr || cos == nullptr || src == nullptr || dst == nullptr)
    {
        throw std::invalid_argument("rope: input tensors must be non-null");
    }
    if (sin->graph() != cos->graph() || sin->graph() != src->graph() ||
        sin->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "rope: input tensors must belong to the same graph");
    }
    if (sin->dtype() != cos->dtype() || sin->dtype() != src->dtype() ||
        sin->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "rope: input tensors must have the same dtype");
    }
    validate_same_shape_and_merge(src, dst, "rope");
    const Index rope_axis = sin->ndim() - 1;
    for(Index d = 0; d < rope_axis; ++d)
    {
        merge_axis(sin->mutable_axes()[d], src->mutable_axes()[d]);
        merge_axis(cos->mutable_axes()[d], src->mutable_axes()[d]);
    }

    auto op = std::make_shared<TensorRopeOp>(sin, cos, src, dst);
    src->graph()->add_op(op);
}

void TensorRopeOp::lower_to_tile(const LoweringContext &ctx) const
{
    // Match nntile::tensor::rope_async (src/tensor/rope.cc).
    tile_lower::assert_same_elementwise_layout(src, dst, "ROPE src/dst");

    const TensorAxisLayout *lay_src = ctx.tiling.find(src);
    const TensorAxisLayout *lay_sin = ctx.tiling.find(sin);
    if (lay_src == nullptr || lay_sin == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile ROPE: missing tiling for src and/or sin");
    }

    const auto &tiles_sin = tile_lower::tiles_of(ctx.tile_map, sin);
    const auto &tiles_cos = tile_lower::tiles_of(ctx.tile_map, cos);
    const auto &tiles_src = tile_lower::tiles_of(ctx.tile_map, src);
    const auto &tiles_dst = tile_lower::tiles_of(ctx.tile_map, dst);

    const Index sin_ndim = sin->ndim();
    const Index rope_axis = sin_ndim - 1;
    std::vector<Index> src_coord;
    std::vector<Index> sincos_coord(static_cast<size_t>(sin_ndim));

    for (Index lin = 0; lin < lay_src->grid_volume(); ++lin)
    {
        lay_src->grid_coord_from_linear(lin, src_coord);
        for (Index d = 0; d < sin_ndim; ++d)
        {
            if (d == rope_axis)
            {
                Index src_lo = 0;
                Index src_hi = 0;
                lay_src->tile_axis_global_range(
                    src_coord, rope_axis, src_lo, src_hi);
                (void)src_hi;
                sincos_coord[static_cast<size_t>(rope_axis)] =
                    src_coord[static_cast<size_t>(rope_axis)];
            }
            else
            {
                sincos_coord[static_cast<size_t>(d)] =
                    src_coord[static_cast<size_t>(d)];
            }
        }
        const Index j = lay_sin->grid_linear(sincos_coord);
        Index src_lo = 0;
        Index src_hi = 0;
        Index sin_lo = 0;
        Index sin_hi = 0;
        lay_src->tile_axis_global_range(
            src_coord, rope_axis, src_lo, src_hi);
        lay_sin->tile_axis_global_range(
            sincos_coord, rope_axis, sin_lo, sin_hi);
        const Index sin_pair0 = src_lo / 2 - sin_lo;
        tile::rope(tiles_sin[static_cast<size_t>(j)],
            tiles_cos[static_cast<size_t>(j)],
            tiles_src[static_cast<size_t>(lin)],
            tiles_dst[static_cast<size_t>(lin)],
            sin_pair0);
    }
}

} // namespace nntile::tensor
