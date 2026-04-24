/*! @copyright (c) 2022-present Skolkovo Institute of Science and Technology
 *                              (Skoltech), Russia. All rights reserved.
 *                 2023-present Artificial Intelligence Research Institute
 *                              (AIRI), Russia. All rights reserved.
 *
 * NNTile is software framework for fast training of big neural networks on
 * distributed-memory heterogeneous systems based on StarPU runtime system.
 *
 * @file src/graph/tensor/sumprod_slice.cc
 * TensorGraph sumprod_slice operation implementation.
 *
 * @version 1.1.0
 * */

#include "nntile/graph/tensor/sumprod_slice.hh"

#include <stdexcept>

#include "nntile/base_types.hh"
#include "nntile/graph/tensor.hh"
#include "nntile/graph/tensor/tensor_graph_tiling.hh"
#include "nntile/graph/tensor/tile_lowering_helpers.hh"
#include "nntile/graph/tile/lowering_context.hh"
#include "nntile/graph/tile/sumprod_slice.hh"
#include "nntile/tensor/sumprod_slice.hh"

namespace nntile::graph::tensor
{



void sumprod_slice(
    TensorGraph::TensorNode* src1,
    TensorGraph::TensorNode* src2,
    TensorGraph::TensorNode* dst,
    Index axis,
    int redux,
    Scalar alpha,
    Scalar beta)
{
    if(src1 == nullptr || src2 == nullptr || dst == nullptr)
    {
        throw std::invalid_argument(
            "sumprod_slice: input tensors must be non-null");
    }
    if(src1->graph() != src2->graph() || src1->graph() != dst->graph())
    {
        throw std::invalid_argument(
            "sumprod_slice: input tensors must belong to the same graph");
    }
    if(src1->dtype() != src2->dtype() || src1->dtype() != dst->dtype())
    {
        throw std::invalid_argument(
            "sumprod_slice: input tensors must have the same dtype");
    }
    if(axis < 0 || axis >= src1->ndim())
    {
        throw std::invalid_argument(
            "sumprod_slice: axis out of range");
    }
    if(src1 == src2 || src1 == dst || src2 == dst)
    {
        throw std::invalid_argument(
            "sumprod_slice: src1, src2, and dst must be distinct tensors");
    }

    validate_same_shape_and_merge(src1, src2, "sumprod_slice");
    validate_slice_shape_and_merge(dst, src1, axis, "sumprod_slice");

    auto op = std::make_shared<TensorSumprodSliceOp>(
        src1, src2, dst, axis, redux, alpha, beta);
    src1->graph()->add_op(op);
}

void TensorSumprodSliceOp::lower_to_tile(const LoweringContext& ctx) const
{
    // Match nntile::tensor::sumprod_slice_async (src/tensor/sumprod_slice.cc):
    // one dst tile aggregates src1/src2 tiles along `axis`.
    const TensorAxisLayout* lay_s1 = ctx.tiling.find(src1);
    const TensorAxisLayout* lay_d = ctx.tiling.find(dst);
    if(lay_s1 == nullptr || lay_d == nullptr)
    {
        throw std::runtime_error(
            "lower_to_tile SUMPROD_SLICE: missing tiling for src1 and/or dst");
    }

    tile_lower::assert_same_elementwise_layout(src1, src2, "SUMPROD_SLICE");

    const auto& tiles_s1 = tile_lower::tiles_of(ctx.tile_map, src1);
    const auto& tiles_s2 = tile_lower::tiles_of(ctx.tile_map, src2);
    const auto& tiles_d = tile_lower::tiles_of(ctx.tile_map, dst);

    std::vector<Index> dst_coord;
    std::vector<Index> s1_coord(static_cast<size_t>(src1->ndim()));

    for(Index lin_d = 0; lin_d < lay_d->grid_volume(); ++lin_d)
    {
        lay_d->grid_coord_from_linear(lin_d, dst_coord);
        TileGraph::TileNode* dst_tile = tiles_d[static_cast<size_t>(lin_d)];

        for(Index j = 0; j < axis; ++j)
        {
            s1_coord[static_cast<size_t>(j)] =
                dst_coord[static_cast<size_t>(j)];
        }
        for(Index j = axis + 1; j < src1->ndim(); ++j)
        {
            s1_coord[static_cast<size_t>(j)] =
                dst_coord[static_cast<size_t>(j - 1)];
        }

        const Index nseg_along_axis =
            lay_s1->grid_shape()[static_cast<size_t>(axis)];

        s1_coord[static_cast<size_t>(axis)] = 0;
        Index lin_s0 = lay_s1->grid_linear(s1_coord);
        tile_graph::sumprod_slice(
            alpha,
            tiles_s1[static_cast<size_t>(lin_s0)],
            tiles_s2[static_cast<size_t>(lin_s0)],
            beta,
            dst_tile,
            axis,
            redux);

        for(Index jj = 1; jj < nseg_along_axis; ++jj)
        {
            s1_coord[static_cast<size_t>(axis)] = jj;
            const Index lin_s = lay_s1->grid_linear(s1_coord);
            tile_graph::sumprod_slice(
                alpha,
                tiles_s1[static_cast<size_t>(lin_s)],
                tiles_s2[static_cast<size_t>(lin_s)],
                Scalar(1.0),
                dst_tile,
                axis,
                redux);
        }
    }
}

} // namespace nntile::graph::tensor
